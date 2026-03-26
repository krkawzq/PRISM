from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

GeneMAEMode = Literal["pretrain", "classify", "embed"]
ClassificationHeadName = Literal["linear", "mlp"]

__all__ = [
    "ClassificationHeadName",
    "GeneMAE",
    "GeneMAEConfig",
    "GeneMAEForwardOutput",
    "GeneMAEMode",
    "GeneNet",
    "GeneNetConfig",
    "GeneNetOutput",
    "build_gene_mae",
    "build_gene_net",
    "count_parameters",
    "extract_masked_targets_and_weights",
    "masked_regression_loss",
]


@dataclass(frozen=True, slots=True)
class GeneMAEConfig:
    """GeneMAE model hyperparameters."""

    n_genes: int
    n_classes: int = 2
    d_model: int = 64
    d_gene_id: int = 16
    n_heads: int = 4
    head_expand: float = 1.0
    n_layers: int = 4
    attn_dropout: float = 0.0
    ffn_expand: float = 4.0
    ffn_dropout: float = 0.1
    signal_bins: int = 64
    mask_ratio: float = 0.20
    use_learnable_mask_token: bool = True
    use_cls_token: bool = True
    classification_head: ClassificationHeadName = "linear"
    embed_dropout: float = 0.1
    path_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_genes < 1:
            raise ValueError(f"n_genes 必须 >= 1，收到 {self.n_genes}")
        if self.n_classes < 1:
            raise ValueError(f"n_classes 必须 >= 1，收到 {self.n_classes}")
        if self.d_model < 1:
            raise ValueError(f"d_model 必须 >= 1，收到 {self.d_model}")
        if self.d_gene_id < 1:
            raise ValueError(f"d_gene_id 必须 >= 1，收到 {self.d_gene_id}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads 必须 >= 1，收到 {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model 必须能被 n_heads 整除，收到 d_model={self.d_model}, n_heads={self.n_heads}"
            )
        if self.head_expand <= 0:
            raise ValueError(f"head_expand 必须 > 0，收到 {self.head_expand}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers 必须 >= 1，收到 {self.n_layers}")
        if self.ffn_expand <= 0:
            raise ValueError(f"ffn_expand 必须 > 0，收到 {self.ffn_expand}")
        if self.signal_bins < 2:
            raise ValueError(f"signal_bins 必须 >= 2，收到 {self.signal_bins}")
        if not 0.0 <= self.attn_dropout < 1.0:
            raise ValueError(f"attn_dropout 必须在 [0, 1) 内，收到 {self.attn_dropout}")
        if not 0.0 <= self.ffn_dropout < 1.0:
            raise ValueError(f"ffn_dropout 必须在 [0, 1) 内，收到 {self.ffn_dropout}")
        if not 0.0 <= self.embed_dropout < 1.0:
            raise ValueError(
                f"embed_dropout 必须在 [0, 1) 内，收到 {self.embed_dropout}"
            )
        if not 0.0 <= self.path_dropout < 1.0:
            raise ValueError(f"path_dropout 必须在 [0, 1) 内，收到 {self.path_dropout}")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio 必须在 (0, 1) 内，收到 {self.mask_ratio}")
        if not self.use_cls_token:
            raise ValueError("当前要求 use_cls_token=True")


@dataclass(frozen=True, slots=True)
class GeneMAEForwardOutput:
    """GeneMAE forward output container."""

    masked_predictions: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    cls_embedding: torch.Tensor | None = None
    token_embeddings: torch.Tensor | None = None
    mask: torch.Tensor | None = None


class ScalarEmbedding(nn.Module):
    """Embed scalar inputs with soft bin assignment."""

    def __init__(self, n_bins: int, output_dim: int) -> None:
        super().__init__()
        if n_bins < 2:
            raise ValueError(f"n_bins 必须 >= 2，收到 {n_bins}")
        if output_dim < 1:
            raise ValueError(f"output_dim 必须 >= 1，收到 {output_dim}")

        self.n_bins = int(n_bins)
        self.output_dim = int(output_dim)
        self.bin_scorer = nn.Linear(1, self.n_bins)
        self.bin_embeddings = nn.Parameter(torch.empty(self.n_bins, self.output_dim))
        nn.init.trunc_normal_(self.bin_embeddings, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values_3d = _ensure_last_dim(values)
        logits = self.bin_scorer(values_3d)
        weights = F.softmax(logits, dim=-1)
        return torch.matmul(weights, self.bin_embeddings)


class GeneIdentifier(nn.Module):
    """Learned token identity vectors for CLS + gene positions."""

    def __init__(self, n_genes: int, d_gene_id: int) -> None:
        super().__init__()
        if n_genes < 1:
            raise ValueError(f"n_genes 必须 >= 1，收到 {n_genes}")
        if d_gene_id < 1:
            raise ValueError(f"d_gene_id 必须 >= 1，收到 {d_gene_id}")

        self.n_tokens = int(n_genes) + 1
        self.d_gene_id = int(d_gene_id)
        self.identifiers = nn.Parameter(torch.empty(self.n_tokens, self.d_gene_id))
        nn.init.trunc_normal_(self.identifiers, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")
        return self.identifiers.unsqueeze(0).expand(batch_size, -1, -1)


class GeneAwareAttention(nn.Module):
    """Attention that uses token identity only for Q/K pathways."""

    def __init__(self, config: GeneMAEConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_gene_id = config.d_gene_id
        self.d_head_v = config.d_model // config.n_heads
        self.d_head_qk = max(1, int(self.d_head_v * config.head_expand))
        self.scale = self.d_head_qk**-0.5

        qk_input_dim = config.d_model + config.d_gene_id
        self.q_proj = nn.Linear(qk_input_dim, self.n_heads * self.d_head_qk, bias=False)
        self.k_proj = nn.Linear(qk_input_dim, self.n_heads * self.d_head_qk, bias=False)
        self.v_proj = nn.Linear(
            config.d_model, self.n_heads * self.d_head_v, bias=False
        )
        self.out_proj = nn.Linear(
            self.n_heads * self.d_head_v, config.d_model, bias=False
        )
        self.attn_dropout = nn.Dropout(config.attn_dropout)

    def forward(self, x: torch.Tensor, gene_id: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_with_id = torch.cat([x, gene_id], dim=-1)
        q = self.q_proj(x_with_id).view(
            batch_size, seq_len, self.n_heads, self.d_head_qk
        )
        k = self.k_proj(x_with_id).view(
            batch_size, seq_len, self.n_heads, self.d_head_qk
        )
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head_v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block with fresh gene-id injection."""

    def __init__(self, config: GeneMAEConfig) -> None:
        super().__init__()
        input_dim = config.d_model + config.d_gene_id
        hidden_dim = int(config.d_model * config.ffn_expand)
        self.value_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.ffn_dropout)

    def forward(self, x: torch.Tensor, gene_id: torch.Tensor) -> torch.Tensor:
        x_with_id = torch.cat([x, gene_id], dim=-1)
        hidden = self.value_proj(x_with_id) * F.silu(self.gate_proj(x_with_id))
        return self.out_proj(self.dropout(hidden))


class DropPath(nn.Module):
    """Stochastic depth over the residual branch."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob 必须在 [0, 1) 内，收到 {drop_prob}")
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * mask / keep_prob


class GeneMAEBlock(nn.Module):
    """Transformer block with pre-norm on the data stream only."""

    def __init__(self, config: GeneMAEConfig, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.attention = GeneAwareAttention(config)
        self.ffn = SwiGLUFFN(config)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor, gene_id: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x), gene_id))
        x = x + self.drop_path(self.ffn(self.norm2(x), gene_id))
        return x


class GeneMAE(nn.Module):
    """Gene-aware masked autoencoder for single-cell feature experiments."""

    def __init__(self, config: GeneMAEConfig) -> None:
        super().__init__()
        self.config = config

        self.signal_embedding = ScalarEmbedding(config.signal_bins, config.d_model)
        self.cls_token = nn.Parameter(torch.empty(config.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.gene_identifier = GeneIdentifier(config.n_genes, config.d_gene_id)

        if config.use_learnable_mask_token:
            self.mask_token = nn.Parameter(torch.empty(config.d_model))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            self.register_buffer("mask_token", torch.zeros(config.d_model))

        self.embed_dropout = nn.Dropout(config.embed_dropout)
        self.blocks = nn.ModuleList(
            [
                GeneMAEBlock(config, drop_path_rate=rate)
                for rate in _drop_path_schedule(
                    n_layers=config.n_layers,
                    max_drop_path=config.path_dropout,
                )
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.regression_head = nn.Linear(config.d_model, 1)
        self.classification_head = _build_classification_head(config)

        self.apply(self._init_weights)

    def forward(
        self,
        signal: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        mode: GeneMAEMode = "pretrain",
    ) -> GeneMAEForwardOutput:
        if mode == "pretrain":
            return self.forward_pretrain(signal, mask=mask)
        if mode == "classify":
            return self.forward_classify(signal, mask=mask)
        if mode == "embed":
            return self.encode(signal, mask=mask)
        raise ValueError(f"未知 mode: {mode!r}")

    def forward_pretrain(
        self,
        signal: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        return_token_embeddings: bool = True,
    ) -> GeneMAEForwardOutput:
        if mask is None:
            mask = self.create_mask(batch_size=signal.shape[0], device=signal.device)
        encoded = self._encode(signal, mask=mask)
        gene_tokens = encoded[:, 1:, :]
        masked_repr = gene_tokens[mask]
        masked_predictions = self.regression_head(masked_repr).squeeze(-1)
        return GeneMAEForwardOutput(
            masked_predictions=masked_predictions,
            cls_embedding=encoded[:, 0, :],
            token_embeddings=gene_tokens if return_token_embeddings else None,
            mask=mask,
        )

    def forward_classify(
        self,
        signal: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        return_token_embeddings: bool = True,
    ) -> GeneMAEForwardOutput:
        encoded = self._encode(signal, mask=mask)
        cls_embedding = encoded[:, 0, :]
        return GeneMAEForwardOutput(
            logits=self.classification_head(cls_embedding),
            cls_embedding=cls_embedding,
            token_embeddings=encoded[:, 1:, :] if return_token_embeddings else None,
            mask=mask,
        )

    def encode(
        self,
        signal: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        return_cls_embedding: bool = True,
    ) -> GeneMAEForwardOutput:
        encoded = self._encode(signal, mask=mask)
        return GeneMAEForwardOutput(
            cls_embedding=encoded[:, 0, :] if return_cls_embedding else None,
            token_embeddings=encoded[:, 1:, :],
            mask=mask,
        )

    def create_mask(
        self,
        *,
        batch_size: int,
        device: torch.device | str,
        mask_ratio: float | None = None,
    ) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")
        resolved_ratio = (
            self.config.mask_ratio if mask_ratio is None else float(mask_ratio)
        )
        if not 0.0 < resolved_ratio < 1.0:
            raise ValueError(f"mask_ratio 必须在 (0, 1) 内，收到 {resolved_ratio}")

        n_mask = max(1, int(round(self.config.n_genes * resolved_ratio)))
        noise = torch.rand(batch_size, self.config.n_genes, device=device)
        _, topk_idx = torch.topk(noise, k=n_mask, dim=1, largest=False)
        mask = torch.zeros(
            batch_size,
            self.config.n_genes,
            dtype=torch.bool,
            device=device,
        )
        mask.scatter_(1, topk_idx, True)
        return mask

    def masked_targets(
        self,
        signal: torch.Tensor,
        mask: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return extract_masked_targets_and_weights(signal, mask, weights=weights)

    def _encode(
        self,
        signal: torch.Tensor,
        *,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        self._validate_inputs(signal, mask)
        batch_size = signal.shape[0]
        x = self._embed_inputs(signal, mask=mask)
        gene_id = self.gene_identifier(batch_size)
        for block in self.blocks:
            x = block(x, gene_id)
        return self.final_norm(x)

    def _embed_inputs(
        self,
        signal: torch.Tensor,
        *,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        gene_emb = self.signal_embedding(signal)
        if mask is not None:
            mask_token = self.mask_token.view(1, 1, -1).expand_as(gene_emb)
            gene_emb = torch.where(mask.unsqueeze(-1), mask_token, gene_emb)
        cls_emb = self.cls_token.view(1, 1, -1).expand(signal.shape[0], 1, -1)
        x = torch.cat([cls_emb, gene_emb], dim=1)
        return self.embed_dropout(x)

    def _validate_inputs(
        self,
        signal: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        if signal.ndim != 2:
            raise ValueError(f"signal 必须为二维，收到 shape={tuple(signal.shape)}")
        if signal.shape[1] != self.config.n_genes:
            raise ValueError(
                f"signal 第二维必须等于 n_genes={self.config.n_genes}，"
                f"收到 shape={tuple(signal.shape)}"
            )
        if mask is not None:
            if mask.shape != signal.shape:
                raise ValueError(
                    f"mask shape 必须等于 signal shape={tuple(signal.shape)}，"
                    f"收到 {tuple(mask.shape)}"
                )
            if mask.dtype != torch.bool:
                raise ValueError(f"mask 必须为 bool，收到 {mask.dtype}")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def masked_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted masked regression loss."""

    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions shape 必须等于 targets shape，收到 {tuple(predictions.shape)} 和 {tuple(targets.shape)}"
        )
    residual = (predictions - targets) ** 2
    if weights is None:
        return residual.mean()
    if weights.shape != predictions.shape:
        raise ValueError(
            f"weights shape 必须等于 predictions shape，收到 {tuple(weights.shape)} 和 {tuple(predictions.shape)}"
        )
    return (residual * weights).sum() / (weights.sum() + eps)


def extract_masked_targets_and_weights(
    signal: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Extract masked targets and optional per-position weights."""

    if signal.shape != mask.shape:
        raise ValueError(
            f"signal shape 必须等于 mask shape，收到 {tuple(signal.shape)} 和 {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise ValueError(f"mask 必须为 bool，收到 {mask.dtype}")
    if weights is not None and weights.shape != signal.shape:
        raise ValueError(
            f"weights shape 必须等于 signal shape，收到 {tuple(weights.shape)} 和 {tuple(signal.shape)}"
        )
    targets = signal[mask]
    selected_weights = None if weights is None else weights[mask]
    return targets, selected_weights


def build_gene_mae(**kwargs: Any) -> GeneMAE:
    """Construct GeneMAE from keyword config arguments."""

    config_kwargs = cast(dict[str, Any], kwargs)
    return GeneMAE(GeneMAEConfig(**config_kwargs))


def build_gene_net(**kwargs: Any) -> GeneMAE:
    """Backward-compatible alias for build_gene_mae."""

    return build_gene_mae(**kwargs)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable parameters by top-level module name."""

    counts: dict[str, int] = {}
    total = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n_params = int(param.numel())
        total += n_params
        top_level = name.split(".", 1)[0]
        counts[top_level] = counts.get(top_level, 0) + n_params
    counts["_total"] = total
    return counts


def _build_classification_head(config: GeneMAEConfig) -> nn.Module:
    if config.classification_head == "linear":
        return nn.Linear(config.d_model, config.n_classes)
    if config.classification_head == "mlp":
        return nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.ffn_dropout),
            nn.Linear(config.d_model, config.n_classes),
        )
    raise ValueError(f"未知 classification_head: {config.classification_head!r}")


def _drop_path_schedule(*, n_layers: int, max_drop_path: float) -> list[float]:
    if n_layers == 1:
        return [float(max_drop_path)]
    return [float(max_drop_path) * i / (n_layers - 1) for i in range(n_layers)]


def _ensure_last_dim(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values.unsqueeze(-1)
    if values.ndim == 2:
        return values.unsqueeze(-1)
    if values.ndim == 3 and values.shape[-1] == 1:
        return values
    raise ValueError(
        f"输入必须为 (B,), (B, S) 或 (B, S, 1)，收到 shape={tuple(values.shape)}"
    )


GeneNetConfig = GeneMAEConfig
GeneNetOutput = GeneMAEForwardOutput
GeneNet = GeneMAE
