from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from .gene_mae import (
    ClassificationHeadName,
    DropPath,
    GeneIdentifier,
    ScalarEmbedding,
    extract_masked_targets_and_weights,
)

StaticGeneNetMode = Literal["pretrain", "classify", "embed"]

__all__ = [
    "StaticGeneNet",
    "StaticGeneNetConfig",
    "StaticGeneNetForwardOutput",
    "StaticGeneNetMode",
    "build_static_gene_net",
    "count_parameters",
]


@dataclass(frozen=True, slots=True)
class StaticGeneNetConfig:
    n_genes: int
    n_classes: int = 2
    d_model: int = 64
    d_gene_id: int = 16
    n_heads: int = 4
    n_layers: int = 4
    static_qk_dim: int = 16
    ffn_expand: float = 4.0
    ffn_dropout: float = 0.1
    embed_dropout: float = 0.1
    path_dropout: float = 0.0
    signal_bins: int = 64
    mask_ratio: float = 0.20
    use_learnable_mask_token: bool = True
    use_cls_token: bool = True
    classification_head: ClassificationHeadName = "linear"
    share_layer_params: bool = False
    attention_dropout: float = 0.0
    use_gene_id: bool = False

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
        if self.n_layers < 1:
            raise ValueError(f"n_layers 必须 >= 1，收到 {self.n_layers}")
        if self.static_qk_dim < 1:
            raise ValueError(f"static_qk_dim 必须 >= 1，收到 {self.static_qk_dim}")
        if self.ffn_expand <= 0:
            raise ValueError(f"ffn_expand 必须 > 0，收到 {self.ffn_expand}")
        if self.signal_bins < 2:
            raise ValueError(f"signal_bins 必须 >= 2，收到 {self.signal_bins}")
        if not 0.0 <= self.ffn_dropout < 1.0:
            raise ValueError(f"ffn_dropout 必须在 [0, 1) 内，收到 {self.ffn_dropout}")
        if not 0.0 <= self.embed_dropout < 1.0:
            raise ValueError(
                f"embed_dropout 必须在 [0, 1) 内，收到 {self.embed_dropout}"
            )
        if not 0.0 <= self.path_dropout < 1.0:
            raise ValueError(f"path_dropout 必须在 [0, 1) 内，收到 {self.path_dropout}")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(
                f"attention_dropout 必须在 [0, 1) 内，收到 {self.attention_dropout}"
            )
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio 必须在 (0, 1) 内，收到 {self.mask_ratio}")
        if not self.use_cls_token:
            raise ValueError("当前要求 use_cls_token=True")


@dataclass(frozen=True, slots=True)
class StaticGeneNetForwardOutput:
    masked_predictions: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    cls_embedding: torch.Tensor | None = None
    token_embeddings: torch.Tensor | None = None
    mask: torch.Tensor | None = None


class StaticGeneAttention(nn.Module):
    """Static multi-head gene communication graph.

    Each source token owns learned q/k vectors and an outgoing activity scalar.
    The qk interaction forms a fixed send-to graph per head. Values remain input-
    dependent, so the topology is static while the propagated signal is dynamic.
    """

    def __init__(self, config: StaticGeneNetConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_tokens = config.n_genes + 1
        self.d_head = config.d_model // config.n_heads
        self.static_qk_dim = config.static_qk_dim
        self.scale = self.static_qk_dim**-0.5

        self.q_vectors = nn.Parameter(
            torch.empty(self.n_heads, self.n_tokens, self.static_qk_dim)
        )
        self.k_vectors = nn.Parameter(
            torch.empty(self.n_heads, self.n_tokens, self.static_qk_dim)
        )
        self.send_log_scale = nn.Parameter(torch.zeros(self.n_heads, self.n_tokens))
        self.value_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        nn.init.trunc_normal_(self.q_vectors, std=0.02)
        nn.init.trunc_normal_(self.k_vectors, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if seq_len != self.n_tokens:
            raise ValueError(
                f"seq_len 必须等于 n_tokens={self.n_tokens}，收到 {seq_len}"
            )

        values = self.value_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        values = values.transpose(1, 2)

        logits = (
            torch.einsum("htd,hsd->hts", self.q_vectors, self.k_vectors) * self.scale
        )
        send_weights = F.softmax(logits, dim=-1)
        send_weights = self.attention_dropout(send_weights)

        send_activity = F.softplus(self.send_log_scale).unsqueeze(-1)
        send_weights = send_weights * send_activity

        propagated = torch.einsum("hts,bhtd->bhsd", send_weights, values)
        propagated = propagated.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(propagated)


class SwiGLUFFN(nn.Module):
    def __init__(self, config: StaticGeneNetConfig) -> None:
        super().__init__()
        self.use_gene_id = bool(config.use_gene_id)
        input_dim = config.d_model + (config.d_gene_id if self.use_gene_id else 0)
        hidden_dim = int(config.d_model * config.ffn_expand)
        self.value_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.ffn_dropout)

    def forward(
        self, x: torch.Tensor, gene_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.use_gene_id:
            if gene_id is None:
                raise ValueError("use_gene_id=True 时 gene_id 不能为空")
            hidden_input = torch.cat([x, gene_id], dim=-1)
        else:
            hidden_input = x
        hidden = self.value_proj(hidden_input) * F.silu(self.gate_proj(hidden_input))
        return self.out_proj(self.dropout(hidden))


class StaticGeneNetBlock(nn.Module):
    def __init__(
        self, config: StaticGeneNetConfig, drop_path_rate: float = 0.0
    ) -> None:
        super().__init__()
        self.n_tokens = config.n_genes + 1
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.attention = StaticGeneAttention(config)
        self.ffn = SwiGLUFFN(config)
        self.drop_path = DropPath(drop_path_rate)
        self.attn_residual_lambda = nn.Parameter(torch.ones(self.n_tokens))

    def forward(
        self, x: torch.Tensor, gene_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        attn_out = self.attention(self.norm1(x))
        attn_scale = self.attn_residual_lambda.view(1, self.n_tokens, 1)
        x = x + self.drop_path(attn_out * attn_scale)
        x = x + self.drop_path(self.ffn(self.norm2(x), gene_id))
        return x


class StaticGeneNet(nn.Module):
    """Static transformer-style network for gene-level modeling."""

    def __init__(self, config: StaticGeneNetConfig) -> None:
        super().__init__()
        self.config = config
        self.signal_embedding = ScalarEmbedding(config.signal_bins, config.d_model)
        self.cls_token = nn.Parameter(torch.empty(config.d_model))
        self.gene_identifier = (
            GeneIdentifier(config.n_genes, config.d_gene_id)
            if config.use_gene_id
            else None
        )
        self.input_token_bias = (
            nn.Linear(config.d_gene_id, config.d_model, bias=False)
            if config.use_gene_id
            else None
        )

        if config.use_learnable_mask_token:
            self.mask_token = nn.Parameter(torch.empty(config.d_model))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            self.register_buffer("mask_token", torch.zeros(config.d_model))

        self.embed_dropout = nn.Dropout(config.embed_dropout)
        if config.share_layer_params:
            self.shared_block = StaticGeneNetBlock(
                config, drop_path_rate=config.path_dropout
            )
            self.blocks = None
        else:
            self.shared_block = None
            self.blocks = nn.ModuleList(
                [
                    StaticGeneNetBlock(config, drop_path_rate=rate)
                    for rate in _drop_path_schedule(
                        n_layers=config.n_layers,
                        max_drop_path=config.path_dropout,
                    )
                ]
            )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.regression_head = nn.Linear(config.d_model, 1)
        self.classification_head = _build_classification_head(config)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def forward(
        self,
        signal: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        mode: StaticGeneNetMode = "pretrain",
    ) -> StaticGeneNetForwardOutput:
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
    ) -> StaticGeneNetForwardOutput:
        if mask is None:
            mask = self.create_mask(batch_size=signal.shape[0], device=signal.device)
        encoded = self._encode(signal, mask=mask)
        gene_tokens = encoded[:, 1:, :]
        masked_repr = gene_tokens[mask]
        masked_predictions = self.regression_head(masked_repr).squeeze(-1)
        return StaticGeneNetForwardOutput(
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
    ) -> StaticGeneNetForwardOutput:
        encoded = self._encode(signal, mask=mask)
        cls_embedding = encoded[:, 0, :]
        return StaticGeneNetForwardOutput(
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
    ) -> StaticGeneNetForwardOutput:
        encoded = self._encode(signal, mask=mask)
        return StaticGeneNetForwardOutput(
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
        self, signal: torch.Tensor, *, mask: torch.Tensor | None
    ) -> torch.Tensor:
        self._validate_inputs(signal, mask)
        gene_id = (
            None
            if self.gene_identifier is None
            else self.gene_identifier(signal.shape[0])
        )
        x = self._embed_inputs(signal, mask=mask, gene_id=gene_id)
        if self.shared_block is not None:
            for _ in range(self.config.n_layers):
                x = self.shared_block(x, gene_id)
        elif self.blocks is not None:
            for block in self.blocks:
                x = block(x, gene_id)
        return self.final_norm(x)

    def _embed_inputs(
        self,
        signal: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        gene_id: torch.Tensor | None,
    ) -> torch.Tensor:
        gene_emb = self.signal_embedding(signal)
        if mask is not None:
            mask_token = self.mask_token.view(1, 1, -1).expand_as(gene_emb)
            gene_emb = torch.where(mask.unsqueeze(-1), mask_token, gene_emb)
        cls_emb = self.cls_token.view(1, 1, -1).expand(signal.shape[0], 1, -1)
        x = torch.cat([cls_emb, gene_emb], dim=1)
        if self.input_token_bias is not None:
            if gene_id is None:
                raise ValueError("use_gene_id=True 时 gene_id 不能为空")
            x = x + self.input_token_bias(gene_id)
        return self.embed_dropout(x)

    def _validate_inputs(self, signal: torch.Tensor, mask: torch.Tensor | None) -> None:
        if signal.ndim != 2:
            raise ValueError(f"signal 必须为二维，收到 shape={tuple(signal.shape)}")
        if signal.shape[1] != self.config.n_genes:
            raise ValueError(
                f"signal 第二维必须等于 n_genes={self.config.n_genes}，收到 shape={tuple(signal.shape)}"
            )
        if mask is not None:
            if mask.shape != signal.shape:
                raise ValueError(
                    f"mask shape 必须等于 signal shape={tuple(signal.shape)}，收到 {tuple(mask.shape)}"
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


def build_static_gene_net(**kwargs: Any) -> StaticGeneNet:
    config_kwargs = cast(dict[str, Any], kwargs)
    return StaticGeneNet(StaticGeneNetConfig(**config_kwargs))


def count_parameters(model: nn.Module) -> dict[str, int]:
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


def _build_classification_head(config: StaticGeneNetConfig) -> nn.Module:
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
