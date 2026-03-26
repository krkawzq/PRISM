from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from .gene_mae import (
    ClassificationHeadName,
    GeneMAE,
    GeneMAEConfig,
    count_parameters,
)

GeneJEPAMode = Literal["pretrain", "classify", "embed"]
PredictorLossName = Literal["l2", "smooth_l1"]

__all__ = [
    "GeneJEPA",
    "GeneJEPAConfig",
    "GeneJEPAForwardOutput",
    "GeneJEPAMode",
    "PredictorLossName",
    "build_gene_jepa",
    "cosine_ema_schedule",
    "count_parameters",
    "jepa_train_step",
]


@dataclass(frozen=True, slots=True)
class GeneJEPAConfig:
    """GeneJEPA hyperparameters."""

    encoder: GeneMAEConfig
    predictor_d_model: int = 32
    predictor_n_heads: int = 4
    predictor_n_layers: int = 2
    predictor_dropout: float = 0.1
    ema_momentum: float = 0.996
    ema_momentum_end: float = 1.0
    loss_type: PredictorLossName = "l2"
    classification_head: ClassificationHeadName = "linear"

    def __post_init__(self) -> None:
        if self.predictor_d_model < 1:
            raise ValueError(
                f"predictor_d_model 必须 >= 1，收到 {self.predictor_d_model}"
            )
        if self.predictor_n_heads < 1:
            raise ValueError(
                f"predictor_n_heads 必须 >= 1，收到 {self.predictor_n_heads}"
            )
        if self.predictor_d_model % self.predictor_n_heads != 0:
            raise ValueError(
                "predictor_d_model 必须能被 predictor_n_heads 整除，"
                f"收到 {self.predictor_d_model} 和 {self.predictor_n_heads}"
            )
        if self.predictor_n_layers < 1:
            raise ValueError(
                f"predictor_n_layers 必须 >= 1，收到 {self.predictor_n_layers}"
            )
        if not 0.0 <= self.predictor_dropout < 1.0:
            raise ValueError(
                f"predictor_dropout 必须在 [0, 1) 内，收到 {self.predictor_dropout}"
            )
        if not 0.0 <= self.ema_momentum <= 1.0:
            raise ValueError(f"ema_momentum 必须在 [0, 1] 内，收到 {self.ema_momentum}")
        if not 0.0 <= self.ema_momentum_end <= 1.0:
            raise ValueError(
                f"ema_momentum_end 必须在 [0, 1] 内，收到 {self.ema_momentum_end}"
            )
        if self.ema_momentum_end < self.ema_momentum:
            raise ValueError(
                "要求 ema_momentum_end >= ema_momentum，"
                f"收到 {self.ema_momentum_end} < {self.ema_momentum}"
            )


@dataclass(frozen=True, slots=True)
class GeneJEPAForwardOutput:
    """GeneJEPA forward output container."""

    loss: torch.Tensor | None = None
    cls_loss: torch.Tensor | None = None
    token_loss: torch.Tensor | None = None
    predicted_cls_latent: torch.Tensor | None = None
    target_cls_latent: torch.Tensor | None = None
    predicted_latents: torch.Tensor | None = None
    target_latents: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    cls_embedding: torch.Tensor | None = None
    token_embeddings: torch.Tensor | None = None
    mask: torch.Tensor | None = None


class PredictorBlock(nn.Module):
    """Lightweight transformer block used by the JEPA predictor."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = d_model * 4
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed)
        x = x + attended
        x = x + self.ffn(self.norm2(x))
        return x


class JEPAPredictor(nn.Module):
    """Predict target latents from context latents and masked positions."""

    def __init__(self, config: GeneJEPAConfig) -> None:
        super().__init__()
        encoder_dim = config.encoder.d_model
        self.predictor_d_model = config.predictor_d_model
        self.input_proj = nn.Linear(encoder_dim, self.predictor_d_model)
        self.mask_token = nn.Parameter(torch.empty(self.predictor_d_model))
        self.pos_embed = nn.Parameter(
            torch.empty(1, config.encoder.n_genes + 1, self.predictor_d_model)
        )
        self.blocks = nn.ModuleList(
            [
                PredictorBlock(
                    d_model=config.predictor_d_model,
                    n_heads=config.predictor_n_heads,
                    dropout=config.predictor_dropout,
                )
                for _ in range(config.predictor_n_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.predictor_d_model)
        self.output_proj = nn.Linear(self.predictor_d_model, encoder_dim)
        self._reset_parameters()

    def forward(self, context_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = context_tokens.shape
        if mask.shape != (batch_size, seq_len - 1):
            raise ValueError(
                f"mask shape 必须为 ({batch_size}, {seq_len - 1})，收到 {tuple(mask.shape)}"
            )

        x = self.input_proj(context_tokens)
        x = x + self.pos_embed[:, :seq_len, :]

        full_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        full_mask[:, 1:] = mask
        mask_token = self.mask_token.view(1, 1, -1) + self.pos_embed[:, :seq_len, :]
        x = torch.where(full_mask.unsqueeze(-1), mask_token.expand_as(x), x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.output_proj(x)

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class EMA:
    """Exponential moving average updater."""

    def __init__(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum 必须在 [0, 1] 内，收到 {momentum}")
        self.momentum = float(momentum)

    @torch.no_grad()
    def update(self, target_model: nn.Module, source_model: nn.Module) -> None:
        for target_param, source_param in zip(
            target_model.parameters(), source_model.parameters(), strict=True
        ):
            target_param.data.mul_(self.momentum).add_(
                source_param.data, alpha=1.0 - self.momentum
            )

    def set_momentum(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum 必须在 [0, 1] 内，收到 {momentum}")
        self.momentum = float(momentum)


def cosine_ema_schedule(
    base_momentum: float,
    end_momentum: float,
    step: int,
    total_steps: int,
) -> float:
    """Cosine EMA schedule from base_momentum to end_momentum."""

    progress = step / max(total_steps - 1, 1)
    return (
        end_momentum
        - (end_momentum - base_momentum) * (math.cos(math.pi * progress) + 1.0) / 2.0
    )


class GeneJEPA(nn.Module):
    """Gene-level JEPA with MAE-style interfaces."""

    def __init__(self, config: GeneJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = GeneMAE(copy.deepcopy(config.encoder))
        self.target_encoder = GeneMAE(copy.deepcopy(config.encoder))
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = JEPAPredictor(config)
        self.ema = EMA(config.ema_momentum)
        self.classification_head = _build_classification_head(config)
        self._init_classification_head()

    def forward(
        self,
        signal: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        mode: GeneJEPAMode = "pretrain",
    ) -> GeneJEPAForwardOutput:
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
    ) -> GeneJEPAForwardOutput:
        if mask is None:
            mask = self.create_mask(batch_size=signal.shape[0], device=signal.device)
        self._validate_mask(mask=mask, signal=signal)

        with torch.no_grad():
            self.target_encoder.eval()
            target_output = self.target_encoder.encode(signal, mask=None)
            if (
                target_output.token_embeddings is None
                or target_output.cls_embedding is None
            ):
                raise RuntimeError("target encoder 未返回完整 embedding")
            target_cls_latent = target_output.cls_embedding.detach()
            target_latents = target_output.token_embeddings[mask].detach()

        context_output = self.context_encoder.encode(signal, mask=mask)
        if (
            context_output.token_embeddings is None
            or context_output.cls_embedding is None
        ):
            raise RuntimeError("context encoder 未返回完整 embedding")

        context_tokens = torch.cat(
            [
                context_output.cls_embedding.unsqueeze(1),
                context_output.token_embeddings,
            ],
            dim=1,
        )
        predicted_sequence = self.predictor(context_tokens, mask)
        predicted_cls_latent = predicted_sequence[:, 0, :]
        predicted_latents = predicted_sequence[:, 1:, :][mask]
        cls_loss = _latent_prediction_loss(
            predicted_cls_latent,
            target_cls_latent,
            loss_type=self.config.loss_type,
        )
        token_loss = _latent_prediction_loss(
            predicted_latents,
            target_latents,
            loss_type=self.config.loss_type,
        )
        loss = 0.5 * (cls_loss + token_loss)

        return GeneJEPAForwardOutput(
            loss=loss,
            cls_loss=cls_loss,
            token_loss=token_loss,
            predicted_cls_latent=predicted_cls_latent,
            target_cls_latent=target_cls_latent,
            predicted_latents=predicted_latents,
            target_latents=target_latents,
            cls_embedding=context_output.cls_embedding,
            token_embeddings=(
                context_output.token_embeddings if return_token_embeddings else None
            ),
            mask=mask,
        )

    def forward_classify(
        self,
        signal: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        return_token_embeddings: bool = True,
    ) -> GeneJEPAForwardOutput:
        encoded = self.context_encoder.encode(signal, mask=mask)
        if encoded.cls_embedding is None:
            raise RuntimeError("context encoder 未返回 cls embedding")
        return GeneJEPAForwardOutput(
            logits=self.classification_head(encoded.cls_embedding),
            cls_embedding=encoded.cls_embedding,
            token_embeddings=(
                encoded.token_embeddings if return_token_embeddings else None
            ),
            mask=mask,
        )

    def encode(
        self,
        signal: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        return_cls_embedding: bool = True,
    ) -> GeneJEPAForwardOutput:
        encoded = self.context_encoder.encode(signal, mask=mask)
        return GeneJEPAForwardOutput(
            cls_embedding=encoded.cls_embedding if return_cls_embedding else None,
            token_embeddings=encoded.token_embeddings,
            mask=mask,
        )

    def create_mask(
        self,
        *,
        batch_size: int,
        device: torch.device | str,
        mask_ratio: float | None = None,
    ) -> torch.Tensor:
        return self.context_encoder.create_mask(
            batch_size=batch_size,
            device=device,
            mask_ratio=(
                self.config.encoder.mask_ratio if mask_ratio is None else mask_ratio
            ),
        )

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        self.ema.update(self.target_encoder, self.context_encoder)

    def set_ema_momentum(self, momentum: float) -> None:
        self.ema.set_momentum(momentum)

    def get_pretrain_parameters(self) -> list[nn.Parameter]:
        return list(self.context_encoder.parameters()) + list(
            self.predictor.parameters()
        )

    def get_classify_parameters(
        self, *, freeze_backbone: bool = True
    ) -> list[nn.Parameter]:
        if freeze_backbone:
            return list(self.classification_head.parameters())
        return list(self.context_encoder.parameters()) + list(
            self.classification_head.parameters()
        )

    def _validate_mask(self, *, mask: torch.Tensor, signal: torch.Tensor) -> None:
        if mask.shape != signal.shape:
            raise ValueError(
                f"mask shape 必须等于 signal shape={tuple(signal.shape)}，收到 {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise ValueError(f"mask 必须为 bool，收到 {mask.dtype}")

    def _init_classification_head(self) -> None:
        for module in self.classification_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def build_gene_jepa(
    *,
    encoder: GeneMAEConfig | None = None,
    **kwargs: Any,
) -> GeneJEPA:
    """Construct GeneJEPA from keyword config arguments."""

    config_kwargs = cast(dict[str, Any], kwargs)
    if encoder is None:
        encoder_kwargs = cast(dict[str, Any], config_kwargs.pop("encoder_kwargs", {}))
        encoder = GeneMAEConfig(**encoder_kwargs)
    return GeneJEPA(GeneJEPAConfig(encoder=encoder, **config_kwargs))


def jepa_train_step(
    model: GeneJEPA,
    signal: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    step: int = 0,
    total_steps: int = 1000,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Single GeneJEPA pretraining step."""

    model.train()
    model.target_encoder.eval()
    ema_momentum = cosine_ema_schedule(
        model.config.ema_momentum,
        model.config.ema_momentum_end,
        step,
        total_steps,
    )
    model.set_ema_momentum(ema_momentum)

    output = model.forward_pretrain(signal, mask=mask)
    if output.loss is None:
        raise RuntimeError("预训练输出缺少 loss")
    optimizer.zero_grad()
    output.loss.backward()
    optimizer.step()
    model.update_target_encoder()
    return {
        "loss": float(output.loss.item()),
        "cls_loss": float(output.cls_loss.item())
        if output.cls_loss is not None
        else 0.0,
        "token_loss": float(output.token_loss.item())
        if output.token_loss is not None
        else 0.0,
        "ema_momentum": float(ema_momentum),
    }


def _latent_prediction_loss(
    predicted_latents: torch.Tensor,
    target_latents: torch.Tensor,
    *,
    loss_type: PredictorLossName,
) -> torch.Tensor:
    if predicted_latents.shape != target_latents.shape:
        raise ValueError(
            "predicted_latents shape 必须等于 target_latents shape，"
            f"收到 {tuple(predicted_latents.shape)} 和 {tuple(target_latents.shape)}"
        )
    if loss_type == "l2":
        return F.mse_loss(predicted_latents, target_latents)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(predicted_latents, target_latents)
    raise ValueError(f"未知 loss_type: {loss_type!r}")


def _build_classification_head(config: GeneJEPAConfig) -> nn.Module:
    if config.classification_head == "linear":
        return nn.Linear(config.encoder.d_model, config.encoder.n_classes)
    if config.classification_head == "mlp":
        return nn.Sequential(
            nn.Linear(config.encoder.d_model, config.encoder.d_model),
            nn.GELU(),
            nn.Dropout(config.predictor_dropout),
            nn.Linear(config.encoder.d_model, config.encoder.n_classes),
        )
    raise ValueError(f"未知 classification_head: {config.classification_head!r}")
