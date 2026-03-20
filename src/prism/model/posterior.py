"""后验推断与信号提取。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from ._kernel import posterior as compute_posterior
from ._typing import (
    DTYPE_NP,
    DTYPE_TORCH,
    EPS,
    GeneBatch,
    GridDistribution,
    TorchTensor,
)

SignalChannel = Literal[
    "signal",
    "confidence",
    "surprisal",
    "surprisal_norm",
    "sharpness",
    "posterior",
]

ALL_CHANNELS: frozenset[SignalChannel] = frozenset(
    {"signal", "confidence", "surprisal", "surprisal_norm", "sharpness", "posterior"}
)
CORE_CHANNELS: frozenset[SignalChannel] = frozenset(
    {"signal", "confidence", "surprisal"}
)

__all__ = [
    "ALL_CHANNELS",
    "CORE_CHANNELS",
    "Posterior",
    "PosteriorBatchReport",
    "PosteriorSummary",
    "SignalChannel",
    "SignalExtractor",
]


@dataclass(frozen=True, slots=True)
class PosteriorSummary:
    gene_names: list[str]
    support: np.ndarray  # (B, M)
    prior_weights: np.ndarray  # (B, M)
    posterior: np.ndarray  # (B, C, M)
    signal: np.ndarray  # (B, C)
    confidence: np.ndarray  # (B, C)
    surprisal: np.ndarray  # (B, C)
    surprisal_norm: np.ndarray | None = None
    sharpness: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class PosteriorBatchReport:
    gene_names: list[str]
    support: np.ndarray
    prior_weights: np.ndarray
    signal: np.ndarray
    confidence: np.ndarray
    surprisal: np.ndarray
    surprisal_norm: np.ndarray | None = None
    sharpness: np.ndarray | None = None
    posterior: np.ndarray | None = None


def _validate_channels(
    channels: set[SignalChannel] | None,
) -> tuple[SignalChannel, ...]:
    if channels is None:
        return tuple(CORE_CHANNELS)

    unknown = set(channels) - set(ALL_CHANNELS)
    if unknown:
        raise ValueError(f"未知通道: {sorted(unknown)}")
    return tuple(channels)


def _gather_at_index(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    batch_idx = np.arange(values.shape[0])[:, None]
    return values[batch_idx, index]


def _normalize_s_hat(s_hat: float | np.ndarray, n_cells: int) -> np.ndarray:
    s_hat_np = np.asarray(s_hat, dtype=DTYPE_NP)
    if s_hat_np.ndim == 0:
        s_hat_np = np.full(n_cells, float(s_hat_np), dtype=DTYPE_NP)
    else:
        s_hat_np = s_hat_np.reshape(-1)

    if s_hat_np.shape != (n_cells,):
        raise ValueError(f"s_hat shape 期望 ({n_cells},)，收到 {s_hat_np.shape}")
    if np.any(~np.isfinite(s_hat_np)) or np.any(s_hat_np <= 0):
        raise ValueError("s_hat 必须为正且全部有限")
    return s_hat_np


def _extract_confidence(post: np.ndarray) -> np.ndarray:
    grid_size = post.shape[-1]
    entropy = -(post * np.log(np.clip(post, EPS, None))).sum(axis=-1)
    return np.clip(1.0 - entropy / math.log(grid_size), 0.0, 1.0)


def _extract_sharpness_curve(weights: np.ndarray) -> np.ndarray:
    log_weights = np.log(np.clip(weights, EPS, None))
    sharpness_curve = np.zeros_like(log_weights)
    if weights.shape[-1] >= 3:
        sharpness_curve[:, 1:-1] = -(
            log_weights[:, :-2] - 2.0 * log_weights[:, 1:-1] + log_weights[:, 2:]
        )
        sharpness_curve[:, 0] = sharpness_curve[:, 1]
        sharpness_curve[:, -1] = sharpness_curve[:, -2]
    return np.clip(sharpness_curve, 0.0, None)


class Posterior:
    """从已拟合先验中提取 MAP 信号与后验指标。"""

    def __init__(self, gene_names: list[str], priors: GridDistribution) -> None:
        if not gene_names:
            raise ValueError("gene_names 不能为空")
        if len(gene_names) != len(set(gene_names)):
            raise ValueError("gene_names 必须去重")

        priors = priors.batched()
        priors.check_shape()
        if priors.B != len(gene_names):
            raise ValueError(
                f"gene_names 长度 ({len(gene_names)}) 与 priors.B ({priors.B}) 不匹配"
            )

        self.gene_names = list(gene_names)
        self._gene_to_idx = {name: idx for idx, name in enumerate(self.gene_names)}
        self._priors = priors
        self._base_grid_t = torch.linspace(0.0, 1.0, self._priors.M, dtype=DTYPE_TORCH)

    def query(
        self,
        gene_name: str,
        x_vals: np.ndarray,
        n_vals: np.ndarray,
        s_hat: float | np.ndarray,
        channels: set[SignalChannel] | None = None,
    ) -> dict[str, np.ndarray]:
        batch = GeneBatch(
            gene_names=[gene_name],
            counts=np.asarray(x_vals, dtype=DTYPE_NP).reshape(-1, 1),
            totals=np.asarray(n_vals, dtype=DTYPE_NP).reshape(-1),
        )
        result = self.extract(batch, s_hat, channels)
        return {key: value[0] for key, value in result.items()}

    def summarize(
        self,
        batch: GeneBatch,
        s_hat: float | np.ndarray,
    ) -> PosteriorSummary:
        payload = self.extract(batch, s_hat, channels=set(ALL_CHANNELS))
        return PosteriorSummary(
            gene_names=list(batch.gene_names),
            support=payload.pop("support"),
            prior_weights=payload.pop("prior_weights"),
            posterior=payload.pop("posterior"),
            signal=payload.pop("signal"),
            confidence=payload.pop("confidence"),
            surprisal=payload.pop("surprisal"),
            surprisal_norm=payload.pop("surprisal_norm"),
            sharpness=payload.pop("sharpness"),
        )

    def summarize_batch(
        self,
        batch: GeneBatch,
        s_hat: float | np.ndarray,
        *,
        include_posterior: bool = False,
        include_surprisal_norm: bool = True,
        include_sharpness: bool = True,
    ) -> PosteriorBatchReport:
        channels: set[SignalChannel] = {"signal", "confidence", "surprisal"}
        if include_surprisal_norm:
            channels.add("surprisal_norm")
        if include_sharpness:
            channels.add("sharpness")
        if include_posterior:
            channels.add("posterior")

        payload = self.extract(batch, s_hat, channels=channels)
        return PosteriorBatchReport(
            gene_names=list(batch.gene_names),
            support=payload.pop("support")
            if include_posterior
            else self._support_for(batch),
            prior_weights=payload.pop("prior_weights")
            if include_posterior
            else self._prior_weights_for(batch),
            signal=payload.pop("signal"),
            confidence=payload.pop("confidence"),
            surprisal=payload.pop("surprisal"),
            surprisal_norm=payload.pop("surprisal_norm", None),
            sharpness=payload.pop("sharpness", None),
            posterior=payload.pop("posterior", None),
        )

    def extract(
        self,
        batch: GeneBatch,
        s_hat: float | np.ndarray,
        channels: set[SignalChannel] | None = None,
    ) -> dict[str, np.ndarray]:
        requested = _validate_channels(channels)

        batch.check_shape()
        self._validate_batch(batch)
        s_hat_np = _normalize_s_hat(s_hat, batch.C)

        indices = self._resolve_genes(batch.gene_names)
        posterior_np, support_np, weights_np = self._compute_posterior(
            batch, indices, s_hat_np
        )
        map_idx = np.argmax(posterior_np, axis=-1)

        result: dict[str, np.ndarray] = {}
        if "signal" in requested:
            result["signal"] = _gather_at_index(support_np, map_idx)
        if "confidence" in requested:
            result["confidence"] = _extract_confidence(posterior_np)
        if "surprisal" in requested:
            result["surprisal"] = -np.log(
                np.clip(_gather_at_index(weights_np, map_idx), EPS, None)
            )
        if "surprisal_norm" in requested:
            surprisal = result.get("surprisal")
            if surprisal is None:
                surprisal = -np.log(
                    np.clip(_gather_at_index(weights_np, map_idx), EPS, None)
                )
            max_surprisal = -np.log(np.clip(weights_np, EPS, None)).max(axis=-1)
            result["surprisal_norm"] = surprisal / np.maximum(
                max_surprisal[:, None], EPS
            )
        if "sharpness" in requested:
            sharpness_curve = _extract_sharpness_curve(weights_np)
            result["sharpness"] = _gather_at_index(sharpness_curve, map_idx)
        if "posterior" in requested:
            result["posterior"] = posterior_np

        if channels is not None and "posterior" in channels:
            result["support"] = support_np
            result["prior_weights"] = weights_np

        return result

    def _resolve_genes(self, gene_names: list[str]) -> np.ndarray:
        indices = []
        for name in gene_names:
            if name not in self._gene_to_idx:
                raise KeyError(f"基因 {name!r} 不在 extractor 中")
            indices.append(self._gene_to_idx[name])
        return np.asarray(indices, dtype=np.int64)

    def _support_for(self, batch: GeneBatch) -> np.ndarray:
        indices = self._resolve_genes(batch.gene_names)
        grid_min_np = np.asarray(self._priors.grid_min)[indices].astype(
            DTYPE_NP, copy=False
        )
        grid_max_np = np.asarray(self._priors.grid_max)[indices].astype(
            DTYPE_NP, copy=False
        )
        grid_min_t = torch.as_tensor(grid_min_np, dtype=DTYPE_TORCH)
        grid_max_t = torch.as_tensor(grid_max_np, dtype=DTYPE_TORCH)
        support_t = (
            grid_min_t[:, None]
            + (grid_max_t - grid_min_t)[:, None] * self._base_grid_t[None, :]
        )
        return support_t.cpu().numpy()

    def _prior_weights_for(self, batch: GeneBatch) -> np.ndarray:
        indices = self._resolve_genes(batch.gene_names)
        return np.asarray(self._priors.weights[indices], dtype=DTYPE_NP)

    def _validate_batch(self, batch: GeneBatch) -> None:
        if batch.counts.ndim != 2:
            raise ValueError(f"counts 必须为二维，收到 shape={batch.counts.shape}")
        if batch.totals.ndim != 1:
            raise ValueError(f"totals 必须为一维，收到 shape={batch.totals.shape}")
        if np.any(batch.counts < 0) or np.any(batch.totals <= 0):
            raise ValueError("counts 不能为负，totals 必须为正")
        if np.any(batch.counts > batch.totals[:, None]):
            raise ValueError("要求逐元素满足 counts <= totals")

    def _compute_posterior(
        self,
        batch: GeneBatch,
        indices: np.ndarray,
        s_hat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights_np = np.asarray(self._priors.weights[indices], dtype=DTYPE_NP)
        grid_min_np = np.asarray(self._priors.grid_min)[indices].astype(
            DTYPE_NP, copy=False
        )
        grid_max_np = np.asarray(self._priors.grid_max)[indices].astype(
            DTYPE_NP, copy=False
        )

        grid_min_t = torch.as_tensor(grid_min_np, dtype=DTYPE_TORCH)
        grid_max_t = torch.as_tensor(grid_max_np, dtype=DTYPE_TORCH)
        support_t = (
            grid_min_t[:, None]
            + (grid_max_t - grid_min_t)[:, None] * self._base_grid_t[None, :]
        )

        counts_t = torch.as_tensor(batch.counts.T, dtype=DTYPE_TORCH)
        totals_t = torch.as_tensor(batch.totals, dtype=DTYPE_TORCH).unsqueeze(0)
        s_hat_t = torch.as_tensor(s_hat, dtype=DTYPE_TORCH).unsqueeze(0).unsqueeze(-1)
        weights_t = torch.as_tensor(weights_np, dtype=DTYPE_TORCH)

        x = counts_t.unsqueeze(-1)
        n = totals_t.unsqueeze(-1)
        p = (support_t[:, None, :] / s_hat_t).clamp(EPS, 1.0 - EPS)
        log_coeff = (
            torch.lgamma(n + 1.0) - torch.lgamma(x + 1.0) - torch.lgamma(n - x + 1.0)
        )
        log_lik = log_coeff + x * torch.log(p) + (n - x) * torch.log1p(-p)
        posterior_t = compute_posterior(log_lik, weights_t)

        return (
            posterior_t.cpu().numpy(),
            support_t.cpu().numpy(),
            weights_np,
        )


SignalExtractor = Posterior
