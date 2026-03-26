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
    "posterior_entropy",
    "prior_entropy",
    "mutual_information",
    "posterior",
]

ALL_CHANNELS: frozenset[SignalChannel] = frozenset(
    {
        "signal",
        "posterior_entropy",
        "prior_entropy",
        "mutual_information",
        "posterior",
    }
)
CORE_CHANNELS: frozenset[SignalChannel] = frozenset(
    {"signal", "posterior_entropy", "prior_entropy", "mutual_information"}
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
    posterior_entropy: np.ndarray  # (B, C)
    prior_entropy: np.ndarray  # (B, C)
    mutual_information: np.ndarray  # (B, C)


@dataclass(frozen=True, slots=True)
class PosteriorBatchReport:
    gene_names: list[str]
    support: np.ndarray
    prior_weights: np.ndarray
    signal: np.ndarray
    posterior_entropy: np.ndarray
    prior_entropy: np.ndarray
    mutual_information: np.ndarray
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


def _extract_entropy(post: np.ndarray) -> np.ndarray:
    entropy = -(post * np.log(np.clip(post, EPS, None))).sum(axis=-1)
    return np.clip(np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def _extract_entropy_t(post: TorchTensor) -> TorchTensor:
    entropy = -(post * torch.log(torch.clamp(post, min=EPS))).sum(dim=-1)
    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(entropy, min=0.0)


def _extract_prior_entropy(weights: np.ndarray) -> np.ndarray:
    entropy = -(weights * np.log(np.clip(weights, EPS, None))).sum(axis=-1)
    return np.clip(np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def _extract_prior_entropy_t(weights: TorchTensor) -> TorchTensor:
    entropy = -(weights * torch.log(torch.clamp(weights, min=EPS))).sum(dim=-1)
    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(entropy, min=0.0)


class Posterior:
    """从已拟合先验中提取 MAP 信号与后验指标。"""

    def __init__(
        self,
        gene_names: list[str],
        priors: GridDistribution,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
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
        self.device = torch.device(device)
        self._weights_t = torch.as_tensor(
            np.asarray(self._priors.weights, dtype=DTYPE_NP),
            dtype=DTYPE_TORCH,
            device=self.device,
        )
        self._grid_min_t_all = torch.as_tensor(
            np.asarray(self._priors.grid_min, dtype=DTYPE_NP),
            dtype=DTYPE_TORCH,
            device=self.device,
        )
        self._grid_max_t_all = torch.as_tensor(
            np.asarray(self._priors.grid_max, dtype=DTYPE_NP),
            dtype=DTYPE_TORCH,
            device=self.device,
        )
        self._totals_cache: dict[tuple[int, int], TorchTensor] = {}
        self._s_hat_scalar_cache: dict[int, TorchTensor] = {}
        self._base_grid_t = torch.linspace(
            0.0,
            1.0,
            self._priors.M,
            dtype=DTYPE_TORCH,
            device=self.device,
        )

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
            posterior_entropy=payload.pop("posterior_entropy"),
            prior_entropy=payload.pop("prior_entropy"),
            mutual_information=payload.pop("mutual_information"),
        )

    def summarize_batch(
        self,
        batch: GeneBatch,
        s_hat: float | np.ndarray,
        *,
        include_posterior: bool = False,
    ) -> PosteriorBatchReport:
        channels: set[SignalChannel] = {
            "signal",
            "posterior_entropy",
            "prior_entropy",
            "mutual_information",
        }
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
            posterior_entropy=payload.pop("posterior_entropy"),
            prior_entropy=payload.pop("prior_entropy"),
            mutual_information=payload.pop("mutual_information"),
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
        posterior_t, support_t, weights_t = self._compute_posterior(
            batch, indices, s_hat_np
        )
        map_idx_t = torch.argmax(posterior_t, dim=-1)
        support_expanded = support_t[:, None, :].expand(-1, batch.C, -1)

        result: dict[str, np.ndarray] = {}
        if "signal" in requested:
            signal_t = torch.gather(
                support_expanded, 2, map_idx_t.unsqueeze(-1)
            ).squeeze(-1)
            result["signal"] = signal_t.cpu().numpy()
        posterior_entropy_t: TorchTensor | None = None
        prior_entropy_t: TorchTensor | None = None
        if "posterior_entropy" in requested or "mutual_information" in requested:
            posterior_entropy_t = _extract_entropy_t(posterior_t)
        if "posterior_entropy" in requested:
            if posterior_entropy_t is None:
                raise RuntimeError("posterior entropy 提取失败")
            posterior_entropy = posterior_entropy_t
            result["posterior_entropy"] = posterior_entropy.cpu().numpy()
        if "prior_entropy" in requested or "mutual_information" in requested:
            prior_entropy_t = _extract_prior_entropy_t(weights_t)
        if "prior_entropy" in requested:
            if prior_entropy_t is None:
                raise RuntimeError("prior entropy 提取失败")
            prior_entropy = prior_entropy_t
            result["prior_entropy"] = (
                prior_entropy[:, None].expand(-1, batch.C).cpu().numpy()
            )
        if "mutual_information" in requested:
            if prior_entropy_t is None or posterior_entropy_t is None:
                raise RuntimeError("mutual information 提取失败")
            prior_entropy = prior_entropy_t
            posterior_entropy = posterior_entropy_t
            result["mutual_information"] = (
                torch.clamp(prior_entropy[:, None] - posterior_entropy, min=0.0)
                .cpu()
                .numpy()
            )
        if "posterior" in requested:
            result["posterior"] = posterior_t.cpu().numpy()

        if channels is not None and "posterior" in channels:
            result["support"] = support_t.cpu().numpy()
            result["prior_weights"] = weights_t.cpu().numpy()

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
        indices_t = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        grid_min_t = self._grid_min_t_all.index_select(0, indices_t)
        grid_max_t = self._grid_max_t_all.index_select(0, indices_t)
        support_t = (
            grid_min_t[:, None]
            + (grid_max_t - grid_min_t)[:, None] * self._base_grid_t[None, :]
        )
        return support_t.cpu().numpy()

    def _prior_weights_for(self, batch: GeneBatch) -> np.ndarray:
        indices = self._resolve_genes(batch.gene_names)
        indices_t = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return self._weights_t.index_select(0, indices_t).cpu().numpy()

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
    ) -> tuple[TorchTensor, TorchTensor, TorchTensor]:
        indices_t = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        weights_t = self._weights_t.index_select(0, indices_t)
        grid_min_t = self._grid_min_t_all.index_select(0, indices_t)
        grid_max_t = self._grid_max_t_all.index_select(0, indices_t)
        support_t = (
            grid_min_t[:, None]
            + (grid_max_t - grid_min_t)[:, None] * self._base_grid_t[None, :]
        )

        with torch.inference_mode():
            counts_t = torch.as_tensor(
                batch.counts.T,
                dtype=DTYPE_TORCH,
                device=self.device,
            )
            totals_t = self._totals_tensor(batch.totals)
            s_hat_t = self._s_hat_tensor(s_hat, batch.C)
            x = counts_t.unsqueeze(-1)
            n = totals_t.unsqueeze(-1)
            p = (support_t[:, None, :] / s_hat_t).clamp(EPS, 1.0 - EPS)
            log_coeff = (
                torch.lgamma(n + 1.0)
                - torch.lgamma(x + 1.0)
                - torch.lgamma(n - x + 1.0)
            )
            log_lik = log_coeff + x * torch.log(p) + (n - x) * torch.log1p(-p)
            posterior_t = compute_posterior(log_lik, weights_t)

        return posterior_t, support_t, weights_t

    def _totals_tensor(self, totals: np.ndarray) -> TorchTensor:
        key = (int(totals.shape[0]), int(totals.__array_interface__["data"][0]))
        cached = self._totals_cache.get(key)
        if cached is not None:
            return cached
        tensor = torch.as_tensor(
            totals,
            dtype=DTYPE_TORCH,
            device=self.device,
        ).unsqueeze(0)
        self._totals_cache[key] = tensor
        return tensor

    def _s_hat_tensor(self, s_hat: np.ndarray, n_cells: int) -> TorchTensor:
        if s_hat.ndim == 1 and np.all(s_hat == s_hat[0]):
            key = n_cells
            cached = self._s_hat_scalar_cache.get(key)
            if cached is not None:
                return cached
            tensor = torch.full(
                (1, n_cells, 1),
                float(s_hat[0]),
                dtype=DTYPE_TORCH,
                device=self.device,
            )
            self._s_hat_scalar_cache[key] = tensor
            return tensor
        return (
            torch.as_tensor(
                s_hat,
                dtype=DTYPE_TORCH,
                device=self.device,
            )
            .unsqueeze(0)
            .unsqueeze(-1)
        )


SignalExtractor = Posterior
