from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import DTYPE_NP
from .exposure import validate_binomial_observations
from .infer import (
    PosteriorDistribution,
    _infer_from_arrays,
    _resolve_numpy_dtype,
    _resolve_distribution,
    _resolve_torch_dtype,
)
from .types import InferenceResult, PriorGrid


def _normalize_kbulk_payload(
    gene_names: list[str],
    counts: np.ndarray,
    effective_exposure: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    resolved_gene_names = [str(name) for name in gene_names]
    if not resolved_gene_names:
        raise ValueError("gene_names cannot be empty")
    if len(resolved_gene_names) != len(set(resolved_gene_names)):
        raise ValueError("gene_names must be unique")
    counts_array = np.asarray(counts, dtype=DTYPE_NP)
    exposure_array = np.asarray(effective_exposure, dtype=DTYPE_NP).reshape(-1)
    if counts_array.ndim != 2:
        raise ValueError(f"counts must be 2D, got shape={counts_array.shape}")
    if counts_array.shape != (exposure_array.shape[0], len(resolved_gene_names)):
        raise ValueError(
            "counts shape must equal (n_samples, n_genes), "
            f"got {counts_array.shape} vs {(exposure_array.shape[0], len(resolved_gene_names))}"
        )
    if np.any(~np.isfinite(counts_array)) or np.any(counts_array < 0):
        raise ValueError("counts must be finite and non-negative")
    if np.any(~np.isfinite(exposure_array)) or np.any(exposure_array <= 0):
        raise ValueError("effective_exposure must be finite and positive")
    return resolved_gene_names, counts_array, exposure_array


@dataclass(frozen=True, slots=True)
class KBulkBatch:
    gene_names: list[str]
    counts: np.ndarray
    effective_exposure: np.ndarray

    def __post_init__(self) -> None:
        gene_names, counts, effective_exposure = _normalize_kbulk_payload(
            list(self.gene_names),
            self.counts,
            self.effective_exposure,
        )
        object.__setattr__(self, "gene_names", gene_names)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "effective_exposure", effective_exposure)

    @property
    def n_samples(self) -> int:
        return int(self.effective_exposure.shape[0])

    @property
    def n_genes(self) -> int:
        return int(len(self.gene_names))

    def check_shape(self) -> None:
        _normalize_kbulk_payload(
            list(self.gene_names),
            self.counts,
            self.effective_exposure,
        )


def infer_kbulk_samples(
    gene_names: list[str],
    aggregated_counts: np.ndarray,
    effective_exposure: np.ndarray,
    prior: PriorGrid,
    *,
    device: str | torch.device = "cpu",
    include_posterior: bool = False,
    torch_dtype: str = "float64",
    result_dtype: str | None = None,
    posterior_distribution: PosteriorDistribution = "auto",
    nb_overdispersion: float = 0.01,
    compile_model: bool = True,
    observation_chunk_size: int | None = None,
) -> InferenceResult:
    batch = KBulkBatch(
        gene_names=list(gene_names),
        counts=aggregated_counts,
        effective_exposure=effective_exposure,
    )
    prior = prior.select_genes(batch.gene_names).as_gene_specific()
    resolved_distribution = _resolve_distribution(prior, posterior_distribution)

    device_obj = torch.device(device)
    dtype_obj = _resolve_torch_dtype(torch_dtype)
    result_dtype_name = torch_dtype if result_dtype is None else result_dtype
    result_dtype_obj = _resolve_numpy_dtype(result_dtype_name)
    if resolved_distribution == "binomial":
        validate_binomial_observations(batch.counts, batch.effective_exposure)

    return _infer_from_arrays(
        gene_names=list(batch.gene_names),
        counts=batch.counts,
        effective_exposure_values=(
            None if resolved_distribution == "poisson" else batch.effective_exposure
        ),
        prior=prior,
        resolved_distribution=resolved_distribution,
        device=device_obj,
        include_posterior=include_posterior,
        dtype=dtype_obj,
        result_dtype=result_dtype_obj,
        nb_overdispersion=nb_overdispersion,
        compile_model=compile_model,
        observation_chunk_size=observation_chunk_size,
    )


def infer_kbulk(
    batch: KBulkBatch,
    prior: PriorGrid,
    **kwargs: object,
) -> InferenceResult:
    return infer_kbulk_samples(
        batch.gene_names,
        aggregated_counts=batch.counts,
        effective_exposure=batch.effective_exposure,
        prior=prior,
        **kwargs,
    )


class KBulkAggregator:
    def __init__(self, prior: PriorGrid) -> None:
        self.prior = prior

    def infer(self, batch: KBulkBatch, **kwargs: object) -> InferenceResult:
        return infer_kbulk(batch, self.prior, **kwargs)


__all__ = [
    "KBulkAggregator",
    "KBulkBatch",
    "infer_kbulk",
    "infer_kbulk_samples",
]
