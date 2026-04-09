from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from .constants import DTYPE_NP
from .exposure import validate_binomial_observations
from .infer import (
    PosteriorDistribution,
    _BinomialPosteriorInferencer,
    _NegativeBinomialPosteriorInferencer,
    _PoissonPosteriorInferencer,
    _get_inferencer,
    _resolve_distribution,
)
from .numeric import (
    entropy,
)
from .types import InferenceResult, PriorGrid


@dataclass(frozen=True, slots=True)
class KBulkBatch:
    gene_names: list[str]
    counts: np.ndarray
    effective_exposure: np.ndarray

    def __post_init__(self) -> None:
        self.check_shape()

    @property
    def n_samples(self) -> int:
        return int(np.asarray(self.effective_exposure).reshape(-1).shape[0])

    @property
    def n_genes(self) -> int:
        return int(len(self.gene_names))

    def check_shape(self) -> None:
        if not self.gene_names:
            raise ValueError("gene_names cannot be empty")
        if len(self.gene_names) != len(set(self.gene_names)):
            raise ValueError("gene_names must be unique")
        counts = np.asarray(self.counts, dtype=DTYPE_NP)
        exposure = np.asarray(self.effective_exposure, dtype=DTYPE_NP).reshape(-1)
        if counts.ndim != 2:
            raise ValueError(f"counts must be 2D, got shape={counts.shape}")
        if counts.shape != (exposure.shape[0], len(self.gene_names)):
            raise ValueError(
                "counts shape must equal (n_samples, n_genes), "
                f"got {counts.shape} vs {(exposure.shape[0], len(self.gene_names))}"
            )
        if np.any(~np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("counts must be finite and non-negative")
        if np.any(~np.isfinite(exposure)) or np.any(exposure <= 0):
            raise ValueError("effective_exposure must be finite and positive")

def _resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported torch_dtype: {name}")


def infer_kbulk_samples(
    gene_names: list[str],
    aggregated_counts: np.ndarray,
    effective_exposure: np.ndarray,
    prior: PriorGrid,
    *,
    device: str | torch.device = "cpu",
    include_posterior: bool = False,
    torch_dtype: str = "float64",
    posterior_distribution: PosteriorDistribution = "auto",
    nb_overdispersion: float = 0.01,
    compile_model: bool = True,
) -> InferenceResult:
    batch = KBulkBatch(
        gene_names=list(gene_names),
        counts=np.asarray(aggregated_counts, dtype=DTYPE_NP),
        effective_exposure=np.asarray(effective_exposure, dtype=DTYPE_NP),
    )
    prior = prior.select_genes(batch.gene_names).as_gene_specific()
    resolved_distribution = _resolve_distribution(prior, posterior_distribution)

    device_obj = torch.device(device)
    dtype_obj = _resolve_torch_dtype(torch_dtype)
    inferencer = _get_inferencer(
        resolved_distribution,
        device=device_obj,
        dtype=dtype_obj,
        nb_overdispersion=nb_overdispersion,
        compile_model=compile_model,
    )
    counts_t = torch.as_tensor(batch.counts.T, dtype=dtype_obj, device=device_obj)
    support_t = torch.as_tensor(prior.support, dtype=dtype_obj, device=device_obj)
    prior_probabilities_t = torch.as_tensor(
        prior.prior_probabilities, dtype=dtype_obj, device=device_obj
    )

    if resolved_distribution == "poisson":
        (
            posterior_probabilities_t,
            map_support_t,
            posterior_entropy_t,
            prior_entropy_t,
            mutual_information_t,
        ) = cast(_PoissonPosteriorInferencer, inferencer)(
            counts_t,
            support_t,
            prior_probabilities_t,
        )
    else:
        if resolved_distribution == "binomial":
            validate_binomial_observations(batch.counts, batch.effective_exposure)
        exposure_t = (
            torch.as_tensor(
                batch.effective_exposure, dtype=dtype_obj, device=device_obj
            )
            .unsqueeze(0)
            .expand(batch.n_genes, -1)
        )
        (
            posterior_probabilities_t,
            map_support_t,
            posterior_entropy_t,
            prior_entropy_t,
            mutual_information_t,
        ) = cast(
            _BinomialPosteriorInferencer | _NegativeBinomialPosteriorInferencer,
            inferencer,
        )(
            counts_t,
            support_t,
            prior_probabilities_t,
            exposure_t,
        )

    return InferenceResult(
        gene_names=list(batch.gene_names),
        support_domain=prior.support_domain,
        support=np.asarray(prior.support, dtype=DTYPE_NP),
        prior_probabilities=np.asarray(prior.prior_probabilities, dtype=DTYPE_NP),
        map_support=map_support_t.detach().cpu().numpy().T.astype(DTYPE_NP, copy=False),
        posterior_entropy=posterior_entropy_t.detach()
        .cpu()
        .numpy()
        .T.astype(DTYPE_NP, copy=False),
        prior_entropy=prior_entropy_t.detach()
        .cpu()
        .numpy()
        .T.astype(DTYPE_NP, copy=False),
        mutual_information=mutual_information_t.detach()
        .cpu()
        .numpy()
        .T.astype(DTYPE_NP, copy=False),
        posterior_probabilities=(
            posterior_probabilities_t.detach()
            .cpu()
            .numpy()
            .transpose(1, 0, 2)
            .astype(DTYPE_NP, copy=False)
            if include_posterior
            else None
        ),
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
