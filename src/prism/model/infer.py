from __future__ import annotations

from typing import Literal, cast

import numpy as np
import torch

from .exposure import effective_exposure, validate_binomial_observations
from .numeric import (
    entropy,
    log_binomial_likelihood_support,
    log_negative_binomial_likelihood_support,
    log_poisson_likelihood_support,
    posterior_from_log_likelihood,
)
from .types import GeneBatch, InferenceResult, ObservationBatch, PriorGrid

PosteriorDistribution = Literal["auto", "binomial", "negative_binomial", "poisson"]
ResolvedPosteriorDistribution = Literal["binomial", "negative_binomial", "poisson"]
_CPU_INFERENCE_TARGET_BYTES = 2 * 1024**3


def _resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported torch_dtype: {name}")


def _resolve_numpy_dtype(name: str) -> np.dtype:
    if name == "float32":
        return np.dtype(np.float32)
    if name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"unsupported result_dtype: {name}")


def _resolve_distribution(
    prior: PriorGrid,
    posterior_distribution: PosteriorDistribution,
) -> Literal["binomial", "negative_binomial", "poisson"]:
    resolved = (
        prior.distribution_name
        if posterior_distribution == "auto"
        else posterior_distribution
    )
    if resolved != prior.distribution_name:
        raise ValueError(
            "posterior distribution mismatch with prior distribution; "
            f"prior={prior.distribution_name!r}, posterior={resolved!r}"
        )
    if prior.support_domain == "rate" and resolved != "poisson":
        raise ValueError(
            "rate support requires poisson posterior inference; "
            f"got posterior_distribution={resolved!r}"
        )
    if prior.support_domain == "probability" and resolved == "poisson":
        raise ValueError("poisson posterior inference requires rate support")
    return cast(Literal["binomial", "negative_binomial", "poisson"], resolved)


class _BasePosteriorInferencer(torch.nn.Module):
    def _summarize(
        self,
        posterior_probabilities: torch.Tensor,
        support: torch.Tensor,
        prior_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        map_index = torch.argmax(posterior_probabilities, dim=-1)
        map_support = torch.gather(support, 1, map_index)
        posterior_entropy = cast(torch.Tensor, entropy(posterior_probabilities))
        prior_entropy = cast(torch.Tensor, entropy(prior_probabilities))
        mutual_information = torch.clamp(
            prior_entropy[:, None] - posterior_entropy,
            min=0.0,
        )
        return map_support, posterior_entropy, mutual_information


class _BinomialPosteriorInferencer(_BasePosteriorInferencer):
    def forward(
        self,
        counts: torch.Tensor,
        support: torch.Tensor,
        prior_probabilities: torch.Tensor,
        effective_exposure_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log_likelihood = log_binomial_likelihood_support(
            counts,
            effective_exposure_values,
            support,
        )
        posterior_probabilities = posterior_from_log_likelihood(
            log_likelihood,
            prior_probabilities,
        )
        posterior_probabilities = cast(torch.Tensor, posterior_probabilities)
        map_support, posterior_entropy, mutual_information = (
            self._summarize(
                posterior_probabilities,
                support,
                prior_probabilities,
            )
        )
        return (
            posterior_probabilities,
            map_support,
            posterior_entropy,
            mutual_information,
        )


class _NegativeBinomialPosteriorInferencer(_BasePosteriorInferencer):
    def __init__(self, overdispersion: float) -> None:
        super().__init__()
        self.overdispersion = float(overdispersion)

    def forward(
        self,
        counts: torch.Tensor,
        support: torch.Tensor,
        prior_probabilities: torch.Tensor,
        effective_exposure_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log_likelihood = log_negative_binomial_likelihood_support(
            counts,
            effective_exposure_values,
            support,
            overdispersion=self.overdispersion,
        )
        posterior_probabilities = posterior_from_log_likelihood(
            log_likelihood,
            prior_probabilities,
        )
        posterior_probabilities = cast(torch.Tensor, posterior_probabilities)
        map_support, posterior_entropy, mutual_information = (
            self._summarize(
                posterior_probabilities,
                support,
                prior_probabilities,
            )
        )
        return (
            posterior_probabilities,
            map_support,
            posterior_entropy,
            mutual_information,
        )


class _PoissonPosteriorInferencer(_BasePosteriorInferencer):
    def forward(
        self,
        counts: torch.Tensor,
        support: torch.Tensor,
        prior_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log_likelihood = log_poisson_likelihood_support(counts, support)
        posterior_probabilities = posterior_from_log_likelihood(
            log_likelihood,
            prior_probabilities,
        )
        posterior_probabilities = cast(torch.Tensor, posterior_probabilities)
        map_support, posterior_entropy, mutual_information = (
            self._summarize(
                posterior_probabilities,
                support,
                prior_probabilities,
            )
        )
        return (
            posterior_probabilities,
            map_support,
            posterior_entropy,
            mutual_information,
        )


_COMPILED_INFERENCERS: dict[
    tuple[str, str, str, float | None, bool],
    torch.nn.Module,
] = {}


def _cache_device_key(device: torch.device) -> str:
    return str(device)


def _maybe_compile(module: torch.nn.Module) -> torch.nn.Module:
    try:
        return cast(torch.nn.Module, torch.compile(module))
    except Exception:
        return module


def _get_inferencer(
    distribution: ResolvedPosteriorDistribution,
    *,
    device: torch.device,
    dtype: torch.dtype,
    nb_overdispersion: float,
    compile_model: bool,
) -> torch.nn.Module:
    dtype_name = str(dtype).removeprefix("torch.")
    overdispersion_key = (
        nb_overdispersion if distribution == "negative_binomial" else None
    )
    cache_key = (
        distribution,
        _cache_device_key(device),
        dtype_name,
        overdispersion_key,
        bool(compile_model),
    )
    cached = _COMPILED_INFERENCERS.get(cache_key)
    if cached is not None:
        return cached

    if distribution == "binomial":
        module: torch.nn.Module = _BinomialPosteriorInferencer()
    elif distribution == "negative_binomial":
        module = _NegativeBinomialPosteriorInferencer(nb_overdispersion)
    else:
        module = _PoissonPosteriorInferencer()

    module = module.to(device=device, dtype=dtype)
    compiled = _maybe_compile(module) if compile_model else module
    _COMPILED_INFERENCERS[cache_key] = compiled
    return compiled


def _estimate_inference_working_set_bytes(
    *,
    n_genes: int,
    n_observations: int,
    n_support_points: int,
    dtype: torch.dtype,
) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    return int(n_genes * n_observations * n_support_points * element_size * 4)


def _resolve_observation_chunk_size(
    *,
    n_observations: int,
    n_genes: int,
    n_support_points: int,
    dtype: torch.dtype,
    device: torch.device,
    observation_chunk_size: int | None,
) -> int:
    if n_observations < 1:
        raise ValueError("n_observations must be >= 1")
    if observation_chunk_size is not None:
        if observation_chunk_size < 1:
            raise ValueError("observation_chunk_size must be >= 1")
        return min(int(observation_chunk_size), n_observations)

    estimated_bytes = _estimate_inference_working_set_bytes(
        n_genes=n_genes,
        n_observations=n_observations,
        n_support_points=n_support_points,
        dtype=dtype,
    )
    if device.type == "cuda":
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        target_bytes = max(1, int(total_memory * 0.35))
    else:
        target_bytes = _CPU_INFERENCE_TARGET_BYTES
    if estimated_bytes <= target_bytes:
        return n_observations
    bytes_per_observation = max(1, estimated_bytes // n_observations)
    return max(1, min(n_observations, target_bytes // bytes_per_observation))


def _iter_observation_slices(n_observations: int, chunk_size: int) -> list[slice]:
    return [
        slice(start, min(start + chunk_size, n_observations))
        for start in range(0, n_observations, chunk_size)
    ]


def _to_numpy_chunk(
    values: torch.Tensor,
    *,
    result_dtype: np.dtype,
) -> np.ndarray:
    return values.detach().cpu().numpy().astype(result_dtype, copy=False)


def _infer_from_arrays(
    *,
    gene_names: list[str],
    counts: np.ndarray,
    effective_exposure_values: np.ndarray | None,
    prior: PriorGrid,
    resolved_distribution: ResolvedPosteriorDistribution,
    device: torch.device,
    include_posterior: bool,
    dtype: torch.dtype,
    result_dtype: np.dtype,
    nb_overdispersion: float,
    compile_model: bool,
    observation_chunk_size: int | None,
) -> InferenceResult:
    n_observations, n_genes = counts.shape
    support_np = np.asarray(prior.support)
    prior_probabilities_np = np.asarray(prior.prior_probabilities)
    chunk_size = _resolve_observation_chunk_size(
        n_observations=n_observations,
        n_genes=n_genes,
        n_support_points=int(support_np.shape[1]),
        dtype=dtype,
        device=device,
        observation_chunk_size=observation_chunk_size,
    )
    inferencer = _get_inferencer(
        resolved_distribution,
        device=device,
        dtype=dtype,
        nb_overdispersion=nb_overdispersion,
        compile_model=compile_model,
    )

    counts_t = torch.as_tensor(counts, dtype=dtype, device=device).transpose(0, 1)
    support_t = torch.as_tensor(support_np, dtype=dtype, device=device)
    prior_probabilities_t = torch.as_tensor(
        prior_probabilities_np,
        dtype=dtype,
        device=device,
    )
    prior_entropy_vector_np = _to_numpy_chunk(
        cast(torch.Tensor, entropy(prior_probabilities_t)),
        result_dtype=result_dtype,
    )
    if resolved_distribution == "poisson":
        exposure_t = None
    else:
        exposure_t = (
            torch.as_tensor(
                effective_exposure_values,
                dtype=dtype,
                device=device,
            )
            .unsqueeze(0)
            .expand(n_genes, -1)
        )

    map_support_np = np.empty((n_observations, n_genes), dtype=result_dtype)
    posterior_entropy_np = np.empty((n_observations, n_genes), dtype=result_dtype)
    prior_entropy_np = np.empty((n_observations, n_genes), dtype=result_dtype)
    prior_entropy_np[:] = prior_entropy_vector_np
    mutual_information_np = np.empty((n_observations, n_genes), dtype=result_dtype)
    posterior_probabilities_np = (
        np.empty((n_observations, n_genes, support_np.shape[1]), dtype=result_dtype)
        if include_posterior
        else None
    )

    with torch.inference_mode():
        for observation_slice in _iter_observation_slices(n_observations, chunk_size):
            chunk_start = int(observation_slice.start or 0)
            chunk_stop = int(observation_slice.stop or n_observations)
            counts_chunk = counts_t[:, observation_slice]
            if resolved_distribution == "poisson":
                (
                    posterior_probabilities_t,
                    map_support_t,
                    posterior_entropy_t,
                    mutual_information_t,
                ) = cast(_PoissonPosteriorInferencer, inferencer)(
                    counts_chunk,
                    support_t,
                    prior_probabilities_t,
                )
            else:
                (
                    posterior_probabilities_t,
                    map_support_t,
                    posterior_entropy_t,
                    mutual_information_t,
                ) = cast(
                    _BinomialPosteriorInferencer | _NegativeBinomialPosteriorInferencer,
                    inferencer,
                )(
                    counts_chunk,
                    support_t,
                    prior_probabilities_t,
                    cast(torch.Tensor, exposure_t)[:, observation_slice],
                )

            map_support_np[chunk_start:chunk_stop] = _to_numpy_chunk(
                map_support_t.transpose(0, 1),
                result_dtype=result_dtype,
            )
            posterior_entropy_np[chunk_start:chunk_stop] = _to_numpy_chunk(
                posterior_entropy_t.transpose(0, 1),
                result_dtype=result_dtype,
            )
            mutual_information_np[chunk_start:chunk_stop] = _to_numpy_chunk(
                mutual_information_t.transpose(0, 1),
                result_dtype=result_dtype,
            )
            if posterior_probabilities_np is not None:
                posterior_probabilities_np[chunk_start:chunk_stop] = _to_numpy_chunk(
                    posterior_probabilities_t.permute(1, 0, 2),
                    result_dtype=result_dtype,
                )

    return InferenceResult(
        gene_names=list(gene_names),
        support_domain=prior.support_domain,
        support=np.asarray(support_np, dtype=result_dtype),
        prior_probabilities=np.asarray(prior_probabilities_np, dtype=result_dtype),
        map_support=map_support_np,
        posterior_entropy=posterior_entropy_np,
        prior_entropy=prior_entropy_np,
        mutual_information=mutual_information_np,
        posterior_probabilities=posterior_probabilities_np,
    )


def infer_posteriors(
    batch: ObservationBatch | GeneBatch,
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
    if isinstance(batch, GeneBatch):
        batch = batch.to_observation_batch()
    batch.check_shape()
    prior = prior.select_genes(batch.gene_names).as_gene_specific()
    resolved_distribution = _resolve_distribution(prior, posterior_distribution)

    device_obj = torch.device(device)
    dtype_obj = _resolve_torch_dtype(torch_dtype)
    result_dtype_name = torch_dtype if result_dtype is None else result_dtype
    result_dtype_obj = _resolve_numpy_dtype(result_dtype_name)
    exposure_values = None
    if resolved_distribution != "poisson":
        exposure_values = effective_exposure(batch.reference_counts, prior.scale)
        if resolved_distribution == "binomial":
            validate_binomial_observations(batch.counts, exposure_values)

    return _infer_from_arrays(
        gene_names=list(batch.gene_names),
        counts=batch.counts,
        effective_exposure_values=exposure_values,
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


__all__ = ["PosteriorDistribution", "infer_posteriors"]
