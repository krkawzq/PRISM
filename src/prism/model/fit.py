from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Literal, cast

import numpy as np
import torch

from .constants import DTYPE_NP, EPS
from .exposure import (
    effective_exposure,
    mean_reference_count,
    validate_binomial_observations,
)
from .numeric import (
    _BinomialObservationTerms,
    _NegativeBinomialObservationTerms,
    _PoissonObservationTerms,
    _ProbabilitySupportTerms,
    _RateSupportTerms,
    _binomial_observation_terms,
    _log_binomial_likelihood_from_terms,
    _log_negative_binomial_likelihood_from_terms,
    _log_poisson_likelihood_from_terms,
    _negative_binomial_observation_terms,
    _poisson_observation_terms,
    _probability_support_terms,
    _rate_support_terms,
)
from .types import (
    GeneBatch,
    ObservationBatch,
    PriorFitConfig,
    PriorFitResult,
    PriorGrid,
    make_distribution_grid,
)

FitProgressCallback = Callable[[int, int, float, float, float, bool], None]
LikelihoodName = Literal["binomial", "negative_binomial", "poisson"]
_CPU_LOG_LIKELIHOOD_CACHE_LIMIT_BYTES = 4 * 1024**3
_CPU_OBSERVATION_TERMS_CACHE_LIMIT_BYTES = 4 * 1024**3
_ADAPTIVE_RIGHT_EDGE_MASS_THRESHOLD = 0.01
_ADAPTIVE_RIGHT_TAIL_MASS_THRESHOLD = 0.05
_ADAPTIVE_MAX_REFINEMENT_ROUNDS = 2


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


def _iter_cell_slices(n_cells: int, chunk_size: int) -> list[slice]:
    return [
        slice(start, min(start + chunk_size, n_cells))
        for start in range(0, n_cells, chunk_size)
    ]


def _resolve_cell_chunk_size(n_cells: int, requested_chunk_size: int) -> int:
    if n_cells < 1:
        raise ValueError("n_cells must be >= 1")
    if requested_chunk_size < 1:
        raise ValueError("requested_chunk_size must be >= 1")
    return min(int(requested_chunk_size), int(n_cells))


def _resolve_support_tail_points(n_support_points: int) -> int:
    if n_support_points < 1:
        raise ValueError("n_support_points must be >= 1")
    return max(2, min(int(n_support_points), int(np.ceil(n_support_points * 0.02))))


def _resolve_probability_support_scale(scale: float | None) -> float:
    if scale is None:
        raise ValueError("scale is required for probability support construction")
    resolved = float(scale)
    if not np.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return resolved


def _resolve_effective_exposure_values(
    batch: ObservationBatch,
    *,
    scale: float | None,
    effective_exposure_values: np.ndarray | None,
) -> np.ndarray:
    if effective_exposure_values is not None:
        return np.asarray(effective_exposure_values, dtype=DTYPE_NP).reshape(-1)
    return np.asarray(
        effective_exposure(batch.reference_counts, _resolve_probability_support_scale(scale)),
        dtype=DTYPE_NP,
    ).reshape(-1)


def _observed_probability_values(
    batch: ObservationBatch,
    *,
    effective_exposure_values: np.ndarray,
) -> np.ndarray:
    return np.asarray(batch.counts, dtype=DTYPE_NP) / np.maximum(
        effective_exposure_values[:, None],
        EPS,
    )


def _observed_scaled_support_floor(
    batch: ObservationBatch,
) -> np.ndarray:
    return np.max(np.asarray(batch.counts, dtype=DTYPE_NP), axis=0)


def _default_scaled_support_max(
    batch: ObservationBatch,
    scale: float | None = None,
    *,
    method: Literal["observed_max", "quantile"],
    effective_exposure_values: np.ndarray | None = None,
) -> np.ndarray:
    probability_support_scale = _resolve_probability_support_scale(scale)
    resolved_exposure_values = _resolve_effective_exposure_values(
        batch,
        scale=probability_support_scale,
        effective_exposure_values=effective_exposure_values,
    )
    observed_probabilities = _observed_probability_values(
        batch,
        effective_exposure_values=resolved_exposure_values,
    )
    if method == "observed_max":
        scaled_support_from_neff = np.max(observed_probabilities, axis=0) * probability_support_scale
    else:
        scaled_support_from_neff = (
            np.nanpercentile(observed_probabilities, 95, axis=0)
            * probability_support_scale
        )
    # Keep a raw-count floor on the shared scaled axis so low-mean-N batches do
    # not collapse the initial support below observed count magnitudes.
    scaled_support_floor = _observed_scaled_support_floor(batch)
    return np.maximum(
        np.asarray(scaled_support_from_neff, dtype=DTYPE_NP),
        scaled_support_floor,
    )


def _default_probability_support_max(
    batch: ObservationBatch,
    scale: float | None = None,
    *,
    method: Literal["observed_max", "quantile"],
    effective_exposure_values: np.ndarray | None = None,
) -> np.ndarray:
    probability_support_scale = _resolve_probability_support_scale(scale)
    scaled_support_max = _default_scaled_support_max(
        batch,
        scale=probability_support_scale,
        method=method,
        effective_exposure_values=effective_exposure_values,
    )
    return np.clip(
        np.asarray(scaled_support_max, dtype=DTYPE_NP) / probability_support_scale,
        EPS,
        1.0,
    )


def _build_probability_support(
    n_support_points: int,
    support_max: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device,
    spacing: Literal["linear", "sqrt"],
) -> torch.Tensor:
    max_values = torch.as_tensor(support_max, dtype=dtype, device=device)
    if spacing == "sqrt":
        base = torch.sqrt(
            torch.linspace(0.0, 1.0, n_support_points, dtype=dtype, device=device)
        )
        base = base / base[-1]
    else:
        base = torch.linspace(0.0, 1.0, n_support_points, dtype=dtype, device=device)
    return (max_values[:, None] * base[None, :]).clamp(0.0, 1.0)


def _scale_support_max(
    support_max: np.ndarray,
    support_scale: float,
    *,
    upper_bound: float | None = None,
) -> np.ndarray:
    scaled = np.asarray(support_max, dtype=DTYPE_NP) * float(support_scale)
    return np.clip(scaled, EPS, upper_bound)


def _build_rate_support(
    counts: np.ndarray,
    n_support_points: int,
    *,
    support_scale: float,
    dtype: torch.dtype,
    device: torch.device,
    spacing: Literal["linear", "sqrt"],
) -> torch.Tensor:
    if spacing == "sqrt":
        base = torch.sqrt(
            torch.linspace(0.0, 1.0, n_support_points, dtype=dtype, device=device)
        )
        base = base / base[-1]
    else:
        base = torch.linspace(0.0, 1.0, n_support_points, dtype=dtype, device=device)
    max_values = _scale_support_max(
        np.nanmax(counts, axis=0),
        support_scale,
    )
    max_tensor = torch.as_tensor(max_values, dtype=dtype, device=device)
    return (max_tensor[:, None] * base[None, :]).clamp_min(EPS)


def _validate_initial_probabilities(
    initial_probabilities: np.ndarray | torch.Tensor,
    *,
    n_genes: int,
    n_support_points: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    values = torch.as_tensor(initial_probabilities, dtype=dtype, device=device)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.shape != (n_genes, n_support_points):
        raise ValueError(
            "initial_probabilities must have shape "
            f"{(n_genes, n_support_points)}, got {values.shape}"
        )
    values = values.clamp_min(EPS)
    return values / values.sum(dim=-1, keepdim=True)


def _initialize_probabilities(
    *,
    n_genes: int,
    n_support_points: int,
    initial_probabilities: np.ndarray | torch.Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if initial_probabilities is not None:
        return _validate_initial_probabilities(
            initial_probabilities,
            n_genes=n_genes,
            n_support_points=n_support_points,
            dtype=dtype,
            device=device,
        )
    return torch.full(
        (n_genes, n_support_points),
        1.0 / n_support_points,
        dtype=dtype,
        device=device,
    )


def _interpolate_probabilities_to_support(
    old_support: np.ndarray | torch.Tensor,
    old_probabilities: np.ndarray | torch.Tensor,
    new_support: np.ndarray | torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    old_support_t = torch.as_tensor(old_support, dtype=dtype, device=device)
    old_probabilities_t = torch.as_tensor(old_probabilities, dtype=dtype, device=device)
    new_support_t = torch.as_tensor(new_support, dtype=dtype, device=device)

    right_idx = torch.searchsorted(
        old_support_t.contiguous(),
        new_support_t.contiguous(),
        right=False,
    )
    support_size = old_support_t.shape[1]
    right_idx = right_idx.clamp(0, support_size - 1)
    left_idx = (right_idx - 1).clamp(0, support_size - 1)

    left_support = torch.gather(old_support_t, 1, left_idx)
    right_support = torch.gather(old_support_t, 1, right_idx)
    left_probabilities = torch.gather(old_probabilities_t, 1, left_idx)
    right_probabilities = torch.gather(old_probabilities_t, 1, right_idx)

    interpolation_span = (right_support - left_support).clamp_min(EPS)
    interpolation_weight = (new_support_t - left_support) / interpolation_span
    projected = left_probabilities + interpolation_weight * (
        right_probabilities - left_probabilities
    )
    in_range = (new_support_t >= old_support_t[:, :1]) & (
        new_support_t <= old_support_t[:, -1:]
    )
    projected = torch.where(
        in_range,
        projected,
        torch.zeros_like(projected),
    )
    projected = projected.clamp_min(EPS)
    return projected / projected.sum(dim=-1, keepdim=True)


def _compute_support_diagnostics(
    support: torch.Tensor,
    probabilities: np.ndarray | torch.Tensor,
    *,
    quantile_hi: float,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    support_t = support.to(device=device, dtype=dtype)
    probabilities_t = torch.as_tensor(probabilities, dtype=dtype, device=device)
    probabilities_t = probabilities_t.clamp_min(EPS)
    probabilities_t = probabilities_t / probabilities_t.sum(dim=-1, keepdim=True)
    cdf = probabilities_t.cumsum(dim=-1)
    cdf = cdf / cdf[:, -1:].clamp_min(EPS)
    quantile_mask = cdf >= float(quantile_hi)
    quantile_index = quantile_mask.to(torch.int64).argmax(dim=-1)
    quantile_index = torch.where(
        quantile_mask.any(dim=-1),
        quantile_index,
        torch.full_like(quantile_index, support_t.shape[1] - 1),
    )
    map_index = torch.argmax(probabilities_t, dim=-1)
    tail_points = _resolve_support_tail_points(int(support_t.shape[1]))
    return {
        "n_support_points": int(support_t.shape[1]),
        "tail_points": tail_points,
        "support_upper": support_t[:, -1],
        "quantile_index": quantile_index,
        "quantile_at_right_edge": quantile_index == (support_t.shape[1] - 1),
        "map_index": map_index,
        "map_at_right_edge": map_index == (support_t.shape[1] - 1),
        "right_edge_mass": probabilities_t[:, -1],
        "right_tail_mass": probabilities_t[:, -tail_points:].sum(dim=-1),
    }


def _support_diagnostics_expansion_mask(
    diagnostics: dict[str, torch.Tensor | int],
) -> torch.Tensor:
    n_support_points = int(cast(int, diagnostics["n_support_points"]))
    quantile_index = cast(torch.Tensor, diagnostics["quantile_index"])
    quantile_at_right_edge = cast(torch.Tensor, diagnostics["quantile_at_right_edge"])
    map_at_right_edge = cast(torch.Tensor, diagnostics["map_at_right_edge"])
    right_edge_mass = cast(torch.Tensor, diagnostics["right_edge_mass"])
    right_tail_mass = cast(torch.Tensor, diagnostics["right_tail_mass"])
    near_right_edge = quantile_index >= max(0, n_support_points - 2)
    return (
        quantile_at_right_edge
        | map_at_right_edge
        | (right_edge_mass >= _ADAPTIVE_RIGHT_EDGE_MASS_THRESHOLD)
        | (
            near_right_edge
            & (right_tail_mass >= _ADAPTIVE_RIGHT_TAIL_MASS_THRESHOLD)
        )
    )


def _summarize_support_diagnostics(
    diagnostics: dict[str, torch.Tensor | int],
    *,
    expansion_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    map_at_right_edge = cast(torch.Tensor, diagnostics["map_at_right_edge"])
    quantile_at_right_edge = cast(torch.Tensor, diagnostics["quantile_at_right_edge"])
    right_edge_mass = cast(torch.Tensor, diagnostics["right_edge_mass"])
    right_tail_mass = cast(torch.Tensor, diagnostics["right_tail_mass"])
    summary: dict[str, Any] = {
        "n_genes": int(map_at_right_edge.numel()),
        "tail_points": int(cast(int, diagnostics["tail_points"])),
        "n_map_at_right_edge": int(map_at_right_edge.to(torch.int64).sum().item()),
        "n_quantile_at_right_edge": int(
            quantile_at_right_edge.to(torch.int64).sum().item()
        ),
        "max_right_edge_mass": float(right_edge_mass.max().item()),
        "max_right_tail_mass": float(right_tail_mass.max().item()),
    }
    if expansion_mask is not None:
        summary["n_genes_needing_expansion"] = int(
            expansion_mask.to(torch.int64).sum().item()
        )
    return summary


def _adaptive_refine_support(
    support: torch.Tensor,
    probabilities: np.ndarray | torch.Tensor,
    *,
    config: PriorFitConfig,
    dtype: torch.dtype,
    device: torch.device,
    upper_bound: float | None,
) -> torch.Tensor:
    support_t = support.to(device=device, dtype=dtype)
    diagnostics = _compute_support_diagnostics(
        support_t,
        probabilities,
        quantile_hi=float(config.adaptive_support_quantile_hi),
        dtype=dtype,
        device=device,
    )
    quantile_index = cast(torch.Tensor, diagnostics["quantile_index"])
    quantile_hi_value = torch.gather(
        support_t,
        1,
        quantile_index.unsqueeze(-1),
    ).squeeze(-1)
    max_value = cast(torch.Tensor, diagnostics["support_upper"])
    minimum_upper = torch.full_like(max_value, max_value.new_tensor(EPS))
    if support_t.shape[1] > 1:
        minimum_upper = torch.maximum(
            minimum_upper,
            max_value / float(support_t.shape[1] - 1),
        )
    shrink_upper = torch.minimum(
        max_value,
        torch.maximum(
            quantile_hi_value * float(config.adaptive_support_scale),
            minimum_upper,
        ),
    )
    expanded_upper = torch.maximum(
        max_value * float(config.adaptive_support_scale),
        minimum_upper,
    )
    if upper_bound is not None:
        expanded_upper = expanded_upper.clamp_max(float(upper_bound))
    expansion_mask = _support_diagnostics_expansion_mask(diagnostics)
    upper = torch.where(
        expansion_mask,
        expanded_upper,
        shrink_upper,
    )
    base = torch.linspace(0.0, 1.0, support_t.shape[1], dtype=dtype, device=device)
    return upper[:, None] * base[None, :]


class _BaseEMStep(torch.nn.Module):
    def _finalize(
        self,
        log_joint: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _finalize_em_log_joint(log_joint)


class _BinomialEMStep(_BaseEMStep):
    def forward(
        self,
        counts: torch.Tensor,
        effective_exposure_values: torch.Tensor,
        log_coeff: torch.Tensor,
        invalid: torch.Tensor,
        log_support: torch.Tensor,
        log1m_support: torch.Tensor,
        log_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_likelihood = (
            log_coeff
            + counts * log_support.unsqueeze(-2)
            + (effective_exposure_values - counts) * log1m_support.unsqueeze(-2)
        )
        log_joint = log_likelihood.masked_fill(invalid, float("-inf")) + log_prior
        return self._finalize(log_joint)


class _NegativeBinomialEMStep(_BaseEMStep):
    def __init__(self, overdispersion: float) -> None:
        super().__init__()
        self.overdispersion = float(overdispersion)
        self.r = float(1.0 / overdispersion)

    def forward(
        self,
        counts: torch.Tensor,
        effective_exposure_values: torch.Tensor,
        base_term: torch.Tensor,
        r_tensor: torch.Tensor,
        support_values: torch.Tensor,
        log_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu = effective_exposure_values * support_values.unsqueeze(-2)
        log_r_plus_mu = torch.log(r_tensor + mu)
        log_joint = (
            base_term
            + counts * torch.log(mu + EPS)
            - (counts + self.r) * log_r_plus_mu
            + log_prior
        )
        return self._finalize(log_joint)


class _PoissonEMStep(_BaseEMStep):
    def forward(
        self,
        counts: torch.Tensor,
        rate_values: torch.Tensor,
        log_rate_values: torch.Tensor,
        log_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_joint = (
            counts * log_rate_values.unsqueeze(-2)
            - rate_values.unsqueeze(-2)
            + log_prior
        )
        return self._finalize(log_joint)


_COMPILED_EM_STEPS: dict[
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


def _get_em_step_module(
    likelihood: LikelihoodName,
    *,
    device: torch.device,
    dtype: torch.dtype,
    nb_overdispersion: float,
    compile_model: bool,
) -> torch.nn.Module:
    dtype_name = str(dtype).removeprefix("torch.")
    overdispersion_key = (
        nb_overdispersion if likelihood == "negative_binomial" else None
    )
    cache_key = (
        likelihood,
        _cache_device_key(device),
        dtype_name,
        overdispersion_key,
        bool(compile_model),
    )
    cached = _COMPILED_EM_STEPS.get(cache_key)
    if cached is not None:
        return cached
    if likelihood == "binomial":
        module: torch.nn.Module = _BinomialEMStep()
    elif likelihood == "negative_binomial":
        module = _NegativeBinomialEMStep(nb_overdispersion)
    else:
        module = _PoissonEMStep()
    module = module.to(device=device, dtype=dtype)
    compiled = _maybe_compile(module) if compile_model else module
    _COMPILED_EM_STEPS[cache_key] = compiled
    return compiled


ObservationTerms = (
    _BinomialObservationTerms
    | _NegativeBinomialObservationTerms
    | _PoissonObservationTerms
)
SupportTerms = _ProbabilitySupportTerms | _RateSupportTerms


def _prepare_support_terms(
    *,
    likelihood: LikelihoodName,
    support: torch.Tensor,
) -> SupportTerms:
    if likelihood == "poisson":
        return _rate_support_terms(support)
    return _probability_support_terms(support)


def _build_observation_terms(
    *,
    likelihood: LikelihoodName,
    counts_chunk: torch.Tensor,
    exposure_chunk: torch.Tensor | None,
    nb_overdispersion: float,
) -> ObservationTerms:
    if likelihood == "poisson":
        return _poisson_observation_terms(counts_chunk)
    if exposure_chunk is None:
        raise ValueError("exposure_chunk is required for non-poisson likelihoods")
    if likelihood == "negative_binomial":
        return _negative_binomial_observation_terms(
            counts_chunk,
            exposure_chunk,
            nb_overdispersion,
        )
    return _binomial_observation_terms(
        counts_chunk,
        exposure_chunk,
    )


def _compute_log_likelihood_chunk(
    *,
    likelihood: LikelihoodName,
    observation_terms: ObservationTerms,
    support_terms: SupportTerms,
) -> torch.Tensor:
    if likelihood == "poisson":
        return _log_poisson_likelihood_from_terms(
            cast(_PoissonObservationTerms, observation_terms),
            cast(_RateSupportTerms, support_terms),
        )
    if likelihood == "negative_binomial":
        return _log_negative_binomial_likelihood_from_terms(
            cast(_NegativeBinomialObservationTerms, observation_terms),
            cast(_ProbabilitySupportTerms, support_terms),
        )
    return _log_binomial_likelihood_from_terms(
        cast(_BinomialObservationTerms, observation_terms),
        cast(_ProbabilitySupportTerms, support_terms),
    )


def _log_prior_from_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.log(probabilities.clamp_min(EPS)).unsqueeze(-2)


def _finalize_em_log_joint(
    log_joint: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_normalizer = torch.logsumexp(log_joint, dim=-1, keepdim=True)
    if torch.any(~torch.isfinite(log_normalizer)):
        raise ValueError(
            "encountered observations with zero likelihood under the current "
            "support/prior configuration"
        )
    posterior_sum = torch.exp(log_joint - log_normalizer).sum(dim=-2)
    return posterior_sum, log_normalizer.sum()


def _finalize_em_step(
    log_likelihood: torch.Tensor,
    log_prior: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _finalize_em_log_joint(log_likelihood + log_prior)


def _estimate_log_likelihood_cache_bytes(
    *,
    n_genes: int,
    n_cells: int,
    n_support_points: int,
    dtype: torch.dtype,
) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    return int(n_genes * n_cells * n_support_points * element_size)


def _estimate_observation_terms_cache_bytes(
    *,
    likelihood: LikelihoodName,
    n_genes: int,
    n_cells: int,
    dtype: torch.dtype,
) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    if likelihood == "poisson":
        return int(n_genes * n_cells * element_size)
    if likelihood == "negative_binomial":
        return int(n_genes * n_cells * 3 * element_size)
    return int(n_genes * n_cells * ((3 * element_size) + 1))


def _should_cache_log_likelihood(
    *,
    device: torch.device,
    estimated_bytes: int,
) -> bool:
    if estimated_bytes <= 0:
        return False
    if device.type == "cuda":
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        return estimated_bytes <= int(total_memory * 0.55)
    return estimated_bytes <= _CPU_LOG_LIKELIHOOD_CACHE_LIMIT_BYTES


def _should_cache_observation_terms(
    *,
    device: torch.device,
    estimated_bytes: int,
) -> bool:
    if estimated_bytes <= 0:
        return False
    if device.type == "cuda":
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        return estimated_bytes <= int(total_memory * 0.25)
    return estimated_bytes <= _CPU_OBSERVATION_TERMS_CACHE_LIMIT_BYTES


def _run_em_pass(
    batch: ObservationBatch,
    *,
    config: PriorFitConfig,
    support: torch.Tensor,
    effective_exposure_values: np.ndarray | None,
    initial_probabilities: np.ndarray | torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
    compile_model: bool,
    progress_callback: FitProgressCallback | None,
    phase_index: int,
) -> tuple[torch.Tensor, float, list[float]]:
    counts_t = torch.as_tensor(batch.counts, dtype=dtype, device=device).transpose(0, 1)
    probabilities_t = _initialize_probabilities(
        n_genes=batch.n_genes,
        n_support_points=support.shape[1],
        initial_probabilities=initial_probabilities,
        dtype=dtype,
        device=device,
    )
    if config.likelihood == "poisson":
        exposure_t = None
    else:
        exposure_t = (
            torch.as_tensor(
                effective_exposure_values,
                dtype=dtype,
                device=device,
            )
            .unsqueeze(0)
            .expand(batch.n_genes, -1)
        )
    effective_cell_chunk_size = _resolve_cell_chunk_size(
        batch.n_cells,
        config.cell_chunk_size,
    )
    cell_slices = _iter_cell_slices(batch.n_cells, effective_cell_chunk_size)
    likelihood = cast(LikelihoodName, config.likelihood)
    support_terms = _prepare_support_terms(
        likelihood=likelihood,
        support=support,
    )
    cache_log_likelihood = _should_cache_log_likelihood(
        device=device,
        estimated_bytes=_estimate_log_likelihood_cache_bytes(
            n_genes=batch.n_genes,
            n_cells=batch.n_cells,
            n_support_points=int(support.shape[1]),
            dtype=dtype,
        ),
    )
    cache_observation_terms = (not cache_log_likelihood) and _should_cache_observation_terms(
        device=device,
        estimated_bytes=_estimate_observation_terms_cache_bytes(
            likelihood=likelihood,
            n_genes=batch.n_genes,
            n_cells=batch.n_cells,
            dtype=dtype,
        ),
    )
    log_likelihood_cache: list[torch.Tensor | None] = []
    observation_terms_cache: list[ObservationTerms | None] = []
    if cache_log_likelihood:
        log_likelihood_cache = [None] * len(cell_slices)
    if cache_observation_terms:
        observation_terms_cache = [None] * len(cell_slices)
    step_module = None
    if not cache_log_likelihood:
        step_module = _get_em_step_module(
            likelihood,
            device=device,
            dtype=dtype,
            nb_overdispersion=config.nb_overdispersion,
            compile_model=compile_model,
        )
    max_steps = (
        config.max_em_iterations if config.max_em_iterations is not None else 10000
    )
    objective_history: list[float] = []
    objective_denominator = float(batch.n_cells * batch.n_genes)
    posterior_sum = torch.zeros_like(probabilities_t)
    log_marginal_sum = torch.zeros((), dtype=dtype, device=device)

    with torch.inference_mode():
        for step in range(1, max_steps + 1):
            posterior_sum.zero_()
            log_marginal_sum.zero_()
            log_prior = _log_prior_from_probabilities(probabilities_t)
            for chunk_index, cell_slice in enumerate(cell_slices):
                if cache_log_likelihood:
                    log_likelihood_chunk = log_likelihood_cache[chunk_index]
                    if log_likelihood_chunk is None:
                        counts_chunk = counts_t[:, cell_slice]
                        exposure_chunk = (
                            None if exposure_t is None else exposure_t[:, cell_slice]
                        )
                        observation_terms = _build_observation_terms(
                            likelihood=likelihood,
                            counts_chunk=counts_chunk,
                            exposure_chunk=exposure_chunk,
                            nb_overdispersion=config.nb_overdispersion,
                        )
                        log_likelihood_chunk = _compute_log_likelihood_chunk(
                            likelihood=likelihood,
                            observation_terms=observation_terms,
                            support_terms=support_terms,
                        )
                        log_likelihood_cache[chunk_index] = log_likelihood_chunk
                    posterior_sum_chunk, log_marginal_sum_chunk = _finalize_em_step(
                        log_likelihood_chunk,
                        log_prior,
                    )
                else:
                    observation_terms = (
                        observation_terms_cache[chunk_index]
                        if cache_observation_terms
                        else None
                    )
                    if observation_terms is None:
                        counts_chunk = counts_t[:, cell_slice]
                        exposure_chunk = (
                            None if exposure_t is None else exposure_t[:, cell_slice]
                        )
                        observation_terms = _build_observation_terms(
                            likelihood=likelihood,
                            counts_chunk=counts_chunk,
                            exposure_chunk=exposure_chunk,
                            nb_overdispersion=config.nb_overdispersion,
                        )
                        if cache_observation_terms:
                            observation_terms_cache[chunk_index] = observation_terms
                    if likelihood == "poisson":
                        poisson_terms = cast(_PoissonObservationTerms, observation_terms)
                        rate_terms = cast(_RateSupportTerms, support_terms)
                        posterior_sum_chunk, log_marginal_sum_chunk = cast(
                            _PoissonEMStep,
                            step_module,
                        )(
                            poisson_terms.counts,
                            rate_terms.values,
                            rate_terms.log_values,
                            log_prior,
                        )
                    elif likelihood == "negative_binomial":
                        negative_binomial_terms = cast(
                            _NegativeBinomialObservationTerms,
                            observation_terms,
                        )
                        probability_terms = cast(_ProbabilitySupportTerms, support_terms)
                        posterior_sum_chunk, log_marginal_sum_chunk = cast(
                            _NegativeBinomialEMStep,
                            step_module,
                        )(
                            negative_binomial_terms.counts,
                            negative_binomial_terms.exposure,
                            negative_binomial_terms.base_term,
                            negative_binomial_terms.r_tensor,
                            probability_terms.values,
                            log_prior,
                        )
                    else:
                        binomial_terms = cast(_BinomialObservationTerms, observation_terms)
                        probability_terms = cast(_ProbabilitySupportTerms, support_terms)
                        posterior_sum_chunk, log_marginal_sum_chunk = cast(
                            _BinomialEMStep,
                            step_module,
                        )(
                            binomial_terms.counts,
                            binomial_terms.exposure,
                            binomial_terms.log_coeff,
                            binomial_terms.invalid,
                            probability_terms.log_values,
                            probability_terms.log1m_values,
                            log_prior,
                        )
                posterior_sum.add_(posterior_sum_chunk)
                log_marginal_sum.add_(log_marginal_sum_chunk)
            updated = posterior_sum.clamp_min(EPS)
            updated = updated / updated.sum(dim=-1, keepdim=True)
            delta = float(torch.max(torch.abs(updated - probabilities_t)).item())
            probabilities_t = updated
            objective = float(((-log_marginal_sum) / objective_denominator).item())
            objective_history.append(objective)
            if progress_callback is not None:
                progress_callback(
                    step,
                    max_steps,
                    objective,
                    objective,
                    float(phase_index),
                    delta < config.convergence_tolerance,
                )
            if delta < config.convergence_tolerance:
                break

    return (
        probabilities_t.detach(),
        objective_history[-1],
        objective_history,
    )


def _build_support(
    batch: ObservationBatch,
    *,
    scale: float | None = None,
    config: PriorFitConfig,
    dtype: torch.dtype,
    device: torch.device,
    effective_exposure_values: np.ndarray | None = None,
    support_max: np.ndarray | None,
) -> torch.Tensor:
    if config.likelihood == "poisson":
        return _build_rate_support(
            batch.counts,
            config.n_support_points,
            support_scale=config.support_scale,
            dtype=dtype,
            device=device,
            spacing=config.support_spacing,
        )
    resolved_max = (
        _default_probability_support_max(
            batch,
            scale=scale,
            method=config.support_max_from,
            effective_exposure_values=effective_exposure_values,
        )
        if support_max is None
        else np.asarray(support_max, dtype=DTYPE_NP).reshape(-1)
    )
    if resolved_max.shape != (batch.n_genes,):
        raise ValueError(
            f"support_max must have shape {(batch.n_genes,)}, got {resolved_max.shape}"
        )
    if (
        np.any(~np.isfinite(resolved_max))
        or np.any(resolved_max <= 0)
        or np.any(resolved_max > 1)
    ):
        raise ValueError("support_max must lie in (0, 1]")
    resolved_max = _scale_support_max(
        resolved_max,
        config.support_scale,
        upper_bound=1.0,
    )
    return _build_probability_support(
        config.n_support_points,
        resolved_max,
        dtype=dtype,
        device=device,
        spacing=config.support_spacing,
    )


def fit_gene_priors(
    batch: ObservationBatch | GeneBatch,
    *,
    scale: float,
    config: PriorFitConfig = PriorFitConfig(),
    device: str | torch.device = "cpu",
    torch_dtype: str = "float64",
    result_dtype: str | None = None,
    support_max: np.ndarray | None = None,
    initial_probabilities: np.ndarray | None = None,
    compile_model: bool = True,
    progress_callback: FitProgressCallback | None = None,
) -> PriorFitResult:
    if isinstance(batch, GeneBatch):
        batch = batch.to_observation_batch()
    batch.check_shape()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    effective_exposure_values = (
        None
        if config.likelihood == "poisson"
        else np.asarray(
            effective_exposure(batch.reference_counts, scale),
            dtype=DTYPE_NP,
        ).reshape(-1)
    )
    if config.likelihood == "binomial":
        validate_binomial_observations(
            batch.counts,
            effective_exposure_values,
        )
    mean_reference_count_value = mean_reference_count(batch.reference_counts)

    device_obj = torch.device(device)
    dtype_obj = _resolve_torch_dtype(torch_dtype)
    result_dtype_name = torch_dtype if result_dtype is None else result_dtype
    result_dtype_obj = _resolve_numpy_dtype(result_dtype_name)
    support = _build_support(
        batch,
        scale=scale,
        config=config,
        dtype=dtype_obj,
        device=device_obj,
        effective_exposure_values=effective_exposure_values,
        support_max=support_max,
    )
    posterior_mean_probabilities_t, final_objective, objective_history = _run_em_pass(
        batch,
        config=config,
        support=support,
        effective_exposure_values=effective_exposure_values,
        initial_probabilities=initial_probabilities,
        device=device_obj,
        dtype=dtype_obj,
        compile_model=compile_model,
        progress_callback=progress_callback,
        phase_index=1,
    )
    phase1_diagnostics = _compute_support_diagnostics(
        support,
        posterior_mean_probabilities_t,
        quantile_hi=float(config.adaptive_support_quantile_hi),
        dtype=dtype_obj,
        device=device_obj,
    )
    phase1_expansion_mask = _support_diagnostics_expansion_mask(phase1_diagnostics)
    adaptive_rounds_run = 0
    max_support_upper_delta = 0.0
    adaptive_expanded_mask = torch.zeros(
        batch.n_genes,
        dtype=torch.bool,
        device=device_obj,
    )

    if config.use_adaptive_support:
        support_upper_bound = None if config.likelihood == "poisson" else 1.0
        for adaptive_round in range(1, _ADAPTIVE_MAX_REFINEMENT_ROUNDS + 1):
            refined_support = _adaptive_refine_support(
                support,
                posterior_mean_probabilities_t,
                config=config,
                dtype=dtype_obj,
                device=device_obj,
                upper_bound=support_upper_bound,
            )
            support_upper_delta = refined_support[:, -1] - support[:, -1]
            adaptive_expanded_mask |= support_upper_delta > 1e-12
            max_support_upper_delta = max(
                max_support_upper_delta,
                float(torch.clamp_min(support_upper_delta, 0.0).max().item()),
            )
            refined_init = _interpolate_probabilities_to_support(
                support,
                posterior_mean_probabilities_t,
                refined_support,
                dtype=dtype_obj,
                device=device_obj,
            )
            support = refined_support
            adaptive_rounds_run = adaptive_round
            posterior_mean_probabilities_t, final_objective, phase_history = _run_em_pass(
                batch,
                config=config,
                support=support,
                effective_exposure_values=effective_exposure_values,
                initial_probabilities=refined_init,
                device=device_obj,
                dtype=dtype_obj,
                compile_model=compile_model,
                progress_callback=progress_callback,
                phase_index=adaptive_round + 1,
            )
            objective_history.extend(phase_history)
            round_diagnostics = _compute_support_diagnostics(
                support,
                posterior_mean_probabilities_t,
                quantile_hi=float(config.adaptive_support_quantile_hi),
                dtype=dtype_obj,
                device=device_obj,
            )
            if adaptive_round >= _ADAPTIVE_MAX_REFINEMENT_ROUNDS:
                break
            if not bool(_support_diagnostics_expansion_mask(round_diagnostics).any().item()):
                break
            if support_upper_bound is not None and bool(
                torch.all(support[:, -1] >= (support_upper_bound - 1e-12)).item()
            ):
                break
    final_diagnostics = _compute_support_diagnostics(
        support,
        posterior_mean_probabilities_t,
        quantile_hi=float(config.adaptive_support_quantile_hi),
        dtype=dtype_obj,
        device=device_obj,
    )
    posterior_mean_probabilities = (
        posterior_mean_probabilities_t.detach().cpu().numpy().astype(
            result_dtype_obj,
            copy=False,
        )
    )

    prior = PriorGrid(
        gene_names=list(batch.gene_names),
        distribution=make_distribution_grid(
            config.likelihood,
            support=support.detach().cpu().numpy().astype(result_dtype_obj, copy=False),
            probabilities=posterior_mean_probabilities,
        ),
        scale=float(scale),
    )
    result_config = asdict(config)
    result_config["torch_dtype"] = torch_dtype
    result_config["result_dtype"] = result_dtype_name
    result_config["compile_model"] = bool(compile_model)
    result_config["mean_reference_count"] = mean_reference_count_value
    result_config["support_diagnostics"] = {
        "phase1": _summarize_support_diagnostics(
            phase1_diagnostics,
            expansion_mask=phase1_expansion_mask,
        ),
        "final": _summarize_support_diagnostics(
            final_diagnostics,
            expansion_mask=_support_diagnostics_expansion_mask(final_diagnostics),
        ),
        "adaptive": {
            "enabled": bool(config.use_adaptive_support),
            "rounds_run": int(adaptive_rounds_run),
            "n_expanded_genes": int(adaptive_expanded_mask.to(torch.int64).sum().item()),
            "max_support_upper_delta": float(max_support_upper_delta),
        },
    }
    return PriorFitResult(
        gene_names=list(batch.gene_names),
        prior=prior,
        posterior_mean_probabilities=posterior_mean_probabilities,
        objective_history=objective_history,
        final_objective=final_objective,
        config=result_config,
    )


__all__ = ["fit_gene_priors"]
