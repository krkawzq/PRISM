from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Literal, cast

import numpy as np
import torch

from .constants import DTYPE_NP, EPS
from .exposure import (
    effective_exposure,
    mean_reference_count,
    validate_binomial_observations,
)
from .numeric import (
    log_binomial_likelihood_support,
    log_negative_binomial_likelihood_support,
    log_poisson_likelihood_support,
    posterior_from_log_likelihood,
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


def _resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported torch_dtype: {name}")


def _iter_cell_slices(n_cells: int, chunk_size: int) -> list[slice]:
    return [
        slice(start, min(start + chunk_size, n_cells))
        for start in range(0, n_cells, chunk_size)
    ]


def _default_probability_support_max(
    batch: ObservationBatch,
    scale: float,
    *,
    method: Literal["observed_max", "quantile"],
) -> np.ndarray:
    if method == "observed_max":
        mean_values = np.max(batch.counts, axis=0)
    else:
        mean_values = np.nanpercentile(batch.counts, 95, axis=0)
    mean_reference = max(mean_reference_count(batch.reference_counts), EPS)
    max_support = np.clip(
        mean_values * float(scale) / mean_reference / float(scale), EPS, 1.0
    )
    return np.asarray(max_support, dtype=DTYPE_NP)


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
    return (max_values[:, None] * base[None, :]).clamp(EPS, 1.0)


def _build_rate_support(
    counts: np.ndarray,
    n_support_points: int,
    *,
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
    max_values = np.clip(np.nanmax(counts, axis=0), EPS, None)
    max_tensor = torch.as_tensor(max_values, dtype=dtype, device=device)
    return (max_tensor[:, None] * base[None, :]).clamp_min(EPS)


def _validate_initial_probabilities(
    initial_probabilities: np.ndarray,
    *,
    n_genes: int,
    n_support_points: int,
) -> np.ndarray:
    values = np.asarray(initial_probabilities, dtype=DTYPE_NP)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape != (n_genes, n_support_points):
        raise ValueError(
            "initial_probabilities must have shape "
            f"{(n_genes, n_support_points)}, got {values.shape}"
        )
    values = np.clip(values, EPS, None)
    return values / values.sum(axis=-1, keepdims=True)


def _initialize_probabilities(
    *,
    n_genes: int,
    n_support_points: int,
    initial_probabilities: np.ndarray | None,
) -> np.ndarray:
    if initial_probabilities is not None:
        return _validate_initial_probabilities(
            initial_probabilities,
            n_genes=n_genes,
            n_support_points=n_support_points,
        )
    return np.full((n_genes, n_support_points), 1.0 / n_support_points, dtype=DTYPE_NP)


def _interpolate_probabilities_to_support(
    old_support: np.ndarray,
    old_probabilities: np.ndarray,
    new_support: np.ndarray,
) -> np.ndarray:
    projected = np.zeros_like(new_support, dtype=DTYPE_NP)
    for idx in range(old_probabilities.shape[0]):
        projected[idx] = np.interp(
            new_support[idx],
            old_support[idx],
            old_probabilities[idx],
            left=0.0,
            right=0.0,
        )
    projected = np.clip(projected, EPS, None)
    return projected / projected.sum(axis=-1, keepdims=True)


def _adaptive_refine_support(
    support: torch.Tensor,
    probabilities: np.ndarray,
    *,
    config: PriorFitConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    support_np = support.detach().cpu().numpy()
    refined = np.zeros_like(support_np, dtype=DTYPE_NP)
    for gene_idx in range(support_np.shape[0]):
        gene_support = support_np[gene_idx]
        gene_probabilities = probabilities[gene_idx]
        cdf = np.cumsum(gene_probabilities)
        cdf = cdf / max(cdf[-1], EPS)
        mode_value = float(gene_support[int(np.argmax(gene_probabilities))])
        min_value = float(gene_support[0])
        max_value = float(gene_support[-1])
        quantile_hi_value = float(
            gene_support[
                min(
                    int(np.searchsorted(cdf, config.adaptive_support_quantile_hi)),
                    len(gene_support) - 1,
                )
            ]
        )
        full_width = max(max_value - min_value, EPS)
        target_width = max(
            full_width * config.adaptive_support_fraction,
            full_width / max(len(gene_support) - 1, 1),
        )
        lower = max(min_value, mode_value - 0.5 * target_width)
        upper = min(quantile_hi_value, mode_value + 0.5 * target_width)
        if upper <= lower:
            upper = min(max_value, max(mode_value, quantile_hi_value))
            lower = max(min_value, upper - target_width)
        if upper <= lower:
            upper = min(max_value, lower + full_width / max(len(gene_support) - 1, 1))
        refined[gene_idx] = np.linspace(
            lower, upper, gene_support.shape[0], dtype=DTYPE_NP
        )
    return torch.as_tensor(refined, dtype=dtype, device=device)


class _BaseEMStep(torch.nn.Module):
    def _finalize(
        self,
        log_likelihood: torch.Tensor,
        prior_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        posterior = cast(
            torch.Tensor,
            posterior_from_log_likelihood(log_likelihood, prior_probabilities),
        )
        log_prior = torch.log(prior_probabilities.clamp_min(EPS)).unsqueeze(-2)
        log_marginal_sum = torch.logsumexp(log_likelihood + log_prior, dim=-1).sum()
        return posterior.sum(dim=-2), log_marginal_sum


class _BinomialEMStep(_BaseEMStep):
    def forward(
        self,
        counts: torch.Tensor,
        support: torch.Tensor,
        prior_probabilities: torch.Tensor,
        effective_exposure_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_likelihood = log_binomial_likelihood_support(
            counts,
            effective_exposure_values,
            support,
        )
        return self._finalize(log_likelihood, prior_probabilities)


class _NegativeBinomialEMStep(_BaseEMStep):
    def __init__(self, overdispersion: float) -> None:
        super().__init__()
        self.overdispersion = float(overdispersion)

    def forward(
        self,
        counts: torch.Tensor,
        support: torch.Tensor,
        prior_probabilities: torch.Tensor,
        effective_exposure_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_likelihood = log_negative_binomial_likelihood_support(
            counts,
            effective_exposure_values,
            support,
            overdispersion=self.overdispersion,
        )
        return self._finalize(log_likelihood, prior_probabilities)


class _PoissonEMStep(_BaseEMStep):
    def forward(
        self,
        counts: torch.Tensor,
        support: torch.Tensor,
        prior_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_likelihood = log_poisson_likelihood_support(counts, support)
        return self._finalize(log_likelihood, prior_probabilities)


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


def _run_em_pass(
    batch: ObservationBatch,
    *,
    scale: float,
    config: PriorFitConfig,
    support: torch.Tensor,
    initial_probabilities: np.ndarray | None,
    device: torch.device,
    dtype: torch.dtype,
    compile_model: bool,
    progress_callback: FitProgressCallback | None,
    phase_index: int,
) -> tuple[np.ndarray, float, list[float]]:
    counts_t = torch.as_tensor(batch.counts.T, dtype=dtype, device=device)
    probabilities_t = torch.as_tensor(
        _initialize_probabilities(
            n_genes=batch.n_genes,
            n_support_points=support.shape[1],
            initial_probabilities=initial_probabilities,
        ),
        dtype=dtype,
        device=device,
    )
    if config.likelihood == "poisson":
        exposure_t = None
    else:
        exposure_t = (
            torch.as_tensor(
                effective_exposure(batch.reference_counts, scale),
                dtype=dtype,
                device=device,
            )
            .unsqueeze(0)
            .expand(batch.n_genes, -1)
        )
    cell_slices = _iter_cell_slices(batch.n_cells, config.cell_chunk_size)
    step_module = _get_em_step_module(
        cast(LikelihoodName, config.likelihood),
        device=device,
        dtype=dtype,
        nb_overdispersion=config.nb_overdispersion,
        compile_model=compile_model,
    )
    max_steps = (
        config.max_em_iterations if config.max_em_iterations is not None else 10000
    )
    objective_history: list[float] = []

    with torch.no_grad():
        for step in range(1, max_steps + 1):
            posterior_sum = torch.zeros_like(probabilities_t)
            log_marginal_sum = torch.zeros((), dtype=dtype, device=device)
            for cell_slice in cell_slices:
                counts_chunk = counts_t[:, cell_slice]
                if exposure_t is None:
                    posterior_sum_chunk, log_marginal_sum_chunk = cast(
                        _PoissonEMStep, step_module
                    )(
                        counts_chunk,
                        support,
                        probabilities_t,
                    )
                else:
                    posterior_sum_chunk, log_marginal_sum_chunk = cast(
                        _BinomialEMStep | _NegativeBinomialEMStep,
                        step_module,
                    )(
                        counts_chunk,
                        support,
                        probabilities_t,
                        exposure_t[:, cell_slice],
                    )
                posterior_sum = posterior_sum + posterior_sum_chunk
                log_marginal_sum = log_marginal_sum + log_marginal_sum_chunk
            updated = posterior_sum.clamp_min(EPS)
            updated = updated / updated.sum(dim=-1, keepdim=True)
            delta = float(torch.max(torch.abs(updated - probabilities_t)).item())
            probabilities_t = updated
            objective = float(
                ((-log_marginal_sum) / (batch.n_cells * batch.n_genes)).item()
            )
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
        probabilities_t.detach().cpu().numpy().astype(DTYPE_NP, copy=False),
        objective_history[-1],
        objective_history,
    )


def _build_support(
    batch: ObservationBatch,
    *,
    scale: float,
    config: PriorFitConfig,
    dtype: torch.dtype,
    device: torch.device,
    support_max: np.ndarray | None,
) -> torch.Tensor:
    if config.likelihood == "poisson":
        return _build_rate_support(
            batch.counts,
            config.n_support_points,
            dtype=dtype,
            device=device,
            spacing=config.support_spacing,
        )
    resolved_max = (
        _default_probability_support_max(batch, scale, method=config.support_max_from)
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
    if config.likelihood == "binomial":
        validate_binomial_observations(
            batch.counts,
            effective_exposure(batch.reference_counts, scale),
        )

    device_obj = torch.device(device)
    dtype_obj = _resolve_torch_dtype(torch_dtype)
    support = _build_support(
        batch,
        scale=scale,
        config=config,
        dtype=dtype_obj,
        device=device_obj,
        support_max=support_max,
    )
    posterior_mean_probabilities, final_objective, objective_history = _run_em_pass(
        batch,
        scale=scale,
        config=config,
        support=support,
        initial_probabilities=initial_probabilities,
        device=device_obj,
        dtype=dtype_obj,
        compile_model=compile_model,
        progress_callback=progress_callback,
        phase_index=1,
    )

    if config.use_adaptive_support:
        refined_support = _adaptive_refine_support(
            support,
            posterior_mean_probabilities,
            config=config,
            dtype=dtype_obj,
            device=device_obj,
        )
        refined_init = _interpolate_probabilities_to_support(
            support.detach().cpu().numpy(),
            posterior_mean_probabilities,
            refined_support.detach().cpu().numpy(),
        )
        support = refined_support
        posterior_mean_probabilities, final_objective, phase2_history = _run_em_pass(
            batch,
            scale=scale,
            config=config,
            support=support,
            initial_probabilities=refined_init,
            device=device_obj,
            dtype=dtype_obj,
            compile_model=compile_model,
            progress_callback=progress_callback,
            phase_index=2,
        )
        objective_history.extend(phase2_history)

    prior = PriorGrid(
        gene_names=list(batch.gene_names),
        distribution=make_distribution_grid(
            config.likelihood,
            support=support.detach().cpu().numpy().astype(DTYPE_NP, copy=False),
            probabilities=posterior_mean_probabilities,
        ),
        scale=float(scale),
    )
    result_config = asdict(config)
    result_config["torch_dtype"] = torch_dtype
    result_config["compile_model"] = bool(compile_model)
    result_config["mean_reference_count"] = mean_reference_count(batch.reference_counts)
    return PriorFitResult(
        gene_names=list(batch.gene_names),
        prior=prior,
        posterior_mean_probabilities=posterior_mean_probabilities,
        objective_history=objective_history,
        final_objective=final_objective,
        config=result_config,
    )


__all__ = ["fit_gene_priors"]
