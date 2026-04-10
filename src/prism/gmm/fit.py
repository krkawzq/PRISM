from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from prism.model.constants import DTYPE_NP
from prism.model.types import DistributionGrid, PriorGrid

from .optimize import (
    MixtureOptimizationResult,
    evaluate_mixture_parameters,
    optimize_mixture_parameters,
)
from .schema import (
    DistributionGMMReport,
    DistributionGMMSearch,
    GMMSearchConfig,
    GMMTrainingConfig,
    PriorGMMReport,
    SupportAxis,
)
from .search import (
    _prepare_distribution_grid,
    _prepare_prior_grid,
    search_distribution_gmm,
    search_prior_gmm,
)


def _validate_search_matches(
    search: DistributionGMMSearch,
    *,
    support: np.ndarray,
    probabilities: np.ndarray,
    support_domain: str,
) -> None:
    atol = float(search.config.get("support_match_atol", 1e-9))
    rtol = float(search.config.get("support_match_rtol", 1e-9))
    if search.support_domain != support_domain:
        raise ValueError(
            "search support_domain does not match the requested fit input; "
            f"{search.support_domain!r} != {support_domain!r}"
        )
    if search.support.shape != support.shape:
        raise ValueError("search support shape does not match the requested fit input")
    if search.probabilities.shape != probabilities.shape:
        raise ValueError(
            "search probability shape does not match the requested fit input"
        )
    if not np.allclose(search.support, support, atol=atol, rtol=rtol):
        raise ValueError("search support does not match the requested fit input")
    if not np.allclose(search.probabilities, probabilities, atol=atol, rtol=rtol):
        raise ValueError("search probabilities do not match the requested fit input")


def _resolve_search_config(
    search: DistributionGMMSearch | None,
    *,
    fallback: GMMSearchConfig,
) -> GMMSearchConfig:
    if search is None:
        return fallback
    resolved: dict[str, object] = {}
    for name in GMMSearchConfig.__dataclass_fields__:
        if name in search.config:
            resolved[name] = search.config[name]
        else:
            resolved[name] = getattr(fallback, name)
    return GMMSearchConfig(**resolved)


def _sort_components(
    selected_k: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    lefts: np.ndarray,
    rights: np.ndarray,
    component_masses: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    sorted_weights = np.zeros_like(weights)
    sorted_means = np.zeros_like(means)
    sorted_stds = np.zeros_like(stds)
    sorted_lefts = np.zeros_like(lefts)
    sorted_rights = np.zeros_like(rights)
    sorted_component_masses = (
        None if component_masses is None else np.zeros_like(component_masses)
    )
    for row_idx, k in enumerate(selected_k.tolist()):
        order = np.argsort(means[row_idx, :k], kind="stable")
        sorted_weights[row_idx, :k] = weights[row_idx, :k][order]
        sorted_means[row_idx, :k] = means[row_idx, :k][order]
        sorted_stds[row_idx, :k] = stds[row_idx, :k][order]
        sorted_lefts[row_idx, :k] = lefts[row_idx, :k][order]
        sorted_rights[row_idx, :k] = rights[row_idx, :k][order]
        if sorted_component_masses is not None:
            sorted_component_masses[row_idx, :, :k] = component_masses[row_idx, :, :k][
                :,
                order,
            ]
    return (
        sorted_weights,
        sorted_means,
        sorted_stds,
        sorted_lefts,
        sorted_rights,
        sorted_component_masses,
    )


def _run_refit(
    search: DistributionGMMSearch,
    *,
    row_slice: slice | np.ndarray | None = None,
    selected_k: np.ndarray,
    initial_weights: np.ndarray,
    initial_means: np.ndarray,
    initial_stds: np.ndarray,
    initial_lefts: np.ndarray,
    initial_rights: np.ndarray,
    training_config: GMMTrainingConfig,
    device: str | torch.device,
) -> MixtureOptimizationResult:
    selected_rows = slice(None) if row_slice is None else row_slice
    return optimize_mixture_parameters(
        probabilities=search.probabilities[selected_rows],
        bin_edges=search.bin_edges[selected_rows],
        support_mask=search.support_mask[selected_rows],
        lower_bounds=search.lower_bounds[selected_rows],
        upper_bounds=search.upper_bounds[selected_rows],
        selected_k=selected_k,
        initial_weights=initial_weights,
        initial_means=initial_means,
        initial_stds=initial_stds,
        initial_left_truncations=initial_lefts,
        initial_right_truncations=initial_rights,
        max_iterations=training_config.max_iterations,
        learning_rate=training_config.learning_rate,
        convergence_tolerance=training_config.convergence_tolerance,
        overshoot_penalty=training_config.overshoot_penalty,
        mean_margin_fraction=training_config.mean_margin_fraction,
        torch_dtype=training_config.torch_dtype,
        device=device,
        compile_model=training_config.compile_model,
        optimize_weights=training_config.optimize_weights,
        optimize_means=training_config.optimize_means,
        optimize_stds=training_config.optimize_stds,
        optimize_left_truncations=training_config.optimize_left_truncations,
        optimize_right_truncations=training_config.optimize_right_truncations,
        sigma_floor_fraction=training_config.sigma_floor_fraction,
        min_window_fraction=training_config.min_window_fraction,
        truncation_mode=training_config.truncation_mode,
        truncation_regularization_strength=training_config.truncation_regularization_strength,
        initial_weight_logit_floor=training_config.initial_weight_logit_floor,
        inactive_weight_floor=training_config.inactive_weight_floor,
        masked_logit_value=training_config.masked_logit_value,
        logit_clip=training_config.logit_clip,
        inverse_softplus_clip=training_config.inverse_softplus_clip,
    )


def _metric_values(
    result: MixtureOptimizationResult,
    *,
    metric: str,
) -> np.ndarray:
    if metric == "jsd":
        return np.asarray(result.jsd, dtype=DTYPE_NP)
    if metric == "l1":
        return np.asarray(result.l1_error, dtype=DTYPE_NP)
    if metric == "cross_entropy":
        return np.asarray(result.cross_entropy, dtype=DTYPE_NP)
    raise ValueError(f"unsupported pruning_error_metric: {metric!r}")


def _component_significance(
    *,
    weights: np.ndarray,
    component_masses: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric == "weight":
        return np.asarray(weights, dtype=DTYPE_NP)
    if metric == "peak_mass":
        peak_mass = np.max(component_masses, axis=0)
        return np.asarray(weights * peak_mass, dtype=DTYPE_NP)
    raise ValueError(f"unsupported pruning_significance_metric: {metric!r}")


def _refit_single_row_with_pruning(
    search: DistributionGMMSearch,
    *,
    row_idx: int,
    selected_k: int,
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    lefts: np.ndarray,
    rights: np.ndarray,
    component_masses: np.ndarray,
    jsd: float,
    cross_entropy: float,
    l1: float,
    training_config: GMMTrainingConfig,
    device: str | torch.device,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    current_k = int(selected_k)
    current_weights = np.asarray(weights, dtype=DTYPE_NP).copy()
    current_means = np.asarray(means, dtype=DTYPE_NP).copy()
    current_stds = np.asarray(stds, dtype=DTYPE_NP).copy()
    current_lefts = np.asarray(lefts, dtype=DTYPE_NP).copy()
    current_rights = np.asarray(rights, dtype=DTYPE_NP).copy()
    current_component_masses = np.asarray(component_masses, dtype=DTYPE_NP).copy()
    current_metrics = {
        "jsd": float(jsd),
        "cross_entropy": float(cross_entropy),
        "l1": float(l1),
    }
    metric_name = training_config.pruning_error_metric
    threshold = float(training_config.pruning_error_threshold)
    best_k = current_k
    best_weights = current_weights.copy()
    best_means = current_means.copy()
    best_stds = current_stds.copy()
    best_lefts = current_lefts.copy()
    best_rights = current_rights.copy()
    best_component_masses = current_component_masses.copy()
    best_metrics = dict(current_metrics)
    if metric_name not in current_metrics:
        raise ValueError(f"unsupported pruning_error_metric: {metric_name!r}")

    for _ in range(training_config.pruning_max_refits):
        current_metric = float(current_metrics[metric_name])
        if current_metric <= threshold:
            break
        if current_k <= training_config.pruning_min_components:
            break
        significance = _component_significance(
            weights=current_weights[:current_k],
            component_masses=current_component_masses[:, :current_k],
            metric=training_config.pruning_significance_metric,
        )
        drop_idx = int(np.argmin(significance))
        next_k = current_k - 1
        next_weights = np.zeros((1, search.component_weights.shape[1]), dtype=DTYPE_NP)
        next_means = np.zeros_like(next_weights)
        next_stds = np.ones_like(next_weights)
        next_lefts = np.repeat(search.lower_bounds[[row_idx], None], next_weights.shape[1], axis=1)
        next_rights = np.repeat(search.upper_bounds[[row_idx], None], next_weights.shape[1], axis=1)
        keep = [idx for idx in range(current_k) if idx != drop_idx]
        next_weights[0, :next_k] = current_weights[keep]
        next_means[0, :next_k] = current_means[keep]
        next_stds[0, :next_k] = current_stds[keep]
        next_lefts[0, :next_k] = current_lefts[keep]
        next_rights[0, :next_k] = current_rights[keep]
        refit_result = optimize_mixture_parameters(
            probabilities=search.probabilities[[row_idx]],
            bin_edges=search.bin_edges[[row_idx]],
            support_mask=search.support_mask[[row_idx]],
            lower_bounds=search.lower_bounds[[row_idx]],
            upper_bounds=search.upper_bounds[[row_idx]],
            selected_k=np.asarray([next_k], dtype=np.int64),
            initial_weights=next_weights,
            initial_means=next_means,
            initial_stds=next_stds,
            initial_left_truncations=next_lefts,
            initial_right_truncations=next_rights,
            max_iterations=training_config.max_iterations,
            learning_rate=training_config.learning_rate,
            convergence_tolerance=training_config.convergence_tolerance,
            overshoot_penalty=training_config.overshoot_penalty,
            mean_margin_fraction=training_config.mean_margin_fraction,
            torch_dtype=training_config.torch_dtype,
            device=device,
            compile_model=training_config.compile_model,
            optimize_weights=training_config.optimize_weights,
            optimize_means=training_config.optimize_means,
            optimize_stds=training_config.optimize_stds,
            optimize_left_truncations=training_config.optimize_left_truncations,
            optimize_right_truncations=training_config.optimize_right_truncations,
            sigma_floor_fraction=training_config.sigma_floor_fraction,
            min_window_fraction=training_config.min_window_fraction,
            truncation_mode=training_config.truncation_mode,
            truncation_regularization_strength=training_config.truncation_regularization_strength,
            initial_weight_logit_floor=training_config.initial_weight_logit_floor,
            inactive_weight_floor=training_config.inactive_weight_floor,
            masked_logit_value=training_config.masked_logit_value,
            logit_clip=training_config.logit_clip,
            inverse_softplus_clip=training_config.inverse_softplus_clip,
        )
        (
            sorted_weights,
            sorted_means,
            sorted_stds,
            sorted_lefts,
            sorted_rights,
            sorted_component_masses,
        ) = _sort_components(
            np.asarray([next_k], dtype=np.int64),
            refit_result.component_weights,
            refit_result.component_means,
            refit_result.component_stds,
            refit_result.component_left_truncations,
            refit_result.component_right_truncations,
            refit_result.component_masses,
        )
        candidate_metrics = {
            "jsd": float(refit_result.jsd[0]),
            "cross_entropy": float(refit_result.cross_entropy[0]),
            "l1": float(refit_result.l1_error[0]),
        }
        candidate_metric = float(candidate_metrics[metric_name])
        if candidate_metric < float(best_metrics[metric_name]):
            best_k = next_k
            best_weights = sorted_weights[0].copy()
            best_means = sorted_means[0].copy()
            best_stds = sorted_stds[0].copy()
            best_lefts = sorted_lefts[0].copy()
            best_rights = sorted_rights[0].copy()
            best_component_masses = sorted_component_masses[0].copy()
            best_metrics = candidate_metrics
        if candidate_metric >= current_metric - 1e-12:
            break
        current_k = next_k
        current_weights = sorted_weights[0].copy()
        current_means = sorted_means[0].copy()
        current_stds = sorted_stds[0].copy()
        current_lefts = sorted_lefts[0].copy()
        current_rights = sorted_rights[0].copy()
        current_component_masses = sorted_component_masses[0].copy()
        current_metrics = candidate_metrics

    return (
        best_k,
        best_weights,
        best_means,
        best_stds,
        best_lefts,
        best_rights,
        best_component_masses,
        float(best_metrics["jsd"]),
        float(best_metrics["cross_entropy"]),
        float(best_metrics["l1"]),
    )


def _refit_hard_rows_with_multistart(
    search: DistributionGMMSearch,
    *,
    selected_k: np.ndarray,
    fitted: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    lefts: np.ndarray,
    rights: np.ndarray,
    component_masses: np.ndarray,
    jsd: np.ndarray,
    cross_entropy: np.ndarray,
    l1: np.ndarray,
    training_config: GMMTrainingConfig,
    device: str | torch.device,
) -> None:
    if training_config.multi_start_count <= 1:
        return
    rng = np.random.default_rng(training_config.multi_start_seed)
    metric_name = training_config.multi_start_metric
    component_count = weights.shape[1]
    for row_idx, k in enumerate(selected_k.tolist()):
        current_metric = {
            "jsd": float(jsd[row_idx]),
            "cross_entropy": float(cross_entropy[row_idx]),
            "l1": float(l1[row_idx]),
        }[metric_name]
        if current_metric <= training_config.multi_start_trigger_threshold:
            continue
        lower_bound = float(search.lower_bounds[row_idx])
        upper_bound = float(search.upper_bounds[row_idx])
        span = max(upper_bound - lower_bound, 1e-12)
        best_metric = current_metric
        best_fitted = fitted[row_idx].copy()
        best_weights = weights[row_idx].copy()
        best_means = means[row_idx].copy()
        best_stds = stds[row_idx].copy()
        best_lefts = lefts[row_idx].copy()
        best_rights = rights[row_idx].copy()
        best_component_masses = component_masses[row_idx].copy()
        best_jsd = float(jsd[row_idx])
        best_cross_entropy = float(cross_entropy[row_idx])
        best_l1 = float(l1[row_idx])
        for _ in range(training_config.multi_start_count - 1):
            init_weights = np.zeros((1, component_count), dtype=DTYPE_NP)
            init_means = np.zeros_like(init_weights)
            init_stds = np.ones_like(init_weights)
            init_lefts = np.repeat(search.lower_bounds[[row_idx], None], component_count, axis=1)
            init_rights = np.repeat(search.upper_bounds[[row_idx], None], component_count, axis=1)
            perturbed_weights = best_weights[:k] * np.exp(
                rng.normal(0.0, training_config.multi_start_jitter_scale, size=k)
            )
            perturbed_weights = np.clip(perturbed_weights, 1e-12, None)
            perturbed_weights = perturbed_weights / max(
                float(np.sum(perturbed_weights)),
                1e-12,
            )
            perturbed_means = best_means[:k] + rng.normal(
                0.0,
                training_config.multi_start_jitter_scale * span,
                size=k,
            )
            perturbed_stds = best_stds[:k] * np.exp(
                rng.normal(0.0, training_config.multi_start_jitter_scale, size=k)
            )
            init_weights[0, :k] = perturbed_weights
            init_means[0, :k] = perturbed_means
            init_stds[0, :k] = np.clip(perturbed_stds, 1e-12, None)
            init_lefts[0, :k] = best_lefts[:k]
            init_rights[0, :k] = best_rights[:k]
            candidate = optimize_mixture_parameters(
                probabilities=search.probabilities[[row_idx]],
                bin_edges=search.bin_edges[[row_idx]],
                support_mask=search.support_mask[[row_idx]],
                lower_bounds=search.lower_bounds[[row_idx]],
                upper_bounds=search.upper_bounds[[row_idx]],
                selected_k=np.asarray([k], dtype=np.int64),
                initial_weights=init_weights,
                initial_means=init_means,
                initial_stds=init_stds,
                initial_left_truncations=init_lefts,
                initial_right_truncations=init_rights,
                max_iterations=training_config.max_iterations,
                learning_rate=training_config.learning_rate,
                convergence_tolerance=training_config.convergence_tolerance,
                overshoot_penalty=training_config.overshoot_penalty,
                mean_margin_fraction=training_config.mean_margin_fraction,
                torch_dtype=training_config.torch_dtype,
                device=device,
                compile_model=training_config.compile_model,
                optimize_weights=training_config.optimize_weights,
                optimize_means=training_config.optimize_means,
                optimize_stds=training_config.optimize_stds,
                optimize_left_truncations=training_config.optimize_left_truncations,
                optimize_right_truncations=training_config.optimize_right_truncations,
                sigma_floor_fraction=training_config.sigma_floor_fraction,
                min_window_fraction=training_config.min_window_fraction,
                truncation_mode=training_config.truncation_mode,
                truncation_regularization_strength=training_config.truncation_regularization_strength,
                initial_weight_logit_floor=training_config.initial_weight_logit_floor,
                inactive_weight_floor=training_config.inactive_weight_floor,
                masked_logit_value=training_config.masked_logit_value,
                logit_clip=training_config.logit_clip,
                inverse_softplus_clip=training_config.inverse_softplus_clip,
            )
            (
                sorted_weights,
                sorted_means,
                sorted_stds,
                sorted_lefts,
                sorted_rights,
                sorted_component_masses,
            ) = _sort_components(
                np.asarray([k], dtype=np.int64),
                candidate.component_weights,
                candidate.component_means,
                candidate.component_stds,
                candidate.component_left_truncations,
                candidate.component_right_truncations,
                candidate.component_masses,
            )
            candidate_metric = float(_metric_values(candidate, metric=metric_name)[0])
            if candidate_metric >= best_metric:
                continue
            best_metric = candidate_metric
            best_fitted = candidate.fitted_probabilities[0].copy()
            best_weights = sorted_weights[0].copy()
            best_means = sorted_means[0].copy()
            best_stds = sorted_stds[0].copy()
            best_lefts = sorted_lefts[0].copy()
            best_rights = sorted_rights[0].copy()
            best_component_masses = sorted_component_masses[0].copy()
            best_jsd = float(candidate.jsd[0])
            best_cross_entropy = float(candidate.cross_entropy[0])
            best_l1 = float(candidate.l1_error[0])
        fitted[row_idx] = best_fitted
        weights[row_idx] = best_weights
        means[row_idx] = best_means
        stds[row_idx] = best_stds
        lefts[row_idx] = best_lefts
        rights[row_idx] = best_rights
        component_masses[row_idx] = best_component_masses
        jsd[row_idx] = best_jsd
        cross_entropy[row_idx] = best_cross_entropy
        l1[row_idx] = best_l1


def _fit_search_result(
    search: DistributionGMMSearch,
    *,
    training_config: GMMTrainingConfig,
    device: str | torch.device,
) -> DistributionGMMReport:
    n_genes = search.support.shape[0]
    fitted = np.zeros_like(search.probabilities)
    weights = np.zeros_like(search.component_weights)
    means = np.zeros_like(search.component_means)
    stds = np.zeros_like(search.component_stds)
    lefts = np.zeros_like(search.component_left_truncations)
    rights = np.zeros_like(search.component_right_truncations)
    component_masses = np.zeros(
        (
            search.probabilities.shape[0],
            search.probabilities.shape[1],
            search.component_weights.shape[1],
        ),
        dtype=DTYPE_NP,
    )
    jsd = np.zeros(n_genes, dtype=DTYPE_NP)
    cross_entropy = np.zeros(n_genes, dtype=DTYPE_NP)
    l1 = np.zeros(n_genes, dtype=DTYPE_NP)
    search_selected_k = np.asarray(search.selected_k, dtype=np.int64)
    selected_k = search_selected_k.copy()

    for start in range(0, n_genes, training_config.gene_chunk_size):
        stop = min(start + training_config.gene_chunk_size, n_genes)
        result = _run_refit(
            search,
            row_slice=slice(start, stop),
            selected_k=selected_k[start:stop],
            initial_weights=search.component_weights[start:stop],
            initial_means=search.component_means[start:stop],
            initial_stds=search.component_stds[start:stop],
            initial_lefts=search.component_left_truncations[start:stop],
            initial_rights=search.component_right_truncations[start:stop],
            training_config=training_config,
            device=device,
        )
        fitted[start:stop] = result.fitted_probabilities
        weights[start:stop] = result.component_weights
        means[start:stop] = result.component_means
        stds[start:stop] = result.component_stds
        lefts[start:stop] = result.component_left_truncations
        rights[start:stop] = result.component_right_truncations
        component_masses[start:stop] = result.component_masses
        jsd[start:stop] = result.jsd
        cross_entropy[start:stop] = result.cross_entropy
        l1[start:stop] = result.l1_error

    weights, means, stds, lefts, rights, component_masses = _sort_components(
        selected_k,
        weights,
        means,
        stds,
        lefts,
        rights,
        component_masses,
    )

    if training_config.pruning_enabled and training_config.pruning_max_refits > 0:
        for row_idx in range(n_genes):
            (
                pruned_k,
                pruned_weights,
                pruned_means,
                pruned_stds,
                pruned_lefts,
                pruned_rights,
                pruned_component_masses,
                pruned_jsd,
                pruned_cross_entropy,
                pruned_l1,
            ) = _refit_single_row_with_pruning(
                search,
                row_idx=row_idx,
                selected_k=int(selected_k[row_idx]),
                weights=weights[row_idx],
                means=means[row_idx],
                stds=stds[row_idx],
                lefts=lefts[row_idx],
                rights=rights[row_idx],
                component_masses=component_masses[row_idx],
                jsd=float(jsd[row_idx]),
                cross_entropy=float(cross_entropy[row_idx]),
                l1=float(l1[row_idx]),
                training_config=training_config,
                device=device,
            )
            selected_k[row_idx] = pruned_k
            weights[row_idx] = pruned_weights
            means[row_idx] = pruned_means
            stds[row_idx] = pruned_stds
            lefts[row_idx] = pruned_lefts
            rights[row_idx] = pruned_rights
            component_masses[row_idx] = pruned_component_masses
            jsd[row_idx] = pruned_jsd
            cross_entropy[row_idx] = pruned_cross_entropy
            l1[row_idx] = pruned_l1

    changed_rows = np.flatnonzero(search_selected_k != selected_k)
    if changed_rows.size > 0:
        reevaluated = evaluate_mixture_parameters(
            probabilities=search.probabilities[changed_rows],
            bin_edges=search.bin_edges[changed_rows],
            support_mask=search.support_mask[changed_rows],
            lower_bounds=search.lower_bounds[changed_rows],
            upper_bounds=search.upper_bounds[changed_rows],
            selected_k=selected_k[changed_rows],
            component_weights=weights[changed_rows],
            component_means=means[changed_rows],
            component_stds=stds[changed_rows],
            component_left_truncations=lefts[changed_rows],
            component_right_truncations=rights[changed_rows],
            torch_dtype=training_config.torch_dtype,
            device=device,
        )
        fitted[changed_rows] = reevaluated.fitted_probabilities
        component_masses[changed_rows] = reevaluated.component_masses
        jsd[changed_rows] = reevaluated.jsd
        cross_entropy[changed_rows] = reevaluated.cross_entropy
        l1[changed_rows] = reevaluated.l1_error

    _refit_hard_rows_with_multistart(
        search,
        selected_k=selected_k,
        fitted=fitted,
        weights=weights,
        means=means,
        stds=stds,
        lefts=lefts,
        rights=rights,
        component_masses=component_masses,
        jsd=jsd,
        cross_entropy=cross_entropy,
        l1=l1,
        training_config=training_config,
        device=device,
    )

    for row_idx, k in enumerate(selected_k.tolist()):
        if k < weights.shape[1]:
            weights[row_idx, k:] = 0.0
            means[row_idx, k:] = 0.0
            stds[row_idx, k:] = 1.0
            lefts[row_idx, k:] = search.lower_bounds[row_idx]
            rights[row_idx, k:] = search.upper_bounds[row_idx]
            component_masses[row_idx, :, k:] = 0.0

    greedy_probabilities = np.asarray(search.greedy_probabilities, dtype=DTYPE_NP).copy()
    greedy_error = np.asarray(
        [search.error_path[row_idx, int(k) - 1] for row_idx, k in enumerate(selected_k)],
        dtype=DTYPE_NP,
    )
    for row_idx, k in enumerate(selected_k.tolist()):
        if int(search_selected_k[row_idx]) != k:
            greedy_probabilities[row_idx] = fitted[row_idx]
            greedy_error[row_idx] = jsd[row_idx]

    return DistributionGMMReport(
        support_domain=search.support_domain,
        support=search.support,
        support_mask=search.support_mask,
        probabilities=search.probabilities,
        bin_edges=search.bin_edges,
        lower_bounds=search.lower_bounds,
        upper_bounds=search.upper_bounds,
        selected_k=selected_k,
        fitted_probabilities=fitted,
        residual_probabilities=search.probabilities - fitted,
        component_weights=weights,
        component_means=means,
        component_stds=stds,
        component_left_truncations=lefts,
        component_right_truncations=rights,
        greedy_probabilities=greedy_probabilities,
        jsd=jsd,
        cross_entropy=cross_entropy,
        l1_error=l1,
        greedy_error=greedy_error,
        explored_k=search.explored_k,
        config={
            **dict(search.config),
            **asdict(training_config),
        },
    )


def fit_distribution_gmm(
    distribution: DistributionGrid,
    *,
    search: DistributionGMMSearch | None = None,
    search_config: GMMSearchConfig = GMMSearchConfig(),
    training_config: GMMTrainingConfig = GMMTrainingConfig(),
    device: str | torch.device = "cpu",
) -> DistributionGMMReport:
    if search is None:
        search = search_distribution_gmm(
            distribution,
            config=search_config,
            device=device,
            torch_dtype=training_config.torch_dtype,
        )
    prepared = _prepare_distribution_grid(
        distribution,
        config=_resolve_search_config(search, fallback=search_config),
    )
    _validate_search_matches(
        search,
        support=prepared.support,
        probabilities=prepared.probabilities,
        support_domain=prepared.support_domain,
    )
    return _fit_search_result(
        search,
        training_config=training_config,
        device=device,
    )


def fit_prior_gmm(
    prior: PriorGrid,
    *,
    search: DistributionGMMSearch | None = None,
    search_config: GMMSearchConfig = GMMSearchConfig(),
    training_config: GMMTrainingConfig = GMMTrainingConfig(),
    device: str | torch.device = "cpu",
    support_axis: SupportAxis = "raw",
) -> PriorGMMReport:
    if search is None:
        search = search_prior_gmm(
            prior,
            config=search_config,
            device=device,
            torch_dtype=training_config.torch_dtype,
            support_axis=support_axis,
        )
    prepared = _prepare_prior_grid(
        prior,
        support_axis=support_axis,
        config=_resolve_search_config(search, fallback=search_config),
    )
    _validate_search_matches(
        search,
        support=prepared.support,
        probabilities=prepared.probabilities,
        support_domain=prepared.support_domain,
    )
    gene_specific = prior.as_gene_specific()
    report = _fit_search_result(
        search,
        training_config=training_config,
        device=device,
    )
    if support_axis == "scaled" or prior.support_domain == "rate":
        scaled_support = np.asarray(report.support, dtype=DTYPE_NP)
    else:
        scaled_support = np.asarray(report.support, dtype=DTYPE_NP) * float(prior.scale)
    return PriorGMMReport(
        gene_names=list(gene_specific.gene_names),
        support_domain=report.support_domain,
        support=report.support,
        scaled_support=scaled_support,
        support_mask=report.support_mask,
        probabilities=report.probabilities,
        lower_bounds=report.lower_bounds,
        upper_bounds=report.upper_bounds,
        selected_k=report.selected_k,
        fitted_probabilities=report.fitted_probabilities,
        residual_probabilities=report.residual_probabilities,
        component_weights=report.component_weights,
        component_means=report.component_means,
        component_stds=report.component_stds,
        component_left_truncations=report.component_left_truncations,
        component_right_truncations=report.component_right_truncations,
        jsd=report.jsd,
        cross_entropy=report.cross_entropy,
        l1_error=report.l1_error,
        explored_k=report.explored_k,
        scale=float(prior.scale),
        config={
            **dict(report.config),
            "support_axis": support_axis,
        },
    )


__all__ = ["fit_distribution_gmm", "fit_prior_gmm"]
