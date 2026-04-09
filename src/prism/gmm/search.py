from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from prism.model.constants import DTYPE_NP, EPS, SupportDomain
from prism.model.types import DistributionGrid, PriorGrid

from .numeric import build_bin_edges, resolve_torch_dtype, truncated_gaussian_bin_masses
from .optimize import MixtureOptimizationResult, evaluate_mixture_parameters, optimize_mixture_parameters
from .schema import DistributionGMMSearch, GMMSearchConfig, SupportAxis


@dataclass(frozen=True, slots=True)
class _PreparedGridBatch:
    support_domain: SupportDomain
    support: np.ndarray
    probabilities: np.ndarray
    support_mask: np.ndarray
    bin_edges: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    @property
    def n_genes(self) -> int:
        return int(self.support.shape[0])

    @property
    def n_support_points(self) -> int:
        return int(self.support.shape[1])


@dataclass(frozen=True, slots=True)
class _StagePrefixBatch:
    probabilities: np.ndarray
    bin_edges: np.ndarray
    support_mask: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    selected_k: np.ndarray
    component_weights: np.ndarray
    component_means: np.ndarray
    component_stds: np.ndarray
    component_lefts: np.ndarray
    component_rights: np.ndarray
    row_gene_indices: np.ndarray
    row_target_k: np.ndarray


def _merge_sorted_support_points(
    support: np.ndarray,
    probabilities: np.ndarray,
    *,
    merge_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    merged_support: list[float] = []
    merged_probabilities: list[float] = []
    for support_value, probability_value in zip(support, probabilities, strict=True):
        current_support = float(support_value)
        current_probability = float(probability_value)
        if not merged_support:
            merged_support.append(current_support)
            merged_probabilities.append(current_probability)
            continue
        if abs(current_support - merged_support[-1]) <= merge_tolerance:
            merged_probabilities[-1] += current_probability
            continue
        merged_support.append(current_support)
        merged_probabilities.append(current_probability)
    support_np = np.asarray(merged_support, dtype=DTYPE_NP)
    probabilities_np = np.asarray(merged_probabilities, dtype=DTYPE_NP)
    probabilities_np = np.clip(probabilities_np, 0.0, None)
    probabilities_np = probabilities_np / max(float(probabilities_np.sum()), EPS)
    return support_np, probabilities_np


def _prepare_support_probability_rows(
    support: np.ndarray,
    probabilities: np.ndarray,
    *,
    support_domain: SupportDomain,
    config: GMMSearchConfig,
) -> _PreparedGridBatch:
    support_np = np.asarray(support, dtype=DTYPE_NP)
    probabilities_np = np.asarray(probabilities, dtype=DTYPE_NP)
    if support_np.ndim == 1:
        support_np = support_np[None, :]
        probabilities_np = probabilities_np[None, :]
    if support_np.shape != probabilities_np.shape:
        raise ValueError("support and probabilities must match")

    normalized_support: list[np.ndarray] = []
    normalized_probabilities: list[np.ndarray] = []
    edges_list: list[np.ndarray] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    max_points = 0
    for row_idx in range(support_np.shape[0]):
        row_support = support_np[row_idx].reshape(-1)
        row_probabilities = probabilities_np[row_idx].reshape(-1)
        order = np.argsort(row_support, kind="stable")
        sorted_support = row_support[order]
        sorted_probabilities = row_probabilities[order]
        merged_support, merged_probabilities = _merge_sorted_support_points(
            sorted_support,
            sorted_probabilities,
            merge_tolerance=config.merge_tolerance,
        )
        edges, lower_bound, upper_bound = build_bin_edges(
            merged_support,
            support_domain=support_domain,
            midpoint_fraction=config.bin_edge_midpoint_fraction,
            single_point_rate_half_width_scale=config.single_point_rate_half_width_scale,
            single_point_rate_half_width_floor=config.single_point_rate_half_width_floor,
        )
        normalized_support.append(merged_support)
        normalized_probabilities.append(merged_probabilities)
        edges_list.append(edges)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
        max_points = max(max_points, merged_support.shape[0])

    padded_support = np.zeros((support_np.shape[0], max_points), dtype=DTYPE_NP)
    padded_probabilities = np.zeros_like(padded_support)
    padded_mask = np.zeros_like(padded_support, dtype=bool)
    padded_edges = np.zeros((support_np.shape[0], max_points + 1), dtype=DTYPE_NP)
    for row_idx, (row_support, row_probabilities, row_edges) in enumerate(
        zip(normalized_support, normalized_probabilities, edges_list, strict=True)
    ):
        n_points = row_support.shape[0]
        padded_support[row_idx, :n_points] = row_support
        padded_probabilities[row_idx, :n_points] = row_probabilities
        padded_mask[row_idx, :n_points] = True
        padded_edges[row_idx, : n_points + 1] = row_edges
        if n_points < max_points:
            padded_support[row_idx, n_points:] = row_support[-1]
            padded_edges[row_idx, n_points + 1 :] = row_edges[-1]
    return _PreparedGridBatch(
        support_domain=support_domain,
        support=padded_support,
        probabilities=padded_probabilities,
        support_mask=padded_mask,
        bin_edges=padded_edges,
        lower_bounds=np.asarray(lower_bounds, dtype=DTYPE_NP),
        upper_bounds=np.asarray(upper_bounds, dtype=DTYPE_NP),
    )


def _prepare_distribution_grid(
    distribution: DistributionGrid,
    *,
    config: GMMSearchConfig,
) -> _PreparedGridBatch:
    gene_specific = distribution.as_gene_specific()
    return _prepare_support_probability_rows(
        np.asarray(gene_specific.support, dtype=DTYPE_NP),
        np.asarray(gene_specific.probabilities, dtype=DTYPE_NP),
        support_domain=gene_specific.support_domain,
        config=config,
    )


def _prepare_prior_grid(
    prior: PriorGrid,
    *,
    support_axis: SupportAxis,
    config: GMMSearchConfig,
) -> _PreparedGridBatch:
    gene_specific = prior.as_gene_specific()
    if support_axis == "scaled":
        support = np.asarray(gene_specific.scaled_support, dtype=DTYPE_NP)
        support_domain: SupportDomain = "rate"
    else:
        support = np.asarray(gene_specific.support, dtype=DTYPE_NP)
        support_domain = gene_specific.support_domain
    return _prepare_support_probability_rows(
        support,
        np.asarray(gene_specific.prior_probabilities, dtype=DTYPE_NP),
        support_domain=support_domain,
        config=config,
    )


def _window_radii(n_points: int, *, count: int) -> np.ndarray:
    if n_points <= 1:
        return np.asarray([0], dtype=np.int64)
    raw = np.geomspace(1.0, float(n_points), num=count)
    radii = np.unique(np.clip(np.round(raw).astype(np.int64) - 1, 0, n_points - 1))
    if radii[0] != 0:
        radii = np.concatenate([np.asarray([0], dtype=np.int64), radii])
    return radii


def _peak_indices(
    residual: np.ndarray,
    *,
    config: GMMSearchConfig,
) -> np.ndarray:
    values = np.asarray(residual, dtype=DTYPE_NP).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    left = np.concatenate([np.asarray([-np.inf]), values[:-1]])
    right = np.concatenate([values[1:], np.asarray([-np.inf])])
    local_max = np.flatnonzero(
        (values >= left) & (values >= right) & (values > config.peak_min_value)
    )
    if not config.include_boundary_peaks:
        local_max = local_max[(local_max > 0) & (local_max < values.size - 1)]
    if local_max.size == 0:
        argmax_idx = int(np.argmax(values))
        if values[argmax_idx] <= config.peak_min_value:
            return np.zeros(0, dtype=np.int64)
        local_max = np.asarray([argmax_idx], dtype=np.int64)
    order = np.argsort(values[local_max], kind="stable")[::-1]
    peaks = local_max[order]
    if config.peak_limit_per_stage is not None:
        peaks = peaks[: config.peak_limit_per_stage]
    return peaks.astype(np.int64, copy=False)


def _candidate_sigma_values(
    *,
    bin_edges: np.ndarray,
    left_idx: int,
    right_idx: int,
    min_sigma: float,
    config: GMMSearchConfig,
) -> np.ndarray:
    width = max(float(bin_edges[right_idx + 1] - bin_edges[left_idx]), min_sigma)
    max_sigma = max(width, min_sigma * config.candidate_sigma_max_scale)
    if config.candidate_sigma_count == 1:
        return np.asarray([min_sigma], dtype=DTYPE_NP)
    return np.geomspace(min_sigma, max_sigma, num=config.candidate_sigma_count).astype(
        DTYPE_NP
    )


def _fallback_candidate(
    support: np.ndarray,
    probabilities: np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
    min_sigma: float,
) -> tuple[float, float, float, float]:
    centroid = float(np.sum(support * probabilities))
    variance = float(np.sum(probabilities * (support - centroid) ** 2))
    std = max(math.sqrt(max(variance, 0.0)), min_sigma)
    return centroid, std, lower_bound, upper_bound


def _build_peak_candidates(
    support: np.ndarray,
    probabilities: np.ndarray,
    bin_edges: np.ndarray,
    residual: np.ndarray,
    *,
    peak_idx: int,
    lower_bound: float,
    upper_bound: float,
    min_sigma: float,
    config: GMMSearchConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radii = _window_radii(support.size, count=config.candidate_window_count)
    means: list[float] = []
    stds: list[float] = []
    lefts: list[float] = []
    rights: list[float] = []
    for radius in radii.tolist():
        left_idx = max(0, peak_idx - radius)
        right_idx = min(support.size - 1, peak_idx + radius)
        left_bound = float(bin_edges[left_idx])
        right_bound = float(bin_edges[right_idx + 1])
        local_residual = residual[left_idx : right_idx + 1]
        local_support = support[left_idx : right_idx + 1]
        local_mass = float(np.sum(local_residual))
        mu_candidates = [float(support[peak_idx])]
        if local_mass > 0:
            mu_candidates.append(
                float(np.sum(local_support * local_residual) / max(local_mass, EPS))
            )
        window_width = max(right_bound - left_bound, min_sigma)
        if peak_idx == 0 or left_idx == 0:
            mu_candidates.append(
                left_bound - config.boundary_mean_margin_scale * window_width
            )
        if peak_idx == support.size - 1 or right_idx == support.size - 1:
            mu_candidates.append(
                right_bound + config.boundary_mean_margin_scale * window_width
            )
        sigma_values = _candidate_sigma_values(
            bin_edges=bin_edges,
            left_idx=left_idx,
            right_idx=right_idx,
            min_sigma=min_sigma,
            config=config,
        )
        for mu_candidate in mu_candidates:
            for sigma_candidate in sigma_values.tolist():
                means.append(mu_candidate)
                stds.append(float(sigma_candidate))
                lefts.append(left_bound)
                rights.append(right_bound)
    fallback = _fallback_candidate(
        support,
        probabilities,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        min_sigma=min_sigma,
    )
    means.append(fallback[0])
    stds.append(fallback[1])
    lefts.append(fallback[2])
    rights.append(fallback[3])
    return (
        np.asarray(means, dtype=DTYPE_NP),
        np.asarray(stds, dtype=DTYPE_NP),
        np.asarray(lefts, dtype=DTYPE_NP),
        np.asarray(rights, dtype=DTYPE_NP),
    )


def _score_peak_candidates(
    residual: np.ndarray,
    support_mask: np.ndarray,
    bin_edges: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    lefts: np.ndarray,
    rights: np.ndarray,
    *,
    config: GMMSearchConfig,
    device: str | torch.device,
    torch_dtype: str,
) -> tuple[float, float, float, float, float, float]:
    dtype_obj = resolve_torch_dtype(torch_dtype)
    device_obj = torch.device(device)
    with torch.no_grad():
        residual_t = torch.as_tensor(
            residual[None, :],
            dtype=dtype_obj,
            device=device_obj,
        )
        bin_edges_t = torch.as_tensor(
            bin_edges[None, :],
            dtype=dtype_obj,
            device=device_obj,
        )
        support_mask_t = torch.as_tensor(
            support_mask[None, :],
            dtype=torch.bool,
            device=device_obj,
        )
        means_t = torch.as_tensor(means[None, :], dtype=dtype_obj, device=device_obj)
        stds_t = torch.as_tensor(stds[None, :], dtype=dtype_obj, device=device_obj)
        lefts_t = torch.as_tensor(lefts[None, :], dtype=dtype_obj, device=device_obj)
        rights_t = torch.as_tensor(rights[None, :], dtype=dtype_obj, device=device_obj)
        candidate_masses = truncated_gaussian_bin_masses(
            bin_edges_t,
            means_t,
            stds_t,
            lefts_t,
            rights_t,
            support_mask_t,
        )[0]
        max_mass = candidate_masses.max(dim=0).values.clamp_min(EPS)
        significant = candidate_masses > config.mass_floor * max_mass.unsqueeze(0)
        ratio = residual_t[0].unsqueeze(-1) / candidate_masses.clamp_min(EPS)
        ratio = torch.where(significant, ratio, torch.full_like(ratio, float("inf")))
        alpha_max = torch.min(ratio, dim=0).values
        alpha = torch.minimum(
            alpha_max * config.candidate_weight_slack,
            residual_t[0].sum(),
        )
        residual_after = torch.clamp(
            residual_t[0].unsqueeze(-1) - alpha.unsqueeze(0) * candidate_masses,
            min=0.0,
        )
        baseline = torch.sum(residual_t[0] * residual_t[0])
        improvement = baseline - torch.sum(residual_after * residual_after, dim=0)
        valid = (alpha >= config.min_component_mass) & torch.isfinite(improvement)
        if not torch.any(valid):
            best_idx = int(torch.argmax(alpha).detach().cpu().item())
            return (
                float(alpha[best_idx].detach().cpu().item()),
                float(improvement[best_idx].detach().cpu().item()),
                float(means[best_idx]),
                float(stds[best_idx]),
                float(lefts[best_idx]),
                float(rights[best_idx]),
            )
        masked_improvement = torch.where(
            valid,
            improvement,
            torch.full_like(improvement, -float("inf")),
        )
        best_idx = int(torch.argmax(masked_improvement).detach().cpu().item())
    return (
        float(alpha[best_idx].detach().cpu().item()),
        float(improvement[best_idx].detach().cpu().item()),
        float(means[best_idx]),
        float(stds[best_idx]),
        float(lefts[best_idx]),
        float(rights[best_idx]),
    )


def _extract_stage_components_for_gene(
    prepared: _PreparedGridBatch,
    *,
    row_idx: int,
    residual: np.ndarray,
    frontier_k: int,
    config: GMMSearchConfig,
    device: str | torch.device,
    torch_dtype: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = prepared.support_mask[row_idx]
    row_support = prepared.support[row_idx, mask]
    row_probabilities = prepared.probabilities[row_idx, mask]
    row_bin_edges = prepared.bin_edges[row_idx, : row_support.shape[0] + 1]
    row_residual = residual[mask]
    lower_bound = float(prepared.lower_bounds[row_idx])
    upper_bound = float(prepared.upper_bounds[row_idx])
    if row_support.size == 0:
        return (
            np.zeros(0, dtype=DTYPE_NP),
            np.zeros(0, dtype=DTYPE_NP),
            np.zeros(0, dtype=DTYPE_NP),
            np.zeros(0, dtype=DTYPE_NP),
            np.zeros(0, dtype=DTYPE_NP),
        )
    if row_support.size == 1:
        min_sigma = max((upper_bound - lower_bound) * config.min_sigma_factor, EPS)
        mu, sigma, left, right = _fallback_candidate(
            row_support,
            row_probabilities,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            min_sigma=min_sigma,
        )
        return (
            np.asarray([1.0], dtype=DTYPE_NP),
            np.asarray([mu], dtype=DTYPE_NP),
            np.asarray([sigma], dtype=DTYPE_NP),
            np.asarray([left], dtype=DTYPE_NP),
            np.asarray([right], dtype=DTYPE_NP),
        )

    positive_diffs = np.diff(row_support)
    positive_diffs = positive_diffs[positive_diffs > 0]
    default_gap = float(config.default_support_gap)
    min_diff = float(np.min(positive_diffs)) if positive_diffs.size else default_gap
    min_sigma = max(min_diff * config.min_sigma_factor, EPS)
    peaks = _peak_indices(row_residual, config=config)
    candidate_rows: list[tuple[float, float, float, float, float, float]] = []
    for peak_idx in peaks.tolist():
        means, stds, lefts, rights = _build_peak_candidates(
            row_support,
            row_probabilities,
            row_bin_edges,
            row_residual,
            peak_idx=peak_idx,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            min_sigma=min_sigma,
            config=config,
        )
        alpha, improvement, mean, std, left, right = _score_peak_candidates(
            row_residual,
            np.ones(row_support.shape[0], dtype=bool),
            row_bin_edges,
            means,
            stds,
            lefts,
            rights,
            config=config,
            device=device,
            torch_dtype=torch_dtype,
        )
        if alpha < config.min_component_mass:
            continue
        candidate_rows.append((improvement, alpha, mean, std, left, right))

    if not candidate_rows:
        if frontier_k > 0:
            return (
                np.zeros(0, dtype=DTYPE_NP),
                np.zeros(0, dtype=DTYPE_NP),
                np.zeros(0, dtype=DTYPE_NP),
                np.zeros(0, dtype=DTYPE_NP),
                np.zeros(0, dtype=DTYPE_NP),
            )
        mu, sigma, left, right = _fallback_candidate(
            row_support,
            row_probabilities,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            min_sigma=min_sigma,
        )
        return (
            np.asarray([1.0], dtype=DTYPE_NP),
            np.asarray([mu], dtype=DTYPE_NP),
            np.asarray([sigma], dtype=DTYPE_NP),
            np.asarray([left], dtype=DTYPE_NP),
            np.asarray([right], dtype=DTYPE_NP),
        )

    candidate_rows.sort(key=lambda item: item[0], reverse=True)
    remaining = max(config.max_components - frontier_k, 0)
    if remaining == 0:
        return (
            np.zeros(0, dtype=DTYPE_NP),
            np.zeros(0, dtype=DTYPE_NP),
            np.zeros(0, dtype=DTYPE_NP),
            np.zeros(0, dtype=DTYPE_NP),
            np.zeros(0, dtype=DTYPE_NP),
        )
    selected = candidate_rows[:remaining]
    weights = np.asarray([row[1] for row in selected], dtype=DTYPE_NP)
    means = np.asarray([row[2] for row in selected], dtype=DTYPE_NP)
    stds = np.asarray([row[3] for row in selected], dtype=DTYPE_NP)
    lefts = np.asarray([row[4] for row in selected], dtype=DTYPE_NP)
    rights = np.asarray([row[5] for row in selected], dtype=DTYPE_NP)
    if frontier_k == 0:
        lefts[:] = lower_bound
        rights[:] = upper_bound
    return weights, means, stds, lefts, rights


def _build_stage_prefix_batch(
    prepared: _PreparedGridBatch,
    *,
    frontier_k: np.ndarray,
    frontier_weights: np.ndarray,
    frontier_means: np.ndarray,
    frontier_stds: np.ndarray,
    frontier_lefts: np.ndarray,
    frontier_rights: np.ndarray,
    stage_components: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    max_components: int,
) -> _StagePrefixBatch | None:
    if not stage_components:
        return None

    batch_probabilities: list[np.ndarray] = []
    batch_bin_edges: list[np.ndarray] = []
    batch_support_mask: list[np.ndarray] = []
    batch_lower_bounds: list[float] = []
    batch_upper_bounds: list[float] = []
    batch_selected_k: list[int] = []
    batch_weights: list[np.ndarray] = []
    batch_means: list[np.ndarray] = []
    batch_stds: list[np.ndarray] = []
    batch_lefts: list[np.ndarray] = []
    batch_rights: list[np.ndarray] = []
    row_gene_indices: list[int] = []
    row_target_k: list[int] = []

    for gene_idx, (new_weights, new_means, new_stds, new_lefts, new_rights) in stage_components.items():
        start_k = int(frontier_k[gene_idx])
        for offset in range(1, new_weights.shape[0] + 1):
            target_k = start_k + offset
            weights = np.zeros(max_components, dtype=DTYPE_NP)
            means = np.zeros(max_components, dtype=DTYPE_NP)
            stds = np.ones(max_components, dtype=DTYPE_NP)
            lefts = np.zeros(max_components, dtype=DTYPE_NP)
            rights = np.zeros(max_components, dtype=DTYPE_NP)
            if start_k > 0:
                weights[:start_k] = frontier_weights[gene_idx, :start_k]
                means[:start_k] = frontier_means[gene_idx, :start_k]
                stds[:start_k] = frontier_stds[gene_idx, :start_k]
                lefts[:start_k] = frontier_lefts[gene_idx, :start_k]
                rights[:start_k] = frontier_rights[gene_idx, :start_k]
            weights[start_k:target_k] = new_weights[:offset]
            means[start_k:target_k] = new_means[:offset]
            stds[start_k:target_k] = new_stds[:offset]
            lefts[start_k:target_k] = new_lefts[:offset]
            rights[start_k:target_k] = new_rights[:offset]
            active_total = max(float(np.sum(weights[:target_k])), EPS)
            weights[:target_k] = weights[:target_k] / active_total
            batch_probabilities.append(prepared.probabilities[gene_idx].copy())
            batch_bin_edges.append(prepared.bin_edges[gene_idx].copy())
            batch_support_mask.append(prepared.support_mask[gene_idx].copy())
            batch_lower_bounds.append(float(prepared.lower_bounds[gene_idx]))
            batch_upper_bounds.append(float(prepared.upper_bounds[gene_idx]))
            batch_selected_k.append(target_k)
            batch_weights.append(weights)
            batch_means.append(means)
            batch_stds.append(stds)
            batch_lefts.append(lefts)
            batch_rights.append(rights)
            row_gene_indices.append(gene_idx)
            row_target_k.append(target_k)

    return _StagePrefixBatch(
        probabilities=np.asarray(batch_probabilities, dtype=DTYPE_NP),
        bin_edges=np.asarray(batch_bin_edges, dtype=DTYPE_NP),
        support_mask=np.asarray(batch_support_mask, dtype=bool),
        lower_bounds=np.asarray(batch_lower_bounds, dtype=DTYPE_NP),
        upper_bounds=np.asarray(batch_upper_bounds, dtype=DTYPE_NP),
        selected_k=np.asarray(batch_selected_k, dtype=np.int64),
        component_weights=np.asarray(batch_weights, dtype=DTYPE_NP),
        component_means=np.asarray(batch_means, dtype=DTYPE_NP),
        component_stds=np.asarray(batch_stds, dtype=DTYPE_NP),
        component_lefts=np.asarray(batch_lefts, dtype=DTYPE_NP),
        component_rights=np.asarray(batch_rights, dtype=DTYPE_NP),
        row_gene_indices=np.asarray(row_gene_indices, dtype=np.int64),
        row_target_k=np.asarray(row_target_k, dtype=np.int64),
    )


def _evaluate_prefix_batch(
    batch: _StagePrefixBatch,
    *,
    config: GMMSearchConfig,
    device: str | torch.device,
    torch_dtype: str,
) -> MixtureOptimizationResult:
    if config.search_refit_enabled:
        max_iterations = config.search_refit_max_iterations
        if np.any(batch.selected_k == 1):
            max_iterations = max(
                max_iterations,
                config.search_refit_min_iterations_first_component,
            )
        return optimize_mixture_parameters(
            probabilities=batch.probabilities,
            bin_edges=batch.bin_edges,
            support_mask=batch.support_mask,
            lower_bounds=batch.lower_bounds,
            upper_bounds=batch.upper_bounds,
            selected_k=batch.selected_k,
            initial_weights=batch.component_weights,
            initial_means=batch.component_means,
            initial_stds=batch.component_stds,
            initial_left_truncations=batch.component_lefts,
            initial_right_truncations=batch.component_rights,
            max_iterations=max_iterations,
            learning_rate=config.search_refit_learning_rate,
            convergence_tolerance=config.search_refit_convergence_tolerance,
            overshoot_penalty=config.search_refit_overshoot_penalty,
            mean_margin_fraction=config.search_refit_mean_margin_fraction,
            torch_dtype=torch_dtype,
            device=device,
            compile_model=config.search_refit_compile_model,
            optimize_weights=config.search_refit_optimize_weights,
            optimize_means=config.search_refit_optimize_means,
            optimize_stds=config.search_refit_optimize_stds,
            optimize_left_truncations=config.search_refit_optimize_left_truncations,
            optimize_right_truncations=config.search_refit_optimize_right_truncations,
            sigma_floor_fraction=config.search_refit_sigma_floor_fraction,
            min_window_fraction=config.search_refit_min_window_fraction,
            initial_weight_logit_floor=config.search_refit_initial_weight_logit_floor,
            inactive_weight_floor=config.search_refit_inactive_weight_floor,
            masked_logit_value=config.search_refit_masked_logit_value,
            logit_clip=config.search_refit_logit_clip,
            inverse_softplus_clip=config.search_refit_inverse_softplus_clip,
        )
    return evaluate_mixture_parameters(
        probabilities=batch.probabilities,
        bin_edges=batch.bin_edges,
        support_mask=batch.support_mask,
        lower_bounds=batch.lower_bounds,
        upper_bounds=batch.upper_bounds,
        selected_k=batch.selected_k,
        component_weights=batch.component_weights,
        component_means=batch.component_means,
        component_stds=batch.component_stds,
        component_left_truncations=batch.component_lefts,
        component_right_truncations=batch.component_rights,
        torch_dtype=torch_dtype,
        device=device,
    )


def _select_component_count(
    error_path: np.ndarray,
    explored_k: np.ndarray,
    *,
    error_threshold: float,
) -> np.ndarray:
    selected = np.ones(error_path.shape[0], dtype=np.int64)
    for row_idx in range(error_path.shape[0]):
        max_k = int(explored_k[row_idx])
        row_errors = error_path[row_idx, :max_k]
        within = np.flatnonzero(row_errors <= error_threshold)
        if within.size > 0:
            selected[row_idx] = int(within[0] + 1)
            continue
        selected[row_idx] = int(np.argmin(row_errors) + 1)
    return selected


def _finalize_search(
    prepared: _PreparedGridBatch,
    *,
    selected_k: np.ndarray,
    explored_k: np.ndarray,
    weight_history: np.ndarray,
    mean_history: np.ndarray,
    std_history: np.ndarray,
    left_history: np.ndarray,
    right_history: np.ndarray,
    fitted_history: np.ndarray,
    error_path: np.ndarray,
    residual_mass_path: np.ndarray,
    residual_peak_path: np.ndarray,
    config: GMMSearchConfig,
) -> DistributionGMMSearch:
    component_weights = np.zeros((prepared.n_genes, config.max_components), dtype=DTYPE_NP)
    component_means = np.zeros_like(component_weights)
    component_stds = np.ones_like(component_weights)
    component_lefts = np.zeros_like(component_weights)
    component_rights = np.zeros_like(component_weights)
    greedy_probabilities = np.zeros_like(prepared.probabilities)
    for row_idx, k in enumerate(selected_k.tolist()):
        step_idx = k - 1
        component_weights[row_idx] = weight_history[row_idx, step_idx]
        component_means[row_idx] = mean_history[row_idx, step_idx]
        component_stds[row_idx] = std_history[row_idx, step_idx]
        component_lefts[row_idx] = left_history[row_idx, step_idx]
        component_rights[row_idx] = right_history[row_idx, step_idx]
        greedy_probabilities[row_idx] = fitted_history[row_idx, :, step_idx]
        if k < config.max_components:
            component_weights[row_idx, k:] = 0.0
    return DistributionGMMSearch(
        support_domain=prepared.support_domain,
        support=prepared.support,
        support_mask=prepared.support_mask,
        probabilities=prepared.probabilities,
        bin_edges=prepared.bin_edges,
        lower_bounds=prepared.lower_bounds,
        upper_bounds=prepared.upper_bounds,
        selected_k=selected_k,
        component_weights=component_weights,
        component_means=component_means,
        component_stds=component_stds,
        component_left_truncations=component_lefts,
        component_right_truncations=component_rights,
        greedy_probabilities=greedy_probabilities,
        error_path=error_path,
        residual_mass_path=residual_mass_path,
        residual_peak_path=residual_peak_path,
        explored_k=explored_k,
        config=asdict(config),
    )


def _search_prepared_batch(
    prepared: _PreparedGridBatch,
    *,
    config: GMMSearchConfig,
    device: str | torch.device,
    torch_dtype: str,
) -> DistributionGMMSearch:
    n_genes = prepared.n_genes
    max_components = config.max_components
    explored_k = np.zeros(n_genes, dtype=np.int64)
    frontier_k = np.zeros(n_genes, dtype=np.int64)
    done = np.zeros(n_genes, dtype=bool)
    frontier_weights = np.zeros((n_genes, max_components), dtype=DTYPE_NP)
    frontier_means = np.zeros((n_genes, max_components), dtype=DTYPE_NP)
    frontier_stds = np.ones((n_genes, max_components), dtype=DTYPE_NP)
    frontier_lefts = np.repeat(prepared.lower_bounds[:, None], max_components, axis=1)
    frontier_rights = np.repeat(prepared.upper_bounds[:, None], max_components, axis=1)
    frontier_fitted = np.zeros_like(prepared.probabilities)

    weight_history = np.zeros((n_genes, max_components, max_components), dtype=DTYPE_NP)
    mean_history = np.zeros_like(weight_history)
    std_history = np.ones_like(weight_history)
    left_history = np.repeat(
        prepared.lower_bounds[:, None, None],
        max_components,
        axis=1,
    )
    left_history = np.repeat(left_history, max_components, axis=2).astype(DTYPE_NP)
    right_history = np.repeat(
        prepared.upper_bounds[:, None, None],
        max_components,
        axis=1,
    )
    right_history = np.repeat(right_history, max_components, axis=2).astype(DTYPE_NP)
    fitted_history = np.zeros((n_genes, prepared.n_support_points, max_components), dtype=DTYPE_NP)
    error_path = np.full((n_genes, max_components), np.inf, dtype=DTYPE_NP)
    residual_mass_path = np.full((n_genes, max_components), np.inf, dtype=DTYPE_NP)
    residual_peak_path = np.full((n_genes, max_components), np.inf, dtype=DTYPE_NP)

    while np.any(~done):
        stage_components: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for row_idx in range(n_genes):
            if done[row_idx]:
                continue
            if frontier_k[row_idx] >= max_components:
                done[row_idx] = True
                continue
            residual = np.clip(
                prepared.probabilities[row_idx] - frontier_fitted[row_idx],
                0.0,
                None,
            )
            if frontier_k[row_idx] > 0:
                if (
                    residual.sum() <= config.residual_mass_threshold
                    or residual.max() <= config.residual_peak_threshold
                ):
                    done[row_idx] = True
                    continue
            extracted = _extract_stage_components_for_gene(
                prepared,
                row_idx=row_idx,
                residual=residual,
                frontier_k=int(frontier_k[row_idx]),
                config=config,
                device=device,
                torch_dtype=torch_dtype,
            )
            if extracted[0].size == 0:
                done[row_idx] = True
                continue
            stage_components[row_idx] = extracted

        batch = _build_stage_prefix_batch(
            prepared,
            frontier_k=frontier_k,
            frontier_weights=frontier_weights,
            frontier_means=frontier_means,
            frontier_stds=frontier_stds,
            frontier_lefts=frontier_lefts,
            frontier_rights=frontier_rights,
            stage_components=stage_components,
            max_components=max_components,
        )
        if batch is None:
            break
        result = _evaluate_prefix_batch(
            batch,
            config=config,
            device=device,
            torch_dtype=torch_dtype,
        )
        for batch_row, (gene_idx, target_k) in enumerate(
            zip(batch.row_gene_indices.tolist(), batch.row_target_k.tolist(), strict=True)
        ):
            step_idx = target_k - 1
            weight_history[gene_idx, step_idx] = result.component_weights[batch_row]
            mean_history[gene_idx, step_idx] = result.component_means[batch_row]
            std_history[gene_idx, step_idx] = result.component_stds[batch_row]
            left_history[gene_idx, step_idx] = result.component_left_truncations[batch_row]
            right_history[gene_idx, step_idx] = result.component_right_truncations[batch_row]
            fitted_history[gene_idx, :, step_idx] = result.fitted_probabilities[batch_row]
            error_path[gene_idx, step_idx] = result.jsd[batch_row]
            residual = np.clip(
                prepared.probabilities[gene_idx] - result.fitted_probabilities[batch_row],
                0.0,
                None,
            )
            residual_mass_path[gene_idx, step_idx] = residual.sum()
            residual_peak_path[gene_idx, step_idx] = residual.max()
            explored_k[gene_idx] = max(explored_k[gene_idx], target_k)

        for gene_idx, extracted in stage_components.items():
            new_frontier_k = int(frontier_k[gene_idx] + extracted[0].shape[0])
            step_idx = new_frontier_k - 1
            frontier_k[gene_idx] = new_frontier_k
            frontier_weights[gene_idx] = weight_history[gene_idx, step_idx]
            frontier_means[gene_idx] = mean_history[gene_idx, step_idx]
            frontier_stds[gene_idx] = std_history[gene_idx, step_idx]
            frontier_lefts[gene_idx] = left_history[gene_idx, step_idx]
            frontier_rights[gene_idx] = right_history[gene_idx, step_idx]
            frontier_fitted[gene_idx] = fitted_history[gene_idx, :, step_idx]
            if (
                residual_mass_path[gene_idx, step_idx] <= config.residual_mass_threshold
                or residual_peak_path[gene_idx, step_idx] <= config.residual_peak_threshold
                or frontier_k[gene_idx] >= max_components
            ):
                done[gene_idx] = True

    if np.any(explored_k == 0):
        raise RuntimeError("search failed to discover an initial mixture for one or more rows")

    for row_idx in range(n_genes):
        last_idx = int(explored_k[row_idx] - 1)
        if last_idx + 1 < max_components:
            error_path[row_idx, last_idx + 1 :] = error_path[row_idx, last_idx]
            residual_mass_path[row_idx, last_idx + 1 :] = residual_mass_path[row_idx, last_idx]
            residual_peak_path[row_idx, last_idx + 1 :] = residual_peak_path[row_idx, last_idx]

    selected_k = _select_component_count(
        error_path,
        explored_k,
        error_threshold=config.error_threshold,
    )
    return _finalize_search(
        prepared,
        selected_k=selected_k,
        explored_k=explored_k,
        weight_history=weight_history,
        mean_history=mean_history,
        std_history=std_history,
        left_history=left_history,
        right_history=right_history,
        fitted_history=fitted_history,
        error_path=error_path,
        residual_mass_path=residual_mass_path,
        residual_peak_path=residual_peak_path,
        config=config,
    )


def search_distribution_gmm(
    distribution: DistributionGrid,
    *,
    config: GMMSearchConfig = GMMSearchConfig(),
    device: str | torch.device = "cpu",
    torch_dtype: str = "float64",
) -> DistributionGMMSearch:
    prepared = _prepare_distribution_grid(distribution, config=config)
    return _search_prepared_batch(
        prepared,
        config=config,
        device=device,
        torch_dtype=torch_dtype,
    )


def search_prior_gmm(
    prior: PriorGrid,
    *,
    config: GMMSearchConfig = GMMSearchConfig(),
    device: str | torch.device = "cpu",
    torch_dtype: str = "float64",
    support_axis: SupportAxis = "raw",
) -> DistributionGMMSearch:
    prepared = _prepare_prior_grid(
        prior,
        support_axis=support_axis,
        config=config,
    )
    return _search_prepared_batch(
        prepared,
        config=config,
        device=device,
        torch_dtype=torch_dtype,
    )


__all__ = ["search_distribution_gmm", "search_prior_gmm"]
