from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from prism.gmm import (
    DistributionGMMSearch,
    GMMSearchConfig,
    GMMTrainingConfig,
    fit_distribution_gmm,
    fit_prior_gmm,
    search_distribution_gmm,
)
from prism.gmm.numeric import (
    truncated_gaussian_bin_masses,
    truncated_gaussian_bin_masses_dense_1d,
)
from prism.gmm.optimize import evaluate_mixture_parameters, optimize_mixture_parameters
from prism.gmm.search import (
    _masked_quantile_columns,
    _peak_indices,
    _select_component_count,
    _select_frontier_target_k,
)
from prism.model import PriorGrid, make_distribution_grid


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(array / math.sqrt(2.0)))


def _build_edges(support: np.ndarray, *, probability_domain: bool) -> np.ndarray:
    values = np.asarray(support, dtype=np.float64).reshape(-1)
    if values.size == 1:
        return np.asarray([0.0, 1.0] if probability_domain else [0.0, 1.0])
    midpoints = 0.5 * (values[:-1] + values[1:])
    left = values[0] - 0.5 * (values[1] - values[0])
    right = values[-1] + 0.5 * (values[-1] - values[-2])
    edges = np.concatenate([[left], midpoints, [right]])
    if probability_domain:
        edges = np.clip(edges, 0.0, 1.0)
    else:
        edges = np.clip(edges, 0.0, None)
    edges = np.maximum.accumulate(edges)
    for idx in range(1, edges.shape[0]):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1e-12
    return edges


def _truncated_gaussian_grid(
    support: np.ndarray,
    components: list[tuple[float, float, float, float, float]],
    *,
    probability_domain: bool,
) -> np.ndarray:
    edges = _build_edges(support, probability_domain=probability_domain)
    total = np.zeros_like(support, dtype=np.float64)
    for weight, mean, std, left, right in components:
        clipped_left = max(left, float(edges[0]))
        clipped_right = min(right, float(edges[-1]))
        denom = float(
            _normal_cdf(np.asarray([(clipped_right - mean) / std]))[0]
            - _normal_cdf(np.asarray([(clipped_left - mean) / std]))[0]
        )
        denom = max(denom, 1e-12)
        lo = np.maximum(edges[:-1], clipped_left)
        hi = np.minimum(edges[1:], clipped_right)
        bin_mass = _normal_cdf((hi - mean) / std) - _normal_cdf((lo - mean) / std)
        bin_mass = np.where(hi > lo, bin_mass / denom, 0.0)
        total += float(weight) * bin_mass
    total = np.clip(total, 0.0, None)
    return total / max(float(np.sum(total)), 1e-12)


def _dataclass_kwargs(instance: object) -> dict[str, object]:
    return {
        field.name: getattr(instance, field.name)
        for field in dataclasses.fields(instance)
    }


def test_search_distribution_gmm_prefers_one_component_for_single_peak() -> None:
    support = np.linspace(0.0, 1.0, 49, dtype=np.float64)
    probabilities = _truncated_gaussian_grid(
        support,
        [(1.0, 0.35, 0.07, 0.0, 1.0)],
        probability_domain=True,
    )
    distribution = make_distribution_grid(
        "binomial",
        support=support,
        probabilities=probabilities,
    )
    search = search_distribution_gmm(
        distribution,
        config=GMMSearchConfig(
            max_components=4,
            error_threshold=5e-3,
            peak_limit_per_stage=4,
            candidate_window_count=4,
            candidate_sigma_count=5,
            search_refit_max_iterations=20,
        ),
        torch_dtype="float64",
    )
    assert search.selected_k.tolist() == [1]
    assert search.explored_k.tolist() == [1]
    assert np.isclose(np.sum(search.greedy_probabilities[0]), 1.0)
    assert search.error_path[0, 0] < 0.03


def test_search_distribution_gmm_can_disable_search_refit() -> None:
    support = np.linspace(0.0, 1.0, 65, dtype=np.float64)
    probabilities = _truncated_gaussian_grid(
        support,
        [
            (0.4, 0.2, 0.06, 0.0, 1.0),
            (0.6, 0.75, 0.07, 0.0, 1.0),
        ],
        probability_domain=True,
    )
    distribution = make_distribution_grid(
        "binomial",
        support=support,
        probabilities=probabilities,
    )
    search = search_distribution_gmm(
        distribution,
        config=GMMSearchConfig(
            max_components=4,
            error_threshold=2e-2,
            peak_limit_per_stage=None,
            candidate_window_count=4,
            candidate_sigma_count=5,
            search_refit_enabled=False,
        ),
        torch_dtype="float64",
    )
    assert search.config["search_refit_enabled"] is False
    assert search.explored_k[0] >= 2
    assert search.selected_k[0] >= 2
    assert np.isclose(np.sum(search.greedy_probabilities[0]), 1.0)
    assert search.error_path[0, search.selected_k[0] - 1] < 0.12


def test_search_frontier_strategy_prefers_best_prefix() -> None:
    target_k = np.asarray([1, 2, 3], dtype=np.int64)
    metric_values = np.asarray([0.08, 0.03, 0.05], dtype=np.float64)
    assert (
        _select_frontier_target_k(
            frontier_k=0,
            stage_target_k=target_k,
            stage_metric_values=metric_values,
            strategy="best_prefix",
        )
        == 2
    )
    assert (
        _select_frontier_target_k(
            frontier_k=0,
            stage_target_k=target_k,
            stage_metric_values=metric_values,
            strategy="single_step",
        )
        == 1
    )
    assert (
        _select_frontier_target_k(
            frontier_k=0,
            stage_target_k=target_k,
            stage_metric_values=metric_values,
            strategy="full_stage",
        )
        == 3
    )


def test_search_peak_indices_collapse_plateau() -> None:
    peaks = _peak_indices(
        np.asarray([0.0, 0.3, 0.3, 0.3, 0.1], dtype=np.float64),
        config=GMMSearchConfig(
            peak_plateau_tolerance=1e-12,
            peak_limit_per_stage=None,
        ),
    )
    assert peaks.tolist() == [2]


def test_masked_quantile_columns_matches_columnwise_quantile() -> None:
    values = np.asarray(
        [
            [1.0, 5.0, 2.0],
            [3.0, 7.0, 4.0],
            [2.0, 6.0, 8.0],
            [9.0, 1.0, 6.0],
        ],
        dtype=np.float64,
    )
    mask = np.asarray(
        [
            [True, False, True],
            [True, True, False],
            [False, True, True],
            [True, True, True],
        ],
        dtype=bool,
    )
    expected = np.asarray(
        [
            np.quantile(values[mask[:, col_idx], col_idx], 0.25)
            for col_idx in range(values.shape[1])
        ],
        dtype=np.float64,
    )
    resolved = _masked_quantile_columns(values, mask, quantile=0.25)
    assert np.allclose(resolved, expected)


def test_truncated_gaussian_dense_1d_matches_generic_kernel() -> None:
    bin_edges = np.asarray([0.0, 0.2, 0.5, 1.0], dtype=np.float64)
    means = np.asarray([0.25, 0.7], dtype=np.float64)
    stds = np.asarray([0.08, 0.12], dtype=np.float64)
    lefts = np.asarray([0.0, 0.15], dtype=np.float64)
    rights = np.asarray([1.0, 1.0], dtype=np.float64)
    generic = truncated_gaussian_bin_masses(
        torch.as_tensor(bin_edges[None, :], dtype=torch.float64),
        torch.as_tensor(means[None, :], dtype=torch.float64),
        torch.as_tensor(stds[None, :], dtype=torch.float64),
        torch.as_tensor(lefts[None, :], dtype=torch.float64),
        torch.as_tensor(rights[None, :], dtype=torch.float64),
        torch.ones((1, bin_edges.shape[0] - 1), dtype=torch.bool),
    )[0]
    dense = truncated_gaussian_bin_masses_dense_1d(
        torch.as_tensor(bin_edges, dtype=torch.float64),
        torch.as_tensor(means, dtype=torch.float64),
        torch.as_tensor(stds, dtype=torch.float64),
        torch.as_tensor(lefts, dtype=torch.float64),
        torch.as_tensor(rights, dtype=torch.float64),
    )
    assert np.allclose(dense.numpy(), generic.numpy())


def test_select_component_count_supports_marginal_gain_and_penalized_error() -> None:
    metric_path = np.asarray([[0.3, 0.12, 0.11, 0.109]], dtype=np.float64)
    explored_k = np.asarray([4], dtype=np.int64)
    marginal = _select_component_count(
        metric_path,
        explored_k,
        error_threshold=0.0,
        mode="marginal_gain",
        min_improvement=0.02,
        min_improvement_patience=2,
        penalty_weight=0.0,
    )
    penalized = _select_component_count(
        np.asarray([[0.10, 0.09, 0.085]], dtype=np.float64),
        np.asarray([3], dtype=np.int64),
        error_threshold=0.0,
        mode="penalized_error",
        min_improvement=0.0,
        min_improvement_patience=1,
        penalty_weight=0.01,
    )
    assert marginal.tolist() == [2]
    assert penalized.tolist() == [1]


def test_training_config_validates_pruning_metrics() -> None:
    with pytest.raises(ValueError, match="pruning_error_metric"):
        GMMTrainingConfig(pruning_error_metric="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pruning_significance_metric"):
        GMMTrainingConfig(pruning_significance_metric="bad")  # type: ignore[arg-type]


def test_compile_policy_supports_legacy_bool_and_rejects_conflicts() -> None:
    config = GMMTrainingConfig(compile_model=True)
    assert config.compile_policy == "always"
    assert config.compile_model is True

    search_config = GMMSearchConfig(search_refit_compile_policy="auto")
    assert search_config.search_refit_compile_policy == "auto"
    assert search_config.search_refit_compile_model is None

    with pytest.raises(ValueError, match="compile_model conflicts"):
        GMMTrainingConfig(
            compile_policy="auto",
            compile_model=True,
        )
    with pytest.raises(ValueError, match="search_refit_compile_model conflicts"):
        GMMSearchConfig(
            search_refit_compile_policy="auto",
            search_refit_compile_model=True,
        )


def test_distribution_gmm_search_rejects_non_prefix_mask_and_masked_probability() -> None:
    support = np.asarray(
        [
            [0.0, 0.25, 0.5, 0.75],
            [0.0, 0.5, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    probabilities = np.asarray(
        [
            [0.2, 0.3, 0.3, 0.2],
            [0.25, 0.25, 0.2, 0.3],
        ],
        dtype=np.float64,
    )
    distribution = make_distribution_grid(
        "binomial",
        support=support,
        probabilities=probabilities,
    )
    search = search_distribution_gmm(
        distribution,
        config=GMMSearchConfig(
            max_components=3,
            peak_limit_per_stage=3,
            candidate_window_count=3,
            candidate_sigma_count=3,
            search_refit_enabled=False,
        ),
        torch_dtype="float64",
    )
    kwargs = _dataclass_kwargs(search)

    bad_mask = np.asarray(search.support_mask, dtype=bool).copy()
    bad_mask[0, 1] = False
    bad_mask[0, 2] = True
    kwargs["support_mask"] = bad_mask
    with pytest.raises(ValueError, match="prefix-contiguous"):
        type(search)(**kwargs)

    kwargs = _dataclass_kwargs(search)
    bad_probabilities = np.asarray(search.probabilities, dtype=np.float64).copy()
    masked_idx = int(np.flatnonzero(~search.support_mask[1])[0])
    bad_probabilities[1, masked_idx] = 1e-3
    bad_probabilities[1, 0] -= 1e-3
    kwargs["probabilities"] = bad_probabilities
    with pytest.raises(ValueError, match="masked-out probabilities"):
        type(search)(**kwargs)


def test_fit_distribution_gmm_rejects_search_with_mismatched_bin_edges() -> None:
    support = np.linspace(0.0, 1.0, 17, dtype=np.float64)
    probabilities = _truncated_gaussian_grid(
        support,
        [(1.0, 0.45, 0.08, 0.0, 1.0)],
        probability_domain=True,
    )
    distribution = make_distribution_grid(
        "binomial",
        support=support,
        probabilities=probabilities,
    )
    search = search_distribution_gmm(
        distribution,
        config=GMMSearchConfig(
            max_components=3,
            peak_limit_per_stage=3,
            candidate_window_count=3,
            candidate_sigma_count=3,
            search_refit_enabled=False,
        ),
        torch_dtype="float64",
    )
    kwargs = _dataclass_kwargs(search)
    bad_bin_edges = np.asarray(search.bin_edges, dtype=np.float64).copy()
    bad_lower_bounds = np.asarray(search.lower_bounds, dtype=np.float64).copy()
    bad_bin_edges[0, 0] -= 0.05
    bad_lower_bounds[0] = bad_bin_edges[0, 0]
    kwargs["bin_edges"] = bad_bin_edges
    kwargs["lower_bounds"] = bad_lower_bounds
    mismatched_search = type(search)(**kwargs)
    with pytest.raises(ValueError, match="bin_edges"):
        fit_distribution_gmm(distribution, search=mismatched_search)


def test_distribution_report_validates_residual_consistency_and_freezes_config() -> None:
    support = np.linspace(0.0, 1.0, 33, dtype=np.float64)
    probabilities = _truncated_gaussian_grid(
        support,
        [(1.0, 0.5, 0.1, 0.0, 1.0)],
        probability_domain=True,
    )
    distribution = make_distribution_grid(
        "binomial",
        support=support,
        probabilities=probabilities,
    )
    report = fit_distribution_gmm(
        distribution,
        search_config=GMMSearchConfig(
            max_components=3,
            peak_limit_per_stage=3,
            candidate_window_count=3,
            candidate_sigma_count=3,
            search_refit_enabled=False,
        ),
        training_config=GMMTrainingConfig(
            max_iterations=5,
            gene_chunk_size=1,
            compile_model=False,
        ),
    )
    with pytest.raises(TypeError):
        report.config["mutated"] = True  # type: ignore[index]

    kwargs = _dataclass_kwargs(report)
    bad_residual = np.asarray(report.residual_probabilities, dtype=np.float64).copy()
    bad_residual[0, 0] += 1e-3
    kwargs["residual_probabilities"] = bad_residual
    with pytest.raises(ValueError, match="fitted_probabilities plus residual_probabilities"):
        type(report)(**kwargs)


def test_evaluate_and_optimize_do_not_leak_workspace_state_between_calls() -> None:
    support = np.linspace(0.0, 1.0, 25, dtype=np.float64)
    bin_edges = _build_edges(support, probability_domain=True)[None, :]
    support_mask = np.ones((1, support.shape[0]), dtype=bool)
    lower_bounds = np.asarray([bin_edges[0, 0]], dtype=np.float64)
    upper_bounds = np.asarray([bin_edges[0, -1]], dtype=np.float64)
    selected_k = np.asarray([1], dtype=np.int64)
    component_weights = np.asarray([[1.0]], dtype=np.float64)
    component_stds = np.asarray([[0.08]], dtype=np.float64)
    component_lefts = np.asarray([[0.0]], dtype=np.float64)
    component_rights = np.asarray([[1.0]], dtype=np.float64)
    probabilities_first = _truncated_gaussian_grid(
        support,
        [(1.0, 0.25, 0.08, 0.0, 1.0)],
        probability_domain=True,
    )[None, :]
    probabilities_second = _truncated_gaussian_grid(
        support,
        [(1.0, 0.75, 0.08, 0.0, 1.0)],
        probability_domain=True,
    )[None, :]

    first = evaluate_mixture_parameters(
        probabilities=probabilities_first,
        bin_edges=bin_edges,
        support_mask=support_mask,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        selected_k=selected_k,
        component_weights=component_weights,
        component_means=np.asarray([[0.25]], dtype=np.float64),
        component_stds=component_stds,
        component_left_truncations=component_lefts,
        component_right_truncations=component_rights,
        torch_dtype="float64",
    )
    second = evaluate_mixture_parameters(
        probabilities=probabilities_second,
        bin_edges=bin_edges,
        support_mask=support_mask,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        selected_k=selected_k,
        component_weights=component_weights,
        component_means=np.asarray([[0.75]], dtype=np.float64),
        component_stds=component_stds,
        component_left_truncations=component_lefts,
        component_right_truncations=component_rights,
        torch_dtype="float64",
    )
    optimized = optimize_mixture_parameters(
        probabilities=probabilities_second,
        bin_edges=bin_edges,
        support_mask=support_mask,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        selected_k=selected_k,
        initial_weights=component_weights,
        initial_means=np.asarray([[0.75]], dtype=np.float64),
        initial_stds=component_stds,
        initial_left_truncations=component_lefts,
        initial_right_truncations=component_rights,
        max_iterations=0,
        learning_rate=0.05,
        convergence_tolerance=1e-7,
        overshoot_penalty=0.0,
        mean_margin_fraction=0.5,
        torch_dtype="float64",
    )

    assert not np.allclose(first.fitted_probabilities, second.fitted_probabilities)
    assert np.allclose(second.fitted_probabilities, probabilities_second, atol=1e-6)
    assert np.allclose(optimized.fitted_probabilities, second.fitted_probabilities)
    assert np.allclose(optimized.component_means, np.asarray([[0.75]], dtype=np.float64))


def test_search_distribution_gmm_handles_close_peaks_with_shoulder() -> None:
    support = np.linspace(0.0, 1.0, 81, dtype=np.float64)
    probabilities = _truncated_gaussian_grid(
        support,
        [
            (0.52, 0.42, 0.07, 0.0, 1.0),
            (0.48, 0.58, 0.07, 0.0, 1.0),
        ],
        probability_domain=True,
    )
    distribution = make_distribution_grid(
        "binomial",
        support=support,
        probabilities=probabilities,
    )
    search = search_distribution_gmm(
        distribution,
        config=GMMSearchConfig(
            max_components=4,
            error_threshold=2e-2,
            peak_limit_per_stage=None,
            candidate_window_count=5,
            candidate_sigma_count=6,
            candidate_rerank_top_n=4,
            frontier_update_strategy="best_prefix",
            selection_metric="jsd",
            search_refit_enabled=False,
        ),
        torch_dtype="float64",
    )
    assert search.selected_k[0] >= 2
    assert search.error_path[0, search.selected_k[0] - 1] < 0.08


def test_search_distribution_gmm_handles_boundary_peak_with_weak_interior_peak() -> None:
    support = np.linspace(0.0, 1.0, 81, dtype=np.float64)
    probabilities = _truncated_gaussian_grid(
        support,
        [
            (0.8, -0.04, 0.07, 0.0, 1.0),
            (0.2, 0.68, 0.05, 0.0, 1.0),
        ],
        probability_domain=True,
    )
    distribution = make_distribution_grid(
        "binomial",
        support=support,
        probabilities=probabilities,
    )
    search = search_distribution_gmm(
        distribution,
        config=GMMSearchConfig(
            max_components=4,
            error_threshold=2e-2,
            peak_limit_per_stage=None,
            candidate_window_count=5,
            candidate_sigma_count=6,
            candidate_rerank_top_n=4,
            candidate_alpha_strategy="least_squares",
            search_refit_enabled=False,
        ),
        torch_dtype="float64",
    )
    assert search.selected_k[0] >= 2
    assert search.error_path[0, search.selected_k[0] - 1] < 0.08


def test_search_distribution_gmm_prefers_single_component_for_broad_single_mode() -> None:
    support = np.linspace(0.0, 1.0, 81, dtype=np.float64)
    probabilities = _truncated_gaussian_grid(
        support,
        [(1.0, 0.5, 0.22, 0.0, 1.0)],
        probability_domain=True,
    )
    distribution = make_distribution_grid(
        "binomial",
        support=support,
        probabilities=probabilities,
    )
    search = search_distribution_gmm(
        distribution,
        config=GMMSearchConfig(
            max_components=4,
            error_threshold=2e-2,
            peak_limit_per_stage=None,
            candidate_window_count=5,
            candidate_sigma_count=6,
            candidate_alpha_strategy="least_squares",
            search_refit_enabled=False,
        ),
        torch_dtype="float64",
    )
    assert search.selected_k.tolist() == [1]
    assert search.error_path[0, 0] < 0.08


def test_fit_distribution_gmm_handles_boundary_peak() -> None:
    support = np.linspace(0.0, 1.0, 65, dtype=np.float64)
    probabilities = _truncated_gaussian_grid(
        support,
        [(1.0, -0.05, 0.08, 0.0, 1.0)],
        probability_domain=True,
    )
    distribution = make_distribution_grid(
        "binomial",
        support=support,
        probabilities=probabilities,
    )
    report = fit_distribution_gmm(
        distribution,
        search_config=GMMSearchConfig(
            max_components=3,
            error_threshold=5e-3,
            peak_limit_per_stage=3,
            candidate_window_count=4,
            candidate_sigma_count=5,
            search_refit_max_iterations=20,
        ),
        training_config=GMMTrainingConfig(
            max_iterations=120,
            learning_rate=0.05,
            convergence_tolerance=1e-7,
            gene_chunk_size=16,
            compile_model=False,
        ),
    )
    assert report.selected_k.tolist() == [1]
    assert report.jsd[0] < 0.03
    assert report.component_left_truncations[0, 0] <= 0.05
    assert np.isclose(np.sum(report.fitted_probabilities[0]), 1.0)


def test_fit_distribution_gmm_supports_parallel_gene_specific_grids() -> None:
    support = np.asarray(
        [0.0, 0.01, 0.03, 0.08, 0.14, 0.23, 0.35, 0.48, 0.63, 0.79, 1.0],
        dtype=np.float64,
    )
    probabilities_a = _truncated_gaussian_grid(
        support,
        [(1.0, 0.18, 0.06, 0.0, 1.0)],
        probability_domain=True,
    )
    probabilities_b = _truncated_gaussian_grid(
        support,
        [
            (0.42, 0.18, 0.05, 0.0, 1.0),
            (0.58, 0.74, 0.06, 0.0, 1.0),
        ],
        probability_domain=True,
    )
    distribution = make_distribution_grid(
        "binomial",
        support=np.vstack([support, support]),
        probabilities=np.vstack([probabilities_a, probabilities_b]),
    )
    report = fit_distribution_gmm(
        distribution,
        search_config=GMMSearchConfig(
            max_components=4,
            error_threshold=2e-2,
            peak_limit_per_stage=None,
            candidate_window_count=5,
            candidate_sigma_count=5,
            search_refit_max_iterations=20,
        ),
        training_config=GMMTrainingConfig(
            max_iterations=100,
            learning_rate=0.05,
            convergence_tolerance=1e-7,
            gene_chunk_size=2,
            compile_model=False,
        ),
    )
    assert (
        report.fitted_probabilities.shape
        == distribution.as_gene_specific().support.shape
    )
    assert np.allclose(np.sum(report.fitted_probabilities, axis=1), 1.0)
    assert report.selected_k[0] == 1
    assert report.selected_k[1] >= 2
    assert np.all(report.jsd < 0.08)


def test_fit_distribution_gmm_refit_chunking_supports_more_genes_than_chunk_size() -> None:
    support = np.asarray(
        [0.0, 0.05, 0.12, 0.2, 0.35, 0.55, 0.75, 1.0],
        dtype=np.float64,
    )
    probabilities = np.vstack(
        [
            _truncated_gaussian_grid(
                support,
                [(1.0, 0.2, 0.07, 0.0, 1.0)],
                probability_domain=True,
            ),
            _truncated_gaussian_grid(
                support,
                [
                    (0.45, 0.2, 0.06, 0.0, 1.0),
                    (0.55, 0.72, 0.07, 0.0, 1.0),
                ],
                probability_domain=True,
            ),
            _truncated_gaussian_grid(
                support,
                [(1.0, 0.78, 0.08, 0.0, 1.0)],
                probability_domain=True,
            ),
        ]
    )
    distribution = make_distribution_grid(
        "binomial",
        support=np.repeat(support[None, :], 3, axis=0),
        probabilities=probabilities,
    )
    report = fit_distribution_gmm(
        distribution,
        search_config=GMMSearchConfig(
            max_components=4,
            error_threshold=2e-2,
            peak_limit_per_stage=None,
            candidate_window_count=4,
            candidate_sigma_count=5,
            search_refit_enabled=False,
        ),
        training_config=GMMTrainingConfig(
            max_iterations=40,
            learning_rate=0.05,
            convergence_tolerance=1e-7,
            gene_chunk_size=2,
            compile_model=False,
        ),
    )
    assert report.fitted_probabilities.shape == probabilities.shape
    assert np.allclose(np.sum(report.fitted_probabilities, axis=1), 1.0)
    assert report.selected_k[0] == 1
    assert report.selected_k[1] >= 2
    assert report.selected_k[2] == 1
    assert np.all(np.isfinite(report.jsd))


def test_fit_distribution_gmm_accepts_unsorted_duplicate_support() -> None:
    distribution = make_distribution_grid(
        "binomial",
        support=np.asarray([0.5, 0.2, 0.5, 0.8], dtype=np.float64),
        probabilities=np.asarray([0.2, 0.3, 0.1, 0.4], dtype=np.float64),
    )
    report = fit_distribution_gmm(
        distribution,
        search_config=GMMSearchConfig(
            max_components=3,
            error_threshold=0.2,
            peak_limit_per_stage=3,
            candidate_window_count=3,
            candidate_sigma_count=3,
            search_refit_enabled=False,
        ),
        training_config=GMMTrainingConfig(
            max_iterations=10,
            learning_rate=0.05,
            convergence_tolerance=1e-7,
            gene_chunk_size=1,
            compile_model=False,
        ),
    )
    assert report.support.shape == (1, 3)
    assert np.allclose(report.support[0], np.asarray([0.2, 0.5, 0.8], dtype=np.float64))
    assert np.all(report.support_mask[0])
    assert np.allclose(
        report.probabilities[0],
        np.asarray([0.3, 0.3, 0.4], dtype=np.float64),
    )
    assert np.isclose(np.sum(report.fitted_probabilities[0]), 1.0)


def test_fit_distribution_gmm_optional_pruning_can_drop_weak_component() -> None:
    support = np.linspace(0.0, 1.0, 49, dtype=np.float64)
    probabilities = _truncated_gaussian_grid(
        support,
        [(1.0, 0.35, 0.07, 0.0, 1.0)],
        probability_domain=True,
    )
    distribution = make_distribution_grid(
        "binomial",
        support=support,
        probabilities=probabilities,
    )
    base_search = search_distribution_gmm(
        distribution,
        config=GMMSearchConfig(
            max_components=3,
            error_threshold=1e-2,
            peak_limit_per_stage=3,
            candidate_window_count=4,
            candidate_sigma_count=5,
            search_refit_enabled=False,
        ),
        torch_dtype="float64",
    )
    overselected_search = DistributionGMMSearch(
        support_domain=base_search.support_domain,
        support=base_search.support,
        support_mask=base_search.support_mask,
        probabilities=base_search.probabilities,
        bin_edges=base_search.bin_edges,
        lower_bounds=base_search.lower_bounds,
        upper_bounds=base_search.upper_bounds,
        selected_k=np.asarray([2], dtype=np.int64),
        component_weights=np.asarray([[0.55, 0.45, 0.0]], dtype=np.float64),
        component_means=np.asarray(
            [[0.2, 0.85, 0.0]],
            dtype=np.float64,
        ),
        component_stds=np.asarray(
            [[0.05, 0.05, 1.0]],
            dtype=np.float64,
        ),
        component_left_truncations=np.asarray(
            [[base_search.lower_bounds[0], base_search.lower_bounds[0], base_search.lower_bounds[0]]],
            dtype=np.float64,
        ),
        component_right_truncations=np.asarray(
            [[base_search.upper_bounds[0], base_search.upper_bounds[0], base_search.upper_bounds[0]]],
            dtype=np.float64,
        ),
        greedy_probabilities=base_search.greedy_probabilities,
        error_path=np.asarray([[0.02, 0.02, 0.02]], dtype=np.float64),
        residual_mass_path=np.asarray([[0.05, 0.05, 0.05]], dtype=np.float64),
        residual_peak_path=np.asarray([[0.02, 0.02, 0.02]], dtype=np.float64),
        explored_k=np.asarray([2], dtype=np.int64),
        config=dict(base_search.config),
    )
    report = fit_distribution_gmm(
        distribution,
        search=overselected_search,
        training_config=GMMTrainingConfig(
            max_iterations=80,
            learning_rate=0.05,
            convergence_tolerance=1e-7,
            gene_chunk_size=1,
            compile_model=False,
            optimize_weights=False,
            optimize_means=False,
            optimize_stds=False,
            optimize_left_truncations=False,
            optimize_right_truncations=False,
            pruning_enabled=True,
            pruning_error_threshold=0.0,
            pruning_max_refits=2,
            pruning_min_components=1,
            pruning_significance_metric="weight",
        ),
    )
    assert report.selected_k.tolist() == [1]
    recomputed = evaluate_mixture_parameters(
        probabilities=report.probabilities,
        bin_edges=report.bin_edges,
        support_mask=report.support_mask,
        lower_bounds=report.lower_bounds,
        upper_bounds=report.upper_bounds,
        selected_k=report.selected_k,
        component_weights=report.component_weights,
        component_means=report.component_means,
        component_stds=report.component_stds,
        component_left_truncations=report.component_left_truncations,
        component_right_truncations=report.component_right_truncations,
        torch_dtype="float64",
    )
    assert np.allclose(report.fitted_probabilities, recomputed.fitted_probabilities)
    assert np.allclose(report.residual_probabilities, report.probabilities - report.fitted_probabilities)
    assert np.allclose(report.greedy_probabilities, report.fitted_probabilities)
    assert np.allclose(report.jsd, recomputed.jsd)
    assert np.allclose(report.cross_entropy, recomputed.cross_entropy)
    assert np.allclose(report.l1_error, recomputed.l1_error)
    assert np.allclose(report.greedy_error, report.jsd)


def test_fit_distribution_gmm_supports_fixed_bounds_truncation_and_multistart() -> None:
    support = np.linspace(0.0, 1.0, 49, dtype=np.float64)
    probabilities = _truncated_gaussian_grid(
        support,
        [
            (0.45, 0.25, 0.07, 0.0, 1.0),
            (0.55, 0.72, 0.08, 0.0, 1.0),
        ],
        probability_domain=True,
    )
    distribution = make_distribution_grid(
        "binomial",
        support=support,
        probabilities=probabilities,
    )
    report = fit_distribution_gmm(
        distribution,
        search_config=GMMSearchConfig(
            max_components=4,
            error_threshold=2e-2,
            peak_limit_per_stage=None,
            candidate_window_count=4,
            candidate_sigma_count=5,
            search_refit_enabled=False,
        ),
        training_config=GMMTrainingConfig(
            max_iterations=40,
            learning_rate=0.05,
            convergence_tolerance=1e-7,
            gene_chunk_size=1,
            compile_model=False,
            truncation_mode="fixed_bounds",
            truncation_regularization_strength=0.1,
            multi_start_count=3,
            multi_start_trigger_threshold=0.0,
            multi_start_jitter_scale=0.05,
            multi_start_seed=7,
        ),
    )
    active = int(report.selected_k[0])
    assert np.allclose(
        report.component_left_truncations[0, :active],
        report.lower_bounds[0],
    )
    assert np.allclose(
        report.component_right_truncations[0, :active],
        report.upper_bounds[0],
    )
    assert np.isfinite(report.jsd[0])


def test_fit_prior_gmm_preserves_gene_names_and_support_bounds() -> None:
    support = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
    probabilities = _truncated_gaussian_grid(
        support,
        [(1.0, 0.5, 0.09, 0.0, 1.0)],
        probability_domain=True,
    )
    prior = PriorGrid(
        gene_names=["g1"],
        distribution=make_distribution_grid(
            "binomial",
            support=support,
            probabilities=probabilities,
        ),
        scale=12.0,
    )
    report = fit_prior_gmm(
        prior,
        search_config=GMMSearchConfig(
            max_components=3,
            error_threshold=2e-2,
            peak_limit_per_stage=3,
            candidate_window_count=4,
            candidate_sigma_count=5,
            search_refit_max_iterations=20,
        ),
        training_config=GMMTrainingConfig(
            max_iterations=80,
            learning_rate=0.05,
            convergence_tolerance=1e-7,
            gene_chunk_size=1,
            compile_model=False,
        ),
    )
    assert report.gene_names == ["g1"]
    assert report.lower_bounds[0] < support[0]
    assert report.upper_bounds[0] > support[-1]
    mixture = report.to_mixture("g1")
    assert mixture.lower_bound == report.lower_bounds[0]
    assert mixture.upper_bound == report.upper_bounds[0]
    assert np.allclose(np.sum(report.fitted_probabilities, axis=1), 1.0)


def test_fit_prior_gmm_accepts_unsorted_duplicate_support_for_raw_and_scaled() -> None:
    prior = PriorGrid(
        gene_names=["g1"],
        distribution=make_distribution_grid(
            "binomial",
            support=np.asarray([0.5, 0.2, 0.5, 0.8], dtype=np.float64),
            probabilities=np.asarray([0.2, 0.3, 0.1, 0.4], dtype=np.float64),
        ),
        scale=12.0,
    )
    for support_axis, expected_support, expected_domain in [
        (
            "raw",
            np.asarray([0.2, 0.5, 0.8], dtype=np.float64),
            "probability",
        ),
        (
            "scaled",
            np.asarray([2.4, 6.0, 9.6], dtype=np.float64),
            "rate",
        ),
    ]:
        report = fit_prior_gmm(
            prior,
            search_config=GMMSearchConfig(
                max_components=3,
                error_threshold=0.2,
                peak_limit_per_stage=3,
                candidate_window_count=3,
                candidate_sigma_count=3,
                search_refit_enabled=False,
            ),
            training_config=GMMTrainingConfig(
                max_iterations=10,
                learning_rate=0.05,
                convergence_tolerance=1e-7,
                gene_chunk_size=1,
                compile_model=False,
            ),
            support_axis=support_axis,
        )
        assert report.support_domain == expected_domain
        assert report.support.shape == (1, 3)
        assert np.allclose(report.support[0], expected_support)
        assert np.allclose(
            report.scaled_support[0],
            np.asarray([2.4, 6.0, 9.6], dtype=np.float64),
        )
        assert np.allclose(
            report.probabilities[0],
            np.asarray([0.3, 0.3, 0.4], dtype=np.float64),
        )
        assert np.isclose(np.sum(report.fitted_probabilities[0]), 1.0)
