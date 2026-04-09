from __future__ import annotations

import math

import numpy as np

from prism.gmm import (
    GMMSearchConfig,
    GMMTrainingConfig,
    fit_distribution_gmm,
    fit_prior_gmm,
    search_distribution_gmm,
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
            candidate_peak_count=3,
            candidate_window_count=4,
            candidate_sigma_count=5,
        ),
        torch_dtype="float64",
    )
    assert search.selected_k.tolist() == [1]
    assert np.isclose(np.sum(search.greedy_probabilities[0]), 1.0)
    assert search.error_path[0, 0] < 0.02


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
            candidate_peak_count=3,
            candidate_window_count=4,
            candidate_sigma_count=5,
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
            candidate_peak_count=4,
            candidate_window_count=5,
            candidate_sigma_count=5,
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


def test_fit_prior_gmm_preserves_gene_names_and_raw_support_default() -> None:
    support = np.linspace(0.0, 1.0, 33, dtype=np.float64)
    probabilities_a = _truncated_gaussian_grid(
        support,
        [(1.0, 0.25, 0.07, 0.0, 1.0)],
        probability_domain=True,
    )
    probabilities_b = _truncated_gaussian_grid(
        support,
        [(0.35, 0.18, 0.05, 0.0, 1.0), (0.65, 0.72, 0.08, 0.0, 1.0)],
        probability_domain=True,
    )
    prior = PriorGrid(
        gene_names=["g1", "g2"],
        distribution=make_distribution_grid(
            "binomial",
            support=np.vstack([support, support]),
            probabilities=np.vstack([probabilities_a, probabilities_b]),
        ),
        scale=12.0,
    )
    report = fit_prior_gmm(
        prior,
        search_config=GMMSearchConfig(
            max_components=4,
            error_threshold=2e-2,
            candidate_peak_count=4,
            candidate_window_count=4,
            candidate_sigma_count=5,
        ),
        training_config=GMMTrainingConfig(
            max_iterations=100,
            learning_rate=0.05,
            convergence_tolerance=1e-7,
            gene_chunk_size=2,
            compile_model=False,
        ),
    )
    assert report.gene_names == ["g1", "g2"]
    assert np.allclose(report.support, np.vstack([support, support]))
    assert np.allclose(report.scaled_support, report.support * 12.0)
    assert np.allclose(np.sum(report.fitted_probabilities, axis=1), 1.0)
    assert np.all(report.jsd < 0.08)
