#!/usr/bin/env python3
"""Benchmark PRISM GMM search/fit hot paths on synthetic distributions."""

from __future__ import annotations

import argparse
import cProfile
import math
from pstats import SortKey, Stats
from time import perf_counter

import numpy as np

try:
    from prism.gmm import (
        GMMSearchConfig,
        GMMTrainingConfig,
        fit_distribution_gmm,
        search_distribution_gmm,
    )
    from prism.model import make_distribution_grid
except ImportError as exc:
    raise ImportError(
        "PRISM is not installed in the active environment. Run `pip install -e .` "
        "from the repository root before executing scripts/experiments/benchmark_gmm_hotpath.py."
    ) from exc


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(array / math.sqrt(2.0)))


def _build_edges(support: np.ndarray) -> np.ndarray:
    points = np.asarray(support, dtype=np.float64).reshape(-1)
    if points.size == 1:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    midpoints = 0.5 * (points[:-1] + points[1:])
    left = points[0] - 0.5 * (points[1] - points[0])
    right = points[-1] + 0.5 * (points[-1] - points[-2])
    edges = np.concatenate([[left], midpoints, [right]])
    edges = np.clip(edges, 0.0, 1.0)
    edges = np.maximum.accumulate(edges)
    for idx in range(1, edges.shape[0]):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1e-12
    return edges


def _truncated_gaussian_grid(
    support: np.ndarray,
    components: list[tuple[float, float, float, float, float]],
) -> np.ndarray:
    edges = _build_edges(support)
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


def _gene_components(gene_idx: int, *, max_true_components: int) -> list[tuple[float, float, float, float, float]]:
    count = 1 + (gene_idx % max_true_components)
    anchors = np.linspace(0.14, 0.86, count, dtype=np.float64)
    raw_weights = np.arange(1, count + 1, dtype=np.float64)
    raw_weights = np.roll(raw_weights, gene_idx % count)
    weights = raw_weights / raw_weights.sum()
    components: list[tuple[float, float, float, float, float]] = []
    for component_idx, anchor in enumerate(anchors.tolist()):
        shift = 0.02 * math.sin((gene_idx + 1) * (component_idx + 1))
        mean = min(max(anchor + shift, 0.02), 0.98)
        std = 0.035 + 0.01 * ((gene_idx + component_idx) % 4)
        components.append((float(weights[component_idx]), mean, std, 0.0, 1.0))
    return components


def build_distribution(
    *,
    n_genes: int,
    n_support: int,
    max_true_components: int,
) -> object:
    support = np.linspace(0.0, 1.0, n_support, dtype=np.float64)
    support_matrix = np.repeat(support[None, :], n_genes, axis=0)
    probabilities = np.stack(
        [
            _truncated_gaussian_grid(
                support,
                _gene_components(
                    gene_idx,
                    max_true_components=max_true_components,
                ),
            )
            for gene_idx in range(n_genes)
        ],
        axis=0,
    )
    return make_distribution_grid(
        "binomial",
        support=support_matrix,
        probabilities=probabilities,
    )


def _profile_call(label: str, fn: object, *, top_n: int) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    fn()
    profiler.disable()
    print(f"\n[{label}] top {top_n} cumulative functions")
    Stats(profiler).sort_stats(SortKey.CUMULATIVE).print_stats(top_n)


def _summarize(label: str, timings: list[float]) -> None:
    array = np.asarray(timings, dtype=np.float64)
    print(
        f"{label}: mean={array.mean():.4f}s min={array.min():.4f}s "
        f"max={array.max():.4f}s repeats={array.shape[0]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", type=int, default=16)
    parser.add_argument("--support", type=int, default=97)
    parser.add_argument("--max-components", type=int, default=4)
    parser.add_argument("--true-components", type=int, default=1)
    parser.add_argument("--fit-iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--torch-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--search-refit", action="store_true")
    parser.add_argument("--profile", choices=("none", "search", "fit", "both"), default="none")
    parser.add_argument("--profile-top", type=int, default=15)
    args = parser.parse_args()

    distribution = build_distribution(
        n_genes=args.genes,
        n_support=args.support,
        max_true_components=args.true_components,
    )
    search_config = GMMSearchConfig(
        max_components=args.max_components,
        peak_limit_per_stage=args.max_components,
        candidate_window_count=6,
        candidate_sigma_count=8,
        search_refit_enabled=args.search_refit,
    )
    training_config = GMMTrainingConfig(
        max_iterations=args.fit_iterations,
        gene_chunk_size=min(args.genes, 256),
        torch_dtype=args.torch_dtype,
        compile_policy="never",
    )

    for _ in range(args.warmup):
        search = search_distribution_gmm(
            distribution,
            config=search_config,
            torch_dtype=args.torch_dtype,
            device=args.device,
        )
        fit_distribution_gmm(
            distribution,
            search=search,
            training_config=training_config,
            device=args.device,
        )

    search_timings: list[float] = []
    fit_timings: list[float] = []
    cached_search = None
    for _ in range(args.repeat):
        start = perf_counter()
        cached_search = search_distribution_gmm(
            distribution,
            config=search_config,
            torch_dtype=args.torch_dtype,
            device=args.device,
        )
        search_timings.append(perf_counter() - start)

        start = perf_counter()
        fit_distribution_gmm(
            distribution,
            search=cached_search,
            training_config=training_config,
            device=args.device,
        )
        fit_timings.append(perf_counter() - start)

    print(
        "Synthetic benchmark:",
        f"genes={args.genes}",
        f"support={args.support}",
        f"max_components={args.max_components}",
        f"fit_iterations={args.fit_iterations}",
        f"search_refit={args.search_refit}",
        f"device={args.device}",
        f"torch_dtype={args.torch_dtype}",
    )
    _summarize("search", search_timings)
    _summarize("fit", fit_timings)

    if args.profile in {"search", "both"}:
        _profile_call(
            "search",
            lambda: search_distribution_gmm(
                distribution,
                config=search_config,
                torch_dtype=args.torch_dtype,
                device=args.device,
            ),
            top_n=args.profile_top,
        )
    if args.profile in {"fit", "both"}:
        if cached_search is None:
            cached_search = search_distribution_gmm(
                distribution,
                config=search_config,
                torch_dtype=args.torch_dtype,
                device=args.device,
            )
        _profile_call(
            "fit",
            lambda: fit_distribution_gmm(
                distribution,
                search=cached_search,
                training_config=training_config,
                device=args.device,
            ),
            top_n=args.profile_top,
        )


if __name__ == "__main__":
    main()
