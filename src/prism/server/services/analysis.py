from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, cast

import numpy as np

from prism.baseline.metrics import (
    evaluate_representations,
    log1p_normalize_total,
    normalize_total,
    raw_umi,
)
from prism.model import (
    KBulkBatch,
    ObservationBatch,
    Posterior,
    PriorFitConfig,
    PriorFitResult,
    SignalChannel,
    effective_exposure,
    fit_gene_priors,
    infer_kbulk,
)
from prism.model import OptimizerName, SchedulerName

from ..state import AppState, LoadedState
from .datasets import (
    GeneCandidate,
    GeneNotFoundError,
    compute_reference_counts,
    resolve_gene_positions,
    resolve_gene_query,
    search_gene_candidates as search_gene_candidates_raw,
    slice_gene_counts,
)


@dataclass(frozen=True, slots=True)
class RepresentationMetric:
    mean: float
    median: float
    std: float
    var: float
    p95: float
    nonzero_frac: float
    depth_corr: float
    depth_mi: float
    sparsity_corr: float | None
    fisher_ratio: float | None
    kruskal_h: float | None
    kruskal_p: float | None
    auroc_ovr: float | None
    zero_consistency: float | None
    zero_rank_tau: float | None
    dropout_recovery: float | None
    treatment_cv: float | None


@dataclass(frozen=True, slots=True)
class GeneSummary:
    gene_name: str
    gene_index: int
    total_counts: float
    mean_count: float
    median_count: float
    p90_count: float
    p99_count: float
    max_count: float
    detected_cells: int
    detected_frac: float
    zero_frac: float
    count_total_correlation: float


@dataclass(frozen=True, slots=True)
class GeneBrowsePage:
    items: list[GeneCandidate]
    page: int
    page_size: int
    total_items: int
    total_pages: int
    query: str
    sort_by: str
    descending: bool
    scope: str


@dataclass(frozen=True, slots=True)
class GeneFitParams:
    S: float | None = None
    reference_mode: Literal["checkpoint", "all"] = "checkpoint"
    grid_size: int = 512
    sigma_bins: float = 1.0
    align_loss_weight: float = 1.0
    lr: float = 0.05
    n_iter: int = 100
    lr_min_ratio: float = 0.1
    grad_clip: float | None = None
    init_temperature: float = 1.0
    cell_chunk_size: int = 512
    optimizer: OptimizerName = "adamw"
    scheduler: SchedulerName = "cosine"
    torch_dtype: Literal["float64", "float32"] = "float64"
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class KBulkParams:
    k: int = 8
    n_samples: int = 24
    max_groups: int = 4
    min_cells_per_group: int = 24
    random_seed: int = 0
    grid_size: int = 256
    sigma_bins: float = 1.0
    align_loss_weight: float = 1.0
    lr: float = 0.05
    n_iter: int = 50
    lr_min_ratio: float = 0.1
    init_temperature: float = 1.0
    cell_chunk_size: int = 256
    optimizer: OptimizerName = "adamw"
    scheduler: SchedulerName = "cosine"
    torch_dtype: Literal["float64", "float32"] = "float64"
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class KBulkGroupAnalysis:
    label: str
    n_cells: int
    support_mu: np.ndarray
    prior_weights: np.ndarray
    sampled_map_mu: np.ndarray
    sampled_map_p: np.ndarray
    posterior_entropy: np.ndarray
    mean_map_mu: float
    std_map_mu: float
    mean_posterior_entropy: float


@dataclass(frozen=True, slots=True)
class KBulkComparison:
    label_key: str
    reference_mode: str
    S: float
    k: int
    n_samples: int
    groups: list[KBulkGroupAnalysis]


@dataclass(frozen=True, slots=True)
class GeneAnalysis:
    cache_key: str
    gene_name: str
    gene_index: int
    source: str
    S: float
    S_source: str
    reference_mode: str
    reference_gene_count: int
    counts: np.ndarray
    totals: np.ndarray
    reference_counts: np.ndarray
    observed_mu_proxy: np.ndarray
    map_p: np.ndarray
    signal: np.ndarray
    posterior_entropy: np.ndarray
    prior_entropy: np.ndarray
    mutual_information: np.ndarray
    support_p: np.ndarray
    support_mu: np.ndarray
    prior_weights: np.ndarray
    posterior_samples: np.ndarray
    posterior_cell_indices: np.ndarray
    summary: GeneSummary
    representations: dict[str, np.ndarray]
    representation_metrics: dict[str, RepresentationMetric]
    fit_params: dict[str, object] | None = None
    fit_result: PriorFitResult | None = None
    kbulk: KBulkComparison | None = None


def build_dataset_summary(state: AppState) -> dict[str, object]:
    loaded = state.require_loaded()
    cache_key = state.make_cache_key("summary", "dataset")
    cached = state.get_cache("summary", cache_key)
    if cached is not None:
        return cast(dict[str, object], cached)
    checkpoint = loaded.checkpoint
    summary = {
        "n_cells": loaded.n_cells,
        "n_genes": loaded.n_genes,
        "fitted_genes": len(loaded.fitted_gene_names),
        "layer": loaded.dataset.layer or "X",
        "h5ad_path": str(loaded.dataset.h5ad_path),
        "ckpt_path": "" if checkpoint is None else str(checkpoint.ckpt_path),
        "S": None if checkpoint is None else float(checkpoint.checkpoint.scale.S),
        "S_source": None
        if checkpoint is None
        else str(checkpoint.checkpoint.metadata.get("S_source", "checkpoint")),
        "mean_reference_count": None
        if checkpoint is None
        else float(checkpoint.checkpoint.scale.mean_reference_count),
        "reference_genes": 0
        if checkpoint is None
        else len(checkpoint.reference_gene_names),
        "label_key": loaded.dataset.treatment_label_key,
        "model_source": "dataset-only" if checkpoint is None else "checkpoint",
        "mean_total": float(np.mean(loaded.dataset.totals)),
        "median_total": float(np.median(loaded.dataset.totals)),
    }
    state.set_cache("summary", cache_key, summary)
    return summary


def search_gene_candidates(
    state: AppState, query: str, limit: int | None = None
) -> list[GeneCandidate]:
    loaded = state.require_loaded()
    limit = state.config.top_gene_limit if limit is None else limit
    cache_key = state.make_cache_key("search", query, limit)
    cached = state.get_cache("search", cache_key)
    if cached is not None:
        return cast(list[GeneCandidate], cached)
    candidates = search_gene_candidates_raw(
        query=query,
        gene_names=loaded.dataset.gene_names,
        gene_names_lower=loaded.dataset.gene_names_lower,
        gene_total_counts=loaded.dataset.gene_total_counts,
        gene_detected_counts=loaded.dataset.gene_detected_counts,
        ranked_indices=loaded.dataset.ranked_gene_indices,
        n_cells=loaded.n_cells,
        limit=limit,
    )
    state.set_cache("search", cache_key, candidates)
    return candidates


def browse_gene_candidates(
    state: AppState,
    *,
    query: str = "",
    sort_by: str = "total_umi",
    descending: bool = True,
    page: int = 1,
    page_size: int | None = None,
    scope: str = "auto",
) -> GeneBrowsePage:
    loaded = state.require_loaded()
    resolved_page_size = (
        state.config.browse_page_size if page_size is None else max(1, page_size)
    )
    resolved_scope = scope if scope in {"auto", "all", "fitted"} else "auto"
    dataset_candidates = search_gene_candidates(state, query, limit=loaded.n_genes)
    if resolved_scope == "fitted" or (
        resolved_scope == "auto" and loaded.checkpoint is not None
    ):
        fitted = set(loaded.fitted_gene_names)
        items = [item for item in dataset_candidates if item.gene_name in fitted]
    else:
        items = dataset_candidates
    key_map = {
        "total_umi": lambda item: item.total_umi,
        "detected_cells": lambda item: item.detected_cells,
        "detected_fraction": lambda item: item.detected_fraction,
        "gene_name": lambda item: item.gene_name.lower(),
        "gene_index": lambda item: item.gene_index,
    }
    sort_key = key_map.get(sort_by, key_map["total_umi"])
    items = sorted(items, key=sort_key, reverse=descending)
    total_items = len(items)
    total_pages = max((total_items + resolved_page_size - 1) // resolved_page_size, 1)
    resolved_page = min(max(page, 1), total_pages)
    start = (resolved_page - 1) * resolved_page_size
    end = start + resolved_page_size
    return GeneBrowsePage(
        items=items[start:end],
        page=resolved_page,
        page_size=resolved_page_size,
        total_items=total_items,
        total_pages=total_pages,
        query=query,
        sort_by=sort_by,
        descending=descending,
        scope=resolved_scope,
    )


def summarize_gene_expression(loaded: LoadedState, gene_index: int) -> GeneSummary:
    gene_name = str(loaded.dataset.gene_names[gene_index])
    counts = slice_gene_counts(loaded.dataset.matrix, gene_index)
    totals = np.asarray(loaded.dataset.totals, dtype=np.float64)
    count_total_correlation = _safe_corr(counts, totals)
    return GeneSummary(
        gene_name=gene_name,
        gene_index=gene_index,
        total_counts=float(np.sum(counts)),
        mean_count=float(np.mean(counts)),
        median_count=float(np.median(counts)),
        p90_count=float(np.quantile(counts, 0.9)),
        p99_count=float(np.quantile(counts, 0.99)),
        max_count=float(np.max(counts)),
        detected_cells=int(np.count_nonzero(counts > 0)),
        detected_frac=float(np.mean(counts > 0)),
        zero_frac=float(np.mean(counts <= 0)),
        count_total_correlation=count_total_correlation,
    )


def analyze_gene(
    state: AppState,
    query: str,
    *,
    fit_params: GeneFitParams | None = None,
    kbulk_params: KBulkParams | None = None,
) -> GeneAnalysis:
    loaded = state.require_loaded()
    gene_index = resolve_gene_query(
        query,
        loaded.dataset.gene_names,
        loaded.dataset.gene_names_lower,
        loaded.dataset.gene_to_idx,
        loaded.dataset.gene_lower_to_idx,
    )
    gene_name = str(loaded.dataset.gene_names[gene_index])
    kbulk_key = (
        "none"
        if kbulk_params is None
        else f"kbulk-{kbulk_params.k}-{kbulk_params.n_samples}-{kbulk_params.max_groups}-{kbulk_params.random_seed}"
    )
    cache_key = state.make_cache_key(
        "analysis",
        gene_name,
        "fit" if fit_params is not None else "checkpoint",
        kbulk_key,
    )
    if fit_params is None and kbulk_params is None:
        cached = state.get_cache("analysis", cache_key)
        if cached is not None:
            return cast(GeneAnalysis, cached)
    analysis = _analyze_gene_uncached(
        loaded,
        gene_index=gene_index,
        fit_params=fit_params,
        kbulk_params=kbulk_params,
        cache_key=cache_key,
    )
    if fit_params is None and kbulk_params is None:
        state.set_cache("analysis", cache_key, analysis)
    return analysis


def _analyze_gene_uncached(
    loaded: LoadedState,
    *,
    gene_index: int,
    fit_params: GeneFitParams | None,
    kbulk_params: KBulkParams | None,
    cache_key: str,
) -> GeneAnalysis:
    gene_name = str(loaded.dataset.gene_names[gene_index])
    counts = slice_gene_counts(loaded.dataset.matrix, gene_index)
    if (
        fit_params is None
        and loaded.checkpoint is not None
        and gene_name in set(loaded.fitted_gene_names)
    ):
        priors = loaded.checkpoint.checkpoint.priors.subset(gene_name)
        reference_names = list(loaded.checkpoint.reference_gene_names)
        reference_mode = "checkpoint"
        S = float(loaded.checkpoint.checkpoint.scale.S)
        S_source = str(
            loaded.checkpoint.checkpoint.metadata.get("S_source", "checkpoint")
        )
        fit_result = None
        fit_payload = None
        source = "checkpoint"
    else:
        params = GeneFitParams() if fit_params is None else fit_params
        reference_names = _resolve_reference_gene_names(
            loaded, gene_name, params.reference_mode
        )
        reference_mode = (
            params.reference_mode if loaded.checkpoint is not None else "all"
        )
        reference_positions = resolve_gene_positions(
            reference_names, loaded.dataset.gene_to_idx
        )
        reference_counts = compute_reference_counts(
            loaded.dataset.matrix, reference_positions
        )
        S = float(np.mean(reference_counts)) if params.S is None else float(params.S)
        S_source = "default:N_avg" if params.S is None else "user"
        fit_result = fit_gene_priors(
            ObservationBatch(
                gene_names=[gene_name],
                counts=np.asarray(counts, dtype=np.float64).reshape(-1, 1),
                reference_counts=reference_counts,
            ),
            S=S,
            config=PriorFitConfig(
                grid_size=params.grid_size,
                sigma_bins=params.sigma_bins,
                align_loss_weight=params.align_loss_weight,
                lr=params.lr,
                n_iter=params.n_iter,
                lr_min_ratio=params.lr_min_ratio,
                grad_clip=params.grad_clip,
                init_temperature=params.init_temperature,
                cell_chunk_size=params.cell_chunk_size,
                optimizer=params.optimizer,
                scheduler=params.scheduler,
                torch_dtype=params.torch_dtype,
            ),
            device=params.device,
        )
        priors = fit_result.priors
        fit_payload = asdict(params)
        source = "on-demand-fit"

    reference_positions = resolve_gene_positions(
        reference_names, loaded.dataset.gene_to_idx
    )
    reference_counts = compute_reference_counts(
        loaded.dataset.matrix, reference_positions
    )
    posterior = Posterior([gene_name], priors, device="cpu")
    summary = posterior.summarize(
        ObservationBatch(
            gene_names=[gene_name],
            counts=np.asarray(counts, dtype=np.float64).reshape(-1, 1),
            reference_counts=reference_counts,
        )
    )
    observed_mu_proxy = counts * S / max(float(np.mean(reference_counts)), 1e-12)
    signal = np.asarray(summary.map_mu[:, 0], dtype=np.float64)
    map_p = np.asarray(summary.map_p[:, 0], dtype=np.float64)
    posterior_entropy = np.asarray(summary.posterior_entropy[:, 0], dtype=np.float64)
    prior_entropy = np.asarray(summary.prior_entropy[:, 0], dtype=np.float64)
    mutual_information = np.asarray(summary.mutual_information[:, 0], dtype=np.float64)
    posterior_samples = np.asarray(summary.posterior[:, 0, :], dtype=np.float64)
    representations = {
        "raw_count": raw_umi(
            np.asarray(counts, dtype=np.float64).reshape(-1, 1)
        ).reshape(-1),
        "normalize_total": normalize_total(
            np.asarray(counts, dtype=np.float64).reshape(-1, 1), reference_counts
        ).reshape(-1),
        "log1p_normalize_total": log1p_normalize_total(
            np.asarray(counts, dtype=np.float64).reshape(-1, 1), reference_counts
        ).reshape(-1),
        "signal": signal,
    }
    metrics_raw = evaluate_representations(
        representations,
        totals=reference_counts,
        raw_counts=counts,
        labels=loaded.dataset.treatment_labels,
        zero_fraction=loaded.dataset.cell_zero_fraction,
    )
    metrics = {name: _to_metric(values) for name, values in metrics_raw.items()}
    representative = _representative_indices(signal, n=12)
    kbulk = None
    if kbulk_params is not None:
        kbulk = _compute_kbulk_comparison(
            loaded,
            gene_name=gene_name,
            counts=counts,
            reference_mode=reference_mode,
            S=S,
            params=kbulk_params,
        )
    return GeneAnalysis(
        cache_key=cache_key,
        gene_name=gene_name,
        gene_index=gene_index,
        source=source,
        S=S,
        S_source=S_source,
        reference_mode=reference_mode,
        reference_gene_count=len(reference_names),
        counts=counts,
        totals=np.asarray(loaded.dataset.totals, dtype=np.float64),
        reference_counts=reference_counts,
        observed_mu_proxy=observed_mu_proxy,
        map_p=map_p,
        signal=signal,
        posterior_entropy=posterior_entropy,
        prior_entropy=prior_entropy,
        mutual_information=mutual_information,
        support_p=np.asarray(summary.p_grid[0], dtype=np.float64),
        support_mu=np.asarray(summary.mu_grid[0], dtype=np.float64),
        prior_weights=np.asarray(summary.prior_weights[0], dtype=np.float64),
        posterior_samples=posterior_samples[representative],
        posterior_cell_indices=representative,
        summary=summarize_gene_expression(loaded, gene_index),
        representations=representations,
        representation_metrics=metrics,
        fit_params=fit_payload,
        fit_result=fit_result,
        kbulk=kbulk,
    )


def _resolve_reference_gene_names(
    loaded: LoadedState, gene_name: str, mode: str
) -> list[str]:
    if mode == "checkpoint" and loaded.checkpoint is not None:
        names = [
            name for name in loaded.checkpoint.reference_gene_names if name != gene_name
        ]
        if names:
            return names
    return [
        str(name)
        for name in loaded.dataset.gene_names.tolist()
        if str(name) != gene_name
    ]


def _representative_indices(values: np.ndarray, n: int) -> np.ndarray:
    if values.size <= n:
        return np.arange(values.size, dtype=np.int64)
    order = np.argsort(values)
    anchors = np.linspace(0, len(order) - 1, n).round().astype(np.int64)
    return np.unique(order[anchors])


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(
        np.corrcoef(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))[
            0, 1
        ]
    )


def _compute_kbulk_comparison(
    loaded: LoadedState,
    *,
    gene_name: str,
    counts: np.ndarray,
    reference_mode: str,
    S: float,
    params: KBulkParams,
) -> KBulkComparison | None:
    labels = loaded.dataset.treatment_labels
    label_key = loaded.dataset.treatment_label_key
    if labels is None or label_key is None:
        return None
    reference_names = _resolve_reference_gene_names(loaded, gene_name, reference_mode)
    reference_positions = resolve_gene_positions(
        reference_names, loaded.dataset.gene_to_idx
    )
    if not reference_positions:
        return None
    unique_labels, label_sizes = np.unique(labels, return_counts=True)
    order = np.argsort(label_sizes)[::-1]
    groups: list[KBulkGroupAnalysis] = []
    rng = np.random.default_rng(params.random_seed)
    fit_config = PriorFitConfig(
        grid_size=params.grid_size,
        sigma_bins=params.sigma_bins,
        align_loss_weight=params.align_loss_weight,
        lr=params.lr,
        n_iter=params.n_iter,
        lr_min_ratio=params.lr_min_ratio,
        grad_clip=None,
        init_temperature=params.init_temperature,
        cell_chunk_size=params.cell_chunk_size,
        optimizer=params.optimizer,
        scheduler=params.scheduler,
        torch_dtype=params.torch_dtype,
    )
    for order_idx in order.tolist():
        group_label = unique_labels[order_idx]
        group_indices = np.flatnonzero(labels == group_label)
        if group_indices.size < max(params.k, params.min_cells_per_group):
            continue
        group_reference_counts = compute_reference_counts(
            loaded.dataset.matrix,
            reference_positions,
            cell_indices=group_indices,
        )
        valid_mask = np.asarray(group_reference_counts > 0, dtype=bool)
        if int(np.count_nonzero(valid_mask)) < max(
            params.k, params.min_cells_per_group
        ):
            continue
        group_reference_counts = np.asarray(
            group_reference_counts[valid_mask], dtype=np.float64
        )
        group_counts = np.asarray(
            counts[group_indices][valid_mask], dtype=np.float64
        ).reshape(-1, 1)
        fit_result = fit_gene_priors(
            ObservationBatch(
                gene_names=[gene_name],
                counts=group_counts,
                reference_counts=group_reference_counts,
            ),
            S=S,
            config=fit_config,
            device=params.device,
        )
        priors = fit_result.priors
        n_eff = effective_exposure(group_reference_counts, S)
        sampled_mu: list[float] = []
        sampled_p: list[float] = []
        sampled_entropy: list[float] = []
        n_draws = max(1, params.n_samples)
        for _ in range(n_draws):
            chosen = rng.choice(
                np.arange(group_counts.shape[0]), size=params.k, replace=False
            )
            kbulk = infer_kbulk(
                KBulkBatch(
                    gene_names=[gene_name],
                    counts=group_counts[chosen],
                    effective_exposure=n_eff[chosen],
                ),
                priors,
                include_posterior=False,
            )
            sampled_mu.append(float(kbulk.map_mu[0]))
            sampled_p.append(float(kbulk.map_p[0]))
            sampled_entropy.append(float(kbulk.posterior_entropy[0]))
        groups.append(
            KBulkGroupAnalysis(
                label=str(group_label),
                n_cells=int(group_counts.shape[0]),
                support_mu=np.asarray(priors.mu_grid[0], dtype=np.float64),
                prior_weights=np.asarray(priors.weights[0], dtype=np.float64),
                sampled_map_mu=np.asarray(sampled_mu, dtype=np.float64),
                sampled_map_p=np.asarray(sampled_p, dtype=np.float64),
                posterior_entropy=np.asarray(sampled_entropy, dtype=np.float64),
                mean_map_mu=float(np.mean(sampled_mu)),
                std_map_mu=float(np.std(sampled_mu)),
                mean_posterior_entropy=float(np.mean(sampled_entropy)),
            )
        )
        if len(groups) >= params.max_groups:
            break
    if not groups:
        return None
    return KBulkComparison(
        label_key=label_key,
        reference_mode=reference_mode,
        S=S,
        k=params.k,
        n_samples=params.n_samples,
        groups=groups,
    )


def _to_metric(values: dict[str, float | None]) -> RepresentationMetric:
    def optional(name: str) -> float | None:
        value = values.get(name)
        return None if value is None else float(value)

    return RepresentationMetric(
        mean=float(values.get("mean") or 0.0),
        median=float(values.get("median") or 0.0),
        std=float(values.get("std") or 0.0),
        var=float(values.get("var") or 0.0),
        p95=float(values.get("p95") or 0.0),
        nonzero_frac=float(values.get("nonzero_frac") or 0.0),
        depth_corr=float(values.get("depth_corr") or 0.0),
        depth_mi=float(values.get("depth_mi") or 0.0),
        sparsity_corr=optional("sparsity_corr"),
        fisher_ratio=optional("fisher_ratio"),
        kruskal_h=optional("kruskal_h"),
        kruskal_p=optional("kruskal_p"),
        auroc_ovr=optional("auroc_ovr"),
        zero_consistency=optional("zero_consistency"),
        zero_rank_tau=optional("zero_rank_tau"),
        dropout_recovery=optional("dropout_recovery"),
        treatment_cv=optional("treatment_cv"),
    )
