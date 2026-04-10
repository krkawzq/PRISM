from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from prism.model import (
    KBulkBatch,
    ModelCheckpoint,
    ObservationBatch,
    Posterior,
    PosteriorSummary,
    PriorFitConfig,
    PriorFitResult,
    PriorGrid,
    effective_exposure,
    fit_gene_priors,
    mean_reference_count,
    infer_kbulk,
)

from .checkpoints import CheckpointState
from .datasets import (
    GeneCandidate,
    GeneNotFoundError,
    compute_reference_counts,
    resolve_gene_query,
    search_gene_candidates as search_gene_candidates_impl,
    slice_gene_counts,
    slice_gene_matrix,
)


BrowseScope = Literal["auto", "fitted", "all"]
BrowseSort = Literal[
    "total_count",
    "detected_cells",
    "detected_fraction",
    "gene_name",
    "gene_index",
]
AnalysisMode = Literal["raw", "checkpoint", "fit"]
PriorSource = Literal["global", "label"]


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    ckpt_path: str
    gene_count: int
    has_global_prior: bool
    n_label_priors: int
    label_preview: tuple[str, ...]
    distribution: str
    support_domain: str | None
    scale: float | None
    mean_reference_count: float | None
    n_reference_genes: int
    n_overlap_reference_genes: int
    suggested_label_key: str | None


@dataclass(frozen=True, slots=True)
class GeneBrowsePage:
    query: str
    scope: BrowseScope
    sort_by: BrowseSort
    descending: bool
    page: int
    page_size: int
    total_items: int
    total_pages: int
    items: list[GeneCandidate]


@dataclass(frozen=True, slots=True)
class GeneSummary:
    total_count: float
    mean_count: float
    median_count: float
    p90_count: float
    p99_count: float
    max_count: float
    detected_cells: int
    detected_fraction: float
    zero_fraction: float
    count_total_correlation: float


@dataclass(frozen=True, slots=True)
class GeneFitParams:
    scale: float | None = None
    reference_source: Literal["checkpoint", "dataset"] = "checkpoint"
    n_support_points: int = 512
    max_em_iterations: int | None = 200
    convergence_tolerance: float = 1e-6
    cell_chunk_size: int = 512
    support_max_from: Literal["observed_max", "quantile"] = "observed_max"
    support_spacing: Literal["linear", "sqrt"] = "linear"
    support_scale: float = 1.5
    use_adaptive_support: bool = False
    adaptive_support_scale: float = 1.5
    adaptive_support_quantile_hi: float = 0.99
    likelihood: Literal["binomial", "negative_binomial", "poisson"] = "binomial"
    nb_overdispersion: float = 0.01
    torch_dtype: Literal["float32", "float64"] = "float64"
    compile_model: bool = True
    device: str = "cpu"

    def to_fit_config(self) -> PriorFitConfig:
        return PriorFitConfig(
            n_support_points=self.n_support_points,
            max_em_iterations=self.max_em_iterations,
            convergence_tolerance=self.convergence_tolerance,
            cell_chunk_size=self.cell_chunk_size,
            support_max_from=self.support_max_from,
            support_spacing=self.support_spacing,
            support_scale=self.support_scale,
            use_adaptive_support=self.use_adaptive_support,
            adaptive_support_scale=self.adaptive_support_scale,
            adaptive_support_quantile_hi=self.adaptive_support_quantile_hi,
            likelihood=self.likelihood,
            nb_overdispersion=self.nb_overdispersion,
        )


@dataclass(frozen=True, slots=True)
class KBulkParams:
    class_key: str | None = None
    k: int = 8
    n_samples: int = 24
    sample_seed: int = 0
    max_classes: int = 6
    sample_batch_size: int = 32
    kbulk_prior_source: PriorSource = "global"
    torch_dtype: Literal["float32", "float64"] = "float64"
    compile_model: bool = True
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class KBulkGroupSummary:
    label: str
    n_cells: int
    realized_samples: int
    mean_signal: float
    std_signal: float
    mean_entropy: float
    std_entropy: float
    signal_values: np.ndarray
    entropy_values: np.ndarray


@dataclass(frozen=True, slots=True)
class KBulkAnalysis:
    cache_key: str
    gene_name: str
    gene_index: int
    class_key: str
    prior_source: PriorSource
    k: int
    n_samples: int
    sample_seed: int
    max_classes: int
    sample_batch_size: int
    groups: list[KBulkGroupSummary]
    available_class_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GeneAnalysis:
    cache_key: str
    gene_name: str
    gene_index: int
    source: str
    mode: AnalysisMode
    prior_source: PriorSource
    label_key: str | None
    label: str | None
    n_cells: int
    reference_gene_count: int
    reference_source: str
    raw_summary: GeneSummary
    counts: np.ndarray
    totals: np.ndarray
    reference_counts: np.ndarray
    raw_proxy: np.ndarray
    prior: PriorGrid | None
    checkpoint_prior: PriorGrid | None
    posterior: PosteriorSummary | None
    fit_result: PriorFitResult | None
    checkpoint_summary: CheckpointSummary | None
    available_label_keys: tuple[str, ...]
    available_labels: tuple[str, ...]


def build_dataset_summary(state: AppState) -> dict[str, object] | None:
    loaded = state.loaded
    if loaded is None:
        return None
    key = state.make_cache_key("summary", "dataset")

    def factory() -> dict[str, object]:
        totals = np.asarray(loaded.dataset.totals, dtype=np.float64)
        checkpoint_summary = build_checkpoint_summary(state)
        return {
            "n_cells": loaded.n_cells,
            "n_genes": loaded.n_genes,
            "layer": loaded.dataset.layer or "(X)",
            "h5ad_path": str(loaded.dataset.h5ad_path),
            "label_keys": loaded.label_keys,
            "total_count_mean": float(np.mean(totals)) if totals.size else 0.0,
            "total_count_median": float(np.median(totals)) if totals.size else 0.0,
            "total_count_p99": float(np.percentile(totals, 99)) if totals.size else 0.0,
            "checkpoint": checkpoint_summary,
        }

    return cast(dict[str, object], state.get_or_create_cache("summary", key, factory))


def build_checkpoint_summary(state: AppState) -> CheckpointSummary | None:
    loaded = state.loaded
    checkpoint_state = None if loaded is None else loaded.checkpoint
    if loaded is None or checkpoint_state is None:
        return None
    checkpoint_state = cast(CheckpointState, checkpoint_state)
    key = state.make_cache_key("summary", "checkpoint")

    def factory() -> CheckpointSummary:
        checkpoint = checkpoint_state.checkpoint
        global_prior = checkpoint.prior
        scale_metadata = checkpoint.scale_metadata
        return CheckpointSummary(
            ckpt_path=str(checkpoint_state.ckpt_path),
            gene_count=len(checkpoint.gene_names),
            has_global_prior=checkpoint.has_global_prior,
            n_label_priors=len(checkpoint.label_priors),
            label_preview=tuple(sorted(checkpoint.label_priors)[:8]),
            distribution=checkpoint_state.posterior_distribution,
            support_domain=(
                None if global_prior is None else str(global_prior.support_domain)
            ),
            scale=None if global_prior is None else float(global_prior.scale),
            mean_reference_count=(
                None
                if scale_metadata is None
                else float(scale_metadata.mean_reference_count)
            ),
            n_reference_genes=len(checkpoint_state.reference_gene_names),
            n_overlap_reference_genes=len(checkpoint_state.reference_positions),
            suggested_label_key=checkpoint_state.suggested_label_key,
        )

    return cast(CheckpointSummary, state.get_or_create_cache("summary", key, factory))


def search_gene_candidates(
    state: AppState,
    query: str,
    *,
    limit: int | None = None,
) -> list[GeneCandidate]:
    loaded = state.loaded
    if loaded is None:
        return []
    resolved_limit = (
        state.config.top_gene_limit if limit is None else max(1, int(limit))
    )
    key = state.make_cache_key("search", query, resolved_limit)
    return cast(
        list[GeneCandidate],
        state.get_or_create_cache(
            "search",
            key,
            lambda: search_gene_candidates_impl(
                query=query,
                gene_names=loaded.dataset.gene_names,
                gene_names_lower=loaded.dataset.gene_names_lower,
                gene_total_counts=loaded.dataset.gene_total_counts,
                gene_detected_counts=loaded.dataset.gene_detected_counts,
                ranked_indices=loaded.dataset.ranked_gene_indices,
                n_cells=loaded.n_cells,
                limit=resolved_limit,
            ),
        ),
    )


def browse_gene_candidates(
    state: AppState,
    *,
    query: str = "",
    sort_by: BrowseSort = "total_count",
    descending: bool = True,
    page: int = 1,
    scope: BrowseScope = "auto",
) -> GeneBrowsePage | None:
    loaded = state.loaded
    if loaded is None:
        return None
    page = max(1, int(page))
    page_size = state.config.browse_page_size
    sort_key = _normalize_sort_by(sort_by)
    scope_key = _normalize_scope(scope, has_checkpoint=loaded.checkpoint is not None)
    key = state.make_cache_key(
        "search", "browse", query, sort_key, descending, page, scope_key
    )

    def factory() -> GeneBrowsePage:
        indices = _filter_gene_indices(loaded, query=query, scope=scope_key)
        indices = _sort_gene_indices(
            loaded, indices, sort_by=sort_key, descending=descending
        )
        total_items = len(indices)
        total_pages = max(1, (total_items + page_size - 1) // page_size)
        page_resolved = min(page, total_pages)
        start = (page_resolved - 1) * page_size
        end = start + page_size
        items = [_candidate_from_index(loaded, idx) for idx in indices[start:end]]
        return GeneBrowsePage(
            query=query,
            scope=scope_key,
            sort_by=sort_key,
            descending=bool(descending),
            page=page_resolved,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            items=items,
        )

    return cast(GeneBrowsePage, state.get_or_create_cache("search", key, factory))


def build_gene_analysis(
    state: AppState,
    *,
    query: str,
    mode: AnalysisMode = "checkpoint",
    prior_source: PriorSource = "global",
    label_key: str | None = None,
    label: str | None = None,
    fit_params: GeneFitParams | None = None,
) -> GeneAnalysis:
    loaded = state.require_loaded()
    gene_index = _resolve_gene_index(loaded, query)
    gene_name = str(loaded.dataset.gene_names[gene_index])
    fit_payload = None if fit_params is None else asdict(fit_params)
    key = state.make_cache_key(
        "analysis",
        "gene",
        gene_name,
        mode,
        prior_source,
        label_key,
        label,
        fit_payload,
    )

    def factory() -> GeneAnalysis:
        return _build_gene_analysis_uncached(
            state,
            loaded=loaded,
            gene_index=gene_index,
            mode=mode,
            prior_source=prior_source,
            label_key=label_key,
            label=label,
            fit_params=fit_params,
            cache_key=key,
        )

    return cast(GeneAnalysis, state.get_or_create_cache("analysis", key, factory))


def compute_kbulk_analysis(
    state: AppState,
    *,
    query: str,
    params: KBulkParams,
    label_key: str | None = None,
    label: str | None = None,
) -> KBulkAnalysis:
    loaded = state.require_loaded()
    if loaded.checkpoint is None:
        raise ValueError("kBulk analysis requires a loaded checkpoint")
    gene_index = _resolve_gene_index(loaded, query)
    gene_name = str(loaded.dataset.gene_names[gene_index])
    class_key = _resolve_class_key(loaded, label_key or params.class_key)
    key = state.make_cache_key(
        "kbulk",
        gene_name,
        class_key,
        label,
        asdict(params),
    )

    def factory() -> KBulkAnalysis:
        return _compute_kbulk_uncached(
            loaded=loaded,
            gene_index=gene_index,
            class_key=class_key,
            label=label,
            params=params,
            cache_key=key,
        )

    return cast(KBulkAnalysis, state.get_or_create_cache("kbulk", key, factory))


def summarize_gene_expression(
    loaded: LoadedState,
    gene_index: int,
    *,
    cell_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, GeneSummary]:
    counts = slice_gene_counts(loaded.dataset.matrix, gene_index)
    totals = np.asarray(loaded.dataset.totals, dtype=np.float64)
    if cell_indices is not None:
        counts = np.asarray(counts[cell_indices], dtype=np.float64)
        totals = np.asarray(totals[cell_indices], dtype=np.float64)
    else:
        counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        raise ValueError("selected gene has no cells to summarize")
    detected = int(np.count_nonzero(counts > 0))
    zero_fraction = float(np.mean(counts <= 0))
    if (
        counts.size < 2
        or np.allclose(counts, counts[0])
        or np.allclose(totals, totals[0])
    ):
        corr = 0.0
    else:
        corr = float(np.corrcoef(counts, totals)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    summary = GeneSummary(
        total_count=float(np.sum(counts)),
        mean_count=float(np.mean(counts)),
        median_count=float(np.median(counts)),
        p90_count=float(np.percentile(counts, 90)),
        p99_count=float(np.percentile(counts, 99)),
        max_count=float(np.max(counts)),
        detected_cells=detected,
        detected_fraction=float(detected / max(counts.shape[0], 1)),
        zero_fraction=zero_fraction,
        count_total_correlation=corr,
    )
    return counts, totals, summary


def _build_gene_analysis_uncached(
    state: AppState,
    *,
    loaded: LoadedState,
    gene_index: int,
    mode: AnalysisMode,
    prior_source: PriorSource,
    label_key: str | None,
    label: str | None,
    fit_params: GeneFitParams | None,
    cache_key: str,
) -> GeneAnalysis:
    gene_name = str(loaded.dataset.gene_names[gene_index])
    resolved_label_key, resolved_label, cell_indices, available_labels = (
        _resolve_label_selection(
            loaded,
            prior_source=prior_source,
            label_key=label_key,
            label=label,
        )
    )
    counts, totals, raw_summary = summarize_gene_expression(
        loaded,
        gene_index,
        cell_indices=cell_indices,
    )
    checkpoint_summary = build_checkpoint_summary(state)
    available_label_keys = loaded.label_keys
    if mode == "raw":
        return GeneAnalysis(
            cache_key=cache_key,
            gene_name=gene_name,
            gene_index=gene_index,
            source="raw-only",
            mode="raw",
            prior_source=prior_source,
            label_key=resolved_label_key,
            label=resolved_label,
            n_cells=int(counts.shape[0]),
            reference_gene_count=0,
            reference_source="none",
            raw_summary=raw_summary,
            counts=counts,
            totals=totals,
            reference_counts=np.ones_like(counts, dtype=np.float64),
            raw_proxy=np.asarray(counts, dtype=np.float64),
            prior=None,
            checkpoint_prior=None,
            posterior=None,
            fit_result=None,
            checkpoint_summary=checkpoint_summary,
            available_label_keys=available_label_keys,
            available_labels=available_labels,
        )

    if mode == "checkpoint" and loaded.checkpoint is None:
        raise ValueError("checkpoint-backed analysis requires a loaded checkpoint")

    checkpoint_prior = _try_select_checkpoint_prior(
        loaded.checkpoint,
        gene_name=gene_name,
        prior_source=prior_source,
        label=resolved_label,
    )
    reference_source = "checkpoint"
    if mode == "checkpoint" and checkpoint_prior is None:
        raise ValueError(
            f"gene {gene_name!r} is not available for checkpoint posterior analysis"
        )
    if loaded.checkpoint is None:
        reference_source = "dataset"
    reference_positions = _resolve_reference_positions(
        loaded,
        target_gene_index=gene_index,
        source=reference_source,
    )
    reference_counts = compute_reference_counts(
        loaded.dataset.matrix,
        reference_positions,
        cell_indices=cell_indices,
    )
    counts, totals, reference_counts, raw_summary = _filter_positive_reference_cells(
        counts=counts,
        totals=totals,
        reference_counts=reference_counts,
    )
    prior = checkpoint_prior
    fit_result: PriorFitResult | None = None
    source = f"checkpoint/{prior_source}"

    if mode == "fit":
        resolved_fit_params = GeneFitParams() if fit_params is None else fit_params
        fit_reference_source = resolved_fit_params.reference_source
        if fit_reference_source == "checkpoint" and loaded.checkpoint is None:
            fit_reference_source = "dataset"
        reference_positions = _resolve_reference_positions(
            loaded,
            target_gene_index=gene_index,
            source=fit_reference_source,
        )
        reference_counts = compute_reference_counts(
            loaded.dataset.matrix,
            reference_positions,
            cell_indices=cell_indices,
        )
        counts, totals, reference_counts, raw_summary = (
            _filter_positive_reference_cells(
                counts=counts,
                totals=totals,
                reference_counts=reference_counts,
            )
        )
        scale = _resolve_fit_scale(
            checkpoint=None
            if loaded.checkpoint is None
            else loaded.checkpoint.checkpoint,
            prior_source=prior_source,
            label=resolved_label,
            reference_counts=reference_counts,
            explicit_scale=resolved_fit_params.scale,
        )
        observation_batch = ObservationBatch(
            gene_names=[gene_name],
            counts=np.asarray(counts[:, None], dtype=np.float64),
            reference_counts=np.asarray(reference_counts, dtype=np.float64),
        )
        fit_result = fit_gene_priors(
            observation_batch,
            scale=scale,
            config=resolved_fit_params.to_fit_config(),
            device=resolved_fit_params.device,
            torch_dtype=resolved_fit_params.torch_dtype,
            compile_model=resolved_fit_params.compile_model,
        )
        prior = fit_result.prior.select_genes(gene_name)
        reference_source = fit_reference_source
        source = f"fit/{prior_source}"

    resolved_prior = cast(PriorGrid, prior)
    posterior = _run_posterior(
        counts=counts,
        reference_counts=reference_counts,
        prior=resolved_prior,
        checkpoint=loaded.checkpoint,
        fit_params=fit_params,
        gene_name=gene_name,
    )
    raw_proxy = _compute_raw_proxy(counts, reference_counts, resolved_prior.scale)
    return GeneAnalysis(
        cache_key=cache_key,
        gene_name=gene_name,
        gene_index=gene_index,
        source=source,
        mode=mode,
        prior_source=prior_source,
        label_key=resolved_label_key,
        label=resolved_label,
        n_cells=int(counts.shape[0]),
        reference_gene_count=len(reference_positions),
        reference_source=reference_source,
        raw_summary=raw_summary,
        counts=np.asarray(counts, dtype=np.float64),
        totals=np.asarray(totals, dtype=np.float64),
        reference_counts=np.asarray(reference_counts, dtype=np.float64),
        raw_proxy=raw_proxy,
        prior=resolved_prior,
        checkpoint_prior=checkpoint_prior,
        posterior=posterior,
        fit_result=fit_result,
        checkpoint_summary=checkpoint_summary,
        available_label_keys=available_label_keys,
        available_labels=available_labels,
    )


def _compute_kbulk_uncached(
    *,
    loaded: LoadedState,
    gene_index: int,
    class_key: str,
    label: str | None,
    params: KBulkParams,
    cache_key: str,
) -> KBulkAnalysis:
    assert loaded.checkpoint is not None
    values = np.asarray(loaded.dataset.label_values[class_key], dtype=str)
    gene_name = str(loaded.dataset.gene_names[gene_index])
    counts_all = slice_gene_counts(loaded.dataset.matrix, gene_index)
    reference_positions = _resolve_reference_positions(
        loaded,
        target_gene_index=gene_index,
        source="checkpoint",
    )
    reference_counts = compute_reference_counts(
        loaded.dataset.matrix,
        reference_positions,
    )
    unique_labels, label_counts = np.unique(values, return_counts=True)
    order = np.argsort(label_counts)[::-1]
    ordered_labels = unique_labels[order]
    groups: list[KBulkGroupSummary] = []
    rng = np.random.default_rng(params.sample_seed)

    for group_label in ordered_labels.tolist():
        if len(groups) >= params.max_classes:
            break
        if label is not None and group_label != label:
            continue
        cell_indices = np.flatnonzero(values == group_label)
        if cell_indices.shape[0] < params.k:
            continue
        prior = _select_checkpoint_prior(
            loaded.checkpoint,
            gene_name=gene_name,
            prior_source=params.kbulk_prior_source,
            label=group_label if params.kbulk_prior_source == "label" else None,
        )
        positive_mask = np.asarray(reference_counts[cell_indices], dtype=np.float64) > 0
        cell_indices = cell_indices[positive_mask]
        if cell_indices.shape[0] < params.k:
            continue
        cell_reference = np.asarray(reference_counts[cell_indices], dtype=np.float64)
        cell_effective_exposure = effective_exposure(cell_reference, prior.scale)
        realized = _resolve_realized_kbulk_samples(
            n_cells=int(cell_indices.shape[0]),
            k=params.k,
            requested=params.n_samples,
        )
        signal_values = np.empty(realized, dtype=np.float64)
        entropy_values = np.empty(realized, dtype=np.float64)
        offset = 0
        while offset < realized:
            batch_size = min(params.sample_batch_size, realized - offset)
            combos = np.vstack(
                [
                    rng.choice(cell_indices, size=params.k, replace=False)
                    for _ in range(batch_size)
                ]
            )
            flat = combos.reshape(-1)
            chunk_counts = np.asarray(counts_all[flat], dtype=np.float64).reshape(
                batch_size,
                params.k,
            )
            chunk_effective_exposure = np.asarray(
                cell_effective_exposure[np.searchsorted(cell_indices, flat)],
                dtype=np.float64,
            ).reshape(batch_size, params.k)
            result = infer_kbulk(
                KBulkBatch(
                    gene_names=[gene_name],
                    counts=chunk_counts.sum(axis=1, dtype=np.float64)[:, None],
                    effective_exposure=chunk_effective_exposure.sum(
                        axis=1, dtype=np.float64
                    ),
                ),
                prior,
                device=params.device,
                include_posterior=False,
                torch_dtype=params.torch_dtype,
                posterior_distribution=loaded.checkpoint.posterior_distribution,
                nb_overdispersion=loaded.checkpoint.nb_overdispersion,
                compile_model=params.compile_model,
            )
            map_support = np.asarray(result.map_support[:, 0], dtype=np.float64)
            if prior.support_domain == "rate":
                signal_values[offset : offset + batch_size] = map_support
            else:
                signal_values[offset : offset + batch_size] = map_support * float(
                    prior.scale
                )
            entropy_values[offset : offset + batch_size] = np.asarray(
                result.posterior_entropy[:, 0],
                dtype=np.float64,
            )
            offset += batch_size

        groups.append(
            KBulkGroupSummary(
                label=str(group_label),
                n_cells=int(cell_indices.shape[0]),
                realized_samples=realized,
                mean_signal=float(np.mean(signal_values)),
                std_signal=float(np.std(signal_values)),
                mean_entropy=float(np.mean(entropy_values)),
                std_entropy=float(np.std(entropy_values)),
                signal_values=signal_values,
                entropy_values=entropy_values,
            )
        )

    if not groups:
        raise ValueError("no eligible classes available for kBulk analysis")

    return KBulkAnalysis(
        cache_key=cache_key,
        gene_name=gene_name,
        gene_index=gene_index,
        class_key=class_key,
        prior_source=params.kbulk_prior_source,
        k=params.k,
        n_samples=params.n_samples,
        sample_seed=params.sample_seed,
        max_classes=params.max_classes,
        sample_batch_size=params.sample_batch_size,
        groups=groups,
        available_class_keys=loaded.label_keys,
    )


def _resolve_realized_kbulk_samples(
    *,
    n_cells: int,
    k: int,
    requested: int,
) -> int:
    max_unique = comb(n_cells, k) if n_cells <= 32 else requested
    return max(1, min(int(requested), int(max_unique)))


def _resolve_gene_index(loaded: LoadedState, query: str) -> int:
    return resolve_gene_query(
        query,
        loaded.dataset.gene_names,
        loaded.dataset.gene_names_lower,
        loaded.dataset.gene_to_idx,
        loaded.dataset.gene_lower_to_idx,
    )


def _candidate_from_index(loaded: LoadedState, idx: int) -> GeneCandidate:
    detected_cells = int(loaded.dataset.gene_detected_counts[idx])
    return GeneCandidate(
        gene_name=str(loaded.dataset.gene_names[idx]),
        gene_index=int(idx),
        total_count=int(round(float(loaded.dataset.gene_total_counts[idx]))),
        detected_cells=detected_cells,
        detected_fraction=float(detected_cells / max(loaded.n_cells, 1)),
    )


def _normalize_sort_by(value: str) -> BrowseSort:
    if value in {"total_umi", "total_count"}:
        return "total_count"
    if value in {"detected_cells", "detected_fraction", "gene_name", "gene_index"}:
        return cast(BrowseSort, value)
    return "total_count"


def _normalize_scope(value: str, *, has_checkpoint: bool) -> BrowseScope:
    if value == "auto":
        return "fitted" if has_checkpoint else "all"
    if value in {"fitted", "all"}:
        return cast(BrowseScope, value)
    return "fitted" if has_checkpoint else "all"


def _filter_gene_indices(
    loaded: LoadedState,
    *,
    query: str,
    scope: BrowseScope,
) -> list[int]:
    token = query.strip().lower()
    fitted = None
    if scope == "fitted" and loaded.checkpoint is not None:
        fitted = set(loaded.checkpoint.checkpoint.gene_names)
    result: list[int] = []
    for idx, gene_name in enumerate(loaded.dataset.gene_names_lower):
        if token and token not in gene_name:
            continue
        if fitted is not None and str(loaded.dataset.gene_names[idx]) not in fitted:
            continue
        result.append(int(idx))
    return result


def _sort_gene_indices(
    loaded: LoadedState,
    indices: list[int],
    *,
    sort_by: BrowseSort,
    descending: bool,
) -> list[int]:
    if sort_by == "detected_cells":
        key = lambda idx: int(loaded.dataset.gene_detected_counts[idx])
    elif sort_by == "detected_fraction":
        key = lambda idx: (
            float(loaded.dataset.gene_detected_counts[idx])
            / max(
                loaded.n_cells,
                1,
            )
        )
    elif sort_by == "gene_name":
        key = lambda idx: str(loaded.dataset.gene_names[idx]).lower()
    elif sort_by == "gene_index":
        key = lambda idx: int(idx)
    else:
        key = lambda idx: float(loaded.dataset.gene_total_counts[idx])
    return sorted(indices, key=key, reverse=descending)


def _resolve_label_selection(
    loaded: LoadedState,
    *,
    prior_source: PriorSource,
    label_key: str | None,
    label: str | None,
) -> tuple[str | None, str | None, np.ndarray | None, tuple[str, ...]]:
    if prior_source != "label":
        return None, None, None, ()
    resolved_key = _resolve_class_key(loaded, label_key)
    values = np.asarray(loaded.dataset.label_values[resolved_key], dtype=str)
    available_labels = tuple(np.unique(values).tolist())
    if not available_labels:
        raise ValueError(f"label key {resolved_key!r} has no available labels")
    resolved_label = label if label in available_labels else available_labels[0]
    cell_indices = np.flatnonzero(values == resolved_label).astype(np.int64)
    if cell_indices.size == 0:
        raise ValueError(
            f"label {resolved_label!r} under {resolved_key!r} matched zero cells"
        )
    return resolved_key, resolved_label, cell_indices, available_labels


def _resolve_class_key(loaded: LoadedState, label_key: str | None) -> str:
    if label_key is not None and label_key in loaded.dataset.label_values:
        return label_key
    if (
        loaded.checkpoint is not None
        and loaded.checkpoint.suggested_label_key is not None
        and loaded.checkpoint.suggested_label_key in loaded.dataset.label_values
    ):
        return loaded.checkpoint.suggested_label_key
    if loaded.label_keys:
        return loaded.label_keys[0]
    raise ValueError("dataset does not expose any label columns")


def _resolve_reference_positions(
    loaded: LoadedState,
    *,
    target_gene_index: int,
    source: str,
) -> list[int]:
    if source == "dataset":
        positions = [
            idx for idx in range(loaded.n_genes) if idx != int(target_gene_index)
        ]
        if not positions:
            raise ValueError("dataset does not have enough genes for reference counts")
        return positions
    if loaded.checkpoint is None:
        raise ValueError("checkpoint reference source requires a loaded checkpoint")
    positions = [
        int(idx)
        for idx in loaded.checkpoint.reference_positions
        if int(idx) != int(target_gene_index)
    ]
    if not positions:
        positions = [int(idx) for idx in loaded.checkpoint.reference_positions]
    if not positions:
        raise ValueError("checkpoint does not contain usable reference genes")
    return positions


def _select_checkpoint_prior(
    checkpoint_state: CheckpointState,
    *,
    gene_name: str,
    prior_source: PriorSource,
    label: str | None,
) -> PriorGrid:
    checkpoint = checkpoint_state.checkpoint
    if gene_name not in checkpoint.gene_names:
        raise ValueError(f"gene {gene_name!r} is not available in checkpoint priors")
    if prior_source == "global":
        if checkpoint.prior is None:
            raise ValueError("checkpoint is missing a global prior")
        return checkpoint.prior.select_genes(gene_name)
    if label is None:
        raise ValueError("label prior source requires a selected label")
    if label not in checkpoint.label_priors:
        raise ValueError(
            f"checkpoint does not contain label prior {label!r}; "
            f"available labels: {', '.join(sorted(checkpoint.label_priors)[:8])}"
        )
    return checkpoint.label_priors[label].select_genes(gene_name)


def _resolve_fit_scale(
    *,
    checkpoint: ModelCheckpoint | None,
    prior_source: PriorSource,
    label: str | None,
    reference_counts: np.ndarray,
    explicit_scale: float | None,
) -> float:
    if explicit_scale is not None:
        return float(explicit_scale)
    if checkpoint is not None:
        scale_metadata = checkpoint.get_scale_metadata(
            label if prior_source == "label" else None
        )
        if scale_metadata is not None:
            return float(scale_metadata.scale)
        if (
            prior_source == "label"
            and label is not None
            and label in checkpoint.label_priors
        ):
            return float(checkpoint.label_priors[label].scale)
        if checkpoint.prior is not None:
            return float(checkpoint.prior.scale)
    return float(mean_reference_count(reference_counts))


def _run_posterior(
    *,
    counts: np.ndarray,
    reference_counts: np.ndarray,
    prior: PriorGrid,
    checkpoint: CheckpointState | None,
    fit_params: GeneFitParams | None,
    gene_name: str,
) -> PosteriorSummary:
    observation_batch = ObservationBatch(
        gene_names=[gene_name],
        counts=np.asarray(counts[:, None], dtype=np.float64),
        reference_counts=np.asarray(reference_counts, dtype=np.float64),
    )
    if fit_params is None:
        device = "cpu"
        torch_dtype = "float64"
        compile_model = True
    else:
        device = fit_params.device
        torch_dtype = fit_params.torch_dtype
        compile_model = fit_params.compile_model
    posterior_distribution = (
        checkpoint.posterior_distribution if checkpoint is not None else "auto"
    )
    nb_overdispersion = (
        checkpoint.nb_overdispersion
        if checkpoint is not None
        else (0.01 if fit_params is None else fit_params.nb_overdispersion)
    )
    return Posterior(
        [gene_name],
        prior,
        device=device,
        torch_dtype=torch_dtype,
        posterior_distribution=posterior_distribution,
        nb_overdispersion=nb_overdispersion,
        compile_model=compile_model,
    ).summarize(observation_batch)


def _compute_raw_proxy(
    counts: np.ndarray,
    reference_counts: np.ndarray,
    scale: float,
) -> np.ndarray:
    reference_mean = max(float(mean_reference_count(reference_counts)), 1e-12)
    return np.asarray(counts, dtype=np.float64) * float(scale) / reference_mean


def _try_select_checkpoint_prior(
    checkpoint_state: CheckpointState | None,
    *,
    gene_name: str,
    prior_source: PriorSource,
    label: str | None,
) -> PriorGrid | None:
    if checkpoint_state is None:
        return None
    try:
        return _select_checkpoint_prior(
            checkpoint_state,
            gene_name=gene_name,
            prior_source=prior_source,
            label=label,
        )
    except ValueError:
        return None


def _filter_positive_reference_cells(
    *,
    counts: np.ndarray,
    totals: np.ndarray,
    reference_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, GeneSummary]:
    mask = np.asarray(reference_counts, dtype=np.float64) > 0
    if not np.any(mask):
        raise ValueError("no cells have positive reference counts for this analysis")
    counts_filtered = np.asarray(counts[mask], dtype=np.float64)
    totals_filtered = np.asarray(totals[mask], dtype=np.float64)
    reference_filtered = np.asarray(reference_counts[mask], dtype=np.float64)
    detected = int(np.count_nonzero(counts_filtered > 0))
    if (
        counts_filtered.size < 2
        or np.allclose(counts_filtered, counts_filtered[0])
        or np.allclose(
            totals_filtered,
            totals_filtered[0],
        )
    ):
        corr = 0.0
    else:
        corr = float(np.corrcoef(counts_filtered, totals_filtered)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    summary = GeneSummary(
        total_count=float(np.sum(counts_filtered)),
        mean_count=float(np.mean(counts_filtered)),
        median_count=float(np.median(counts_filtered)),
        p90_count=float(np.percentile(counts_filtered, 90)),
        p99_count=float(np.percentile(counts_filtered, 99)),
        max_count=float(np.max(counts_filtered)),
        detected_cells=detected,
        detected_fraction=float(detected / max(counts_filtered.shape[0], 1)),
        zero_fraction=float(np.mean(counts_filtered <= 0)),
        count_total_correlation=corr,
    )
    return counts_filtered, totals_filtered, reference_filtered, summary


if TYPE_CHECKING:
    from ..state import AppState, LoadedState
