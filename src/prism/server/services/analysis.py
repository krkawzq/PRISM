from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
)

from prism.baseline.metrics import (
    evaluate_representations,
    log1p_normalize_total,
    normalize_total,
    raw_umi,
)
from prism.model import (
    GeneBatch,
    GridDistribution,
    PoolFitReport,
    Posterior,
    PriorEngineSetting,
    PriorEngineTrainingConfig,
    PriorFitReport,
    fit_pool_scale_report,
)
from prism.model._typing import OptimizerName, SchedulerName

from ..state import AppState, LoadedState
from .datasets import (
    GeneCandidate,
    GeneNotFoundError,
    resolve_gene_query,
    search_gene_candidates as search_gene_candidates_raw,
    slice_gene_counts,
)


console = Console()


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
    treatment_table: list[dict[str, object]]


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
class GeneAnalysis:
    cache_key: str
    gene_name: str
    gene_index: int
    source: str
    s_hat: float
    counts: np.ndarray
    totals: np.ndarray
    x_eff: np.ndarray
    signal: np.ndarray
    confidence: np.ndarray
    prior_entropy: np.ndarray
    mutual_information: np.ndarray
    support: np.ndarray
    prior_weights: np.ndarray
    posterior_samples: np.ndarray
    posterior_cell_indices: np.ndarray
    summary: GeneSummary
    representations: dict[str, np.ndarray]
    representation_metrics: dict[str, RepresentationMetric]
    pool_report: PoolFitReport | None = None
    prior_report: PriorFitReport | None = None
    fit_params: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class GeneFitParams:
    r: float = 0.05
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


def build_dataset_summary(state: AppState) -> dict[str, object]:
    loaded = state.require_loaded()
    cache_key = state.make_cache_key("summary", "dataset_summary")
    cached = state.get_cache("summary", cache_key)
    if cached is not None:
        print("[prism-server] dataset summary cache hit", flush=True)
        return cast(dict[str, object], cached)
    summary = {
        "n_cells": loaded.n_cells,
        "n_genes": loaded.n_genes,
        "fitted_genes": len(loaded.fitted_gene_names),
        "layer": loaded.dataset.layer or "X",
        "h5ad_path": str(loaded.dataset.h5ad_path),
        "ckpt_path": ""
        if loaded.model.ckpt_path is None
        else str(loaded.model.ckpt_path),
        "s_hat": default_s_hat(state, loaded),
        "mean_total": float(np.mean(loaded.dataset.totals)),
        "median_total": float(np.median(loaded.dataset.totals)),
        "model_source": loaded.model.source,
    }
    state.set_cache("summary", cache_key, summary)
    print("[prism-server] dataset summary cache miss -> stored", flush=True)
    return summary


def top_fitted_genes(state: AppState, limit: int | None = None) -> list[GeneCandidate]:
    loaded = state.require_loaded()
    limit = state.config.top_gene_limit if limit is None else limit
    cache_key = state.make_cache_key("top_genes", limit)
    cached = state.get_cache("top_genes", cache_key)
    if cached is not None:
        print(f"[prism-server] top genes cache hit limit={limit}", flush=True)
        return cast(list[GeneCandidate], cached)

    ranked = (
        loaded.ranked_fitted_gene_indices
        if loaded.ranked_fitted_gene_indices.size > 0
        else loaded.dataset.ranked_gene_indices
    )
    candidates = [_candidate_from_index(loaded, int(idx)) for idx in ranked[:limit]]
    state.set_cache("top_genes", cache_key, candidates)
    return candidates


def search_gene_candidates(
    state: AppState, query: str, limit: int | None = None
) -> list[GeneCandidate]:
    loaded = state.require_loaded()
    limit = state.config.top_gene_limit if limit is None else limit
    cache_key = state.make_cache_key("search", query, limit)
    cached = state.get_cache("search", cache_key)
    if cached is not None:
        print(
            f"[prism-server] search cache hit query={query!r} limit={limit}",
            flush=True,
        )
        return cast(list[GeneCandidate], cached)

    gene_names = loaded.dataset.gene_names
    gene_names_lower = loaded.dataset.gene_names_lower
    gene_total_counts = loaded.dataset.gene_total_counts
    gene_detected_counts = loaded.dataset.gene_detected_counts
    ranked_indices = loaded.dataset.ranked_gene_indices
    if loaded.fitted_gene_indices.size > 0:
        gene_names = gene_names[loaded.fitted_gene_indices]
        gene_names_lower = tuple(
            loaded.dataset.gene_names_lower[int(idx)]
            for idx in loaded.fitted_gene_indices.tolist()
        )
        gene_total_counts = gene_total_counts[loaded.fitted_gene_indices]
        gene_detected_counts = gene_detected_counts[loaded.fitted_gene_indices]
        ranked_indices = np.arange(loaded.fitted_gene_indices.size, dtype=np.int64)

    candidates = search_gene_candidates_raw(
        query=query,
        gene_names=gene_names,
        gene_names_lower=gene_names_lower,
        gene_total_counts=gene_total_counts,
        gene_detected_counts=gene_detected_counts,
        ranked_indices=ranked_indices,
        n_cells=loaded.n_cells,
        limit=limit,
    )
    resolved = [
        GeneCandidate(
            gene_name=item.gene_name,
            gene_index=int(loaded.dataset.gene_to_idx[item.gene_name]),
            total_umi=item.total_umi,
            detected_cells=item.detected_cells,
            detected_fraction=item.detected_fraction,
        )
        for item in candidates
    ]
    state.set_cache("search", cache_key, resolved)
    return resolved


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
        state.config.gene_browser_page_size if page_size is None else max(1, page_size)
    )
    resolved_scope = scope if scope in {"auto", "all", "fitted"} else "auto"
    resolved_sort = (
        sort_by
        if sort_by
        in {
            "gene_name",
            "gene_index",
            "total_umi",
            "detected_cells",
            "detected_fraction",
        }
        else "total_umi"
    )
    resolved_page = max(1, page)
    cache_key = state.make_cache_key(
        "browse_v2_fulltable",
        query,
        resolved_sort,
        int(descending),
        resolved_page,
        resolved_page_size,
        resolved_scope,
    )
    cached = state.get_cache("search", cache_key)
    if cached is not None:
        print(
            "[prism-server] gene browser cache hit "
            f"query={query!r} sort={resolved_sort} desc={descending} page={resolved_page}",
            flush=True,
        )
        return cast(GeneBrowsePage, cached)

    use_fitted = resolved_scope == "fitted" or (
        resolved_scope == "auto" and loaded.fitted_gene_indices.size > 0
    )
    indices = (
        loaded.fitted_gene_indices.copy()
        if use_fitted
        else np.arange(loaded.n_genes, dtype=np.int64)
    )
    token = query.strip().lower()
    if token:
        indices = np.asarray(
            [
                int(idx)
                for idx in indices.tolist()
                if token in loaded.dataset.gene_names_lower[int(idx)]
            ],
            dtype=np.int64,
        )

    items = [_candidate_from_index(loaded, int(idx)) for idx in indices.tolist()]
    items.sort(
        key=lambda item: cast(
            str | int | float, _gene_browse_sort_key(item, resolved_sort)
        ),
        reverse=descending,
    )
    total_items = len(items)
    total_pages = max(1, (total_items + resolved_page_size - 1) // resolved_page_size)
    clamped_page = min(resolved_page, total_pages)
    start = (clamped_page - 1) * resolved_page_size
    end = start + resolved_page_size
    result = GeneBrowsePage(
        items=items[start:end],
        page=clamped_page,
        page_size=resolved_page_size,
        total_items=total_items,
        total_pages=total_pages,
        query=query,
        sort_by=resolved_sort,
        descending=descending,
        scope="fitted" if use_fitted else "all",
    )
    state.set_cache("search", cache_key, result)
    print(
        "[prism-server] gene browser cache miss -> stored "
        f"query={query!r} total={total_items} sort={resolved_sort} desc={descending} "
        f"page={clamped_page}/{total_pages} scope={result.scope}",
        flush=True,
    )
    return result


def _gene_browse_sort_key(candidate: GeneCandidate, sort_by: str) -> str | int | float:
    if sort_by == "gene_name":
        return candidate.gene_name.lower()
    if sort_by == "gene_index":
        return candidate.gene_index
    if sort_by == "detected_cells":
        return candidate.detected_cells
    if sort_by == "detected_fraction":
        return candidate.detected_fraction
    return candidate.total_umi


def analyze_gene(
    state: AppState, query: str, fit_params: GeneFitParams | None = None
) -> GeneAnalysis:
    loaded = state.require_loaded()
    print(f"[prism-server] analyze gene query={query!r}", flush=True)
    gene_index = resolve_gene_query(
        query,
        loaded.dataset.gene_names,
        loaded.dataset.gene_names_lower,
        loaded.dataset.gene_to_idx,
        loaded.dataset.gene_lower_to_idx,
    )
    gene_name = str(loaded.dataset.gene_names[gene_index])
    print(
        f"[prism-server] resolved gene name={gene_name} index={gene_index} fit_params={'default' if fit_params is None else 'custom'}",
        flush=True,
    )
    analysis_key = _analysis_cache_key(loaded, gene_name, fit_params)
    cached_analysis = state.get_cache("analysis", analysis_key)
    if cached_analysis is not None:
        print(f"[prism-server] gene analysis cache hit gene={gene_name}", flush=True)
        return cast(GeneAnalysis, cached_analysis)
    print(f"[prism-server] gene analysis cache miss gene={gene_name}", flush=True)

    def factory() -> GeneAnalysis:
        if (
            fit_params is None
            and loaded.model.engine is not None
            and loaded.model.engine.is_fitted(gene_name)
        ):
            print(
                f"[prism-server] using checkpoint prior for gene={gene_name} source={loaded.model.source}",
                flush=True,
            )
            priors = loaded.model.engine.get_priors(gene_name)
            if priors is None:
                raise ValueError(f"failed to read prior for gene {gene_name!r}")
            pool_report = ensure_pool_report(state, loaded)
            return _build_analysis(
                loaded=loaded,
                gene_index=gene_index,
                priors=priors,
                s_hat=default_s_hat(state, loaded),
                source=loaded.model.source,
                chunk_size=state.config.analysis_chunk_size,
                pool_report=pool_report,
                cache_key=analysis_key,
            )

        resolved_fit_params = GeneFitParams() if fit_params is None else fit_params
        fit_key = _fit_cache_key(loaded, gene_name, resolved_fit_params)
        cached_fit = cast(
            dict[str, Any],
            state.get_or_create_cache(
                "fit",
                fit_key,
                lambda: _fit_gene_once(
                    state, loaded, gene_index, gene_name, resolved_fit_params
                ),
            ),
        )
        return _build_analysis(
            loaded=loaded,
            gene_index=gene_index,
            priors=cast(GridDistribution, cached_fit["priors"]),
            s_hat=float(cached_fit["s_hat"]),
            source="on-demand-fit",
            fit_params=cast(dict[str, object], cached_fit["fit_params"]),
            chunk_size=state.config.analysis_chunk_size,
            pool_report=cast(PoolFitReport | None, cached_fit.get("pool_report")),
            prior_report=cast(PriorFitReport | None, cached_fit.get("prior_report")),
            cache_key=analysis_key,
        )

    return cast(
        GeneAnalysis, state.get_or_create_cache("analysis", analysis_key, factory)
    )


def ensure_pool_report(
    state: AppState,
    loaded: LoadedState,
) -> PoolFitReport:
    if loaded.model.pool_report is not None:
        print("[prism-server] using pool report from checkpoint", flush=True)
        return loaded.model.pool_report

    report = cast(
        PoolFitReport,
        state.get_or_create_cache(
            "summary",
            state.make_cache_key("summary", "pool_report"),
            lambda: _fit_pool_report_once(state, loaded),
        ),
    )
    state.set_pool_report(report)
    return report


def default_s_hat(state: AppState, loaded: LoadedState) -> float:
    if loaded.model.s_hat is not None:
        return float(loaded.model.s_hat)
    return float(
        pool_s_hat(
            ensure_pool_report(state, loaded), _resolve_pool_r(state, loaded, None)
        )
    )


def pool_s_hat(report: PoolFitReport, r: float) -> float:
    return float(report.point_eta / r)


def _resolve_pool_r(state: AppState, loaded: LoadedState, r: float | None) -> float:
    if r is not None:
        return float(r)
    if loaded.model.r_hint is not None:
        return float(loaded.model.r_hint)
    if loaded.model.checkpoint is not None:
        checkpoint_r = loaded.model.checkpoint.get("r")
        if isinstance(checkpoint_r, (int, float)) and checkpoint_r > 0:
            return float(checkpoint_r)
    return float(state.config.pool_r)


def _build_analysis(
    *,
    loaded: LoadedState,
    gene_index: int,
    priors: GridDistribution,
    s_hat: float,
    source: str,
    chunk_size: int,
    cache_key: str,
    pool_report: PoolFitReport | None = None,
    prior_report: PriorFitReport | None = None,
    fit_params: dict[str, object] | None = None,
) -> GeneAnalysis:
    gene_name = str(loaded.dataset.gene_names[gene_index])
    print(
        f"[prism-server] build analysis gene={gene_name} source={source} s_hat={s_hat:.4f}",
        flush=True,
    )
    counts = slice_gene_counts(loaded.dataset.matrix, gene_index)
    posterior = Posterior([gene_name], priors)

    signal = np.empty(loaded.n_cells, dtype=np.float64)
    confidence = np.empty(loaded.n_cells, dtype=np.float64)
    prior_entropy = np.empty(loaded.n_cells, dtype=np.float64)
    mutual_information = np.empty(loaded.n_cells, dtype=np.float64)
    for cell_slice in _iter_cell_slices(loaded.n_cells, chunk_size):
        batch = GeneBatch(
            gene_names=[gene_name],
            counts=counts[cell_slice, None],
            totals=loaded.dataset.totals[cell_slice],
        )
        result = posterior.extract(
            batch,
            s_hat=s_hat,
            channels={
                "signal",
                "posterior_entropy",
                "prior_entropy",
                "mutual_information",
            },
        )
        signal[cell_slice] = result["signal"][0]
        confidence[cell_slice] = result["posterior_entropy"][0]
        prior_entropy[cell_slice] = result["prior_entropy"][0]
        mutual_information[cell_slice] = result["mutual_information"][0]

    representative_idx = _representative_indices(signal, n=12)
    representative_batch = GeneBatch(
        gene_names=[gene_name],
        counts=counts[representative_idx, None],
        totals=loaded.dataset.totals[representative_idx],
    )
    representative = posterior.extract(
        representative_batch,
        s_hat=s_hat,
        channels={
            "posterior",
            "signal",
            "posterior_entropy",
            "prior_entropy",
            "mutual_information",
        },
    )

    x_eff = counts / np.maximum(loaded.dataset.totals, 1.0) * s_hat
    counts_matrix = counts[:, None]
    representations = {
        "X": raw_umi(counts_matrix)[:, 0],
        "NormalizeTotalX": normalize_total(counts_matrix, loaded.dataset.totals)[:, 0],
        "log1pX": np.log1p(raw_umi(counts_matrix)[:, 0]),
        "Log1pNormalizeTotalX": log1p_normalize_total(
            counts_matrix, loaded.dataset.totals
        )[:, 0],
        "signal": signal,
    }
    group_labels = _group_labels(loaded)
    representation_metrics = {
        name: _representation_metric_from_dict(metrics)
        for name, metrics in evaluate_representations(
            representations,
            totals=loaded.dataset.totals,
            raw_counts=counts,
            labels=group_labels,
            zero_fraction=loaded.dataset.cell_zero_fraction,
        ).items()
    }
    print(
        f"[prism-server] analysis ready gene={gene_name} detected_frac={float(np.mean(counts > 0)):.4f} mean_signal={float(np.mean(signal)):.4f} mean_conf={float(np.mean(confidence)):.4f}",
        flush=True,
    )

    return GeneAnalysis(
        cache_key=cache_key,
        gene_name=gene_name,
        gene_index=gene_index,
        source=source,
        s_hat=s_hat,
        counts=counts,
        totals=loaded.dataset.totals,
        x_eff=x_eff,
        signal=signal,
        confidence=confidence,
        prior_entropy=prior_entropy,
        mutual_information=mutual_information,
        support=representative["support"][0],
        prior_weights=representative["prior_weights"][0],
        posterior_samples=representative["posterior"][0],
        posterior_cell_indices=representative_idx,
        summary=summarize_gene_expression(
            loaded,
            gene_index,
            counts=counts,
            totals=loaded.dataset.totals,
        ),
        representations=representations,
        representation_metrics=representation_metrics,
        pool_report=pool_report,
        prior_report=prior_report,
        fit_params=fit_params,
    )


def _fit_cache_key(
    loaded: LoadedState, gene_name: str, fit_params: GeneFitParams
) -> str:
    payload = {
        "context": loaded.context_key,
        "gene": gene_name,
        **_fit_params_dict(fit_params),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _analysis_cache_key(
    loaded: LoadedState, gene_name: str, fit_params: GeneFitParams | None
) -> str:
    payload: dict[str, object] = {
        "context": loaded.context_key,
        "gene": gene_name,
        "mode": "checkpoint" if fit_params is None else "fit",
    }
    if fit_params is not None:
        payload.update(_fit_params_dict(fit_params))
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _fit_params_dict(fit_params: GeneFitParams) -> dict[str, object]:
    return asdict(fit_params)


def _iter_cell_slices(n_cells: int, chunk_size: int) -> list[slice]:
    return [
        slice(start, min(start + chunk_size, n_cells))
        for start in range(0, n_cells, chunk_size)
    ]


def _representative_indices(values: np.ndarray, n: int) -> np.ndarray:
    if values.size <= n:
        return np.arange(values.size, dtype=np.int64)
    order = np.argsort(values)
    anchors = np.linspace(0, len(order) - 1, n).round().astype(np.int64)
    return np.unique(order[anchors])


def _candidate_from_index(loaded: LoadedState, gene_index: int) -> GeneCandidate:
    return GeneCandidate(
        gene_name=str(loaded.dataset.gene_names[gene_index]),
        gene_index=int(gene_index),
        total_umi=int(round(float(loaded.dataset.gene_total_counts[gene_index]))),
        detected_cells=int(loaded.dataset.gene_detected_counts[gene_index]),
        detected_fraction=float(loaded.dataset.gene_detected_counts[gene_index])
        / max(loaded.n_cells, 1),
    )


def _group_labels(loaded: LoadedState) -> np.ndarray | None:
    obs = loaded.dataset.adata.obs
    for key in ("treatment", "cell_type", "label", "group"):
        if key in obs.columns:
            values = np.asarray(obs[key]).reshape(-1)
            if np.unique(values).size >= 2:
                return values
    return None


def _representation_metric_from_dict(
    metrics: dict[str, float | None],
) -> RepresentationMetric:
    return RepresentationMetric(
        mean=_required_metric(metrics, "mean"),
        median=_required_metric(metrics, "median"),
        std=_required_metric(metrics, "std"),
        var=_required_metric(metrics, "var"),
        p95=_required_metric(metrics, "p95"),
        nonzero_frac=_required_metric(metrics, "nonzero_frac"),
        depth_corr=_required_metric(metrics, "depth_corr"),
        depth_mi=_required_metric(metrics, "depth_mi"),
        sparsity_corr=metrics.get("sparsity_corr"),
        fisher_ratio=metrics.get("fisher_ratio"),
        kruskal_h=metrics.get("kruskal_h"),
        kruskal_p=metrics.get("kruskal_p"),
        auroc_ovr=metrics.get("auroc_ovr"),
        zero_consistency=metrics.get("zero_consistency"),
        zero_rank_tau=metrics.get("zero_rank_tau"),
        dropout_recovery=metrics.get("dropout_recovery"),
        treatment_cv=metrics.get("treatment_cv"),
    )


def _required_metric(metrics: dict[str, float | None], key: str) -> float:
    value = metrics.get(key)
    return 0.0 if value is None else float(value)


def summarize_gene_expression(
    loaded: LoadedState,
    gene_index: int,
    *,
    counts: np.ndarray | None = None,
    totals: np.ndarray | None = None,
) -> GeneSummary:
    if counts is None:
        counts = slice_gene_counts(loaded.dataset.matrix, gene_index)
    if totals is None:
        totals = loaded.dataset.totals
    detected = counts > 0
    corr = 0.0
    if np.std(counts) > 0 and np.std(totals) > 0:
        corr = float(np.corrcoef(totals, counts)[0, 1])
    treatment_table: list[dict[str, object]] = []
    treatment_labels = loaded.dataset.treatment_labels
    treatment_totals = loaded.dataset.treatment_totals
    if treatment_labels is not None and treatment_totals is not None:
        treatment_table = _treatment_table(
            counts,
            treatment_labels=treatment_labels,
            treatment_totals=treatment_totals,
        )
    return GeneSummary(
        gene_name=str(loaded.dataset.gene_names[gene_index]),
        gene_index=int(gene_index),
        total_counts=float(np.sum(counts)),
        mean_count=float(np.mean(counts)),
        median_count=float(np.median(counts)),
        p90_count=float(np.quantile(counts, 0.9)),
        p99_count=float(np.quantile(counts, 0.99)),
        max_count=float(np.max(counts)),
        detected_cells=int(np.sum(detected)),
        detected_frac=float(np.mean(detected)),
        zero_frac=float(np.mean(~detected)),
        count_total_correlation=corr,
        treatment_table=treatment_table,
    )


def _fit_pool_report_once(state: AppState, loaded: LoadedState) -> PoolFitReport:
    resolved_r = _resolve_pool_r(state, loaded, None)
    print(
        f"[prism-server] fitting pool report from totals cells={loaded.n_cells} r={resolved_r:.4f}",
        flush=True,
    )
    start = perf_counter()
    progress_callback = None
    progress: Progress | None = None
    task_id: TaskID | None = None
    if state.config.show_pool_fit_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            console=console,
            transient=False,
        )
        progress.start()
        task_id = cast(TaskID, progress.add_task("EM pool fit", total=120))

        def _on_progress(
            step: int,
            total: int,
            ll: float,
            mu: float,
            sigma: float,
            done: bool,
        ) -> None:
            nonlocal task_id
            if progress is None or task_id is None:
                return
            total_steps = max(total, 1)
            progress.update(
                task_id,
                total=total_steps,
                completed=step,
                description=(
                    f"EM pool fit | iter={step}/{total_steps} "
                    f"ll={ll:.2f} mu={mu:.4f} sigma={sigma:.4f}"
                ),
            )
            if done:
                progress.refresh()

        progress_callback = _on_progress

    try:
        report = fit_pool_scale_report(
            loaded.dataset.totals,
            use_posterior_mu=False,
            progress_callback=progress_callback,
        )
    finally:
        if progress is not None:
            if task_id is not None:
                progress.update(task_id, completed=progress.tasks[0].total or 120)
            progress.stop()
    print(
        f"[prism-server] pool fit done elapsed={perf_counter() - start:.2f}s mu={report.mu:.6f} sigma={report.sigma:.6f} point_eta={report.point_eta:.4f} inferred_s_hat={pool_s_hat(report, resolved_r):.4f}",
        flush=True,
    )
    return report


def _fit_gene_once(
    state: AppState,
    loaded: LoadedState,
    gene_index: int,
    gene_name: str,
    fit_params: GeneFitParams,
) -> dict[str, Any]:
    print(
        f"[prism-server] cache miss gene={gene_name}; fitting on demand with r={fit_params.r:.4f} grid={fit_params.grid_size} sigma={fit_params.sigma_bins:.3f} iter={fit_params.n_iter}",
        flush=True,
    )
    resolved_r = _resolve_pool_r(state, loaded, fit_params.r)
    pool_report = ensure_pool_report(state, loaded)
    engine = loaded.model.engine_for_fit(
        setting=PriorEngineSetting(
            grid_size=fit_params.grid_size,
            sigma_bins=fit_params.sigma_bins,
            align_loss_weight=fit_params.align_loss_weight,
            torch_dtype=fit_params.torch_dtype,
        ),
        device=fit_params.device,
        gene_names=loaded.dataset.gene_names,
    )
    counts = slice_gene_counts(loaded.dataset.matrix, gene_index)
    batch = GeneBatch(
        gene_names=[gene_name],
        counts=counts[:, None],
        totals=loaded.dataset.totals,
    )
    training_cfg = PriorEngineTrainingConfig(
        lr=fit_params.lr,
        n_iter=fit_params.n_iter,
        lr_min_ratio=fit_params.lr_min_ratio,
        grad_clip=fit_params.grad_clip,
        init_temperature=fit_params.init_temperature,
        cell_chunk_size=fit_params.cell_chunk_size,
        optimizer=fit_params.optimizer,
        scheduler=fit_params.scheduler,
    )
    fit_progress: Progress | None = None
    fit_task_id: TaskID | None = None
    fit_callback = None
    fit_start = perf_counter()
    if state.config.show_pool_fit_progress:
        fit_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            console=console,
            transient=False,
        )
        fit_progress.start()
        fit_task_id = cast(
            TaskID, fit_progress.add_task("Prior fit", total=training_cfg.n_iter)
        )

        def _on_fit_progress(
            step: int,
            total: int,
            total_loss: float,
            nll_value: float,
            align_value: float,
            _best_updated: bool,
        ) -> None:
            if fit_progress is None or fit_task_id is None:
                return
            fit_progress.update(
                fit_task_id,
                total=max(total, 1),
                completed=step,
                description=(
                    f"Prior fit | iter={step}/{max(total, 1)} "
                    f"loss={total_loss:.4f} nll={nll_value:.4f} jsd={align_value:.4f}"
                ),
            )

        fit_callback = _on_fit_progress

    try:
        prior_report = engine.fit_report(
            batch,
            s_hat=pool_s_hat(pool_report, resolved_r),
            training_cfg=training_cfg,
            progress_callback=fit_callback,
        )
    finally:
        if fit_progress is not None:
            if fit_task_id is not None:
                fit_progress.update(fit_task_id, completed=training_cfg.n_iter)
            fit_progress.stop()
    priors = GridDistribution(
        grid_min=float(prior_report.grid_min[0]),
        grid_max=float(prior_report.grid_max[0]),
        weights=np.asarray(prior_report.prior_weights[0], dtype=np.float64),
    )
    print(
        f"[prism-server] fit completed gene={gene_name} elapsed={perf_counter() - fit_start:.2f}s final_loss={prior_report.final_loss:.6f} best_loss={prior_report.best_loss:.6f}",
        flush=True,
    )
    return {
        "priors": priors,
        "s_hat": float(pool_s_hat(pool_report, resolved_r)),
        "pool_report": pool_report,
        "prior_report": prior_report,
        "fit_params": _fit_params_dict(fit_params),
    }


def _treatment_table(
    counts: np.ndarray,
    *,
    treatment_labels: np.ndarray,
    treatment_totals: np.ndarray,
) -> list[dict[str, object]]:
    labels = np.asarray(treatment_labels, dtype=str)
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    cells = np.bincount(inverse)
    total_counts = np.bincount(inverse, weights=counts)
    detected_counts = np.bincount(inverse, weights=(counts > 0).astype(np.float64))
    total_umi = np.bincount(inverse, weights=treatment_totals)
    order = np.argsort(total_counts)[::-1]
    rows: list[dict[str, object]] = []
    for idx in order[:12]:
        cell_count = int(cells[idx])
        if cell_count <= 0:
            continue
        rows.append(
            {
                "treatment": str(unique_labels[idx]),
                "cells": cell_count,
                "total_counts": float(total_counts[idx]),
                "mean_count": float(total_counts[idx] / max(cell_count, 1)),
                "detected_frac": float(detected_counts[idx] / max(cell_count, 1)),
                "mean_total_umi": float(total_umi[idx] / max(cell_count, 1)),
            }
        )
    return rows
