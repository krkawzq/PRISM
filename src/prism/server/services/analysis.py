from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np

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
    surprisal: np.ndarray
    sharpness: np.ndarray
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

    if loaded.fitted_gene_indices.size > 0:
        ranked = np.argsort(
            loaded.dataset.gene_total_counts[loaded.fitted_gene_indices]
        )[::-1][:limit]
        candidates = [
            _candidate_from_index(loaded, int(loaded.fitted_gene_indices[idx]))
            for idx in ranked
        ]
        state.set_cache("top_genes", cache_key, candidates)
        return candidates

    ranked = np.argsort(loaded.dataset.gene_total_counts)[::-1][:limit]
    candidates = [_candidate_from_index(loaded, int(idx)) for idx in ranked]
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
    gene_total_counts = loaded.dataset.gene_total_counts
    gene_detected_counts = loaded.dataset.gene_detected_counts
    if loaded.fitted_gene_indices.size > 0:
        gene_names = gene_names[loaded.fitted_gene_indices]
        gene_total_counts = gene_total_counts[loaded.fitted_gene_indices]
        gene_detected_counts = gene_detected_counts[loaded.fitted_gene_indices]

    candidates = search_gene_candidates_raw(
        query=query,
        gene_names=gene_names,
        gene_total_counts=gene_total_counts,
        gene_detected_counts=gene_detected_counts,
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


def analyze_gene(
    state: AppState, query: str, fit_params: GeneFitParams | None = None
) -> GeneAnalysis:
    loaded = state.require_loaded()
    print(f"[prism-server] analyze gene query={query!r}", flush=True)
    gene_index = resolve_gene_query(
        query, loaded.dataset.gene_names, loaded.dataset.gene_to_idx
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
        analysis = _build_analysis(
            loaded=loaded,
            gene_index=gene_index,
            priors=priors,
            s_hat=default_s_hat(state, loaded),
            source=loaded.model.source,
            chunk_size=state.config.analysis_chunk_size,
            pool_report=pool_report,
            cache_key=analysis_key,
        )
        state.set_cache("analysis", analysis_key, analysis)
        return analysis

    if fit_params is None:
        fit_params = GeneFitParams()

    cache_key = _fit_cache_key(loaded, gene_name, fit_params)
    cached = state.get_cached_fit(cache_key)
    if cached is None:
        print(
            f"[prism-server] cache miss gene={gene_name}; fitting on demand with r={fit_params.r:.4f} grid={fit_params.grid_size} sigma={fit_params.sigma_bins:.3f} iter={fit_params.n_iter}",
            flush=True,
        )
        pool_report = ensure_pool_report(state, loaded, r=fit_params.r)
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
        prior_report = engine.fit_report(
            batch,
            s_hat=pool_s_hat(pool_report, fit_params.r),
            training_cfg=PriorEngineTrainingConfig(
                lr=fit_params.lr,
                n_iter=fit_params.n_iter,
                lr_min_ratio=fit_params.lr_min_ratio,
                grad_clip=fit_params.grad_clip,
                init_temperature=fit_params.init_temperature,
                cell_chunk_size=fit_params.cell_chunk_size,
                optimizer=fit_params.optimizer,
                scheduler=fit_params.scheduler,
            ),
        )
        priors = GridDistribution(
            grid_min=float(prior_report.grid_min[0]),
            grid_max=float(prior_report.grid_max[0]),
            weights=np.asarray(prior_report.prior_weights[0], dtype=np.float64),
        )
        cached = {
            "priors": priors,
            "s_hat": float(pool_s_hat(pool_report, fit_params.r)),
            "pool_report": pool_report,
            "prior_report": prior_report,
            "fit_params": _fit_params_dict(fit_params),
        }
        state.set_cached_fit(cache_key, cached)
        print(
            f"[prism-server] fit completed gene={gene_name} final_loss={prior_report.final_loss:.6f} best_loss={prior_report.best_loss:.6f}",
            flush=True,
        )
    else:
        print(f"[prism-server] cache hit gene={gene_name}", flush=True)

    analysis = _build_analysis(
        loaded=loaded,
        gene_index=gene_index,
        priors=cast(GridDistribution, cached["priors"]),
        s_hat=float(cached["s_hat"]),
        source="on-demand-fit",
        fit_params=cast(dict[str, object], cached["fit_params"]),
        chunk_size=state.config.analysis_chunk_size,
        pool_report=cast(PoolFitReport | None, cached.get("pool_report")),
        prior_report=cast(PriorFitReport | None, cached.get("prior_report")),
        cache_key=analysis_key,
    )
    state.set_cache("analysis", analysis_key, analysis)
    return analysis


def ensure_pool_report(
    state: AppState,
    loaded: LoadedState,
    r: float | None = None,
) -> PoolFitReport:
    if loaded.model.pool_report is not None:
        print("[prism-server] using pool report from checkpoint", flush=True)
        return loaded.model.pool_report

    cached = state.pool_report
    if cached is not None:
        print("[prism-server] using cached in-memory pool report", flush=True)
        return cached

    resolved_r = _resolve_pool_r(loaded, r)
    print(
        f"[prism-server] fitting pool report from totals cells={loaded.n_cells} r={resolved_r:.4f}",
        flush=True,
    )
    report = fit_pool_scale_report(loaded.dataset.totals, use_posterior_mu=False)
    state.set_pool_report(report)
    print(
        f"[prism-server] pool fit done mu={report.mu:.6f} sigma={report.sigma:.6f} point_eta={report.point_eta:.4f} inferred_s_hat={pool_s_hat(report, resolved_r):.4f}",
        flush=True,
    )
    return report


def default_s_hat(state: AppState, loaded: LoadedState) -> float:
    if loaded.model.s_hat is not None:
        return float(loaded.model.s_hat)
    return float(
        pool_s_hat(ensure_pool_report(state, loaded), _resolve_pool_r(loaded, None))
    )


def pool_s_hat(report: PoolFitReport, r: float) -> float:
    return float(report.point_eta / r)


def _resolve_pool_r(loaded: LoadedState, r: float | None) -> float:
    if r is not None:
        return float(r)
    if loaded.model.r_hint is not None:
        return float(loaded.model.r_hint)
    return 0.05


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
    surprisal = np.empty(loaded.n_cells, dtype=np.float64)
    sharpness = np.empty(loaded.n_cells, dtype=np.float64)
    for cell_slice in _iter_cell_slices(loaded.n_cells, chunk_size):
        batch = GeneBatch(
            gene_names=[gene_name],
            counts=counts[cell_slice, None],
            totals=loaded.dataset.totals[cell_slice],
        )
        result = posterior.extract(
            batch,
            s_hat=s_hat,
            channels={"signal", "confidence", "surprisal", "sharpness"},
        )
        signal[cell_slice] = result["signal"][0]
        confidence[cell_slice] = result["confidence"][0]
        surprisal[cell_slice] = result["surprisal"][0]
        sharpness[cell_slice] = result["sharpness"][0]

    representative_idx = _representative_indices(signal, n=12)
    representative_batch = GeneBatch(
        gene_names=[gene_name],
        counts=counts[representative_idx, None],
        totals=loaded.dataset.totals[representative_idx],
    )
    representative = posterior.extract(
        representative_batch,
        s_hat=s_hat,
        channels={"posterior", "signal", "confidence", "surprisal", "sharpness"},
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
        surprisal=surprisal,
        sharpness=sharpness,
        support=representative["support"][0],
        prior_weights=representative["prior_weights"][0],
        posterior_samples=representative["posterior"][0],
        posterior_cell_indices=representative_idx,
        summary=summarize_gene_expression(loaded, gene_index),
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
        "h5ad": str(loaded.dataset.h5ad_path),
        "layer": loaded.dataset.layer,
        "gene": gene_name,
        **_fit_params_dict(fit_params),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _analysis_cache_key(
    loaded: LoadedState, gene_name: str, fit_params: GeneFitParams | None
) -> str:
    payload: dict[str, object] = {
        "h5ad": str(loaded.dataset.h5ad_path),
        "layer": loaded.dataset.layer,
        "ckpt": None if loaded.model.ckpt_path is None else str(loaded.model.ckpt_path),
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


def summarize_gene_expression(loaded: LoadedState, gene_index: int) -> GeneSummary:
    counts = slice_gene_counts(loaded.dataset.matrix, gene_index)
    totals = loaded.dataset.totals
    detected = counts > 0
    corr = 0.0
    if np.std(counts) > 0 and np.std(totals) > 0:
        corr = float(np.corrcoef(totals, counts)[0, 1])
    treatment_table: list[dict[str, object]] = []
    if (
        "treatment" in loaded.dataset.adata.obs.columns
        and "total_umi" in loaded.dataset.adata.obs.columns
    ):
        obs: Any = loaded.dataset.adata.obs.copy()
        obs["gene_count"] = counts
        grouped: Any = obs.groupby("treatment", observed=False).agg(
            cells=("gene_count", "size"),
            total_counts=("gene_count", "sum"),
            mean_count=("gene_count", "mean"),
            detected_frac=("gene_count", lambda series: float((series > 0).mean())),
            mean_total_umi=("total_umi", "mean"),
        )
        grouped = grouped.sort_values(["total_counts", "cells"], ascending=False).head(
            12
        )
        treatment_table = [
            {
                "treatment": str(name),
                "cells": int(row["cells"]),
                "total_counts": float(row["total_counts"]),
                "mean_count": float(row["mean_count"]),
                "detected_frac": float(row["detected_frac"]),
                "mean_total_umi": float(row["mean_total_umi"]),
            }
            for name, row in grouped.iterrows()
        ]
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
