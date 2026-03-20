from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from prism.baseline.metrics import (
    depth_correlation,
    log1p_normalize_total,
    normalize_total,
    raw_umi,
)
from prism.model import (
    GeneBatch,
    GridDistribution,
    PoolEstimate,
    Posterior,
    PriorEngine,
    PriorEngineSetting,
    PriorEngineTrainingConfig,
    fit_pool_scale,
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
class GeneAnalysis:
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
    representation_metrics: dict[str, dict[str, float]]
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
    return {
        "n_cells": loaded.n_cells,
        "n_genes": loaded.n_genes,
        "fitted_genes": len(loaded.fitted_gene_names),
        "layer": loaded.layer or "X",
        "h5ad_path": str(loaded.h5ad_path),
        "ckpt_path": "" if loaded.ckpt_path is None else str(loaded.ckpt_path),
        "s_hat": _default_s_hat(state, loaded),
        "mean_total": float(np.mean(loaded.totals)),
        "median_total": float(np.median(loaded.totals)),
    }


def top_fitted_genes(state: AppState, limit: int | None = None) -> list[GeneCandidate]:
    loaded = state.require_loaded()
    limit = state.config.top_gene_limit if limit is None else limit

    if loaded.fitted_gene_indices.size > 0:
        ranked = np.argsort(loaded.gene_total_counts[loaded.fitted_gene_indices])[::-1][
            :limit
        ]
        return [
            _candidate_from_index(loaded, int(loaded.fitted_gene_indices[idx]))
            for idx in ranked
        ]

    ranked = np.argsort(loaded.gene_total_counts)[::-1][:limit]
    return [_candidate_from_index(loaded, int(idx)) for idx in ranked]


def search_gene_candidates(
    state: AppState, query: str, limit: int | None = None
) -> list[GeneCandidate]:
    loaded = state.require_loaded()
    limit = state.config.top_gene_limit if limit is None else limit

    gene_names = loaded.gene_names
    gene_total_counts = loaded.gene_total_counts
    gene_detected_counts = loaded.gene_detected_counts
    if loaded.fitted_gene_indices.size > 0:
        gene_names = loaded.gene_names[loaded.fitted_gene_indices]
        gene_total_counts = loaded.gene_total_counts[loaded.fitted_gene_indices]
        gene_detected_counts = loaded.gene_detected_counts[loaded.fitted_gene_indices]

    candidates = search_gene_candidates_raw(
        query=query,
        gene_names=gene_names,
        gene_total_counts=gene_total_counts,
        gene_detected_counts=gene_detected_counts,
        n_cells=loaded.n_cells,
        limit=limit,
    )
    return [
        GeneCandidate(
            gene_name=item.gene_name,
            gene_index=int(loaded.gene_to_idx[item.gene_name]),
            total_umi=item.total_umi,
            detected_cells=item.detected_cells,
            detected_fraction=item.detected_fraction,
        )
        for item in candidates
    ]


def analyze_gene(
    state: AppState, query: str, fit_params: GeneFitParams | None = None
) -> GeneAnalysis:
    loaded = state.require_loaded()
    gene_index = resolve_gene_query(query, loaded.gene_names, loaded.gene_to_idx)
    gene_name = str(loaded.gene_names[gene_index])

    if (
        fit_params is None
        and loaded.engine is not None
        and loaded.engine.is_fitted(gene_name)
    ):
        priors = loaded.engine.get_priors(gene_name)
        if priors is None:
            raise ValueError(f"failed to read prior for gene {gene_name!r}")
        return _build_analysis(
            loaded=loaded,
            gene_index=gene_index,
            priors=priors,
            s_hat=float(
                loaded.checkpoint_s_hat if loaded.checkpoint_s_hat is not None else 0.0
            ),
            source="checkpoint",
            fit_params=None,
            chunk_size=state.config.analysis_chunk_size,
        )

    if fit_params is None:
        fit_params = GeneFitParams()

    cache_key = _fit_cache_key(loaded, gene_name, fit_params)
    cached = state.get_cached_fit(cache_key)
    if cached is None:
        pooled = _ensure_pool_estimate(state, loaded)
        s_hat = float(pooled.point_eta / fit_params.r)
        counts = slice_gene_counts(loaded.matrix, gene_index)
        batch = GeneBatch(
            gene_names=[gene_name], counts=counts[:, None], totals=loaded.totals
        )
        engine = PriorEngine(
            [gene_name],
            setting=PriorEngineSetting(
                grid_size=fit_params.grid_size,
                sigma_bins=fit_params.sigma_bins,
                align_loss_weight=fit_params.align_loss_weight,
                torch_dtype=fit_params.torch_dtype,
            ),
            device=fit_params.device,
        )
        engine.fit(
            batch,
            s_hat=s_hat,
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
        priors = engine.get_priors(gene_name)
        if priors is None:
            raise ValueError(f"failed to fit prior for gene {gene_name!r}")
        cached = {
            "priors": priors,
            "s_hat": s_hat,
            "fit_params": _fit_params_dict(fit_params),
        }
        state.set_cached_fit(cache_key, cached)

    return _build_analysis(
        loaded=loaded,
        gene_index=gene_index,
        priors=cast(GridDistribution, cached["priors"]),
        s_hat=float(cached["s_hat"]),
        source="fit-cache",
        fit_params=cached["fit_params"],
        chunk_size=state.config.analysis_chunk_size,
    )


def _build_analysis(
    *,
    loaded: LoadedState,
    gene_index: int,
    priors: GridDistribution,
    s_hat: float,
    source: str,
    fit_params: dict[str, object] | None,
    chunk_size: int,
) -> GeneAnalysis:
    gene_name = str(loaded.gene_names[gene_index])
    counts = slice_gene_counts(loaded.matrix, gene_index)
    posterior = Posterior([gene_name], priors)

    signal = np.empty(loaded.n_cells, dtype=np.float64)
    confidence = np.empty(loaded.n_cells, dtype=np.float64)
    surprisal = np.empty(loaded.n_cells, dtype=np.float64)
    sharpness = np.empty(loaded.n_cells, dtype=np.float64)
    for cell_slice in _iter_cell_slices(loaded.n_cells, chunk_size):
        batch = GeneBatch(
            gene_names=[gene_name],
            counts=counts[cell_slice, None],
            totals=loaded.totals[cell_slice],
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

    representative_idx = _representative_indices(signal, n=8)
    representative_batch = GeneBatch(
        gene_names=[gene_name],
        counts=counts[representative_idx, None],
        totals=loaded.totals[representative_idx],
    )
    representative = posterior.extract(
        representative_batch,
        s_hat=s_hat,
        channels={"posterior", "signal", "confidence", "surprisal", "sharpness"},
    )

    x_eff = counts / np.maximum(loaded.totals, 1.0) * s_hat
    counts_matrix = counts[:, None]
    representation_metrics = {
        "raw": _representation_summary(raw_umi(counts_matrix)[:, 0], loaded.totals),
        "normalize_total": _representation_summary(
            normalize_total(counts_matrix, loaded.totals)[:, 0], loaded.totals
        ),
        "log1p_normalize_total": _representation_summary(
            log1p_normalize_total(counts_matrix, loaded.totals)[:, 0], loaded.totals
        ),
        "signal": _representation_summary(signal, loaded.totals),
    }

    return GeneAnalysis(
        gene_name=gene_name,
        gene_index=gene_index,
        source=source,
        s_hat=s_hat,
        counts=counts,
        totals=loaded.totals,
        x_eff=x_eff,
        signal=signal,
        confidence=confidence,
        surprisal=surprisal,
        sharpness=sharpness,
        support=representative["support"][0],
        prior_weights=representative["prior_weights"][0],
        posterior_samples=representative["posterior"][0],
        posterior_cell_indices=representative_idx,
        representation_metrics=representation_metrics,
        fit_params=fit_params,
    )


def _ensure_pool_estimate(state: AppState, loaded: LoadedState) -> PoolEstimate:
    estimate = state.pool_estimate
    if estimate is not None:
        return estimate
    estimate = fit_pool_scale(loaded.totals)
    state.set_pool_estimate(estimate)
    return estimate


def _default_s_hat(state: AppState, loaded: LoadedState) -> float:
    if loaded.checkpoint_s_hat is not None:
        return float(loaded.checkpoint_s_hat)
    pooled = _ensure_pool_estimate(state, loaded)
    return float(pooled.point_eta / 0.05)


def _fit_cache_key(
    loaded: LoadedState, gene_name: str, fit_params: GeneFitParams
) -> str:
    payload = {
        "h5ad": str(loaded.h5ad_path),
        "layer": loaded.layer,
        "gene": gene_name,
        **_fit_params_dict(fit_params),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _fit_params_dict(fit_params: GeneFitParams) -> dict[str, object]:
    return {
        "r": fit_params.r,
        "grid_size": fit_params.grid_size,
        "sigma_bins": fit_params.sigma_bins,
        "align_loss_weight": fit_params.align_loss_weight,
        "lr": fit_params.lr,
        "n_iter": fit_params.n_iter,
        "lr_min_ratio": fit_params.lr_min_ratio,
        "grad_clip": fit_params.grad_clip,
        "init_temperature": fit_params.init_temperature,
        "cell_chunk_size": fit_params.cell_chunk_size,
        "optimizer": fit_params.optimizer,
        "scheduler": fit_params.scheduler,
        "torch_dtype": fit_params.torch_dtype,
        "device": fit_params.device,
    }


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
        gene_name=str(loaded.gene_names[gene_index]),
        gene_index=int(gene_index),
        total_umi=int(round(float(loaded.gene_total_counts[gene_index]))),
        detected_cells=int(loaded.gene_detected_counts[gene_index]),
        detected_fraction=float(loaded.gene_detected_counts[gene_index])
        / max(loaded.n_cells, 1),
    )


def _representation_summary(values: np.ndarray, totals: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p95": float(np.quantile(values, 0.95)),
        "depth_corr": float(depth_correlation(values, totals)),
    }
