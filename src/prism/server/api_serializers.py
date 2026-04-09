from __future__ import annotations

from typing import Any

import numpy as np

from .services.analysis import (
    CheckpointSummary,
    GeneAnalysis,
    GeneBrowsePage,
    GeneCandidate,
    GeneSummary,
    KBulkAnalysis,
)
from .services.figures import (
    figure_to_data_uri,
    plot_kbulk_group_comparison,
    plot_objective_trace,
    plot_posterior_gallery,
    plot_prior_overlay,
    plot_raw_overview,
    plot_signal_interface,
)
from .state import AppState


def serialize_context_snapshot(
    state: AppState,
    *,
    dataset_summary: dict[str, object] | None,
    checkpoint_summary: CheckpointSummary | None,
) -> dict[str, Any]:
    loaded = state.loaded
    if loaded is None or dataset_summary is None:
        return {
            "loaded": False,
            "contextKey": None,
            "dataset": None,
            "checkpoint": None,
        }
    return {
        "loaded": True,
        "contextKey": loaded.context_key,
        "dataset": serialize_dataset_summary(dataset_summary),
        "checkpoint": serialize_checkpoint_summary(checkpoint_summary),
    }


def serialize_dataset_summary(summary: dict[str, object]) -> dict[str, Any]:
    return {
        "nCells": _to_int(summary.get("n_cells")),
        "nGenes": _to_int(summary.get("n_genes")),
        "layer": str(summary.get("layer") or "(X)"),
        "h5adPath": str(summary.get("h5ad_path") or ""),
        "labelKeys": [str(item) for item in _to_tuple(summary.get("label_keys"))],
        "totalCountMean": _to_float(summary.get("total_count_mean")),
        "totalCountMedian": _to_float(summary.get("total_count_median")),
        "totalCountP99": _to_float(summary.get("total_count_p99")),
    }


def serialize_checkpoint_summary(
    summary: CheckpointSummary | None,
) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "ckptPath": summary.ckpt_path,
        "geneCount": summary.gene_count,
        "hasGlobalPrior": summary.has_global_prior,
        "nLabelPriors": summary.n_label_priors,
        "labelPreview": list(summary.label_preview),
        "distribution": summary.distribution,
        "supportDomain": summary.support_domain,
        "scale": summary.scale,
        "meanReferenceCount": summary.mean_reference_count,
        "nReferenceGenes": summary.n_reference_genes,
        "nOverlapReferenceGenes": summary.n_overlap_reference_genes,
        "suggestedLabelKey": summary.suggested_label_key,
    }


def serialize_gene_browse_page(page: GeneBrowsePage | None) -> dict[str, Any] | None:
    if page is None:
        return None
    return {
        "query": page.query,
        "scope": page.scope,
        "sortBy": page.sort_by,
        "descending": page.descending,
        "page": page.page,
        "pageSize": page.page_size,
        "totalItems": page.total_items,
        "totalPages": page.total_pages,
        "items": [serialize_gene_candidate(item) for item in page.items],
    }


def serialize_gene_candidate(candidate: GeneCandidate) -> dict[str, Any]:
    return {
        "geneName": candidate.gene_name,
        "geneIndex": candidate.gene_index,
        "totalCount": candidate.total_count,
        "detectedCells": candidate.detected_cells,
        "detectedFraction": candidate.detected_fraction,
    }


def serialize_gene_analysis(
    state: AppState,
    analysis: GeneAnalysis,
) -> dict[str, Any]:
    return {
        "geneName": analysis.gene_name,
        "geneIndex": analysis.gene_index,
        "source": analysis.source,
        "mode": analysis.mode,
        "priorSource": analysis.prior_source,
        "labelKey": analysis.label_key,
        "label": analysis.label,
        "nCells": analysis.n_cells,
        "referenceGeneCount": analysis.reference_gene_count,
        "referenceSource": analysis.reference_source,
        "availableLabelKeys": list(analysis.available_label_keys),
        "availableLabels": list(analysis.available_labels),
        "rawSummary": serialize_gene_summary(analysis.raw_summary),
        "checkpointSummary": serialize_checkpoint_summary(analysis.checkpoint_summary),
        "prior": _serialize_prior(analysis.prior),
        "checkpointPrior": _serialize_prior(analysis.checkpoint_prior),
        "posterior": _serialize_posterior(analysis),
        "fit": _serialize_fit(analysis),
        "figures": _serialize_analysis_figures(state, analysis),
    }


def serialize_kbulk_analysis(
    state: AppState,
    result: KBulkAnalysis,
) -> dict[str, Any]:
    figure_key = state.make_cache_key("figures", result.cache_key, "kbulk")
    figure = state.get_or_create_cache(
        "figures",
        figure_key,
        lambda: figure_to_data_uri(plot_kbulk_group_comparison(result)),
    )
    return {
        "geneName": result.gene_name,
        "geneIndex": result.gene_index,
        "classKey": result.class_key,
        "priorSource": result.prior_source,
        "k": result.k,
        "nSamples": result.n_samples,
        "sampleSeed": result.sample_seed,
        "maxClasses": result.max_classes,
        "sampleBatchSize": result.sample_batch_size,
        "availableClassKeys": list(result.available_class_keys),
        "groups": [
            {
                "label": group.label,
                "nCells": group.n_cells,
                "realizedSamples": group.realized_samples,
                "meanSignal": group.mean_signal,
                "stdSignal": group.std_signal,
                "meanEntropy": group.mean_entropy,
                "stdEntropy": group.std_entropy,
            }
            for group in result.groups
        ],
        "figure": figure,
    }


def serialize_gene_summary(summary: GeneSummary) -> dict[str, Any]:
    return {
        "totalCount": summary.total_count,
        "meanCount": summary.mean_count,
        "medianCount": summary.median_count,
        "p90Count": summary.p90_count,
        "p99Count": summary.p99_count,
        "maxCount": summary.max_count,
        "detectedCells": summary.detected_cells,
        "detectedFraction": summary.detected_fraction,
        "zeroFraction": summary.zero_fraction,
        "countTotalCorrelation": summary.count_total_correlation,
    }


def _serialize_prior(prior: object | None) -> dict[str, Any] | None:
    if prior is None:
        return None
    support = np.asarray(getattr(prior, "scaled_support"), dtype=np.float64).reshape(-1)
    probabilities = np.asarray(
        getattr(prior, "prior_probabilities"),
        dtype=np.float64,
    ).reshape(-1)
    return {
        "supportDomain": str(getattr(prior, "support_domain")),
        "scale": float(getattr(prior, "scale")),
        "support": support.tolist(),
        "probabilities": probabilities.tolist(),
    }


def _serialize_posterior(analysis: GeneAnalysis) -> dict[str, Any] | None:
    posterior = analysis.posterior
    if posterior is None:
        return None
    signal = np.asarray(posterior.map_scaled_support[:, 0], dtype=np.float64)
    posterior_entropy = np.asarray(posterior.posterior_entropy[:, 0], dtype=np.float64)
    prior_entropy = np.asarray(posterior.prior_entropy[:, 0], dtype=np.float64)
    mutual_information = np.asarray(
        posterior.mutual_information[:, 0], dtype=np.float64
    )
    map_probability = np.asarray(posterior.map_probability[:, 0], dtype=np.float64)
    return {
        "supportDomain": posterior.support_domain,
        "summary": {
            "mapSignal": _summarize_vector(signal),
            "mapProbability": _summarize_vector(map_probability),
            "posteriorEntropy": _summarize_vector(posterior_entropy),
            "priorEntropy": _summarize_vector(prior_entropy),
            "mutualInformation": _summarize_vector(mutual_information),
        },
    }


def _serialize_fit(analysis: GeneAnalysis) -> dict[str, Any] | None:
    fit_result = analysis.fit_result
    if fit_result is None:
        return None
    objective_history = np.asarray(fit_result.objective_history, dtype=np.float64)
    return {
        "objectiveHistory": objective_history.tolist(),
        "finalObjective": float(objective_history[-1])
        if objective_history.size
        else None,
    }


def _serialize_analysis_figures(
    state: AppState,
    analysis: GeneAnalysis,
) -> dict[str, str | None]:
    return {
        "rawOverview": _cached_figure(
            state,
            analysis.cache_key,
            "raw",
            lambda: plot_raw_overview(analysis),
        ),
        "priorOverlay": (
            None
            if analysis.prior is None
            else _cached_figure(
                state,
                analysis.cache_key,
                "prior",
                lambda: plot_prior_overlay(analysis),
            )
        ),
        "signalInterface": (
            None
            if analysis.posterior is None
            else _cached_figure(
                state,
                analysis.cache_key,
                "signal",
                lambda: plot_signal_interface(analysis),
            )
        ),
        "posteriorGallery": (
            None
            if analysis.posterior is None
            or analysis.posterior.posterior_probabilities is None
            else _cached_figure(
                state,
                analysis.cache_key,
                "gallery",
                lambda: plot_posterior_gallery(analysis),
            )
        ),
        "objectiveTrace": (
            None
            if analysis.fit_result is None
            else _cached_figure(
                state,
                analysis.cache_key,
                "objective",
                lambda: plot_objective_trace(analysis),
            )
        ),
    }


def _cached_figure(
    state: AppState,
    cache_key: str,
    figure_name: str,
    factory,
) -> str:
    key = state.make_cache_key("figures", cache_key, figure_name)
    return state.get_or_create_cache(
        "figures",
        key,
        lambda: figure_to_data_uri(factory()),
    )


def _summarize_vector(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _to_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    return ()


def _to_int(value: object) -> int:
    if isinstance(value, (int, float, str)):
        return int(value)
    return 0


def _to_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    return 0.0


__all__ = [
    "serialize_checkpoint_summary",
    "serialize_context_snapshot",
    "serialize_dataset_summary",
    "serialize_gene_analysis",
    "serialize_gene_browse_page",
    "serialize_gene_candidate",
    "serialize_kbulk_analysis",
]
