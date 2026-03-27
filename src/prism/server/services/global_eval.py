from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from prism.model import ObservationBatch, Posterior, SignalChannel

from ..state import AppState, LoadedState
from .checkpoints import CheckpointState
from .datasets import (
    compute_reference_counts,
    resolve_gene_positions,
    slice_gene_matrix,
)


@dataclass(frozen=True, slots=True)
class GlobalRepresentationMetric:
    silhouette: float
    ari: float
    nmi: float
    pca_var_ratio: float
    neighborhood_consistency: float


@dataclass(frozen=True, slots=True)
class GlobalEvaluationResult:
    label_key: str
    n_labels: int
    n_cells: int
    n_genes: int
    representation_metrics: dict[str, GlobalRepresentationMetric]
    top_entropy_genes: list[tuple[str, float]]


@dataclass(frozen=True, slots=True)
class GlobalEvalParams:
    max_cells: int = 2000
    max_genes: int = 256
    gene_batch_size: int = 64
    random_seed: int = 0


def compute_global_evaluation(
    state: AppState, *, params: GlobalEvalParams | None = None
) -> GlobalEvaluationResult:
    loaded = state.require_loaded()
    if loaded.checkpoint is None:
        raise ValueError("global evaluation requires a loaded checkpoint")
    resolved = (
        GlobalEvalParams(
            max_cells=state.config.global_eval_max_cells,
            max_genes=state.config.global_eval_max_genes,
            gene_batch_size=state.config.inference_batch_size,
            random_seed=0,
        )
        if params is None
        else params
    )
    cache_key = state.make_cache_key(
        "global_eval",
        resolved.max_cells,
        resolved.max_genes,
        resolved.gene_batch_size,
        resolved.random_seed,
    )
    cached = state.get_cache("global_eval", cache_key)
    if cached is not None:
        return cast(GlobalEvaluationResult, cached)
    result = _compute_global_evaluation_uncached(loaded, resolved)
    state.set_cache("global_eval", cache_key, result)
    return result


def _compute_global_evaluation_uncached(
    loaded: LoadedState, params: GlobalEvalParams
) -> GlobalEvaluationResult:
    checkpoint = loaded.checkpoint
    if checkpoint is None:
        raise ValueError("global evaluation requires a loaded checkpoint")
    label_key, labels = _resolve_labels(loaded)
    cell_indices = _select_cell_subset(labels, params.max_cells, params.random_seed)
    sampled_labels = labels[cell_indices]
    gene_names = _select_gene_subset(loaded, params.max_genes)
    gene_positions = [loaded.dataset.gene_to_idx[name] for name in gene_names]
    raw = slice_gene_matrix(
        loaded.dataset.matrix, gene_positions, cell_indices=cell_indices
    )
    ref_positions = resolve_gene_positions(
        checkpoint.reference_gene_names, loaded.dataset.gene_to_idx
    )
    sampled_reference_counts = compute_reference_counts(
        loaded.dataset.matrix, ref_positions, cell_indices=cell_indices
    )
    normalized = _normalize_total_matrix(raw, sampled_reference_counts)
    lognorm = np.log1p(normalized)
    signal = _extract_signal_matrix(
        checkpoint,
        gene_names=gene_names,
        counts=raw,
        reference_counts=sampled_reference_counts,
        batch_size=params.gene_batch_size,
    )
    representation_metrics = {
        "X": _evaluate_matrix(raw, sampled_labels),
        "NormalizeTotalX": _evaluate_matrix(normalized, sampled_labels),
        "Log1pNormalizeTotalX": _evaluate_matrix(lognorm, sampled_labels),
        "signal": _evaluate_matrix(signal, sampled_labels),
    }
    top_entropy_genes = _top_entropy_genes(loaded, limit=16)
    return GlobalEvaluationResult(
        label_key=label_key,
        n_labels=int(np.unique(sampled_labels).size),
        n_cells=int(raw.shape[0]),
        n_genes=int(raw.shape[1]),
        representation_metrics=representation_metrics,
        top_entropy_genes=top_entropy_genes,
    )


def _resolve_labels(loaded: LoadedState) -> tuple[str, np.ndarray]:
    labels = loaded.dataset.treatment_labels
    if labels is None:
        raise ValueError(
            "global evaluation requires a label column such as treatment/cell_type/label/group"
        )
    return loaded.dataset.treatment_label_key or "label", labels


def _select_cell_subset(labels: np.ndarray, max_cells: int, seed: int) -> np.ndarray:
    n_cells = labels.shape[0]
    if n_cells <= max_cells:
        return np.arange(n_cells, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(np.arange(n_cells, dtype=np.int64), size=max_cells, replace=False)
    )


def _select_gene_subset(loaded: LoadedState, max_genes: int) -> list[str]:
    fitted = list(loaded.fitted_gene_names)
    fitted.sort(
        key=lambda name: loaded.dataset.gene_total_counts[
            loaded.dataset.gene_to_idx[name]
        ],
        reverse=True,
    )
    return fitted[:max_genes]


def _normalize_total_matrix(
    counts: np.ndarray, reference_counts: np.ndarray
) -> np.ndarray:
    target = float(np.median(reference_counts))
    scale = target / np.maximum(np.asarray(reference_counts, dtype=np.float64), 1e-12)
    return counts * scale[:, None]


def _extract_signal_matrix(
    checkpoint: CheckpointState,
    *,
    gene_names: list[str],
    counts: np.ndarray,
    reference_counts: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    priors = checkpoint.checkpoint.priors.subset(gene_names)
    posterior = Posterior(gene_names, priors)
    signal = np.zeros_like(counts, dtype=np.float32)
    requested = cast(set[SignalChannel], {"signal"})
    for offset in range(0, len(gene_names), batch_size):
        batch_names = gene_names[offset : offset + batch_size]
        batch_counts = counts[:, offset : offset + batch_size]
        extracted = posterior.extract(
            ObservationBatch(
                gene_names=batch_names,
                counts=batch_counts,
                reference_counts=reference_counts,
            ),
            channels=requested,
        )
        signal[:, offset : offset + len(batch_names)] = np.asarray(
            extracted["signal"], dtype=np.float32
        )
    return signal


def _evaluate_matrix(
    values: np.ndarray, labels: np.ndarray
) -> GlobalRepresentationMetric:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape={x.shape}")
    if x.shape[0] < 4:
        return GlobalRepresentationMetric(0.0, 0.0, 0.0, 0.0, 0.0)
    n_components = min(10, x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(x)
    n_labels = max(int(np.unique(labels).size), 2)
    km = KMeans(n_clusters=n_labels, n_init="auto", random_state=0)
    clusters = km.fit_predict(embedding)
    nn = NearestNeighbors(n_neighbors=min(16, max(x.shape[0] - 1, 1)))
    nn.fit(embedding)
    indices = nn.kneighbors(return_distance=False)
    neighbor_score = float(
        np.mean([np.mean(labels[row[1:]] == labels[row[0]]) for row in indices])
    )
    silhouette = 0.0 if n_labels < 2 else float(silhouette_score(embedding, labels))
    return GlobalRepresentationMetric(
        silhouette=silhouette,
        ari=float(adjusted_rand_score(labels, clusters)),
        nmi=float(normalized_mutual_info_score(labels, clusters)),
        pca_var_ratio=float(np.sum(pca.explained_variance_ratio_)),
        neighborhood_consistency=neighbor_score,
    )


def _top_entropy_genes(loaded: LoadedState, *, limit: int) -> list[tuple[str, float]]:
    checkpoint = loaded.checkpoint
    if checkpoint is None:
        raise ValueError("global evaluation requires a loaded checkpoint")
    priors = checkpoint.checkpoint.priors.batched()
    weights = np.asarray(priors.weights, dtype=np.float64)
    entropy = -(weights * np.log(np.clip(weights, 1e-12, None))).sum(axis=-1)
    order = np.argsort(entropy)[::-1][:limit]
    return [(priors.gene_names[idx], float(entropy[idx])) for idx in order.tolist()]
