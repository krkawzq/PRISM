from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from prism.baseline.fg_analysis import (
    FgAnalysisSummary,
    fg_entropy,
    summarize_fg_analysis,
)
from prism.baseline.metrics import treatment_conditional_cv
from prism.model import GeneBatch, Posterior
from prism.model._typing import DTYPE_NP

from ..state import AppState, LoadedState


@dataclass(frozen=True, slots=True)
class GlobalRepresentationMetric:
    silhouette: float
    ari: float
    nmi: float
    pca_var_ratio: float
    neighborhood_consistency: float
    mean_treatment_cv: float | None


@dataclass(frozen=True, slots=True)
class GlobalEvaluationResult:
    label_key: str
    n_labels: int
    n_cells: int
    n_genes: int
    representation_metrics: dict[str, GlobalRepresentationMetric]
    fg_summary: FgAnalysisSummary
    top_entropy_genes: list[tuple[str, float]]


def compute_global_evaluation(
    state: AppState,
    *,
    max_genes: int = 256,
    gene_batch_size: int = 64,
) -> GlobalEvaluationResult:
    loaded = state.require_loaded()
    if loaded.model.engine is None or loaded.model.s_hat is None:
        raise ValueError("global evaluation requires a loaded checkpoint")

    cache_key = _cache_key(loaded, max_genes)
    cached = state.get_cached_global_eval(cache_key)
    if cached is not None:
        print(f"[prism-server] global eval cache hit key={cache_key[:8]}", flush=True)
        return cached

    label_key, labels = _resolve_labels(loaded)
    print(
        f"[prism-server] global eval start label_key={label_key} max_genes={max_genes}",
        flush=True,
    )
    gene_names, gene_positions = _select_gene_subset(loaded, max_genes)
    print(
        f"[prism-server] global eval selected genes count={len(gene_names)} top_gene={gene_names[0] if gene_names else '-'}",
        flush=True,
    )
    raw = _slice_gene_matrix(loaded.dataset.matrix, gene_positions)
    print(f"[prism-server] global eval raw matrix shape={raw.shape}", flush=True)
    totals = loaded.dataset.totals
    normalized = _normalize_total_matrix(raw, totals)
    print("[prism-server] global eval built NormalizeTotalX", flush=True)
    log1p_normalized = np.log1p(normalized)
    print("[prism-server] global eval built Log1pNormalizeTotalX", flush=True)
    signal = _extract_signal_matrix(
        loaded,
        gene_names=gene_names,
        gene_positions=gene_positions,
        batch_size=gene_batch_size,
    )
    print(
        f"[prism-server] global eval built signal matrix shape={signal.shape}",
        flush=True,
    )

    representation_metrics = {
        "X": _evaluate_matrix(raw, labels, name="X"),
        "NormalizeTotalX": _evaluate_matrix(normalized, labels, name="NormalizeTotalX"),
        "Log1pNormalizeTotalX": _evaluate_matrix(
            log1p_normalized, labels, name="Log1pNormalizeTotalX"
        ),
        "signal": _evaluate_matrix(signal, labels, name="signal"),
    }
    print("[prism-server] global eval starting F_g analysis", flush=True)
    fg_summary, top_entropy_genes = _analyze_fg(loaded)
    result = GlobalEvaluationResult(
        label_key=label_key,
        n_labels=int(np.unique(labels).size),
        n_cells=int(raw.shape[0]),
        n_genes=int(raw.shape[1]),
        representation_metrics=representation_metrics,
        fg_summary=fg_summary,
        top_entropy_genes=top_entropy_genes,
    )
    state.set_cached_global_eval(cache_key, result)
    print(
        f"[prism-server] global eval done cells={result.n_cells} genes={result.n_genes}",
        flush=True,
    )
    return result


def _resolve_labels(loaded: LoadedState) -> tuple[str, np.ndarray]:
    obs = loaded.dataset.adata.obs
    for key in ("treatment", "cell_type", "label", "group"):
        if key in obs.columns:
            values = np.asarray(obs[key]).reshape(-1)
            if np.unique(values).size >= 2:
                return key, values
    raise ValueError(
        "global evaluation requires a label column like treatment/cell_type/label/group"
    )


def _select_gene_subset(
    loaded: LoadedState, max_genes: int
) -> tuple[list[str], np.ndarray]:
    if loaded.fitted_gene_indices.size == 0:
        raise ValueError("checkpoint has no fitted genes overlapping the dataset")
    ranked_local = np.argsort(
        loaded.dataset.gene_total_counts[loaded.fitted_gene_indices]
    )[::-1]
    keep = loaded.fitted_gene_indices[ranked_local[:max_genes]]
    keep = np.asarray(keep, dtype=np.int64)
    gene_names = [str(loaded.dataset.gene_names[idx]) for idx in keep.tolist()]
    return gene_names, keep


def _slice_gene_matrix(matrix, gene_positions: np.ndarray) -> np.ndarray:
    subset = matrix[:, gene_positions]
    if sparse.issparse(subset):
        return np.asarray(subset.toarray(), dtype=DTYPE_NP)
    return np.asarray(subset, dtype=DTYPE_NP)


def _normalize_total_matrix(counts: np.ndarray, totals: np.ndarray) -> np.ndarray:
    target = float(np.median(totals))
    scale = target / np.maximum(np.asarray(totals, dtype=DTYPE_NP), 1e-12)
    return counts * scale[:, None]


def _extract_signal_matrix(
    loaded: LoadedState,
    *,
    gene_names: list[str],
    gene_positions: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if loaded.model.engine is None or loaded.model.s_hat is None:
        raise ValueError("global evaluation requires checkpoint engine and s_hat")
    priors = loaded.model.engine.get_priors(gene_names)
    if priors is None:
        raise ValueError("failed to read checkpoint priors for global evaluation")
    posterior = Posterior(gene_names, priors)
    counts = _slice_gene_matrix(loaded.dataset.matrix, gene_positions)
    signal = np.zeros_like(counts, dtype=np.float32)
    offsets = range(0, len(gene_names), batch_size)
    total_batches = max((len(gene_names) + batch_size - 1) // batch_size, 1)
    for batch_index, offset in enumerate(offsets, start=1):
        batch_gene_names = gene_names[offset : offset + batch_size]
        batch_counts = counts[:, offset : offset + batch_size]
        print(
            f"[prism-server] global eval signal batch {batch_index}/{total_batches} genes={len(batch_gene_names)} first={batch_gene_names[0] if batch_gene_names else '-'}",
            flush=True,
        )
        batch = GeneBatch(
            gene_names=batch_gene_names,
            counts=batch_counts,
            totals=loaded.dataset.totals,
        )
        extracted = posterior.extract(
            batch, s_hat=float(loaded.model.s_hat), channels={"signal"}
        )
        signal[:, offset : offset + len(batch_gene_names)] = extracted[
            "signal"
        ].T.astype(np.float32, copy=False)
    return signal


def _evaluate_matrix(
    matrix: np.ndarray, labels: np.ndarray, *, name: str
) -> GlobalRepresentationMetric:
    matrix = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(labels)
    print(
        f"[prism-server] global eval matrix start name={name} shape={matrix.shape} labels={np.unique(labels).size}",
        flush=True,
    )
    n_components = max(2, min(20, matrix.shape[0] - 1, matrix.shape[1]))
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(matrix)
    print(
        f"[prism-server] global eval PCA done name={name} n_components={n_components}",
        flush=True,
    )
    silhouette = (
        float(silhouette_score(embedding, labels))
        if np.unique(labels).size >= 2
        else 0.0
    )
    n_clusters = int(np.unique(labels).size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(embedding)
    print(
        f"[prism-server] global eval clustering done name={name} n_clusters={n_clusters}",
        flush=True,
    )
    ari = float(adjusted_rand_score(labels, clusters))
    nmi = float(normalized_mutual_info_score(labels, clusters))
    pca_var_ratio = float(
        np.sum(pca.explained_variance_ratio_[: min(10, n_components)])
    )
    neighborhood_consistency = _neighborhood_consistency(embedding)
    mean_treatment_cv = _mean_treatment_cv(matrix, labels)
    print(
        f"[prism-server] global eval metrics name={name} silhouette={silhouette:.4f} ari={ari:.4f} nmi={nmi:.4f} pca_var={pca_var_ratio:.4f} neigh={neighborhood_consistency:.4f}",
        flush=True,
    )
    return GlobalRepresentationMetric(
        silhouette=silhouette,
        ari=ari,
        nmi=nmi,
        pca_var_ratio=pca_var_ratio,
        neighborhood_consistency=neighborhood_consistency,
        mean_treatment_cv=mean_treatment_cv,
    )


def _neighborhood_consistency(embedding: np.ndarray, k: int = 15) -> float:
    if embedding.shape[0] <= 2:
        return 0.0
    n_neighbors = min(k + 1, embedding.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(embedding)
    _, indices = nn.kneighbors(embedding)
    scores: list[float] = []
    for row_idx, neigh in enumerate(indices):
        neigh = neigh[neigh != row_idx]
        if neigh.size == 0:
            continue
        local = embedding[neigh]
        center = np.mean(local, axis=0)
        dispersion = float(np.mean(np.sum((local - center) ** 2, axis=1)))
        scores.append(1.0 / (1.0 + dispersion))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def _mean_treatment_cv(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    if np.unique(labels).size < 2:
        return None
    cvs = [
        treatment_conditional_cv(matrix[:, idx], labels)
        for idx in range(matrix.shape[1])
    ]
    if not cvs:
        return None
    return float(np.mean(cvs))


def _cache_key(loaded: LoadedState, max_genes: int) -> str:
    payload = (
        str(loaded.dataset.h5ad_path),
        loaded.dataset.layer,
        str(loaded.model.ckpt_path),
        int(max_genes),
        tuple(loaded.fitted_gene_names[:max_genes]),
    )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def _analyze_fg(
    loaded: LoadedState,
) -> tuple[FgAnalysisSummary, list[tuple[str, float]]]:
    if loaded.model.engine is None:
        raise ValueError("F_g analysis requires checkpoint engine")
    gene_names = list(loaded.fitted_gene_names)
    priors = loaded.model.engine.get_priors(gene_names)
    if priors is None:
        raise ValueError("failed to read priors for F_g analysis")
    gene_idx = np.asarray(
        [loaded.dataset.gene_to_idx[name] for name in gene_names], dtype=np.int64
    )
    mean_expression = np.asarray(
        loaded.dataset.gene_total_counts[gene_idx], dtype=float
    ) / max(loaded.n_cells, 1)
    subset = _slice_gene_matrix(loaded.dataset.matrix, gene_idx)
    normalized = _normalize_total_matrix(subset, loaded.dataset.totals)
    hvg_scores = np.var(normalized, axis=0) / np.maximum(
        np.mean(normalized, axis=0), 1e-12
    )
    summary = summarize_fg_analysis(
        priors, mean_expression=mean_expression, hvg_scores=hvg_scores
    )
    entropy = fg_entropy(priors)
    order = np.argsort(entropy)[::-1][:20]
    top_entropy_genes = [
        (gene_names[int(idx)], float(entropy[int(idx)])) for idx in order
    ]
    print(
        f"[prism-server] F_g analysis done genes={len(gene_names)} entropy_mean={summary.entropy_mean:.4f} hvg_rho={summary.hvg_spearman:.4f}",
        flush=True,
    )
    return summary, top_entropy_genes
