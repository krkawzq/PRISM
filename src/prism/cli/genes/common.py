from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
from rich.console import Console
from scipy import sparse

from prism.cli.common import normalize_choice as normalize_choice_shared, resolve_prior_source as resolve_prior_source_shared
from prism.io import GeneListSpec, read_gene_list as read_gene_list_shared
from prism.model import load_checkpoint

console = Console()
EPS = 1e-12
SIGNAL_LAYER = "signal"
SUPPORTED_HVG_FLAVORS = (
    "seurat",
    "cell_ranger",
    "seurat_v3",
    "seurat_v3_paper",
)
SUPPORTED_PRIOR_SOURCES = ("global", "label")
SUPPORTED_RANK_METHODS = (
    "hvg",
    "signal-hvg",
    "prior-entropy",
    "prior-entropy-reverse",
    "lognorm-variance",
    "lognorm-dispersion",
    "signal-variance",
    "signal-dispersion",
)
SIGNAL_RANK_METHODS = ("signal-hvg", "signal-variance", "signal-dispersion")
LOGNORM_RANK_METHODS = ("lognorm-variance", "lognorm-dispersion")
PRIOR_ENTROPY_METHODS = ("prior-entropy", "prior-entropy-reverse")


@dataclass(frozen=True, slots=True)
class RankingResult:
    gene_names: np.ndarray
    scores: np.ndarray
    descending: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_choice(
    value: str,
    *,
    supported: tuple[str, ...],
    option_name: str,
) -> str:
    return normalize_choice_shared(
        value,
        supported=supported,
        option_name=option_name,
    )


def normalize_rank_method(method: str) -> str:
    return normalize_choice(
        method,
        supported=SUPPORTED_RANK_METHODS,
        option_name="--method",
    )


def normalize_hvg_flavor(hvg_flavor: str) -> str:
    return normalize_choice(
        hvg_flavor,
        supported=SUPPORTED_HVG_FLAVORS,
        option_name="--hvg-flavor",
    )


def normalize_prior_source(prior_source: str) -> str:
    return resolve_prior_source_shared(prior_source)


def read_gene_list(path: Path) -> list[str]:
    return read_gene_list_shared(path)


def load_var_names(path: Path) -> list[str]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        return [str(name) for name in adata.var_names.tolist()]
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def maybe_subsample_adata(
    adata: ad.AnnData, *, max_cells: int | None, random_seed: int
) -> tuple[ad.AnnData, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "n_total_cells": int(adata.n_obs),
        "n_used_cells": int(adata.n_obs),
        "cell_sampling_applied": False,
        "max_cells": None if max_cells is None else int(max_cells),
        "random_seed": int(random_seed),
    }
    if max_cells is None or adata.n_obs <= max_cells:
        return adata, metadata
    if max_cells <= 0:
        raise ValueError("--max-cells must be positive when provided")
    rng = np.random.default_rng(random_seed)
    indices = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
    sampled = adata[indices].copy()
    metadata.update(
        {
            "n_used_cells": int(sampled.n_obs),
            "cell_sampling_applied": True,
        }
    )
    return sampled, metadata


def compute_hvg_ranking_from_adata(
    adata: ad.AnnData, *, flavor: str
) -> tuple[np.ndarray, np.ndarray]:
    import scanpy as sc

    try:
        sc.pp.highly_variable_genes(adata, flavor=cast(Any, flavor), inplace=True)
    except ImportError:
        if flavor not in {"seurat_v3", "seurat_v3_paper"}:
            raise
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor="seurat", inplace=True)
    if "highly_variable_rank" in adata.var:
        score = -np.nan_to_num(
            np.asarray(adata.var["highly_variable_rank"], dtype=np.float64), nan=np.inf
        )
    elif "dispersions_norm" in adata.var:
        score = np.asarray(adata.var["dispersions_norm"], dtype=np.float64)
    elif "variances_norm" in adata.var:
        score = np.asarray(adata.var["variances_norm"], dtype=np.float64)
    else:
        raise ValueError("scanpy did not produce an HVG ranking field")
    return np.asarray(adata.var_names), np.nan_to_num(
        score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf
    )


def compute_lognorm_ranking(
    adata: ad.AnnData, *, method: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    matrix = adata.X
    if matrix is None:
        raise ValueError("input h5ad has empty X")
    if sparse.issparse(matrix):
        counts = np.asarray(cast(Any, matrix).toarray(), dtype=np.float64)
    else:
        counts = np.asarray(matrix, dtype=np.float64)
    totals = counts.sum(axis=1)
    target = float(np.median(totals))
    values = np.log1p(counts * (target / np.maximum(totals, 1.0))[:, None])
    mean = np.mean(values, axis=0)
    var = np.var(values, axis=0)
    if method == "lognorm-variance":
        score = var
        score_definition = "variance(log1p(normalize_total(X)))"
    elif method == "lognorm-dispersion":
        score = var / np.maximum(mean, EPS)
        score_definition = (
            "variance(log1p(normalize_total(X))) / mean(log1p(normalize_total(X)))"
        )
    else:
        raise ValueError(f"unsupported lognorm method: {method}")
    return (
        np.asarray(adata.var_names),
        np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf),
        {"normalization_target": target, "score_definition": score_definition},
    )


def compute_signal_ranking(
    adata: ad.AnnData, *, method: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if SIGNAL_LAYER not in adata.layers:
        raise KeyError(f"input file is missing required layer: {SIGNAL_LAYER!r}")
    values = np.asarray(adata.layers[SIGNAL_LAYER], dtype=np.float64)
    mean = np.mean(values, axis=0)
    var = np.var(values, axis=0)
    if method == "signal-variance":
        score = var
        score_definition = "variance(signal)"
    elif method == "signal-dispersion":
        score = var / np.maximum(mean, EPS)
        score_definition = "variance(signal) / mean(signal)"
    elif method == "signal-hvg":
        signal = np.clip(
            np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None
        )
        signal_adata = ad.AnnData(
            X=np.log1p(signal),
            obs=cast(Any, adata.obs.copy()),
            var=cast(Any, adata.var.copy()),
        )
        gene_names, score = compute_hvg_ranking_from_adata(signal_adata, flavor="seurat")
        return (
            gene_names,
            score,
            {
                "layer": SIGNAL_LAYER,
                "hvg_flavor": "seurat",
                "score_definition": "HVG rank over log1p(signal)",
            },
        )
    else:
        raise ValueError(f"unsupported signal method: {method}")
    return (
        np.asarray(adata.var_names),
        np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf),
        {"layer": SIGNAL_LAYER, "score_definition": score_definition},
    )


def checkpoint_prior_entropy_scores(
    input_path: Path, *, prior_source: str, label: str | None
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = load_checkpoint(input_path)
    if prior_source == "global":
        if checkpoint.priors is None:
            raise ValueError("checkpoint does not contain global priors")
        priors = checkpoint.priors.batched()
    elif prior_source == "label":
        if label is None or not label.strip():
            raise ValueError("--label is required when --prior-source label")
        if label not in checkpoint.label_priors:
            raise ValueError(
                f"unknown label prior: {label!r}; available labels: {sorted(checkpoint.label_priors)}"
            )
        priors = checkpoint.label_priors[label].batched()
    else:
        raise ValueError(f"unsupported prior source: {prior_source}")
    weights = np.asarray(priors.weights, dtype=np.float64)
    scores = -(weights * np.log(np.clip(weights, EPS, None))).sum(axis=-1)
    return np.asarray(priors.gene_names), scores


def compute_ranking(
    input_path: Path,
    *,
    method: str,
    hvg_flavor: str,
    prior_source: str,
    label: str | None,
    max_cells: int | None,
    random_seed: int,
) -> RankingResult:
    method_resolved = normalize_rank_method(method)
    prior_source_resolved = normalize_prior_source(prior_source)
    hvg_flavor_resolved = normalize_hvg_flavor(hvg_flavor)
    if method_resolved in PRIOR_ENTROPY_METHODS:
        gene_names, scores = checkpoint_prior_entropy_scores(
            input_path,
            prior_source=prior_source_resolved,
            label=label,
        )
        return RankingResult(
            gene_names=gene_names,
            scores=np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0),
            descending=(method_resolved == "prior-entropy"),
            metadata={
                "score_definition": "prior entropy of F_g",
                "prior_source": prior_source_resolved,
                "label": label,
                "n_total_cells": None,
                "n_used_cells": None,
                "cell_sampling_applied": False,
                "max_cells": None if max_cells is None else int(max_cells),
                "random_seed": int(random_seed),
                "cell_sampling_ignored": True,
            },
        )

    adata = ad.read_h5ad(input_path)
    adata, sampling_metadata = maybe_subsample_adata(
        adata,
        max_cells=max_cells,
        random_seed=random_seed,
    )
    if method_resolved == "hvg":
        gene_names, scores = compute_hvg_ranking_from_adata(
            adata, flavor=hvg_flavor_resolved
        )
        metadata = {"hvg_flavor": hvg_flavor_resolved}
    elif method_resolved in LOGNORM_RANK_METHODS:
        gene_names, scores, metadata = compute_lognorm_ranking(
            adata,
            method=method_resolved,
        )
    elif method_resolved in SIGNAL_RANK_METHODS:
        gene_names, scores, metadata = compute_signal_ranking(
            adata,
            method=method_resolved,
        )
    else:
        raise ValueError(f"unsupported method: {method}")
    metadata = dict(metadata)
    metadata.update(sampling_metadata)
    return RankingResult(
        gene_names=np.asarray(gene_names),
        scores=np.asarray(scores, dtype=np.float64),
        descending=True,
        metadata=metadata,
    )


def filter_gene_scores(
    gene_names: np.ndarray,
    scores: np.ndarray,
    *,
    restrict_genes: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {"n_input_genes": int(gene_names.shape[0])}
    if restrict_genes is None:
        metadata.update(
            {
                "n_requested_restrict_genes": None,
                "n_missing_restrict_genes": 0,
                "missing_restrict_genes": [],
                "n_ranked_genes": int(gene_names.shape[0]),
            }
        )
        return gene_names, scores, metadata

    requested: list[str] = []
    seen_requested: set[str] = set()
    for gene in restrict_genes:
        if gene not in seen_requested:
            requested.append(gene)
            seen_requested.add(gene)

    gene_to_idx = {str(gene): idx for idx, gene in enumerate(gene_names.tolist())}
    selected_indices: list[int] = []
    missing: list[str] = []
    for gene in requested:
        idx = gene_to_idx.get(gene)
        if idx is None:
            missing.append(gene)
        else:
            selected_indices.append(idx)
    if not selected_indices:
        raise ValueError("no restricted genes were found in the input data")
    index_array = np.asarray(selected_indices, dtype=np.int64)
    filtered_gene_names = np.asarray(gene_names[index_array], dtype=gene_names.dtype)
    filtered_scores = np.asarray(scores[index_array], dtype=np.float64)
    metadata.update(
        {
            "n_requested_restrict_genes": int(len(requested)),
            "n_missing_restrict_genes": int(len(missing)),
            "missing_restrict_genes": missing,
            "n_ranked_genes": int(filtered_gene_names.shape[0]),
        }
    )
    return filtered_gene_names, filtered_scores, metadata


def rank_gene_scores(
    gene_names: np.ndarray, scores: np.ndarray, *, descending: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    return order, gene_names[order], scores[order]


def build_gene_list_spec(
    *,
    input_path: Path,
    method: str,
    ranked_gene_names: np.ndarray,
    ranked_scores: np.ndarray,
    metadata: dict[str, Any],
) -> GeneListSpec:
    return GeneListSpec(
        gene_names=[str(gene) for gene in ranked_gene_names.tolist()],
        scores=[float(score) for score in ranked_scores.tolist()],
        source_path=str(input_path),
        method=method,
        metadata=dict(metadata),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
