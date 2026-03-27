from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
from rich.console import Console
from scipy import sparse

from prism.model import load_checkpoint

console = Console()
EPS = 1e-12


def read_gene_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_var_names(path: Path) -> list[str]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        return [str(name) for name in adata.var_names.tolist()]
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


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
    else:
        raise ValueError("scanpy did not produce an HVG ranking field")
    return np.asarray(adata.var_names), np.nan_to_num(
        score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf
    )


def compute_lognorm_ranking(
    adata: ad.AnnData, *, dispersion: bool
) -> tuple[np.ndarray, np.ndarray]:
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
    score = var / np.maximum(mean, EPS) if dispersion else var
    return np.asarray(adata.var_names), np.nan_to_num(
        score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
