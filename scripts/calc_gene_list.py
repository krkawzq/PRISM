#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, cast

import anndata as ad
import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prism.model import load_checkpoint

EPS = 1e-12
SIGNAL_LAYER = "signal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ranked gene lists from h5ad or checkpoint inputs."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-ranked-genes", type=Path, default=None)
    parser.add_argument("--ranked-limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument(
        "--method",
        choices=(
            "hvg",
            "signal-hvg",
            "prior-entropy",
            "prior-entropy-reverse",
            "lognorm-variance",
            "lognorm-dispersion",
            "signal-variance",
            "signal-dispersion",
        ),
        required=True,
    )
    parser.add_argument(
        "--hvg-flavor",
        choices=("seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"),
        default="seurat_v3",
    )
    return parser.parse_args()


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
    matrix_any = cast(Any, matrix)
    counts = (
        matrix_any.toarray() if sparse.issparse(matrix_any) else np.asarray(matrix_any)
    )
    totals = np.asarray(counts.sum(axis=1), dtype=np.float64)
    target = float(np.median(totals))
    values = np.log1p(
        np.asarray(counts, dtype=np.float64)
        * (target / np.maximum(totals, 1.0))[:, None]
    )
    mean = np.mean(values, axis=0)
    var = np.var(values, axis=0)
    if method == "lognorm-variance":
        score = var
        score_definition = "variance(log1p(normalize_total(X)))"
    else:
        score = var / np.maximum(mean, EPS)
        score_definition = (
            "variance(log1p(normalize_total(X))) / mean(log1p(normalize_total(X)))"
        )
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
        gene_names, score = compute_hvg_ranking_from_adata(
            signal_adata, flavor="seurat"
        )
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


def compute_prior_entropy_ranking(
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = load_checkpoint(checkpoint_path)
    priors = checkpoint.priors.batched()
    weights = np.asarray(priors.weights, dtype=np.float64)
    entropy = -(weights * np.log(np.clip(weights, EPS, None))).sum(axis=-1)
    return np.asarray(priors.gene_names), np.nan_to_num(
        entropy, nan=0.0, posinf=0.0, neginf=0.0
    )


def build_gene_list_payload(
    *,
    input_path: Path,
    method: str,
    top_k: int,
    gene_names: np.ndarray,
    scores: np.ndarray,
    metadata: dict[str, Any],
    descending: bool = True,
) -> dict[str, Any]:
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    order = order[:top_k]
    return {
        "source_path": str(input_path),
        "method": method,
        "top_k": int(top_k),
        "gene_indices": [int(i) for i in order.tolist()],
        "gene_names": [str(gene_names[i]) for i in order.tolist()],
        "scores": [float(scores[i]) for i in order.tolist()],
        "metadata": metadata,
    }


def write_ranked_gene_names(
    output_path: Path,
    gene_names: np.ndarray,
    scores: np.ndarray,
    *,
    descending: bool,
    limit: int | None,
) -> None:
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    if limit is not None:
        order = order[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(str(gene_names[idx]) for idx in order.tolist()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_ranked_genes = (
        None
        if args.output_ranked_genes is None
        else args.output_ranked_genes.expanduser().resolve()
    )
    descending = True

    if args.method == "hvg":
        adata = ad.read_h5ad(input_path)
        gene_names, scores = compute_hvg_ranking_from_adata(
            adata, flavor=args.hvg_flavor
        )
        metadata = {"hvg_flavor": args.hvg_flavor}
    elif args.method in {"lognorm-variance", "lognorm-dispersion"}:
        adata = ad.read_h5ad(input_path)
        gene_names, scores, metadata = compute_lognorm_ranking(
            adata, method=args.method
        )
    elif args.method in {"signal-hvg", "signal-variance", "signal-dispersion"}:
        adata = ad.read_h5ad(input_path)
        gene_names, scores, metadata = compute_signal_ranking(adata, method=args.method)
    elif args.method in {"prior-entropy", "prior-entropy-reverse"}:
        gene_names, scores = compute_prior_entropy_ranking(input_path)
        metadata = {
            "score_definition": "prior entropy of F_g",
            "score_mean": float(np.mean(scores)),
            "score_max": float(np.max(scores)),
            "score_min": float(np.min(scores)),
        }
        descending = args.method == "prior-entropy"
    else:
        raise ValueError(f"unsupported method: {args.method}")

    payload = build_gene_list_payload(
        input_path=input_path,
        method=args.method,
        top_k=args.top_k,
        gene_names=gene_names,
        scores=scores,
        metadata=metadata,
        descending=descending,
    )
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if output_ranked_genes is not None:
        write_ranked_gene_names(
            output_ranked_genes,
            gene_names,
            scores,
            descending=descending,
            limit=args.ranked_limit,
        )
    print(f"saved {output_json}")
    if output_ranked_genes is not None:
        print(f"saved ranked genes: {output_ranked_genes}")


if __name__ == "__main__":
    main()
