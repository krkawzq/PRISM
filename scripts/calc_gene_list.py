#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

EPS = 1e-12
SIGNAL_LAYER = "signal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ranked gene lists by HVG, variance, or prior entropy and export JSON plus optional ordered gene names."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--output-ranked-genes",
        type=Path,
        default=None,
        help="Optional text file with one gene name per line in ranked order.",
    )
    parser.add_argument(
        "--ranked-limit",
        type=int,
        default=None,
        help="Optional cap for --output-ranked-genes. Defaults to all ranked genes.",
    )
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
        help="Scanpy HVG flavor used when --method hvg.",
    )
    return parser.parse_args()


def compute_hvg_ranking(
    input_path: Path, *, flavor: str
) -> tuple[np.ndarray, np.ndarray]:
    adata = ad.read_h5ad(input_path)
    return compute_hvg_ranking_from_adata(adata, flavor=flavor, layer=None)


def compute_hvg_ranking_from_adata(
    adata: ad.AnnData, *, flavor: str, layer: str | None
) -> tuple[np.ndarray, np.ndarray]:
    try:
        sc.pp.highly_variable_genes(
            adata,
            flavor=cast(Any, flavor),
            layer=layer,
            inplace=True,
        )
    except ImportError:
        if flavor not in {"seurat_v3", "seurat_v3_paper"}:
            raise
        fallback_flavor = "seurat"
        print(
            "warning: scanpy HVG flavor requires scikit-misc; falling back to normalize_total + log1p + 'seurat'"
        )
        if layer is not None:
            raise ValueError(
                "layer-backed HVG fallback is not supported when scikit-misc is missing"
            )
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(
            adata,
            flavor=cast(Any, fallback_flavor),
            layer=layer,
            inplace=True,
        )
        flavor = fallback_flavor
    if "highly_variable_rank" in adata.var:
        rank = np.asarray(adata.var["highly_variable_rank"], dtype=np.float64)
        score = -np.nan_to_num(rank, nan=np.inf)
    elif "dispersions_norm" in adata.var:
        score = np.asarray(adata.var["dispersions_norm"], dtype=np.float64)
    elif "variances_norm" in adata.var:
        score = np.asarray(adata.var["variances_norm"], dtype=np.float64)
    else:
        raise ValueError("scanpy 未产出可用于排序的 HVG 字段")
    score = np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    return np.asarray(adata.var_names), score


def compute_signal_hvg_ranking(
    input_path: Path, *, flavor: str
) -> tuple[np.ndarray, np.ndarray]:
    adata = ad.read_h5ad(input_path)
    if SIGNAL_LAYER not in adata.layers:
        raise KeyError(f"输入文件缺少必需 layer: {SIGNAL_LAYER!r}")
    if flavor in {"seurat_v3", "seurat_v3_paper"}:
        print(
            "warning: signal-hvg uses continuous signal values; switching HVG flavor to 'seurat'"
        )
        flavor = "seurat"
    signal = np.asarray(adata.layers[SIGNAL_LAYER], dtype=np.float64)
    signal = np.clip(np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    signal = np.log1p(signal)
    signal_adata = ad.AnnData(X=signal, obs=adata.obs.copy(), var=adata.var.copy())
    return compute_hvg_ranking_from_adata(signal_adata, flavor=flavor, layer=None)


def compute_totals(adata: ad.AnnData) -> np.ndarray:
    matrix = adata.X
    if matrix is None:
        raise ValueError("输入 h5ad 的 X 为空")
    matrix_any = cast(Any, matrix)
    if sparse.issparse(matrix_any):
        totals = np.asarray(matrix_any.sum(axis=1)).reshape(-1)
    else:
        totals = np.asarray(matrix_any).sum(axis=1)
    return np.asarray(totals, dtype=np.float32)


def log1p_normalize_total(
    counts: np.ndarray,
    totals: np.ndarray,
    *,
    target: float,
) -> np.ndarray:
    scale = np.maximum(totals.astype(np.float64, copy=False), 1.0)
    normalized = counts.astype(np.float64, copy=False) * (target / scale)[:, None]
    return np.log1p(normalized)


def compute_lognorm_ranking(
    input_path: Path, *, method: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    adata = ad.read_h5ad(input_path)
    matrix = adata.X
    if matrix is None:
        raise ValueError("输入 h5ad 的 X 为空")
    totals = compute_totals(adata)
    target = float(np.median(totals))
    matrix_any = cast(Any, matrix)
    counts = (
        matrix_any.toarray() if sparse.issparse(matrix_any) else np.asarray(matrix_any)
    )
    values = np.asarray(
        log1p_normalize_total(
            np.asarray(counts, dtype=np.float32), totals, target=target
        ),
        dtype=np.float32,
    )
    mean = np.mean(values, axis=0, dtype=np.float64)
    var = np.var(values, axis=0, dtype=np.float64)
    if method == "lognorm-variance":
        score = var
        score_definition = "variance(log1p_normalize_total(X))"
    elif method == "lognorm-dispersion":
        score = var / np.maximum(mean, EPS)
        score_definition = (
            "variance(log1p_normalize_total(X)) / mean(log1p_normalize_total(X))"
        )
    else:
        raise ValueError(f"未知 lognorm 排序方法: {method!r}")
    score = np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    return (
        np.asarray(adata.var_names),
        np.asarray(score, dtype=np.float64),
        {
            "normalization_target": target,
            "score_definition": score_definition,
        },
    )


def compute_signal_ranking(
    input_path: Path, *, method: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    adata = ad.read_h5ad(input_path)
    if SIGNAL_LAYER not in adata.layers:
        raise KeyError(f"输入文件缺少必需 layer: {SIGNAL_LAYER!r}")
    values = np.asarray(adata.layers[SIGNAL_LAYER], dtype=np.float32)
    mean = np.mean(values, axis=0, dtype=np.float64)
    var = np.var(values, axis=0, dtype=np.float64)
    if method == "signal-variance":
        score = var
        score_definition = "variance(signal)"
    elif method == "signal-dispersion":
        score = var / np.maximum(mean, EPS)
        score_definition = "variance(signal) / mean(signal)"
    else:
        raise ValueError(f"未知 signal 排序方法: {method!r}")
    score = np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    return (
        np.asarray(adata.var_names),
        np.asarray(score, dtype=np.float64),
        {
            "layer": SIGNAL_LAYER,
            "score_definition": score_definition,
        },
    )


def compute_prior_entropy_ranking(input_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with input_path.open("rb") as fh:
        checkpoint = pickle.load(fh)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint 不是合法字典")
    gene_names = checkpoint.get("gene_names")
    engine = checkpoint.get("engine")
    if not isinstance(gene_names, list) or not all(
        isinstance(name, str) for name in gene_names
    ):
        raise TypeError("checkpoint 中缺少合法 gene_names")
    if engine is None or not hasattr(engine, "get_priors"):
        raise TypeError("checkpoint 中缺少可读取先验的 engine")
    priors = engine.get_priors(gene_names)
    if priors is None:
        raise ValueError("checkpoint 中存在未拟合基因，无法计算先验熵")
    weights = np.asarray(priors.weights, dtype=np.float64)
    entropy = -(weights * np.log(np.clip(weights, EPS, None))).sum(axis=-1)
    entropy = np.clip(
        np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None
    )
    return np.asarray(gene_names), entropy


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
    if top_k < 1:
        raise ValueError(f"top_k 必须 >= 1，收到 {top_k}")
    if top_k > gene_names.shape[0]:
        raise ValueError(f"top_k={top_k} 超过基因数 {gene_names.shape[0]}")
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


def rank_gene_indices(scores: np.ndarray, *, descending: bool = True) -> np.ndarray:
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    return np.asarray(order, dtype=np.int64)


def write_ranked_gene_names(
    output_path: Path,
    gene_names: np.ndarray,
    scores: np.ndarray,
    *,
    descending: bool,
    limit: int | None,
) -> None:
    order = rank_gene_indices(scores, descending=descending)
    if limit is not None:
        if limit < 1:
            raise ValueError(f"ranked_limit 必须 >= 1，收到 {limit}")
        order = order[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(gene_names[idx]) for idx in order.tolist()]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        gene_names, scores = compute_hvg_ranking(input_path, flavor=args.hvg_flavor)
        payload = build_gene_list_payload(
            input_path=input_path,
            method="hvg",
            top_k=args.top_k,
            gene_names=gene_names,
            scores=scores,
            metadata={"hvg_flavor": args.hvg_flavor},
        )
    elif args.method == "signal-hvg":
        gene_names, scores = compute_signal_hvg_ranking(
            input_path, flavor=args.hvg_flavor
        )
        payload = build_gene_list_payload(
            input_path=input_path,
            method="signal-hvg",
            top_k=args.top_k,
            gene_names=gene_names,
            scores=scores,
            metadata={"hvg_flavor": args.hvg_flavor, "layer": SIGNAL_LAYER},
        )
    elif args.method in {"lognorm-variance", "lognorm-dispersion"}:
        gene_names, scores, metadata = compute_lognorm_ranking(
            input_path, method=args.method
        )
        payload = build_gene_list_payload(
            input_path=input_path,
            method=args.method,
            top_k=args.top_k,
            gene_names=gene_names,
            scores=scores,
            metadata=metadata,
        )
    elif args.method in {"signal-variance", "signal-dispersion"}:
        gene_names, scores, metadata = compute_signal_ranking(
            input_path, method=args.method
        )
        payload = build_gene_list_payload(
            input_path=input_path,
            method=args.method,
            top_k=args.top_k,
            gene_names=gene_names,
            scores=scores,
            metadata=metadata,
        )
    elif args.method in {"prior-entropy", "prior-entropy-reverse"}:
        gene_names, scores = compute_prior_entropy_ranking(input_path)
        descending = args.method == "prior-entropy"
        payload = build_gene_list_payload(
            input_path=input_path,
            method=args.method,
            top_k=args.top_k,
            gene_names=gene_names,
            scores=scores,
            metadata={
                "score_mean": float(np.mean(scores)),
                "score_max": float(np.max(scores)),
                "score_min": float(np.min(scores)),
                "score_definition": "prior entropy of F_g",
            },
            descending=descending,
        )
    else:
        raise ValueError(f"未知 method: {args.method!r}")

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
        rank_scope = "all" if args.ranked_limit is None else str(args.ranked_limit)
        print(f"saved ranked genes: {output_ranked_genes} (limit={rank_scope})")
    print(f"method: {payload['method']}")
    print(f"top_k : {payload['top_k']}")
    print(f"first5 : {payload['gene_names'][:5]}")


if __name__ == "__main__":
    main()
