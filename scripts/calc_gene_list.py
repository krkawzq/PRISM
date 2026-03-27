#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, cast

import anndata as ad
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prism.model import load_checkpoint

EPS = 1e-12
SIGNAL_LAYER = "signal"
console = Console()

install_rich_traceback(show_locals=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ranked gene lists from h5ad or checkpoint inputs."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-ranked-genes", type=Path, required=True)
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
        "--restrict-genes",
        type=Path,
        default=None,
        help="Optional text file with one gene per line. Only these genes are ranked.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Maximum number of cells to use for h5ad-based methods. Randomly subsamples if needed.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used for cell subsampling.",
    )
    parser.add_argument(
        "--hvg-flavor",
        choices=("seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"),
        default="seurat_v3",
    )
    parser.add_argument(
        "--prior-source",
        choices=("global", "label"),
        default="global",
        help="Checkpoint prior source used by prior-entropy methods.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label name used when --prior-source label.",
    )
    args = parser.parse_args()
    if args.max_cells is not None and args.max_cells <= 0:
        raise ValueError("--max-cells must be positive when provided")
    return args


def read_gene_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
                "restrict_genes_path": None,
                "n_requested_restrict_genes": None,
                "n_missing_restrict_genes": 0,
                "missing_restrict_genes": [],
                "n_ranked_genes": int(gene_names.shape[0]),
            }
        )
        return gene_names, scores, metadata

    requested = []
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
    *,
    prior_source: str,
    label: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = load_checkpoint(checkpoint_path)
    if prior_source == "global":
        if checkpoint.priors is None:
            raise ValueError("checkpoint does not contain global priors")
        priors = checkpoint.priors.batched()
    elif prior_source == "label":
        if label is None or not label.strip():
            raise ValueError("--label is required when --prior-source label")
        if label not in checkpoint.label_priors:
            available = sorted(checkpoint.label_priors)
            raise ValueError(
                f"checkpoint does not contain label priors for {label!r}; available labels: {available}"
            )
        priors = checkpoint.label_priors[label].batched()
    else:
        raise ValueError(f"unsupported prior source: {prior_source}")
    weights = np.asarray(priors.weights, dtype=np.float64)
    entropy = -(weights * np.log(np.clip(weights, EPS, None))).sum(axis=-1)
    return np.asarray(priors.gene_names), np.nan_to_num(
        entropy, nan=0.0, posinf=0.0, neginf=0.0
    )


def rank_gene_scores(
    gene_names: np.ndarray, scores: np.ndarray, *, descending: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    return order, gene_names[order], scores[order]


def build_gene_list_payload(
    *,
    input_path: Path,
    method: str,
    ranked_gene_names: np.ndarray,
    ranked_scores: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_path": str(input_path),
        "method": method,
        "gene_names": [str(gene) for gene in ranked_gene_names.tolist()],
        "scores": [float(score) for score in ranked_scores.tolist()],
        "metadata": metadata,
    }


def write_ranked_gene_names(output_path: Path, ranked_gene_names: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(str(gene) for gene in ranked_gene_names.tolist()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_ranked_genes = args.output_ranked_genes.expanduser().resolve()
    restrict_genes_path = (
        None
        if args.restrict_genes is None
        else args.restrict_genes.expanduser().resolve()
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_ranked_genes.parent.mkdir(parents=True, exist_ok=True)

    restrict_genes = (
        None if restrict_genes_path is None else read_gene_list(restrict_genes_path)
    )
    descending = True
    metadata: dict[str, Any] = {
        "restrict_genes_path": None
        if restrict_genes_path is None
        else str(restrict_genes_path),
    }

    job_table = Table(show_header=False, box=None)
    job_table.add_row("Input", str(input_path))
    job_table.add_row("Method", args.method)
    job_table.add_row(
        "Restrict genes", str(restrict_genes_path) if restrict_genes_path else "None"
    )
    job_table.add_row(
        "Max cells", str(args.max_cells) if args.max_cells is not None else "All"
    )
    console.print(Panel(job_table, title="Calc Gene List", border_style="cyan"))

    if args.method == "hvg":
        with console.status("Loading AnnData and computing HVG ranking..."):
            adata = ad.read_h5ad(input_path)
            adata, sampling_metadata = maybe_subsample_adata(
                adata, max_cells=args.max_cells, random_seed=args.random_seed
            )
            gene_names, scores = compute_hvg_ranking_from_adata(
                adata, flavor=args.hvg_flavor
            )
        metadata.update(sampling_metadata)
        metadata.update({"hvg_flavor": args.hvg_flavor})
    elif args.method in {"lognorm-variance", "lognorm-dispersion"}:
        with console.status("Loading AnnData and computing log-normalized ranking..."):
            adata = ad.read_h5ad(input_path)
            adata, sampling_metadata = maybe_subsample_adata(
                adata, max_cells=args.max_cells, random_seed=args.random_seed
            )
            gene_names, scores, method_metadata = compute_lognorm_ranking(
                adata, method=args.method
            )
        metadata.update(sampling_metadata)
        metadata.update(method_metadata)
    elif args.method in {"signal-hvg", "signal-variance", "signal-dispersion"}:
        with console.status("Loading AnnData and computing signal ranking..."):
            adata = ad.read_h5ad(input_path)
            adata, sampling_metadata = maybe_subsample_adata(
                adata, max_cells=args.max_cells, random_seed=args.random_seed
            )
            gene_names, scores, method_metadata = compute_signal_ranking(
                adata, method=args.method
            )
        metadata.update(sampling_metadata)
        metadata.update(method_metadata)
    elif args.method in {"prior-entropy", "prior-entropy-reverse"}:
        with console.status(
            "Loading checkpoint and computing prior entropy ranking..."
        ):
            gene_names, scores = compute_prior_entropy_ranking(
                input_path,
                prior_source=args.prior_source,
                label=args.label,
            )
        metadata.update(
            {
                "score_definition": "prior entropy of F_g",
                "score_mean": float(np.mean(scores)),
                "score_max": float(np.max(scores)),
                "score_min": float(np.min(scores)),
                "prior_source": args.prior_source,
                "label": args.label,
                "n_total_cells": None,
                "n_used_cells": None,
                "cell_sampling_applied": False,
                "max_cells": None if args.max_cells is None else int(args.max_cells),
                "random_seed": int(args.random_seed),
                "cell_sampling_ignored": True,
            }
        )
        descending = args.method == "prior-entropy"
    else:
        raise ValueError(f"unsupported method: {args.method}")

    with console.status("Filtering requested genes and sorting results..."):
        gene_names, scores, filter_metadata = filter_gene_scores(
            gene_names,
            scores,
            restrict_genes=restrict_genes,
        )
        _, ranked_gene_names, ranked_scores = rank_gene_scores(
            gene_names,
            scores,
            descending=descending,
        )
    metadata.update(filter_metadata)
    payload = build_gene_list_payload(
        input_path=input_path,
        method=args.method,
        ranked_gene_names=ranked_gene_names,
        ranked_scores=ranked_scores,
        metadata=metadata,
    )
    with console.status("Writing outputs..."):
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_ranked_gene_names(output_ranked_genes, ranked_gene_names)

    summary = Table(title="Ranking Summary")
    summary.add_column("Field")
    summary.add_column("Value", overflow="fold")
    summary.add_row("Ranked genes", str(len(ranked_gene_names)))
    summary.add_row("Cells used", str(metadata.get("n_used_cells", "N/A")))
    summary.add_row(
        "Sampling applied", str(metadata.get("cell_sampling_applied", False))
    )
    summary.add_row(
        "Missing restricted genes", str(metadata.get("n_missing_restrict_genes", 0))
    )
    summary.add_row("JSON output", str(output_json))
    summary.add_row("Ranked genes output", str(output_ranked_genes))
    console.print(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(
            Panel(str(exc), title="calc_gene_list failed", border_style="red")
        )
        raise SystemExit(1) from exc
