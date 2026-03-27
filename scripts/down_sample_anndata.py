#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback
from typing import cast

console = Console()

install_rich_traceback(show_locals=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a stratified subsampled AnnData file."
    )
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_h5ad", type=Path)
    parser.add_argument("--label-column", type=str, default="treatment")
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--per-class-min",
        type=int,
        default=1,
        help="Minimum number of cells retained per class.",
    )
    return parser.parse_args()


def stratified_sample_indices(
    labels: pd.Series,
    *,
    fraction: float,
    seed: int,
    per_class_min: int,
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction 必须在 (0, 1] 内，收到 {fraction}")
    if per_class_min < 1:
        raise ValueError(f"per_class_min 必须 >= 1，收到 {per_class_min}")

    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    categories = labels.astype("category")
    codes = categories.cat.codes.to_numpy(copy=False)
    if np.any(codes < 0):
        raise ValueError("标签列包含缺失值，无法分层采样")

    for code in np.unique(codes):
        class_idx = np.flatnonzero(codes == code)
        target_n = max(per_class_min, int(round(class_idx.size * fraction)))
        target_n = min(target_n, class_idx.size)
        chosen = rng.choice(class_idx, size=target_n, replace=False)
        sampled.append(np.sort(chosen))

    return np.sort(np.concatenate(sampled)).astype(np.int64, copy=False)


def main() -> None:
    args = parse_args()
    input_path = args.input_h5ad.expanduser().resolve()
    output_path = args.output_h5ad.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    job = Table(show_header=False, box=None)
    job.add_row("Input", str(input_path))
    job.add_row("Output", str(output_path))
    job.add_row("Label column", args.label_column)
    job.add_row("Fraction", str(args.fraction))
    job.add_row("Seed", str(args.seed))
    console.print(Panel(job, title="Down Sample AnnData", border_style="cyan"))

    with console.status("Loading AnnData and sampling cells..."):
        adata = ad.read_h5ad(input_path)
    if args.label_column not in adata.obs:
        raise KeyError(f"输入文件缺少标签列: {args.label_column!r}")

    with console.status("Computing stratified sample..."):
        sampled_idx = stratified_sample_indices(
            cast(pd.Series, adata.obs[args.label_column]),
            fraction=args.fraction,
            seed=args.seed,
            per_class_min=args.per_class_min,
        )
        sampled = adata[sampled_idx].copy()
        sampled.uns["sampling"] = {
            "source_h5ad": str(input_path),
            "label_column": args.label_column,
            "fraction": float(args.fraction),
            "seed": int(args.seed),
            "per_class_min": int(args.per_class_min),
            "n_obs_before": int(adata.n_obs),
            "n_obs_after": int(sampled.n_obs),
        }
        sampled.write_h5ad(output_path)

    counts = (
        sampled.obs[args.label_column].astype("category").value_counts().sort_index()
    )

    summary = Table(title="Sampling Summary")
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("Cells", f"{adata.n_obs} -> {sampled.n_obs}")
    summary.add_row("Genes", str(adata.n_vars))
    summary.add_row("Output", str(output_path))
    console.print(summary)

    class_table = Table(title="Per-Class Counts")
    class_table.add_column(args.label_column)
    class_table.add_column("Cells", justify="right")
    for label, count in counts.items():
        class_table.add_row(str(label), str(int(count)))
    console.print(class_table)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(
            Panel(str(exc), title="down_sample_anndata failed", border_style="red")
        )
        raise SystemExit(1) from exc
