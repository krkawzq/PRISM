#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


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

    adata = ad.read_h5ad(input_path)
    if args.label_column not in adata.obs:
        raise KeyError(f"输入文件缺少标签列: {args.label_column!r}")

    sampled_idx = stratified_sample_indices(
        adata.obs[args.label_column],
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
    print(f"saved {output_path}")
    print(f"cells: {adata.n_obs} -> {sampled.n_obs}")
    print(f"genes: {adata.n_vars}")
    for label, count in counts.items():
        print(f"{label}: {int(count)}")


if __name__ == "__main__":
    main()
