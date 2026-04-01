from __future__ import annotations

from pathlib import Path
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import typer
from rich.console import Console

from prism.cli.common import print_key_value_table
from prism.io import write_h5ad_atomic
console = Console()


def stratified_sample_indices(
    labels: pd.Series,
    *,
    fraction: float,
    seed: int,
    per_class_min: int,
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")
    if per_class_min < 1:
        raise ValueError("--per-class-min must be >= 1")

    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    categories = labels.astype("category")
    codes = categories.cat.codes.to_numpy(copy=False)
    if np.any(codes < 0):
        raise ValueError("label column contains missing values")

    for code in np.unique(codes):
        class_idx = np.flatnonzero(codes == code)
        target_n = max(per_class_min, int(round(class_idx.size * fraction)))
        target_n = min(target_n, class_idx.size)
        chosen = rng.choice(class_idx, size=target_n, replace=False)
        sampled.append(np.sort(chosen))

    return np.sort(np.concatenate(sampled)).astype(np.int64, copy=False)


def downsample_command(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input AnnData file."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output downsampled AnnData file."
    ),
    label_key: str = typer.Option(
        "treatment",
        "--label-key",
        "--label-column",
        help="obs column used for stratified sampling.",
    ),
    fraction: float = typer.Option(
        ...,
        "--fraction",
        help="Fraction of cells retained per class.",
    ),
    seed: int = typer.Option(42, "--seed", min=0, help="Random seed."),
    per_class_min: int = typer.Option(
        1,
        "--per-class-min",
        min=1,
        help="Minimum number of cells retained per class.",
    ),
) -> int:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_path)
    if label_key not in adata.obs:
        raise KeyError(f"obs column {label_key!r} not found")

    sampled_idx = stratified_sample_indices(
        cast(pd.Series, adata.obs[label_key]),
        fraction=fraction,
        seed=seed,
        per_class_min=per_class_min,
    )
    sampled = adata[sampled_idx].copy()
    label_counts = (
        sampled.obs[label_key].astype("category").value_counts().sort_index()
    )
    sampled.uns["sampling"] = {
        "method": "stratified_fraction",
        "source_h5ad": str(input_path),
        "label_key": str(label_key),
        "fraction": float(fraction),
        "seed": int(seed),
        "per_class_min": int(per_class_min),
        "n_obs_before": int(adata.n_obs),
        "n_obs_after": int(sampled.n_obs),
        "n_vars": int(sampled.n_vars),
        "label_counts_after": {
            str(label): int(count) for label, count in label_counts.items()
        },
    }
    write_h5ad_atomic(sampled, output_path)

    print_key_value_table(
        console,
        title="Downsample AnnData",
        values={
            "Input": input_path,
            "Label key": label_key,
            "Fraction": fraction,
            "Cells": f"{adata.n_obs} -> {sampled.n_obs}",
            "Genes": sampled.n_vars,
            "Output": output_path,
        },
    )
    return 0


__all__ = ["downsample_command", "stratified_sample_indices"]
