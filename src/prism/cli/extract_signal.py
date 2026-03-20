from __future__ import annotations

import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from scipy import sparse

from prism.model import CORE_CHANNELS, GeneBatch, Posterior, PriorEngine
from prism.model._typing import DTYPE_NP

console = Console()


def extract_signal(
    ckpt_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input checkpoint pickle path."
    ),
    h5ad_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad file."
    ),
    output_path: Path = typer.Argument(..., help="Output h5ad file path."),
    layer: str | None = typer.Option(
        None, help="AnnData layer name to use as input matrix."
    ),
    batch_size: int = typer.Option(256, min=1, help="Genes per extraction batch."),
    channels: list[str] | None = typer.Option(
        None,
        "--channel",
        help="Channels to write. Repeatable; defaults to core channels.",
    ),
    dtype: str = typer.Option(
        "float32", help="Output layer dtype: float32 or float64."
    ),
) -> int:
    ckpt_path = ckpt_path.resolve()
    h5ad_path = h5ad_path.resolve()
    output_path = output_path.resolve()

    console.print(f"[bold cyan]Loading[/bold cyan] checkpoint {ckpt_path}")
    with ckpt_path.open("rb") as fh:
        checkpoint = pickle.load(fh)

    engine = checkpoint.get("engine")
    s_hat = checkpoint.get("s_hat")
    if not isinstance(engine, PriorEngine):
        raise TypeError("checkpoint 中缺少合法的 PriorEngine")
    if s_hat is None:
        raise KeyError("checkpoint 中缺少 s_hat")

    console.print(f"[bold cyan]Reading[/bold cyan] {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    matrix = _select_matrix(adata, layer)
    gene_names = np.asarray(adata.var_names.astype(str))
    gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}

    fitted_gene_names = [
        name
        for name in engine.gene_names
        if name in gene_to_idx and engine.is_fitted(name)
    ]
    if not fitted_gene_names:
        raise ValueError("h5ad 与 checkpoint 没有可提取的已拟合基因交集")

    priors = engine.get_priors(fitted_gene_names)
    if priors is None:
        raise ValueError("无法从 checkpoint 中读取已拟合先验")

    posterior = Posterior(fitted_gene_names, priors)
    requested_channels = _resolve_channels(channels)
    output_dtype = _resolve_dtype(dtype)
    totals = _compute_totals(matrix)
    gene_positions = np.asarray(
        [gene_to_idx[name] for name in fitted_gene_names], dtype=np.int64
    )

    output_layers = {
        channel: np.full((adata.n_obs, adata.n_vars), np.nan, dtype=output_dtype)
        for channel in requested_channels
    }

    batch_offsets = list(range(0, len(fitted_gene_names), batch_size))
    with typer.progressbar(batch_offsets, label="Extracting signals") as progress:
        for offset in progress:
            batch_gene_names = fitted_gene_names[offset : offset + batch_size]
            batch_positions = gene_positions[offset : offset + batch_size]
            batch_counts = _slice_gene_counts_by_index(matrix, batch_positions)
            batch = GeneBatch(
                gene_names=batch_gene_names,
                counts=batch_counts,
                totals=totals,
            )
            extracted = posterior.extract(
                batch, s_hat=s_hat, channels=set(requested_channels)
            )
            for channel in requested_channels:
                output_layers[channel][:, batch_positions] = extracted[
                    channel
                ].T.astype(output_dtype, copy=False)

    for channel, values in output_layers.items():
        adata.layers[channel] = values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)

    _print_extract_summary(
        n_cells=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        extracted_genes=len(fitted_gene_names),
        channels=requested_channels,
        output_path=output_path,
    )
    return 0


def _resolve_channels(channels: list[str] | None) -> list[str]:
    if not channels:
        return sorted(CORE_CHANNELS)

    valid = set(CORE_CHANNELS) | {"surprisal_norm", "sharpness"}
    unknown = [channel for channel in channels if channel not in valid]
    if unknown:
        raise ValueError(f"未知 channel: {unknown}")
    return list(dict.fromkeys(channels))


def _resolve_dtype(dtype: str) -> np.dtype:
    if dtype == "float32":
        return np.dtype(np.float32)
    if dtype == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"不支持的 dtype: {dtype!r}")


def _select_matrix(adata: ad.AnnData, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} 不存在")
    return adata.layers[layer]


def _compute_totals(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        totals = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        totals = np.asarray(matrix, dtype=DTYPE_NP).sum(axis=1)
    return np.asarray(totals, dtype=DTYPE_NP).reshape(-1)


def _slice_gene_counts_by_index(matrix, gene_positions: np.ndarray) -> np.ndarray:
    subset = matrix[:, gene_positions]
    if sparse.issparse(subset):
        return np.asarray(subset.toarray(), dtype=DTYPE_NP)
    return np.asarray(subset, dtype=DTYPE_NP)


def _print_extract_summary(
    *,
    n_cells: int,
    n_genes: int,
    extracted_genes: int,
    channels: list[str],
    output_path: Path,
) -> None:
    table = Table(title="Extract Summary")
    table.add_column("Cells", justify="right")
    table.add_column("Genes", justify="right")
    table.add_column("Extracted", justify="right")
    table.add_column("Channels")
    table.add_row(
        str(n_cells),
        str(n_genes),
        str(extracted_genes),
        ", ".join(channels),
    )
    console.print(table)
    console.print(f"[bold green]Saved[/bold green] {output_path}")
