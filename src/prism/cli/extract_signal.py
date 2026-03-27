from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import cast

import anndata as ad
import numpy as np
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from scipy import sparse

from prism.model import (
    CORE_CHANNELS,
    ObservationBatch,
    Posterior,
    SignalChannel,
    load_checkpoint,
)

extract_app = typer.Typer(
    help="Extract signal layers from checkpoints.", no_args_is_help=True
)
console = Console()


@extract_app.command("signals")
def extract_signals_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input checkpoint path."
    ),
    h5ad_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad file."
    ),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output h5ad path."),
    layer: str | None = typer.Option(None, help="Input AnnData layer. Defaults to X."),
    genes_path: Path | None = typer.Option(
        None,
        "--genes",
        exists=True,
        dir_okay=False,
        help="Optional text file restricting extracted genes.",
    ),
    output_mode: str = typer.Option(
        "fitted-only", help="Output layout: fitted-only or full-matrix."
    ),
    batch_size: int = typer.Option(128, min=1, help="Genes per extraction batch."),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda."),
    dtype: str = typer.Option("float32", help="Output dtype: float32 or float64."),
    channels: list[str] | None = typer.Option(
        None,
        "--channel",
        help="Repeatable channel selection. Defaults to core channels.",
    ),
    dry_run: bool = typer.Option(
        False, help="Show the execution plan without writing output."
    ),
) -> int:
    start_time = perf_counter()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    h5ad_path = h5ad_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    genes_path = None if genes_path is None else genes_path.expanduser().resolve()

    checkpoint = load_checkpoint(checkpoint_path)
    reference_gene_names = _require_reference_genes(checkpoint.metadata)
    selected_channels = _resolve_channels(channels)
    output_dtype = _resolve_dtype(dtype)

    console.print(f"[bold cyan]Reading[/bold cyan] {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    matrix = _select_matrix(adata, layer)
    dataset_gene_names = [str(name) for name in adata.var_names.tolist()]
    gene_to_idx = {name: idx for idx, name in enumerate(dataset_gene_names)}

    ref_positions = [
        gene_to_idx[name] for name in reference_gene_names if name in gene_to_idx
    ]
    if not ref_positions:
        raise ValueError(
            "checkpoint reference genes do not overlap with the input dataset"
        )
    reference_counts = _compute_reference_counts(matrix, ref_positions)

    requested_genes = (
        checkpoint.gene_names if genes_path is None else _read_gene_list(genes_path)
    )
    selected_genes = [
        name
        for name in requested_genes
        if name in gene_to_idx and name in set(checkpoint.gene_names)
    ]
    if not selected_genes:
        raise ValueError(
            "no selected genes overlap between the dataset and the checkpoint"
        )

    _print_extract_plan(
        checkpoint_path=checkpoint_path,
        h5ad_path=h5ad_path,
        layer=layer,
        n_cells=int(adata.n_obs),
        n_dataset_genes=int(adata.n_vars),
        n_checkpoint_genes=len(checkpoint.gene_names),
        n_selected_genes=len(selected_genes),
        n_reference_genes=len(reference_gene_names),
        output_mode=output_mode,
        channels=",".join(selected_channels),
        output_path=output_path,
        device=device,
    )
    if dry_run:
        return 0

    posterior = Posterior(
        selected_genes, checkpoint.priors.subset(selected_genes), device=device
    )
    if output_mode == "fitted-only":
        output_adata = adata[:, [gene_to_idx[name] for name in selected_genes]].copy()
        output_positions = {name: idx for idx, name in enumerate(selected_genes)}
        shape = (int(output_adata.n_obs), int(output_adata.n_vars))
    elif output_mode == "full-matrix":
        output_adata = adata.copy()
        output_positions = gene_to_idx
        shape = (int(output_adata.n_obs), int(output_adata.n_vars))
    else:
        raise ValueError("output_mode must be 'fitted-only' or 'full-matrix'")

    layer_arrays = {
        channel: np.full(shape, np.nan, dtype=output_dtype)
        for channel in selected_channels
    }
    batches = [
        selected_genes[start : start + batch_size]
        for start in range(0, len(selected_genes), batch_size)
    ]
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("extracting", total=len(batches))
        for batch_index, batch_names in enumerate(batches, start=1):
            batch_counts = _slice_gene_counts(
                matrix, [gene_to_idx[name] for name in batch_names]
            )
            extracted = posterior.extract(
                ObservationBatch(
                    gene_names=batch_names,
                    counts=batch_counts,
                    reference_counts=reference_counts,
                ),
                channels=cast(set[SignalChannel], set(selected_channels)),
            )
            for channel in selected_channels:
                values = np.asarray(extracted[channel], dtype=np.float64)
                for local_idx, gene_name in enumerate(batch_names):
                    layer_arrays[channel][:, output_positions[gene_name]] = values[
                        :, local_idx
                    ].astype(output_dtype, copy=False)
            progress.update(
                task_id,
                advance=1,
                description=f"extracting ({batch_index}/{len(batches)})",
            )

    for channel, values in layer_arrays.items():
        output_adata.layers[channel] = values
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_adata.write_h5ad(output_path)
    _print_extract_summary(
        output_path=output_path,
        elapsed_sec=perf_counter() - start_time,
        n_genes=len(selected_genes),
        channels=selected_channels,
    )
    return 0


def _require_reference_genes(metadata: dict[str, object]) -> list[str]:
    value = metadata.get("reference_gene_names")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("checkpoint metadata is missing reference_gene_names")
    return list(value)


def _resolve_channels(channels: list[str] | None) -> list[str]:
    if not channels:
        return sorted(CORE_CHANNELS)
    valid = set(CORE_CHANNELS) | {"map_p", "map_mu"}
    unknown = [channel for channel in channels if channel not in valid]
    if unknown:
        raise ValueError(f"unknown channels: {unknown}")
    return list(dict.fromkeys(channels))


def _resolve_dtype(value: str) -> np.dtype:
    if value == "float32":
        return np.dtype(np.float32)
    if value == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"unsupported dtype: {value}")


def _read_gene_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _select_matrix(adata: ad.AnnData, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} does not exist")
    return adata.layers[layer]


def _slice_gene_counts(matrix, positions: list[int]) -> np.ndarray:
    subset = matrix[:, positions]
    if sparse.issparse(subset):
        return np.asarray(subset.toarray(), dtype=np.float64)
    return np.asarray(subset, dtype=np.float64)


def _compute_reference_counts(matrix, positions: list[int]) -> np.ndarray:
    subset = matrix[:, positions]
    if sparse.issparse(subset):
        totals = np.asarray(subset.sum(axis=1)).reshape(-1)
    else:
        totals = np.asarray(subset, dtype=np.float64).sum(axis=1)
    return np.asarray(totals, dtype=np.float64).reshape(-1)


def _print_extract_plan(**values: object) -> None:
    table = Table(title="Extract Plan")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(key, str(value))
    console.print(table)


def _print_extract_summary(
    *, output_path: Path, elapsed_sec: float, n_genes: int, channels: list[str]
) -> None:
    table = Table(title="Extract Summary")
    table.add_column("Genes", justify="right")
    table.add_column("Channels")
    table.add_row(str(n_genes), ", ".join(channels))
    console.print(table)
    console.print(f"[bold green]Saved[/bold green] {output_path}")
    console.print(f"[bold green]Elapsed[/bold green] {elapsed_sec:.2f}s")
