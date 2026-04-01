from __future__ import annotations

from pathlib import Path
from time import perf_counter

import anndata as ad
import numpy as np
import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from prism.model import load_checkpoint
from prism.cli.checkpoint_validation import resolve_cli_checkpoint_distribution

from .common import (
    compute_reference_counts,
    extract_batch,
    print_extract_plan,
    print_extract_summary,
    read_gene_list,
    require_reference_genes,
    resolve_channels,
    resolve_dtype,
    resolve_prior_source,
    select_matrix,
    slice_gene_counts,
    console,
)


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
    prior_source: str = typer.Option("global", help="Prior source: global or label."),
    label_key: str | None = typer.Option(
        None, help="Obs column used when --prior-source label."
    ),
    batch_size: int = typer.Option(
        128,
        min=1,
        help="Number of genes per extraction batch. Controls memory usage during signal extraction.",
    ),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda."),
    torch_dtype: str = typer.Option(
        "float32", help="Torch dtype for inference: float32 or float64."
    ),
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
    resolve_cli_checkpoint_distribution(
        checkpoint,
        command_name="prism extract signals",
        allow_distributions={"binomial", "negative_binomial", "poisson"},
    )
    reference_gene_names = require_reference_genes(checkpoint.metadata)
    selected_channels = resolve_channels(channels)
    output_dtype = resolve_dtype(dtype)
    prior_source_resolved = resolve_prior_source(prior_source)

    console.print(f"[bold cyan]Reading[/bold cyan] {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    matrix = select_matrix(adata, layer)
    dataset_gene_names = [str(name) for name in adata.var_names.tolist()]
    gene_to_idx = {name: idx for idx, name in enumerate(dataset_gene_names)}
    ref_positions = [
        gene_to_idx[name] for name in reference_gene_names if name in gene_to_idx
    ]
    if not ref_positions:
        raise ValueError(
            "checkpoint reference genes do not overlap with the input dataset"
        )
    reference_counts = compute_reference_counts(matrix, ref_positions)
    available_gene_names = checkpoint.gene_names
    if prior_source_resolved == "global" and checkpoint.priors is None:
        raise ValueError("checkpoint does not contain global priors")
    if prior_source_resolved == "label" and not checkpoint.label_priors:
        raise ValueError("checkpoint does not contain label-specific priors")
    requested_genes = (
        available_gene_names if genes_path is None else read_gene_list(genes_path)
    )
    selected_genes = [
        name
        for name in requested_genes
        if name in gene_to_idx and name in set(available_gene_names)
    ]
    if not selected_genes:
        raise ValueError(
            "no selected genes overlap between the dataset and the checkpoint"
        )
    print_extract_plan(
        checkpoint_path=checkpoint_path,
        h5ad_path=h5ad_path,
        layer=layer,
        n_cells=int(adata.n_obs),
        n_dataset_genes=int(adata.n_vars),
        n_checkpoint_genes=len(checkpoint.gene_names),
        n_selected_genes=len(selected_genes),
        n_reference_genes=len(reference_gene_names),
        prior_source=prior_source_resolved,
        label_key=label_key,
        output_mode=output_mode,
        channels=",".join(selected_channels),
        output_path=output_path,
        device=device,
    )
    if dry_run:
        return 0
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
            batch_counts = slice_gene_counts(
                matrix, [gene_to_idx[name] for name in batch_names]
            )
            extracted = extract_batch(
                checkpoint=checkpoint,
                adata=adata,
                batch_names=batch_names,
                batch_counts=batch_counts,
                reference_counts=reference_counts,
                prior_source=prior_source_resolved,
                label_key=label_key,
                device=device,
                torch_dtype=torch_dtype,
                selected_channels=selected_channels,
            )
            batch_positions = np.asarray(
                [output_positions[name] for name in batch_names], dtype=np.int64
            )
            for channel in selected_channels:
                values = np.asarray(extracted[channel], dtype=output_dtype)
                layer_arrays[channel][:, batch_positions] = values
            progress.update(
                task_id,
                advance=1,
                description=f"extracting ({batch_index}/{len(batches)})",
            )
    for channel, values in layer_arrays.items():
        output_adata.layers[channel] = values
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_adata.write_h5ad(output_path)
    print_extract_summary(
        output_path=output_path,
        elapsed_sec=perf_counter() - start_time,
        n_genes=len(selected_genes),
        channels=selected_channels,
    )
    return 0


__all__ = ["extract_signals_command"]
