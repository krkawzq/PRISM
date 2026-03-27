from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, UTC
from pathlib import Path
from time import perf_counter
from typing import Any, cast

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
    ModelCheckpoint,
    ObservationBatch,
    PriorFitConfig,
    PriorGrid,
    ScaleMetadata,
    fit_gene_priors,
    save_checkpoint,
)

fit_app = typer.Typer(help="Fit prior checkpoints.", no_args_is_help=True)
console = Console()


@fit_app.command("priors")
def fit_priors_command(
    h5ad_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad file."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output checkpoint path."
    ),
    reference_genes_path: Path = typer.Option(
        ...,
        "--reference-genes",
        exists=True,
        dir_okay=False,
        help="Text file with reference gene names used to compute reference counts.",
    ),
    fit_genes_path: Path | None = typer.Option(
        None,
        "--fit-genes",
        exists=True,
        dir_okay=False,
        help="Optional text file with genes to fit. Defaults to all genes in the dataset.",
    ),
    layer: str | None = typer.Option(None, help="Input AnnData layer. Defaults to X."),
    S: float | None = typer.Option(
        None,
        min=1e-12,
        help="Global scale marker used by the model. Defaults to mean reference count N_avg.",
    ),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda."),
    gene_batch_size: int = typer.Option(
        64, min=1, help="Number of genes per fit batch."
    ),
    shard: str = typer.Option(
        "0/1", help="Shard specification as rank/world, e.g. 0/4."
    ),
    grid_size: int = typer.Option(512, min=2, help="Prior grid size."),
    sigma_bins: float = typer.Option(
        1.0, min=0.0, help="Gaussian smoothing sigma in grid bins."
    ),
    align_loss_weight: float = typer.Option(
        1.0, min=0.0, help="Posterior-prior alignment weight."
    ),
    lr: float = typer.Option(0.05, min=1e-12, help="Optimizer learning rate."),
    n_iter: int = typer.Option(100, min=1, help="Optimization iterations."),
    lr_min_ratio: float = typer.Option(
        0.1, min=0.0, help="Scheduler minimum lr ratio."
    ),
    grad_clip: float | None = typer.Option(
        None, min=0.0, help="Optional gradient clipping threshold."
    ),
    init_temperature: float = typer.Option(
        1.0, min=1e-12, help="Initialization temperature."
    ),
    cell_chunk_size: int = typer.Option(
        512, min=1, help="Likelihood chunk size over cells."
    ),
    optimizer: str = typer.Option("adamw", help="Optimizer name."),
    scheduler: str = typer.Option("cosine", help="Scheduler name."),
    torch_dtype: str = typer.Option("float64", help="Torch dtype: float64 or float32."),
    dry_run: bool = typer.Option(
        False, help="Show the execution plan without fitting."
    ),
) -> int:
    start_time = perf_counter()
    h5ad_path = h5ad_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    reference_genes_path = reference_genes_path.expanduser().resolve()
    fit_genes_path = (
        None if fit_genes_path is None else fit_genes_path.expanduser().resolve()
    )

    rank, world_size = _parse_shard(shard)
    fit_config = PriorFitConfig(
        grid_size=grid_size,
        sigma_bins=sigma_bins,
        align_loss_weight=align_loss_weight,
        lr=lr,
        n_iter=n_iter,
        lr_min_ratio=lr_min_ratio,
        grad_clip=grad_clip,
        init_temperature=init_temperature,
        cell_chunk_size=cell_chunk_size,
        optimizer=cast(Any, optimizer),
        scheduler=cast(Any, scheduler),
        torch_dtype=cast(Any, torch_dtype),
    )

    console.print(f"[bold cyan]Reading[/bold cyan] {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    matrix = _select_matrix(adata, layer)
    gene_names = [str(name) for name in adata.var_names.tolist()]
    gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}

    reference_gene_names, missing_reference = _resolve_gene_list(
        reference_genes_path, gene_to_idx
    )
    if not reference_gene_names:
        raise ValueError("reference gene list has no overlap with the dataset")
    fit_gene_names, missing_fit = _resolve_fit_gene_list(
        fit_genes_path, gene_names, gene_to_idx
    )
    if not fit_gene_names:
        raise ValueError("fit gene list is empty after intersecting with the dataset")
    shard_gene_names = _shard_gene_names(
        fit_gene_names, rank=rank, world_size=world_size
    )
    if not shard_gene_names:
        raise ValueError(f"shard {rank}/{world_size} has no assigned genes")

    reference_counts = _compute_reference_counts(
        matrix, [gene_to_idx[name] for name in reference_gene_names]
    )
    default_S = float(np.mean(reference_counts))
    resolved_S = default_S if S is None else float(S)
    S_source = "default:N_avg" if S is None else "user"
    _print_fit_plan(
        h5ad_path=h5ad_path,
        layer=layer,
        n_cells=int(adata.n_obs),
        n_dataset_genes=int(adata.n_vars),
        n_reference_genes=len(reference_gene_names),
        n_fit_genes=len(fit_gene_names),
        n_shard_genes=len(shard_gene_names),
        S=resolved_S,
        S_source=S_source,
        N_avg=default_S,
        device=device,
        gene_batch_size=gene_batch_size,
        shard=f"{rank}/{world_size}",
        output_path=output_path,
    )
    if missing_reference:
        console.print(
            f"[yellow]Skipped[/yellow] {len(missing_reference)} missing reference genes"
        )
    if missing_fit:
        console.print(f"[yellow]Skipped[/yellow] {len(missing_fit)} missing fit genes")
    if dry_run:
        return 0

    batch_names = [
        shard_gene_names[start : start + gene_batch_size]
        for start in range(0, len(shard_gene_names), gene_batch_size)
    ]
    fitted_p_grids: list[np.ndarray] = []
    fitted_weights: list[np.ndarray] = []
    batch_summaries: list[dict[str, Any]] = []
    mean_reference_count = float(np.mean(reference_counts))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("fitting priors", total=len(batch_names))
        for batch_index, names in enumerate(batch_names, start=1):
            batch_counts = _slice_gene_counts(
                matrix, [gene_to_idx[name] for name in names]
            )
            result = fit_gene_priors(
                ObservationBatch(
                    gene_names=list(names),
                    counts=batch_counts,
                    reference_counts=reference_counts,
                ),
                S=resolved_S,
                config=fit_config,
                device=device,
            )
            priors = result.priors.batched()
            fitted_p_grids.append(np.asarray(priors.p_grid, dtype=np.float64))
            fitted_weights.append(np.asarray(priors.weights, dtype=np.float64))
            batch_summaries.append(
                {
                    "batch_index": batch_index,
                    "n_genes": len(names),
                    "final_loss": float(result.final_loss),
                    "best_loss": float(result.best_loss),
                }
            )
            progress.update(
                task_id,
                advance=1,
                description=f"fitting priors ({batch_index}/{len(batch_names)})",
            )

    merged_priors = PriorGrid(
        gene_names=list(shard_gene_names),
        p_grid=np.concatenate(fitted_p_grids, axis=0),
        weights=np.concatenate(fitted_weights, axis=0),
        S=resolved_S,
    )
    checkpoint = ModelCheckpoint(
        gene_names=list(shard_gene_names),
        priors=merged_priors,
        scale=ScaleMetadata(S=resolved_S, mean_reference_count=mean_reference_count),
        fit_config=asdict(fit_config),
        metadata={
            "schema_version": 1,
            "source_h5ad_path": str(h5ad_path),
            "layer": layer,
            "reference_gene_names": list(reference_gene_names),
            "requested_fit_gene_names": list(fit_gene_names),
            "shard_gene_names": list(shard_gene_names),
            "missing_reference_genes": list(missing_reference),
            "missing_fit_genes": list(missing_fit),
            "n_cells": int(adata.n_obs),
            "gene_batch_size": int(gene_batch_size),
            "shard_rank": int(rank),
            "shard_world_size": int(world_size),
            "S_source": S_source,
            "default_S_from_reference_mean": default_S,
            "created_at": datetime.now(UTC).isoformat(),
            "batch_summaries": batch_summaries,
        },
    )
    save_checkpoint(checkpoint, output_path)
    _print_fit_summary(
        output_path=output_path,
        elapsed_sec=perf_counter() - start_time,
        checkpoint=checkpoint,
    )
    return 0


def _parse_shard(value: str) -> tuple[int, int]:
    parts = value.split("/")
    if len(parts) != 2:
        raise ValueError("shard must be formatted as rank/world, e.g. 0/4")
    rank = int(parts[0])
    world_size = int(parts[1])
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid shard specification: {value}")
    return rank, world_size


def _read_gene_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_gene_list(
    path: Path, gene_to_idx: dict[str, int]
) -> tuple[list[str], list[str]]:
    requested = _read_gene_list(path)
    ordered: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for name in requested:
        if name in seen:
            continue
        seen.add(name)
        if name in gene_to_idx:
            ordered.append(name)
        else:
            missing.append(name)
    return ordered, missing


def _resolve_fit_gene_list(
    fit_genes_path: Path | None,
    dataset_gene_names: list[str],
    gene_to_idx: dict[str, int],
) -> tuple[list[str], list[str]]:
    if fit_genes_path is None:
        return list(dataset_gene_names), []
    return _resolve_gene_list(fit_genes_path, gene_to_idx)


def _shard_gene_names(
    gene_names: list[str], *, rank: int, world_size: int
) -> list[str]:
    return [name for idx, name in enumerate(gene_names) if idx % world_size == rank]


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


def _print_fit_plan(**values: Any) -> None:
    table = Table(title="Fit Plan")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(key, str(value))
    console.print(table)


def _print_fit_summary(
    *, output_path: Path, elapsed_sec: float, checkpoint: ModelCheckpoint
) -> None:
    table = Table(title="Fit Summary")
    table.add_column("Genes", justify="right")
    table.add_column("S", justify="right")
    table.add_column("Mean ref count", justify="right")
    table.add_row(
        str(len(checkpoint.gene_names)),
        f"{checkpoint.scale.S:.4f}",
        f"{checkpoint.scale.mean_reference_count:.4f}",
    )
    console.print(table)
    console.print(f"[bold green]Saved[/bold green] {output_path}")
    console.print(f"[bold green]Elapsed[/bold green] {elapsed_sec:.2f}s")
