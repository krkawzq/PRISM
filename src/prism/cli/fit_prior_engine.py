from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True, slots=True)
class _FitTask:
    scope_kind: str
    scope_name: str
    cell_indices: np.ndarray
    label_value: str | None = None


@fit_app.command("priors")
def fit_priors_command(
    h5ad_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad file."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output checkpoint path."
    ),
    reference_genes_path: Path | None = typer.Option(
        None,
        "--reference-genes",
        exists=True,
        dir_okay=False,
        help="Optional text file with reference gene names used to compute reference counts. Defaults to all genes in the dataset.",
    ),
    fit_genes_path: Path | None = typer.Option(
        None,
        "--fit-genes",
        exists=True,
        dir_okay=False,
        help="Optional text file with genes to fit. Defaults to all genes in the dataset.",
    ),
    layer: str | None = typer.Option(None, help="Input AnnData layer. Defaults to X."),
    label_key: str | None = typer.Option(
        None,
        help="Optional obs column used for class-specific fitting.",
    ),
    label_values: list[str] | None = typer.Option(
        None,
        "--label-value",
        help="Optional repeatable label values to fit when --label-key is set.",
    ),
    fit_mode: str = typer.Option(
        "global",
        help="Fit scope: global, by-label, or both.",
    ),
    n_samples: int | None = typer.Option(
        None,
        min=1,
        help="Optional random cell subsample size per fit scope.",
    ),
    sample_seed: int = typer.Option(0, min=0, help="Random seed for cell subsampling."),
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
    reference_genes_path = (
        None
        if reference_genes_path is None
        else reference_genes_path.expanduser().resolve()
    )
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
    fit_mode_resolved = _resolve_fit_mode(fit_mode)

    if reference_genes_path is None:
        reference_gene_names = list(gene_names)
        missing_reference: list[str] = []
    else:
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

    reference_positions = [gene_to_idx[name] for name in reference_gene_names]
    reference_counts = _compute_reference_counts(matrix, reference_positions)
    default_S = float(np.mean(reference_counts))
    resolved_S = default_S if S is None else float(S)
    S_source = "default:N_avg" if S is None else "user"
    label_groups = _resolve_label_groups(
        adata=adata,
        label_key=label_key,
        label_values=label_values,
        fit_mode=fit_mode_resolved,
    )
    _print_fit_plan(
        h5ad_path=h5ad_path,
        layer=layer,
        n_cells=int(adata.n_obs),
        n_dataset_genes=int(adata.n_vars),
        n_reference_genes=len(reference_gene_names),
        n_fit_genes=len(fit_gene_names),
        n_shard_genes=len(shard_gene_names),
        fit_mode=fit_mode_resolved,
        label_key=label_key,
        n_label_groups=len(label_groups),
        n_samples=n_samples,
        sample_seed=sample_seed,
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

    fit_tasks = _build_fit_tasks(
        fit_mode_resolved, label_groups, n_cells=int(adata.n_obs)
    )
    global_gene_names: list[str] | None = None
    global_p_grids: list[np.ndarray] = []
    global_weights: list[np.ndarray] = []
    global_scale: ScaleMetadata | None = None
    global_metadata: dict[str, Any] | None = None
    label_priors: dict[str, PriorGrid] = {}
    label_scales: dict[str, ScaleMetadata] = {}
    label_metadata: dict[str, dict[str, Any]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("fitting priors", total=len(fit_tasks))
        for task_index, task in enumerate(fit_tasks, start=1):
            sampled_indices = _sample_indices(
                task.cell_indices, n_samples=n_samples, seed=sample_seed + task_index
            )
            task_reference_counts = _compute_reference_counts(
                matrix, reference_positions, cell_indices=sampled_indices
            )
            valid_mask = task_reference_counts > 0
            if int(np.count_nonzero(valid_mask)) == 0:
                raise ValueError(
                    f"fit scope {task.scope_name!r} has no cells with positive reference counts"
                )
            sampled_indices = sampled_indices[valid_mask]
            task_reference_counts = task_reference_counts[valid_mask]
            batches = [
                shard_gene_names[start : start + gene_batch_size]
                for start in range(0, len(shard_gene_names), gene_batch_size)
            ]
            fitted_p_grids: list[np.ndarray] = []
            fitted_weights: list[np.ndarray] = []
            batch_summaries: list[dict[str, Any]] = []
            for batch_index, names in enumerate(batches, start=1):
                batch_counts = _slice_gene_counts(
                    matrix,
                    [gene_to_idx[name] for name in names],
                    cell_indices=sampled_indices,
                )
                result = fit_gene_priors(
                    ObservationBatch(
                        gene_names=list(names),
                        counts=batch_counts,
                        reference_counts=task_reference_counts,
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
            merged_priors = PriorGrid(
                gene_names=list(shard_gene_names),
                p_grid=np.concatenate(fitted_p_grids, axis=0),
                weights=np.concatenate(fitted_weights, axis=0),
                S=resolved_S,
            )
            task_scale = ScaleMetadata(
                S=resolved_S, mean_reference_count=float(np.mean(task_reference_counts))
            )
            task_meta = {
                "scope": task.scope_name,
                "label_key": label_key,
                "label_value": task.label_value,
                "n_cells_total": int(task.cell_indices.shape[0]),
                "n_cells_used": int(sampled_indices.shape[0]),
                "n_samples_requested": None if n_samples is None else int(n_samples),
                "batch_summaries": batch_summaries,
            }
            if task.scope_kind == "global":
                global_gene_names = list(shard_gene_names)
                global_p_grids = [np.asarray(merged_priors.p_grid, dtype=np.float64)]
                global_weights = [np.asarray(merged_priors.weights, dtype=np.float64)]
                global_scale = task_scale
                global_metadata = task_meta
            else:
                assert task.label_value is not None
                label_priors[task.label_value] = merged_priors
                label_scales[task.label_value] = task_scale
                label_metadata[task.label_value] = task_meta
            progress.update(
                task_id,
                advance=1,
                description=f"fitting priors ({task_index}/{len(fit_tasks)})",
            )

    merged_priors = None
    if global_gene_names is not None:
        merged_priors = PriorGrid(
            gene_names=list(global_gene_names),
            p_grid=np.concatenate(global_p_grids, axis=0),
            weights=np.concatenate(global_weights, axis=0),
            S=resolved_S,
        )
    checkpoint = ModelCheckpoint(
        gene_names=_checkpoint_gene_names(
            merged_priors, label_priors, shard_gene_names
        ),
        priors=merged_priors,
        scale=global_scale,
        fit_config=asdict(fit_config),
        metadata={
            "schema_version": 2,
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
            "fit_mode": fit_mode_resolved,
            "label_key": label_key,
            "label_values": [
                task.label_value for task in fit_tasks if task.label_value is not None
            ],
            "n_samples_requested": None if n_samples is None else int(n_samples),
            "sample_seed": int(sample_seed),
            "S_source": S_source,
            "default_S_from_reference_mean": default_S,
            "created_at": datetime.now(UTC).isoformat(),
            "global_fit": global_metadata,
            "label_fits": label_metadata,
        },
        label_priors=label_priors,
        label_scales=label_scales,
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


def _resolve_fit_mode(value: str) -> str:
    resolved = value.strip().lower()
    if resolved not in {"global", "by-label", "both"}:
        raise ValueError("fit_mode must be one of: global, by-label, both")
    return resolved


def _resolve_label_groups(
    *,
    adata: ad.AnnData,
    label_key: str | None,
    label_values: list[str] | None,
    fit_mode: str,
) -> list[tuple[str, np.ndarray]]:
    if fit_mode == "global":
        return []
    if label_key is None:
        raise ValueError("--label-key is required when fit_mode is by-label or both")
    if label_key not in adata.obs.columns:
        raise KeyError(f"obs column {label_key!r} does not exist")
    labels = np.asarray(adata.obs[label_key].astype(str)).reshape(-1)
    chosen_values = (
        list(dict.fromkeys(label_values))
        if label_values
        else sorted(np.unique(labels).tolist())
    )
    groups: list[tuple[str, np.ndarray]] = []
    for value in chosen_values:
        indices = np.flatnonzero(labels == value).astype(np.int64)
        if indices.size == 0:
            raise ValueError(f"label value {value!r} has no cells")
        groups.append((value, indices))
    return groups


def _build_fit_tasks(
    fit_mode: str, label_groups: list[tuple[str, np.ndarray]], *, n_cells: int
) -> list[_FitTask]:
    tasks: list[_FitTask] = []
    if fit_mode in {"global", "both"}:
        if label_groups:
            all_indices = np.unique(
                np.concatenate([indices for _, indices in label_groups])
            )
        else:
            all_indices = np.arange(n_cells, dtype=np.int64)
        tasks.append(
            _FitTask(scope_kind="global", scope_name="global", cell_indices=all_indices)
        )
    if fit_mode in {"by-label", "both"}:
        tasks.extend(
            _FitTask(
                scope_kind="label",
                scope_name=f"label:{label}",
                cell_indices=indices,
                label_value=label,
            )
            for label, indices in label_groups
        )
    return tasks


def _sample_indices(
    cell_indices: np.ndarray, *, n_samples: int | None, seed: int
) -> np.ndarray:
    if cell_indices.size == 0:
        return cell_indices
    if n_samples is None or cell_indices.size <= n_samples:
        return np.asarray(cell_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(cell_indices, size=n_samples, replace=False)
    return np.sort(np.asarray(chosen, dtype=np.int64))


def _checkpoint_gene_names(
    global_priors: PriorGrid | None,
    label_priors: dict[str, PriorGrid],
    fallback_gene_names: list[str],
) -> list[str]:
    if global_priors is not None:
        return list(global_priors.gene_names)
    if label_priors:
        first = next(iter(label_priors.values()))
        return list(first.gene_names)
    return list(fallback_gene_names)


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


def _slice_gene_counts(
    matrix, positions: list[int], *, cell_indices: np.ndarray | None = None
) -> np.ndarray:
    subset = matrix[:, positions]
    if cell_indices is not None:
        subset = subset[cell_indices, :]
    if sparse.issparse(subset):
        return np.asarray(subset.toarray(), dtype=np.float64)
    return np.asarray(subset, dtype=np.float64)


def _compute_reference_counts(
    matrix, positions: list[int], *, cell_indices: np.ndarray | None = None
) -> np.ndarray:
    subset = matrix[:, positions]
    if cell_indices is not None:
        subset = subset[cell_indices, :]
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
    s_value = "-" if checkpoint.scale is None else f"{checkpoint.scale.S:.4f}"
    mean_ref = (
        "-"
        if checkpoint.scale is None
        else f"{checkpoint.scale.mean_reference_count:.4f}"
    )
    table = Table(title="Fit Summary")
    table.add_column("Genes", justify="right")
    table.add_column("S", justify="right")
    table.add_column("Mean ref count", justify="right")
    table.add_row(
        str(len(checkpoint.gene_names)),
        s_value,
        mean_ref,
    )
    console.print(table)
    console.print(f"[bold green]Saved[/bold green] {output_path}")
    console.print(f"[bold green]Elapsed[/bold green] {elapsed_sec:.2f}s")
