from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from prism.model import ModelCheckpoint, PriorGrid, load_checkpoint, save_checkpoint

checkpoint_app = typer.Typer(help="Checkpoint utilities.", no_args_is_help=True)
console = Console()


@checkpoint_app.command("merge")
def merge_checkpoints_command(
    checkpoint_paths: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input checkpoint paths."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Merged checkpoint path."
    ),
    allow_partial: bool = typer.Option(
        False,
        help="Allow merged checkpoints to miss genes from the declared requested set.",
    ),
) -> int:
    if len(checkpoint_paths) < 2:
        raise ValueError("merge requires at least two checkpoints")
    checkpoints = [
        load_checkpoint(path.expanduser().resolve()) for path in checkpoint_paths
    ]
    merged = _merge_checkpoints(
        checkpoints,
        [path.expanduser().resolve() for path in checkpoint_paths],
        allow_partial=allow_partial,
    )
    output_path = output_path.expanduser().resolve()
    save_checkpoint(merged, output_path)
    _print_merge_summary(output_path, checkpoints, merged)
    return 0


@checkpoint_app.command("inspect")
def inspect_checkpoint_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Checkpoint path."
    ),
) -> int:
    checkpoint = load_checkpoint(checkpoint_path.expanduser().resolve())
    metadata = checkpoint.metadata
    table = Table(title="Checkpoint")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("genes", str(len(checkpoint.gene_names)))
    table.add_row("S", f"{checkpoint.scale.S:.4f}")
    table.add_row(
        "mean_reference_count", f"{checkpoint.scale.mean_reference_count:.4f}"
    )
    table.add_row("S_source", str(metadata.get("S_source", "")))
    table.add_row(
        "default_S_from_reference_mean",
        str(metadata.get("default_S_from_reference_mean", "")),
    )
    table.add_row("source_h5ad_path", str(metadata.get("source_h5ad_path", "")))
    table.add_row("layer", str(metadata.get("layer", "")))
    table.add_row(
        "reference_genes",
        str(len(_safe_string_list(metadata.get("reference_gene_names")))),
    )
    table.add_row(
        "requested_fit_genes",
        str(len(_safe_string_list(metadata.get("requested_fit_gene_names")))),
    )
    table.add_row(
        "shard",
        f"{metadata.get('shard_rank', 0)}/{metadata.get('shard_world_size', 1)}",
    )
    console.print(table)
    return 0


def _merge_checkpoints(
    checkpoints: list[ModelCheckpoint],
    source_paths: list[Path],
    *,
    allow_partial: bool,
) -> ModelCheckpoint:
    first = checkpoints[0]
    _validate_shared_metadata(checkpoints, source_paths)

    rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for checkpoint in checkpoints:
        priors = checkpoint.priors.batched()
        for idx, gene_name in enumerate(priors.gene_names):
            if gene_name in rows:
                raise ValueError(
                    f"duplicate fitted gene across checkpoints: {gene_name}"
                )
            rows[gene_name] = (
                np.asarray(priors.p_grid[idx], dtype=np.float64),
                np.asarray(priors.weights[idx], dtype=np.float64),
            )

    requested_gene_names = _safe_string_list(
        first.metadata.get("requested_fit_gene_names")
    )
    ordered_gene_names = (
        requested_gene_names if requested_gene_names else list(rows.keys())
    )
    missing = [name for name in ordered_gene_names if name not in rows]
    if missing and not allow_partial:
        raise ValueError(
            f"merged checkpoints still miss {len(missing)} genes, e.g. {missing[:5]}"
        )
    merged_gene_names = [name for name in ordered_gene_names if name in rows]
    if not merged_gene_names:
        raise ValueError("merged checkpoint contains no genes")
    p_grid = np.stack([rows[name][0] for name in merged_gene_names], axis=0)
    weights = np.stack([rows[name][1] for name in merged_gene_names], axis=0)
    metadata = dict(first.metadata)
    metadata.update(
        {
            "shard_rank": 0,
            "shard_world_size": 1,
            "shard_gene_names": list(merged_gene_names),
            "source_checkpoints": [str(path) for path in source_paths],
            "merge_allow_partial": bool(allow_partial),
            "missing_after_merge": missing,
        }
    )
    return ModelCheckpoint(
        gene_names=list(merged_gene_names),
        priors=PriorGrid(
            gene_names=list(merged_gene_names),
            p_grid=p_grid,
            weights=weights,
            S=float(first.priors.S),
        ),
        scale=first.scale,
        fit_config=dict(first.fit_config),
        metadata=metadata,
    )


def _validate_shared_metadata(
    checkpoints: list[ModelCheckpoint], paths: list[Path]
) -> None:
    first = checkpoints[0]
    for path, checkpoint in zip(paths[1:], checkpoints[1:], strict=True):
        if checkpoint.priors.S != first.priors.S:
            raise ValueError(f"{path} has a different S")
        if checkpoint.fit_config != first.fit_config:
            raise ValueError(f"{path} has a different fit configuration")
        for key in (
            "source_h5ad_path",
            "layer",
            "reference_gene_names",
            "requested_fit_gene_names",
        ):
            if checkpoint.metadata.get(key) != first.metadata.get(key):
                raise ValueError(f"{path} has different metadata for {key!r}")


def _safe_string_list(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return []
    return list(value)


def _print_merge_summary(
    output_path: Path, checkpoints: list[ModelCheckpoint], merged: ModelCheckpoint
) -> None:
    table = Table(title="Merged Checkpoint")
    table.add_column("Input genes", justify="right")
    table.add_column("Merged genes", justify="right")
    table.add_column("S", justify="right")
    table.add_row(
        str(sum(len(checkpoint.gene_names) for checkpoint in checkpoints)),
        str(len(merged.gene_names)),
        f"{merged.scale.S:.4f}",
    )
    console.print(table)
    console.print(f"[bold green]Saved[/bold green] {output_path}")
