from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from prism.model import load_checkpoint

from .common import console, safe_string_list


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
    table.add_row("global_priors", str(checkpoint.priors is not None))
    table.add_row("label_priors", str(len(checkpoint.label_priors)))
    table.add_row("S", "-" if checkpoint.scale is None else f"{checkpoint.scale.S:.4f}")
    table.add_row(
        "mean_reference_count",
        "-"
        if checkpoint.scale is None
        else f"{checkpoint.scale.mean_reference_count:.4f}",
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
        str(len(safe_string_list(metadata.get("reference_gene_names")))),
    )
    table.add_row(
        "requested_fit_genes",
        str(len(safe_string_list(metadata.get("requested_fit_gene_names")))),
    )
    table.add_row(
        "shard",
        f"{metadata.get('shard_rank', 0)}/{metadata.get('shard_world_size', 1)}",
    )
    console.print(table)
    return 0


__all__ = ["inspect_checkpoint_command"]
