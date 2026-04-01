from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from prism.model import load_checkpoint
from prism.cli.checkpoint_validation import resolve_cli_checkpoint_distribution

from .common import console, safe_string_list


def inspect_checkpoint_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Checkpoint path."
    ),
    show_labels: bool = typer.Option(
        False,
        "--show-labels",
        help="Print available label prior names when present.",
    ),
    label_limit: int = typer.Option(
        50,
        min=1,
        help="Maximum number of label names to print with --show-labels.",
    ),
) -> int:
    checkpoint = load_checkpoint(checkpoint_path.expanduser().resolve())
    metadata = resolve_cli_checkpoint_distribution(
        checkpoint,
        command_name="prism checkpoint inspect",
    )
    table = Table(title="Checkpoint")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("genes", str(len(checkpoint.gene_names)))
    table.add_row("global_priors", str(checkpoint.priors is not None))
    table.add_row("label_priors", str(len(checkpoint.label_priors)))
    table.add_row("fit_distribution", str(metadata["fit_distribution"]))
    table.add_row(
        "posterior_distribution",
        str(metadata["posterior_distribution"]),
    )
    table.add_row(
        "grid_domain",
        str(metadata["grid_domain"]),
    )
    table.add_row(
        "distribution_resolution",
        str(metadata.get("distribution_resolution", "")),
    )
    table.add_row(
        "legacy_compatibility",
        str(bool(metadata.get("legacy_compatibility", False))),
    )
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
    if checkpoint.label_priors:
        labels = sorted(checkpoint.label_priors)
        preview = ", ".join(labels[:label_limit])
        if len(labels) > label_limit:
            preview = f"{preview}, ..."
        table.add_row("label_preview", preview)
    table.add_row(
        "shard",
        f"{metadata.get('shard_rank', 0)}/{metadata.get('shard_world_size', 1)}",
    )
    console.print(table)
    if show_labels and checkpoint.label_priors:
        label_table = Table(title="Label Priors")
        label_table.add_column("Index", justify="right")
        label_table.add_column("Label")
        for idx, label in enumerate(
            sorted(checkpoint.label_priors)[:label_limit], start=1
        ):
            label_table.add_row(str(idx), label)
        console.print(label_table)
    return 0


__all__ = ["inspect_checkpoint_command"]
