from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from prism.cli.common import resolve_bool, resolve_int
from prism.model import load_checkpoint


def inspect_checkpoint_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Checkpoint path."
    ),
    show_labels: bool = typer.Option(
        False, "--show-labels", help="Print available label prior names when present."
    ),
    label_limit: int = typer.Option(
        50, min=1, help="Maximum number of label names to print with --show-labels."
    ),
) -> int:
    checkpoint = load_checkpoint(checkpoint_path.expanduser().resolve())
    show_labels = resolve_bool(show_labels)
    label_limit = resolve_int(label_limit)
    table = Table(title="Checkpoint")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("genes", str(len(checkpoint.gene_names)))
    table.add_row("global_prior", str(checkpoint.has_global_prior))
    table.add_row("label_priors", str(len(checkpoint.label_priors)))
    table.add_row(
        "fit_distribution",
        str(
            checkpoint.metadata.get(
                "fit_distribution", checkpoint.fit_config.get("likelihood", "")
            )
        ),
    )
    table.add_row(
        "posterior_distribution",
        str(
            checkpoint.metadata.get(
                "posterior_distribution", checkpoint.fit_config.get("likelihood", "")
            )
        ),
    )
    table.add_row("support_domain", str(checkpoint.metadata.get("support_domain", "")))
    table.add_row(
        "scale",
        "-"
        if checkpoint.scale_metadata is None
        else f"{checkpoint.scale_metadata.scale:.4f}",
    )
    table.add_row(
        "mean_reference_count",
        "-"
        if checkpoint.scale_metadata is None
        else f"{checkpoint.scale_metadata.mean_reference_count:.4f}",
    )
    table.add_row(
        "source_h5ad_path", str(checkpoint.metadata.get("source_h5ad_path", ""))
    )
    table.add_row("layer", str(checkpoint.metadata.get("layer", "")))
    labels = list(checkpoint.available_labels)
    if labels:
        preview = ", ".join(labels[:label_limit])
        if len(labels) > label_limit:
            preview = f"{preview}, ..."
        table.add_row("label_preview", preview)
    from .common import console

    console.print(table)
    if show_labels and labels:
        label_table = Table(title="Label Priors")
        label_table.add_column("Index", justify="right")
        label_table.add_column("Label")
        for idx, label in enumerate(labels[:label_limit], start=1):
            label_table.add_row(str(idx), label)
        console.print(label_table)
    return 0


__all__ = ["inspect_checkpoint_command"]
