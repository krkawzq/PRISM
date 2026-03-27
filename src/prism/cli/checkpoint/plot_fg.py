from __future__ import annotations

from pathlib import Path

import typer

from prism.model import load_checkpoint

from .common import (
    console,
    plot_fg_figure,
    resolve_plot_curve_sets,
    resolve_x_axis,
    plt,
)


def plot_fg_command(
    checkpoint_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Checkpoint path."
    ),
    gene_names: list[str] = typer.Option(
        ..., "--gene", help="Repeatable gene name to plot."
    ),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output figure path."),
    labels: list[str] | None = typer.Option(
        None,
        "--label",
        help="Optional repeatable label priors to include. If omitted, include all label priors.",
    ),
    x_axis: str = typer.Option("mu", help="x axis: mu or p."),
    mass_quantile: float = typer.Option(
        0.995,
        min=0.5,
        max=1.0,
        help="Upper cumulative mass used to truncate the displayed axis.",
    ),
    include_global: bool = typer.Option(
        True,
        "--include-global/--no-include-global",
        help="Include the global prior when present.",
    ),
) -> int:
    checkpoint = load_checkpoint(checkpoint_path.expanduser().resolve())
    curve_sets = resolve_plot_curve_sets(
        checkpoint, gene_names=gene_names, labels=labels, include_global=include_global
    )
    fig = plot_fg_figure(
        curve_sets, x_axis=resolve_x_axis(x_axis), mass_quantile=float(mass_quantile)
    )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[bold green]Saved[/bold green] {output_path}")
    return 0


__all__ = ["plot_fg_command"]
