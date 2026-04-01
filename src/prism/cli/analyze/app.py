from __future__ import annotations

import typer

from .checkpoint_summary import checkpoint_summary_command
from .overlap_de import overlap_de_command

analyze_app = typer.Typer(
    help="Tabular analyses and statistics from checkpoints and extracted outputs. Use 'plot' for figure-based visualization.",
    no_args_is_help=True,
)
analyze_app.command("overlap-de")(overlap_de_command)
analyze_app.command("checkpoint-summary")(checkpoint_summary_command)

__all__ = ["analyze_app"]
