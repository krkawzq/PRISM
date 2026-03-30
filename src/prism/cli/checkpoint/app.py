from __future__ import annotations

import typer

from .inspect import inspect_checkpoint_command
from .merge import merge_checkpoints_command
from .overlap_de import overlap_de_command
from .plot_fg import plot_fg_command

checkpoint_app = typer.Typer(
    help="Inspect and manipulate checkpoints.", no_args_is_help=True
)
checkpoint_app.command("merge")(merge_checkpoints_command)
checkpoint_app.command("inspect")(inspect_checkpoint_command)
checkpoint_app.command("plot-fg")(plot_fg_command)
checkpoint_app.command("overlap-de")(overlap_de_command)

__all__ = ["checkpoint_app"]
