from __future__ import annotations

from prism.cli.common import create_typer_app

from .inspect import inspect_checkpoint_command
from .merge import merge_checkpoints_command

checkpoint_app = create_typer_app(help="Inspect and manipulate checkpoints.")
checkpoint_app.command("inspect")(inspect_checkpoint_command)
checkpoint_app.command("merge")(merge_checkpoints_command)

__all__ = ["checkpoint_app"]
