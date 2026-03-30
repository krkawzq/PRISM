from __future__ import annotations

import typer

from .kbulk import extract_kbulk_command
from .kbulk_mean import extract_kbulk_mean_command
from .signals import extract_signals_command

extract_app = typer.Typer(
    help="Extract signals and derived outputs.", no_args_is_help=True
)
extract_app.command("signals")(extract_signals_command)
extract_app.command("kbulk")(extract_kbulk_command)
extract_app.command("kbulk-mean")(extract_kbulk_mean_command)

__all__ = ["extract_app"]
