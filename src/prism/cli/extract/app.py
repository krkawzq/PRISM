from __future__ import annotations

import typer

from .signals import extract_signals_command

extract_app = typer.Typer(
    help="Extract signals and derived outputs.", no_args_is_help=True
)
extract_app.command("signals")(extract_signals_command)

__all__ = ["extract_app"]
