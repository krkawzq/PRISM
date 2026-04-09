from __future__ import annotations

from prism.cli.common import create_typer_app

from .kbulk import extract_kbulk_command
from .kbulk_mean import extract_kbulk_mean_command
from .signals import extract_signals_command

extract_app = create_typer_app(help="Extract signals and derived outputs.")
extract_app.command("signals")(extract_signals_command)
extract_app.command("kbulk")(extract_kbulk_command)
extract_app.command("kbulk-mean")(extract_kbulk_mean_command)

__all__ = ["extract_app"]
