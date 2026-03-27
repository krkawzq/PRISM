from __future__ import annotations

import typer

from .priors import fit_priors_command

fit_app = typer.Typer(help="Fit model artifacts.", no_args_is_help=True)
fit_app.command("priors")(fit_priors_command)

__all__ = ["fit_app"]
