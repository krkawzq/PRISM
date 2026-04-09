from __future__ import annotations

from prism.cli.common import create_typer_app

from .priors import fit_priors_command

fit_app = create_typer_app(help="Fit model priors.")
fit_app.command("priors")(fit_priors_command)

__all__ = ["fit_app"]
