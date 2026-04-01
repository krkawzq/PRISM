from __future__ import annotations

import typer

from .downsample import downsample_command
from .subset_genes import subset_genes_command

data_app = typer.Typer(help="Prepare and transform datasets.", no_args_is_help=True)
data_app.command("subset-genes")(subset_genes_command)
data_app.command("downsample")(downsample_command)

__all__ = ["data_app"]
