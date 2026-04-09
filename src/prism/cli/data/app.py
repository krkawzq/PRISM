from __future__ import annotations

from prism.cli.common import create_typer_app

from .downsample import downsample_command
from .subset_genes import subset_genes_command

data_app = create_typer_app(help="Prepare and transform datasets.")
data_app.command("subset-genes")(subset_genes_command)
data_app.command("downsample")(downsample_command)

__all__ = ["data_app"]
