from __future__ import annotations

import typer

from .intersect import intersect_genes_command
from .rank import rank_genes_command
from .subset import subset_genes_command

genes_app = typer.Typer(help="Build and manipulate gene lists.", no_args_is_help=True)
genes_app.command("intersect")(intersect_genes_command)
genes_app.command("subset")(subset_genes_command)
genes_app.command("rank")(rank_genes_command)

__all__ = ["genes_app"]
