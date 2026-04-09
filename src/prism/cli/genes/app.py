from __future__ import annotations

from prism.cli.common import create_typer_app

from .filter import filter_genes_command
from .intersect import intersect_genes_command
from .merge import merge_genes_command
from .rank import rank_genes_command
from .subset import subset_genes_command

genes_app = create_typer_app(help="Build and manipulate gene lists.")
genes_app.command("intersect")(intersect_genes_command)
genes_app.command("subset")(subset_genes_command)
genes_app.command("rank")(rank_genes_command)
genes_app.command("merge")(merge_genes_command)
genes_app.command("filter")(filter_genes_command)

__all__ = ["genes_app"]
