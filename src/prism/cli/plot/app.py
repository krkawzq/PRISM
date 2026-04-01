from __future__ import annotations

import typer

from .batch_grid import plot_batch_grid_command
from .distributions import plot_distributions_command
from .label_summary import plot_label_summary_command
from .overlap import plot_overlap_command
from .priors import plot_priors_command

plot_app = typer.Typer(
    help="Render figures from checkpoints and extracted outputs. Use 'analyze' for tabular statistics.",
    no_args_is_help=True,
)
plot_app.command("priors")(plot_priors_command)
plot_app.command("batch-grid")(plot_batch_grid_command)
plot_app.command("overlap")(plot_overlap_command)
plot_app.command("distributions")(plot_distributions_command)
plot_app.command("label-summary")(plot_label_summary_command)

__all__ = ["plot_app"]
