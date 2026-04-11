from __future__ import annotations

from prism.cli.common import create_typer_app

from .batch_grid import plot_batch_grid_command
from .distributions import plot_distributions_command
from .label_summary import plot_label_summary_command
from .priors import plot_priors_command

plot_app = create_typer_app(
    help="Render figures from checkpoints and extracted outputs."
)
plot_app.command(
    "priors",
    help="Compare prior curves across genes, labels, and checkpoints.",
)(plot_priors_command)
plot_app.command(
    "batch-grid",
    help="Render label priors on a batch x perturbation grid.",
)(plot_batch_grid_command)
plot_app.command(
    "distributions",
    help="Inspect extracted layer distributions across genes or groups.",
)(plot_distributions_command)
plot_app.command(
    "label-summary",
    help="Summarize pairwise similarity across label-specific priors.",
)(plot_label_summary_command)

__all__ = ["plot_app"]
