from __future__ import annotations

import typer

from prism.cli.analyze.app import analyze_app
from prism.cli.checkpoint.app import checkpoint_app
from prism.cli.data.app import data_app
from prism.cli.extract.app import extract_app
from prism.cli.fit.app import fit_app
from prism.cli.genes.app import genes_app
from prism.cli.plot.app import plot_app
from prism.cli.serve.app import serve

app = typer.Typer(
    name="prism",
    help="PRISM command line interface.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

app.add_typer(fit_app, name="fit", help="Fit model artifacts.")
app.add_typer(data_app, name="data", help="Prepare and transform datasets.")
app.add_typer(extract_app, name="extract", help="Extract signals and derived outputs.")
app.add_typer(analyze_app, name="analyze", help="Run analysis commands.")
app.add_typer(plot_app, name="plot", help="Render plots from checkpoints and outputs.")
app.add_typer(checkpoint_app, name="checkpoint", help="Inspect and merge checkpoints.")
app.add_typer(genes_app, name="genes", help="Build and manipulate gene lists.")
app.command("serve")(serve)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
