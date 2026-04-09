from __future__ import annotations

from .checkpoint.app import checkpoint_app
from .common.runtime import create_typer_app
from .data.app import data_app
from .extract.app import extract_app
from .fit.app import fit_app
from .genes.app import genes_app
from .plot.app import plot_app
from .serve.app import serve_command

app = create_typer_app(name="prism", help="PRISM command line interface.")
app.add_typer(fit_app, name="fit", help="Fit model artifacts.")
app.add_typer(data_app, name="data", help="Prepare and transform datasets.")
app.add_typer(extract_app, name="extract", help="Extract signals and derived outputs.")
app.add_typer(
    checkpoint_app,
    name="checkpoint",
    help="Inspect and merge checkpoints.",
)
app.add_typer(genes_app, name="genes", help="Build and manipulate gene lists.")
app.add_typer(
    plot_app, name="plot", help="Render figures from checkpoints and outputs."
)
app.command("serve", help="Start the local PRISM server.")(serve_command)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
