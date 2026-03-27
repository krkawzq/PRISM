from __future__ import annotations

import typer

from prism.cli.extract_signal import extract_app
from prism.cli.fit_prior_engine import fit_app
from prism.cli.gene_lists import genes_app
from prism.cli.merge_ckpt import checkpoint_app
from prism.cli.serve import serve

app = typer.Typer(
    name="prism",
    help="PRISM command line interface.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

app.add_typer(fit_app, name="fit", help="Fit model artifacts.")
app.add_typer(extract_app, name="extract", help="Extract signals and derived outputs.")
app.add_typer(checkpoint_app, name="checkpoint", help="Inspect and merge checkpoints.")
app.add_typer(genes_app, name="genes", help="Build and manipulate gene lists.")
app.command("serve")(serve)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
