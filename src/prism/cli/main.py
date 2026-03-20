from __future__ import annotations

import typer
from rich.console import Console

from .extract_signal import extract_signal
from .fit_prior_engine import fit_prior_engine
from .merge_ckpt import merge_ckpt
from .serve import serve

app = typer.Typer(
    name="prism",
    help="PRISM command line tools",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

app.command("fit")(fit_prior_engine)
app.command("extract")(extract_signal)
app.command("merge-ckpt")(merge_ckpt)
app.command("serve")(serve)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
