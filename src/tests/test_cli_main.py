from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from prism.cli.fit.priors import fit_priors_command
from prism.cli.main import app


def test_main_help_lists_current_top_level_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("fit", "data", "extract", "checkpoint", "genes", "plot", "serve"):
        assert command in result.stdout


def test_main_supports_serve_command_help() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "--host" in result.stdout
    assert "--port" in result.stdout


def test_fit_priors_command_supports_dry_run(tmp_path) -> None:
    adata = ad.AnnData(
        X=np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64),
        obs=pd.DataFrame(index=["c1", "c2"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    input_path = tmp_path / "input.h5ad"
    output_path = tmp_path / "checkpoint.pkl"
    adata.write_h5ad(input_path)

    result = fit_priors_command(
        h5ad_path=input_path,
        output_path=output_path,
        dry_run=True,
    )

    assert result == 0
    assert not output_path.exists()
