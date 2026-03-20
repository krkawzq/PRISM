#!/usr/bin/env python3
# pyright: reportMissingImports=false

from __future__ import annotations

from pathlib import Path
import sys

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.axes import Axes
from rich.console import Console
from rich.table import Table
from scipy import sparse

from prism.model import fit_pool_scale
from prism.model._typing import DTYPE_NP, EPS
from prism.model.estimator import _log_marginal, _posterior_grid_weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

app = typer.Typer(add_completion=False)
console = Console()


def _select_matrix(adata: ad.AnnData, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} 不存在")
    return adata.layers[layer]


def _compute_totals(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        totals = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        totals = np.asarray(matrix, dtype=DTYPE_NP).sum(axis=1)
    return np.asarray(totals, dtype=DTYPE_NP).reshape(-1)


def _build_output_path(h5ad_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        resolved = output_path.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    out_dir = Path("output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{h5ad_path.stem}_rS_estimation.png"


def _geometric_mean(values: np.ndarray) -> float:
    values_np = np.asarray(values, dtype=DTYPE_NP)
    clipped = np.clip(values_np, EPS, None)
    return float(np.exp(np.mean(np.log(clipped))))


def _summarize_eta(values: np.ndarray) -> dict[str, float]:
    values_np = np.asarray(values, dtype=DTYPE_NP)
    return {
        "median": float(np.median(values_np)),
        "mean": float(np.mean(values_np)),
        "geometric_mean": _geometric_mean(values_np),
    }


def _parameterized_summary(mu: float, sigma: float) -> dict[str, float]:
    return {
        "median": float(np.exp(mu)),
        "mean": float(np.exp(mu + 0.5 * sigma**2)),
        "mode": float(np.exp(mu - sigma**2)),
    }


def _posterior_eta_arrays(
    totals: np.ndarray,
    mu: float,
    sigma: float,
    n_quad: int,
) -> tuple[np.ndarray, np.ndarray]:
    totals_int = np.rint(np.asarray(totals, dtype=DTYPE_NP)).astype(np.int64)
    unique_n, inverse = np.unique(totals_int, return_inverse=True)
    log_eta_grid, posterior_weight = _posterior_grid_weights(
        unique_n.astype(DTYPE_NP),
        mu,
        sigma,
        n_quad,
    )
    eta_grid = np.exp(log_eta_grid)
    eta_map_unique = eta_grid[np.argmax(posterior_weight, axis=1)]
    eta_mean_unique = posterior_weight @ eta_grid
    return eta_map_unique[inverse], eta_mean_unique[inverse]


def _make_summary_table(
    ax: Axes,
    rows: list[str],
    cols: list[str],
    values: np.ndarray,
) -> None:
    ax.axis("off")

    comparable = values[1:, :] - values[0:1, :]
    abs_diff = np.abs(comparable)
    max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    min_idx = np.unravel_index(np.argmin(abs_diff), abs_diff.shape)
    max_cell = (int(max_idx[0]) + 1, int(max_idx[1]))
    min_cell = (int(min_idx[0]) + 1, int(min_idx[1]))

    table = ax.table(
        cellText=[[f"{value:,.3f}" for value in row] for row in values],
        rowLabels=rows,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold")
        elif col_idx == -1:
            cell.set_facecolor("#f8fafc")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff")

    for row_idx, col_idx, color in (
        (min_cell[0], min_cell[1], "#dcfce7"),
        (max_cell[0], max_cell[1], "#fee2e2"),
    ):
        table[(row_idx, col_idx)].set_facecolor(color)
        table[(row_idx, col_idx)].set_text_props(weight="bold")

    ax.set_title("rS extraction summary", fontsize=13, pad=12)
    ax.text(
        0.0,
        0.02,
        "Green/red mark the smallest/largest absolute deviation from the parameterized row\n"
        "within the same column among posterior-derived summaries.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )


def _print_console_summary(
    *,
    mu: float,
    sigma: float,
    r: float,
    s_hat: float,
    totals: np.ndarray,
    output_path: Path,
) -> None:
    table = Table(title="rS Verification")
    table.add_column("Cells", justify="right")
    table.add_column("Mean total", justify="right")
    table.add_column("Median total", justify="right")
    table.add_column("mu", justify="right")
    table.add_column("sigma", justify="right")
    table.add_column("r", justify="right")
    table.add_column("s_hat", justify="right")
    table.add_row(
        str(totals.size),
        f"{float(np.mean(totals)):.3f}",
        f"{float(np.median(totals)):.3f}",
        f"{mu:.4f}",
        f"{sigma:.4f}",
        f"{r:.4f}",
        f"{s_hat:.4f}",
    )
    console.print(table)
    console.print(f"[bold green]Saved[/bold green] {output_path}")


@app.command()
def main(
    h5ad_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad file."
    ),
    output_path: Path | None = typer.Option(
        None, "--output", "-o", help="Output figure path."
    ),
    layer: str | None = typer.Option(None, help="AnnData layer name to use."),
    r: float = typer.Option(
        0.05, min=1e-12, max=1.0, help="Capture efficiency used to convert rS to S."
    ),
    n_quad: int = typer.Option(128, min=2, help="Gauss-Hermite quadrature points."),
    max_points: int = typer.Option(
        5000, min=100, help="Maximum scatter points to draw in the eta plot."
    ),
) -> int:
    h5ad_path = h5ad_path.expanduser().resolve()
    resolved_output_path = _build_output_path(h5ad_path, output_path)

    console.print(f"[bold cyan]Reading[/bold cyan] {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    matrix = _select_matrix(adata, layer)
    totals = _compute_totals(matrix)

    console.print("[bold cyan]Fitting[/bold cyan] Poisson-LogNormal pool scale")
    estimate = fit_pool_scale(totals, n_quad=n_quad)
    mu = float(estimate.mu)
    sigma = float(estimate.sigma)
    s_hat = float(estimate.point_eta / r)

    eta_map, eta_mean = _posterior_eta_arrays(totals, mu, sigma, n_quad)
    param_summary = _parameterized_summary(mu, sigma)
    map_summary = _summarize_eta(eta_map)
    mean_summary = _summarize_eta(eta_mean)

    rows = ["Parameterized", "Posterior MAP", "Posterior Mean"]
    cols = ["median", "mean", "mode / geometric_mean"]
    table_values = np.asarray(
        [
            [param_summary["median"], param_summary["mean"], param_summary["mode"]],
            [map_summary["median"], map_summary["mean"], map_summary["geometric_mean"]],
            [
                mean_summary["median"],
                mean_summary["mean"],
                mean_summary["geometric_mean"],
            ],
        ],
        dtype=DTYPE_NP,
    )

    figure, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_hist, ax_eta, ax_scatter, ax_table = axes.ravel()

    max_count = int(np.max(np.rint(totals)))
    count_grid = np.arange(max_count + 1, dtype=DTYPE_NP)
    pmf = np.exp(_log_marginal(count_grid, mu, sigma, n_quad))

    bins = np.arange(-0.5, max_count + 1.5, 1.0)
    ax_hist.hist(
        totals, bins=bins, color="#94a3b8", alpha=0.75, label="Observed histogram"
    )
    ax_hist.plot(
        count_grid,
        pmf * totals.size,
        color="#1d4ed8",
        lw=2.0,
        label="Theoretical PMF x n",
    )
    ax_hist.set_title("Total UMI counts with Poisson-LogNormal fit")
    ax_hist.set_xlabel("N_c")
    ax_hist.set_ylabel("Cell count")
    ax_hist.text(
        0.98,
        0.95,
        f"mu = {mu:.4f}\nsigma = {sigma:.4f}\nrS = {estimate.point_eta:.4f}",
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#cbd5e1",
        },
    )
    ax_hist.legend(frameon=False)

    ax_eta.hist(
        eta_map,
        bins=80,
        density=True,
        color="#99f6e4",
        alpha=0.7,
        label="Posterior MAP histogram",
    )
    ax_eta.axvline(
        param_summary["median"], color="#1d4ed8", lw=2, ls="--", label="param median"
    )
    ax_eta.axvline(
        param_summary["mean"], color="#c2410c", lw=2, ls="-.", label="param mean"
    )
    ax_eta.axvline(
        param_summary["mode"], color="#0f766e", lw=2, ls=":", label="param mode"
    )
    ax_eta.set_title("Posterior MAP eta with analytical rS markers")
    ax_eta.set_xlabel("eta_hat_c")
    ax_eta.set_ylabel("Density")
    ax_eta.legend(frameon=False, fontsize=9)

    rng = np.random.default_rng(0)
    if totals.size > max_points:
        scatter_idx = rng.choice(totals.size, size=max_points, replace=False)
    else:
        scatter_idx = np.arange(totals.size)

    ax_scatter.scatter(
        totals[scatter_idx],
        eta_map[scatter_idx],
        s=10,
        alpha=0.35,
        color="#c2410c",
        edgecolors="none",
    )
    lim = max(
        float(np.quantile(totals, 0.995)),
        float(np.quantile(eta_map, 0.995)),
    )
    ax_scatter.plot([0.0, lim], [0.0, lim], color="#0f172a", lw=1.5, ls="--")
    ax_scatter.set_xlim(0.0, lim)
    ax_scatter.set_ylim(0.0, lim)
    ax_scatter.set_title("Observed counts vs posterior MAP eta")
    ax_scatter.set_xlabel("N_c")
    ax_scatter.set_ylabel("eta_hat_c")

    _make_summary_table(ax_table, rows, cols, table_values)

    figure.suptitle(f"rS estimation check: {h5ad_path.name}", fontsize=16)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(resolved_output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    _print_console_summary(
        mu=mu,
        sigma=sigma,
        r=r,
        s_hat=s_hat,
        totals=totals,
        output_path=resolved_output_path,
    )
    return 0


if __name__ == "__main__":
    app()
