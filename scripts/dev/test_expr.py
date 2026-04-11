#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import anndata as ad
import matplotlib
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from prism.io import (
        compute_reference_counts,
        read_gene_list,
        select_matrix,
        slice_gene_matrix,
    )
    from prism.model import (
        ObservationBatch,
        Posterior,
        PriorFitConfig,
        PriorGrid,
        effective_exposure,
        fit_gene_priors,
        make_distribution_grid,
        summarize_reference_scale,
    )
    from prism.model.numeric import log_binomial_likelihood_support
except ImportError as exc:
    raise ImportError(
        "PRISM is not installed in the active environment. Run `uv run` or "
        "`uv sync` from the repository root before executing scripts/dev/test_expr.py."
    ) from exc

console = Console()
install_rich_traceback(show_locals=False)

EXPR_MODES: tuple[str, ...] = ("raw", "normalize_total", "log1p_normalize_total")
MODE_LABELS = {
    "raw": "raw",
    "normalize_total": "normalize_total",
    "log1p_normalize_total": "log1p_normalize_total",
}
MODE_COLORS = {
    "raw": ("#1d4ed8", "#60a5fa"),
    "normalize_total": ("#c2410c", "#fb923c"),
    "log1p_normalize_total": ("#047857", "#6ee7b7"),
}


@dataclass(frozen=True, slots=True)
class ScopeSpec:
    name: str
    cell_indices: np.ndarray | None


@dataclass(frozen=True, slots=True)
class FitArtifact:
    mode: str
    prior: PriorGrid
    posterior_average: np.ndarray
    nll: np.ndarray
    prior_post_jsd: np.ndarray
    gaussian_jsd: np.ndarray
    gaussian_compare_jsd: np.ndarray
    compare_jsd: dict[str, np.ndarray]
    fit_seconds: float
    n_iterations: int
    final_objective: float
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare prior fits across raw counts, total-normalized counts, and "
            "log1p(total-normalized counts) using the current PRISM model API."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_h5ad", type=Path, help="Input AnnData file.")
    parser.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix. Writes <prefix>.svg and <prefix>.csv.",
    )
    parser.add_argument(
        "--gene",
        action="append",
        default=[],
        help="Gene to inspect. Can be repeated.",
    )
    parser.add_argument(
        "--gene-list",
        type=Path,
        default=None,
        help="Optional text file used as the source of candidate genes.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=6,
        help="Number of genes to use when --gene is not provided.",
    )
    parser.add_argument(
        "--reference-genes",
        type=Path,
        default=None,
        help="Optional text file defining reference genes.",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default=None,
        help="obs column used to select label-specific subsets.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Label value to inspect. Can be repeated.",
    )
    parser.add_argument(
        "--include-global",
        action="store_true",
        help="Include a global fit alongside any requested labels.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="AnnData layer to read. Defaults to X.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=4000,
        help="Maximum number of cells per scope after subsampling.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Base random seed used for subsampling.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Optional fixed model scale. Defaults to a mode-specific suggestion.",
    )
    parser.add_argument(
        "--normalize-target",
        type=float,
        default=None,
        help="Target total count used by normalized modes. Defaults to the median reference count.",
    )
    parser.add_argument(
        "--n-support-points",
        type=int,
        default=256,
        help="Number of support points per gene.",
    )
    parser.add_argument(
        "--max-em-iterations",
        type=int,
        default=100,
        help="Maximum EM iterations.",
    )
    parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=1e-6,
        help="EM convergence tolerance.",
    )
    parser.add_argument(
        "--cell-chunk-size",
        type=int,
        default=512,
        help="Number of cells per likelihood chunk.",
    )
    parser.add_argument(
        "--support-max-from",
        choices=("observed_max", "quantile"),
        default="observed_max",
        help="Rule used to determine the upper support bound.",
    )
    parser.add_argument(
        "--support-spacing",
        choices=("linear", "sqrt"),
        default="linear",
        help="Spacing used across the support grid.",
    )
    parser.add_argument(
        "--adaptive-support",
        action="store_true",
        help="Enable adaptive support refinement.",
    )
    parser.add_argument(
        "--adaptive-support-scale",
        type=float,
        default=1.0,
        help="Expansion factor applied to the adaptive support range.",
    )
    parser.add_argument(
        "--adaptive-support-quantile",
        type=float,
        default=0.99,
        help="Posterior quantile used for adaptive refinement.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Torch dtype used for fitting and inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch.compile for model kernels.",
    )
    return parser.parse_args()


def resolve_output_prefix(input_path: Path, output_prefix: Path | None) -> Path:
    if output_prefix is not None:
        return output_prefix.expanduser().resolve()
    return (Path("outputs") / "dev" / f"{input_path.stem}.expr-compare").resolve()


def resolve_selected_genes(
    *,
    adata: ad.AnnData,
    genes: list[str],
    gene_list_path: Path | None,
    top_n: int,
) -> list[str]:
    if genes:
        return list(dict.fromkeys(str(gene) for gene in genes))
    if gene_list_path is not None:
        return read_gene_list(gene_list_path.expanduser().resolve())[:top_n]
    return [str(name) for name in adata.var_names.tolist()[:top_n]]


def resolve_reference_positions(
    *,
    dataset_gene_names: list[str],
    reference_gene_list: list[str] | None,
) -> list[int]:
    gene_to_idx = {name: idx for idx, name in enumerate(dataset_gene_names)}
    if reference_gene_list is None:
        return list(range(len(dataset_gene_names)))
    selected = [
        gene_to_idx[name] for name in reference_gene_list if name in gene_to_idx
    ]
    if not selected:
        raise ValueError("reference gene list has no overlap with the dataset")
    return list(dict.fromkeys(selected))


def resolve_target_positions(
    *,
    dataset_gene_names: list[str],
    selected_genes: list[str],
) -> tuple[list[str], list[int]]:
    gene_to_idx = {name: idx for idx, name in enumerate(dataset_gene_names)}
    ordered_genes: list[str] = []
    ordered_positions: list[int] = []
    missing: list[str] = []
    for gene in selected_genes:
        idx = gene_to_idx.get(gene)
        if idx is None:
            missing.append(gene)
            continue
        ordered_genes.append(gene)
        ordered_positions.append(idx)
    if missing:
        raise ValueError(f"selected genes missing from dataset: {missing[:5]}")
    if not ordered_positions:
        raise ValueError("no selected genes overlap with the dataset")
    return ordered_genes, ordered_positions


def resolve_scopes(adata: ad.AnnData, args: argparse.Namespace) -> list[ScopeSpec]:
    scopes: list[ScopeSpec] = []
    if args.include_global or not args.label:
        scopes.append(ScopeSpec(name="global", cell_indices=None))
    if args.label_key is None:
        if args.label:
            raise ValueError("--label-key is required when using --label")
        return scopes
    if args.label_key not in adata.obs.columns:
        raise KeyError(f"obs column {args.label_key!r} does not exist")
    labels = np.asarray(adata.obs[args.label_key].astype(str)).reshape(-1)
    for label in list(dict.fromkeys(args.label)):
        indices = np.flatnonzero(labels == label).astype(np.int64)
        if indices.size == 0:
            raise ValueError(f"label {label!r} has no cells in {args.label_key!r}")
        scopes.append(ScopeSpec(name=f"label:{label}", cell_indices=indices))
    return scopes


def normalize_weights(values: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(values, dtype=np.float64), 1e-12, None)
    return weights / weights.sum(axis=-1, keepdims=True)


def jsd(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_p = normalize_weights(left)
    right_p = normalize_weights(right)
    midpoint = 0.5 * (left_p + right_p)
    return 0.5 * (
        np.sum(left_p * np.log(left_p / midpoint), axis=-1)
        + np.sum(right_p * np.log(right_p / midpoint), axis=-1)
    )


def build_gaussian_prior(prior: PriorGrid) -> PriorGrid:
    support = np.asarray(prior.support, dtype=np.float64)
    scaled_support = np.asarray(prior.scaled_support, dtype=np.float64)
    weights = normalize_weights(prior.prior_probabilities)
    mean = np.sum(weights * scaled_support, axis=-1, keepdims=True)
    variance = np.sum(
        weights * np.square(scaled_support - mean), axis=-1, keepdims=True
    )
    std = np.sqrt(np.clip(variance, 1e-12, None))
    gaussian_weights = normalize_weights(
        np.exp(-0.5 * np.square((scaled_support - mean) / std))
    )
    return PriorGrid(
        gene_names=list(prior.gene_names),
        distribution=make_distribution_grid(
            prior.distribution_name,
            support=support,
            probabilities=gaussian_weights,
        ),
        scale=float(prior.scale),
    )


def compare_priors_on_scaled_support(
    left: PriorGrid,
    right: PriorGrid,
) -> np.ndarray:
    left_scaled = np.asarray(left.scaled_support, dtype=np.float64)
    right_scaled = np.asarray(right.scaled_support, dtype=np.float64)
    left_weights = normalize_weights(left.prior_probabilities)
    right_weights = normalize_weights(right.prior_probabilities)
    scores = np.empty(left_scaled.shape[0], dtype=np.float64)
    for idx in range(left_scaled.shape[0]):
        merged_support = np.unique(
            np.concatenate([left_scaled[idx], right_scaled[idx]], axis=0)
        )
        left_interp = normalize_weights(
            np.interp(
                merged_support, left_scaled[idx], left_weights[idx], left=0.0, right=0.0
            )
        )
        right_interp = normalize_weights(
            np.interp(
                merged_support,
                right_scaled[idx],
                right_weights[idx],
                left=0.0,
                right=0.0,
            )
        )
        scores[idx] = float(jsd(left_interp[None, :], right_interp[None, :])[0])
    return scores


def subsample_rows(n_rows: int, max_cells: int | None, seed: int) -> np.ndarray:
    if max_cells is None or n_rows <= max_cells:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_cells, replace=False)).astype(np.int64)


def normalize_total(
    counts: np.ndarray,
    reference_counts: np.ndarray,
    *,
    target: float,
) -> np.ndarray:
    scale = float(target) / np.maximum(
        np.asarray(reference_counts, dtype=np.float64), 1e-12
    )
    return np.asarray(counts, dtype=np.float64) * scale[:, None]


def build_mode_batch(
    *,
    matrix,
    target_positions: list[int],
    reference_positions: list[int],
    target_gene_names: list[str],
    mode: str,
    normalize_target: float | None,
    max_cells: int | None,
    seed: int,
    cell_indices: np.ndarray | None,
    scale_override: float | None,
) -> tuple[ObservationBatch, float, dict[str, Any]]:
    raw_counts = slice_gene_matrix(
        matrix,
        target_positions,
        cell_indices=cell_indices,
        dtype=np.float64,
    )
    raw_reference_counts = compute_reference_counts(
        matrix,
        reference_positions,
        cell_indices=cell_indices,
        dtype=np.float64,
    )
    sampled_rows = subsample_rows(raw_counts.shape[0], max_cells, seed)
    raw_counts = raw_counts[sampled_rows]
    raw_reference_counts = raw_reference_counts[sampled_rows]
    valid_mask = raw_reference_counts > 0
    if not np.any(valid_mask):
        raise ValueError("no cells with positive reference counts after sampling")
    raw_counts = raw_counts[valid_mask]
    raw_reference_counts = raw_reference_counts[valid_mask]
    resolved_target = float(
        np.median(raw_reference_counts)
        if normalize_target is None
        else normalize_target
    )
    if not np.isfinite(resolved_target) or resolved_target <= 0:
        raise ValueError(f"normalize target must be positive, got {resolved_target}")

    if mode == "raw":
        counts = raw_counts
        reference_counts = raw_reference_counts
        default_scale = summarize_reference_scale(reference_counts).suggested_scale
    elif mode == "normalize_total":
        counts = normalize_total(
            raw_counts, raw_reference_counts, target=resolved_target
        )
        reference_counts = np.full(
            raw_counts.shape[0], resolved_target, dtype=np.float64
        )
        default_scale = resolved_target
    elif mode == "log1p_normalize_total":
        counts = np.log1p(
            normalize_total(raw_counts, raw_reference_counts, target=resolved_target)
        )
        reference_counts = np.full(
            raw_counts.shape[0], resolved_target, dtype=np.float64
        )
        default_scale = resolved_target
    else:
        raise ValueError(f"unsupported mode: {mode}")

    resolved_scale = float(default_scale if scale_override is None else scale_override)
    batch = ObservationBatch(
        gene_names=list(target_gene_names),
        counts=np.asarray(counts, dtype=np.float64),
        reference_counts=np.asarray(reference_counts, dtype=np.float64),
    )
    metadata = {
        "mode": mode,
        "n_cells_used": int(batch.n_cells),
        "n_genes": int(batch.n_genes),
        "mean_reference_count": float(np.mean(batch.reference_counts)),
        "scale": float(resolved_scale),
        "normalize_target": None if mode == "raw" else float(resolved_target),
    }
    return batch, resolved_scale, metadata


def compute_gene_nll(
    batch: ObservationBatch,
    prior: PriorGrid,
    *,
    device: str,
    torch_dtype: str,
) -> np.ndarray:
    dtype = torch.float64 if torch_dtype == "float64" else torch.float32
    device_obj = torch.device(device)
    counts_t = torch.as_tensor(batch.counts.T, dtype=dtype, device=device_obj)
    support_t = torch.as_tensor(prior.support, dtype=dtype, device=device_obj)
    probabilities_t = torch.as_tensor(
        prior.prior_probabilities, dtype=dtype, device=device_obj
    )
    exposure = effective_exposure(batch.reference_counts, prior.scale)
    exposure_t = (
        torch.as_tensor(exposure, dtype=dtype, device=device_obj)
        .unsqueeze(0)
        .expand(batch.n_genes, -1)
    )
    log_likelihood = log_binomial_likelihood_support(counts_t, exposure_t, support_t)
    log_prior = torch.log(probabilities_t.clamp_min(1e-12)).unsqueeze(-2)
    log_marginal = torch.logsumexp(log_likelihood + log_prior, dim=-1)
    return np.asarray(
        (-log_marginal.mean(dim=-1)).detach().cpu().numpy(), dtype=np.float64
    )


def fit_mode(
    *,
    batch: ObservationBatch,
    scale: float,
    mode: str,
    args: argparse.Namespace,
) -> FitArtifact:
    config = PriorFitConfig(
        n_support_points=args.n_support_points,
        max_em_iterations=args.max_em_iterations,
        convergence_tolerance=args.convergence_tolerance,
        cell_chunk_size=args.cell_chunk_size,
        support_max_from=args.support_max_from,
        support_spacing=args.support_spacing,
        use_adaptive_support=bool(args.adaptive_support),
        adaptive_support_scale=args.adaptive_support_scale,
        adaptive_support_quantile=args.adaptive_support_quantile,
        likelihood="binomial",
    )
    start = perf_counter()
    fit_result = fit_gene_priors(
        batch,
        scale=scale,
        config=config,
        device=args.device,
        torch_dtype=args.torch_dtype,
        compile_model=bool(args.compile_model),
    )
    fit_seconds = perf_counter() - start
    posterior = Posterior(
        list(batch.gene_names),
        fit_result.prior,
        device=args.device,
        torch_dtype=args.torch_dtype,
        compile_model=bool(args.compile_model),
    ).summarize_batch(batch, include_posterior=True)
    if posterior.posterior_probabilities is None:
        raise RuntimeError("posterior inference unexpectedly returned no posterior")
    posterior_average = normalize_weights(
        np.asarray(posterior.posterior_probabilities, dtype=np.float64).mean(axis=0)
    )
    gaussian_prior = build_gaussian_prior(fit_result.prior)
    return FitArtifact(
        mode=mode,
        prior=fit_result.prior,
        posterior_average=posterior_average,
        nll=compute_gene_nll(
            batch,
            fit_result.prior,
            device=args.device,
            torch_dtype=args.torch_dtype,
        ),
        prior_post_jsd=jsd(fit_result.prior.prior_probabilities, posterior_average),
        gaussian_jsd=jsd(
            fit_result.prior.prior_probabilities,
            gaussian_prior.prior_probabilities,
        ),
        gaussian_compare_jsd=compare_priors_on_scaled_support(
            fit_result.prior, gaussian_prior
        ),
        compare_jsd={},
        fit_seconds=float(fit_seconds),
        n_iterations=int(len(fit_result.objective_history)),
        final_objective=float(fit_result.final_objective),
        metadata={
            "mode": mode,
            "scale": float(scale),
            "fit_seconds": float(fit_seconds),
            "n_iterations": int(len(fit_result.objective_history)),
        },
    )


def build_figure(
    *,
    target_gene_names: list[str],
    scope_results: dict[str, dict[str, FitArtifact]],
    output_svg_path: Path,
) -> None:
    scope_names = list(scope_results)
    n_rows = len(target_gene_names)
    n_cols = len(scope_names) * len(EXPR_MODES)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )
    for row_idx, gene_name in enumerate(target_gene_names):
        for scope_idx, scope_name in enumerate(scope_names):
            for mode_idx, mode in enumerate(EXPR_MODES):
                col_idx = scope_idx * len(EXPR_MODES) + mode_idx
                ax = axes[row_idx][col_idx]
                artifact = scope_results[scope_name][mode]
                x = np.asarray(artifact.prior.scaled_support[row_idx], dtype=np.float64)
                prior = np.asarray(
                    artifact.prior.prior_probabilities[row_idx], dtype=np.float64
                )
                posterior = np.asarray(
                    artifact.posterior_average[row_idx], dtype=np.float64
                )
                prior_color, post_color = MODE_COLORS[mode]
                annotation_lines = [
                    f"objective={artifact.final_objective:.4f}",
                    f"nll={artifact.nll[row_idx]:.4f}",
                    f"prior-post jsd={artifact.prior_post_jsd[row_idx]:.4f}",
                    f"gauss jsd={artifact.gaussian_jsd[row_idx]:.4f}",
                    f"gauss cmp={artifact.gaussian_compare_jsd[row_idx]:.4f}",
                ]
                for other_mode in EXPR_MODES:
                    if other_mode == mode:
                        continue
                    annotation_lines.append(
                        f"vs {MODE_LABELS[other_mode]}={artifact.compare_jsd[other_mode][row_idx]:.4f}"
                    )
                ax.plot(x, prior, lw=2.0, color=prior_color, label=MODE_LABELS[mode])
                ax.plot(
                    x,
                    posterior,
                    lw=1.8,
                    ls="--",
                    color=post_color,
                    label=f"{MODE_LABELS[mode]} posterior",
                )
                ax.set_xlim(0.0, max(float(np.max(x)), 1e-12))
                ax.set_xlabel("scaled support")
                ax.set_ylabel(
                    f"{gene_name}\nprobability" if col_idx == 0 else "probability"
                )
                if row_idx == 0:
                    ax.set_title(f"{scope_name}\n{MODE_LABELS[mode]}")
                ax.grid(alpha=0.2)
                ax.text(
                    0.98,
                    0.98,
                    "\n".join(annotation_lines),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "fc": "white",
                        "alpha": 0.85,
                        "ec": "none",
                    },
                )
                if row_idx == 0 and col_idx == 0:
                    ax.legend(frameon=False)
    fig.tight_layout()
    output_svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(
    *,
    output_csv_path: Path,
    scope_results: dict[str, dict[str, FitArtifact]],
    target_gene_names: list[str],
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "mode",
        "gene",
        "fit_seconds",
        "n_iterations",
        "final_objective",
        "scale",
        "nll",
        "prior_post_jsd",
        "gaussian_jsd",
        "gaussian_compare_jsd",
    ] + [f"compare_jsd_vs_{mode}" for mode in EXPR_MODES]
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scope_name, mode_results in scope_results.items():
            for mode, artifact in mode_results.items():
                for gene_idx, gene_name in enumerate(target_gene_names):
                    row = {
                        "scope": scope_name,
                        "mode": mode,
                        "gene": gene_name,
                        "fit_seconds": f"{artifact.fit_seconds:.6f}",
                        "n_iterations": artifact.n_iterations,
                        "final_objective": f"{artifact.final_objective:.6f}",
                        "scale": f"{float(artifact.metadata['scale']):.6f}",
                        "nll": f"{artifact.nll[gene_idx]:.6f}",
                        "prior_post_jsd": f"{artifact.prior_post_jsd[gene_idx]:.6f}",
                        "gaussian_jsd": f"{artifact.gaussian_jsd[gene_idx]:.6f}",
                        "gaussian_compare_jsd": f"{artifact.gaussian_compare_jsd[gene_idx]:.6f}",
                    }
                    for other_mode in EXPR_MODES:
                        if other_mode == mode:
                            row[f"compare_jsd_vs_{other_mode}"] = ""
                        else:
                            row[f"compare_jsd_vs_{other_mode}"] = (
                                f"{artifact.compare_jsd[other_mode][gene_idx]:.6f}"
                            )
                    writer.writerow(row)


def print_intro(
    *,
    input_path: Path,
    output_prefix: Path,
    target_gene_names: list[str],
    scopes: list[ScopeSpec],
    args: argparse.Namespace,
) -> None:
    table = Table(show_header=False, box=None)
    table.add_row("Input", str(input_path))
    table.add_row("Output prefix", str(output_prefix))
    table.add_row("Genes", ", ".join(target_gene_names))
    table.add_row("Scopes", ", ".join(scope.name for scope in scopes))
    table.add_row("Layer", args.layer or "X")
    table.add_row("Max cells", "all" if args.max_cells is None else str(args.max_cells))
    table.add_row("Device", args.device)
    table.add_row("Compile", str(bool(args.compile_model)))
    console.print(Panel(table, title="Expr Compare", border_style="cyan"))


def print_summary(
    *,
    output_prefix: Path,
    scope_results: dict[str, dict[str, FitArtifact]],
) -> None:
    summary = Table(title="Expression Comparison Summary")
    summary.add_column("Scope")
    summary.add_column("Mode")
    summary.add_column("Fit", justify="right")
    summary.add_column("Steps", justify="right")
    summary.add_column("Objective", justify="right")
    for scope_name, mode_results in scope_results.items():
        for mode in EXPR_MODES:
            artifact = mode_results[mode]
            summary.add_row(
                scope_name,
                mode,
                f"{artifact.fit_seconds:.2f}s",
                str(artifact.n_iterations),
                f"{artifact.final_objective:.4f}",
            )
    console.print(summary)
    console.print(f"[bold green]Saved[/bold green] {output_prefix.with_suffix('.svg')}")
    console.print(f"[bold green]Saved[/bold green] {output_prefix.with_suffix('.csv')}")


def main() -> None:
    args = parse_args()
    input_path = args.input_h5ad.expanduser().resolve()
    output_prefix = resolve_output_prefix(input_path, args.output_prefix)
    reference_gene_list = (
        None
        if args.reference_genes is None
        else read_gene_list(args.reference_genes.expanduser().resolve())
    )

    with console.status("Loading AnnData..."):
        adata = ad.read_h5ad(input_path)
        dataset_gene_names = [str(name) for name in adata.var_names.tolist()]
        target_gene_names, target_positions = resolve_target_positions(
            dataset_gene_names=dataset_gene_names,
            selected_genes=resolve_selected_genes(
                adata=adata,
                genes=args.gene,
                gene_list_path=args.gene_list,
                top_n=args.top_n,
            ),
        )
        reference_positions = resolve_reference_positions(
            dataset_gene_names=dataset_gene_names,
            reference_gene_list=reference_gene_list,
        )
        scopes = resolve_scopes(adata, args)
        matrix = select_matrix(adata, args.layer)

    print_intro(
        input_path=input_path,
        output_prefix=output_prefix,
        target_gene_names=target_gene_names,
        scopes=scopes,
        args=args,
    )

    scope_results: dict[str, dict[str, FitArtifact]] = {}
    for scope_index, scope in enumerate(scopes, start=1):
        mode_results: dict[str, FitArtifact] = {}
        for mode in EXPR_MODES:
            with console.status(f"Fitting {mode} for {scope.name}..."):
                batch, scale, metadata = build_mode_batch(
                    matrix=matrix,
                    target_positions=target_positions,
                    reference_positions=reference_positions,
                    target_gene_names=target_gene_names,
                    mode=mode,
                    normalize_target=args.normalize_target,
                    max_cells=args.max_cells,
                    seed=args.random_seed + scope_index,
                    cell_indices=scope.cell_indices,
                    scale_override=args.scale,
                )
                artifact = fit_mode(
                    batch=batch,
                    scale=scale,
                    mode=mode,
                    args=args,
                )
                artifact.metadata.update(metadata)
                mode_results[mode] = artifact
        for mode in EXPR_MODES:
            for other_mode in EXPR_MODES:
                if other_mode == mode:
                    continue
                mode_results[mode].compare_jsd[other_mode] = (
                    compare_priors_on_scaled_support(
                        mode_results[mode].prior,
                        mode_results[other_mode].prior,
                    )
                )
        scope_results[scope.name] = mode_results

    with console.status("Writing outputs..."):
        build_figure(
            target_gene_names=target_gene_names,
            scope_results=scope_results,
            output_svg_path=output_prefix.with_suffix(".svg"),
        )
        write_summary_csv(
            output_csv_path=output_prefix.with_suffix(".csv"),
            scope_results=scope_results,
            target_gene_names=target_gene_names,
        )

    print_summary(output_prefix=output_prefix, scope_results=scope_results)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(Panel(str(exc), title="test_expr failed", border_style="red"))
        raise SystemExit(1) from exc
