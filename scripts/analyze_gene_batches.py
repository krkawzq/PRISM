#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import anndata as ad
import matplotlib
import numpy as np
import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.model import (
    GeneBatch,
    Posterior,
    PriorEngine,
    PriorEngineSetting,
    PriorEngineTrainingConfig,
    fit_pool_scale_report,
)
from prism.model._typing import DTYPE_NP
from prism.server.services.datasets import (
    resolve_gene_query,
    select_matrix,
    slice_gene_counts,
)

app = typer.Typer(add_completion=False)
console = Console()


@dataclass(frozen=True, slots=True)
class GeneFitResult:
    label: str
    batch_value: str | None
    n_cells: int
    s_hat: float
    counts: np.ndarray
    totals: np.ndarray
    x_eff: np.ndarray
    signal: np.ndarray
    posterior_entropy: np.ndarray
    prior_entropy: np.ndarray
    mutual_information: np.ndarray
    support: np.ndarray
    prior_weights: np.ndarray
    posterior_samples: np.ndarray
    posterior_signal: np.ndarray
    posterior_entropy_samples: np.ndarray
    pool_mu: float
    pool_sigma: float
    pool_point_eta: float
    prior_grid_max: float
    final_loss: float
    best_loss: float


@dataclass(frozen=True, slots=True)
class GlobalReferences:
    support: np.ndarray
    fitted_density: np.ndarray
    pseudo_raw_density: np.ndarray
    pseudo_aligned_density: np.ndarray
    common_cutoff: float
    batch_scales: dict[str, float]


@dataclass(frozen=True, slots=True)
class PoolFitSnapshot:
    label: str
    batch_value: str | None
    n_cells: int
    mu: float
    sigma: float
    point_eta: float
    loglik: float
    loglik_history: tuple[float, ...]
    totals_sample: np.ndarray
    eta_posterior_mean_sample: np.ndarray
    eta_prior_grid: np.ndarray
    eta_prior_density: np.ndarray


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    if x_np.size != y_np.size or x_np.size < 2:
        return float("nan")
    if np.allclose(x_np, x_np[0]) or np.allclose(y_np, y_np[0]):
        return float("nan")
    return float(np.corrcoef(x_np, y_np)[0, 1])


def _read_ranked_gene_list(path: Path) -> list[str]:
    genes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    genes = [gene for gene in genes if gene]
    if not genes:
        raise ValueError(f"ranked gene list is empty: {path}")
    return genes


def _resolve_device(device: str) -> str:
    requested = device.strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA unavailable; falling back to cpu.[/yellow]")
        return "cpu"
    return requested


def _infer_total_key(adata: ad.AnnData, total_key: str | None) -> str | None:
    if total_key:
        if total_key not in adata.obs.columns:
            raise KeyError(f"total key {total_key!r} not found in obs")
        return total_key
    for candidate in ("total_umi", "UMI_count", "ncounts", "total_counts"):
        if candidate in adata.obs.columns:
            return candidate
    return None


def _compute_totals(matrix) -> np.ndarray:
    try:
        totals = np.asarray(matrix.sum(axis=1)).reshape(-1)
    except AttributeError:
        totals = np.asarray(matrix[:, :], dtype=DTYPE_NP).sum(axis=1)
    return np.asarray(totals, dtype=DTYPE_NP)


def _sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)


def _sample_for_plot(values: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    values_np = np.asarray(values, dtype=np.float64)
    if values_np.size <= max_points:
        return values_np
    rng = np.random.default_rng(seed)
    idx = rng.choice(values_np.size, size=max_points, replace=False)
    return values_np[np.sort(idx)]


def _sample_indices(n_items: int, max_points: int, seed: int) -> np.ndarray:
    if n_items <= max_points:
        return np.arange(n_items, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_items, size=max_points, replace=False)
    return np.sort(idx.astype(np.int64, copy=False))


def _pool_snapshot(
    label: str, batch_value: str | None, totals: np.ndarray, pool_report
) -> PoolFitSnapshot:
    sample_idx = _sample_indices(len(totals), min(12000, len(totals)), seed=17)
    totals_sample = np.asarray(totals, dtype=np.float64)[sample_idx]
    posterior_sample = np.asarray(pool_report.eta_posterior_mean, dtype=np.float64)[
        sample_idx
    ]
    return PoolFitSnapshot(
        label=label,
        batch_value=batch_value,
        n_cells=int(len(totals)),
        mu=float(pool_report.mu),
        sigma=float(pool_report.sigma),
        point_eta=float(pool_report.point_eta),
        loglik=float(pool_report.loglik),
        loglik_history=tuple(float(x) for x in pool_report.loglik_history),
        totals_sample=totals_sample,
        eta_posterior_mean_sample=posterior_sample,
        eta_prior_grid=np.asarray(pool_report.eta_prior_grid, dtype=np.float64),
        eta_prior_density=np.asarray(pool_report.eta_prior_density, dtype=np.float64),
    )


def _pick_indices_by_quantile(values: np.ndarray, n: int) -> np.ndarray:
    if values.size <= n:
        return np.arange(values.size, dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, n)
    order = np.argsort(values)
    picked = [int(order[int(round(q * (values.size - 1)))]) for q in quantiles]
    return np.asarray(sorted(set(picked)), dtype=np.int64)


def _top_batch_values(labels: np.ndarray, top_n: int) -> list[str]:
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [str(unique[idx]) for idx in order[:top_n].tolist()]


def _select_batch_values(
    labels: np.ndarray, batch_limit: int
) -> tuple[list[str], list[tuple[str, int]]]:
    top_values = _top_batch_values(labels, batch_limit)
    batch_sizes = [(value, int(np.sum(labels == value))) for value in top_values]
    return top_values, batch_sizes


def _randomize_selected_batches(
    *,
    n_cells: int,
    batch_sizes: list[tuple[str, int]],
    seed: int,
) -> tuple[np.ndarray, list[str], list[tuple[str, int]]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_cells)
    randomized = np.full(n_cells, "__unselected__", dtype=object)
    random_values: list[str] = []
    random_sizes: list[tuple[str, int]] = []
    cursor = 0
    for idx, (_label, size) in enumerate(batch_sizes, start=1):
        random_label = f"rand_batch_{idx:03d}"
        chosen = perm[cursor : cursor + size]
        randomized[chosen] = random_label
        random_values.append(random_label)
        random_sizes.append((random_label, int(size)))
        cursor += size
    return randomized.astype(str), random_values, random_sizes


def _mu_axis_upper(global_result: GeneFitResult, mass: float = 0.95) -> float:
    support = np.asarray(global_result.support, dtype=np.float64)
    weights = np.asarray(global_result.prior_weights, dtype=np.float64)
    if support.size == 0:
        return 1.0
    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if total <= 0:
        return float(np.max(support))
    cdf = np.cumsum(weights / total)
    idx = int(np.searchsorted(cdf, mass, side="left"))
    idx = min(max(idx, 0), support.size - 1)
    upper = float(support[idx])
    if support.size == 1:
        return max(upper, 1.0)
    step = float(np.median(np.diff(support))) if support.size > 1 else 0.0
    full_max = float(np.max(support))
    padded = upper + max(step * 2.0, upper * 0.03, 1e-8)
    return min(max(padded, support[1] if support.size > 1 else upper), full_max)


def _curve_quantile(support: np.ndarray, density: np.ndarray, q: float) -> float:
    support_np = np.asarray(support, dtype=np.float64)
    density_np = np.clip(np.asarray(density, dtype=np.float64), 0.0, None)
    if support_np.size == 0:
        return 1.0
    if support_np.size == 1:
        return float(support_np[0])
    dx = np.diff(support_np)
    mid_mass = 0.5 * (density_np[:-1] + density_np[1:]) * dx
    total = float(np.sum(mid_mass))
    if total <= 0:
        return float(support_np[-1])
    cdf = np.concatenate([[0.0], np.cumsum(mid_mass) / total])
    return float(np.interp(q, cdf, support_np))


def _prior_mode(support: np.ndarray, weights: np.ndarray) -> float:
    support_np = np.asarray(support, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    if support_np.size == 0:
        return float("nan")
    return float(support_np[int(np.argmax(weights_np))])


def _prior_mean(support: np.ndarray, weights: np.ndarray) -> float:
    support_np = np.asarray(support, dtype=np.float64)
    density = _weights_to_density(support_np, weights)
    return float(np.trapezoid(support_np * density, support_np))


def _weights_to_density(support: np.ndarray, weights: np.ndarray) -> np.ndarray:
    support_np = np.asarray(support, dtype=np.float64)
    weights_np = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if support_np.size == 0:
        return weights_np.copy()
    if support_np.size == 1:
        return np.ones_like(weights_np, dtype=np.float64)
    step = float(np.median(np.diff(support_np)))
    step = max(step, 1e-12)
    density = weights_np / step
    area = float(np.trapezoid(density, support_np))
    if area <= 0:
        density = np.ones_like(support_np, dtype=np.float64)
        area = float(np.trapezoid(density, support_np))
    return density / max(area, 1e-12)


def _resample_density_to_cutoff(
    support: np.ndarray,
    density: np.ndarray,
    cutoff: float,
    target_grid: np.ndarray,
) -> np.ndarray:
    support_np = np.asarray(support, dtype=np.float64)
    density_np = np.clip(np.asarray(density, dtype=np.float64), 0.0, None)
    target_support = np.asarray(target_grid, dtype=np.float64) * cutoff
    sampled = np.interp(
        target_support, support_np, density_np, left=float(density_np[0]), right=0.0
    )
    sampled = np.clip(sampled, 0.0, None)
    sampled_unit = sampled * cutoff
    area = float(np.trapezoid(sampled_unit, target_grid))
    if area <= 0:
        sampled_unit = np.ones_like(target_grid)
        area = float(np.trapezoid(sampled_unit, target_grid))
    return sampled_unit / max(area, 1e-12)


def _scaled_unit_density(
    curve_unit: np.ndarray, scale: float, grid: np.ndarray
) -> np.ndarray:
    source = np.clip(np.asarray(curve_unit, dtype=np.float64), 0.0, None)
    grid_np = np.asarray(grid, dtype=np.float64)
    warped_x = np.clip(grid_np / max(scale, 1e-8), 0.0, 1.0)
    warped = np.interp(warped_x, grid_np, source, left=float(source[0]), right=0.0)
    warped = warped / max(scale, 1e-8)
    warped = np.clip(warped, 0.0, None)
    area = float(np.trapezoid(warped, grid_np))
    if area <= 0:
        return source / max(float(np.trapezoid(source, grid_np)), 1e-12)
    return warped / area


def _overlap_area(lhs: np.ndarray, rhs: np.ndarray, grid: np.ndarray) -> float:
    return float(np.trapezoid(np.minimum(lhs, rhs), np.asarray(grid, dtype=np.float64)))


def _aggregate_weighted(curves: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    curve_stack = np.stack(curves, axis=0)
    weight_np = np.asarray(weights, dtype=np.float64)
    weight_np = weight_np / max(float(np.sum(weight_np)), 1e-12)
    aggregated = np.tensordot(weight_np, curve_stack, axes=(0, 0))
    return np.asarray(aggregated, dtype=np.float64)


def _fit_batch_scales(
    curves_unit: list[np.ndarray],
    weights: np.ndarray,
    grid: np.ndarray,
    n_iter: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    weight_np = np.asarray(weights, dtype=np.float64)
    weight_np = weight_np / max(float(np.sum(weight_np)), 1e-12)
    scales = np.ones(len(curves_unit), dtype=np.float64)
    reference = _aggregate_weighted(curves_unit, weight_np)
    search = np.exp(np.linspace(np.log(0.7), np.log(1.4), 49))
    fine = np.exp(np.linspace(np.log(0.9), np.log(1.1), 41))
    for _ in range(n_iter):
        aligned_curves: list[np.ndarray] = []
        for idx, curve in enumerate(curves_unit):
            best_scale = scales[idx]
            best_score = -1.0
            for candidate in search * scales[idx]:
                warped = _scaled_unit_density(curve, float(candidate), grid)
                score = _overlap_area(warped, reference, grid)
                if score > best_score:
                    best_score = score
                    best_scale = float(candidate)
            local_search = np.clip(best_scale * fine, 0.5, 2.0)
            for candidate in local_search:
                warped = _scaled_unit_density(curve, float(candidate), grid)
                score = _overlap_area(warped, reference, grid)
                if score > best_score:
                    best_score = score
                    best_scale = float(candidate)
            scales[idx] = best_scale
            aligned_curves.append(_scaled_unit_density(curve, best_scale, grid))
        log_scales = np.log(np.clip(scales, 1e-8, None))
        centered = log_scales - float(np.sum(weight_np * log_scales))
        scales = np.exp(centered)
        aligned_curves = [
            _scaled_unit_density(curve, float(scale), grid)
            for curve, scale in zip(curves_unit, scales, strict=True)
        ]
        reference = _aggregate_weighted(aligned_curves, weight_np)
        reference = reference / max(float(np.trapezoid(reference, grid)), 1e-12)
    return scales, reference


def _build_global_references(
    fitted_global: GeneFitResult,
    batch_results: list[GeneFitResult],
    grid_size: int = 512,
) -> GlobalReferences:
    if not batch_results:
        raise ValueError("batch_results cannot be empty")
    q99_values = [
        _curve_quantile(
            result.support,
            _weights_to_density(result.support, result.prior_weights),
            0.99,
        )
        for result in batch_results
    ]
    common_cutoff = max(q99_values)
    common_cutoff = max(common_cutoff, 1e-6)
    grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    fitted_density = _resample_density_to_cutoff(
        fitted_global.support,
        _weights_to_density(fitted_global.support, fitted_global.prior_weights),
        common_cutoff,
        grid,
    )
    weights = np.asarray([result.n_cells for result in batch_results], dtype=np.float64)
    batch_curves = [
        _resample_density_to_cutoff(
            result.support,
            _weights_to_density(result.support, result.prior_weights),
            common_cutoff,
            grid,
        )
        for result in batch_results
    ]
    pseudo_raw = _aggregate_weighted(batch_curves, weights)
    pseudo_raw = pseudo_raw / max(float(np.trapezoid(pseudo_raw, grid)), 1e-12)
    scales, pseudo_aligned = _fit_batch_scales(batch_curves, weights, grid)
    support_mu = grid * common_cutoff
    density_scale = 1.0 / common_cutoff
    return GlobalReferences(
        support=support_mu,
        fitted_density=fitted_density * density_scale,
        pseudo_raw_density=pseudo_raw * density_scale,
        pseudo_aligned_density=pseudo_aligned * density_scale,
        common_cutoff=common_cutoff,
        batch_scales={
            result.label: float(scale)
            for result, scale in zip(batch_results, scales, strict=True)
        },
    )


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def _resolve_selected_genes(
    *,
    gene: str | None,
    ranked_genes: Path | None,
    gene_rank: int | None,
    top_ranked: int | None,
    gene_names: np.ndarray,
    gene_names_lower: tuple[str, ...],
    gene_to_idx: dict[str, int],
    gene_lower_to_idx: dict[str, int],
) -> tuple[list[str], list[int], str | None]:
    if gene is None and ranked_genes is None:
        raise typer.BadParameter("provide GENE or --ranked-genes")
    if gene is not None and ranked_genes is not None:
        raise typer.BadParameter("provide either GENE or --ranked-genes, not both")
    if (gene_rank is not None or top_ranked is not None) and ranked_genes is None:
        raise typer.BadParameter("--gene-rank/--top-ranked require --ranked-genes")
    if gene_rank is not None and top_ranked is not None:
        raise typer.BadParameter("use either --gene-rank or --top-ranked")

    note: str | None = None
    if gene is not None:
        selected_tokens = [gene]
    else:
        ranked = _read_ranked_gene_list(ranked_genes)
        if top_ranked is not None:
            if top_ranked > len(ranked):
                raise ValueError(
                    f"top_ranked={top_ranked} exceeds ranked list length {len(ranked)}"
                )
            selected_tokens = ranked[:top_ranked]
            note = f"ranked source={ranked_genes} top={top_ranked}"
        else:
            rank = 1 if gene_rank is None else gene_rank
            if rank > len(ranked):
                raise ValueError(
                    f"gene_rank={rank} exceeds ranked list length {len(ranked)}"
                )
            selected_tokens = [ranked[rank - 1]]
            note = f"ranked source={ranked_genes} rank={rank}"

    resolved_names: list[str] = []
    resolved_indices: list[int] = []
    seen: set[int] = set()
    for token in selected_tokens:
        idx = resolve_gene_query(
            token,
            gene_names,
            gene_names_lower,
            gene_to_idx,
            gene_lower_to_idx,
        )
        if idx in seen:
            continue
        seen.add(idx)
        resolved_indices.append(int(idx))
        resolved_names.append(str(gene_names[idx]))
    return resolved_names, resolved_indices, note


def _slice_gene_matrix(matrix, gene_indices: list[int]) -> np.ndarray:
    columns = [slice_gene_counts(matrix, idx) for idx in gene_indices]
    if not columns:
        raise ValueError("no genes selected")
    return np.column_stack(columns).astype(DTYPE_NP, copy=False)


def _iter_gene_chunks(n_genes: int, chunk_size: int) -> list[slice]:
    if chunk_size < 1:
        raise ValueError(f"fit_gene_batch_size must be >= 1, got {chunk_size}")
    return [
        slice(start, min(start + chunk_size, n_genes))
        for start in range(0, n_genes, chunk_size)
    ]


def _compute_shared_ratio_quantiles(
    counts_matrix: np.ndarray,
    totals: np.ndarray,
    quantile: float,
) -> np.ndarray:
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"shared_grid_quantile must be in (0, 1), got {quantile}")
    counts_np = np.asarray(counts_matrix, dtype=np.float64)
    totals_np = np.maximum(np.asarray(totals, dtype=np.float64).reshape(-1), 1.0)
    ratios = np.empty(counts_np.shape[1], dtype=np.float64)
    for gene_idx in range(counts_np.shape[1]):
        ratios[gene_idx] = float(
            np.quantile(counts_np[:, gene_idx] / totals_np, quantile)
        )
    return np.clip(ratios, 1e-12, 1.0 - 1e-12)


def _fit_group_multi(
    *,
    gene_names: list[str],
    counts_matrix: np.ndarray,
    totals: np.ndarray,
    label: str,
    batch_value: str | None,
    r: float,
    device: str,
    grid_size: int,
    sigma_bins: float,
    align_loss_weight: float,
    torch_dtype: str,
    lr: float,
    n_iter: int,
    lr_min_ratio: float,
    grad_clip: float | None,
    init_temperature: float,
    cell_chunk_size: int,
    optimizer: str,
    scheduler: str,
    posterior_chunk_size: int,
    representative_cells: int,
    fit_gene_batch_size: int,
    shared_support_ratio_max: np.ndarray | None,
    progress: Progress | None = None,
    group_task_id: TaskID | None = None,
) -> tuple[dict[str, GeneFitResult], PoolFitSnapshot]:
    def update(
        message: str, *, advance: float = 0.0, completed: float | None = None
    ) -> None:
        if progress is None or group_task_id is None:
            return
        kwargs: dict[str, object] = {"description": message}
        if completed is not None:
            kwargs["completed"] = completed
        elif advance:
            kwargs["advance"] = advance
        progress.update(group_task_id, **kwargs)

    n_genes = len(gene_names)
    counts_matrix = np.asarray(counts_matrix, dtype=DTYPE_NP)
    totals = np.asarray(totals, dtype=DTYPE_NP).reshape(-1)
    if counts_matrix.shape != (totals.shape[0], n_genes):
        raise ValueError(
            f"counts_matrix shape mismatch: expected {(totals.shape[0], n_genes)}, got {counts_matrix.shape}"
        )

    gene_slices = _iter_gene_chunks(n_genes, fit_gene_batch_size)
    pool_steps = 120
    prior_steps = max(n_iter, 1) * len(gene_slices)
    posterior_steps = max(math.ceil(totals.shape[0] / posterior_chunk_size), 1) * len(
        gene_slices
    )
    total_steps = (
        pool_steps + prior_steps + posterior_steps + n_genes + len(gene_slices) + 1
    )
    update(f"{label} | setup genes={n_genes}", completed=0)

    last_pool_step = 0

    def on_pool_progress(
        step: int, total: int, ll: float, mu: float, sigma: float, _done: bool
    ) -> None:
        nonlocal last_pool_step, pool_steps
        pool_steps = max(total, 1)
        delta = max(step - last_pool_step, 0)
        last_pool_step = step
        update(
            f"{label} | pool fit {step}/{pool_steps} ll={ll:.2f} mu={mu:.4f} sigma={sigma:.4f}",
            advance=float(delta),
        )

    pool_report = fit_pool_scale_report(
        totals,
        use_posterior_mu=False,
        progress_callback=on_pool_progress,
    )
    s_hat = float(pool_report.point_eta / r)
    pool_snapshot = _pool_snapshot(label, batch_value, totals, pool_report)
    update(f"{label} | pool fit done s_hat={s_hat:.4f}", advance=1.0)

    n_cells = totals.shape[0]
    signal = np.empty((n_cells, n_genes), dtype=np.float64)
    posterior_entropy = np.empty((n_cells, n_genes), dtype=np.float64)
    prior_entropy = np.empty((n_cells, n_genes), dtype=np.float64)
    mutual_information = np.empty((n_cells, n_genes), dtype=np.float64)
    results: dict[str, GeneFitResult] = {}
    posterior_done = 0
    finalized_done = 0
    for chunk_idx, gene_slice in enumerate(gene_slices, start=1):
        chunk_gene_names = gene_names[gene_slice]
        chunk_counts = counts_matrix[:, gene_slice]
        last_prior_step = 0

        def on_prior_progress(
            step: int,
            total: int,
            total_loss: float,
            nll_value: float,
            align_value: float,
            _best_updated: bool,
        ) -> None:
            nonlocal last_prior_step
            local_total = max(total, 1)
            delta = max(step - last_prior_step, 0)
            last_prior_step = step
            update(
                f"{label} | chunk {chunk_idx}/{len(gene_slices)} | prior {step}/{local_total} loss={total_loss:.4f} nll={nll_value:.4f} jsd={align_value:.4f}",
                advance=float(delta),
            )

        batch = GeneBatch(
            gene_names=chunk_gene_names, counts=chunk_counts, totals=totals
        )
        engine = PriorEngine(
            chunk_gene_names,
            setting=PriorEngineSetting(
                grid_size=grid_size,
                sigma_bins=sigma_bins,
                align_loss_weight=align_loss_weight,
                torch_dtype=torch_dtype,
            ),
            device=device,
        )
        prior_report = engine.fit_report(
            batch,
            s_hat=s_hat,
            training_cfg=PriorEngineTrainingConfig(
                lr=lr,
                n_iter=n_iter,
                lr_min_ratio=lr_min_ratio,
                grad_clip=grad_clip,
                init_temperature=init_temperature,
                cell_chunk_size=cell_chunk_size,
                optimizer=optimizer,
                scheduler=scheduler,
            ),
            progress_callback=on_prior_progress,
            grid_max_override=(
                None
                if shared_support_ratio_max is None
                else np.asarray(
                    shared_support_ratio_max[gene_slice] * s_hat, dtype=DTYPE_NP
                )
            ),
        )
        update(
            f"{label} | prior fit chunk {chunk_idx}/{len(gene_slices)} done best={float(prior_report.best_loss):.4f}",
            advance=1.0,
        )
        priors = engine.get_priors(chunk_gene_names)
        if priors is None:
            raise RuntimeError("failed to fetch fitted priors")
        posterior = Posterior(chunk_gene_names, priors, device=device)
        chunk_posterior_steps = max(math.ceil(n_cells / posterior_chunk_size), 1)
        for start in range(0, n_cells, posterior_chunk_size):
            end = min(start + posterior_chunk_size, n_cells)
            extracted = posterior.extract(
                GeneBatch(
                    gene_names=chunk_gene_names,
                    counts=chunk_counts[start:end],
                    totals=totals[start:end],
                ),
                s_hat=s_hat,
                channels={
                    "signal",
                    "posterior_entropy",
                    "prior_entropy",
                    "mutual_information",
                },
            )
            signal[start:end, gene_slice] = extracted["signal"].T
            posterior_entropy[start:end, gene_slice] = extracted["posterior_entropy"].T
            prior_entropy[start:end, gene_slice] = extracted["prior_entropy"].T
            mutual_information[start:end, gene_slice] = extracted[
                "mutual_information"
            ].T
            posterior_done += 1
            done_in_chunk = posterior_done - (chunk_idx - 1) * chunk_posterior_steps
            update(
                f"{label} | chunk {chunk_idx}/{len(gene_slices)} | posterior {done_in_chunk}/{chunk_posterior_steps}",
                advance=1.0,
            )
        for local_idx, gene_name in enumerate(chunk_gene_names):
            gene_idx = gene_slice.start + local_idx
            gene_signal = signal[:, gene_idx]
            representative_idx = _pick_indices_by_quantile(
                gene_signal, representative_cells
            )
            rep = posterior.extract(
                GeneBatch(
                    gene_names=[gene_name],
                    counts=chunk_counts[representative_idx, local_idx][:, None],
                    totals=totals[representative_idx],
                ),
                s_hat=s_hat,
                channels={"posterior", "signal", "posterior_entropy"},
            )
            counts = chunk_counts[:, local_idx]
            x_eff = counts / np.maximum(totals, 1.0) * s_hat
            results[gene_name] = GeneFitResult(
                label=label,
                batch_value=batch_value,
                n_cells=n_cells,
                s_hat=s_hat,
                counts=np.asarray(counts, dtype=np.float64),
                totals=np.asarray(totals, dtype=np.float64),
                x_eff=np.asarray(x_eff, dtype=np.float64),
                signal=gene_signal,
                posterior_entropy=posterior_entropy[:, gene_idx],
                prior_entropy=prior_entropy[:, gene_idx],
                mutual_information=mutual_information[:, gene_idx],
                support=np.asarray(prior_report.support[local_idx], dtype=np.float64),
                prior_weights=np.asarray(
                    prior_report.prior_weights[local_idx], dtype=np.float64
                ),
                posterior_samples=np.asarray(rep["posterior"][0], dtype=np.float64),
                posterior_signal=np.asarray(rep["signal"][0], dtype=np.float64),
                posterior_entropy_samples=np.asarray(
                    rep["posterior_entropy"][0], dtype=np.float64
                ),
                pool_mu=float(pool_report.mu),
                pool_sigma=float(pool_report.sigma),
                pool_point_eta=float(pool_report.point_eta),
                prior_grid_max=float(prior_report.grid_max[local_idx]),
                final_loss=float(prior_report.final_loss),
                best_loss=float(prior_report.best_loss),
            )
            finalized_done += 1
            update(f"{label} | finalize {finalized_done}/{n_genes}", advance=1.0)
    update(f"{label} | done genes={n_genes}", completed=total_steps)
    return results, pool_snapshot


def _plot_batch_sizes(batch_sizes: list[tuple[str, int]], out_path: Path) -> None:
    labels = [label for label, _ in batch_sizes]
    values = [count for _, count in batch_sizes]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(np.arange(len(labels)), values, color="#2563eb")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Cells")
    ax.set_title("Top batches by cell count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    with out_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch", "n_cells"])
        writer.writerows(batch_sizes)


def _plot_pool_fit_summary(
    global_pool: PoolFitSnapshot,
    batch_pools: list[PoolFitSnapshot],
    out_path: Path,
) -> None:
    series = [global_pool] + batch_pools
    labels = [item.label for item in series]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    x = np.arange(len(series))
    metrics = [
        ("point_eta", [item.point_eta for item in series]),
        ("mu", [item.mu for item in series]),
        ("sigma", [item.sigma for item in series]),
        ("loglik / cell", [item.loglik / max(item.n_cells, 1) for item in series]),
    ]
    for ax, (title, values) in zip(axes, metrics, strict=True):
        ax.plot(x, values, marker="o", color="#2563eb", lw=1.8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    with out_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "n_cells",
                "mu",
                "sigma",
                "point_eta",
                "loglik",
                "loglik_per_cell",
            ]
        )
        for item in series:
            writer.writerow(
                [
                    item.label,
                    item.n_cells,
                    item.mu,
                    item.sigma,
                    item.point_eta,
                    item.loglik,
                    item.loglik / max(item.n_cells, 1),
                ]
            )


def _plot_pool_fit_gallery(
    global_pool: PoolFitSnapshot,
    batch_pools: list[PoolFitSnapshot],
    out_path: Path,
) -> None:
    series = [global_pool] + batch_pools
    n = len(series)
    fig, axes = plt.subplots(n, 3, figsize=(15, max(2.8 * n, 4.5)), squeeze=False)
    for row, item in zip(axes, series, strict=True):
        ax0, ax1, ax2 = row
        totals = np.asarray(item.totals_sample, dtype=np.float64)
        posterior = np.asarray(item.eta_posterior_mean_sample, dtype=np.float64)
        ax0.hist(totals, bins=60, color="#155e75", alpha=0.82)
        ax0.axvline(
            np.median(totals), color="#c2410c", lw=1.5, ls="--", label="median N_c"
        )
        ax0.axvline(item.point_eta, color="#1d4ed8", lw=1.5, ls=":", label="point eta")
        ax0.set_title(f"{item.label} | totals")
        ax0.set_xlabel("N_c")
        ax0.set_ylabel("count")
        ax0.legend(frameon=False, fontsize=8)

        ax1.hist(posterior, bins=60, density=True, color="#0f766e", alpha=0.35)
        ax1.plot(item.eta_prior_grid, item.eta_prior_density, color="#1d4ed8", lw=2.0)
        ax1.set_title("Pool fit density")
        ax1.set_xlabel("eta")
        ax1.set_ylabel("density")

        lim = max(
            float(np.quantile(totals, 0.995)) if totals.size else 1.0,
            float(np.quantile(posterior, 0.995)) if posterior.size else 1.0,
            1.0,
        )
        ax2.scatter(
            totals, posterior, s=8, alpha=0.35, color="#c2410c", edgecolor="none"
        )
        ax2.plot([0, lim], [0, lim], color="#1f2430", lw=1.0, ls=":")
        ax2.set_xlim(0, lim)
        ax2.set_ylim(0, lim)
        ax2.set_title(f"N_c vs E[eta|N_c]\nmu={item.mu:.3f}, sigma={item.sigma:.3f}")
        ax2.set_xlabel("N_c")
        ax2.set_ylabel("posterior mean eta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pool_fit_traces(
    global_pool: PoolFitSnapshot,
    batch_pools: list[PoolFitSnapshot],
    out_path: Path,
) -> None:
    series = [global_pool] + batch_pools
    fig, ax = plt.subplots(figsize=(12, 7))
    for item in series:
        values = np.asarray(item.loglik_history, dtype=np.float64)
        if values.size == 0:
            continue
        ax.plot(
            np.arange(1, values.size + 1), values, lw=1.4, alpha=0.9, label=item.label
        )
    ax.set_title("Pool-scale EM log-likelihood traces")
    ax.set_xlabel("iteration")
    ax.set_ylabel("log-likelihood")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    with out_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "iteration", "loglik"])
        for item in series:
            for idx, value in enumerate(item.loglik_history, start=1):
                writer.writerow([item.label, idx, float(value)])


def _plot_prior_overlay(
    fitted_global: GeneFitResult,
    batch_results: list[GeneFitResult],
    refs: GlobalReferences,
    out_path: Path,
) -> None:
    x_max = refs.common_cutoff
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        refs.support,
        refs.fitted_density,
        color="black",
        lw=2.6,
        label="selected-global-fit",
        zorder=5,
    )
    ax.plot(
        refs.support,
        refs.pseudo_raw_density,
        color="#ea580c",
        lw=2.0,
        ls="--",
        label="pseudo-global-raw",
        zorder=4,
    )
    ax.plot(
        refs.support,
        refs.pseudo_aligned_density,
        color="#059669",
        lw=2.1,
        ls="-.",
        label="pseudo-global-aligned",
        zorder=4,
    )
    cmap = plt.get_cmap("tab20", max(len(batch_results), 1))
    for idx, result in enumerate(batch_results):
        result_density = _weights_to_density(result.support, result.prior_weights)
        ax.plot(
            result.support,
            result_density,
            color=cmap(idx),
            lw=1.4,
            alpha=0.85,
            label=result.label,
        )
    ax.set_xlabel("mu support")
    ax.set_ylabel("F_g")
    ax.set_title("Batch priors with fitted and pseudo global references")
    ax.set_xlim(0.0, x_max)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    with out_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "support", "prior_density"])
        for support, weight in zip(refs.support, refs.fitted_density, strict=True):
            writer.writerow(["selected-global-fit", float(support), float(weight)])
        for support, weight in zip(refs.support, refs.pseudo_raw_density, strict=True):
            writer.writerow(["pseudo-global-raw", float(support), float(weight)])
        for support, weight in zip(
            refs.support, refs.pseudo_aligned_density, strict=True
        ):
            writer.writerow(["pseudo-global-aligned", float(support), float(weight)])
        for result in batch_results:
            result_density = _weights_to_density(result.support, result.prior_weights)
            for support, weight in zip(result.support, result_density, strict=True):
                writer.writerow([result.label, float(support), float(weight)])


def _plot_prior_grid(
    fitted_global: GeneFitResult,
    batch_results: list[GeneFitResult],
    refs: GlobalReferences,
    out_path: Path,
) -> None:
    n = len(batch_results)
    x_max = refs.common_cutoff
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(12, max(2.8 * n, 4.0)),
        squeeze=False,
    )
    for ax, result in zip(axes.ravel(), batch_results, strict=True):
        result_density = _weights_to_density(result.support, result.prior_weights)
        ax.plot(
            refs.support,
            refs.fitted_density,
            color="black",
            lw=2.2,
            label="selected-global-fit",
        )
        ax.plot(
            refs.support,
            refs.pseudo_raw_density,
            color="#ea580c",
            lw=1.6,
            ls="--",
            label="pseudo-global-raw",
        )
        ax.plot(
            refs.support,
            refs.pseudo_aligned_density,
            color="#059669",
            lw=1.8,
            ls="-.",
            label="pseudo-global-aligned",
        )
        ax.plot(
            result.support,
            result_density,
            color="#2563eb",
            lw=1.8,
            label=result.label,
        )
        ax.set_title(f"{result.label} | n={result.n_cells:,}")
        ax.set_xlabel("mu support")
        ax.set_ylabel("F_g")
        ax.set_xlim(0.0, x_max)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right")
    fig.suptitle(
        "Per-batch F_g against fitted and pseudo global references", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    with out_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch_label", "curve_label", "support", "prior_density"])
        for result in batch_results:
            for support, weight in zip(refs.support, refs.fitted_density, strict=True):
                writer.writerow(
                    [result.label, "selected-global-fit", float(support), float(weight)]
                )
            for support, weight in zip(
                refs.support, refs.pseudo_raw_density, strict=True
            ):
                writer.writerow(
                    [result.label, "pseudo-global-raw", float(support), float(weight)]
                )
            for support, weight in zip(
                refs.support, refs.pseudo_aligned_density, strict=True
            ):
                writer.writerow(
                    [
                        result.label,
                        "pseudo-global-aligned",
                        float(support),
                        float(weight),
                    ]
                )
            for support, weight in zip(
                result.support,
                _weights_to_density(result.support, result.prior_weights),
                strict=True,
            ):
                writer.writerow(
                    [result.label, result.label, float(support), float(weight)]
                )


def _plot_distributions(
    global_result: GeneFitResult, batch_results: list[GeneFitResult], out_path: Path
) -> None:
    series = [global_result] + batch_results
    labels = [result.label for result in series]
    signal_data = [
        _sample_for_plot(result.signal, 5000, seed=11 + idx)
        for idx, result in enumerate(series)
    ]
    entropy_data = [
        _sample_for_plot(result.posterior_entropy, 5000, seed=211 + idx)
        for idx, result in enumerate(series)
    ]
    mi_data = [
        _sample_for_plot(result.mutual_information, 5000, seed=411 + idx)
        for idx, result in enumerate(series)
    ]
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    for ax, data, title in zip(
        axes,
        [signal_data, entropy_data, mi_data],
        [
            "Signal distributions",
            "Posterior entropy distributions",
            "Mutual information distributions",
        ],
        strict=True,
    ):
        parts = ax.violinplot(data, showmedians=True, widths=0.85)
        for body in parts["bodies"]:
            body.set_facecolor("#60a5fa")
            body.set_edgecolor("#1d4ed8")
            body.set_alpha(0.7)
        ax.set_title(title)
    axes[-1].set_xticks(np.arange(1, len(labels) + 1))
    axes[-1].set_xticklabels(labels, rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    with out_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "metric", "value"])
        for result in series:
            for value in result.signal:
                writer.writerow([result.label, "signal", float(value)])
            for value in result.posterior_entropy:
                writer.writerow([result.label, "posterior_entropy", float(value)])
            for value in result.mutual_information:
                writer.writerow([result.label, "mutual_information", float(value)])


def _plot_batch_summary(
    global_result: GeneFitResult, batch_results: list[GeneFitResult], out_path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    x = np.arange(len(batch_results))
    labels = [result.label for result in batch_results]
    summaries = [
        ("s_hat", [result.s_hat for result in batch_results], global_result.s_hat),
        (
            "mean signal",
            [float(np.mean(result.signal)) for result in batch_results],
            float(np.mean(global_result.signal)),
        ),
        (
            "mean posterior entropy",
            [float(np.mean(result.posterior_entropy)) for result in batch_results],
            float(np.mean(global_result.posterior_entropy)),
        ),
        (
            "mean mutual information",
            [float(np.mean(result.mutual_information)) for result in batch_results],
            float(np.mean(global_result.mutual_information)),
        ),
    ]
    for ax, (title, values, global_value) in zip(axes, summaries, strict=True):
        ax.plot(x, values, marker="o", color="#2563eb", lw=1.6)
        ax.axhline(global_value, color="black", ls="--", lw=1.6, label="global")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    with out_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "label", "value", "global_value"])
        for title, values, global_value in summaries:
            for label, value in zip(labels, values, strict=True):
                writer.writerow([title, label, float(value), float(global_value)])


def _plot_posterior_gallery(
    fitted_global: GeneFitResult,
    batch_results: list[GeneFitResult],
    refs: GlobalReferences,
    out_path: Path,
) -> None:
    n = len(batch_results)
    x_max = refs.common_cutoff
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(14, max(3.0 * n, 4.5)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )
    for row_axes, result in zip(axes, batch_results, strict=True):
        ax_prior, ax_post = row_axes
        result_density = _weights_to_density(result.support, result.prior_weights)
        ax_prior.plot(
            refs.support,
            refs.fitted_density,
            color="black",
            lw=1.6,
            ls="--",
            label="selected-global-fit",
        )
        ax_prior.plot(
            refs.support,
            refs.pseudo_raw_density,
            color="#ea580c",
            lw=1.3,
            ls=":",
            label="pseudo-global-raw",
        )
        ax_prior.plot(
            refs.support,
            refs.pseudo_aligned_density,
            color="#059669",
            lw=1.4,
            ls="-.",
            label="pseudo-global-aligned",
        )
        ax_prior.plot(
            result.support,
            result_density,
            color="#1d4ed8",
            lw=2.0,
            label="batch F_g",
        )
        scale_note = refs.batch_scales.get(result.label, 1.0)
        ax_prior.set_title(
            f"{result.label} | n={result.n_cells:,} | prior comparison | scale={scale_note:.3f}"
        )
        ax_prior.set_xlabel("mu support")
        ax_prior.set_ylabel("prior")
        ax_prior.set_xlim(0.0, x_max)
        for idx in range(result.posterior_samples.shape[0]):
            ax_post.plot(
                result.support,
                result.posterior_samples[idx],
                lw=1.0,
                alpha=0.75,
                label=None if idx else "representative cell posteriors",
            )
        ax_post.plot(
            refs.support,
            refs.fitted_density,
            color="black",
            lw=1.2,
            ls=":",
            label="selected-global-fit",
        )
        ax_post.plot(
            refs.support,
            refs.pseudo_aligned_density,
            color="#059669",
            lw=1.2,
            ls="-.",
            label="pseudo-global-aligned",
        )
        ax_post.set_xlabel("mu support")
        ax_post.set_ylabel("posterior")
        ax_post.set_xlim(0.0, x_max)
        ax_post.set_title("Representative cell posteriors")
        annotation_lines = ["cells chosen by signal quantiles"]
        for idx, (signal_value, entropy_value) in enumerate(
            zip(
                result.posterior_signal,
                result.posterior_entropy_samples,
                strict=True,
            )
        ):
            annotation_lines.append(
                f"c{idx + 1}: signal={float(signal_value):.2f}, Hpost={float(entropy_value):.2f}"
            )
        annotation_lines.append(f"batch align scale={scale_note:.3f}")
        ax_post.text(
            0.985,
            0.97,
            "\n".join(annotation_lines),
            transform=ax_post.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "alpha": 0.82,
                "edgecolor": "#cbd5e1",
            },
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    post_handles, post_labels = axes[0, 1].get_legend_handles_labels()
    handles += [h for h, l in zip(post_handles, post_labels) if l not in labels]
    labels += [l for l in post_labels if l not in labels]
    fig.legend(handles, labels, frameon=False, loc="upper right")
    fig.suptitle(
        "Representative posteriors by batch\nLeft: batch prior against fitted/raw/aligned global references. Right: representative cell posterior curves.",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    with out_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "batch_label",
                "curve_type",
                "sample_index",
                "sample_signal",
                "sample_posterior_entropy",
                "support",
                "value",
            ]
        )
        for result in batch_results:
            for support, weight in zip(refs.support, refs.fitted_density, strict=True):
                writer.writerow(
                    [
                        result.label,
                        "selected-global-fit",
                        -1,
                        "",
                        "",
                        float(support),
                        float(weight),
                    ]
                )
            for support, weight in zip(
                refs.support, refs.pseudo_raw_density, strict=True
            ):
                writer.writerow(
                    [
                        result.label,
                        "pseudo-global-raw",
                        -1,
                        "",
                        "",
                        float(support),
                        float(weight),
                    ]
                )
            for support, weight in zip(
                refs.support, refs.pseudo_aligned_density, strict=True
            ):
                writer.writerow(
                    [
                        result.label,
                        "pseudo-global-aligned",
                        -1,
                        "",
                        "",
                        float(support),
                        float(weight),
                    ]
                )
            for support, weight in zip(
                result.support,
                _weights_to_density(result.support, result.prior_weights),
                strict=True,
            ):
                writer.writerow(
                    [
                        result.label,
                        "batch_prior",
                        -1,
                        "",
                        "",
                        float(support),
                        float(weight),
                    ]
                )
            for idx, sample in enumerate(result.posterior_samples):
                for support, value in zip(result.support, sample, strict=True):
                    writer.writerow(
                        [
                            result.label,
                            "posterior",
                            idx,
                            float(result.posterior_signal[idx]),
                            float(result.posterior_entropy_samples[idx]),
                            float(support),
                            float(value),
                        ]
                    )


def _plot_global_scatter(global_result: GeneFitResult, out_path: Path) -> None:
    idx = _sample_indices(global_result.x_eff.size, 15000, seed=123)
    x_eff = global_result.x_eff[idx]
    signal = global_result.signal[idx]
    posterior_entropy = global_result.posterior_entropy[idx]
    mutual_information = global_result.mutual_information[idx]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc0 = axes[0].scatter(
        x_eff,
        signal,
        c=posterior_entropy,
        s=8,
        alpha=0.5,
        cmap="viridis",
        edgecolors="none",
    )
    axes[0].set_title("Global signal vs X_eff")
    axes[0].set_xlabel("X_eff")
    axes[0].set_ylabel("signal")
    fig.colorbar(sc0, ax=axes[0]).set_label("posterior entropy")
    sc1 = axes[1].scatter(
        x_eff,
        mutual_information,
        c=signal,
        s=8,
        alpha=0.5,
        cmap="plasma",
        edgecolors="none",
    )
    axes[1].set_title("Global mutual information vs X_eff")
    axes[1].set_xlabel("X_eff")
    axes[1].set_ylabel("mutual information")
    fig.colorbar(sc1, ax=axes[1]).set_label("signal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    with out_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_eff", "signal", "posterior_entropy", "mutual_information"])
        for row in zip(
            x_eff, signal, posterior_entropy, mutual_information, strict=True
        ):
            writer.writerow(
                [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
            )


def _write_summary_csv(
    global_result: GeneFitResult, batch_results: list[GeneFitResult], out_path: Path
) -> None:
    rows = [global_result] + batch_results
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "batch_value",
                "n_cells",
                "s_hat",
                "pool_mu",
                "pool_sigma",
                "pool_point_eta",
                "prior_grid_max",
                "final_loss",
                "best_loss",
                "mean_count",
                "detected_fraction",
                "prior_mode",
                "prior_mean",
                "prior_q50",
                "prior_q80",
                "prior_q90",
                "prior_q99",
                "mean_signal",
                "median_signal",
                "mean_posterior_entropy",
                "mean_prior_entropy",
                "mean_mutual_information",
            ],
        )
        writer.writeheader()
        for row in rows:
            prior_density = _weights_to_density(row.support, row.prior_weights)
            writer.writerow(
                {
                    "label": row.label,
                    "batch_value": "" if row.batch_value is None else row.batch_value,
                    "n_cells": row.n_cells,
                    "s_hat": row.s_hat,
                    "pool_mu": row.pool_mu,
                    "pool_sigma": row.pool_sigma,
                    "pool_point_eta": row.pool_point_eta,
                    "prior_grid_max": row.prior_grid_max,
                    "final_loss": row.final_loss,
                    "best_loss": row.best_loss,
                    "mean_count": float(np.mean(row.counts)),
                    "detected_fraction": float(np.mean(row.counts > 0)),
                    "prior_mode": _prior_mode(row.support, row.prior_weights),
                    "prior_mean": _prior_mean(row.support, row.prior_weights),
                    "prior_q50": _curve_quantile(row.support, prior_density, 0.50),
                    "prior_q80": _curve_quantile(row.support, prior_density, 0.80),
                    "prior_q90": _curve_quantile(row.support, prior_density, 0.90),
                    "prior_q99": _curve_quantile(row.support, prior_density, 0.99),
                    "mean_signal": float(np.mean(row.signal)),
                    "median_signal": float(np.median(row.signal)),
                    "mean_posterior_entropy": float(np.mean(row.posterior_entropy)),
                    "mean_prior_entropy": float(np.mean(row.prior_entropy)),
                    "mean_mutual_information": float(np.mean(row.mutual_information)),
                }
            )


def _write_global_reference_csv(refs: GlobalReferences, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "support",
                "selected_global_fit",
                "pseudo_global_raw",
                "pseudo_global_aligned",
                "common_cutoff",
            ]
        )
        for row in zip(
            refs.support,
            refs.fitted_density,
            refs.pseudo_raw_density,
            refs.pseudo_aligned_density,
            strict=True,
        ):
            writer.writerow(
                [
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(refs.common_cutoff),
                ]
            )


def _write_alignment_scale_csv(refs: GlobalReferences, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch_label", "scale"])
        for label, scale in refs.batch_scales.items():
            writer.writerow([label, float(scale)])


def _write_s_hat_mu_analysis_csv(
    gene_name: str,
    global_result: GeneFitResult,
    batch_results: list[GeneFitResult],
    out_path: Path,
) -> None:
    rows = [global_result] + batch_results
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gene_name",
                "label",
                "batch_value",
                "n_cells",
                "s_hat",
                "prior_mean",
                "prior_mode",
                "prior_q50",
                "prior_q80",
                "prior_q90",
                "prior_q99",
                "mean_signal",
            ]
        )
        for row in rows:
            prior_density = _weights_to_density(row.support, row.prior_weights)
            writer.writerow(
                [
                    gene_name,
                    row.label,
                    "" if row.batch_value is None else row.batch_value,
                    row.n_cells,
                    row.s_hat,
                    _prior_mean(row.support, row.prior_weights),
                    _prior_mode(row.support, row.prior_weights),
                    _curve_quantile(row.support, prior_density, 0.50),
                    _curve_quantile(row.support, prior_density, 0.80),
                    _curve_quantile(row.support, prior_density, 0.90),
                    _curve_quantile(row.support, prior_density, 0.99),
                    float(np.mean(row.signal)),
                ]
            )


def _summarize_s_hat_mu_relationship(
    gene_name: str,
    global_result: GeneFitResult,
    batch_results: list[GeneFitResult],
) -> dict[str, float | str | int]:
    if not batch_results:
        return {
            "gene_name": gene_name,
            "n_batches": 0,
            "corr_s_hat_prior_q80": float("nan"),
            "corr_s_hat_prior_mean": float("nan"),
            "corr_s_hat_mean_signal": float("nan"),
            "cv_s_hat": float("nan"),
            "cv_prior_q80": float("nan"),
            "cv_prior_mean": float("nan"),
            "cv_mean_signal": float("nan"),
            "var_ratio_q80_over_s_hat": float("nan"),
            "var_ratio_prior_mean_over_s_hat": float("nan"),
            "var_ratio_mean_signal_over_s_hat": float("nan"),
            "global_prior_q80": float("nan"),
            "global_prior_mean": float("nan"),
            "global_mean_signal": float(np.mean(global_result.signal)),
        }
    s_hat = np.asarray([row.s_hat for row in batch_results], dtype=np.float64)
    prior_mean = np.asarray(
        [_prior_mean(row.support, row.prior_weights) for row in batch_results],
        dtype=np.float64,
    )
    prior_q80 = np.asarray(
        [
            _curve_quantile(
                row.support,
                _weights_to_density(row.support, row.prior_weights),
                0.80,
            )
            for row in batch_results
        ],
        dtype=np.float64,
    )
    mean_signal = np.asarray(
        [float(np.mean(row.signal)) for row in batch_results], dtype=np.float64
    )

    def cv(values: np.ndarray) -> float:
        mean = float(np.mean(values))
        return float(np.std(values) / mean) if abs(mean) > 1e-12 else float("nan")

    s_hat_var = float(np.var(s_hat))
    return {
        "gene_name": gene_name,
        "n_batches": int(len(batch_results)),
        "corr_s_hat_prior_q80": _safe_corr(s_hat, prior_q80),
        "corr_s_hat_prior_mean": _safe_corr(s_hat, prior_mean),
        "corr_s_hat_mean_signal": _safe_corr(s_hat, mean_signal),
        "cv_s_hat": cv(s_hat),
        "cv_prior_q80": cv(prior_q80),
        "cv_prior_mean": cv(prior_mean),
        "cv_mean_signal": cv(mean_signal),
        "var_ratio_q80_over_s_hat": float(np.var(prior_q80) / s_hat_var)
        if s_hat_var > 0
        else float("nan"),
        "var_ratio_prior_mean_over_s_hat": float(np.var(prior_mean) / s_hat_var)
        if s_hat_var > 0
        else float("nan"),
        "var_ratio_mean_signal_over_s_hat": float(np.var(mean_signal) / s_hat_var)
        if s_hat_var > 0
        else float("nan"),
        "global_prior_q80": _curve_quantile(
            global_result.support,
            _weights_to_density(global_result.support, global_result.prior_weights),
            0.80,
        ),
        "global_prior_mean": _prior_mean(
            global_result.support, global_result.prior_weights
        ),
        "global_mean_signal": float(np.mean(global_result.signal)),
    }


def _write_s_hat_mu_overall_summary(
    rows: list[dict[str, float | str | int]], out_path: Path
) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_summary(
    gene_name: str, global_result: GeneFitResult, batch_results: list[GeneFitResult]
) -> None:
    table = Table(title=f"Batch gene fit summary | {gene_name}")
    for column in ("label", "cells", "s_hat", "mean signal", "mean entropy", "mean MI"):
        table.add_column(column)
    for result in [global_result] + batch_results:
        table.add_row(
            result.label,
            f"{result.n_cells:,}",
            f"{result.s_hat:.3f}",
            f"{float(np.mean(result.signal)):.4f}",
            f"{float(np.mean(result.posterior_entropy)):.4f}",
            f"{float(np.mean(result.mutual_information)):.4f}",
        )
    console.print(table)


def _save_gene_outputs(
    *,
    gene_name: str,
    output_root: Path,
    global_result: GeneFitResult,
    batch_results: list[GeneFitResult],
    refs: GlobalReferences,
) -> None:
    gene_dir = output_root / f"gene_{_sanitize_label(gene_name)}"
    gene_dir.mkdir(parents=True, exist_ok=True)
    _plot_prior_overlay(
        global_result, batch_results, refs, gene_dir / "prior_overlay.png"
    )
    _plot_prior_grid(
        global_result, batch_results, refs, gene_dir / "prior_small_multiples.png"
    )
    _plot_distributions(
        global_result, batch_results, gene_dir / "signal_entropy_mi_distributions.png"
    )
    _plot_batch_summary(global_result, batch_results, gene_dir / "batch_summary.png")
    _plot_posterior_gallery(
        global_result, batch_results, refs, gene_dir / "posterior_gallery.png"
    )
    _plot_global_scatter(global_result, gene_dir / "global_scatter.png")
    _write_summary_csv(global_result, batch_results, gene_dir / "summary.csv")
    _write_global_reference_csv(refs, gene_dir / "global_references.csv")
    _write_alignment_scale_csv(refs, gene_dir / "alignment_scales.csv")
    _print_summary(gene_name, global_result, batch_results)


@app.command()
def main(
    h5ad: Path = typer.Argument(
        ..., exists=True, readable=True, help="Input h5ad file."
    ),
    gene: str | None = typer.Argument(None, help="Gene name or gene index."),
    batch_key: str = typer.Option("batch", help="obs column used to split batches."),
    random_select_batch: bool = typer.Option(
        False,
        help="Randomly reassign cells into the selected batch sizes instead of using real batch membership.",
    ),
    random_seed: int = typer.Option(0, help="Random seed for --random-select-batch."),
    ranked_genes: Path | None = typer.Option(
        None,
        exists=True,
        readable=True,
        help="Optional ranked gene list file, one gene per line.",
    ),
    gene_rank: int | None = typer.Option(
        None, min=1, help="1-based rank to select from --ranked-genes."
    ),
    top_ranked: int | None = typer.Option(
        None, min=1, help="Analyze top-N genes from --ranked-genes in one run."
    ),
    layer: str | None = typer.Option(None, help="Layer name. Defaults to X."),
    total_key: str | None = typer.Option(
        None, help="obs column used as totals. Auto-infer when omitted."
    ),
    outdir: Path | None = typer.Option(
        None, help="Directory for saved figures and CSV summary."
    ),
    r: float = typer.Option(0.05, min=1e-12, help="Pool-scale r hyperparameter."),
    device: str = typer.Option(
        "cpu", help="Torch device for fitting and posterior extraction."
    ),
    grid_size: int = typer.Option(512, min=16, help="Prior grid size."),
    sigma_bins: float = typer.Option(
        1.0, min=0.0, help="Gaussian smoothing sigma in bins."
    ),
    align_loss_weight: float = typer.Option(
        1.0, min=0.0, help="Alignment loss weight."
    ),
    torch_dtype: str = typer.Option("float64", help="Torch dtype: float64 or float32."),
    lr: float = typer.Option(0.05, min=1e-8, help="Learning rate."),
    n_iter: int = typer.Option(100, min=1, help="Prior fitting iterations."),
    lr_min_ratio: float = typer.Option(
        0.1, min=0.0, help="Scheduler minimum LR ratio."
    ),
    grad_clip: float | None = typer.Option(None, help="Optional gradient clipping."),
    init_temperature: float = typer.Option(
        1.0, min=1e-8, help="Initialization temperature."
    ),
    cell_chunk_size: int = typer.Option(
        512, min=1, help="Cell chunk size used during prior fitting."
    ),
    optimizer: str = typer.Option("adamw", help="Optimizer name."),
    scheduler: str = typer.Option("cosine", help="Scheduler name."),
    fit_gene_batch_size: int = typer.Option(
        16, min=1, help="Number of genes fitted jointly per chunk to avoid OOM."
    ),
    shared_grid_quantile: float = typer.Option(
        0.99,
        min=1e-6,
        max=0.999999,
        help="Shared pooled count/total quantile used to define a common support-ratio max across all fits.",
    ),
    posterior_chunk_size: int = typer.Option(
        4096, min=1, help="Cell chunk size used during posterior extraction."
    ),
    representative_cells: int = typer.Option(
        4, min=2, help="Representative cells per fit for posterior gallery."
    ),
) -> int:
    resolved_device = _resolve_device(device)
    adata = ad.read_h5ad(h5ad, backed="r")
    try:
        if batch_key not in adata.obs.columns:
            raise KeyError(f"batch key {batch_key!r} not found in obs")
        matrix = select_matrix(adata, layer)
        gene_names_all = np.asarray(adata.var_names.astype(str))
        gene_names_lower = tuple(str(name).lower() for name in gene_names_all.tolist())
        gene_to_idx = {
            str(name): int(idx) for idx, name in enumerate(gene_names_all.tolist())
        }
        gene_lower_to_idx: dict[str, int] = {}
        for idx, name in enumerate(gene_names_lower):
            gene_lower_to_idx.setdefault(name, idx)

        selected_gene_names, selected_gene_indices, selection_note = (
            _resolve_selected_genes(
                gene=gene,
                ranked_genes=ranked_genes,
                gene_rank=gene_rank,
                top_ranked=top_ranked,
                gene_names=gene_names_all,
                gene_names_lower=gene_names_lower,
                gene_to_idx=gene_to_idx,
                gene_lower_to_idx=gene_lower_to_idx,
            )
        )
        counts_matrix = _slice_gene_matrix(matrix, selected_gene_indices)

        resolved_total_key = _infer_total_key(adata, total_key)
        if resolved_total_key is None:
            console.print(
                "[yellow]No total column found in obs; computing totals from matrix.[/yellow]"
            )
            totals = _compute_totals(matrix)
        else:
            totals = np.asarray(adata.obs[resolved_total_key], dtype=DTYPE_NP).reshape(
                -1
            )
        shared_support_ratio_max = _compute_shared_ratio_quantiles(
            counts_matrix,
            totals,
            shared_grid_quantile,
        )

        real_batch_labels = np.asarray(adata.obs[batch_key].astype(str)).reshape(-1)
        all_batch_values = _top_batch_values(
            real_batch_labels, int(np.unique(real_batch_labels).size)
        )
        batch_limit = (
            len(all_batch_values)
            if top_ranked is None
            else min(top_ranked, len(all_batch_values))
        )
        top_values, batch_sizes = _select_batch_values(real_batch_labels, batch_limit)
        batch_mode = "randomized" if random_select_batch else "real"
        if random_select_batch:
            batch_labels, top_values, batch_sizes = _randomize_selected_batches(
                n_cells=adata.n_obs,
                batch_sizes=batch_sizes,
                seed=random_seed,
            )
        else:
            batch_labels = real_batch_labels
        default_name = (
            _sanitize_label(selected_gene_names[0])
            if len(selected_gene_names) == 1
            else f"top{len(selected_gene_names)}"
        )
        output_dir = (
            outdir.expanduser().resolve()
            if outdir is not None
            else (
                PROJECT_ROOT
                / "output"
                / f"batch_gene_{h5ad.stem}_{default_name}_{batch_mode}"
            ).resolve()
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[bold cyan]Genes[/bold cyan]: {len(selected_gene_names)}")
        console.print(
            f"[bold cyan]Gene list[/bold cyan]: {', '.join(selected_gene_names[:8])}{' ...' if len(selected_gene_names) > 8 else ''}"
        )
        if selection_note is not None:
            console.print(f"[bold cyan]Selection[/bold cyan]: {selection_note}")
        console.print(f"[bold cyan]Cells[/bold cyan]: {adata.n_obs:,}")
        console.print(f"[bold cyan]Batch mode[/bold cyan]: {batch_mode}")
        if random_select_batch:
            console.print(f"[bold cyan]Random seed[/bold cyan]: {random_seed}")
        console.print(f"[bold cyan]Batches[/bold cyan]: {len(top_values)}")
        console.print(f"[bold cyan]Fit gene chunk[/bold cyan]: {fit_gene_batch_size}")
        console.print(f"[bold cyan]Shared grid q[/bold cyan]: {shared_grid_quantile}")
        console.print(f"[bold cyan]Output[/bold cyan]: {output_dir}")

        batch_results_by_gene = {name: [] for name in selected_gene_names}
        batch_pool_snapshots: list[PoolFitSnapshot] = []
        with _make_progress() as progress:
            n_gene_chunks = len(
                _iter_gene_chunks(len(selected_gene_names), fit_gene_batch_size)
            )
            global_total = (
                120
                + max(n_iter, 1) * n_gene_chunks
                + max(math.ceil(len(totals) / posterior_chunk_size), 1) * n_gene_chunks
                + len(selected_gene_names)
                + n_gene_chunks
                + 1
            )
            global_task = progress.add_task("global | waiting", total=global_total)
            global_results, global_pool_snapshot = _fit_group_multi(
                gene_names=selected_gene_names,
                counts_matrix=counts_matrix,
                totals=totals,
                label="global",
                batch_value=None,
                r=r,
                device=resolved_device,
                grid_size=grid_size,
                sigma_bins=sigma_bins,
                align_loss_weight=align_loss_weight,
                torch_dtype=torch_dtype,
                lr=lr,
                n_iter=n_iter,
                lr_min_ratio=lr_min_ratio,
                grad_clip=grad_clip,
                init_temperature=init_temperature,
                cell_chunk_size=cell_chunk_size,
                optimizer=optimizer,
                scheduler=scheduler,
                fit_gene_batch_size=fit_gene_batch_size,
                shared_support_ratio_max=shared_support_ratio_max,
                posterior_chunk_size=posterior_chunk_size,
                representative_cells=representative_cells,
                progress=progress,
                group_task_id=global_task,
            )
            for batch_value in top_values:
                mask = batch_labels == batch_value
                n_batch_cells = int(np.sum(mask))
                batch_total = (
                    120
                    + max(n_iter, 1) * n_gene_chunks
                    + max(math.ceil(n_batch_cells / posterior_chunk_size), 1)
                    * n_gene_chunks
                    + len(selected_gene_names)
                    + n_gene_chunks
                    + 1
                )
                batch_task = progress.add_task(
                    f"batch_{batch_value} | waiting", total=batch_total
                )
                batch_results, batch_pool_snapshot = _fit_group_multi(
                    gene_names=selected_gene_names,
                    counts_matrix=counts_matrix[mask],
                    totals=totals[mask],
                    label=f"batch_{batch_value}",
                    batch_value=batch_value,
                    r=r,
                    device=resolved_device,
                    grid_size=grid_size,
                    sigma_bins=sigma_bins,
                    align_loss_weight=align_loss_weight,
                    torch_dtype=torch_dtype,
                    lr=lr,
                    n_iter=n_iter,
                    lr_min_ratio=lr_min_ratio,
                    grad_clip=grad_clip,
                    init_temperature=init_temperature,
                    cell_chunk_size=cell_chunk_size,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    fit_gene_batch_size=fit_gene_batch_size,
                    shared_support_ratio_max=shared_support_ratio_max,
                    posterior_chunk_size=posterior_chunk_size,
                    representative_cells=representative_cells,
                    progress=progress,
                    group_task_id=batch_task,
                )
                batch_pool_snapshots.append(batch_pool_snapshot)
                for gene_name in selected_gene_names:
                    batch_results_by_gene[gene_name].append(batch_results[gene_name])

        _plot_batch_sizes(batch_sizes, output_dir / "batch_sizes.png")
        _plot_pool_fit_summary(
            global_pool_snapshot,
            batch_pool_snapshots,
            output_dir / "pool_fit_summary.png",
        )
        _plot_pool_fit_gallery(
            global_pool_snapshot,
            batch_pool_snapshots,
            output_dir / "pool_fit_gallery.png",
        )
        _plot_pool_fit_traces(
            global_pool_snapshot,
            batch_pool_snapshots,
            output_dir / "pool_fit_traces.png",
        )
        (output_dir / "selected_genes.txt").write_text(
            "\n".join(selected_gene_names) + "\n", encoding="utf-8"
        )
        with (output_dir / "run_config.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["key", "value"])
            writer.writerow(["h5ad", str(h5ad)])
            writer.writerow(["batch_key", batch_key])
            writer.writerow(["batch_mode", batch_mode])
            writer.writerow(["random_seed", random_seed if random_select_batch else ""])
            writer.writerow(["n_selected_genes", len(selected_gene_names)])
            writer.writerow(["n_selected_batches", len(top_values)])
            writer.writerow(["fit_gene_batch_size", fit_gene_batch_size])
            writer.writerow(["shared_grid_quantile", shared_grid_quantile])
            writer.writerow(["r", r])
            writer.writerow(["device", resolved_device])
        with (output_dir / "selected_batches.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["batch_label", "n_cells"])
            writer.writerows(batch_sizes)
        s_hat_mu_rows: list[dict[str, float | str | int]] = []
        for gene_name in selected_gene_names:
            refs = _build_global_references(
                global_results[gene_name], batch_results_by_gene[gene_name]
            )
            _save_gene_outputs(
                gene_name=gene_name,
                output_root=output_dir,
                global_result=global_results[gene_name],
                batch_results=batch_results_by_gene[gene_name],
                refs=refs,
            )
            _write_s_hat_mu_analysis_csv(
                gene_name,
                global_results[gene_name],
                batch_results_by_gene[gene_name],
                output_dir
                / f"gene_{_sanitize_label(gene_name)}"
                / "s_hat_mu_analysis.csv",
            )
            s_hat_mu_rows.append(
                _summarize_s_hat_mu_relationship(
                    gene_name,
                    global_results[gene_name],
                    batch_results_by_gene[gene_name],
                )
            )
        _write_s_hat_mu_overall_summary(
            s_hat_mu_rows,
            output_dir / "s_hat_mu_relationship_summary.csv",
        )
        console.print("[bold green]Saved figures and summaries.[/bold green]")
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(app())
