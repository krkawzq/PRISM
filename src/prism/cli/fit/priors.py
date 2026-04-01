from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import anndata as ad
import numpy as np
import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from prism.io import read_string_list
from prism.model import (
    ModelCheckpoint,
    ObservationBatch,
    PriorFitConfig,
    PriorGrid,
    ScaleMetadata,
    fit_gene_priors,
    fit_gene_priors_em,
    load_checkpoint,
    save_checkpoint,
)

from .common import (
    build_fit_tasks,
    checkpoint_gene_names,
    compute_reference_counts,
    ensure_dense_matrix,
    parse_shard,
    print_fit_plan,
    print_fit_summary,
    resolve_fit_gene_list,
    resolve_fit_mode,
    resolve_gene_list,
    resolve_label_groups,
    read_gene_list,
    sample_indices,
    select_matrix,
    shard_gene_names,
    slice_gene_counts,
    console,
)


def fit_priors_command(
    h5ad_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad file."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output checkpoint path."
    ),
    reference_genes_path: Path | None = typer.Option(
        None,
        "--reference-genes",
        exists=True,
        dir_okay=False,
        help="Optional text file with reference gene names used to compute reference counts. Defaults to all genes in the dataset.",
    ),
    fit_genes_path: Path | None = typer.Option(
        None,
        "--fit-genes",
        exists=True,
        dir_okay=False,
        help="Optional text file with genes to fit. Defaults to all genes in the dataset.",
    ),
    layer: str | None = typer.Option(None, help="Input AnnData layer. Defaults to X."),
    label_key: str | None = typer.Option(
        None, help="Optional obs column used for class-specific fitting."
    ),
    label_values: list[str] | None = typer.Option(
        None,
        "--label-value",
        help="Optional repeatable label values to fit when --label-key is set.",
    ),
    label_list_path: Path | None = typer.Option(
        None,
        "--label-list",
        exists=True,
        dir_okay=False,
        help="Optional text/JSON file with label values.",
    ),
    fit_mode: str = typer.Option(
        "global", help="Fit scope: global, by-label, or both."
    ),
    n_samples: int | None = typer.Option(
        None, min=1, help="Optional random cell subsample size per fit scope."
    ),
    sample_seed: int = typer.Option(0, min=0, help="Random seed for cell subsampling."),
    S: float | None = typer.Option(
        None,
        "--S",
        "--s",
        min=1e-12,
        help="Global scale marker used by the model. Defaults to mean reference count N_avg.",
    ),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda."),
    gene_batch_size: int = typer.Option(
        64,
        min=1,
        help="Number of genes to fit per batch. Controls memory usage during prior fitting.",
    ),
    shard: str = typer.Option(
        "0/1", help="Shard specification as rank/world, e.g. 0/4."
    ),
    grid_size: int = typer.Option(512, min=2, help="Prior grid size."),
    sigma_bins: float = typer.Option(
        1.0, min=0.0, help="Gaussian smoothing sigma in grid bins."
    ),
    align_loss_weight: float = typer.Option(
        1.0, min=0.0, help="Posterior-prior alignment weight."
    ),
    align_every: int = typer.Option(
        1, min=1, help="Recompute the alignment term every N optimization steps."
    ),
    lr: float = typer.Option(0.05, min=1e-12, help="Optimizer learning rate."),
    n_iter: int = typer.Option(100, min=1, help="Optimization iterations."),
    lr_min_ratio: float = typer.Option(
        0.1, min=0.0, help="Scheduler minimum lr ratio."
    ),
    grad_clip: float | None = typer.Option(
        None, min=0.0, help="Optional gradient clipping threshold."
    ),
    early_stop_tol: float | None = typer.Option(
        None,
        min=0.0,
        help="Optional minimum loss improvement required to reset early stopping.",
    ),
    early_stop_patience: int | None = typer.Option(
        None,
        min=1,
        help="Optional number of non-improving steps allowed before stopping early.",
    ),
    init_method: str = typer.Option(
        "posterior_mean",
        help="Initialization method: posterior_mean, uniform, or random.",
    ),
    init_seed: int = typer.Option(
        0, min=0, help="Random seed used by random initialization."
    ),
    init_temperature: float = typer.Option(
        1.0, min=1e-12, help="Initialization temperature."
    ),
    cell_chunk_size: int = typer.Option(
        512,
        min=1,
        help="Number of cells per likelihood computation chunk. Controls GPU memory usage.",
    ),
    optimizer: str = typer.Option("adamw", help="Optimizer name."),
    scheduler: str = typer.Option("cosine", help="Scheduler name."),
    torch_dtype: str = typer.Option("float64", help="Torch dtype: float64 or float32."),
    grid_max_method: str = typer.Option(
        "observed_max",
        help="Grid max bound method: observed_max or quantile.",
    ),
    grid_strategy: str = typer.Option(
        "linear",
        help="Grid spacing strategy: linear or sqrt.",
    ),
    sigma_anneal_start: float | None = typer.Option(
        None,
        min=0.0,
        help="Optional sigma_bins value at the start of training (annealing from large to small).",
    ),
    sigma_anneal_end: float | None = typer.Option(
        None,
        min=0.0,
        help="Optional sigma_bins value at the end of training.",
    ),
    adaptive_grid: bool = typer.Option(
        False,
        help="Enable two-phase adaptive grid refinement.",
    ),
    adaptive_grid_fraction: float = typer.Option(
        0.3,
        min=0.01,
        max=1.0,
        help="Fraction of grid points concentrated in high-mass region during adaptive refinement.",
    ),
    adaptive_grid_quantile_lo: float = typer.Option(
        0.01,
        min=0.0,
        max=0.999999,
        help="Lower posterior mass quantile used to define the adaptive-grid focus region.",
    ),
    adaptive_grid_quantile_hi: float = typer.Option(
        0.99,
        min=0.000001,
        max=1.0,
        help="Upper posterior mass quantile used to define the adaptive-grid focus region.",
    ),
    cell_sample_fraction: float = typer.Option(
        1.0,
        min=0.01,
        max=1.0,
        help="Fraction of cell chunks to sample per step (mini-batch SGD). 1.0 = use all.",
    ),
    cell_sample_seed: int = typer.Option(
        0,
        min=0,
        help="Random seed for cell chunk sampling.",
    ),
    align_mode: str = typer.Option(
        "jsd",
        help="Alignment loss mode: jsd, kl, or weighted_jsd.",
    ),
    shrinkage_weight: float = typer.Option(
        0.0,
        min=0.0,
        max=1.0,
        help="Weight for shrinking per-gene priors toward the batch mean. 0 = no shrinkage.",
    ),
    ensemble_restarts: int = typer.Option(
        1,
        min=1,
        help="Number of random restarts; best result is kept.",
    ),
    likelihood: str = typer.Option(
        "binomial",
        help="Ablation fit distribution: binomial, negative_binomial, or poisson.",
    ),
    nb_overdispersion: float = typer.Option(
        0.01,
        min=1e-6,
        help="Overdispersion parameter for negative binomial likelihood (1/r).",
    ),
    warm_start_checkpoint: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional checkpoint used to initialize prior weights when matching genes are available.",
    ),
    fit_method: str = typer.Option(
        "gradient",
        help="Prior fitting method: gradient or em.",
    ),
    em_tol: float = typer.Option(
        1e-6,
        min=0.0,
        help="EM early-stop tolerance on the max absolute weight update.",
    ),
    dry_run: bool = typer.Option(
        False, help="Show the execution plan without fitting."
    ),
) -> int:
    start_time = perf_counter()
    h5ad_path = h5ad_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    reference_genes_path = (
        None
        if reference_genes_path is None
        else reference_genes_path.expanduser().resolve()
    )
    fit_genes_path = (
        None if fit_genes_path is None else fit_genes_path.expanduser().resolve()
    )
    label_list_path = (
        None if label_list_path is None else label_list_path.expanduser().resolve()
    )
    warm_start_checkpoint = (
        None
        if warm_start_checkpoint is None
        else warm_start_checkpoint.expanduser().resolve()
    )

    rank, world_size = parse_shard(shard)
    fit_method_resolved = fit_method.strip().lower()
    if fit_method_resolved not in {"gradient", "em"}:
        raise ValueError("fit_method must be one of: gradient, em")
    grid_max_method_resolved = grid_max_method.strip().lower()
    if grid_max_method_resolved not in {"observed_max", "quantile"}:
        raise ValueError("grid_max_method must be one of: observed_max, quantile")
    grid_strategy_resolved = grid_strategy.strip().lower()
    if grid_strategy_resolved not in {"linear", "sqrt"}:
        raise ValueError("grid_strategy must be one of: linear, sqrt")
    align_mode_resolved = align_mode.strip().lower()
    if align_mode_resolved not in {"jsd", "kl", "weighted_jsd"}:
        raise ValueError("align_mode must be one of: jsd, kl, weighted_jsd")
    likelihood_resolved = likelihood.strip().lower()
    if likelihood_resolved not in {"binomial", "negative_binomial", "poisson"}:
        raise ValueError(
            "likelihood must be one of: binomial, negative_binomial, poisson"
        )
    fit_config = PriorFitConfig(
        grid_size=grid_size,
        sigma_bins=sigma_bins,
        align_loss_weight=align_loss_weight,
        align_every=align_every,
        lr=lr,
        n_iter=n_iter,
        lr_min_ratio=lr_min_ratio,
        grad_clip=grad_clip,
        early_stop_tol=early_stop_tol,
        early_stop_patience=early_stop_patience,
        init_method=cast(Any, init_method),
        init_seed=init_seed,
        init_temperature=init_temperature,
        cell_chunk_size=cell_chunk_size,
        optimizer=cast(Any, optimizer),
        scheduler=cast(Any, scheduler),
        torch_dtype=cast(Any, torch_dtype),
        grid_max_method=cast(Any, grid_max_method_resolved),
        grid_strategy=cast(Any, grid_strategy_resolved),
        sigma_anneal_start=sigma_anneal_start,
        sigma_anneal_end=sigma_anneal_end,
        adaptive_grid=adaptive_grid,
        adaptive_grid_fraction=adaptive_grid_fraction,
        adaptive_grid_quantile_lo=adaptive_grid_quantile_lo,
        adaptive_grid_quantile_hi=adaptive_grid_quantile_hi,
        cell_sample_fraction=cell_sample_fraction,
        cell_sample_seed=cell_sample_seed,
        align_mode=cast(Any, align_mode_resolved),
        shrinkage_weight=shrinkage_weight,
        ensemble_restarts=ensemble_restarts,
        likelihood=cast(Any, likelihood_resolved),
        nb_overdispersion=nb_overdispersion,
    )
    warm_start = (
        None
        if warm_start_checkpoint is None
        else load_checkpoint(warm_start_checkpoint)
    )

    console.print(f"[bold cyan]Reading[/bold cyan] {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    matrix = ensure_dense_matrix(select_matrix(adata, layer))
    console.print(
        f"[bold cyan]Matrix[/bold cyan] densified to numpy array with shape {matrix.shape}"
    )
    gene_names = [str(name) for name in adata.var_names.tolist()]
    gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    fit_mode_resolved = resolve_fit_mode(fit_mode)

    if reference_genes_path is None:
        reference_gene_names = list(gene_names)
        missing_reference: list[str] = []
    else:
        reference_gene_names, missing_reference = resolve_gene_list(
            reference_genes_path, gene_to_idx
        )
        if not reference_gene_names:
            raise ValueError("reference gene list has no overlap with the dataset")
    fit_gene_names, missing_fit = resolve_fit_gene_list(
        fit_genes_path, gene_names, gene_to_idx
    )
    if not fit_gene_names:
        raise ValueError("fit gene list is empty after intersecting with the dataset")
    shard_gene_names_list = shard_gene_names(
        fit_gene_names, rank=rank, world_size=world_size
    )
    if not shard_gene_names_list:
        raise ValueError(f"shard {rank}/{world_size} has no assigned genes")

    reference_positions = [gene_to_idx[name] for name in reference_gene_names]
    reference_counts = compute_reference_counts(matrix, reference_positions)
    default_S = float(np.mean(reference_counts))
    resolved_S = default_S if S is None else float(S)
    S_source = "default:N_avg" if S is None else "user"
    resolved_label_values = None
    if label_values or label_list_path is not None:
        merged_labels: list[str] = []
        if label_values:
            merged_labels.extend(label_values)
        if label_list_path is not None:
            merged_labels.extend(read_string_list(label_list_path))
        resolved_label_values = list(dict.fromkeys(merged_labels))
    label_groups = resolve_label_groups(
        adata=adata,
        label_key=label_key,
        label_values=resolved_label_values,
        fit_mode=fit_mode_resolved,
    )
    label_group_sizes = [int(indices.shape[0]) for _, indices in label_groups]
    print_fit_plan(
        h5ad_path=h5ad_path,
        layer=layer,
        n_cells=int(adata.n_obs),
        n_dataset_genes=int(adata.n_vars),
        n_reference_genes=len(reference_gene_names),
        n_fit_genes=len(fit_gene_names),
        n_shard_genes=len(shard_gene_names_list),
        fit_mode=fit_mode_resolved,
        label_key=label_key,
        n_label_groups=len(label_groups),
        n_requested_label_values=(
            None if resolved_label_values is None else len(resolved_label_values)
        ),
        n_samples=n_samples,
        sample_seed=sample_seed,
        S=resolved_S,
        S_source=S_source,
        N_avg=default_S,
        device=device,
        fit_method=fit_method_resolved,
        em_tol=em_tol,
        align_every=align_every,
        early_stop_tol=early_stop_tol,
        early_stop_patience=early_stop_patience,
        init_method=init_method,
        init_seed=init_seed,
        warm_start_checkpoint=warm_start_checkpoint,
        grid_max_method=grid_max_method_resolved,
        grid_strategy=grid_strategy_resolved,
        gene_batch_size=gene_batch_size,
        shard=f"{rank}/{world_size}",
        output_path=output_path,
        label_group_size_mean=(
            None
            if not label_group_sizes
            else round(float(np.mean(label_group_sizes)), 2)
        ),
        label_group_size_min=(
            None if not label_group_sizes else int(np.min(label_group_sizes))
        ),
        label_group_size_max=(
            None if not label_group_sizes else int(np.max(label_group_sizes))
        ),
    )
    if missing_reference:
        console.print(
            f"[yellow]Skipped[/yellow] {len(missing_reference)} missing reference genes"
        )
    if missing_fit:
        console.print(f"[yellow]Skipped[/yellow] {len(missing_fit)} missing fit genes")
    if dry_run:
        return 0

    fit_tasks = build_fit_tasks(
        fit_mode_resolved, label_groups, n_cells=int(adata.n_obs)
    )
    global_priors: PriorGrid | None = None
    global_scale: ScaleMetadata | None = None
    global_metadata: dict[str, Any] | None = None
    label_priors: dict[str, PriorGrid] = {}
    label_scales: dict[str, ScaleMetadata] = {}
    label_metadata: dict[str, dict[str, Any]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("fitting priors", total=len(fit_tasks))
        for task_index, task in enumerate(fit_tasks, start=1):
            sampled_indices = sample_indices(
                task.cell_indices, n_samples=n_samples, seed=sample_seed + task_index
            )
            task_reference_counts = compute_reference_counts(
                matrix, reference_positions, cell_indices=sampled_indices
            )
            valid_mask = task_reference_counts > 0
            if int(np.count_nonzero(valid_mask)) == 0:
                raise ValueError(
                    f"fit scope {task.scope_name!r} has no cells with positive reference counts"
                )
            sampled_indices = sampled_indices[valid_mask]
            task_reference_counts = task_reference_counts[valid_mask]
            batches = [
                shard_gene_names_list[start : start + gene_batch_size]
                for start in range(0, len(shard_gene_names_list), gene_batch_size)
            ]
            fitted_p_grids: list[np.ndarray] = []
            fitted_weights: list[np.ndarray] = []
            batch_summaries: list[dict[str, Any]] = []
            for batch_index, names in enumerate(batches, start=1):
                batch_counts = slice_gene_counts(
                    matrix,
                    [gene_to_idx[name] for name in names],
                    cell_indices=sampled_indices,
                )
                observation_batch = ObservationBatch(
                    gene_names=list(names),
                    counts=batch_counts,
                    reference_counts=task_reference_counts,
                )
                init_prior_weights = None
                init_source = None
                if warm_start is not None:
                    warm_priors = None
                    if (
                        task.label_value is not None
                        and task.label_value in warm_start.label_priors
                    ):
                        warm_priors = warm_start.label_priors[task.label_value]
                        init_source = f"label:{task.label_value}"
                    elif warm_start.priors is not None:
                        warm_priors = warm_start.priors
                        init_source = "global"
                    if warm_priors is not None and set(names).issubset(
                        set(warm_priors.gene_names)
                    ):
                        init_prior_weights = np.asarray(
                            warm_priors.subset(list(names)).batched().weights,
                            dtype=np.float64,
                        )
                    else:
                        init_source = None
                if fit_method_resolved == "gradient":
                    result = fit_gene_priors(
                        observation_batch,
                        S=resolved_S,
                        config=fit_config,
                        device=device,
                        init_prior_weights=init_prior_weights,
                    )
                else:
                    result = fit_gene_priors_em(
                        observation_batch,
                        S=resolved_S,
                        config=fit_config,
                        device=device,
                        init_prior_weights=init_prior_weights,
                        tol=em_tol,
                    )
                priors = result.priors.batched()
                fitted_p_grids.append(np.asarray(priors.p_grid, dtype=np.float64))
                fitted_weights.append(np.asarray(priors.weights, dtype=np.float64))
                batch_summaries.append(
                    {
                        "batch_index": batch_index,
                        "n_genes": len(names),
                        "final_loss": float(result.final_loss),
                        "best_loss": float(result.best_loss),
                        "n_iter_run": int(len(result.loss_history)),
                        "init_source": init_source,
                    }
                )
            merged_priors = PriorGrid(
                gene_names=list(shard_gene_names_list),
                p_grid=np.concatenate(fitted_p_grids, axis=0),
                weights=np.concatenate(fitted_weights, axis=0),
                S=resolved_S,
                grid_domain="rate" if fit_config.likelihood == "poisson" else "p",
                distribution=fit_config.likelihood,
            )
            task_scale = ScaleMetadata(
                S=resolved_S, mean_reference_count=float(np.mean(task_reference_counts))
            )
            task_meta = {
                "scope": task.scope_name,
                "label_key": label_key,
                "label_value": task.label_value,
                "n_cells_total": int(task.cell_indices.shape[0]),
                "n_cells_used": int(sampled_indices.shape[0]),
                "n_samples_requested": None if n_samples is None else int(n_samples),
                "batch_summaries": batch_summaries,
            }
            if task.scope_kind == "global":
                global_priors = merged_priors
                global_scale = task_scale
                global_metadata = task_meta
            else:
                assert task.label_value is not None
                label_priors[task.label_value] = merged_priors
                label_scales[task.label_value] = task_scale
                label_metadata[task.label_value] = task_meta
            progress.update(
                task_id,
                advance=1,
                description=f"fitting priors ({task_index}/{len(fit_tasks)})",
            )

    checkpoint = ModelCheckpoint(
        gene_names=checkpoint_gene_names(
            global_priors, label_priors, shard_gene_names_list
        ),
        priors=global_priors,
        scale=global_scale,
        fit_config=asdict(fit_config),
        metadata={
            "schema_version": 2,
            "source_h5ad_path": str(h5ad_path),
            "layer": layer,
            "reference_gene_names": list(reference_gene_names),
            "requested_fit_gene_names": list(fit_gene_names),
            "shard_gene_names": list(shard_gene_names_list),
            "missing_reference_genes": list(missing_reference),
            "missing_fit_genes": list(missing_fit),
            "n_cells": int(adata.n_obs),
            "gene_batch_size": int(gene_batch_size),
            "shard_rank": int(rank),
            "shard_world_size": int(world_size),
            "fit_mode": fit_mode_resolved,
            "fit_method": fit_method_resolved,
            "fit_distribution": fit_config.likelihood,
            "posterior_distribution": fit_config.likelihood,
            "grid_domain": "rate" if fit_config.likelihood == "poisson" else "p",
            "em_tol": float(em_tol),
            "nb_overdispersion": float(nb_overdispersion),
            "warm_start_checkpoint": None
            if warm_start_checkpoint is None
            else str(warm_start_checkpoint),
            "grid_max_method": grid_max_method_resolved,
            "grid_strategy": grid_strategy_resolved,
            "label_key": label_key,
            "label_values": [
                task.label_value for task in fit_tasks if task.label_value is not None
            ],
            "label_list_path": None
            if label_list_path is None
            else str(label_list_path),
            "n_samples_requested": None if n_samples is None else int(n_samples),
            "sample_seed": int(sample_seed),
            "S_source": S_source,
            "default_S_from_reference_mean": default_S,
            "created_at": datetime.now(UTC).isoformat(),
            "global_fit": global_metadata,
            "label_fits": label_metadata,
        },
        label_priors=label_priors,
        label_scales=label_scales,
    )
    save_checkpoint(checkpoint, output_path)
    print_fit_summary(
        output_path=output_path,
        elapsed_sec=perf_counter() - start_time,
        checkpoint=checkpoint,
    )
    return 0


__all__ = ["fit_priors_command"]
