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

from prism.cli.common import (
    normalize_choice,
    resolve_bool,
    resolve_float,
    resolve_int,
    resolve_optional_float,
    resolve_optional_int,
    resolve_optional_path,
    resolve_optional_str,
    resolve_str,
    unwrap_typer_value,
)
from prism.model import (
    ModelCheckpoint,
    ObservationBatch,
    PriorFitConfig,
    PriorGrid,
    ScaleMetadata,
    fit_gene_priors,
    load_checkpoint,
    make_distribution_grid,
    save_checkpoint,
)

from .common import (
    build_fit_tasks,
    build_gene_batches,
    checkpoint_gene_names,
    compute_reference_counts,
    console,
    parse_shard,
    print_fit_plan,
    print_fit_summary,
    resolve_fit_gene_list,
    resolve_fit_mode,
    resolve_gene_list,
    resolve_label_groups,
    resolve_requested_labels,
    resolve_scale,
    sample_indices,
    select_matrix,
    shard_gene_names,
    slice_gene_counts,
)


def _normalize_likelihood(value: str) -> str:
    return normalize_choice(
        value,
        supported=("binomial", "negative_binomial", "poisson"),
        option_name="--likelihood",
    )


def _normalize_support_max_from(value: str) -> str:
    return normalize_choice(
        value,
        supported=("observed_max", "quantile"),
        option_name="--support-max-from",
    )


def _normalize_support_spacing(value: str) -> str:
    return normalize_choice(
        value,
        supported=("linear", "sqrt"),
        option_name="--support-spacing",
    )


def _normalize_torch_dtype(value: str) -> str:
    return normalize_choice(
        value,
        supported=("float32", "float64"),
        option_name="--torch-dtype",
    )


def _resolve_optional_list(value: list[str] | None | object) -> list[str] | None:
    resolved = unwrap_typer_value(value)
    if resolved is None:
        return None
    return list(cast(list[str], resolved))


def _build_fit_config(
    *,
    n_support_points: int,
    max_em_iterations: int | None,
    convergence_tolerance: float,
    cell_chunk_size: int,
    support_max_from: str,
    support_spacing: str,
    support_scale: float,
    use_adaptive_support: bool,
    adaptive_support_scale: float,
    adaptive_support_quantile_hi: float,
    likelihood: str,
    nb_overdispersion: float,
) -> PriorFitConfig:
    return PriorFitConfig(
        n_support_points=n_support_points,
        max_em_iterations=max_em_iterations,
        convergence_tolerance=convergence_tolerance,
        cell_chunk_size=cell_chunk_size,
        support_max_from=cast(Any, _normalize_support_max_from(support_max_from)),
        support_spacing=cast(Any, _normalize_support_spacing(support_spacing)),
        support_scale=support_scale,
        use_adaptive_support=use_adaptive_support,
        adaptive_support_scale=adaptive_support_scale,
        adaptive_support_quantile_hi=adaptive_support_quantile_hi,
        likelihood=cast(Any, _normalize_likelihood(likelihood)),
        nb_overdispersion=nb_overdispersion,
    )


def _select_warm_start_prior(
    checkpoint: ModelCheckpoint | None,
    *,
    label_value: str | None,
    gene_names: list[str],
    expected_distribution: str,
) -> np.ndarray | None:
    if checkpoint is None:
        return None
    warm_prior: PriorGrid | None = None
    if label_value is not None and label_value in checkpoint.label_priors:
        warm_prior = checkpoint.label_priors[label_value]
    elif checkpoint.prior is not None:
        warm_prior = checkpoint.prior
    if warm_prior is None or not set(gene_names).issubset(set(warm_prior.gene_names)):
        return None
    if warm_prior.distribution_name != expected_distribution:
        return None
    return np.asarray(
        warm_prior.select_genes(gene_names).as_gene_specific().prior_probabilities,
        dtype=np.float64,
    )


def _merge_priors(
    *, gene_names: list[str], priors: list[PriorGrid], scale: float, likelihood: str
) -> PriorGrid:
    if not priors:
        raise ValueError("cannot merge an empty prior list")
    support = np.concatenate(
        [np.asarray(prior.support, dtype=np.float64) for prior in priors], axis=0
    )
    prior_probabilities = np.concatenate(
        [np.asarray(prior.prior_probabilities, dtype=np.float64) for prior in priors],
        axis=0,
    )
    return PriorGrid(
        gene_names=list(gene_names),
        distribution=make_distribution_grid(
            cast(Any, likelihood), support=support, probabilities=prior_probabilities
        ),
        scale=float(scale),
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
        help="Optional text file with reference gene names used to compute reference counts.",
    ),
    fit_genes_path: Path | None = typer.Option(
        None,
        "--fit-genes",
        exists=True,
        dir_okay=False,
        help="Optional text file with genes to fit.",
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
        help="Optional text file with label values.",
    ),
    fit_mode: str = typer.Option(
        "global", help="Fit scope: global, by-label, or both."
    ),
    n_samples: int | None = typer.Option(
        None, min=1, help="Optional random cell subsample size per fit scope."
    ),
    sample_seed: int = typer.Option(0, min=0, help="Random seed for cell subsampling."),
    scale: float | None = typer.Option(
        None,
        "--scale",
        min=1e-12,
        help="Model scale. Defaults to mean reference count.",
    ),
    device: str = typer.Option("cpu", help="Torch device, e.g. cpu or cuda."),
    gene_batch_size: int = typer.Option(
        64, min=1, help="Number of genes to fit per batch."
    ),
    shard: str = typer.Option(
        "0/1", help="Shard specification as rank/world, e.g. 0/4."
    ),
    n_support_points: int = typer.Option(
        512, min=2, help="Number of support points per gene."
    ),
    max_em_iterations: int | None = typer.Option(
        100, min=1, help="Maximum EM iterations."
    ),
    convergence_tolerance: float = typer.Option(
        1e-6, min=0.0, help="EM convergence tolerance."
    ),
    cell_chunk_size: int = typer.Option(
        512, min=1, help="Number of cells per likelihood chunk."
    ),
    torch_dtype: str = typer.Option("float64", help="Torch dtype: float64 or float32."),
    support_max_from: str = typer.Option(
        "observed_max", help="Support max rule: observed_max or quantile."
    ),
    support_spacing: str = typer.Option(
        "linear", help="Support spacing: linear or sqrt."
    ),
    support_scale: float = typer.Option(
        1.5,
        min=1.0,
        help="Expansion factor applied to the first-pass support max.",
    ),
    use_adaptive_support: bool = typer.Option(
        False, help="Enable two-phase adaptive support refinement."
    ),
    adaptive_support_scale: float = typer.Option(
        1.5,
        min=1.0,
        help="Expansion factor applied to the adaptive support range [0, q_hi].",
    ),
    adaptive_support_quantile_hi: float = typer.Option(
        0.99,
        min=0.000001,
        max=1.0,
        help="Upper posterior quantile used for adaptive refinement.",
    ),
    likelihood: str = typer.Option(
        "binomial", help="Fit distribution: binomial, negative_binomial, or poisson."
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
        help="Optional checkpoint used to initialize prior probabilities when matching genes are available.",
    ),
    compile_model: bool = typer.Option(
        True, help="Compile fit kernels with torch.compile when possible."
    ),
    dry_run: bool = typer.Option(
        False, help="Show the execution plan without fitting."
    ),
) -> int:
    start_time = perf_counter()
    h5ad_path = h5ad_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    reference_genes_path = resolve_optional_path(reference_genes_path)
    fit_genes_path = resolve_optional_path(fit_genes_path)
    label_list_path = resolve_optional_path(label_list_path)
    warm_start_checkpoint = resolve_optional_path(warm_start_checkpoint)
    label_values = _resolve_optional_list(label_values)
    n_samples = resolve_optional_int(n_samples)
    scale = resolve_optional_float(scale)
    max_em_iterations = resolve_optional_int(max_em_iterations)
    layer = resolve_optional_str(layer)
    label_key = resolve_optional_str(label_key)
    fit_mode = resolve_str(fit_mode)
    sample_seed = resolve_int(sample_seed)
    device = resolve_str(device)
    gene_batch_size = resolve_int(gene_batch_size)
    shard = resolve_str(shard)
    n_support_points = resolve_int(n_support_points)
    convergence_tolerance = resolve_float(convergence_tolerance)
    cell_chunk_size = resolve_int(cell_chunk_size)
    torch_dtype = _normalize_torch_dtype(resolve_str(torch_dtype))
    support_max_from = resolve_str(support_max_from)
    support_spacing = resolve_str(support_spacing)
    support_scale = resolve_float(support_scale)
    use_adaptive_support = resolve_bool(use_adaptive_support)
    adaptive_support_scale = resolve_float(adaptive_support_scale)
    adaptive_support_quantile_hi = resolve_float(adaptive_support_quantile_hi)
    likelihood = resolve_str(likelihood)
    nb_overdispersion = resolve_float(nb_overdispersion)
    compile_model = resolve_bool(compile_model)
    dry_run = resolve_bool(dry_run)

    rank, world_size = parse_shard(shard)
    fit_mode_resolved = resolve_fit_mode(fit_mode)
    fit_config = _build_fit_config(
        n_support_points=n_support_points,
        max_em_iterations=max_em_iterations,
        convergence_tolerance=convergence_tolerance,
        cell_chunk_size=cell_chunk_size,
        support_max_from=support_max_from,
        support_spacing=support_spacing,
        support_scale=support_scale,
        use_adaptive_support=use_adaptive_support,
        adaptive_support_scale=adaptive_support_scale,
        adaptive_support_quantile_hi=adaptive_support_quantile_hi,
        likelihood=likelihood,
        nb_overdispersion=nb_overdispersion,
    )
    warm_start = (
        None
        if warm_start_checkpoint is None
        else load_checkpoint(warm_start_checkpoint)
    )

    console.print(f"[bold cyan]Reading[/bold cyan] {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    matrix = select_matrix(adata, layer)
    dataset_gene_names = [str(name) for name in adata.var_names.tolist()]
    gene_to_idx = {name: idx for idx, name in enumerate(dataset_gene_names)}

    if reference_genes_path is None:
        reference_gene_names = list(dataset_gene_names)
        missing_reference: list[str] = []
    else:
        reference_gene_names, missing_reference = resolve_gene_list(
            reference_genes_path, gene_to_idx
        )
        if not reference_gene_names:
            raise ValueError("reference gene list has no overlap with the dataset")

    fit_gene_names, missing_fit = resolve_fit_gene_list(
        fit_genes_path, dataset_gene_names, gene_to_idx
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
    scale_resolution = resolve_scale(reference_counts, scale=scale)
    resolved_scale = scale_resolution.scale

    resolved_label_values = resolve_requested_labels(label_values, label_list_path)
    label_groups = resolve_label_groups(
        adata=adata,
        label_key=label_key,
        label_values=resolved_label_values,
        fit_mode=fit_mode_resolved,
    )
    fit_tasks = build_fit_tasks(
        fit_mode_resolved, label_groups, n_cells=int(adata.n_obs)
    )
    gene_batches = build_gene_batches(
        shard_gene_names_list, gene_to_idx, batch_size=gene_batch_size
    )

    label_group_sizes = [int(indices.shape[0]) for _, indices in label_groups]
    print_fit_plan(
        h5ad_path=h5ad_path,
        layer=layer,
        n_cells=int(adata.n_obs),
        n_dataset_genes=int(adata.n_vars),
        n_reference_genes=len(reference_gene_names),
        n_positive_reference_cells=scale_resolution.n_positive_reference_cells,
        n_fit_genes=len(fit_gene_names),
        n_shard_genes=len(shard_gene_names_list),
        n_fit_tasks=len(fit_tasks),
        n_gene_batches=len(gene_batches),
        fit_mode=fit_mode_resolved,
        label_key=label_key,
        n_label_groups=len(label_groups),
        n_requested_label_values=None
        if resolved_label_values is None
        else len(resolved_label_values),
        n_samples=n_samples,
        sample_seed=sample_seed,
        scale=resolved_scale,
        scale_source=scale_resolution.scale_source,
        default_scale=scale_resolution.default_scale,
        device=device,
        n_support_points=n_support_points,
        max_em_iterations=max_em_iterations,
        convergence_tolerance=convergence_tolerance,
        cell_chunk_size=cell_chunk_size,
        support_max_from=support_max_from,
        support_spacing=support_spacing,
        support_scale=support_scale,
        use_adaptive_support=use_adaptive_support,
        adaptive_support_scale=adaptive_support_scale,
        adaptive_support_quantile_hi=adaptive_support_quantile_hi,
        likelihood=fit_config.likelihood,
        nb_overdispersion=nb_overdispersion,
        warm_start_checkpoint=warm_start_checkpoint,
        compile_model=compile_model,
        gene_batch_size=gene_batch_size,
        shard=f"{rank}/{world_size}",
        output_path=output_path,
        label_group_size_mean=None
        if not label_group_sizes
        else round(float(np.mean(label_group_sizes)), 2),
        label_group_size_min=None
        if not label_group_sizes
        else int(np.min(label_group_sizes)),
        label_group_size_max=None
        if not label_group_sizes
        else int(np.max(label_group_sizes)),
    )
    if missing_reference:
        console.print(
            f"[yellow]Skipped[/yellow] {len(missing_reference)} missing reference genes"
        )
    if missing_fit:
        console.print(f"[yellow]Skipped[/yellow] {len(missing_fit)} missing fit genes")
    if dry_run:
        return 0

    global_prior: PriorGrid | None = None
    global_scale_metadata: ScaleMetadata | None = None
    global_metadata: dict[str, Any] | None = None
    label_priors: dict[str, PriorGrid] = {}
    label_scale_metadata: dict[str, ScaleMetadata] = {}
    label_metadata: dict[str, dict[str, Any]] = {}

    total_batches = len(fit_tasks) * len(gene_batches)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("fitting priors", total=total_batches)
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

            fitted_priors: list[PriorGrid] = []
            batch_summaries: list[dict[str, Any]] = []
            for batch_index, batch in enumerate(gene_batches, start=1):
                progress.update(
                    task_id,
                    description=(
                        "fitting priors "
                        f"[{task.scope_name} | group {task_index}/{len(fit_tasks)} | "
                        f"batch {batch_index}/{len(gene_batches)}]"
                    ),
                )
                batch_counts = slice_gene_counts(
                    matrix,
                    batch.gene_positions,
                    cell_indices=sampled_indices,
                )
                observation_batch = ObservationBatch(
                    gene_names=list(batch.gene_names),
                    counts=batch_counts,
                    reference_counts=task_reference_counts,
                )
                init_probabilities = _select_warm_start_prior(
                    warm_start,
                    label_value=task.label_value,
                    gene_names=list(batch.gene_names),
                    expected_distribution=fit_config.likelihood,
                )
                result = fit_gene_priors(
                    observation_batch,
                    scale=resolved_scale,
                    config=fit_config,
                    device=device,
                    torch_dtype=torch_dtype,
                    initial_probabilities=init_probabilities,
                    compile_model=compile_model,
                )
                fitted_priors.append(result.prior.as_gene_specific())
                batch_summaries.append(
                    {
                        "batch_index": batch_index,
                        "n_genes": len(batch.gene_names),
                        "final_objective": float(result.final_objective),
                        "n_iterations": int(len(result.objective_history)),
                        "warm_start": init_probabilities is not None,
                    }
                )
                progress.update(task_id, advance=1)

            merged_prior = _merge_priors(
                gene_names=list(shard_gene_names_list),
                priors=fitted_priors,
                scale=resolved_scale,
                likelihood=fit_config.likelihood,
            )
            task_scale_metadata = ScaleMetadata(
                scale=resolved_scale,
                mean_reference_count=float(np.mean(task_reference_counts)),
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
                global_prior = merged_prior
                global_scale_metadata = task_scale_metadata
                global_metadata = task_meta
            else:
                assert task.label_value is not None
                label_priors[task.label_value] = merged_prior
                label_scale_metadata[task.label_value] = task_scale_metadata
                label_metadata[task.label_value] = task_meta
            progress.update(
                task_id,
                description=(
                    "fitting priors "
                    f"[{task.scope_name} complete | group {task_index}/{len(fit_tasks)}]"
                ),
            )

    checkpoint = ModelCheckpoint(
        gene_names=checkpoint_gene_names(
            global_prior, label_priors, shard_gene_names_list
        ),
        prior=global_prior,
        scale_metadata=global_scale_metadata,
        fit_config={
            **asdict(fit_config),
            "torch_dtype": torch_dtype,
            "compile_model": bool(compile_model),
        },
        metadata={
            "source_h5ad_path": str(h5ad_path),
            "layer": layer,
            "reference_gene_names": list(reference_gene_names),
            "requested_fit_gene_names": list(fit_gene_names),
            "shard_gene_names": list(shard_gene_names_list),
            "missing_reference_genes": list(missing_reference),
            "missing_fit_genes": list(missing_fit),
            "n_cells": int(adata.n_obs),
            "n_positive_reference_cells": int(
                scale_resolution.n_positive_reference_cells
            ),
            "gene_batch_size": int(gene_batch_size),
            "shard_rank": int(rank),
            "shard_world_size": int(world_size),
            "fit_mode": fit_mode_resolved,
            "fit_distribution": fit_config.likelihood,
            "posterior_distribution": fit_config.likelihood,
            "support_domain": "rate"
            if fit_config.likelihood == "poisson"
            else "probability",
            "nb_overdispersion": float(nb_overdispersion),
            "warm_start_checkpoint": None
            if warm_start_checkpoint is None
            else str(warm_start_checkpoint),
            "label_key": label_key,
            "label_values": [
                task.label_value for task in fit_tasks if task.label_value is not None
            ],
            "label_list_path": None
            if label_list_path is None
            else str(label_list_path),
            "n_samples_requested": None if n_samples is None else int(n_samples),
            "sample_seed": int(sample_seed),
            "scale_source": scale_resolution.scale_source,
            "default_scale_from_reference_mean": scale_resolution.default_scale,
            "created_at": datetime.now(UTC).isoformat(),
            "global_fit": global_metadata,
            "label_fits": label_metadata,
        },
        label_priors=label_priors,
        label_scale_metadata=label_scale_metadata,
    )
    save_checkpoint(checkpoint, output_path)
    print_fit_summary(
        output_path=output_path,
        elapsed_sec=perf_counter() - start_time,
        checkpoint=checkpoint,
    )
    return 0


__all__ = ["fit_priors_command", "_select_warm_start_prior"]
