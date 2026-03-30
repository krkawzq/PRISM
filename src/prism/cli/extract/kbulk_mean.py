from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .common import (
    compute_reference_counts,
    console,
    print_extract_plan,
    read_gene_list,
    resolve_class_groups,
    resolve_dtype,
    select_matrix,
    slice_gene_counts,
)
from .kbulk import (
    _approx_comb_count,
    _build_kbulk_result_adata,
    _estimate_total_realized,
    _format_comb_count,
    _iter_chunks,
    _resolve_target_samples,
    _sample_unique_combinations,
    _serialize_per_class_plan,
    _write_h5ad_atomic,
)


def _entry_int(entry: dict[str, Any], key: str) -> int:
    return int(entry[key])


def _entry_float(entry: dict[str, Any], key: str) -> float:
    return float(entry[key])


def _entry_str(entry: dict[str, Any], key: str) -> str:
    return str(entry[key])


def _resolve_selected_genes(
    *,
    genes_path: Path | None,
    dataset_gene_names: list[str],
) -> list[str]:
    requested = dataset_gene_names if genes_path is None else read_gene_list(genes_path)
    dataset_set = set(dataset_gene_names)
    selected = [gene for gene in requested if gene in dataset_set]
    if not selected:
        raise ValueError("no selected genes overlap with the dataset")
    return list(dict.fromkeys(selected))


def extract_kbulk_mean_command(
    h5ad_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad file."
    ),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output h5ad path."),
    class_key: str = typer.Option(
        ..., help="Obs column defining classes for kBulk-mean construction."
    ),
    genes_path: Path | None = typer.Option(
        None,
        "--genes",
        exists=True,
        dir_okay=False,
        help="Optional text file restricting genes.",
    ),
    reference_genes_path: Path | None = typer.Option(
        None,
        "--reference-genes",
        exists=True,
        dir_okay=False,
        help="Optional text file used for balance/reference statistics.",
    ),
    layer: str | None = typer.Option(None, help="Input AnnData layer. Defaults to X."),
    normalize_total: float | None = typer.Option(
        None,
        help="Optional per-cell normalization target applied before kBulk averaging.",
    ),
    log1p: bool = typer.Option(
        False,
        help="Apply log1p after per-cell normalization and before kBulk averaging.",
    ),
    k: int = typer.Option(..., min=1, help="Number of cells per kBulk sample."),
    n_samples: int = typer.Option(
        ..., min=1, help="Target number of samples per class before balance scaling."
    ),
    balance: bool = typer.Option(
        False, help="Scale per-class sample count by mean reference size."
    ),
    sample_seed: int = typer.Option(
        0, min=0, help="Random seed for kBulk combination sampling."
    ),
    sample_batch_size: int = typer.Option(
        1024, min=1, help="Number of sampled combinations to aggregate per batch."
    ),
    dtype: str = typer.Option("float32", help="Output dtype: float32 or float64."),
    dry_run: bool = typer.Option(
        False, help="Show the execution plan without writing output."
    ),
) -> int:
    start_time = perf_counter()
    h5ad_path = h5ad_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    genes_path = None if genes_path is None else genes_path.expanduser().resolve()
    reference_genes_path = (
        None
        if reference_genes_path is None
        else reference_genes_path.expanduser().resolve()
    )

    output_dtype = resolve_dtype(dtype)
    console.print(f"[bold cyan]Reading[/bold cyan] {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    matrix = select_matrix(adata, layer)
    dataset_gene_names = [str(name) for name in adata.var_names.tolist()]
    gene_to_idx = {name: idx for idx, name in enumerate(dataset_gene_names)}
    selected_genes = _resolve_selected_genes(
        genes_path=genes_path,
        dataset_gene_names=dataset_gene_names,
    )
    reference_gene_names = (
        list(dataset_gene_names)
        if reference_genes_path is None
        else read_gene_list(reference_genes_path)
    )
    resolved_reference_gene_names = [
        name for name in reference_gene_names if name in gene_to_idx
    ]
    reference_positions = [gene_to_idx[name] for name in resolved_reference_gene_names]
    if not reference_positions:
        raise ValueError("reference genes do not overlap with the input dataset")
    reference_counts = np.ascontiguousarray(
        compute_reference_counts(matrix, reference_positions), dtype=np.float32
    )
    class_groups = resolve_class_groups(adata, class_key)

    class_mean_reference = {
        label: float(np.mean(reference_counts[indices]))
        for label, indices in class_groups.items()
    }
    avg_mean_reference = float(np.mean(list(class_mean_reference.values())))
    per_class_plan: list[dict[str, Any]] = []
    total_kbulk = 0
    skipped_classes = 0
    for label, indices in class_groups.items():
        n_cells = int(indices.shape[0])
        n_combos_exact = 0 if k > n_cells else int(math.comb(n_cells, k))
        n_combos_approx = _approx_comb_count(n_cells, k)
        target = _resolve_target_samples(
            n_samples=n_samples,
            balance=balance,
            class_mean_reference=class_mean_reference[label],
            avg_mean_reference=avg_mean_reference,
        )
        realized = min(target, n_combos_exact)
        if n_combos_exact == 0:
            skipped_classes += 1
        total_kbulk += realized
        per_class_plan.append(
            {
                "label": label,
                "n_cells": n_cells,
                "n_combinations": n_combos_approx,
                "mean_reference_count": class_mean_reference[label],
                "target_samples": target,
                "realized_samples": realized,
            }
        )

    print_extract_plan(
        h5ad_path=h5ad_path,
        layer=layer,
        class_key=class_key,
        method="kbulk-mean",
        k=k,
        n_cells=int(adata.n_obs),
        n_classes=len(class_groups),
        skipped_classes=skipped_classes,
        n_selected_genes=len(selected_genes),
        n_reference_genes=len(reference_positions),
        reference_genes_source="dataset" if reference_genes_path is None else "user",
        normalize_total=normalize_total,
        log1p=log1p,
        n_samples=n_samples,
        balance=balance,
        sample_batch_size=sample_batch_size,
        avg_mean_reference=avg_mean_reference,
        total_planned_kbulk=total_kbulk,
        output_path=output_path,
    )
    if per_class_plan:
        preview_lines = [
            f"{_entry_str(entry, 'label')}: cells={_entry_int(entry, 'n_cells')}, combos={_format_comb_count(_entry_float(entry, 'n_combinations'))}, target={_entry_int(entry, 'target_samples')}, realized={_entry_int(entry, 'realized_samples')}"
            for entry in per_class_plan[: min(10, len(per_class_plan))]
        ]
        console.print("\n".join(preview_lines))
    if dry_run:
        return 0

    target_positions = [gene_to_idx[name] for name in selected_genes]
    gene_counts = np.ascontiguousarray(
        slice_gene_counts(matrix, target_positions), dtype=np.float32
    )
    if normalize_total is not None:
        if normalize_total <= 0:
            raise ValueError("--normalize-total must be positive")
        totals = gene_counts.sum(axis=1, keepdims=True).clip(1e-9)
        gene_counts = gene_counts / totals * np.float32(normalize_total)
    if log1p:
        gene_counts = np.log1p(gene_counts, dtype=np.float32)
    total_realized = _estimate_total_realized(per_class_plan)
    n_genes = len(selected_genes)
    mean_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    sum_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    zero_rows = np.empty((total_realized, n_genes), dtype=output_dtype)
    obs_class: list[str] = [""] * total_realized
    obs_kbulk_index = np.empty(total_realized, dtype=np.int64)
    obs_class_sample_index = np.empty(total_realized, dtype=np.int64)
    obs_k = np.full(total_realized, int(k), dtype=np.int64)
    obs_source_n_cells = np.empty(total_realized, dtype=np.int64)
    obs_n_combinations_total = np.empty(total_realized, dtype=np.float64)
    obs_sampling_mode: list[str] = [""] * total_realized
    obs_sum_reference_count = np.empty(total_realized, dtype=np.float32)
    obs_mean_reference_count = np.empty(total_realized, dtype=np.float32)

    rng = np.random.default_rng(sample_seed)
    realized_classes = [
        entry for entry in per_class_plan if _entry_int(entry, "realized_samples") > 0
    ]
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            "constructing kbulk-mean", total=len(realized_classes)
        )
        global_sample_index = 0
        for class_index, entry in enumerate(realized_classes, start=1):
            label = _entry_str(entry, "label")
            indices = class_groups[label]
            target = _entry_int(entry, "target_samples")
            n_combinations_total = _entry_float(entry, "n_combinations")
            mean_reference = _entry_float(entry, "mean_reference_count")
            combos = _sample_unique_combinations(
                indices, k=k, n_samples=target, rng=rng
            )
            indices_i64 = np.asarray(indices, dtype=np.int64)
            for chunk_index, combo_chunk in enumerate(
                _iter_chunks(combos, sample_batch_size), start=1
            ):
                combo_matrix = np.asarray(combo_chunk, dtype=np.int64)
                flat_indices = combo_matrix.reshape(-1)
                chunk_counts = gene_counts[flat_indices].reshape(
                    combo_matrix.shape[0], combo_matrix.shape[1], n_genes
                )
                chunk_reference = reference_counts[flat_indices].reshape(
                    combo_matrix.shape
                )
                sum_counts = chunk_counts.sum(axis=1, dtype=np.float32)
                mean_counts = sum_counts / np.float32(k)
                zero_frac = np.mean(chunk_counts == 0, axis=1, dtype=np.float32)
                sum_reference = chunk_reference.sum(axis=1, dtype=np.float32)
                chunk_size_actual = combo_matrix.shape[0]
                start = global_sample_index
                stop = global_sample_index + chunk_size_actual
                mean_rows[start:stop] = np.asarray(mean_counts, dtype=output_dtype)
                sum_rows[start:stop] = np.asarray(sum_counts, dtype=output_dtype)
                zero_rows[start:stop] = np.asarray(zero_frac, dtype=output_dtype)
                obs_class[start:stop] = [label] * chunk_size_actual
                obs_kbulk_index[start:stop] = np.arange(
                    start + 1, stop + 1, dtype=np.int64
                )
                obs_class_sample_index[start:stop] = np.arange(
                    (chunk_index - 1) * sample_batch_size + 1,
                    (chunk_index - 1) * sample_batch_size + 1 + chunk_size_actual,
                    dtype=np.int64,
                )
                obs_source_n_cells[start:stop] = int(indices_i64.shape[0])
                obs_n_combinations_total[start:stop] = np.float64(n_combinations_total)
                obs_sampling_mode[start:stop] = [
                    "all" if n_combinations_total <= float(target) else "sampled"
                ] * chunk_size_actual
                obs_sum_reference_count[start:stop] = sum_reference
                obs_mean_reference_count[start:stop] = np.float32(mean_reference)
                global_sample_index = stop
            progress.update(
                task_id,
                advance=1,
                description=f"constructing kbulk-mean ({class_index}/{len(realized_classes)})",
            )

    rows = [
        {
            class_key: obs_class[idx],
            "kbulk_index": int(obs_kbulk_index[idx]),
            "class_sample_index": int(obs_class_sample_index[idx]),
            "k": int(obs_k[idx]),
            "source_n_cells": int(obs_source_n_cells[idx]),
            "n_combinations_total": float(obs_n_combinations_total[idx]),
            "sampling_mode": obs_sampling_mode[idx],
            "sum_reference_count": float(obs_sum_reference_count[idx]),
            "mean_reference_count": float(obs_mean_reference_count[idx]),
        }
        for idx in range(total_realized)
    ]

    output = _build_kbulk_result_adata(
        rows=rows,
        gene_names=selected_genes,
        map_mu_rows=mean_rows,
        map_p_rows=np.zeros_like(mean_rows, dtype=output_dtype),
        posterior_entropy_rows=np.zeros_like(mean_rows, dtype=output_dtype),
        prior_entropy_rows=np.zeros_like(mean_rows, dtype=output_dtype),
        mutual_information_rows=np.zeros_like(mean_rows, dtype=output_dtype),
        raw_count_rows=sum_rows,
        dtype=output_dtype,
        metadata={
            "source_h5ad_path": str(h5ad_path),
            "method": "kbulk-mean",
            "class_key": class_key,
            "normalize_total": None
            if normalize_total is None
            else float(normalize_total),
            "log1p": bool(log1p),
            "k": int(k),
            "requested_n_samples": int(n_samples),
            "balance": bool(balance),
            "sample_seed": int(sample_seed),
            "selected_genes": list(selected_genes),
            "reference_gene_names": list(resolved_reference_gene_names),
            "reference_genes_source": "dataset"
            if reference_genes_path is None
            else "user",
            "per_class_plan": _serialize_per_class_plan(per_class_plan),
        },
    )
    output.layers["mean_counts"] = np.asarray(mean_rows, dtype=output_dtype)
    output.layers["sum_counts"] = np.asarray(sum_rows, dtype=output_dtype)
    output.layers["zero_fraction"] = np.asarray(zero_rows, dtype=output_dtype)
    _write_h5ad_atomic(output, output_path)

    for entry in per_class_plan:
        console.print(
            f"{_entry_str(entry, 'label')}: cells={_entry_int(entry, 'n_cells')}, combos={_format_comb_count(_entry_float(entry, 'n_combinations'))}, target={_entry_int(entry, 'target_samples')}, realized={_entry_int(entry, 'realized_samples')}"
        )
    console.print(f"[bold green]Saved[/bold green] {output_path}")
    console.print(
        f"[bold green]Elapsed[/bold green] {perf_counter() - start_time:.2f}s"
    )
    return 0


__all__ = ["extract_kbulk_mean_command"]
