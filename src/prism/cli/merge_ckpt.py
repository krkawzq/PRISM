from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from prism.model import PoolEstimate, PriorEngine, PriorEngineSetting

console = Console()


def merge_ckpt(
    ckpt_paths: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input shard checkpoint paths."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output merged checkpoint path."
    ),
) -> int:
    if len(ckpt_paths) < 2:
        raise ValueError("merge-ckpt 至少需要两个输入 checkpoint")

    resolved_paths = [path.resolve() for path in ckpt_paths]
    checkpoints = [_load_checkpoint(path) for path in resolved_paths]
    _validate_shared_metadata(checkpoints, resolved_paths)

    first = checkpoints[0]
    global_gene_names = list(first["global_gene_names"])
    setting = PriorEngineSetting(**first["setting"])
    merged_engine = PriorEngine(global_gene_names, setting=setting, device="cpu")

    assigned: set[str] = set()
    fit_history: list[dict[str, Any]] = []
    merged_ranks: list[int] = []
    source_gene_counts: list[tuple[str, int, int]] = []

    for path, checkpoint in zip(resolved_paths, checkpoints, strict=True):
        shard_engine = checkpoint["engine"]
        if not isinstance(shard_engine, PriorEngine):
            raise TypeError(f"{path} 中的 engine 不是 PriorEngine")

        shard_gene_names = list(checkpoint["gene_names"])
        if len(shard_gene_names) != len(set(shard_gene_names)):
            raise ValueError(f"{path} 的 gene_names 存在重复")

        overlap = assigned.intersection(shard_gene_names)
        if overlap:
            overlap_preview = sorted(overlap)[:5]
            raise ValueError(
                f"{path} 与已有 shard 存在重复基因，例如 {overlap_preview}"
            )

        indices = [merged_engine._gene_to_idx[name] for name in shard_gene_names]
        shard_logits = shard_engine.get_logits(shard_gene_names)
        shard_priors = shard_engine.get_priors(shard_gene_names)
        if shard_logits is None or shard_priors is None:
            raise ValueError(f"{path} 含有未拟合基因，无法合并")

        merged_engine._logits[indices] = shard_logits
        merged_engine._grid_min[indices] = shard_priors.grid_min
        merged_engine._grid_max[indices] = shard_priors.grid_max
        merged_engine._fitted[indices] = True

        assigned.update(shard_gene_names)
        merged_ranks.append(int(checkpoint["rank"]))
        source_gene_counts.append(
            (path.name, int(checkpoint["rank"]), len(shard_gene_names))
        )

        for item in checkpoint.get("fit_history", []):
            fit_history.append({"rank": int(checkpoint["rank"]), **item})

    missing = [name for name in global_gene_names if name not in assigned]
    if missing:
        raise ValueError(f"合并后仍缺少 {len(missing)} 个基因，例如 {missing[:5]}")

    merged_checkpoint = {
        "engine": merged_engine,
        "pool_estimate": _coerce_pool_estimate(first["pool_estimate"]),
        "s_hat": float(first["s_hat"]),
        "h5ad_path": first["h5ad_path"],
        "layer": first["layer"],
        "gene_start": int(first["gene_start"]),
        "gene_end": int(first["gene_end"]),
        "gene_names": global_gene_names,
        "global_gene_names": global_gene_names,
        "setting": first["setting"],
        "training_config": first["training_config"],
        "fit_history": fit_history,
        "n_cells": int(first["n_cells"]),
        "elapsed_sec": float(
            sum(float(ckpt.get("elapsed_sec", 0.0)) for ckpt in checkpoints)
        ),
        "rank": 0,
        "world_size": int(first["world_size"]),
        "source_ckpts": [str(path) for path in resolved_paths],
        "merged_from_ranks": sorted(merged_ranks),
    }

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        pickle.dump(merged_checkpoint, fh)

    _print_merge_summary(source_gene_counts, output_path, len(global_gene_names))
    return 0


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        checkpoint = pickle.load(fh)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} 不是合法的 checkpoint 字典")
    return checkpoint


def _coerce_pool_estimate(value: Any) -> PoolEstimate:
    if isinstance(value, PoolEstimate):
        return value
    if isinstance(value, dict):
        return PoolEstimate(**value)
    raise TypeError("checkpoint 中的 pool_estimate 不是合法类型")


def _validate_shared_metadata(
    checkpoints: list[dict[str, Any]],
    paths: list[Path],
) -> None:
    first = checkpoints[0]
    shared_keys = [
        "h5ad_path",
        "layer",
        "gene_start",
        "gene_end",
        "global_gene_names",
        "setting",
        "training_config",
        "world_size",
    ]

    first_pool = _coerce_pool_estimate(first["pool_estimate"])
    first_pool_dict = asdict(first_pool)
    for path, checkpoint in zip(paths[1:], checkpoints[1:], strict=True):
        for key in shared_keys:
            if checkpoint.get(key) != first.get(key):
                raise ValueError(f"{path} 的字段 {key!r} 与首个 checkpoint 不一致")

        pool_dict = asdict(_coerce_pool_estimate(checkpoint["pool_estimate"]))
        if pool_dict != first_pool_dict:
            raise ValueError(f"{path} 的 pool_estimate 与首个 checkpoint 不一致")

        if float(checkpoint["s_hat"]) != float(first["s_hat"]):
            raise ValueError(f"{path} 的 s_hat 与首个 checkpoint 不一致")

    expected_ranks = set(range(int(first["world_size"])))
    found_ranks = {int(checkpoint["rank"]) for checkpoint in checkpoints}
    if found_ranks != expected_ranks:
        raise ValueError(
            f"rank 集合不完整：期望 {sorted(expected_ranks)}，收到 {sorted(found_ranks)}"
        )


def _print_merge_summary(
    source_gene_counts: list[tuple[str, int, int]],
    output_path: Path,
    total_genes: int,
) -> None:
    table = Table(title="Merged Checkpoint")
    table.add_column("Shard")
    table.add_column("Rank", justify="right")
    table.add_column("Genes", justify="right")
    for shard_name, rank, n_genes in source_gene_counts:
        table.add_row(shard_name, str(rank), str(n_genes))
    console.print(table)
    console.print(
        f"[bold green]Merged[/bold green] {total_genes} genes -> {output_path}"
    )
