from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prism.model import PoolEstimate, PoolFitReport, PriorEngine


@dataclass(frozen=True, slots=True)
class CheckpointBundle:
    checkpoint: dict[str, Any]
    engine: PriorEngine
    s_hat: float
    pool_report: PoolFitReport | None
    r_hint: float | None


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        checkpoint = pickle.load(fh)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} is not a checkpoint dictionary")
    return checkpoint


def load_checkpoint_bundle(path: Path, gene_names: np.ndarray) -> CheckpointBundle:
    checkpoint = load_checkpoint(path)
    print(f"[prism-server] checkpoint bundle parse path={path}", flush=True)
    engine = checkpoint.get("engine")
    if not isinstance(engine, PriorEngine):
        raise TypeError("checkpoint does not contain a valid PriorEngine")
    validate_checkpoint_against_dataset(checkpoint, gene_names)

    if checkpoint.get("s_hat") is None:
        raise KeyError("checkpoint does not contain s_hat")
    s_hat = float(checkpoint["s_hat"])
    if s_hat <= 0:
        raise ValueError(f"checkpoint s_hat must be > 0, got {s_hat}")

    pool_report = _coerce_pool_report(checkpoint.get("pool_report"))
    r_hint = _infer_r_hint(checkpoint, s_hat)
    print(
        f"[prism-server] checkpoint fields engine_genes={len(engine.gene_names)} has_pool_report={pool_report is not None} r_hint={r_hint if r_hint is not None else '-'}",
        flush=True,
    )
    return CheckpointBundle(
        checkpoint=checkpoint,
        engine=engine,
        s_hat=s_hat,
        pool_report=pool_report,
        r_hint=r_hint,
    )


def validate_checkpoint_against_dataset(
    checkpoint: dict[str, Any],
    gene_names: np.ndarray,
) -> None:
    ckpt_gene_names = checkpoint.get("gene_names")
    if not isinstance(ckpt_gene_names, list):
        raise TypeError("checkpoint does not contain gene_names")

    gene_set = set(map(str, gene_names.tolist()))
    missing = [name for name in ckpt_gene_names if name not in gene_set]
    if missing:
        raise ValueError(
            f"checkpoint contains genes missing from the dataset, e.g. {missing[:5]}"
        )


def _coerce_pool_report(value: Any) -> PoolFitReport | None:
    if value is None:
        return None
    if isinstance(value, PoolFitReport):
        return value
    if isinstance(value, dict):
        return PoolFitReport(**value)
    return None


def _infer_r_hint(checkpoint: dict[str, Any], s_hat: float) -> float | None:
    if s_hat <= 0:
        return None
    pool_estimate = checkpoint.get("pool_estimate")
    if isinstance(pool_estimate, dict):
        try:
            pool_estimate = PoolEstimate(**pool_estimate)
        except TypeError:
            return None
    if isinstance(pool_estimate, PoolEstimate):
        r_hint = float(pool_estimate.point_eta) / s_hat
        return r_hint if r_hint > 0 else None
    return None
