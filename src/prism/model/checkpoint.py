from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from .types import PriorFitResult, PriorGrid, ScaleMetadata


@dataclass(frozen=True, slots=True)
class ModelCheckpoint:
    gene_names: list[str]
    priors: PriorGrid
    scale: ScaleMetadata
    fit_config: dict[str, Any]
    metadata: dict[str, Any]


def checkpoint_from_fit_result(
    result: PriorFitResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> ModelCheckpoint:
    return ModelCheckpoint(
        gene_names=list(result.gene_names),
        priors=result.priors,
        scale=result.scale,
        fit_config=dict(result.config),
        metadata={} if metadata is None else dict(metadata),
    )


def save_checkpoint(checkpoint: ModelCheckpoint, path: str | Path) -> None:
    payload = {
        "gene_names": checkpoint.gene_names,
        "priors": {
            "gene_names": checkpoint.priors.gene_names,
            "p_grid": np.asarray(checkpoint.priors.p_grid),
            "weights": np.asarray(checkpoint.priors.weights),
            "S": float(checkpoint.priors.S),
            "grid_domain": checkpoint.priors.grid_domain,
        },
        "scale": asdict(checkpoint.scale),
        "fit_config": dict(checkpoint.fit_config),
        "metadata": dict(checkpoint.metadata),
    }
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("wb") as handle:
        pickle.dump(payload, handle)


def load_checkpoint(path: str | Path) -> ModelCheckpoint:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    priors_payload = payload["priors"]
    priors = PriorGrid(
        gene_names=list(priors_payload["gene_names"]),
        p_grid=np.asarray(priors_payload["p_grid"], dtype=np.float64),
        weights=np.asarray(priors_payload["weights"], dtype=np.float64),
        S=float(priors_payload["S"]),
        grid_domain=str(priors_payload.get("grid_domain", "p")),
    )
    return ModelCheckpoint(
        gene_names=list(payload["gene_names"]),
        priors=priors,
        scale=ScaleMetadata(**payload["scale"]),
        fit_config=dict(payload.get("fit_config", {})),
        metadata=dict(payload.get("metadata", {})),
    )
