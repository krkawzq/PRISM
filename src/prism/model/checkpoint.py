from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import pickle
from typing import Any, cast

import numpy as np

from .types import PriorFitResult, PriorGrid, ScaleMetadata


@dataclass(frozen=True, slots=True)
class ModelCheckpoint:
    gene_names: list[str]
    priors: PriorGrid | None
    scale: ScaleMetadata | None
    fit_config: dict[str, Any]
    metadata: dict[str, Any]
    label_priors: dict[str, PriorGrid] = field(default_factory=dict)
    label_scales: dict[str, ScaleMetadata] = field(default_factory=dict)


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
        label_priors={},
        label_scales={},
    )


def _serialize_prior_grid(priors: PriorGrid) -> dict[str, Any]:
    return {
        "gene_names": priors.gene_names,
        "p_grid": np.asarray(priors.p_grid),
        "weights": np.asarray(priors.weights),
        "S": float(priors.S),
        "grid_domain": priors.grid_domain,
    }


def _deserialize_prior_grid(payload: dict[str, Any]) -> PriorGrid:
    return PriorGrid(
        gene_names=list(payload["gene_names"]),
        p_grid=np.asarray(payload["p_grid"], dtype=np.float64),
        weights=np.asarray(payload["weights"], dtype=np.float64),
        S=float(payload["S"]),
        grid_domain=cast(Any, str(payload.get("grid_domain", "p"))),
    )


def save_checkpoint(checkpoint: ModelCheckpoint, path: str | Path) -> None:
    payload = {
        "schema_version": 2,
        "gene_names": checkpoint.gene_names,
        "priors": None
        if checkpoint.priors is None
        else _serialize_prior_grid(checkpoint.priors),
        "scale": None if checkpoint.scale is None else asdict(checkpoint.scale),
        "fit_config": dict(checkpoint.fit_config),
        "metadata": dict(checkpoint.metadata),
        "label_priors": {
            str(label): _serialize_prior_grid(priors)
            for label, priors in checkpoint.label_priors.items()
        },
        "label_scales": {
            str(label): asdict(scale)
            for label, scale in checkpoint.label_scales.items()
        },
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
    schema_version = int(payload.get("schema_version", 1))
    if schema_version == 1:
        priors_payload = payload["priors"]
        priors = _deserialize_prior_grid(priors_payload)
        return ModelCheckpoint(
            gene_names=list(payload["gene_names"]),
            priors=priors,
            scale=ScaleMetadata(**payload["scale"]),
            fit_config=dict(payload.get("fit_config", {})),
            metadata=dict(payload.get("metadata", {})),
            label_priors={},
            label_scales={},
        )
    priors_payload = payload.get("priors")
    priors = None if priors_payload is None else _deserialize_prior_grid(priors_payload)
    scale_payload = payload.get("scale")
    label_priors_payload = payload.get("label_priors", {})
    label_scales_payload = payload.get("label_scales", {})
    return ModelCheckpoint(
        gene_names=list(payload["gene_names"]),
        priors=priors,
        scale=None if scale_payload is None else ScaleMetadata(**scale_payload),
        fit_config=dict(payload.get("fit_config", {})),
        metadata=dict(payload.get("metadata", {})),
        label_priors={
            str(label): _deserialize_prior_grid(entry)
            for label, entry in dict(label_priors_payload).items()
        },
        label_scales={
            str(label): ScaleMetadata(**entry)
            for label, entry in dict(label_scales_payload).items()
        },
    )
