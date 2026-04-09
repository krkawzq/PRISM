from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any, cast

import numpy as np

from .constants import DTYPE_NP, DistributionName
from .types import (
    DistributionGrid,
    PriorFitResult,
    PriorGrid,
    ScaleMetadata,
    make_distribution_grid,
)

SCHEMA_VERSION = "0.1.0"


def _require_unique_gene_names(gene_names: list[str]) -> list[str]:
    resolved = [str(name) for name in gene_names]
    if not resolved:
        raise ValueError("gene_names cannot be empty")
    if len(resolved) != len(set(resolved)):
        raise ValueError("gene_names must be unique")
    return resolved


def _serialize_distribution(distribution: DistributionGrid) -> dict[str, Any]:
    return {
        "distribution": distribution.distribution,
        "support": np.asarray(distribution.support, dtype=DTYPE_NP),
        "probabilities": np.asarray(distribution.probabilities, dtype=DTYPE_NP),
    }


def _parse_distribution_name(value: object) -> DistributionName:
    resolved = str(value).strip()
    if resolved not in {"binomial", "negative_binomial", "poisson"}:
        raise ValueError(f"unsupported distribution: {value!r}")
    return cast(DistributionName, resolved)


def _deserialize_distribution(payload: dict[str, Any]) -> DistributionGrid:
    return make_distribution_grid(
        _parse_distribution_name(payload["distribution"]),
        support=np.asarray(payload["support"], dtype=DTYPE_NP),
        probabilities=np.asarray(payload["probabilities"], dtype=DTYPE_NP),
    )


def _serialize_prior(prior: PriorGrid) -> dict[str, Any]:
    return {
        "gene_names": list(prior.gene_names),
        "distribution": _serialize_distribution(prior.distribution),
        "scale": float(prior.scale),
    }


def _deserialize_prior(payload: dict[str, Any]) -> PriorGrid:
    return PriorGrid(
        gene_names=list(payload["gene_names"]),
        distribution=_deserialize_distribution(dict(payload["distribution"])),
        scale=float(payload["scale"]),
    )


def _serialize_scale_metadata(scale: ScaleMetadata) -> dict[str, Any]:
    return {
        "scale": float(scale.scale),
        "mean_reference_count": float(scale.mean_reference_count),
    }


def _deserialize_scale_metadata(payload: dict[str, Any]) -> ScaleMetadata:
    return ScaleMetadata(
        scale=float(payload["scale"]),
        mean_reference_count=float(payload["mean_reference_count"]),
    )


@dataclass(frozen=True, slots=True)
class ModelCheckpoint:
    gene_names: list[str]
    prior: PriorGrid | None = None
    fit_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    scale_metadata: ScaleMetadata | None = None
    label_priors: dict[str, PriorGrid] = field(default_factory=dict)
    label_scale_metadata: dict[str, ScaleMetadata] = field(default_factory=dict)

    def __post_init__(self) -> None:
        gene_names = _require_unique_gene_names(list(self.gene_names))
        if self.prior is None and not self.label_priors:
            raise ValueError("checkpoint must contain a global prior or label priors")
        if self.prior is not None and gene_names != list(self.prior.gene_names):
            raise ValueError("checkpoint gene_names must match prior.gene_names")
        label_priors = {str(key): value for key, value in self.label_priors.items()}
        for key, value in label_priors.items():
            if gene_names != list(value.gene_names):
                raise ValueError(
                    f"checkpoint gene_names must match label_priors[{key!r}].gene_names"
                )
        label_scale_metadata = {
            str(key): value for key, value in self.label_scale_metadata.items()
        }
        extra_scale_keys = set(label_scale_metadata) - set(label_priors)
        if extra_scale_keys:
            raise ValueError(
                "label_scale_metadata keys must be a subset of label_priors keys"
            )
        object.__setattr__(self, "gene_names", gene_names)
        object.__setattr__(self, "fit_config", dict(self.fit_config))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "label_priors", label_priors)
        object.__setattr__(self, "label_scale_metadata", label_scale_metadata)

    @property
    def has_global_prior(self) -> bool:
        return self.prior is not None

    @property
    def has_label_priors(self) -> bool:
        return bool(self.label_priors)

    @property
    def available_labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.label_priors))

    def get_prior(self, label: str | None = None) -> PriorGrid:
        if label is None:
            if self.prior is None:
                raise KeyError("checkpoint has no global prior")
            return self.prior
        if label not in self.label_priors:
            raise KeyError(
                f"unknown label prior: {label!r}; available labels: {sorted(self.label_priors)}"
            )
        return self.label_priors[label]

    def get_scale_metadata(self, label: str | None = None) -> ScaleMetadata | None:
        if label is None:
            return self.scale_metadata
        return self.label_scale_metadata.get(label)


def checkpoint_from_fit_result(
    result: PriorFitResult,
    *,
    metadata: dict[str, Any] | None = None,
    scale_metadata: ScaleMetadata | None = None,
) -> ModelCheckpoint:
    resolved_metadata = {} if metadata is None else dict(metadata)
    resolved_metadata.setdefault("final_objective", float(result.final_objective))
    resolved_metadata.setdefault("fit_distribution", result.prior.distribution_name)
    resolved_metadata.setdefault(
        "posterior_distribution",
        result.prior.distribution_name,
    )
    resolved_metadata.setdefault("distribution", result.prior.distribution_name)
    resolved_metadata.setdefault("support_domain", result.prior.support_domain)
    resolved_scale_metadata = scale_metadata
    if resolved_scale_metadata is None:
        mean_reference_count_value = result.config.get("mean_reference_count")
        if mean_reference_count_value is not None:
            resolved_scale_metadata = ScaleMetadata(
                scale=float(result.prior.scale),
                mean_reference_count=float(mean_reference_count_value),
            )
    return ModelCheckpoint(
        gene_names=list(result.gene_names),
        prior=result.prior,
        fit_config=dict(result.config),
        metadata=resolved_metadata,
        scale_metadata=resolved_scale_metadata,
        label_priors={},
        label_scale_metadata={},
    )


def save_checkpoint(checkpoint: ModelCheckpoint, path: str | Path) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "gene_names": list(checkpoint.gene_names),
        "prior": None
        if checkpoint.prior is None
        else _serialize_prior(checkpoint.prior),
        "fit_config": dict(checkpoint.fit_config),
        "metadata": dict(checkpoint.metadata),
        "scale_metadata": None
        if checkpoint.scale_metadata is None
        else _serialize_scale_metadata(checkpoint.scale_metadata),
        "label_priors": {
            label: _serialize_prior(prior)
            for label, prior in checkpoint.label_priors.items()
        },
        "label_scale_metadata": {
            label: _serialize_scale_metadata(scale)
            for label, scale in checkpoint.label_scale_metadata.items()
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
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema_version: {schema_version}; expected {SCHEMA_VERSION}"
        )
    prior_payload = payload.get("prior")
    scale_payload = payload.get("scale_metadata")
    label_prior_payload = dict(payload.get("label_priors", {}))
    label_scale_payload = dict(payload.get("label_scale_metadata", {}))
    return ModelCheckpoint(
        gene_names=list(payload["gene_names"]),
        prior=None
        if prior_payload is None
        else _deserialize_prior(dict(prior_payload)),
        fit_config=dict(payload.get("fit_config", {})),
        metadata=dict(payload.get("metadata", {})),
        scale_metadata=(
            None
            if scale_payload is None
            else _deserialize_scale_metadata(dict(scale_payload))
        ),
        label_priors={
            str(label): _deserialize_prior(dict(prior_payload))
            for label, prior_payload in label_prior_payload.items()
        },
        label_scale_metadata={
            str(label): _deserialize_scale_metadata(dict(scale_payload))
            for label, scale_payload in label_scale_payload.items()
        },
    )


__all__ = [
    "ModelCheckpoint",
    "SCHEMA_VERSION",
    "checkpoint_from_fit_result",
    "load_checkpoint",
    "save_checkpoint",
]
