from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import pickle
from typing import Any, cast
import warnings

import numpy as np

from .types import PriorFitResult, PriorGrid, ScaleMetadata


_SUPPORTED_DISTRIBUTIONS = {"binomial", "negative_binomial", "poisson"}
_SUPPORTED_GRID_DOMAINS = {"p", "rate"}


@dataclass(frozen=True, slots=True)
class ModelCheckpoint:
    gene_names: list[str]
    priors: PriorGrid | None
    scale: ScaleMetadata | None
    fit_config: dict[str, Any]
    metadata: dict[str, Any]
    label_priors: dict[str, PriorGrid] = field(default_factory=dict)
    label_scales: dict[str, ScaleMetadata] = field(default_factory=dict)


def _normalize_distribution(value: object) -> str | None:
    if value is None:
        return None
    resolved = str(value).strip().lower()
    if not resolved:
        return None
    if resolved not in _SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            "distribution must be one of: "
            + ", ".join(sorted(_SUPPORTED_DISTRIBUTIONS))
            + f"; got {value!r}"
        )
    return resolved


def _normalize_grid_domain(value: object) -> str | None:
    if value is None:
        return None
    resolved = str(value).strip().lower()
    if not resolved:
        return None
    if resolved not in _SUPPORTED_GRID_DOMAINS:
        raise ValueError(
            "grid_domain must be one of: "
            + ", ".join(sorted(_SUPPORTED_GRID_DOMAINS))
            + f"; got {value!r}"
        )
    return resolved


def resolve_checkpoint_distribution(
    *,
    schema_version: int,
    metadata: dict[str, Any],
    priors: PriorGrid | None,
    label_priors: dict[str, PriorGrid],
    checkpoint_path: str | Path | None = None,
) -> tuple[dict[str, Any], bool]:
    resolved_metadata = dict(metadata)
    path_text = (
        "" if checkpoint_path is None else f" for checkpoint {Path(checkpoint_path)}"
    )

    metadata_fit = _normalize_distribution(resolved_metadata.get("fit_distribution"))
    metadata_post = _normalize_distribution(
        resolved_metadata.get("posterior_distribution")
    )
    metadata_domain = _normalize_grid_domain(resolved_metadata.get("grid_domain"))

    prior_distributions = {
        prior.distribution for prior in ([priors] if priors is not None else [])
    }
    prior_distributions.update(prior.distribution for prior in label_priors.values())
    prior_domains = {
        prior.grid_domain for prior in ([priors] if priors is not None else [])
    }
    prior_domains.update(prior.grid_domain for prior in label_priors.values())

    if len(prior_distributions) > 1:
        raise ValueError(
            "checkpoint mixes multiple prior distributions"
            f"{path_text}: {sorted(prior_distributions)}"
        )
    if len(prior_domains) > 1:
        raise ValueError(
            "checkpoint mixes multiple prior grid_domain values"
            f"{path_text}: {sorted(prior_domains)}"
        )

    prior_distribution = next(iter(prior_distributions), None)
    prior_domain = next(iter(prior_domains), None)

    if schema_version <= 1:
        fit_distribution = metadata_fit or prior_distribution or "binomial"
        posterior_distribution = metadata_post or fit_distribution
        grid_domain = metadata_domain or prior_domain or "p"
        is_legacy = True
    else:
        missing_fields = [
            name
            for name, value in (
                ("fit_distribution", metadata_fit),
                ("posterior_distribution", metadata_post),
                ("grid_domain", metadata_domain),
            )
            if value is None
        ]
        if missing_fields:
            raise ValueError(
                "schema>=2 checkpoint is missing required distribution metadata"
                f"{path_text}: {missing_fields}"
            )
        fit_distribution = metadata_fit
        posterior_distribution = metadata_post
        grid_domain = metadata_domain
        is_legacy = False
        if prior_distribution is not None and prior_distribution != fit_distribution:
            raise ValueError(
                "checkpoint metadata fit_distribution disagrees with prior distribution"
                f"{path_text}: metadata={fit_distribution!r}, prior={prior_distribution!r}"
            )
        if prior_domain is not None and prior_domain != grid_domain:
            raise ValueError(
                "checkpoint metadata grid_domain disagrees with prior grid_domain"
                f"{path_text}: metadata={grid_domain!r}, prior={prior_domain!r}"
            )
        if posterior_distribution != fit_distribution:
            raise ValueError(
                "schema>=2 checkpoint requires posterior_distribution to match fit_distribution"
                f"{path_text}: fit={fit_distribution!r}, posterior={posterior_distribution!r}"
            )

    if grid_domain == "rate" and fit_distribution != "poisson":
        raise ValueError(
            "rate grid_domain requires poisson distribution"
            f"{path_text}: distribution={fit_distribution!r}, grid_domain={grid_domain!r}"
        )
    if grid_domain == "p" and fit_distribution == "poisson":
        raise ValueError(
            "poisson distribution requires rate grid_domain"
            f"{path_text}: distribution={fit_distribution!r}, grid_domain={grid_domain!r}"
        )

    resolved_metadata["fit_distribution"] = fit_distribution
    resolved_metadata["posterior_distribution"] = posterior_distribution
    resolved_metadata["grid_domain"] = grid_domain
    resolved_metadata["distribution_resolution"] = (
        "legacy-compatibility" if is_legacy else "explicit"
    )
    resolved_metadata["legacy_compatibility"] = bool(is_legacy)
    if is_legacy:
        warnings.warn(
            "Loaded legacy checkpoint via compatibility path"
            f"{path_text}; inferred distribution={fit_distribution!r}, grid_domain={grid_domain!r}",
            UserWarning,
            stacklevel=2,
        )
    return resolved_metadata, is_legacy


def checkpoint_from_fit_result(
    result: PriorFitResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> ModelCheckpoint:
    resolved_metadata = {} if metadata is None else dict(metadata)
    resolved_metadata.setdefault(
        "fit_distribution", str(result.config.get("likelihood", "binomial"))
    )
    resolved_metadata.setdefault(
        "posterior_distribution",
        str(resolved_metadata.get("fit_distribution", "binomial")),
    )
    resolved_metadata.setdefault("grid_domain", result.priors.grid_domain)
    if "nb_overdispersion" in result.config:
        resolved_metadata.setdefault(
            "nb_overdispersion", result.config.get("nb_overdispersion")
        )
    resolved_metadata, _ = resolve_checkpoint_distribution(
        schema_version=2,
        metadata=resolved_metadata,
        priors=result.priors,
        label_priors={},
    )
    return ModelCheckpoint(
        gene_names=list(result.gene_names),
        priors=result.priors,
        scale=result.scale,
        fit_config=dict(result.config),
        metadata=resolved_metadata,
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
        "distribution": priors.distribution,
    }


def _deserialize_prior_grid(payload: dict[str, Any], *, strict: bool) -> PriorGrid:
    grid_domain = payload.get("grid_domain")
    distribution = payload.get("distribution")
    if strict and grid_domain is None:
        raise ValueError(
            "schema>=2 prior payload is missing required field 'grid_domain'"
        )
    if strict and distribution is None:
        raise ValueError(
            "schema>=2 prior payload is missing required field 'distribution'"
        )
    resolved_grid_domain = _normalize_grid_domain(grid_domain) or "p"
    resolved_distribution = _normalize_distribution(distribution) or "binomial"
    return PriorGrid(
        gene_names=list(payload["gene_names"]),
        p_grid=np.asarray(payload["p_grid"], dtype=np.float64),
        weights=np.asarray(payload["weights"], dtype=np.float64),
        S=float(payload["S"]),
        grid_domain=cast(Any, resolved_grid_domain),
        distribution=cast(Any, resolved_distribution),
    )


def _validate_checkpoint_prior_grid(
    priors: PriorGrid, *, label: str | None = None
) -> None:
    prefix = "label prior" if label is not None else "global prior"
    if not priors.gene_names:
        raise ValueError(f"{prefix} gene_names cannot be empty")
    if len(priors.gene_names) != len(set(priors.gene_names)):
        raise ValueError(f"{prefix} gene_names must be unique")
    if not np.isfinite(priors.S) or priors.S <= 0:
        raise ValueError(f"{prefix} S must be positive")

    p_grid = np.asarray(priors.p_grid, dtype=np.float64)
    weights = np.asarray(priors.weights, dtype=np.float64)
    if p_grid.ndim not in (1, 2):
        raise ValueError(f"{prefix} p_grid must be 1D or 2D")
    if weights.ndim not in (1, 2):
        raise ValueError(f"{prefix} weights must be 1D or 2D")
    if p_grid.ndim == 2 and p_grid.shape != weights.shape:
        raise ValueError(f"{prefix} 2D p_grid and weights must have identical shape")
    if p_grid.ndim == 1 and weights.shape[-1] != p_grid.shape[0]:
        raise ValueError(f"{prefix} 1D p_grid length must match weights grid dimension")
    if weights.ndim == 2 and weights.shape[0] != len(priors.gene_names):
        raise ValueError(
            f"{prefix} weights first dimension must match gene_names length"
        )
    if weights.ndim == 1 and len(priors.gene_names) != 1:
        raise ValueError(f"{prefix} unbatched weights require exactly one gene")
    if np.any(~np.isfinite(p_grid)) or np.any(~np.isfinite(weights)):
        raise ValueError(f"{prefix} p_grid and weights must be finite")
    if priors.grid_domain == "p":
        if np.any(p_grid < 0) or np.any(p_grid > 1):
            raise ValueError(f"{prefix} p_grid must lie in [0, 1]")
    elif priors.grid_domain == "rate":
        if np.any(p_grid < 0):
            raise ValueError(f"{prefix} p_grid must be >= 0")
    else:
        raise ValueError(f"unsupported grid_domain: {priors.grid_domain}")
    if priors.distribution not in _SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"unsupported distribution: {priors.distribution}")
    if np.any(weights < 0):
        raise ValueError(f"{prefix} weights must be non-negative")
    if np.any(np.abs(weights.sum(axis=-1) - 1.0) > 1e-6):
        raise ValueError(f"{prefix} weights must sum to 1 along the grid axis")


def _validate_checkpoint_consistency(checkpoint: ModelCheckpoint) -> None:
    gene_name_set = set(checkpoint.gene_names)
    if len(gene_name_set) != len(checkpoint.gene_names):
        raise ValueError("checkpoint gene_names must be unique")

    if checkpoint.priors is not None:
        _validate_checkpoint_prior_grid(checkpoint.priors)
        prior_gene_set = set(checkpoint.priors.gene_names)
        if prior_gene_set != gene_name_set:
            raise ValueError("checkpoint gene_names must match global prior gene_names")

    label_prior_keys = set(checkpoint.label_priors)
    label_scale_keys = set(checkpoint.label_scales)
    if not label_scale_keys.issubset(label_prior_keys):
        raise ValueError(
            "checkpoint label_scales keys must be a subset of label_priors keys"
        )

    for label, priors in checkpoint.label_priors.items():
        _validate_checkpoint_prior_grid(priors, label=label)
        label_gene_set = set(priors.gene_names)
        if not label_gene_set.issubset(gene_name_set):
            raise ValueError(
                f"label prior {label!r} contains genes outside checkpoint gene_names"
            )


def save_checkpoint(checkpoint: ModelCheckpoint, path: str | Path) -> None:
    metadata, _ = resolve_checkpoint_distribution(
        schema_version=2,
        metadata=dict(checkpoint.metadata),
        priors=checkpoint.priors,
        label_priors=checkpoint.label_priors,
        checkpoint_path=path,
    )
    if "nb_overdispersion" in checkpoint.fit_config:
        metadata.setdefault(
            "nb_overdispersion", checkpoint.fit_config.get("nb_overdispersion")
        )
    payload = {
        "schema_version": 2,
        "gene_names": checkpoint.gene_names,
        "priors": None
        if checkpoint.priors is None
        else _serialize_prior_grid(checkpoint.priors),
        "scale": None if checkpoint.scale is None else asdict(checkpoint.scale),
        "fit_config": dict(checkpoint.fit_config),
        "metadata": metadata,
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
        priors = _deserialize_prior_grid(priors_payload, strict=False)
        metadata, _ = resolve_checkpoint_distribution(
            schema_version=schema_version,
            metadata=dict(payload.get("metadata", {})),
            priors=priors,
            label_priors={},
            checkpoint_path=resolved,
        )
        checkpoint = ModelCheckpoint(
            gene_names=list(payload["gene_names"]),
            priors=priors,
            scale=ScaleMetadata(**payload["scale"]),
            fit_config=dict(payload.get("fit_config", {})),
            metadata=metadata,
            label_priors={},
            label_scales={},
        )
        _validate_checkpoint_consistency(checkpoint)
        return checkpoint
    priors_payload = payload.get("priors")
    priors = (
        None
        if priors_payload is None
        else _deserialize_prior_grid(priors_payload, strict=True)
    )
    scale_payload = payload.get("scale")
    label_priors_payload = payload.get("label_priors", {})
    label_scales_payload = payload.get("label_scales", {})
    label_priors = {
        str(label): _deserialize_prior_grid(entry, strict=True)
        for label, entry in dict(label_priors_payload).items()
    }
    metadata, _ = resolve_checkpoint_distribution(
        schema_version=schema_version,
        metadata=dict(payload.get("metadata", {})),
        priors=priors,
        label_priors=label_priors,
        checkpoint_path=resolved,
    )
    checkpoint = ModelCheckpoint(
        gene_names=list(payload["gene_names"]),
        priors=priors,
        scale=None if scale_payload is None else ScaleMetadata(**scale_payload),
        fit_config=dict(payload.get("fit_config", {})),
        metadata=metadata,
        label_priors=label_priors,
        label_scales={
            str(label): ScaleMetadata(**entry)
            for label, entry in dict(label_scales_payload).items()
        },
    )
    _validate_checkpoint_consistency(checkpoint)
    return checkpoint
