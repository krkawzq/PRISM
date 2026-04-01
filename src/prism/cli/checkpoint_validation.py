from __future__ import annotations

from typing import Any

from prism.model import ModelCheckpoint
from prism.model.checkpoint import resolve_checkpoint_distribution


def resolve_cli_checkpoint_distribution(
    checkpoint: ModelCheckpoint,
    *,
    command_name: str,
    allow_distributions: set[str] | None = None,
    require_grid_domains: set[str] | None = None,
    require_label_priors: bool = False,
    require_global_priors: bool = False,
) -> dict[str, Any]:
    metadata, is_legacy = resolve_checkpoint_distribution(
        schema_version=int(checkpoint.metadata.get("schema_version", 2)),
        metadata=checkpoint.metadata,
        priors=checkpoint.priors,
        label_priors=checkpoint.label_priors,
    )
    distribution = str(metadata["posterior_distribution"])
    grid_domain = str(metadata["grid_domain"])
    if require_label_priors and not checkpoint.label_priors:
        raise ValueError(f"{command_name} requires checkpoint label priors")
    if require_global_priors and checkpoint.priors is None:
        raise ValueError(f"{command_name} requires checkpoint global priors")
    if allow_distributions is not None and distribution not in allow_distributions:
        raise ValueError(
            f"{command_name} does not support checkpoint distribution {distribution!r}; "
            f"supported: {sorted(allow_distributions)} (grid_domain={grid_domain!r})"
        )
    if require_grid_domains is not None and grid_domain not in require_grid_domains:
        raise ValueError(
            f"{command_name} does not support checkpoint grid_domain {grid_domain!r}; "
            f"supported: {sorted(require_grid_domains)} (distribution={distribution!r})"
        )
    resolved = dict(metadata)
    resolved["is_legacy_compatibility"] = bool(is_legacy)
    return resolved


__all__ = ["resolve_cli_checkpoint_distribution"]
