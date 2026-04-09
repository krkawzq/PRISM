from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from prism.model import ModelCheckpoint, load_checkpoint


@dataclass(frozen=True, slots=True)
class CheckpointState:
    ckpt_path: Path
    checkpoint: ModelCheckpoint
    reference_gene_names: tuple[str, ...]
    reference_positions: tuple[int, ...]
    posterior_distribution: str
    nb_overdispersion: float
    suggested_label_key: str | None


def load_checkpoint_state(
    path: Path,
    *,
    dataset_gene_names: list[str],
    gene_to_idx: dict[str, int],
    available_label_keys: tuple[str, ...],
) -> CheckpointState:
    checkpoint = load_checkpoint(path)
    metadata = checkpoint.metadata
    reference_gene_names = _require_reference_gene_names(metadata)
    overlap_names = tuple(name for name in reference_gene_names if name in gene_to_idx)
    if not overlap_names:
        raise ValueError("checkpoint reference genes do not overlap with the dataset")
    suggested_label_key = _resolve_suggested_label_key(metadata, available_label_keys)
    return CheckpointState(
        ckpt_path=path,
        checkpoint=checkpoint,
        reference_gene_names=overlap_names,
        reference_positions=tuple(gene_to_idx[name] for name in overlap_names),
        posterior_distribution=_resolve_posterior_distribution(
            checkpoint.metadata, checkpoint.fit_config
        ),
        nb_overdispersion=_resolve_nb_overdispersion(
            checkpoint.metadata, checkpoint.fit_config
        ),
        suggested_label_key=suggested_label_key,
    )


def _require_reference_gene_names(metadata: dict[str, object]) -> tuple[str, ...]:
    value = metadata.get("reference_gene_names")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("checkpoint metadata is missing reference_gene_names")
    return tuple(value)


def _resolve_posterior_distribution(
    metadata: dict[str, object], fit_config: dict[str, object]
) -> str:
    value = fit_config.get(
        "likelihood",
        metadata.get(
            "posterior_distribution",
            metadata.get("fit_distribution", "binomial"),
        ),
    )
    resolved = str(value).strip()
    if resolved not in {"binomial", "negative_binomial", "poisson"}:
        raise ValueError(f"unsupported posterior distribution: {resolved!r}")
    return resolved


def _resolve_nb_overdispersion(
    metadata: dict[str, object], fit_config: dict[str, object]
) -> float:
    value = fit_config.get("nb_overdispersion", metadata.get("nb_overdispersion", 0.01))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.01


def _resolve_suggested_label_key(
    metadata: dict[str, object], available_label_keys: tuple[str, ...]
) -> str | None:
    value = metadata.get("label_key")
    if isinstance(value, str) and value in available_label_keys:
        return value
    return available_label_keys[0] if available_label_keys else None
