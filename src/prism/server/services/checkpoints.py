from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from prism.model import ModelCheckpoint, load_checkpoint


@dataclass(frozen=True, slots=True)
class CheckpointState:
    ckpt_path: Path
    checkpoint: ModelCheckpoint
    fitted_gene_names: tuple[str, ...]
    reference_gene_names: tuple[str, ...]
    label_prior_names: tuple[str, ...]


def load_checkpoint_state(path: Path, dataset_gene_names: list[str]) -> CheckpointState:
    checkpoint = load_checkpoint(path)
    dataset_gene_set = set(dataset_gene_names)
    fitted_gene_names = tuple()
    if checkpoint.priors is not None:
        fitted_gene_names = tuple(
            name for name in checkpoint.priors.gene_names if name in dataset_gene_set
        )
    label_prior_names = tuple(sorted(checkpoint.label_priors))
    if not fitted_gene_names and not label_prior_names:
        raise ValueError("checkpoint has no usable priors")
    reference_gene_names = _read_reference_gene_names(checkpoint)
    overlap_reference = tuple(
        name for name in reference_gene_names if name in dataset_gene_set
    )
    if not overlap_reference:
        raise ValueError("checkpoint reference genes do not overlap with the dataset")
    return CheckpointState(
        ckpt_path=path,
        checkpoint=checkpoint,
        fitted_gene_names=fitted_gene_names,
        reference_gene_names=overlap_reference,
        label_prior_names=label_prior_names,
    )


def _read_reference_gene_names(checkpoint: ModelCheckpoint) -> tuple[str, ...]:
    value = checkpoint.metadata.get("reference_gene_names")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("checkpoint metadata is missing reference_gene_names")
    return tuple(value)
