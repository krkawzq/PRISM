from __future__ import annotations

from pathlib import Path

from prism.cli.checkpoint.inspect import inspect_checkpoint_command
from prism.cli.checkpoint.merge import merge_checkpoints_command
from prism.model import ModelCheckpoint, PriorGrid, ScaleMetadata, load_checkpoint, make_distribution_grid, save_checkpoint


def _prior(gene_name: str) -> PriorGrid:
    return PriorGrid(
        gene_names=[gene_name],
        distribution=make_distribution_grid(
            "binomial",
            support=[0.1, 0.2],
            probabilities=[0.4, 0.6],
        ),
        scale=2.0,
    )


def _checkpoint(gene_name: str) -> ModelCheckpoint:
    return ModelCheckpoint(
        gene_names=[gene_name],
        prior=_prior(gene_name),
        fit_config={"likelihood": "binomial"},
        metadata={
            "source_h5ad_path": "dataset.h5ad",
            "layer": None,
            "reference_gene_names": ["ref1", "ref2"],
            "requested_fit_gene_names": ["g1", "g2"],
            "fit_mode": "global",
            "label_key": None,
            "fit_distribution": "binomial",
            "posterior_distribution": "binomial",
            "support_domain": "probability",
        },
        scale_metadata=ScaleMetadata(scale=2.0, mean_reference_count=2.0),
    )


def test_inspect_checkpoint_command_runs(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pkl"
    save_checkpoint(_checkpoint("g1"), checkpoint_path)

    result = inspect_checkpoint_command(checkpoint_path=checkpoint_path)

    assert result == 0


def test_merge_checkpoints_command_writes_merged_checkpoint(tmp_path: Path) -> None:
    first_path = tmp_path / "first.pkl"
    second_path = tmp_path / "second.pkl"
    output_path = tmp_path / "merged.pkl"
    save_checkpoint(_checkpoint("g1"), first_path)
    save_checkpoint(_checkpoint("g2"), second_path)

    result = merge_checkpoints_command(
        checkpoint_paths=[first_path, second_path],
        output_path=output_path,
    )

    assert result == 0
    merged = load_checkpoint(output_path)
    assert merged.gene_names == ["g1", "g2"]
    assert merged.metadata["source_checkpoints"] == [
        str(first_path.resolve()),
        str(second_path.resolve()),
    ]
