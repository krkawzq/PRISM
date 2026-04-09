from __future__ import annotations

import argparse
from pathlib import Path

from _10xgenomics_common import (
    DatasetSpec,
    build_dataset,
    default_output_dir,
    default_raw_dir,
    ensure_relative_symlink,
    prepare_simple_category,
)


CELL_GRADIENT_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="pbmc_500_lt_chromium_x",
        dataset_name="500 Human PBMCs, 3' LT v3.1, Chromium X",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/"
            "500_PBMC_3p_LT_Chromium_X/"
            "500_PBMC_3p_LT_Chromium_X_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="cell_gradients",
        tissue="Blood",
        material_type="Cells",
        approx_cells="~500",
        platform="Chromium X",
        notes="Low Throughput",
        output_filename="pbmc_500_lt_chromium_x.h5ad",
    ),
    DatasetSpec(
        key="pbmc_5k_chromium_controller",
        dataset_name="5k Human PBMCs, 3' v3.1, Chromium Controller",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/7.0.1/"
            "SC3pv3_GEX_Human_PBMC/"
            "SC3pv3_GEX_Human_PBMC_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="cell_gradients",
        tissue="Blood",
        material_type="Cells",
        approx_cells="~5.1k",
        platform="Chromium Controller",
        notes="Standard throughput",
        output_filename="pbmc_5k_chromium_controller.h5ad",
    ),
    DatasetSpec(
        key="pbmc_20k_ht_chromium_x",
        dataset_name="20k Human PBMCs, 3' HT v3.1, Chromium X",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/"
            "20k_PBMC_3p_HT_nextgem_Chromium_X/"
            "20k_PBMC_3p_HT_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="cell_gradients",
        tissue="Blood",
        material_type="Cells",
        approx_cells="~24k",
        platform="Chromium X",
        notes="High Throughput",
        output_filename="pbmc_20k_ht_chromium_x.h5ad",
    ),
)


REUSED_PLATFORM_SPEC = DatasetSpec(
    key="pbmc_10k_chromium_x",
    dataset_name="10k Human PBMCs, 3' v3.1, Chromium X",
    url=(
        "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/"
        "10k_PBMC_3p_nextgem_Chromium_X/"
        "10k_PBMC_3p_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
    ),
    analysis_category="platforms",
    tissue="Blood",
    material_type="Cells",
    approx_cells="~12k",
    platform="Chromium X",
    notes="Canonical 10k reference reused by the cell-gradient collection",
    output_filename="pbmc_10k_chromium_x.h5ad",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 10x PBMC cell-gradient datasets and write h5ad files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw_dir("cell_gradients"),
        help="Directory where the original 10x h5 downloads are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir("cell_gradients"),
        help="Directory where the processed h5ad outputs are written.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the source h5 files even if they already exist.",
    )
    return parser.parse_args()


def ensure_reused_10k_reference(*, force_download: bool) -> tuple[Path, Path]:
    platform_raw_root = default_raw_dir("platforms").resolve()
    platform_output_root = default_output_dir("platforms").resolve()
    return build_dataset(
        REUSED_PLATFORM_SPEC,
        raw_root=platform_raw_root,
        output_root=platform_output_root,
        force_download=force_download,
    )


def main() -> None:
    args = parse_args()
    raw_root = args.raw_dir.resolve()
    output_root = args.output_dir.resolve()

    prepare_simple_category(
        CELL_GRADIENT_DATASETS,
        raw_root=raw_root,
        output_root=output_root,
        force_download=args.force_download,
    )

    canonical_raw_path, canonical_output_path = ensure_reused_10k_reference(
        force_download=args.force_download
    )
    reused_raw_link = raw_root / canonical_raw_path.name
    reused_output_link = output_root / "pbmc_10k_chromium_x.h5ad"

    ensure_relative_symlink(canonical_raw_path, reused_raw_link)
    ensure_relative_symlink(canonical_output_path, reused_output_link)


if __name__ == "__main__":
    main()
