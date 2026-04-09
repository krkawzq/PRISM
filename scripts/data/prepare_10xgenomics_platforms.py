from __future__ import annotations

import argparse
from pathlib import Path

from _10xgenomics_common import (
    DatasetSpec,
    default_output_dir,
    default_raw_dir,
    prepare_simple_category,
)


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
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
        notes="Platform comparison reference",
        output_filename="pbmc_10k_chromium_x.h5ad",
    ),
    DatasetSpec(
        key="pbmc_10k_chromium_controller",
        dataset_name="10k Human PBMCs, 3' v3.1, Chromium Controller",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/"
            "10k_PBMC_3p_nextgem_Chromium_Controller/"
            "10k_PBMC_3p_nextgem_Chromium_Controller_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="platforms",
        tissue="Blood",
        material_type="Cells",
        approx_cells="~11.5k",
        platform="Chromium Controller",
        notes="Same donor and chemistry as Chromium X comparator",
        output_filename="pbmc_10k_chromium_controller.h5ad",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 10x platform-comparison PBMC datasets and write h5ad files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw_dir("platforms"),
        help="Directory where the original 10x h5 downloads are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir("platforms"),
        help="Directory where the processed h5ad outputs are written.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the source h5 files even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_simple_category(
        DATASETS,
        raw_root=args.raw_dir.resolve(),
        output_root=args.output_dir.resolve(),
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()
