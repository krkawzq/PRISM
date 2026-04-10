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
        key="pbmc_citrate_cpt",
        dataset_name="PBMCs from Citrate-Treated Cell Preparation Tubes",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/"
            "3p_Citrate_CPT/3p_Citrate_CPT_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="anticoagulants",
        tissue="Blood",
        material_type="Cells",
        approx_cells="~5-10k",
        platform="NextGEM",
        notes="Citrate CPT",
        output_filename="pbmc_citrate_cpt.h5ad",
    ),
    DatasetSpec(
        key="pbmc_acd_a_sepmate_ficoll",
        dataset_name="PBMCs from ACD-A Treated (SepMate-Ficoll)",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/"
            "3p_ACDA_SepMate/3p_ACDA_SepMate_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="anticoagulants",
        tissue="Blood",
        material_type="Cells",
        approx_cells="~5-10k",
        platform="NextGEM",
        notes="ACD-A SepMate-Ficoll",
        output_filename="pbmc_acd_a_sepmate_ficoll.h5ad",
    ),
    DatasetSpec(
        key="pbmc_edta_sepmate_ficoll",
        dataset_name="PBMCs from EDTA-Treated (SepMate-Ficoll)",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/"
            "3p_EDTA_SepMate/3p_EDTA_SepMate_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="anticoagulants",
        tissue="Blood",
        material_type="Cells",
        approx_cells="~5-10k",
        platform="NextGEM",
        notes="EDTA SepMate-Ficoll",
        output_filename="pbmc_edta_sepmate_ficoll.h5ad",
    ),
    DatasetSpec(
        key="pbmc_citrate_sepmate_ficoll",
        dataset_name="PBMCs from Citrate-Treated (SepMate-Ficoll)",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/"
            "3p_Citrate_SepMate/3p_Citrate_SepMate_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="anticoagulants",
        tissue="Blood",
        material_type="Cells",
        approx_cells="~5-10k",
        platform="NextGEM",
        notes="Citrate SepMate-Ficoll",
        output_filename="pbmc_citrate_sepmate_ficoll.h5ad",
    ),
    DatasetSpec(
        key="pbmc_heparin_sepmate_ficoll",
        dataset_name="PBMCs from Heparin-Treated (SepMate-Ficoll)",
        url=(
            "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/"
            "3p_Heparin_SepMate/3p_Heparin_SepMate_filtered_feature_bc_matrix.h5"
        ),
        analysis_category="anticoagulants",
        tissue="Blood",
        material_type="Cells",
        approx_cells="~5-10k",
        platform="NextGEM",
        notes="Heparin SepMate-Ficoll",
        output_filename="pbmc_heparin_sepmate_ficoll.h5ad",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 10x anticoagulant PBMC datasets and write h5ad files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw_dir("anticoagulants"),
        help="Directory where the original 10x h5 downloads are stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir("anticoagulants"),
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
