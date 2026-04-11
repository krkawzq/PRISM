from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

from prism.io.anndata import write_h5ad


DEFAULT_NAME_REPLACER = {
    "C3orf72": "FOXL2NB",
    "C19orf26": "CBARP",
    "KIAA1804": "RP5-862P8.2",
    "RHOXF2": "RHOXF2B",
}

CONTROL_GUIDE_TARGETS = {
    "NegCtrl0_NegCtrl0",
    "NegCtrl1_NegCtrl0",
    "NegCtrl10_NegCtrl0",
    "NegCtrl11_NegCtrl0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach Norman guide identities to an AnnData object and derive "
            "guide_target / guide_target_1 / guide_target_2 metadata."
        )
    )
    parser.add_argument(
        "input_h5ad",
        type=Path,
        help="Input AnnData built from the Cell Ranger matrix files.",
    )
    parser.add_argument(
        "--cell-identities",
        type=Path,
        required=True,
        help="Path to raw_cell_identities.csv(.gz) or cell_identities.csv(.gz).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output h5ad path.",
    )
    parser.add_argument(
        "--replace-target-names",
        dest="replace_target_names",
        action="store_true",
        help="Apply the Norman notebook target-name replacements. Default: on.",
    )
    parser.add_argument(
        "--no-replace-target-names",
        dest="replace_target_names",
        action="store_false",
        help="Keep parsed target names exactly as encoded in the guide label.",
    )
    parser.set_defaults(replace_target_names=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(
        columns={
            "cell BC": "cell_barcode",
            "read count": "guide_read_count",
            "UMI count": "guide_UMI_count",
            "coverage": "guide_coverage",
            "cell_BC": "cell_barcode",
            "read_count": "guide_read_count",
            "UMI_count": "guide_UMI_count",
            "gemgroup": "gem_group",
        }
    ).copy()
    renamed.columns = [
        col.replace(" ", "_") if isinstance(col, str) else col for col in renamed.columns
    ]
    if "cell_barcode" not in renamed.columns:
        raise ValueError("cell identities file must contain a cell barcode column")
    if renamed["cell_barcode"].duplicated().any():
        duplicated = int(renamed["cell_barcode"].duplicated().sum())
        raise ValueError(
            f"cell identities file contains duplicated cell barcodes: {duplicated}"
        )
    renamed = renamed.set_index("cell_barcode", drop=True)
    return renamed


def read_cell_identities(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    return normalize_identity_columns(df)


def parse_guide_target(label: object) -> str:
    if isinstance(label, str):
        return label.split("__")[0]
    return ""


def split_guide_target(label: str) -> tuple[str, str, int]:
    if label == "":
        return "", "", 0
    parts = label.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1], len(parts)
    if len(parts) == 1:
        return parts[0], "", len(parts)
    return "", "", 0


def apply_name_replacements(value: str, *, enabled: bool) -> str:
    if not enabled:
        return value
    return DEFAULT_NAME_REPLACER.get(value, value)


def is_negative_control_target(value: str) -> bool:
    return isinstance(value, str) and value.startswith("NegCtrl")


def collapse_target(label: str, target_1: str, target_2: str) -> str:
    if label in CONTROL_GUIDE_TARGETS:
        return "control"
    if target_1 == "" and target_2 == "":
        return ""
    if is_negative_control_target(target_1) and not is_negative_control_target(target_2):
        return target_2
    if is_negative_control_target(target_2) and not is_negative_control_target(target_1):
        return target_1
    return label


def normalize_obs_dtypes(obs: pd.DataFrame) -> pd.DataFrame:
    normalized = obs.copy()

    for column in normalized.columns:
        series = normalized[column]
        if pd.api.types.is_object_dtype(series):
            non_null = series.dropna()

            if not non_null.empty and non_null.map(lambda x: isinstance(x, bool)).all():
                normalized[column] = series.astype("boolean")
                continue

            if non_null.empty or non_null.map(lambda x: isinstance(x, str)).all():
                normalized[column] = series.astype("string")

    return normalized


def build_guide_metadata(
    obs: pd.DataFrame,
    identities: pd.DataFrame,
    *,
    replace_target_names: bool,
) -> pd.DataFrame:
    merged = obs.merge(identities, left_index=True, right_index=True, how="left")

    if "gem_group_x" in merged.columns or "gem_group_y" in merged.columns:
        left = merged.get("gem_group_x")
        right = merged.get("gem_group_y")
        if left is not None and right is not None:
            left_float = left.astype("Float64")
            right_float = right.astype("Float64")
            disagree = (left_float != right_float) & ~(left_float.isna() | right_float.isna())
            if bool(disagree.any()):
                raise ValueError("gem_group values disagree between AnnData and cell identities")
            merged["gem_group"] = left_float.fillna(right_float).astype("Int64")
        elif left is not None:
            merged["gem_group"] = left
        elif right is not None:
            merged["gem_group"] = right
        merged = merged.drop(columns=["gem_group_x", "gem_group_y"], errors="ignore")

    merged["guide_target_raw"] = merged["guide_identity"].map(parse_guide_target)

    split_values = merged["guide_target_raw"].map(split_guide_target)
    merged["guide_target_1_raw"] = split_values.map(lambda x: x[0])
    merged["guide_target_2_raw"] = split_values.map(lambda x: x[1])
    merged["num_targets"] = split_values.map(lambda x: x[2]).astype("Int64")

    merged["guide_target_1"] = merged["guide_target_1_raw"].map(
        lambda x: apply_name_replacements(x, enabled=replace_target_names)
    )
    merged["guide_target_2"] = merged["guide_target_2_raw"].map(
        lambda x: apply_name_replacements(x, enabled=replace_target_names)
    )
    merged["guide_target"] = (
        merged["guide_target_1"].fillna("").astype(str)
        + "_"
        + merged["guide_target_2"].fillna("").astype(str)
    )

    one_target = merged["num_targets"] == 1
    zero_target = merged["num_targets"] == 0
    merged.loc[one_target, "guide_target"] = merged.loc[one_target, "guide_target_1"]
    merged.loc[zero_target, "guide_target"] = ""
    merged["target"] = [
        collapse_target(label, target_1, target_2)
        for label, target_1, target_2 in zip(
            merged["guide_target"],
            merged["guide_target_1"],
            merged["guide_target_2"],
        )
    ]

    return normalize_obs_dtypes(merged)


def main() -> None:
    args = parse_args()
    input_path = args.input_h5ad.expanduser().resolve()
    identities_path = args.cell_identities.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Use --force to overwrite.")

    adata = ad.read_h5ad(input_path)
    identities = read_cell_identities(identities_path)
    adata.obs = build_guide_metadata(
        adata.obs,
        identities,
        replace_target_names=args.replace_target_names,
    )

    adata.uns["guide_target_builder"] = {
        "cell_identities_path": str(identities_path),
        "replace_target_names": bool(args.replace_target_names),
        "name_replacer": DEFAULT_NAME_REPLACER if args.replace_target_names else {},
        "parsing_rule": "guide_identity.split('__')[0]",
        "control_guide_targets": sorted(CONTROL_GUIDE_TARGETS),
        "target_rule": (
            "control for four canonical double-negative controls; "
            "single perturbations collapse NegCtrl/gene pairs to the non-control gene; "
            "double perturbations preserve guide_target order"
        ),
        "scope": "metadata only; no control inference or perturbation collapsing",
    }

    write_h5ad(adata, output_path)
    print(
        f"Wrote {adata.n_obs:,} cells with guide metadata to {output_path} "
        f"(matched identities: {adata.obs['guide_identity'].notna().sum():,})"
    )


if __name__ == "__main__":
    main()
