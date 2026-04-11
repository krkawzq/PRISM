from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from prism.io.anndata import slice_gene_matrix, write_h5ad


PHASE_LIST = ["G1-S", "S", "G2-M", "M", "M-G1"]

DEFAULT_CELL_PHASE_GENES: OrderedDict[str, list[str]] = OrderedDict(
    {
        "G1-S": [
            "ARGLU1",
            "BRD7",
            "CDC6",
            "CLSPN",
            "ESD",
            "GINS2",
            "GMNN",
            "LUC7L3",
            "MCM5",
            "MCM6",
            "NASP",
            "PCNA",
            "PNN",
            "SLBP",
            "SRSF7",
            "SSR3",
            "ZRANB2",
        ],
        "S": [
            "ASF1B",
            "CALM2",
            "CDC45",
            "CDCA5",
            "CENPM",
            "DHFR",
            "EZH2",
            "FEN1",
            "HIST1H2AC",
            "HIST1H4C",
            "NEAT1",
            "PKMYT1",
            "PRIM1",
            "RFC2",
            "RPA2",
            "RRM2",
            "RSRC2",
            "SRSF5",
            "SVIP",
            "TOP2A",
            "TYMS",
            "UBE2T",
            "ZWINT",
        ],
        "G2-M": [
            "AURKB",
            "BUB3",
            "CCNA2",
            "CCNF",
            "CDCA2",
            "CDCA3",
            "CDCA8",
            "CDK1",
            "CKAP2",
            "DCAF7",
            "HMGB2",
            "HN1",
            "KIF5B",
            "KIF20B",
            "KIF22",
            "KIF23",
            "KIFC1",
            "KPNA2",
            "LBR",
            "MAD2L1",
            "MALAT1",
            "MND1",
            "NDC80",
            "NUCKS1",
            "NUSAP1",
            "PIF1",
            "PSMD11",
            "PSRC1",
            "SMC4",
            "TIMP1",
            "TMEM99",
            "TOP2A",
            "TUBB",
            "TUBB4B",
            "VPS25",
        ],
        "M": [
            "ANP32B",
            "ANP32E",
            "ARL6IP1",
            "AURKA",
            "BIRC5",
            "BUB1",
            "CCNA2",
            "CCNB2",
            "CDC20",
            "CDC27",
            "CDC42EP1",
            "CDCA3",
            "CENPA",
            "CENPE",
            "CENPF",
            "CKAP2",
            "CKAP5",
            "CKS1B",
            "CKS2",
            "DEPDC1",
            "DLGAP5",
            "DNAJA1",
            "DNAJB1",
            "GRK6",
            "GTSE1",
            "HMG20B",
            "HMGB3",
            "HMMR",
            "HN1",
            "HSPA8",
            "KIF2C",
            "KIF5B",
            "KIF20B",
            "LBR",
            "MKI67",
            "MZT1",
            "NUF2",
            "NUSAP1",
            "PBK",
            "PLK1",
            "PRR11",
            "PSMG3",
            "PWP1",
            "RAD51C",
            "RBM8A",
            "RNF126",
            "RNPS1",
            "RRP1",
            "SFPQ",
            "SGOL2",
            "SMARCB1",
            "SRSF3",
            "TACC3",
            "THRAP3",
            "TPX2",
            "TUBB4B",
            "UBE2D3",
            "USP16",
            "WIBG",
            "YWHAH",
            "ZNF207",
        ],
        "M-G1": [
            "AMD1",
            "ANP32E",
            "CBX3",
            "CDC42",
            "CNIH4",
            "CWC15",
            "DKC1",
            "DNAJB6",
            "DYNLL1",
            "EIF4E",
            "FXR1",
            "GRPEL1",
            "GSPT1",
            "HMG20B",
            "HSPA8",
            "ILF2",
            "KIF5B",
            "KPNB1",
            "LARP1",
            "LYAR",
            "MORF4L2",
            "MRPL19",
            "MRPS2",
            "MRPS18B",
            "NUCKS1",
            "PRC1",
            "PTMS",
            "PTTG1",
            "RAN",
            "RHEB",
            "RPL13A",
            "SRSF3",
            "SYNCRIP",
            "TAF9",
            "TMEM138",
            "TOP1",
            "TROAP",
            "UBE2D3",
            "ZNF593",
        ],
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate Norman cells with cell-cycle scores using the original "
            "Perturbseq_GI marker-refinement and phase-calling logic."
        )
    )
    parser.add_argument("input_h5ad", type=Path, help="Input AnnData from stage 2.")
    parser.add_argument("--output", type=Path, required=True, help="Output h5ad path.")
    parser.add_argument(
        "--target-column",
        default="target",
        help="obs column used to identify controls. Default: target",
    )
    parser.add_argument(
        "--control-label",
        default="control",
        help="Value in --target-column treated as control. Default: control",
    )
    parser.add_argument(
        "--gene-name-column",
        default="gene_name",
        help="var column containing human-readable gene names. Default: gene_name",
    )
    parser.add_argument(
        "--good-coverage-column",
        default="good_coverage",
        help="obs column for guide coverage QC. Default: good_coverage",
    )
    parser.add_argument(
        "--number-of-cells-column",
        default="number_of_cells",
        help="obs column for called guide multiplicity. Default: number_of_cells",
    )
    parser.add_argument(
        "--guide-identity-column",
        default="guide_identity",
        help="obs column containing raw guide labels. Default: guide_identity",
    )
    parser.add_argument(
        "--subset-single-cells",
        dest="subset_single_cells",
        action="store_true",
        help="Subset to official Norman single_cell cells before annotation. Default: on.",
    )
    parser.add_argument(
        "--keep-all-cells",
        dest="subset_single_cells",
        action="store_false",
        help="Keep all cells and only write annotations for the single_cell subset.",
    )
    parser.set_defaults(subset_single_cells=True)
    parser.add_argument(
        "--refine-threshold",
        type=float,
        default=0.3,
        help="Correlation threshold used to refine control-derived marker sets. Default: 0.3",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series) or str(series.dtype) == "boolean":
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    truthy = {"true", "1", "t", "yes"}
    return normalized.isin(truthy)


def build_single_cell_mask(
    obs: pd.DataFrame,
    *,
    good_coverage_column: str,
    number_of_cells_column: str,
    guide_identity_column: str,
) -> pd.Series:
    for column in [good_coverage_column, number_of_cells_column, guide_identity_column]:
        if column not in obs.columns:
            raise KeyError(f"Required obs column {column!r} is missing")

    good_coverage = coerce_bool_series(obs[good_coverage_column])
    number_of_cells = pd.to_numeric(obs[number_of_cells_column], errors="coerce")
    guide_identity = obs[guide_identity_column].astype("string")

    mask = (number_of_cells == 1) & good_coverage & (guide_identity != "*")
    return mask.fillna(False).astype(bool).rename("single_cell")


def require_gene_name_column(adata: ad.AnnData, gene_name_column: str) -> pd.Series:
    if gene_name_column not in adata.var.columns:
        raise KeyError(f"Required var column {gene_name_column!r} is missing")
    return adata.var[gene_name_column].astype("string")


def get_gene_positions(adata: ad.AnnData, gene_names: list[str], gene_name_column: str) -> np.ndarray:
    var_gene_names = require_gene_name_column(adata, gene_name_column)
    return np.flatnonzero(var_gene_names.isin(gene_names).to_numpy())


def extract_gene_name_matrix(
    adata: ad.AnnData,
    gene_names: list[str],
    *,
    gene_name_column: str,
) -> pd.DataFrame:
    positions = get_gene_positions(adata, gene_names, gene_name_column)
    if positions.size == 0:
        return pd.DataFrame(index=adata.obs_names)

    matrix = slice_gene_matrix(adata.X, positions.tolist(), dtype=np.float64)
    columns = adata.var.iloc[positions][gene_name_column].astype(str).to_list()
    return pd.DataFrame(matrix, index=adata.obs_names, columns=columns)


def group_corr(
    adata: ad.AnnData,
    gene_list: list[str],
    *,
    gene_name_column: str,
) -> pd.Series:
    expression_matrix = extract_gene_name_matrix(
        adata,
        gene_list,
        gene_name_column=gene_name_column,
    )
    if expression_matrix.empty:
        return pd.Series(dtype=np.float64)
    expression_matrix["total"] = expression_matrix.mean(axis=1)
    return expression_matrix.corr()["total"].iloc[:-1]


def refine_gene_list(
    adata: ad.AnnData,
    gene_list: list[str],
    threshold: float,
    *,
    gene_name_column: str,
) -> tuple[list[str], pd.Series]:
    corrs = group_corr(adata, gene_list, gene_name_column=gene_name_column)
    if corrs.empty:
        return [], corrs
    refined = corrs[corrs >= threshold].index.astype(str).tolist()
    return refined, corrs.sort_index()


def group_score(
    adata: ad.AnnData,
    gene_list: list[str],
    *,
    gene_name_column: str,
) -> pd.Series:
    expression_matrix = extract_gene_name_matrix(
        adata,
        gene_list,
        gene_name_column=gene_name_column,
    )
    if expression_matrix.empty:
        return pd.Series(np.nan, index=adata.obs_names, dtype=np.float64)
    scores = np.log2(expression_matrix + 1.0).sum(axis=1)
    std = float(scores.std())
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(np.nan, index=adata.obs_names, dtype=np.float64)
    return ((scores - scores.mean()) / std).astype(np.float64)


def batch_group_score(
    adata: ad.AnnData,
    gene_lists: OrderedDict[str, list[str]],
    *,
    gene_name_column: str,
) -> OrderedDict[str, pd.Series]:
    out: OrderedDict[str, pd.Series] = OrderedDict()
    for phase, genes in gene_lists.items():
        out[phase] = group_score(adata, genes, gene_name_column=gene_name_column)
    return out


def get_cell_phase_genes(
    control_adata: ad.AnnData,
    *,
    gene_name_column: str,
    refine: bool,
    threshold: float,
) -> tuple[OrderedDict[str, list[str]], dict[str, dict[str, float]]]:
    cell_phase_genes = OrderedDict(
        (phase, genes.copy()) for phase, genes in DEFAULT_CELL_PHASE_GENES.items()
    )
    refinement_corrs: dict[str, dict[str, float]] = {}

    if refine:
        for phase in list(cell_phase_genes.keys()):
            refined, corrs = refine_gene_list(
                control_adata,
                cell_phase_genes[phase],
                threshold,
                gene_name_column=gene_name_column,
            )
            cell_phase_genes[phase] = refined
            refinement_corrs[phase] = {str(k): float(v) for k, v in corrs.items()}

    return cell_phase_genes, refinement_corrs


def get_cell_phase_scores(
    adata: ad.AnnData,
    gene_list: OrderedDict[str, list[str]],
    *,
    gene_name_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase_scores = pd.DataFrame(
        batch_group_score(adata, gene_list, gene_name_column=gene_name_column),
        index=adata.obs_names,
    )

    normalized_phase_scores = phase_scores.sub(phase_scores.mean(axis=1), axis=0).div(
        phase_scores.std(axis=1), axis=0
    )

    corr_input = normalized_phase_scores.transpose().copy()
    corr_input["G1-S"] = [1, 0, 0, 0, 0]
    corr_input["S"] = [0, 1, 0, 0, 0]
    corr_input["G2-M"] = [0, 0, 1, 0, 0]
    corr_input["M"] = [0, 0, 0, 1, 0]
    corr_input["M-G1"] = [0, 0, 0, 0, 1]

    cell_cycle_scores = corr_input.corr().iloc[-len(PHASE_LIST) :].transpose().iloc[:-len(PHASE_LIST)]

    cell_cycle_scores["cell_cycle_phase"] = cell_cycle_scores.idxmax(axis=1)
    cell_cycle_scores["cell_cycle_phase"] = pd.Categorical(
        cell_cycle_scores["cell_cycle_phase"],
        categories=PHASE_LIST,
        ordered=True,
    )

    def progress_ratio(row: pd.Series) -> float:
        phase = row["cell_cycle_phase"]
        if pd.isna(phase):
            return np.nan
        ind = PHASE_LIST.index(str(phase))
        return float(row[PHASE_LIST[(ind - 1) % len(PHASE_LIST)]] - row[PHASE_LIST[(ind + 1) % len(PHASE_LIST)]])

    cell_cycle_scores["cell_cycle_progress"] = cell_cycle_scores.apply(progress_ratio, axis=1)
    cell_cycle_scores.sort_values(
        ["cell_cycle_phase", "cell_cycle_progress"],
        ascending=[True, False],
        inplace=True,
    )

    order = cell_cycle_scores.groupby("cell_cycle_phase", observed=False).cumcount()
    group_sizes = cell_cycle_scores.groupby(
        "cell_cycle_phase", observed=False
    )["cell_cycle_phase"].transform("size")
    denom = (group_sizes - 1).replace(0, np.nan)
    cell_cycle_scores["cell_cycle_order"] = order / denom

    return phase_scores, normalized_phase_scores, cell_cycle_scores


def add_cell_cycle_annotations(
    adata: ad.AnnData,
    scored_obs_names: pd.Index,
    *,
    raw_group_scores: pd.DataFrame,
    normalized_group_scores: pd.DataFrame,
    final_scores: pd.DataFrame,
) -> ad.AnnData:
    obs = adata.obs.copy()

    for phase in PHASE_LIST:
        obs[f"cell_cycle_group_score_{phase}"] = np.nan
        obs[f"cell_cycle_normalized_group_score_{phase}"] = np.nan
        obs[phase] = np.nan

    obs["cell_cycle_phase"] = pd.Series(pd.NA, index=obs.index, dtype="string")
    obs["cell_cycle_progress"] = np.nan
    obs["cell_cycle_order"] = np.nan
    obs["cell_cycle_scored"] = False

    for phase in PHASE_LIST:
        obs.loc[scored_obs_names, f"cell_cycle_group_score_{phase}"] = raw_group_scores.loc[scored_obs_names, phase]
        obs.loc[
            scored_obs_names,
            f"cell_cycle_normalized_group_score_{phase}",
        ] = normalized_group_scores.loc[scored_obs_names, phase]
        obs.loc[scored_obs_names, phase] = final_scores.loc[scored_obs_names, phase]

    obs.loc[scored_obs_names, "cell_cycle_phase"] = (
        final_scores.loc[scored_obs_names, "cell_cycle_phase"].astype("string")
    )
    obs.loc[scored_obs_names, "cell_cycle_progress"] = final_scores.loc[
        scored_obs_names, "cell_cycle_progress"
    ]
    obs.loc[scored_obs_names, "cell_cycle_order"] = final_scores.loc[scored_obs_names, "cell_cycle_order"]
    obs.loc[scored_obs_names, "cell_cycle_scored"] = True

    obs["cell_cycle_phase"] = pd.Categorical(obs["cell_cycle_phase"], categories=PHASE_LIST, ordered=True)
    adata.obs = obs
    return adata


def build_output_uns(
    adata: ad.AnnData,
    *,
    target_column: str,
    control_label: str,
    gene_name_column: str,
    refine_threshold: float,
    subset_single_cells: bool,
    raw_marker_sets: OrderedDict[str, list[str]],
    refined_marker_sets: OrderedDict[str, list[str]],
    refinement_correlations: dict[str, dict[str, float]],
    n_control_cells: int,
) -> None:
    adata.uns["cell_cycle_annotation"] = {
        "source": "Perturbseq_GI cell_cycle.py migrated to AnnData",
        "target_column": target_column,
        "control_label": control_label,
        "gene_name_column": gene_name_column,
        "single_cell_rule": (
            "(number_of_cells == 1) & good_coverage & (guide_identity != '*')"
        ),
        "subset_single_cells": bool(subset_single_cells),
        "refine_threshold": float(refine_threshold),
        "scoring_source": "raw counts in adata.X",
        "n_obs_output": int(adata.n_obs),
        "n_control_cells": int(n_control_cells),
        "phase_list": PHASE_LIST,
        "raw_marker_sets": {phase: genes for phase, genes in raw_marker_sets.items()},
        "refined_marker_sets": {phase: genes for phase, genes in refined_marker_sets.items()},
        "refinement_correlations": refinement_correlations,
        "output_obs_columns": (
            ["single_cell", "cell_cycle_phase", "cell_cycle_progress", "cell_cycle_order", "cell_cycle_scored"]
            + PHASE_LIST
            + [f"cell_cycle_group_score_{phase}" for phase in PHASE_LIST]
            + [f"cell_cycle_normalized_group_score_{phase}" for phase in PHASE_LIST]
        ),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input_h5ad.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Use --force to overwrite.")

    adata = ad.read_h5ad(input_path)

    single_cell = build_single_cell_mask(
        adata.obs,
        good_coverage_column=args.good_coverage_column,
        number_of_cells_column=args.number_of_cells_column,
        guide_identity_column=args.guide_identity_column,
    )
    adata.obs["single_cell"] = single_cell

    if args.target_column not in adata.obs.columns:
        raise KeyError(f"Required obs column {args.target_column!r} is missing")

    analysis_adata = adata[single_cell].copy() if args.subset_single_cells else adata[single_cell].copy()
    if analysis_adata.n_obs == 0:
        raise ValueError("No single_cell observations remain for cell-cycle annotation.")

    control_mask = analysis_adata.obs[args.target_column].astype("string") == args.control_label
    if int(control_mask.sum()) == 0:
        raise ValueError(
            f"No control cells found in {args.target_column!r} matching {args.control_label!r} "
            "within the single_cell subset."
        )
    control_adata = analysis_adata[control_mask].copy()

    refined_marker_sets, refinement_correlations = get_cell_phase_genes(
        control_adata,
        gene_name_column=args.gene_name_column,
        refine=True,
        threshold=args.refine_threshold,
    )
    raw_group_scores, normalized_group_scores, final_scores = get_cell_phase_scores(
        analysis_adata,
        refined_marker_sets,
        gene_name_column=args.gene_name_column,
    )

    if args.subset_single_cells:
        output_adata = analysis_adata.copy()
        output_adata = add_cell_cycle_annotations(
            output_adata,
            output_adata.obs_names,
            raw_group_scores=raw_group_scores,
            normalized_group_scores=normalized_group_scores,
            final_scores=final_scores,
        )
    else:
        output_adata = adata.copy()
        output_adata = add_cell_cycle_annotations(
            output_adata,
            analysis_adata.obs_names,
            raw_group_scores=raw_group_scores,
            normalized_group_scores=normalized_group_scores,
            final_scores=final_scores,
        )

    build_output_uns(
        output_adata,
        target_column=args.target_column,
        control_label=args.control_label,
        gene_name_column=args.gene_name_column,
        refine_threshold=args.refine_threshold,
        subset_single_cells=args.subset_single_cells,
        raw_marker_sets=DEFAULT_CELL_PHASE_GENES,
        refined_marker_sets=refined_marker_sets,
        refinement_correlations=refinement_correlations,
        n_control_cells=int(control_adata.n_obs),
    )

    write_h5ad(output_adata, output_path)
    print(
        f"Wrote {output_adata.n_obs:,} cells with cell-cycle annotations to {output_path} "
        f"(single_cell analyzed: {analysis_adata.n_obs:,}, controls used: {control_adata.n_obs:,})"
    )


if __name__ == "__main__":
    main()
