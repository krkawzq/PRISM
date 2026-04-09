from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
from rich.console import Console
from scipy import sparse

from prism.cli.common import (
    normalize_choice,
    print_key_value_table,
    resolve_prior_source,
)
from prism.io import (
    read_gene_list,
    write_gene_list,
)
from prism.model import load_checkpoint

EPS = 1e-12
SIGNAL_LAYER = "signal"
SUPPORTED_HVG_FLAVORS = ("seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper")
SUPPORTED_PRIOR_SOURCES = ("global", "label")
SUPPORTED_RANK_METHODS = (
    "hvg",
    "signal-hvg",
    "prior-entropy",
    "prior-entropy-reverse",
    "lognorm-variance",
    "lognorm-dispersion",
    "signal-variance",
    "signal-dispersion",
)
SUPPORTED_INTERSECTION_SORTS = ("first", "alpha")
SUPPORTED_MERGE_METHODS = ("rank-sum",)
SUPPORTED_GENE_SET_MODES = ("exact", "intersection", "union")
SIGNAL_RANK_METHODS = ("signal-hvg", "signal-variance", "signal-dispersion")
LOGNORM_RANK_METHODS = ("lognorm-variance", "lognorm-dispersion")
PRIOR_ENTROPY_METHODS = ("prior-entropy", "prior-entropy-reverse")


@dataclass(frozen=True, slots=True)
class RankingResult:
    gene_names: np.ndarray
    scores: np.ndarray
    descending: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        gene_names = np.asarray(self.gene_names)
        scores = np.asarray(self.scores, dtype=np.float64)
        if gene_names.ndim != 1:
            raise ValueError(f"gene_names must be 1D, got shape={gene_names.shape}")
        if scores.ndim != 1:
            raise ValueError(f"scores must be 1D, got shape={scores.shape}")
        if gene_names.shape[0] == 0:
            raise ValueError("gene_names cannot be empty")
        if scores.shape[0] != gene_names.shape[0]:
            raise ValueError(
                "gene_names and scores must have the same length, "
                f"got {gene_names.shape[0]} != {scores.shape[0]}"
            )
        object.__setattr__(self, "gene_names", gene_names)
        object.__setattr__(self, "scores", np.nan_to_num(scores, copy=False))
        object.__setattr__(self, "descending", bool(self.descending))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _iter_config_values(value: object) -> tuple[object, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return ()


@dataclass(frozen=True, slots=True)
class NuisanceRuleSet:
    exact: frozenset[str] = field(default_factory=frozenset)
    prefixes: tuple[str, ...] = field(default_factory=tuple)
    patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)
    name: str = "unnamed"

    @classmethod
    def build(
        cls,
        *,
        name: str = "unnamed",
        exact: tuple[str, ...] | list[str] = (),
        prefixes: tuple[str, ...] | list[str] = (),
        patterns: tuple[str, ...] | list[str] = (),
    ) -> NuisanceRuleSet:
        return cls(
            exact=frozenset(exact),
            prefixes=tuple(prefixes),
            patterns=tuple(re.compile(pattern) for pattern in patterns),
            name=name,
        )

    @classmethod
    def from_dict(
        cls, data: dict[str, object], *, name: str = "custom"
    ) -> NuisanceRuleSet:
        return cls.build(
            name=str(data.get("name", name)),
            exact=tuple(
                str(value)
                for value in _iter_config_values(data.get("exact"))
                if str(value)
            ),
            prefixes=tuple(
                str(value)
                for value in _iter_config_values(data.get("prefixes"))
                if str(value)
            ),
            patterns=tuple(
                str(value)
                for value in _iter_config_values(data.get("patterns"))
                if str(value)
            ),
        )

    def is_nuisance(self, gene: str) -> bool:
        if gene in self.exact:
            return True
        if self.prefixes and gene.startswith(self.prefixes):
            return True
        return any(pattern.fullmatch(gene) for pattern in self.patterns)

    def merge(self, other: NuisanceRuleSet) -> NuisanceRuleSet:
        return NuisanceRuleSet(
            exact=self.exact | other.exact,
            prefixes=self.prefixes + other.prefixes,
            patterns=self.patterns + other.patterns,
            name=f"{self.name}+{other.name}",
        )


def _build_human() -> NuisanceRuleSet:
    return NuisanceRuleSet.build(
        name="human",
        prefixes=("MT-", "RPL", "RPS", "MRPL", "MRPS"),
        patterns=(
            r"HB[A-Z]\d?",
            r"HSP(?:90|A|B|D|E|H)\w*",
            r"DNAJ[A-C]\d+",
            r"FOS[BL]?",
            r"JUN[BD]?",
            r"EGR[1-4]",
            r"IER[1-5][A-Z]?",
            r"DUSP[1-9]\d?",
        ),
        exact=("MALAT1", "NEAT1", "XIST", "FTL", "FTH1"),
    )


def _build_mouse() -> NuisanceRuleSet:
    return NuisanceRuleSet.build(
        name="mouse",
        prefixes=("mt-", "Rpl", "Rps", "Mrpl", "Mrps"),
        patterns=(
            r"Hb[a-z].*",
            r"Hsp(?:90|a|b|d|e|h)\w*",
            r"Dnaj[a-c]\d+",
            r"Fos[bl]?",
            r"Jun[bd]?",
            r"Egr[1-4]",
            r"Ier[1-5][a-z]?",
            r"Dusp[1-9]\d?",
        ),
        exact=("Malat1", "Neat1", "Xist", "Ftl1", "Fth1"),
    )


def _build_ecoli() -> NuisanceRuleSet:
    return NuisanceRuleSet.build(
        name="ecoli",
        prefixes=("rpl", "rps", "rpm"),
        patterns=(
            r"rr[slf][A-H]",
            r"tuf[AB]?",
            r"fus[A-Z]?",
            r"gro[ELS]+",
            r"dna[KJ]\w*",
            r"clp[ABPX]\w*",
            r"csp[A-I]",
        ),
        exact=("tsf", "infA", "infB", "infC", "ssrA"),
    )


def _build_bsub() -> NuisanceRuleSet:
    return NuisanceRuleSet.build(
        name="bsub",
        prefixes=("rpl", "rps", "rpm"),
        patterns=(
            r"rrn[A-Z]",
            r"tuf[A-Z]?",
            r"fus[A-Z]?",
            r"groE[SL]\w*",
            r"dna[KJ]\w*",
            r"clp[CEPQXY]\w*",
            r"csp[A-Z]",
            r"sig[A-Z]",
            r"hbs\w*",
        ),
        exact=("tsf", "infA", "infB", "infC", "ssrA"),
    )


BUILTIN_SPECIES: dict[str, NuisanceRuleSet] = {
    "human": _build_human(),
    "mouse": _build_mouse(),
    "ecoli": _build_ecoli(),
    "bsub": _build_bsub(),
}
SUPPORTED_FILTER_SPECIES = tuple(sorted((*BUILTIN_SPECIES.keys(), "none")))

console = Console()


def normalize_rank_method(method: str) -> str:
    return normalize_choice(
        method, supported=SUPPORTED_RANK_METHODS, option_name="--method"
    )


def normalize_hvg_flavor(hvg_flavor: str) -> str:
    return normalize_choice(
        hvg_flavor, supported=SUPPORTED_HVG_FLAVORS, option_name="--hvg-flavor"
    )


def normalize_prior_source(prior_source: str) -> str:
    return resolve_prior_source(prior_source)


def normalize_intersection_sort(sort: str) -> str:
    return normalize_choice(
        sort,
        supported=SUPPORTED_INTERSECTION_SORTS,
        option_name="--sort",
    )


def normalize_merge_method(method: str) -> str:
    return normalize_choice(
        method,
        supported=SUPPORTED_MERGE_METHODS,
        option_name="--method",
    )


def normalize_gene_set_mode(mode: str) -> str:
    return normalize_choice(
        mode,
        supported=SUPPORTED_GENE_SET_MODES,
        option_name="--gene-set-mode",
    )


def normalize_filter_species(species: str) -> str:
    return normalize_choice(
        species,
        supported=SUPPORTED_FILTER_SPECIES,
        option_name="--species",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_var_names(path: Path) -> list[str]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        return [str(name) for name in adata.var_names.tolist()]
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def maybe_subsample_adata(
    adata: ad.AnnData, *, max_cells: int | None, random_seed: int
) -> tuple[ad.AnnData, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "n_total_cells": int(adata.n_obs),
        "n_used_cells": int(adata.n_obs),
        "cell_sampling_applied": False,
        "max_cells": None if max_cells is None else int(max_cells),
        "random_seed": int(random_seed),
    }
    if max_cells is None or adata.n_obs <= max_cells:
        return adata, metadata
    rng = np.random.default_rng(random_seed)
    indices = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
    sampled = adata[indices].copy()
    metadata.update({"n_used_cells": int(sampled.n_obs), "cell_sampling_applied": True})
    return sampled, metadata


def compute_hvg_ranking_from_adata(
    adata: ad.AnnData, *, flavor: str
) -> tuple[np.ndarray, np.ndarray]:
    import scanpy as sc

    try:
        sc.pp.highly_variable_genes(adata, flavor=cast(Any, flavor), inplace=True)
    except ImportError:
        if flavor not in {"seurat_v3", "seurat_v3_paper"}:
            raise
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor="seurat", inplace=True)
    if "highly_variable_rank" in adata.var:
        score = -np.nan_to_num(
            np.asarray(adata.var["highly_variable_rank"], dtype=np.float64), nan=np.inf
        )
    elif "dispersions_norm" in adata.var:
        score = np.asarray(adata.var["dispersions_norm"], dtype=np.float64)
    elif "variances_norm" in adata.var:
        score = np.asarray(adata.var["variances_norm"], dtype=np.float64)
    else:
        raise ValueError("scanpy did not produce an HVG ranking field")
    return np.asarray(adata.var_names), np.nan_to_num(
        score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf
    )


def compute_lognorm_ranking(
    adata: ad.AnnData, *, method: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    matrix = adata.X
    if matrix is None:
        raise ValueError("input h5ad has empty X")
    counts = (
        np.asarray(cast(Any, matrix).toarray(), dtype=np.float64)
        if sparse.issparse(matrix)
        else np.asarray(matrix, dtype=np.float64)
    )
    totals = counts.sum(axis=1)
    target = float(np.median(totals))
    values = np.log1p(counts * (target / np.maximum(totals, 1.0))[:, None])
    mean = np.mean(values, axis=0)
    var = np.var(values, axis=0)
    if method == "lognorm-variance":
        score = var
        definition = "variance(log1p(normalize_total(X)))"
    else:
        score = var / np.maximum(mean, EPS)
        definition = (
            "variance(log1p(normalize_total(X))) / mean(log1p(normalize_total(X)))"
        )
    return (
        np.asarray(adata.var_names),
        np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf),
        {"normalization_target": target, "score_definition": definition},
    )


def compute_signal_ranking(
    adata: ad.AnnData, *, method: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if SIGNAL_LAYER not in adata.layers:
        raise KeyError(f"input file is missing required layer: {SIGNAL_LAYER!r}")
    layer = adata.layers[SIGNAL_LAYER]
    values = (
        np.asarray(cast(Any, layer).toarray(), dtype=np.float64)
        if sparse.issparse(layer)
        else np.asarray(layer, dtype=np.float64)
    )
    mean = np.mean(values, axis=0)
    var = np.var(values, axis=0)
    if method == "signal-variance":
        score = var
        definition = "variance(signal)"
    elif method == "signal-dispersion":
        score = var / np.maximum(mean, EPS)
        definition = "variance(signal) / mean(signal)"
    elif method == "signal-hvg":
        signal = np.clip(
            np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None
        )
        signal_adata = ad.AnnData(
            X=np.log1p(signal),
            obs=cast(Any, adata.obs.copy()),
            var=cast(Any, adata.var.copy()),
        )
        gene_names, score = compute_hvg_ranking_from_adata(
            signal_adata, flavor="seurat"
        )
        return (
            gene_names,
            score,
            {
                "layer": SIGNAL_LAYER,
                "hvg_flavor": "seurat",
                "score_definition": "HVG rank over log1p(signal)",
            },
        )
    else:
        raise ValueError(f"unsupported signal method: {method}")
    return (
        np.asarray(adata.var_names),
        np.nan_to_num(score, nan=-np.inf, posinf=-np.inf, neginf=-np.inf),
        {"layer": SIGNAL_LAYER, "score_definition": definition},
    )


def checkpoint_prior_entropy_scores(
    input_path: Path, *, prior_source: str, label: str | None
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = load_checkpoint(input_path)
    if prior_source == "global":
        priors = checkpoint.get_prior().as_gene_specific()
    elif prior_source == "label":
        if label is None or not label.strip():
            raise ValueError("--label is required when --prior-source label")
        priors = checkpoint.get_prior(label).as_gene_specific()
    else:
        raise ValueError(f"unsupported prior source: {prior_source}")
    probabilities = np.asarray(priors.prior_probabilities, dtype=np.float64)
    scores = -(probabilities * np.log(np.clip(probabilities, EPS, None))).sum(axis=-1)
    return np.asarray(priors.gene_names), scores


def compute_ranking(
    input_path: Path,
    *,
    method: str,
    hvg_flavor: str,
    prior_source: str,
    label: str | None,
    max_cells: int | None,
    random_seed: int,
) -> RankingResult:
    method_resolved = normalize_rank_method(method)
    prior_source_resolved = resolve_prior_source(prior_source)
    hvg_flavor_resolved = normalize_hvg_flavor(hvg_flavor)
    if method_resolved in PRIOR_ENTROPY_METHODS:
        gene_names, scores = checkpoint_prior_entropy_scores(
            input_path, prior_source=prior_source_resolved, label=label
        )
        return RankingResult(
            gene_names=gene_names,
            scores=np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0),
            descending=(method_resolved == "prior-entropy"),
            metadata={
                "score_definition": "prior entropy",
                "prior_source": prior_source_resolved,
                "label": label,
            },
        )
    adata = ad.read_h5ad(input_path)
    adata, sampling_metadata = maybe_subsample_adata(
        adata, max_cells=max_cells, random_seed=random_seed
    )
    if method_resolved == "hvg":
        gene_names, scores = compute_hvg_ranking_from_adata(
            adata, flavor=hvg_flavor_resolved
        )
        metadata = {"hvg_flavor": hvg_flavor_resolved}
    elif method_resolved in LOGNORM_RANK_METHODS:
        gene_names, scores, metadata = compute_lognorm_ranking(
            adata, method=method_resolved
        )
    else:
        gene_names, scores, metadata = compute_signal_ranking(
            adata, method=method_resolved
        )
    metadata = dict(metadata)
    metadata.update(sampling_metadata)
    return RankingResult(
        gene_names=np.asarray(gene_names),
        scores=np.asarray(scores, dtype=np.float64),
        descending=True,
        metadata=metadata,
    )


def filter_gene_scores(
    gene_names: np.ndarray, scores: np.ndarray, *, restrict_genes: list[str] | None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {"n_input_genes": int(gene_names.shape[0])}
    if restrict_genes is None:
        metadata.update(
            {
                "n_requested_restrict_genes": None,
                "n_missing_restrict_genes": 0,
                "missing_restrict_genes": [],
                "n_ranked_genes": int(gene_names.shape[0]),
            }
        )
        return gene_names, scores, metadata
    requested: list[str] = []
    seen_requested: set[str] = set()
    for gene in restrict_genes:
        if gene not in seen_requested:
            requested.append(gene)
            seen_requested.add(gene)
    gene_to_idx = {str(gene): idx for idx, gene in enumerate(gene_names.tolist())}
    selected_indices: list[int] = []
    missing: list[str] = []
    for gene in requested:
        idx = gene_to_idx.get(gene)
        if idx is None:
            missing.append(gene)
        else:
            selected_indices.append(idx)
    if not selected_indices:
        raise ValueError("no restricted genes were found in the input data")
    index_array = np.asarray(selected_indices, dtype=np.int64)
    filtered_gene_names = np.asarray(gene_names[index_array], dtype=gene_names.dtype)
    filtered_scores = np.asarray(scores[index_array], dtype=np.float64)
    metadata.update(
        {
            "n_requested_restrict_genes": int(len(requested)),
            "n_missing_restrict_genes": int(len(missing)),
            "missing_restrict_genes": missing,
            "n_ranked_genes": int(filtered_gene_names.shape[0]),
        }
    )
    return filtered_gene_names, filtered_scores, metadata


def rank_gene_scores(
    gene_names: np.ndarray, scores: np.ndarray, *, descending: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    return order, gene_names[order], scores[order]


def resolve_filter_rules(
    *, species: str, config_path: Path | None, config_only: bool
) -> NuisanceRuleSet:
    species_resolved = normalize_filter_species(species)
    if config_only and config_path is None:
        raise ValueError("--config-only requires --config")
    if config_only:
        assert config_path is not None
        return load_config_rules(config_path)
    if species_resolved == "none":
        rules = NuisanceRuleSet.build(name="none")
    else:
        rules = BUILTIN_SPECIES[species_resolved]
    if config_path is not None:
        rules = rules.merge(load_config_rules(config_path))
    return rules


def load_config_rules(path: Path) -> NuisanceRuleSet:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load YAML configs. Install with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "Could not parse config as JSON and PyYAML is not installed."
                ) from exc
            data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping at top level")
    return NuisanceRuleSet.from_dict(data, name=path.stem)


def filter_gene_list(
    genes: list[str], *, rules: NuisanceRuleSet
) -> tuple[list[str], list[str], dict[str, Any]]:
    kept: list[str] = []
    removed: list[str] = []
    for gene in genes:
        if rules.is_nuisance(gene):
            removed.append(gene)
        else:
            kept.append(gene)
    metadata = {
        "rule_set": rules.name,
        "n_input_genes": int(len(genes)),
        "n_kept_genes": int(len(kept)),
        "n_removed_genes": int(len(removed)),
    }
    return kept, removed, metadata


def intersect_gene_lists(
    ordered_lists: list[list[str]], *, sort: str
) -> tuple[list[str], dict[str, Any]]:
    if len(ordered_lists) < 2:
        raise ValueError("intersect requires at least two datasets")
    sort_resolved = normalize_intersection_sort(sort)
    overlap = set(ordered_lists[0])
    for genes in ordered_lists[1:]:
        overlap &= set(genes)
    if sort_resolved == "first":
        ordered = [gene for gene in ordered_lists[0] if gene in overlap]
    else:
        ordered = sorted(overlap)
    metadata = {
        "sort": sort_resolved,
        "n_inputs": int(len(ordered_lists)),
        "n_overlap": int(len(ordered)),
        "input_sizes": [int(len(genes)) for genes in ordered_lists],
    }
    return ordered, metadata


def _resolve_gene_universe(inputs: list[list[str]], *, gene_set_mode: str) -> list[str]:
    if gene_set_mode == "exact":
        reference = set(inputs[0])
        for genes in inputs[1:]:
            current = set(genes)
            if current != reference:
                raise ValueError("gene set mismatch across inputs")
        return inputs[0]
    if gene_set_mode == "intersection":
        overlap = set(inputs[0])
        for genes in inputs[1:]:
            overlap &= set(genes)
        return [gene for gene in inputs[0] if gene in overlap]
    if gene_set_mode == "union":
        ordered: list[str] = []
        seen: set[str] = set()
        for genes in inputs:
            for gene in genes:
                if gene in seen:
                    continue
                seen.add(gene)
                ordered.append(gene)
        return ordered
    raise ValueError("unsupported gene_set_mode")


def merge_gene_lists(
    inputs: list[list[str]],
    *,
    method: str,
    gene_set_mode: str,
) -> tuple[list[str], dict[str, Any]]:
    if len(inputs) < 2:
        raise ValueError("merge requires at least two input gene lists")
    method_resolved = normalize_merge_method(method)
    gene_set_mode_resolved = normalize_gene_set_mode(gene_set_mode)
    merged_genes = _resolve_gene_universe(inputs, gene_set_mode=gene_set_mode_resolved)
    rank_maps = [{gene: idx for idx, gene in enumerate(genes)} for genes in inputs]
    rows: list[tuple[float, float, str]] = []
    for gene in merged_genes:
        ranks = [
            float(rank_map.get(gene, len(genes)))
            for rank_map, genes in zip(rank_maps, inputs, strict=True)
        ]
        rank_sum = float(sum(ranks))
        rank_mean = rank_sum / max(len(ranks), 1)
        rows.append((rank_sum, rank_mean, gene))
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    metadata = {
        "merge_method": method_resolved,
        "gene_set_mode": gene_set_mode_resolved,
        "n_inputs": int(len(inputs)),
        "n_output_genes": int(len(rows)),
    }
    return [gene for _, _, gene in rows], metadata


def subset_gene_list(
    genes: list[str],
    *,
    start: int,
    end: int | None,
    top_k: int | None,
    intersect_genes: list[str] | None,
    exclude_genes: list[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    original_genes = list(genes)
    if intersect_genes is not None:
        keep = set(intersect_genes)
        genes = [gene for gene in genes if gene in keep]
    if exclude_genes is not None:
        drop = set(exclude_genes)
        genes = [gene for gene in genes if gene not in drop]
    n_after_filtering = len(genes)
    resolved_end = len(genes) if end is None else min(end, len(genes))
    if resolved_end < start:
        raise ValueError("end must be >= start")
    genes = genes[start:resolved_end]
    if top_k is not None:
        genes = genes[:top_k]
    metadata = {
        "n_input_genes": int(len(original_genes)),
        "n_after_filtering": int(n_after_filtering),
        "start": int(start),
        "end": None if end is None else int(end),
        "top_k": None if top_k is None else int(top_k),
        "n_output_genes": int(len(genes)),
        "intersect_applied": intersect_genes is not None,
        "exclude_applied": exclude_genes is not None,
    }
    return genes, metadata


def print_gene_summary(title: str, **values: object) -> None:
    print_key_value_table(console, title=title, values=values)


__all__ = [
    "BUILTIN_SPECIES",
    "EPS",
    "LOGNORM_RANK_METHODS",
    "NuisanceRuleSet",
    "PRIOR_ENTROPY_METHODS",
    "RankingResult",
    "SIGNAL_LAYER",
    "SIGNAL_RANK_METHODS",
    "SUPPORTED_FILTER_SPECIES",
    "SUPPORTED_GENE_SET_MODES",
    "SUPPORTED_HVG_FLAVORS",
    "SUPPORTED_INTERSECTION_SORTS",
    "SUPPORTED_MERGE_METHODS",
    "SUPPORTED_PRIOR_SOURCES",
    "SUPPORTED_RANK_METHODS",
    "checkpoint_prior_entropy_scores",
    "compute_hvg_ranking_from_adata",
    "compute_lognorm_ranking",
    "compute_ranking",
    "compute_signal_ranking",
    "console",
    "filter_gene_list",
    "filter_gene_scores",
    "intersect_gene_lists",
    "load_config_rules",
    "load_var_names",
    "maybe_subsample_adata",
    "merge_gene_lists",
    "normalize_filter_species",
    "normalize_gene_set_mode",
    "normalize_hvg_flavor",
    "normalize_intersection_sort",
    "normalize_merge_method",
    "normalize_prior_source",
    "normalize_rank_method",
    "print_gene_summary",
    "rank_gene_scores",
    "read_gene_list",
    "resolve_filter_rules",
    "subset_gene_list",
    "write_gene_list",
    "write_json",
]
