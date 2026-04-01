from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Iterable

import typer

from prism.io import GeneListSpec, read_gene_list_spec, write_gene_list_spec, write_gene_list_text

from .common import console


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
        exact: Iterable[str] = (),
        prefixes: Iterable[str] = (),
        patterns: Iterable[str] = (),
    ) -> NuisanceRuleSet:
        return cls(
            exact=frozenset(exact),
            prefixes=tuple(prefixes),
            patterns=tuple(re.compile(pattern) for pattern in patterns),
            name=name,
        )

    @classmethod
    def from_dict(cls, data: dict[str, object], *, name: str = "custom") -> NuisanceRuleSet:
        return cls.build(
            name=str(data.get("name", name)),
            exact=tuple(str(value) for value in data.get("exact", ()) if str(value)),
            prefixes=tuple(
                str(value) for value in data.get("prefixes", ()) if str(value)
            ),
            patterns=tuple(
                str(value) for value in data.get("patterns", ()) if str(value)
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
SUPPORTED_FILTER_SPECIES = tuple(sorted(list(BUILTIN_SPECIES) + ["none"]))


def load_config_rules(path: Path) -> NuisanceRuleSet:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load YAML configs. Install with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
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


def _resolve_rules(
    *,
    species: str,
    config_path: Path | None,
    config_only: bool,
) -> NuisanceRuleSet:
    species_resolved = species.strip().lower()
    if config_only and config_path is None:
        raise ValueError("--config-only requires --config")
    if config_only:
        assert config_path is not None
        return load_config_rules(config_path)
    if species_resolved == "none":
        rules = NuisanceRuleSet.build(name="none")
    elif species_resolved in BUILTIN_SPECIES:
        rules = BUILTIN_SPECIES[species_resolved]
    else:
        raise ValueError(
            "--species must be one of: "
            + ", ".join(SUPPORTED_FILTER_SPECIES)
        )
    if config_path is not None:
        rules = rules.merge(load_config_rules(config_path))
    return rules


def _filter_spec(
    spec: GeneListSpec,
    *,
    rules: NuisanceRuleSet,
) -> tuple[GeneListSpec, GeneListSpec]:
    score_values = list(spec.scores) if spec.scores else [None] * len(spec.gene_names)
    kept_names: list[str] = []
    kept_scores: list[float] = []
    removed_names: list[str] = []
    removed_scores: list[float] = []
    for gene, score in zip(spec.gene_names, score_values, strict=True):
        if rules.is_nuisance(gene):
            removed_names.append(gene)
            if score is not None:
                removed_scores.append(float(score))
        else:
            kept_names.append(gene)
            if score is not None:
                kept_scores.append(float(score))
    shared_metadata = dict(spec.metadata)
    shared_metadata.update(
        {
            "filter_rule_set": rules.name,
            "score_order": spec.metadata.get("score_order"),
        }
    )
    kept = GeneListSpec(
        gene_names=kept_names,
        scores=kept_scores,
        source_path=spec.source_path,
        method="nuisance-filter",
        metadata={
            **shared_metadata,
            "n_input_genes": len(spec.gene_names),
            "n_kept_genes": len(kept_names),
            "n_removed_genes": len(removed_names),
        },
    )
    removed = GeneListSpec(
        gene_names=removed_names,
        scores=removed_scores,
        source_path=spec.source_path,
        method="nuisance-filter:removed",
        metadata={
            **shared_metadata,
            "n_input_genes": len(spec.gene_names),
            "n_kept_genes": len(kept_names),
            "n_removed_genes": len(removed_names),
        },
    )
    return kept, removed


def filter_genes_command(
    input_genes: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input gene-list text or JSON file."
    ),
    output_genes: Path = typer.Option(
        ..., "--output-genes", "-o", help="Output filtered gene list."
    ),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        help="Optional structured JSON for filtered genes.",
    ),
    removed_genes: Path | None = typer.Option(
        None,
        "--removed-genes",
        help="Optional text file for removed genes.",
    ),
    removed_json: Path | None = typer.Option(
        None,
        "--removed-json",
        help="Optional structured JSON for removed genes.",
    ),
    species: str = typer.Option(
        "human",
        help="Built-in species rule-set: " + ", ".join(SUPPORTED_FILTER_SPECIES) + ".",
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        help="Optional JSON/YAML rule config.",
    ),
    config_only: bool = typer.Option(
        False,
        help="Ignore built-in species rules and use only --config.",
    ),
    dry_run: bool = typer.Option(False, help="Preview results without writing files."),
) -> int:
    input_genes = input_genes.expanduser().resolve()
    output_genes = output_genes.expanduser().resolve()
    output_json = None if output_json is None else output_json.expanduser().resolve()
    removed_genes = (
        None if removed_genes is None else removed_genes.expanduser().resolve()
    )
    removed_json = (
        None if removed_json is None else removed_json.expanduser().resolve()
    )
    config_path = None if config_path is None else config_path.expanduser().resolve()

    rules = _resolve_rules(
        species=species,
        config_path=config_path,
        config_only=config_only,
    )
    spec = read_gene_list_spec(input_genes)
    kept, removed = _filter_spec(spec, rules=rules)

    kept_text = list(kept.gene_names)
    removed_text = list(removed.gene_names)

    console.print(
        f"[bold cyan]Filter[/bold cyan] rules={rules.name} input={len(spec.gene_names)} kept={len(kept_text)} removed={len(removed_text)}"
    )
    if dry_run:
        return 0

    write_gene_list_text(output_genes, kept_text)
    if output_json is not None:
        write_gene_list_spec(output_json, kept)
    if removed_genes is not None:
        write_gene_list_text(removed_genes, removed_text)
    if removed_json is not None:
        write_gene_list_spec(removed_json, removed)

    console.print(f"[bold green]Saved[/bold green] {output_genes}")
    if output_json is not None:
        console.print(f"[bold green]Saved[/bold green] {output_json}")
    if removed_genes is not None:
        console.print(f"[bold green]Saved[/bold green] {removed_genes}")
    if removed_json is not None:
        console.print(f"[bold green]Saved[/bold green] {removed_json}")
    return 0


__all__ = ["filter_genes_command"]
