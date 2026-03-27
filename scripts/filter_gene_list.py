#!/usr/bin/env python3
"""Filter nuisance genes from a ranked gene list.

Instead of hard-coding every nuisance gene, this script uses **rule-based
string matching** that generalises across species:

  1. *Prefix rules*      – e.g. "MT-", "Rpl", "rpl" …
  2. *Regex rules*        – arbitrary patterns compiled once
  3. *Exact-match rules*  – a small fallback set for names that defy patterns

Built-in rule-sets are provided for human, mouse, E. coli (K-12) and
B. subtilis.  Users can also supply a YAML / JSON config to extend or
override rules without touching source code.

Usage examples
--------------
  # basic
  python filter_nuisance_genes.py genes.txt --output-genes filtered.txt

  # mouse
  python filter_nuisance_genes.py genes.txt -o filtered.txt --species mouse

  # E. coli
  python filter_nuisance_genes.py genes.txt -o filtered.txt --species ecoli

  # custom rules from config
  python filter_nuisance_genes.py genes.txt -o filtered.txt --config my_rules.yaml

  # write removed genes to a sidecar file
  python filter_nuisance_genes.py genes.txt -o filtered.txt --removed-genes removed.txt

  # dry-run (preview only, no file written)
  python filter_nuisance_genes.py genes.txt -o filtered.txt --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback

logger = logging.getLogger(__name__)
console = Console()

install_rich_traceback(show_locals=False)

# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NuisanceRuleSet:
    """Immutable collection of nuisance-gene matching rules.

    Three layers, evaluated in short-circuit order:
      1. ``exact``   – O(1) set lookup
      2. ``prefixes`` – tuple fed to ``str.startswith``
      3. ``patterns`` – compiled regexes (full-match semantics)
    """

    exact: frozenset[str] = field(default_factory=frozenset)
    prefixes: tuple[str, ...] = field(default_factory=tuple)
    patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)
    name: str = "unnamed"

    # -- construction helpers ------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        name: str = "unnamed",
        exact: Iterable[str] = (),
        prefixes: Iterable[str] = (),
        patterns: Iterable[str] = (),
    ) -> NuisanceRuleSet:
        compiled = tuple(re.compile(p) for p in patterns)
        return cls(
            exact=frozenset(exact),
            prefixes=tuple(prefixes),
            patterns=compiled,
            name=name,
        )

    @classmethod
    def from_dict(cls, d: dict, *, name: str = "custom") -> NuisanceRuleSet:
        """Construct from a plain dict (e.g. parsed YAML / JSON)."""
        return cls.build(
            name=d.get("name", name),
            exact=d.get("exact", ()),
            prefixes=d.get("prefixes", ()),
            patterns=d.get("patterns", ()),
        )

    # -- matching ------------------------------------------------------------

    def is_nuisance(self, gene: str) -> bool:
        if gene in self.exact:
            return True
        if self.prefixes and gene.startswith(self.prefixes):
            return True
        return any(p.fullmatch(gene) for p in self.patterns)

    # -- merging -------------------------------------------------------------

    def merge(self, other: NuisanceRuleSet) -> NuisanceRuleSet:
        """Return a new rule-set that is the union of *self* and *other*."""
        return NuisanceRuleSet(
            exact=self.exact | other.exact,
            prefixes=self.prefixes + other.prefixes,
            patterns=self.patterns + other.patterns,
            name=f"{self.name}+{other.name}",
        )

    # -- pretty summary ------------------------------------------------------

    def summary(self) -> str:
        parts = [f"RuleSet({self.name})"]
        parts.append(f"  exact:    {len(self.exact)} genes")
        parts.append(f"  prefixes: {self.prefixes or '(none)'}")
        parts.append(f"  patterns: {[p.pattern for p in self.patterns] or '(none)'}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Built-in species rules  (string-pattern based, NOT exhaustive hard-coding)
# ---------------------------------------------------------------------------
# Design philosophy:
#   - Ribosomal proteins  → prefix (RPL/RPS, Rpl/Rps, rpl/rps …)
#   - Mitochondrial genes  → prefix (MT-, mt-)
#   - Hemoglobin           → regex  HB[A-Z]\d?  / Hb[a-z].*
#   - Stress / immediate-early genes → regex family patterns
#   - Anything that truly has no pattern → exact set (kept minimal)
#
# Prokaryotes use different conventions:
#   - E. coli: rRNA operons (rrs/rrl/rrf), tuf/fus (translation), groEL/dnaK
#   - B. subtilis: similar, plus sporulation sigma factors


def _build_human() -> NuisanceRuleSet:
    return NuisanceRuleSet.build(
        name="human",
        prefixes=(
            "MT-",  # mitochondrial
            "RPL",  # ribosomal protein large
            "RPS",  # ribosomal protein small
            "MRPL",  # mito ribosomal large
            "MRPS",  # mito ribosomal small
        ),
        patterns=(
            r"HB[A-Z]\d?",  # hemoglobin (HBA1, HBB, HBD …)
            r"HSP(?:90|A|B|D|E|H)\w*",  # heat-shock protein family
            r"DNAJ[A-C]\d+",  # DnaJ co-chaperones
            r"FOS[BL]?",  # Fos family IEGs
            r"JUN[BD]?",  # Jun family IEGs
            r"EGR[1-4]",  # early growth response
            r"IER[1-5][A-Z]?",  # immediate early response
            r"DUSP[1-9]\d?",  # dual-specificity phosphatases
        ),
        exact=(
            "MALAT1",
            "NEAT1",
            "XIST",  # lncRNAs
            "FTL",
            "FTH1",  # ferritin
        ),
    )


def _build_mouse() -> NuisanceRuleSet:
    return NuisanceRuleSet.build(
        name="mouse",
        prefixes=(
            "mt-",
            "Rpl",
            "Rps",
            "Mrpl",
            "Mrps",
        ),
        patterns=(
            r"Hb[a-z].*",  # hemoglobin
            r"Hsp(?:90|a|b|d|e|h)\w*",  # heat-shock
            r"Dnaj[a-c]\d+",
            r"Fos[bl]?",
            r"Jun[bd]?",
            r"Egr[1-4]",
            r"Ier[1-5][a-z]?",
            r"Dusp[1-9]\d?",
        ),
        exact=(
            "Malat1",
            "Neat1",
            "Xist",
            "Ftl1",
            "Fth1",
        ),
    )


def _build_ecoli() -> NuisanceRuleSet:
    """E. coli K-12 / MG1655 nuisance rules.

    Prokaryotic gene names are typically all-lowercase 3–4 letter mnemonics.
    Major nuisance sources in scRNA-seq of bacteria:
      - rRNA operon genes (rrsA-H, rrlA-H, rrfA-H)
      - translation elongation factors (tufA/B, fusA, tsf)
      - chaperones (groEL/ES, dnaK/J, clpB)
      - ribosomal proteins (rpl*, rps*, rpm*)
    """
    return NuisanceRuleSet.build(
        name="ecoli",
        prefixes=(
            "rpl",
            "rps",
            "rpm",  # ribosomal proteins
        ),
        patterns=(
            r"rr[slf][A-H]",  # rRNA operon genes
            r"tuf[AB]?",  # EF-Tu
            r"fus[A-Z]?",  # EF-G
            r"gro[ELS]+",  # GroEL/ES chaperonin
            r"dna[KJ]\w*",  # DnaK/J chaperone
            r"clp[ABPX]\w*",  # Clp protease / chaperone
            r"csp[A-I]",  # cold-shock proteins
        ),
        exact=(
            "tsf",  # EF-Ts
            "infA",
            "infB",
            "infC",  # translation initiation factors
            "ssrA",  # tmRNA
        ),
    )


def _build_bsub() -> NuisanceRuleSet:
    """B. subtilis 168 nuisance rules.

    Similar to E. coli with additions for sporulation-related genes and
    B. subtilis-specific naming (e.g. rrnA-J operons, sig* sigma factors).
    """
    return NuisanceRuleSet.build(
        name="bsub",
        prefixes=(
            "rpl",
            "rps",
            "rpm",
        ),
        patterns=(
            r"rrn[A-Z]",  # rRNA operons
            r"tuf[A-Z]?",
            r"fus[A-Z]?",
            r"groE[SL]\w*",
            r"dna[KJ]\w*",
            r"clp[CEPQXY]\w*",
            r"csp[A-Z]",
            r"sig[A-Z]",  # sigma factors (nuisance in some designs)
            r"hbs\w*",  # HU / histone-like DNA-binding
        ),
        exact=(
            "tsf",
            "infA",
            "infB",
            "infC",
            "ssrA",
        ),
    )


BUILTIN_SPECIES: dict[str, NuisanceRuleSet] = {
    "human": _build_human(),
    "mouse": _build_mouse(),
    "ecoli": _build_ecoli(),
    "bsub": _build_bsub(),
}


# ---------------------------------------------------------------------------
# Config loading (YAML / JSON)
# ---------------------------------------------------------------------------


def load_config_rules(path: Path) -> NuisanceRuleSet:
    """Load a NuisanceRuleSet from a YAML or JSON config file.

    Expected schema (YAML shown, JSON equivalent works too)::

        name: my_rules
        prefixes:
          - "MT-"
          - "RPL"
        patterns:
          - "HB[A-Z]\\d?"
        exact:
          - MALAT1
          - NEAT1
    """
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "PyYAML is required to load .yaml configs.  "
                "Install with: pip install pyyaml"
            )
            sys.exit(1)
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        # try JSON first, fall back to YAML
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore[import-untyped]

                data = yaml.safe_load(text)
            except ImportError:
                logger.error(
                    "Could not parse config as JSON and PyYAML is not "
                    "installed.  Install with: pip install pyyaml"
                )
                sys.exit(1)

    if not isinstance(data, dict):
        logger.error("Config must be a mapping at top level, got %s", type(data))
        sys.exit(1)

    ruleset = NuisanceRuleSet.from_dict(data, name=path.stem)
    logger.info("Loaded config rules from %s:\n%s", path, ruleset.summary())
    return ruleset


# ---------------------------------------------------------------------------
# Core filtering
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    kept: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    rules: NuisanceRuleSet | None = None

    @property
    def total(self) -> int:
        return len(self.kept) + len(self.removed)

    def report(self) -> str:
        lines = [
            f"species/ruleset : {self.rules.name if self.rules else 'none'}",
            f"input genes     : {self.total}",
            f"kept genes      : {len(self.kept)}",
            f"removed genes   : {len(self.removed)}",
        ]
        if self.removed:
            preview = self.removed[:20]
            lines.append(
                f"removed preview : {', '.join(preview)}"
                + (" …" if len(self.removed) > 20 else "")
            )
        return "\n".join(lines)


def filter_genes(
    genes: Sequence[str],
    rules: NuisanceRuleSet,
    *,
    deduplicate: bool = False,
) -> FilterResult:
    result = FilterResult(rules=rules)
    seen: set[str] = set()
    for gene in genes:
        gene = gene.strip()
        if not gene:
            continue
        if deduplicate:
            if gene in seen:
                continue
            seen.add(gene)
        if rules.is_nuisance(gene):
            result.removed.append(gene)
        else:
            result.kept.append(gene)
    return result


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_gene_list(path: Path) -> list[str]:
    if not path.exists():
        logger.error("Input file does not exist: %s", path)
        sys.exit(1)
    text = path.read_text(encoding="utf-8")
    genes = [line.strip() for line in text.splitlines() if line.strip()]
    if not genes:
        logger.warning("Input file is empty or contains no genes: %s", path)
    return genes


def write_gene_list(path: Path, genes: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(genes) + "\n", encoding="utf-8")
    logger.info("Wrote %d genes → %s", len(genes), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Filter nuisance genes from a ranked gene list.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Supported species: " + ", ".join(sorted(BUILTIN_SPECIES)) + ", none\n"
            "Custom rules can be supplied via --config (YAML or JSON)."
        ),
    )
    ap.add_argument(
        "input_genes", type=Path, help="Input gene-list text file (one gene per line)."
    )
    ap.add_argument(
        "-o",
        "--output-genes",
        type=Path,
        required=True,
        help="Output filtered gene list.",
    )
    ap.add_argument(
        "--removed-genes",
        type=Path,
        default=None,
        help="Optional file to write the removed genes.",
    )
    ap.add_argument(
        "--species",
        choices=sorted(BUILTIN_SPECIES) + ["none"],
        default="human",
        help="Built-in species rule-set (default: human).",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML or JSON file with additional / override rules.",
    )
    ap.add_argument(
        "--config-only",
        action="store_true",
        help="Ignore built-in species rules; use only --config.",
    )
    ap.add_argument(
        "--deduplicate", action="store_true", help="Remove duplicate gene names."
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Preview results without writing files."
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, markup=True, show_path=False)],
    )

    intro = Table(show_header=False, box=None)
    intro.add_row("Input", str(args.input_genes.expanduser().resolve()))
    intro.add_row("Output", str(args.output_genes.expanduser().resolve()))
    intro.add_row("Species", args.species)
    intro.add_row(
        "Config", str(args.config.expanduser().resolve()) if args.config else "None"
    )
    intro.add_row("Dry run", str(bool(args.dry_run)))
    console.print(Panel(intro, title="Filter Gene List", border_style="cyan"))

    # -- resolve rules -------------------------------------------------------
    if args.config_only and args.config is None:
        logger.error("--config-only requires --config")
        sys.exit(1)

    if args.config_only:
        rules = load_config_rules(args.config.expanduser().resolve())
    elif args.species == "none":
        rules = NuisanceRuleSet.build(name="none")
    else:
        rules = BUILTIN_SPECIES[args.species]

    if args.config and not args.config_only:
        extra = load_config_rules(args.config.expanduser().resolve())
        rules = rules.merge(extra)

    logger.debug("\n%s", rules.summary())

    # -- read & filter -------------------------------------------------------
    input_path = args.input_genes.expanduser().resolve()
    with console.status("Reading gene list and applying nuisance filters..."):
        genes = read_gene_list(input_path)
        result = filter_genes(genes, rules, deduplicate=args.deduplicate)

    summary = Table(title="Filtering Summary")
    summary.add_column("Field")
    summary.add_column("Value", overflow="fold")
    summary.add_row("Rule set", result.rules.name if result.rules else "none")
    summary.add_row("Input genes", str(result.total))
    summary.add_row("Kept genes", str(len(result.kept)))
    summary.add_row("Removed genes", str(len(result.removed)))
    summary.add_row(
        "Removed preview",
        ", ".join(result.removed[:10]) + (" ..." if len(result.removed) > 10 else "")
        if result.removed
        else "None",
    )
    console.print(summary)

    # -- write ---------------------------------------------------------------
    if args.dry_run:
        console.print(
            Panel("Dry run complete. No files were written.", border_style="yellow")
        )
        return

    output_path = args.output_genes.expanduser().resolve()
    with console.status("Writing filtered gene lists..."):
        write_gene_list(output_path, result.kept)

        if args.removed_genes:
            removed_path = args.removed_genes.expanduser().resolve()
            write_gene_list(removed_path, result.removed)

    outputs = Table(title="Outputs")
    outputs.add_column("File")
    outputs.add_column("Status")
    outputs.add_row(str(output_path), "filtered genes")
    if args.removed_genes:
        outputs.add_row(str(args.removed_genes.expanduser().resolve()), "removed genes")
    console.print(outputs)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(
            Panel(str(exc), title="filter_gene_list failed", border_style="red")
        )
        raise SystemExit(1) from exc
