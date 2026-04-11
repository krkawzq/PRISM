#!/usr/bin/env python3
"""
Run cell-eval on a predicted/real AnnData pair.

This wrapper prefers the installed `cell_eval` package, but falls back to the
 local fork under `forks/cell-eval/src` so experiment scripts can use a stable
 entrypoint without requiring a separate installation step.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import anndata as ad

LOGGER = logging.getLogger("run_cell_eval")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CELL_EVAL_SRC = PROJECT_ROOT / "forks" / "cell-eval" / "src"
KNOWN_PROFILES = ["full", "minimal", "vcc", "de", "anndata", "pds"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper around cell-eval for experiment pipelines.",
    )
    parser.add_argument("--pred-h5ad", type=Path, required=True)
    parser.add_argument("--real-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pert-col", type=str, default="target_gene")
    parser.add_argument("--control-pert", type=str, default="non-targeting")
    parser.add_argument("--de-pred", type=Path, default=None)
    parser.add_argument("--de-real", type=Path, default=None)
    parser.add_argument("--celltype-col", type=str, default=None)
    parser.add_argument("--embed-key", type=str, default=None)
    parser.add_argument("--profile", type=str, default="full", choices=KNOWN_PROFILES)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--de-method", type=str, default="wilcoxon")
    parser.add_argument("--allow-discrete", action="store_true")
    parser.add_argument("--skip-metrics", type=str, default=None)
    parser.add_argument("--break-on-error", action="store_true")
    parser.add_argument("--cell-eval-src", type=Path, default=None)
    return parser.parse_args()


def ensure_cell_eval_import(extra_src: Path | None = None) -> None:
    try:
        import cell_eval  # noqa: F401

        return
    except ImportError:
        pass

    src = (extra_src or DEFAULT_CELL_EVAL_SRC).expanduser().resolve()
    if not src.exists():
        raise ImportError(
            "cell_eval is not installed and local fallback was not found at "
            f"{src}"
        )

    sys.path.insert(0, str(src))
    import cell_eval  # noqa: F401


def build_metric_configs(
    embed_key: str | None,
    num_threads: int,
) -> dict[str, dict[str, Any]] | None:
    if embed_key is None:
        return None
    return {
        "discrimination_score_l2": {"embed_key": embed_key},
        "discrimination_score_cosine": {"embed_key": embed_key},
        "pearson_edistance": {"n_jobs": num_threads},
    }


def run_single_evaluation(
    *,
    pred: ad.AnnData | str,
    real: ad.AnnData | str,
    outdir: Path,
    pert_col: str,
    control_pert: str,
    de_pred: str | None,
    de_real: str | None,
    profile: str,
    num_threads: int,
    batch_size: int,
    de_method: str,
    allow_discrete: bool,
    metric_configs: dict[str, dict[str, Any]] | None,
    skip_metrics: list[str] | None,
    break_on_error: bool,
    prefix: str | None = None,
) -> tuple[str, str]:
    from cell_eval import MetricsEvaluator

    evaluator = MetricsEvaluator(
        adata_pred=pred,
        adata_real=real,
        de_pred=de_pred,
        de_real=de_real,
        control_pert=control_pert,
        pert_col=pert_col,
        de_method=de_method,
        num_threads=num_threads,
        batch_size=batch_size,
        outdir=str(outdir),
        allow_discrete=allow_discrete,
        prefix=prefix,
        skip_de=profile == "pds",
    )
    evaluator.compute(
        profile=profile,  # type: ignore[arg-type]
        metric_configs=metric_configs,
        skip_metrics=skip_metrics,
        basename="results.csv",
        break_on_error=break_on_error,
    )

    results_name = f"{prefix}_results.csv" if prefix else "results.csv"
    agg_name = f"{prefix}_agg_results.csv" if prefix else "agg_results.csv"
    return results_name, agg_name


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    ensure_cell_eval_import(args.cell_eval_src)

    from cell_eval.utils import split_anndata_on_celltype

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_configs = build_metric_configs(args.embed_key, args.num_threads)
    skip_metrics = args.skip_metrics.split(",") if args.skip_metrics else None
    produced_results: list[dict[str, str]] = []

    if args.celltype_col:
        real = ad.read_h5ad(args.real_h5ad.expanduser().resolve())
        pred = ad.read_h5ad(args.pred_h5ad.expanduser().resolve())
        real_split = split_anndata_on_celltype(real, args.celltype_col)
        pred_split = split_anndata_on_celltype(pred, args.celltype_col)

        if set(real_split) != set(pred_split):
            raise ValueError(
                "Predicted and real AnnData have different celltype partitions: "
                f"{sorted(pred_split)} vs {sorted(real_split)}"
            )

        for celltype in sorted(real_split):
            result_name, agg_name = run_single_evaluation(
                pred=pred_split[celltype],
                real=real_split[celltype],
                outdir=output_dir,
                pert_col=args.pert_col,
                control_pert=args.control_pert,
                de_pred=str(args.de_pred) if args.de_pred else None,
                de_real=str(args.de_real) if args.de_real else None,
                profile=args.profile,
                num_threads=args.num_threads,
                batch_size=args.batch_size,
                de_method=args.de_method,
                allow_discrete=args.allow_discrete,
                metric_configs=metric_configs,
                skip_metrics=skip_metrics,
                break_on_error=args.break_on_error,
                prefix=celltype,
            )
            produced_results.append(
                {
                    "celltype": celltype,
                    "results_csv": str(output_dir / result_name),
                    "agg_results_csv": str(output_dir / agg_name),
                }
            )
    else:
        result_name, agg_name = run_single_evaluation(
            pred=str(args.pred_h5ad.expanduser().resolve()),
            real=str(args.real_h5ad.expanduser().resolve()),
            outdir=output_dir,
            pert_col=args.pert_col,
            control_pert=args.control_pert,
            de_pred=str(args.de_pred) if args.de_pred else None,
            de_real=str(args.de_real) if args.de_real else None,
            profile=args.profile,
            num_threads=args.num_threads,
            batch_size=args.batch_size,
            de_method=args.de_method,
            allow_discrete=args.allow_discrete,
            metric_configs=metric_configs,
            skip_metrics=skip_metrics,
            break_on_error=args.break_on_error,
        )
        produced_results.append(
            {
                "results_csv": str(output_dir / result_name),
                "agg_results_csv": str(output_dir / agg_name),
            }
        )

    config_path = output_dir / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "pred_h5ad": str(args.pred_h5ad.expanduser().resolve()),
                "real_h5ad": str(args.real_h5ad.expanduser().resolve()),
                "output_dir": str(output_dir),
                "pert_col": args.pert_col,
                "control_pert": args.control_pert,
                "celltype_col": args.celltype_col,
                "embed_key": args.embed_key,
                "profile": args.profile,
                "num_threads": args.num_threads,
                "batch_size": args.batch_size,
                "de_method": args.de_method,
                "allow_discrete": args.allow_discrete,
                "skip_metrics": skip_metrics,
                "results": produced_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Wrote run configuration to %s", config_path)


if __name__ == "__main__":
    main()
