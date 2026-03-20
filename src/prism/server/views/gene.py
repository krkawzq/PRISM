from __future__ import annotations

from html import escape
from urllib.parse import quote_plus

import numpy as np

from prism.server.services.analysis import GeneAnalysis
from prism.server.services.datasets import GeneCandidate

from .layout import render_nav, render_page, stat_card


def render_gene_page(
    *,
    analysis: GeneAnalysis,
    figures: dict[str, str],
    search_query: str,
    candidates: list[GeneCandidate] | None = None,
    fit_params: dict[str, object] | None = None,
    treatment_block_html: str = "",
) -> str:
    candidate_block = ""
    if candidates:
        rows = "".join(
            f'<tr><td><a href="/gene?q={quote_plus(item.gene_name)}">{escape(item.gene_name)}</a></td>'
            f"<td>{item.gene_index}</td><td>{item.total_umi:,}</td></tr>"
            for item in candidates
            if item.gene_name != analysis.gene_name
        )
        if rows:
            candidate_block = (
                '<section class="panel"><h2>Nearby matches</h2>'
                "<table><thead><tr><th>Gene</th><th>Index</th><th>Total UMI</th></tr></thead>"
                f"<tbody>{rows}</tbody></table></section>"
            )

    params = fit_params or {
        "r": 0.05,
        "grid_size": 512,
        "sigma_bins": 1.0,
        "align_loss_weight": 1.0,
        "lr": 0.05,
        "n_iter": 100,
        "lr_min_ratio": 0.1,
        "init_temperature": 1.0,
        "cell_chunk_size": 512,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "torch_dtype": "float64",
        "device": "cpu",
    }
    stats = "".join(
        [
            stat_card("Gene", analysis.gene_name),
            stat_card("Index", str(analysis.gene_index)),
            stat_card("Source", analysis.source),
            stat_card("s_hat", f"{analysis.s_hat:.3f}"),
            stat_card("Mean count", f"{analysis.summary.mean_count:.3f}"),
            stat_card("Detected frac", f"{analysis.summary.detected_frac:.3f}"),
            stat_card("Mean signal", f"{float(analysis.signal.mean()):.3f}"),
            stat_card("Mean confidence", f"{float(analysis.confidence.mean()):.3f}"),
            stat_card(
                "P95 surprisal",
                f"{float(np.quantile(analysis.surprisal, 0.95)):.3f}",
            ),
            stat_card("Depth corr", f"{analysis.summary.count_total_correlation:.3f}"),
        ]
    )
    metric_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{metrics.mean:.4f}</td><td>{metrics.median:.4f}</td><td>{metrics.std:.4f}</td><td>{metrics.var:.4f}</td><td>{metrics.p95:.4f}</td><td>{metrics.nonzero_frac:.4f}</td><td>{metrics.depth_corr:.4f}</td><td>{metrics.depth_mi:.4f}</td><td>{'-' if metrics.sparsity_corr is None else f'{metrics.sparsity_corr:.4f}'}</td><td>{'-' if metrics.fisher_ratio is None else f'{metrics.fisher_ratio:.4f}'}</td><td>{'-' if metrics.kruskal_h is None else f'{metrics.kruskal_h:.4f}'}</td><td>{'-' if metrics.kruskal_p is None else f'{metrics.kruskal_p:.3e}'}</td><td>{'-' if metrics.auroc_ovr is None else f'{metrics.auroc_ovr:.4f}'}</td><td>{'-' if metrics.zero_consistency is None else f'{metrics.zero_consistency:.4f}'}</td><td>{'-' if metrics.zero_rank_tau is None else f'{metrics.zero_rank_tau:.4f}'}</td><td>{'-' if metrics.dropout_recovery is None else f'{metrics.dropout_recovery:.4f}'}</td><td>{'-' if metrics.treatment_cv is None else f'{metrics.treatment_cv:.4f}'}</td></tr>"
        for name, metrics in analysis.representation_metrics.items()
    )
    fit_form = f"""
      <section class="panel">
        <h2>On-demand gene fitting</h2>
        <form class="loader fit-grid" action="/gene" method="get">
          <input type="hidden" name="q" value="{escape(search_query)}">
          <input type="hidden" name="fit" value="1">
          <input type="text" name="r" value="{params["r"]}" placeholder="r">
          <input type="text" name="grid_size" value="{params["grid_size"]}" placeholder="grid size">
          <input type="text" name="sigma_bins" value="{params["sigma_bins"]}" placeholder="sigma bins">
          <input type="text" name="align_loss_weight" value="{params["align_loss_weight"]}" placeholder="align weight">
          <input type="text" name="lr" value="{params["lr"]}" placeholder="lr">
          <input type="text" name="n_iter" value="{params["n_iter"]}" placeholder="iterations">
          <input type="text" name="lr_min_ratio" value="{params["lr_min_ratio"]}" placeholder="lr min ratio">
          <input type="text" name="init_temperature" value="{params["init_temperature"]}" placeholder="init temperature">
          <input type="text" name="cell_chunk_size" value="{params["cell_chunk_size"]}" placeholder="cell chunk">
          <input type="text" name="optimizer" value="{params["optimizer"]}" placeholder="optimizer">
          <input type="text" name="scheduler" value="{params["scheduler"]}" placeholder="scheduler">
          <input type="text" name="torch_dtype" value="{params["torch_dtype"]}" placeholder="float64 or float32">
          <input type="text" name="device" value="{params["device"]}" placeholder="cpu or cuda">
          <button type="submit">Fit / Refresh</button>
        </form>
      </section>
    """

    stage0_block = ""
    if "stage0" in figures:
        stage0_block = f'<section class="panel figure-wide"><h2>Stage 0: Sampling-pool scale</h2><div class="figure"><img src="{figures["stage0"]}"></div></section>'

    trace_block = ""
    if "loss_trace" in figures:
        trace_block = f'<section class="panel figure-wide"><h2>Stage 1: Optimization trace</h2><div class="figure"><img src="{figures["loss_trace"]}"></div></section>'

    init_block = ""
    if "init_comparison" in figures:
        init_block = f'<section class="panel figure-wide"><h2>Initialization comparison</h2><div class="figure"><img src="{figures["init_comparison"]}"></div></section>'

    body = f"""
      {render_nav(current_query=search_query)}
      {fit_form}
      <section class="panel"><h2>Gene summary</h2><div class="stat-grid">{stats}</div></section>
      <section class="panel"><h2>Baseline metrics</h2><div class="table-wrap"><table class="compact-table"><thead><tr><th>Signal</th><th>Mean</th><th>Median</th><th>Std</th><th>Var</th><th>P95</th><th>Nonzero frac</th><th>Depth corr</th><th>Depth MI</th><th>Sparsity corr</th><th>Fisher ratio</th><th>Kruskal H</th><th>Kruskal p</th><th>AUROC OVR</th><th>Zero-group consistency</th><th>Zero rank tau</th><th>Dropout recovery</th><th>Treatment CV</th></tr></thead><tbody>{metric_rows}</tbody></table></div></section>
      {candidate_block}
      <section class="panel figure-wide"><h2>Gene overview</h2><div class="figure"><img src="{figures["gene_overview"]}"></div></section>
      {treatment_block_html}
      {stage0_block}
      {trace_block}
      {init_block}
      <section class="panel figure-wide"><h2>Prior and self-consistency</h2><div class="figure"><img src="{figures["prior_fit"]}"></div></section>
      <section class="panel figure-wide"><h2>Signal interface</h2><div class="figure"><img src="{figures["signal_interface"]}"></div></section>
      <section class="panel figure-wide"><h2>3D signal space</h2><div class="figure">{figures["signal_3d"]}</div></section>
      <section class="panel figure-wide"><h2>Posterior gallery</h2><div class="figure"><img src="{figures["posterior_gallery"]}"></div></section>
    """
    return render_page(title=f"PRISM Gene: {analysis.gene_name}", body=body)
