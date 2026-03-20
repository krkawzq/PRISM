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
) -> str:
    candidate_block = ""
    if candidates:
        rows = "".join(
            f'<tr><td><a href="/gene?q={quote_plus(item.gene_name)}">{escape(item.gene_name)}</a></td>'
            f"<td>{item.gene_index}</td><td>{item.total_umi:,}</td></tr>"
            for item in candidates
        )
        candidate_block = f"""
        <section class=\"panel\">
          <h2>Similar genes</h2>
          <table><thead><tr><th>Gene</th><th>Index</th><th>Total UMI</th></tr></thead><tbody>{rows}</tbody></table>
        </section>
        """

    params = fit_params or {
        "r": 0.05,
        "grid_size": 512,
        "sigma_bins": 1.0,
        "align_loss_weight": 1.0,
        "lr": 0.05,
        "n_iter": 100,
        "lr_min_ratio": 0.1,
        "grad_clip": "",
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
            stat_card("Mean count", f"{float(analysis.counts.mean()):.3f}"),
            stat_card("Detected frac", f"{float((analysis.counts > 0).mean()):.3f}"),
            stat_card("Mean signal", f"{float(analysis.signal.mean()):.3f}"),
            stat_card("Mean confidence", f"{float(analysis.confidence.mean()):.3f}"),
            stat_card(
                "P95 surprisal", f"{float(np.quantile(analysis.surprisal, 0.95)):.3f}"
            ),
            stat_card("Mean sharpness", f"{float(analysis.sharpness.mean()):.3f}"),
        ]
    )
    metric_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{metrics['mean']:.4f}</td><td>{metrics['std']:.4f}</td><td>{metrics['p95']:.4f}</td><td>{metrics['depth_corr']:.4f}</td></tr>"
        for name, metrics in analysis.representation_metrics.items()
    )
    fit_form = f"""
      <section class=\"panel\">
        <h2>Gene fitting</h2>
        <form class=\"loader\" action=\"/gene\" method=\"get\">
          <input type=\"hidden\" name=\"q\" value=\"{escape(search_query)}\">
          <input type=\"hidden\" name=\"fit\" value=\"1\">
          <input type=\"text\" name=\"r\" value=\"{params["r"]}\" placeholder=\"r\">
          <input type=\"text\" name=\"grid_size\" value=\"{params["grid_size"]}\" placeholder=\"grid size\">
          <input type=\"text\" name=\"sigma_bins\" value=\"{params["sigma_bins"]}\" placeholder=\"sigma bins\">
          <input type=\"text\" name=\"align_loss_weight\" value=\"{params["align_loss_weight"]}\" placeholder=\"align weight\">
          <input type=\"text\" name=\"lr\" value=\"{params["lr"]}\" placeholder=\"lr\">
          <input type=\"text\" name=\"n_iter\" value=\"{params["n_iter"]}\" placeholder=\"iterations\">
          <input type=\"text\" name=\"init_temperature\" value=\"{params["init_temperature"]}\" placeholder=\"init temperature\">
          <input type=\"text\" name=\"cell_chunk_size\" value=\"{params["cell_chunk_size"]}\" placeholder=\"cell chunk\">
          <input type=\"text\" name=\"torch_dtype\" value=\"{params["torch_dtype"]}\" placeholder=\"float64\">
          <input type=\"text\" name=\"device\" value=\"{params["device"]}\" placeholder=\"cpu or cuda\">
          <button type=\"submit\">Fit / Refresh</button>
        </form>
      </section>
    """

    body = f"""
      {render_nav(current_query=search_query)}
      {fit_form}
      <section class=\"panel\">
        <h2>Gene summary</h2>
        <div class=\"stat-grid\">{stats}</div>
      </section>
      <section class=\"panel\">
        <h2>Representation metrics</h2>
        <table><thead><tr><th>Representation</th><th>Mean</th><th>Std</th><th>P95</th><th>Depth corr</th></tr></thead><tbody>{metric_rows}</tbody></table>
      </section>
      {candidate_block}
      <section class=\"figure-grid\">
        <div class=\"panel figure\"><h3>Raw count histogram</h3>{figures["counts_hist"]}</div>
        <div class=\"panel figure\"><h3>Signal histogram</h3>{figures["signal_hist"]}</div>
        <div class=\"panel figure\"><h3>X_eff vs signal</h3>{figures["xeff_signal"]}</div>
        <div class=\"panel figure\"><h3>Confidence vs surprisal</h3>{figures["confidence_surprisal"]}</div>
        <div class=\"panel figure figure-wide\"><h3>Representative posterior curves</h3>{figures["posterior_curves"]}</div>
      </section>
    """
    return render_page(title=f"PRISM Gene: {analysis.gene_name}", body=body)
