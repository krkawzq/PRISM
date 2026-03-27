from __future__ import annotations

from html import escape
from urllib.parse import quote_plus

from prism.server.services.analysis import GeneAnalysis, GeneSummary
from prism.server.services.datasets import GeneCandidate

from .layout import render_nav, render_page, stat_card


def _default_fit_params() -> dict[str, object]:
    return {
        "S": "",
        "reference_mode": "checkpoint",
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


def _default_kbulk_params() -> dict[str, object]:
    return {
        "kbulk_k": 8,
        "kbulk_samples": 24,
        "kbulk_groups": 4,
        "kbulk_min_cells": 24,
        "kbulk_seed": 0,
    }


def _candidate_block(
    *, gene_name: str, candidates: list[GeneCandidate] | None = None
) -> str:
    if not candidates:
        return ""
    rows = "".join(
        f'<tr><td><a href="/gene?q={quote_plus(item.gene_name)}">{escape(item.gene_name)}</a></td><td>{item.gene_index}</td><td>{item.total_umi:,}</td></tr>'
        for item in candidates
        if item.gene_name != gene_name
    )
    if not rows:
        return ""
    return f'<section class="panel"><h2>Nearby matches</h2><table><thead><tr><th>Gene</th><th>Index</th><th>Total UMI</th></tr></thead><tbody>{rows}</tbody></table></section>'


def _fit_form(
    search_query: str, fit_params: dict[str, object] | None, *, has_checkpoint: bool
) -> str:
    params = _default_fit_params() if fit_params is None else fit_params
    reference_options = [
        ("checkpoint", "Checkpoint reference set"),
        ("all", "All genes except target"),
    ]
    optimizer_options = ["adamw", "adam", "sgd", "rmsprop"]
    scheduler_options = ["cosine", "linear", "constant", "step"]
    dtype_options = ["float64", "float32"]
    reference_html = "".join(
        f'<option value="{name}"{" selected" if str(params.get("reference_mode", "checkpoint")) == name else ""}{" disabled" if (name == "checkpoint" and not has_checkpoint) else ""}>{label}</option>'
        for name, label in reference_options
    )
    optimizer_html = "".join(
        f'<option value="{name}"{" selected" if str(params["optimizer"]) == name else ""}>{name}</option>'
        for name in optimizer_options
    )
    scheduler_html = "".join(
        f'<option value="{name}"{" selected" if str(params["scheduler"]) == name else ""}>{name}</option>'
        for name in scheduler_options
    )
    dtype_html = "".join(
        f'<option value="{name}"{" selected" if str(params["torch_dtype"]) == name else ""}>{name}</option>'
        for name in dtype_options
    )
    return f"""
      <section class="panel fit-panel">
        <form class="fit-form" action="/gene" method="get">
          <input type="hidden" name="q" value="{escape(search_query)}">
          <input type="hidden" name="fit" value="1">
          <div class="fit-header fit-hero">
            <div class="fit-title-stack">
              <div>
                <h2>On-demand fit</h2>
                <p>Run a quick single-gene fit directly in the browser flow using the new PRISM prior model.</p>
              </div>
              <div class="fit-badges">
                <span class="fit-badge">manual start</span>
                <span class="fit-badge">S defaults to N_avg</span>
                <span class="fit-badge">reference-aware</span>
              </div>
            </div>
            <div class="fit-action-card">
              <p>The fit uses the selected reference set to build reference counts, then runs a one-gene prior fit and posterior extraction.</p>
              <button type="submit">Fit / Refresh</button>
            </div>
          </div>
          <div class="fit-overview">
            <label class="fit-inline-field fit-inline-field-strong"><span>S</span><input type="text" name="S" value="{escape(str(params.get("S", "")))}" placeholder="blank = N_avg"><small>blank means use mean reference count</small></label>
            <label class="fit-inline-field"><span>reference mode</span><select name="reference_mode">{reference_html}</select><small>use checkpoint references when available</small></label>
            <label class="fit-inline-field"><span>grid size</span><input type="number" min="16" step="1" name="grid_size" value="{params["grid_size"]}"><small>512 is a solid default</small></label>
            <label class="fit-inline-field"><span>iterations</span><input type="number" min="1" step="1" name="n_iter" value="{params["n_iter"]}"><small>start with 100</small></label>
          </div>
          <details class="fit-advanced" open>
            <summary><span>Advanced controls</span><span class="muted">prior shape, optimizer, scheduler, and runtime knobs</span></summary>
            <div class="fit-sections">
              <div class="fit-section fit-section-primary">
                <h3>Prior geometry</h3>
                <div class="fit-fields">
                  <label class="fit-field"><span>sigma bins</span><input type="number" step="any" name="sigma_bins" value="{params["sigma_bins"]}"><small>Gaussian smoothing over the grid</small></label>
                  <label class="fit-field"><span>align weight</span><input type="number" step="any" name="align_loss_weight" value="{params["align_loss_weight"]}"><small>self-consistency weight</small></label>
                </div>
              </div>
              <div class="fit-section">
                <h3>Optimization</h3>
                <div class="fit-fields">
                  <label class="fit-field"><span>learning rate</span><input type="number" step="any" name="lr" value="{params["lr"]}"></label>
                  <label class="fit-field"><span>lr min ratio</span><input type="number" step="any" name="lr_min_ratio" value="{params["lr_min_ratio"]}"></label>
                  <label class="fit-field"><span>init temperature</span><input type="number" step="any" name="init_temperature" value="{params["init_temperature"]}"></label>
                </div>
              </div>
              <div class="fit-section">
                <h3>Runtime</h3>
                <div class="fit-fields">
                  <label class="fit-field"><span>cell chunk</span><input type="number" min="1" step="1" name="cell_chunk_size" value="{params["cell_chunk_size"]}"></label>
                  <label class="fit-field"><span>torch dtype</span><select name="torch_dtype">{dtype_html}</select></label>
                  <label class="fit-field"><span>optimizer</span><select name="optimizer">{optimizer_html}</select></label>
                  <label class="fit-field"><span>scheduler</span><select name="scheduler">{scheduler_html}</select></label>
                  <label class="fit-field"><span>device</span><input type="text" name="device" value="{escape(str(params["device"]))}"></label>
                </div>
              </div>
            </div>
          </details>
        </form>
      </section>
    """


def _kbulk_form(search_query: str, kbulk_params: dict[str, object] | None) -> str:
    params = (
        _default_kbulk_params()
        if kbulk_params is None
        else {
            "kbulk_k": kbulk_params.get("kbulk_k", kbulk_params.get("k", 8)),
            "kbulk_samples": kbulk_params.get(
                "kbulk_samples", kbulk_params.get("n_samples", 24)
            ),
            "kbulk_groups": kbulk_params.get(
                "kbulk_groups", kbulk_params.get("max_groups", 4)
            ),
            "kbulk_min_cells": kbulk_params.get(
                "kbulk_min_cells", kbulk_params.get("min_cells_per_group", 24)
            ),
            "kbulk_seed": kbulk_params.get(
                "kbulk_seed", kbulk_params.get("random_seed", 0)
            ),
        }
    )
    return f"""
      <section class="panel kbulk-panel">
        <form class="kbulk-form" action="/gene" method="get">
          <input type="hidden" name="q" value="{escape(search_query)}">
          <input type="hidden" name="kbulk" value="1">
          <div class="kbulk-header">
            <div>
              <h2>kBulk comparison</h2>
              <p>Fit class-specific F_g on label groups, then compare repeated kBulk MAP estimates across groups.</p>
            </div>
            <button type="submit">Compute kBulk</button>
          </div>
          <div class="kbulk-grid">
            <label><span>k</span><input type="number" min="2" step="1" name="kbulk_k" value="{params["kbulk_k"]}"></label>
            <label><span>samples</span><input type="number" min="1" step="1" name="kbulk_samples" value="{params["kbulk_samples"]}"></label>
            <label><span>groups</span><input type="number" min="1" step="1" name="kbulk_groups" value="{params["kbulk_groups"]}"></label>
            <label><span>min cells</span><input type="number" min="4" step="1" name="kbulk_min_cells" value="{params["kbulk_min_cells"]}"></label>
            <label><span>seed</span><input type="number" min="0" step="1" name="kbulk_seed" value="{params["kbulk_seed"]}"></label>
          </div>
        </form>
      </section>
    """


def render_gene_pending_page(
    *,
    gene_name: str,
    gene_index: int,
    summary: GeneSummary,
    search_query: str,
    fit_params: dict[str, object] | None = None,
    candidates: list[GeneCandidate] | None = None,
    figures: dict[str, str] | None = None,
    has_checkpoint: bool = False,
    kbulk_params: dict[str, object] | None = None,
) -> str:
    figures = {} if figures is None else figures
    stats = "".join(
        [
            stat_card("Gene", gene_name),
            stat_card("Index", str(gene_index)),
            stat_card("Source", "raw-only"),
            stat_card("Mean count", f"{summary.mean_count:.3f}"),
            stat_card("Detected frac", f"{summary.detected_frac:.3f}"),
            stat_card("Zero frac", f"{summary.zero_frac:.3f}"),
            stat_card("P99 count", f"{summary.p99_count:.3f}"),
            stat_card("Depth corr", f"{summary.count_total_correlation:.3f}"),
        ]
    )
    overview_block = (
        ""
        if "gene_overview" not in figures
        else f'<section class="panel figure-wide"><h2>Gene overview</h2><div class="figure"><img src="{figures["gene_overview"]}"></div></section>'
    )
    body = f"""
      {render_nav(current_query=search_query)}
      {_fit_form(search_query, fit_params, has_checkpoint=has_checkpoint)}
      {_kbulk_form(search_query, kbulk_params)}
      <section class="panel"><h2>Ready to fit</h2><p>No prior is currently available for <strong>{escape(gene_name)}</strong>. Use the on-demand fit form to build one directly from the loaded dataset.</p></section>
      <section class="panel"><h2>Raw summary</h2><div class="stat-grid">{stats}</div></section>
      {_candidate_block(gene_name=gene_name, candidates=candidates)}
      {overview_block}
    """
    return render_page(title=f"PRISM Gene: {gene_name}", body=body)


def render_gene_page(
    *,
    analysis: GeneAnalysis,
    figures: dict[str, str],
    search_query: str,
    candidates: list[GeneCandidate] | None = None,
    fit_params: dict[str, object] | None = None,
    kbulk_params: dict[str, object] | None = None,
    has_checkpoint: bool = False,
) -> str:
    stats = "".join(
        [
            stat_card("Gene", analysis.gene_name),
            stat_card("Index", str(analysis.gene_index)),
            stat_card("Source", analysis.source),
            stat_card("S", f"{analysis.S:.3f}"),
            stat_card("S source", analysis.S_source),
            stat_card("Reference mode", analysis.reference_mode),
            stat_card("Reference genes", str(analysis.reference_gene_count)),
            stat_card("Mean signal", f"{float(analysis.signal.mean()):.3f}"),
            stat_card(
                "Mean posterior entropy",
                f"{float(analysis.posterior_entropy.mean()):.3f}",
            ),
            stat_card(
                "Mean prior entropy", f"{float(analysis.prior_entropy.mean()):.3f}"
            ),
            stat_card(
                "Mean mutual info", f"{float(analysis.mutual_information.mean()):.3f}"
            ),
            stat_card("Depth corr", f"{analysis.summary.count_total_correlation:.3f}"),
        ]
    )
    metric_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{metrics.mean:.4f}</td><td>{metrics.median:.4f}</td><td>{metrics.std:.4f}</td><td>{metrics.var:.4f}</td><td>{metrics.p95:.4f}</td><td>{metrics.nonzero_frac:.4f}</td><td>{metrics.depth_corr:.4f}</td><td>{metrics.depth_mi:.4f}</td><td>{'-' if metrics.sparsity_corr is None else f'{metrics.sparsity_corr:.4f}'}</td><td>{'-' if metrics.fisher_ratio is None else f'{metrics.fisher_ratio:.4f}'}</td><td>{'-' if metrics.kruskal_h is None else f'{metrics.kruskal_h:.4f}'}</td><td>{'-' if metrics.kruskal_p is None else f'{metrics.kruskal_p:.3e}'}</td><td>{'-' if metrics.auroc_ovr is None else f'{metrics.auroc_ovr:.4f}'}</td><td>{'-' if metrics.zero_consistency is None else f'{metrics.zero_consistency:.4f}'}</td><td>{'-' if metrics.zero_rank_tau is None else f'{metrics.zero_rank_tau:.4f}'}</td><td>{'-' if metrics.dropout_recovery is None else f'{metrics.dropout_recovery:.4f}'}</td><td>{'-' if metrics.treatment_cv is None else f'{metrics.treatment_cv:.4f}'}</td></tr>"
        for name, metrics in analysis.representation_metrics.items()
    )
    fit_trace_block = (
        ""
        if "loss_trace" not in figures
        else f'<section class="panel figure-wide"><h2>On-demand fit trace</h2><div class="figure"><img src="{figures["loss_trace"]}"></div></section>'
    )
    kbulk_block = ""
    if analysis.kbulk is not None and "kbulk" in figures:
        rows = "".join(
            f"<tr><td>{escape(group.label)}</td><td>{group.n_cells}</td><td>{group.mean_map_mu:.4f}</td><td>{group.std_map_mu:.4f}</td><td>{group.mean_posterior_entropy:.4f}</td></tr>"
            for group in analysis.kbulk.groups
        )
        kbulk_block = f'<section class="panel figure-wide"><h2>kBulk comparison</h2><p class="muted">Label key: <strong>{escape(analysis.kbulk.label_key)}</strong> | k={analysis.kbulk.k} | samples={analysis.kbulk.n_samples}</p><div class="table-wrap"><table><thead><tr><th>Group</th><th>Cells</th><th>Mean MAP mu</th><th>Std MAP mu</th><th>Mean posterior entropy</th></tr></thead><tbody>{rows}</tbody></table></div><div class="figure"><img src="{figures["kbulk"]}"></div></section>'
    body = f"""
      {render_nav(current_query=search_query)}
      {_fit_form(search_query, fit_params, has_checkpoint=has_checkpoint)}
      {_kbulk_form(search_query, kbulk_params)}
      <section class="panel"><h2>Gene summary</h2><div class="stat-grid">{stats}</div></section>
      <section class="panel"><h2>Baseline metrics</h2><div class="table-wrap"><table class="compact-table"><thead><tr><th>Signal</th><th>Mean</th><th>Median</th><th>Std</th><th>Var</th><th>P95</th><th>Nonzero frac</th><th>Depth corr</th><th>Depth MI</th><th>Sparsity corr</th><th>Fisher ratio</th><th>Kruskal H</th><th>Kruskal p</th><th>AUROC OVR</th><th>Zero-group consistency</th><th>Zero rank tau</th><th>Dropout recovery</th><th>Treatment CV</th></tr></thead><tbody>{metric_rows}</tbody></table></div></section>
      {_candidate_block(gene_name=analysis.gene_name, candidates=candidates)}
      <section class="panel figure-wide"><h2>Gene overview</h2><div class="figure"><img src="{figures["gene_overview"]}"></div></section>
      {fit_trace_block}
      <section class="panel figure-wide"><h2>Prior profile</h2><div class="figure"><img src="{figures["prior_fit"]}"></div></section>
      {kbulk_block}
      <section class="panel figure-wide"><h2>Signal interface</h2><div class="figure"><img src="{figures["signal_interface"]}"></div></section>
      <section class="panel figure-wide"><h2>Posterior gallery</h2><div class="figure"><img src="{figures["posterior_gallery"]}"></div></section>
    """
    return render_page(title=f"PRISM Gene: {analysis.gene_name}", body=body)
