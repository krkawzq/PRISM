from __future__ import annotations

from html import escape

import numpy as np

from prism.server.services.analysis import (
    GeneAnalysis,
    GeneFitParams,
    KBulkAnalysis,
    KBulkParams,
)

from .components import render_chip_row, render_detail_grid, render_section_header
from .layout import render_message, render_nav, render_page, stat_card


def render_gene_page(
    *,
    analysis: GeneAnalysis,
    raw_figure: str | None,
    prior_figure: str | None,
    signal_figure: str | None,
    gallery_figure: str | None,
    objective_figure: str | None,
    fit_params: GeneFitParams,
    kbulk_params: KBulkParams,
    kbulk_analysis: KBulkAnalysis | None = None,
    kbulk_figure: str | None = None,
    error_message: str | None = None,
    kbulk_error: str | None = None,
) -> str:
    parts = [render_nav(current_query=analysis.gene_name)]
    if error_message:
        parts.append(render_message(error_message, level="error"))
    if kbulk_error:
        parts.append(render_message(kbulk_error, level="error"))
    parts.append(_render_gene_identity(analysis))
    parts.append(_render_analysis_controls(analysis))
    parts.append(_render_fit_form(analysis, fit_params))
    parts.append(_render_kbulk_form(analysis, kbulk_params))
    parts.append(_render_summary(analysis))
    if raw_figure:
        parts.append(_figure_block("Raw Overview", raw_figure))
    if prior_figure:
        parts.append(_figure_block("Prior Overlay", prior_figure))
    if signal_figure:
        parts.append(_figure_block("Signal Interface", signal_figure))
    if gallery_figure:
        parts.append(_figure_block("Posterior Gallery", gallery_figure))
    if objective_figure:
        parts.append(_figure_block("Objective Trace", objective_figure))
    if kbulk_analysis is not None:
        parts.append(_render_kbulk_summary(kbulk_analysis, kbulk_figure))
    return render_page(title=f"PRISM Gene: {analysis.gene_name}", body="".join(parts))


def _render_gene_identity(analysis: GeneAnalysis) -> str:
    chips = render_chip_row(
        [
            (f"Mode: {analysis.mode}", "info"),
            (f"Source: {analysis.source}", "neutral"),
            (f"Prior: {analysis.prior_source}", "warning"),
        ]
    )
    details = render_detail_grid(
        [
            ("Gene", analysis.gene_name),
            ("Gene index", str(analysis.gene_index)),
            ("Label key", analysis.label_key or "Auto"),
            ("Label", analysis.label or "Auto"),
            ("Reference source", analysis.reference_source),
            ("Reference genes", f"{analysis.reference_gene_count:,}"),
        ]
    )
    stats = [
        stat_card("Cells", f"{analysis.n_cells:,}"),
        stat_card("Raw mean", f"{analysis.raw_summary.mean_count:.4f}"),
        stat_card("Detected frac", f"{analysis.raw_summary.detected_fraction:.4f}"),
        stat_card("Zero frac", f"{analysis.raw_summary.zero_fraction:.4f}"),
    ]
    section_header = render_section_header(
        analysis.gene_name,
        "Current gene workspace, analysis mode, and reference context.",
        eyebrow="Gene Workspace",
    )
    return f"""
    <section class="panel">
      {section_header}
      {chips}
      {details}
      <div class="stat-grid">{"".join(stats)}</div>
    </section>
    """


def _render_analysis_controls(analysis: GeneAnalysis) -> str:
    label_key_options = "".join(
        f'<option value="{escape(key)}"{" selected" if analysis.label_key == key else ""}>{escape(key)}</option>'
        for key in analysis.available_label_keys
    )
    label_options = "".join(
        f'<option value="{escape(value)}"{" selected" if analysis.label == value else ""}>{escape(value)}</option>'
        for value in analysis.available_labels
    )
    section_header = render_section_header(
        "Analysis Controls",
        "Switch between raw, checkpoint, and fit-backed views. Narrow the analysis to a label-specific subset when available.",
        eyebrow="Controls",
    )
    return f"""
    <section class="panel">
      {section_header}
      <form class="toolbar stack-mobile" action="/gene" method="get">
        <input type="hidden" name="q" value="{escape(analysis.gene_name)}">
        <label class="field"><span>Mode</span><select name="mode">
          <option value="raw"{" selected" if analysis.mode == "raw" else ""}>Raw only</option>
          <option value="checkpoint"{" selected" if analysis.mode == "checkpoint" else ""}>Checkpoint posterior</option>
          <option value="fit"{" selected" if analysis.mode == "fit" else ""}>On-demand fit</option>
        </select></label>
        <label class="field"><span>Prior source</span><select name="prior_source">
          <option value="global"{" selected" if analysis.prior_source == "global" else ""}>Global prior</option>
          <option value="label"{" selected" if analysis.prior_source == "label" else ""}>Label prior</option>
        </select></label>
        <label class="field"><span>Label key</span><select name="label_key">
          <option value="">Auto label key</option>
          {label_key_options}
        </select></label>
        <label class="field"><span>Label</span><select name="label">
          <option value="">Auto label</option>
          {label_options}
        </select></label>
        <div class="form-actions form-actions-inline"><button type="submit">Refresh</button></div>
      </form>
    </section>
    """


def _render_fit_form(analysis: GeneAnalysis, params: GeneFitParams) -> str:
    section_header = render_section_header(
        "On-Demand Fit",
        "Adjust fit configuration, rerun a single-gene prior fit, and compare the resulting posterior diagnostics.",
        eyebrow="Fit",
    )
    return f"""
    <section class="panel">
      {section_header}
      <form class="form-grid" action="/gene" method="get">
        {_shared_hidden_inputs(analysis)}
        <input type="hidden" name="mode" value="fit">
        <label><span>Scale</span><input type="text" name="scale" value="{"" if params.scale is None else params.scale}"></label>
        <label><span>Reference source</span><select name="reference_source"><option value="checkpoint"{" selected" if params.reference_source == "checkpoint" else ""}>checkpoint</option><option value="dataset"{" selected" if params.reference_source == "dataset" else ""}>dataset</option></select></label>
        <label><span>Support points</span><input type="number" min="2" name="n_support_points" value="{params.n_support_points}"></label>
        <label><span>Max EM iterations</span><input type="number" min="1" name="max_em_iterations" value="{"" if params.max_em_iterations is None else params.max_em_iterations}"></label>
        <label><span>Convergence tolerance</span><input type="text" name="convergence_tolerance" value="{params.convergence_tolerance}"></label>
        <label><span>Cell chunk size</span><input type="number" min="1" name="cell_chunk_size" value="{params.cell_chunk_size}"></label>
        <label><span>Support max from</span><select name="support_max_from"><option value="observed_max"{" selected" if params.support_max_from == "observed_max" else ""}>observed_max</option><option value="quantile"{" selected" if params.support_max_from == "quantile" else ""}>quantile</option></select></label>
        <label><span>Support spacing</span><select name="support_spacing"><option value="linear"{" selected" if params.support_spacing == "linear" else ""}>linear</option><option value="sqrt"{" selected" if params.support_spacing == "sqrt" else ""}>sqrt</option></select></label>
        <label><span>Adaptive support</span><select name="use_adaptive_support"><option value="0"{" selected" if not params.use_adaptive_support else ""}>off</option><option value="1"{" selected" if params.use_adaptive_support else ""}>on</option></select></label>
        <label><span>Adaptive fraction</span><input type="text" name="adaptive_support_fraction" value="{params.adaptive_support_fraction}"></label>
        <label><span>Adaptive q_hi</span><input type="text" name="adaptive_support_quantile_hi" value="{params.adaptive_support_quantile_hi}"></label>
        <label><span>Likelihood</span><select name="likelihood"><option value="binomial"{" selected" if params.likelihood == "binomial" else ""}>binomial</option><option value="negative_binomial"{" selected" if params.likelihood == "negative_binomial" else ""}>negative_binomial</option><option value="poisson"{" selected" if params.likelihood == "poisson" else ""}>poisson</option></select></label>
        <label><span>NB overdispersion</span><input type="text" name="nb_overdispersion" value="{params.nb_overdispersion}"></label>
        <label><span>Torch dtype</span><select name="torch_dtype"><option value="float64"{" selected" if params.torch_dtype == "float64" else ""}>float64</option><option value="float32"{" selected" if params.torch_dtype == "float32" else ""}>float32</option></select></label>
        <label><span>Compile model</span><select name="compile_model"><option value="1"{" selected" if params.compile_model else ""}>on</option><option value="0"{" selected" if not params.compile_model else ""}>off</option></select></label>
        <label><span>Device</span><input type="text" name="device" value="{escape(params.device)}"></label>
        <div class="form-actions"><button type="submit">Run Fit</button></div>
      </form>
    </section>
    """


def _render_kbulk_form(analysis: GeneAnalysis, params: KBulkParams) -> str:
    label_key_options = "".join(
        f'<option value="{escape(key)}"{" selected" if params.class_key == key else ""}>{escape(key)}</option>'
        for key in analysis.available_label_keys
    )
    section_header = render_section_header(
        "kBulk Analysis",
        "Sample cell groups by class key and compare group-level signal and uncertainty summaries.",
        eyebrow="Groups",
    )
    return f"""
    <section class="panel">
      {section_header}
      <form class="form-grid compact" action="/gene" method="get">
        {_shared_hidden_inputs(analysis)}
        <input type="hidden" name="kbulk" value="1">
        <label><span>Class key</span><select name="class_key"><option value="">Auto</option>{label_key_options}</select></label>
        <label><span>k</span><input type="number" min="1" name="k" value="{params.k}"></label>
        <label><span>Samples</span><input type="number" min="1" name="n_samples" value="{params.n_samples}"></label>
        <label><span>Seed</span><input type="number" min="0" name="sample_seed" value="{params.sample_seed}"></label>
        <label><span>Max classes</span><input type="number" min="1" name="max_classes" value="{params.max_classes}"></label>
        <label><span>Batch size</span><input type="number" min="1" name="sample_batch_size" value="{params.sample_batch_size}"></label>
        <label><span>Prior source</span><select name="kbulk_prior_source"><option value="global"{" selected" if params.kbulk_prior_source == "global" else ""}>global</option><option value="label"{" selected" if params.kbulk_prior_source == "label" else ""}>label</option></select></label>
        <div class="form-actions"><button type="submit">Run kBulk</button></div>
      </form>
    </section>
    """


def _render_summary(analysis: GeneAnalysis) -> str:
    stats = [
        stat_card("Gene", analysis.gene_name),
        stat_card("Index", str(analysis.gene_index)),
        stat_card("Mode", analysis.mode),
        stat_card("Source", analysis.source),
        stat_card("Prior source", analysis.prior_source),
        stat_card("Label key", analysis.label_key or "-"),
        stat_card("Label", analysis.label or "-"),
        stat_card("Cells", f"{analysis.n_cells:,}"),
        stat_card("Reference source", analysis.reference_source),
        stat_card("Reference genes", f"{analysis.reference_gene_count:,}"),
        stat_card("Raw mean", f"{analysis.raw_summary.mean_count:.4f}"),
        stat_card("Raw median", f"{analysis.raw_summary.median_count:.4f}"),
        stat_card("P99", f"{analysis.raw_summary.p99_count:.4f}"),
        stat_card("Detected frac", f"{analysis.raw_summary.detected_fraction:.4f}"),
        stat_card("Zero frac", f"{analysis.raw_summary.zero_fraction:.4f}"),
        stat_card("Depth corr", f"{analysis.raw_summary.count_total_correlation:.4f}"),
    ]
    if analysis.posterior is not None:
        signal = np.asarray(
            analysis.posterior.map_scaled_support[:, 0], dtype=np.float64
        )
        stats.extend(
            [
                stat_card("Mean signal", f"{float(np.mean(signal)):.4f}"),
                stat_card(
                    "Mean entropy",
                    f"{float(np.mean(analysis.posterior.posterior_entropy[:, 0])):.4f}",
                ),
                stat_card(
                    "Mean MI",
                    f"{float(np.mean(analysis.posterior.mutual_information[:, 0])):.4f}",
                ),
            ]
        )
    section_header = render_section_header(
        "Gene Summary",
        "Raw and posterior summary metrics for the currently selected gene context.",
        eyebrow="Summary",
    )
    return f'<section class="panel">{section_header}<div class="stat-grid">{"".join(stats)}</div></section>'


def _render_kbulk_summary(result: KBulkAnalysis, figure_uri: str | None) -> str:
    max_classes = getattr(result, "max_classes", "-")
    sample_batch_size = getattr(result, "sample_batch_size", "-")
    rows = "".join(
        f"<tr><td>{escape(group.label)}</td><td>{group.n_cells:,}</td><td>{group.realized_samples:,}</td><td>{group.mean_signal:.4f}</td><td>{group.std_signal:.4f}</td><td>{group.mean_entropy:.4f}</td><td>{group.std_entropy:.4f}</td></tr>"
        for group in result.groups
    )
    figure = (
        ""
        if figure_uri is None
        else _figure_block("kBulk Group Comparison", figure_uri)
    )
    section_header = render_section_header(
        "kBulk Summary",
        "Aggregated signal and entropy summaries for each eligible class under the current sampling setup.",
        eyebrow="Result",
    )
    chips = render_chip_row(
        [
            (f"Class key: {result.class_key}", "info"),
            (f"Prior: {result.prior_source}", "neutral"),
            (f"k={result.k}", "warning"),
        ]
    )
    return f"""
    <section class="panel">
      {section_header}
      {chips}
      <p class="muted">Requested samples: {result.n_samples} | Max classes: {max_classes} | Batch size: {sample_batch_size}</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Label</th><th>Cells</th><th>Samples</th><th>Mean signal</th><th>Std signal</th><th>Mean entropy</th><th>Std entropy</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </section>
    {figure}
    """


def _figure_block(title: str, src: str) -> str:
    section_header = render_section_header(
        title,
        "Figure output generated from the current analysis state.",
        eyebrow="Figure",
    )
    return f'<section class="panel figure-panel">{section_header}<div class="figure"><img src="{src}" alt="{escape(title)}"></div></section>'


def _shared_hidden_inputs(analysis: GeneAnalysis) -> str:
    return (
        f'<input type="hidden" name="q" value="{escape(analysis.gene_name)}">'
        f'<input type="hidden" name="prior_source" value="{escape(analysis.prior_source)}">'
        f'<input type="hidden" name="label_key" value="{escape(analysis.label_key or "")}">'
        f'<input type="hidden" name="label" value="{escape(analysis.label or "")}">'
    )
