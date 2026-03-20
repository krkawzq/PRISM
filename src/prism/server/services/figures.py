from __future__ import annotations

import base64
import io
import json
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import GeneAnalysis, GeneSummary

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def fig_to_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def fmt_int(v):
    return f"{int(round(float(v))):,}"


def fmt_f(v, d=4):
    v = float(v)
    if not np.isfinite(v):
        return "-"
    return f"{v:.{d}f}"


def _scatter_sample_idx(n, max_points=4000, seed=0):
    if n <= max_points:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(n, dtype=int), size=max_points, replace=False))


def _representative_indices(values, n=12):
    values = np.asarray(values, dtype=float)
    if values.size <= n:
        return np.arange(values.size, dtype=int)
    order = np.argsort(values)
    anchors = np.linspace(0, len(order) - 1, n).round().astype(int)
    return np.unique(order[anchors])


def plot_gene_overview(summary: GeneSummary, counts: np.ndarray, totals: np.ndarray):
    counts = np.asarray(counts, dtype=float)
    totals = np.asarray(totals, dtype=float)
    detected = counts > 0
    frac = counts[detected] / np.maximum(totals[detected], 1.0)
    scatter_idx = _scatter_sample_idx(len(counts), max_points=2500, seed=0)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].hist(counts, bins=60, color="#155e75", alpha=0.88)
    axes[0].set_title("Raw gene-count histogram")
    axes[0].set_xlabel("X_gc")
    axes[0].set_ylabel("Cell count")
    axes[0].set_yscale("log")
    axes[1].scatter(
        totals[scatter_idx],
        counts[scatter_idx],
        s=12,
        alpha=0.5,
        color="#0f766e",
        edgecolor="none",
    )
    axes[1].set_title("Gene count vs total UMI")
    axes[1].set_xlabel("N_c")
    axes[1].set_ylabel("X_gc")
    axes[1].set_xscale("log")
    if frac.size > 0:
        axes[2].hist(frac, bins=50, color="#c2410c", alpha=0.82)
        axes[2].set_title("Observed fraction among detected cells")
        axes[2].set_xlabel("X_gc / N_c")
        axes[2].set_ylabel("Cell count")
    else:
        axes[2].text(0.5, 0.5, "No detected cells", ha="center", va="center")
        axes[2].set_axis_off()
    fig.suptitle(f"{summary.gene_name} overview", fontsize=14)
    fig.tight_layout()
    return fig_to_uri(fig)


def plot_stage0(pool_report, totals):
    totals = np.asarray(totals, dtype=float)
    scatter_idx = _scatter_sample_idx(len(totals), max_points=3000, seed=1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(totals, bins=80, color="#155e75", alpha=0.82)
    axes[0].axvline(
        np.median(totals), color="#c2410c", lw=2, ls="--", label="median N_c"
    )
    axes[0].axvline(
        pool_report.point_eta, color="#1d4ed8", lw=2, ls=":", label="exp(mu)=rS"
    )
    axes[0].set_title("Dataset total UMI counts")
    axes[0].set_xlabel("N_c")
    axes[0].set_ylabel("Cell count")
    axes[0].legend(frameon=False)
    axes[1].hist(
        pool_report.eta_posterior_mean,
        bins=80,
        density=True,
        color="#0f766e",
        alpha=0.35,
    )
    axes[1].plot(
        pool_report.eta_prior_grid,
        pool_report.eta_prior_density,
        color="#1d4ed8",
        lw=2.2,
    )
    axes[1].set_title("Poisson-LogNormal fit over eta_c")
    axes[1].set_xlabel("eta_c = rS * epsilon_c")
    axes[1].set_ylabel("Density")
    axes[2].scatter(
        totals[scatter_idx],
        pool_report.eta_posterior_mean[scatter_idx],
        s=10,
        alpha=0.42,
        color="#c2410c",
        edgecolor="none",
    )
    lim = max(
        float(np.quantile(totals, 0.995)),
        float(np.quantile(pool_report.eta_posterior_mean, 0.995)),
        1.0,
    )
    axes[2].plot([0, lim], [0, lim], color="#1f2430", lw=1.2, ls=":")
    axes[2].set_title("Observed N_c vs posterior mean eta_c")
    axes[2].set_xlabel("N_c")
    axes[2].set_ylabel("E[eta_c | N_c]")
    fig.tight_layout()
    return fig_to_uri(fig)


def plot_loss_trace(prior_report):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
    its = np.arange(len(prior_report.loss_history))
    axes[0].plot(its, prior_report.loss_history, color="#1d4ed8", lw=2.2, label="total")
    axes[0].plot(its, prior_report.nll_history, color="#0f766e", lw=1.8, label="NLL")
    axes[0].plot(its, prior_report.align_history, color="#c2410c", lw=1.8, label="JSD")
    axes[0].set_title("Optimization trace")
    axes[0].set_xlabel("Iteration")
    axes[0].legend(frameon=False)
    axes[1].hist(prior_report.prior_weights[0], bins=40, color="#155e75", alpha=0.84)
    axes[1].set_title("Prior weight distribution")
    axes[1].set_xlabel("Weight")
    axes[1].set_ylabel("Grid count")
    fig.tight_layout()
    return fig_to_uri(fig)


def plot_prior_fit(analysis: GeneAnalysis):
    support = analysis.support
    grid_step = max(float(support[1] - support[0]) if support.size > 1 else 1.0, 1e-12)
    prior_density = analysis.prior_weights / grid_step
    q_hat = (
        analysis.prior_report.q_hat[0]
        if analysis.prior_report is not None
        else analysis.prior_weights
    )
    q_density = q_hat / grid_step
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].fill_between(support, prior_density, color="#0f766e", alpha=0.22)
    axes[0].plot(support, prior_density, color="#0f766e", lw=2.2)
    axes[0].set_title("Estimated prior density")
    axes[0].set_xlabel("mu")
    axes[0].set_ylabel("Density")
    axes[1].plot(support, q_density, color="#c2410c", lw=2.0, ls="--", label="Q_hat")
    axes[1].plot(support, prior_density, color="#1d4ed8", lw=2.0, label="F_g")
    axes[1].set_title("Posterior-prior self-consistency")
    axes[1].set_xlabel("mu")
    axes[1].legend(frameon=False)
    axes[2].hist(
        analysis.x_eff,
        bins=50,
        density=True,
        color="#c2410c",
        alpha=0.28,
        label="X_eff",
    )
    axes[2].hist(
        analysis.signal,
        bins=50,
        density=True,
        color="#0f766e",
        alpha=0.28,
        label="Signal",
    )
    axes[2].plot(support, prior_density, color="#1d4ed8", lw=2.0, label="F_g density")
    axes[2].set_title("X_eff vs inferred signal scale")
    axes[2].set_xlabel("Signal scale")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    return fig_to_uri(fig)


def plot_init_comparison(prior_report):
    support = prior_report.support[0]
    grid_step = max(float(support[1] - support[0]) if support.size > 1 else 1.0, 1e-12)
    init_q_density = prior_report.init_q_hat[0] / grid_step
    init_prior_density = prior_report.init_prior_weights[0] / grid_step
    final_prior_density = prior_report.prior_weights[0] / grid_step
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
    axes[0].plot(support, init_q_density, color="#c2410c", lw=2.0, label="init Q_hat")
    axes[0].plot(support, init_prior_density, color="#0f766e", lw=2.0, label="init F_g")
    axes[0].set_title("Initialization bootstrap")
    axes[0].set_xlabel("mu")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    axes[1].plot(
        support, init_prior_density, color="#94a3b8", lw=2.0, ls="--", label="init F_g"
    )
    axes[1].plot(
        support, final_prior_density, color="#1d4ed8", lw=2.2, label="final F_g"
    )
    axes[1].set_title("Initialization vs final prior")
    axes[1].set_xlabel("mu")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return fig_to_uri(fig)


def plot_signal_interface(analysis: GeneAnalysis):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    x_eff = analysis.x_eff
    lim = max(float(np.max(x_eff)), float(np.max(analysis.signal)), 1.0)
    sc0 = axes[0, 0].scatter(
        x_eff,
        analysis.signal,
        c=analysis.confidence,
        cmap="viridis",
        s=14,
        alpha=0.72,
        edgecolor="none",
    )
    axes[0, 0].plot([0, lim], [0, lim], color="#1f2430", lw=1.1, ls=":")
    axes[0, 0].set_title("Signal vs X_eff")
    axes[0, 0].set_xlabel("X_eff = X_gc / N_c * S_hat")
    axes[0, 0].set_ylabel("Signal")
    fig.colorbar(sc0, ax=axes[0, 0], shrink=0.88).set_label("Confidence")
    sc1 = axes[0, 1].scatter(
        x_eff,
        analysis.confidence,
        c=analysis.signal,
        cmap="cividis",
        s=14,
        alpha=0.72,
        edgecolor="none",
    )
    axes[0, 1].set_title("Confidence vs X_eff")
    axes[0, 1].set_xlabel("X_eff = X_gc / N_c * S_hat")
    axes[0, 1].set_ylabel("Confidence")
    fig.colorbar(sc1, ax=axes[0, 1], shrink=0.88).set_label("Signal")
    sc2 = axes[1, 0].scatter(
        analysis.signal,
        analysis.surprisal,
        c=analysis.confidence,
        cmap="magma",
        s=14,
        alpha=0.72,
        edgecolor="none",
    )
    axes[1, 0].set_title("Surprisal vs signal")
    axes[1, 0].set_xlabel("Signal")
    axes[1, 0].set_ylabel("Surprisal")
    fig.colorbar(sc2, ax=axes[1, 0], shrink=0.88).set_label("Confidence")
    sc3 = axes[1, 1].scatter(
        analysis.signal,
        analysis.sharpness,
        c=analysis.surprisal,
        cmap="plasma",
        s=14,
        alpha=0.72,
        edgecolor="none",
    )
    axes[1, 1].set_title("Sharpness vs signal")
    axes[1, 1].set_xlabel("Signal")
    axes[1, 1].set_ylabel("Sharpness")
    fig.colorbar(sc3, ax=axes[1, 1], shrink=0.88).set_label("Surprisal")
    fig.tight_layout()
    return fig_to_uri(fig)


def _binned_surface(x, y, z, *, bins=128):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    grid_axis = np.linspace(0.0, 1.0, bins)
    if bins <= 1:
        raise ValueError("bins must be greater than 1")
    x_idx = np.clip(np.rint(x * (bins - 1)).astype(int), 0, bins - 1)
    y_idx = np.clip(np.rint(y * (bins - 1)).astype(int), 0, bins - 1)
    count_grid = np.zeros((bins, bins), dtype=float)
    value_grid = np.zeros((bins, bins), dtype=float)
    np.add.at(count_grid, (x_idx, y_idx), 1.0)
    np.add.at(value_grid, (x_idx, y_idx), z)
    surface = np.divide(
        value_grid,
        np.maximum(count_grid, 1e-12),
        out=np.full_like(value_grid, np.nan),
        where=count_grid > 0,
    )
    density = np.log1p(count_grid)
    if np.max(density) > 0:
        density = density / np.max(density)
    else:
        density.fill(0.0)
    x_centers = grid_axis
    y_centers = grid_axis
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    return xx, yy, surface.T, density.T


def _surface_payload(xx, yy, zz, density):
    zz_arr = np.asarray(zz, dtype=float)
    density_arr = np.asarray(density, dtype=float)
    return {
        "x": np.asarray(xx, dtype=float).tolist(),
        "y": np.asarray(yy, dtype=float).tolist(),
        "z": zz_arr.tolist(),
        "density": density_arr.tolist(),
    }


def _clip_and_normalize(values):
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.quantile(values, 0.99))
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(values, lo, hi)
    return (clipped - lo) / max(hi - lo, 1e-12), lo, hi


def plot_signal_interface_3d_html(analysis: GeneAnalysis):
    totals = np.asarray(analysis.totals, dtype=float)
    counts = np.asarray(analysis.counts, dtype=float)
    x_eff = np.asarray(analysis.x_eff, dtype=float)
    signal = np.asarray(analysis.signal, dtype=float)
    confidence = np.asarray(analysis.confidence, dtype=float)
    surprisal = np.asarray(analysis.surprisal, dtype=float)
    sharpness = np.asarray(analysis.sharpness, dtype=float)
    idx = _scatter_sample_idx(len(totals), max_points=4500, seed=7)
    xy_modes = {
        "x_nc": {
            "label": "X_gc vs N_c",
            "x": counts,
            "y": totals,
            "x_label": "X_gc",
            "y_label": "N_c",
        },
        "x_log_nc": {
            "label": "X_gc vs log1p(N_c)",
            "x": counts,
            "y": np.log1p(totals),
            "x_label": "X_gc",
            "y_label": "log1p(N_c)",
        },
        "x_over_n_log_nc": {
            "label": "X_gc / N_c vs log1p(N_c)",
            "x": np.divide(counts, np.maximum(totals, 1.0)),
            "y": np.log1p(totals),
            "x_label": "X_gc / N_c",
            "y_label": "log1p(N_c)",
        },
        "neff_log_nc": {
            "label": "X_eff vs log1p(N_c)",
            "x": x_eff,
            "y": np.log1p(totals),
            "x_label": "X_eff",
            "y_label": "log1p(N_c)",
        },
        "neff_over_n": {
            "label": "X_eff vs N_c",
            "x": x_eff,
            "y": totals,
            "x_label": "X_eff",
            "y_label": "N_c",
        },
    }
    color_metrics = {
        "signal": {"label": "Signal", "values": signal, "palette": "Viridis"},
        "confidence": {
            "label": "Confidence",
            "values": confidence,
            "palette": "Cividis",
        },
        "surprisal": {"label": "Surprisal", "values": surprisal, "palette": "Magma"},
        "sharpness": {"label": "Sharpness", "values": sharpness, "palette": "Plasma"},
        "xeff": {"label": "X_eff", "values": x_eff, "palette": "Turbo"},
    }

    def build_payload(z_values, title, z_label, palette):
        z_norm, z_lo, z_hi = _clip_and_normalize(z_values)
        modes_payload = {}
        for key, mode in xy_modes.items():
            x_norm, x_lo, x_hi = _clip_and_normalize(mode["x"])
            y_norm, y_lo, y_hi = _clip_and_normalize(mode["y"])
            xx, yy, zz, density = _binned_surface(x_norm, y_norm, z_norm, bins=128)
            colors_payload = {}
            for metric_key, metric in color_metrics.items():
                c_norm, c_lo, c_hi = _clip_and_normalize(metric["values"])
                colors_payload[metric_key] = {
                    "points": c_norm[idx].astype(float).tolist(),
                    "label": metric["label"],
                    "palette": metric["palette"],
                    "lo": float(c_lo),
                    "hi": float(c_hi),
                }
            modes_payload[key] = {
                "label": mode["label"],
                "x_label": mode["x_label"],
                "y_label": mode["y_label"],
                "surface": _surface_payload(xx, yy, zz, density),
                "points": {
                    "x": x_norm[idx].astype(float).tolist(),
                    "y": y_norm[idx].astype(float).tolist(),
                    "z": z_norm[idx].astype(float).tolist(),
                    "raw_x": mode["x"][idx].astype(float).tolist(),
                    "raw_y": mode["y"][idx].astype(float).tolist(),
                    "raw_z": z_values[idx].astype(float).tolist(),
                },
                "ranges": {
                    "x": {"lo": float(x_lo), "hi": float(x_hi)},
                    "y": {"lo": float(y_lo), "hi": float(y_hi)},
                    "z": {"lo": float(z_lo), "hi": float(z_hi)},
                },
                "colors": colors_payload,
            }
        return {
            "modes": modes_payload,
            "title": title,
            "z_label": z_label,
            "default_palette": palette,
        }

    payload_signal = build_payload(signal, "Signal", "Signal", "viridis")
    payload_conf = build_payload(confidence, "Confidence", "Confidence", "cividis")
    block_id = f"signal3d-{uuid.uuid4().hex[:8]}"
    return f"""
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <div id="{block_id}" class="interactive-3d-grid">
      <div class="interactive-3d-card">
        <div class="interactive-3d-header">Signal in observation space</div>
        <div class="interactive-3d-toolbar">
          <select id="{block_id}-signal-mode" class="interactive-3d-select">
            <option value="x_nc">X_gc vs N_c</option>
            <option value="x_log_nc">X_gc vs log1p(N_c)</option>
            <option value="x_over_n_log_nc">X_gc / N_c vs log1p(N_c)</option>
            <option value="neff_log_nc">X_eff vs log1p(N_c)</option>
            <option value="neff_over_n">X_eff vs N_c</option>
          </select>
          <select id="{block_id}-signal-color" class="interactive-3d-select">
            <option value="confidence">Color: Confidence</option>
            <option value="signal">Color: Signal</option>
            <option value="surprisal">Color: Surprisal</option>
            <option value="sharpness">Color: Sharpness</option>
            <option value="xeff">Color: X_eff</option>
          </select>
          <span class="muted">surface color = density</span>
          <button type="button" class="interactive-3d-btn" id="{block_id}-signal-both">Both</button>
          <button type="button" class="interactive-3d-btn" id="{block_id}-signal-surface">Surface</button>
          <button type="button" class="interactive-3d-btn" id="{block_id}-signal-points">Points</button>
          <button type="button" class="interactive-3d-btn" id="{block_id}-signal-fullscreen">Fullscreen</button>
          <button type="button" class="interactive-3d-btn" id="{block_id}-signal-reset">Reset</button>
          <span class="interactive-3d-legend viridis"></span>
        </div>
        <div id="{block_id}-signal" class="interactive-3d-plot"></div>
      </div>
      <div class="interactive-3d-card">
        <div class="interactive-3d-header">Confidence in observation space</div>
        <div class="interactive-3d-toolbar">
          <select id="{block_id}-confidence-mode" class="interactive-3d-select">
            <option value="x_nc">X_gc vs N_c</option>
            <option value="x_log_nc">X_gc vs log1p(N_c)</option>
            <option value="x_over_n_log_nc">X_gc / N_c vs log1p(N_c)</option>
            <option value="neff_log_nc">X_eff vs log1p(N_c)</option>
            <option value="neff_over_n">X_eff vs N_c</option>
          </select>
          <select id="{block_id}-confidence-color" class="interactive-3d-select">
            <option value="signal">Color: Signal</option>
            <option value="confidence">Color: Confidence</option>
            <option value="surprisal">Color: Surprisal</option>
            <option value="sharpness">Color: Sharpness</option>
            <option value="xeff">Color: X_eff</option>
          </select>
          <span class="muted">surface color = density</span>
          <button type="button" class="interactive-3d-btn" id="{block_id}-confidence-both">Both</button>
          <button type="button" class="interactive-3d-btn" id="{block_id}-confidence-surface">Surface</button>
          <button type="button" class="interactive-3d-btn" id="{block_id}-confidence-points">Points</button>
          <button type="button" class="interactive-3d-btn" id="{block_id}-confidence-fullscreen">Fullscreen</button>
          <button type="button" class="interactive-3d-btn" id="{block_id}-confidence-reset">Reset</button>
          <span class="interactive-3d-legend cividis"></span>
        </div>
        <div id="{block_id}-confidence" class="interactive-3d-plot"></div>
      </div>
    </div>
    <script>
    (function() {{
      const payloads = {{
        signal: {json.dumps(payload_signal)},
        confidence: {json.dumps(payload_conf)}
      }};

      function tickText(range) {{
        const mid = 0.5 * (range.lo + range.hi);
        const q1 = range.lo + (range.hi - range.lo) * 0.25;
        const q3 = range.lo + (range.hi - range.lo) * 0.75;
        return [range.lo.toFixed(2), q1.toFixed(2), mid.toFixed(2), q3.toFixed(2), range.hi.toFixed(2) + ' (p99)'];
      }}

      function createViewer(plotId, payload, modeId, colorId, prefix) {{
        const plot = document.getElementById(plotId);
        const card = plot.closest('.interactive-3d-card');
        const modeSelect = document.getElementById(modeId);
        const colorSelect = document.getElementById(colorId);

        function activeMode() {{ return payload.modes[modeSelect.value]; }}

        function buildData() {{
          const mode = activeMode();
          const color = mode.colors[colorSelect.value];
          const hover = mode.points.raw_x.map((_, i) => [
            mode.points.raw_x[i],
            mode.points.raw_y[i],
            mode.points.raw_z[i],
          ]);
          return [
            {{
              type: 'surface',
              x: mode.surface.x,
              y: mode.surface.y,
              z: mode.surface.z,
              surfacecolor: mode.surface.density,
              colorscale: 'Viridis',
              opacity: 0.84,
              showscale: true,
              visible: document.getElementById(prefix + '-surface').dataset.state !== 'off',
              colorbar: {{ title: 'Density', len: 0.78, x: 1.02 }},
              hovertemplate: mode.x_label + ': %{{x:.3f}}<br>' + mode.y_label + ': %{{y:.3f}}<br>' + payload.z_label + ': %{{z:.3f}}<extra></extra>',
            }},
            {{
              type: 'scatter3d',
              mode: 'markers',
              x: mode.points.x,
              y: mode.points.y,
              z: mode.points.z,
              customdata: hover,
              marker: {{ size: 2.5, color: color.points, colorscale: color.palette, opacity: 0.4, showscale: false }},
              visible: document.getElementById(prefix + '-points').dataset.state !== 'off',
              hovertemplate: mode.x_label + ': %{{customdata[0]:.3f}}<br>' + mode.y_label + ': %{{customdata[1]:.3f}}<br>' + payload.z_label + ': %{{customdata[2]:.3f}}<extra></extra>',
            }},
          ];
        }}

        function buildLayout() {{
          const mode = activeMode();
          return {{
            title: payload.title + ': ' + mode.label,
            margin: {{ l: 0, r: 0, b: 0, t: 36 }},
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            scene: {{
              aspectmode: 'cube',
              xaxis: {{ title: mode.x_label, range: [0, 1], tickvals: [0, 0.25, 0.5, 0.75, 1], ticktext: tickText(mode.ranges.x) }},
              yaxis: {{ title: mode.y_label, range: [0, 1], tickvals: [0, 0.25, 0.5, 0.75, 1], ticktext: tickText(mode.ranges.y) }},
              zaxis: {{ title: payload.z_label, range: [0, 1], tickvals: [0, 0.25, 0.5, 0.75, 1], ticktext: tickText(mode.ranges.z) }},
              camera: {{ eye: {{ x: 1.45, y: 1.45, z: 1.15 }} }},
            }},
          }};
        }}

        function draw() {{
          Plotly.react(plot, buildData(), buildLayout(), {{ responsive: true, displaylogo: false, scrollZoom: true }});
        }}

        modeSelect.addEventListener('change', draw);
        colorSelect.addEventListener('change', draw);
        document.getElementById(prefix + '-both').addEventListener('click', () => {{
          document.getElementById(prefix + '-surface').dataset.state = 'on';
          document.getElementById(prefix + '-points').dataset.state = 'on';
          draw();
        }});
        document.getElementById(prefix + '-surface').addEventListener('click', () => {{
          document.getElementById(prefix + '-surface').dataset.state = 'on';
          document.getElementById(prefix + '-points').dataset.state = 'off';
          draw();
        }});
        document.getElementById(prefix + '-points').addEventListener('click', () => {{
          document.getElementById(prefix + '-surface').dataset.state = 'off';
          document.getElementById(prefix + '-points').dataset.state = 'on';
          draw();
        }});
        document.getElementById(prefix + '-reset').addEventListener('click', () => {{
          modeSelect.selectedIndex = 0;
          draw();
          Plotly.relayout(plot, {{'scene.camera.eye': {{x: 1.45, y: 1.45, z: 1.15}}}});
        }});
        document.getElementById(prefix + '-fullscreen').addEventListener('click', async () => {{
          if (!document.fullscreenElement) {{
            await card.requestFullscreen();
          }} else if (document.fullscreenElement === card) {{
            await document.exitFullscreen();
          }}
        }});

        document.getElementById(prefix + '-surface').dataset.state = 'on';
        document.getElementById(prefix + '-points').dataset.state = 'on';
        window.addEventListener('resize', draw);
        document.addEventListener('fullscreenchange', draw);
        draw();
      }}

      createViewer('{block_id}-signal', payloads.signal, '{block_id}-signal-mode', '{block_id}-signal-color', '{block_id}-signal');
      createViewer('{block_id}-confidence', payloads.confidence, '{block_id}-confidence-mode', '{block_id}-confidence-color', '{block_id}-confidence');
    }})();
    </script>
    """


def plot_posterior_gallery(analysis: GeneAnalysis):
    idx = _representative_indices(analysis.signal, n=12)
    support = analysis.support
    posterior = analysis.posterior_samples
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for ax, cell_idx in zip(axes.flat, idx):
        local_idx = np.where(analysis.posterior_cell_indices == cell_idx)[0]
        if local_idx.size == 0:
            ax.set_axis_off()
            continue
        curve = posterior[local_idx[0]]
        ax.plot(support, curve, color="#1d4ed8", lw=1.8)
        ax.fill_between(support, curve, color="#1d4ed8", alpha=0.18)
        ax.axvline(analysis.signal[cell_idx], color="#c2410c", ls="--", lw=1.4)
        ax.set_title(
            f"x={fmt_int(analysis.counts[cell_idx])}, N={fmt_int(analysis.totals[cell_idx])}\n"
            f"s={fmt_f(analysis.signal[cell_idx], 2)}, conf={fmt_f(analysis.confidence[cell_idx], 2)}",
            fontsize=10,
        )
        ax.set_xlabel("mu")
        ax.set_ylabel("Posterior")
    for ax in axes.flat[len(idx) :]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig_to_uri(fig)


def treatment_block(summary: GeneSummary):
    rows = []
    for row in summary.treatment_table:
        rows.append(
            f"<tr><td>{row['treatment']}</td><td>{fmt_int(row['cells'])}</td><td>{fmt_f(row['mean_count'], 3)}</td><td>{fmt_f(row['detected_frac'], 3)}</td><td>{fmt_f(row['mean_total_umi'], 1)}</td></tr>"
        )
    if not rows:
        return ""
    return f'<div class="panel"><h2>Top treatment groups</h2><table><thead><tr><th>Treatment</th><th>Cells</th><th>Mean count</th><th>Detected frac</th><th>Mean total UMI</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
