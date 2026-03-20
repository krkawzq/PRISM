"""
scPRISM dashboard aligned to `docs/scPRISM_re.md`.

The UI follows the refactored method flow:
1. Estimate the global scalar `S` from total UMI counts.
2. Fit a per-gene prior `F_g` on a discrete grid.
3. Inspect posterior-derived Signal / Confidence / Surprisal / Sharpness.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import uuid
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

from denoise_algo import (
    discover_h5ad_datasets,
    load_scprism_defaults,
    list_dataset_summaries,
    run_real_gene_fit,
    search_gene_candidates,
    summarize_gene_expression,
)


def parse_args():
    defaults = load_scprism_defaults().get("server", {})
    parser = argparse.ArgumentParser(description="scPRISM dashboard")
    parser.add_argument("--host", default=defaults.get("host", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(defaults.get("port", 8000)))
    return parser.parse_args()


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


def qlink(path, **params):
    return f"{path}?{urlencode(params)}"


def pf(v, default, lo=None, hi=None):
    try:
        out = float(v) if v else default
    except Exception:
        out = default
    if lo is not None:
        out = max(out, lo)
    if hi is not None:
        out = min(out, hi)
    return out


def pi(v, default, lo=None, hi=None):
    try:
        out = int(v) if v else default
    except Exception:
        out = default
    if lo is not None:
        out = max(out, lo)
    if hi is not None:
        out = min(out, hi)
    return out


def _scatter_sample_idx(n, max_points=2500, seed=0):
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


def plot_gene_overview(h5ad_path, gene_query):
    summary = summarize_gene_expression(h5ad_path, gene_query)
    counts = summary["x_vals"]
    totals = summary["totals"]
    detected = counts > 0
    frac = counts[detected] / np.maximum(totals[detected], 1.0)
    scatter_idx = _scatter_sample_idx(len(counts), max_points=2500, seed=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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

    fig.tight_layout()
    return fig_to_uri(fig)


def plot_stage0(stage0, totals):
    totals = np.asarray(totals, dtype=float)
    scatter_idx = _scatter_sample_idx(len(totals), max_points=3000, seed=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(totals, bins=80, color="#155e75", alpha=0.82)
    axes[0].axvline(
        np.median(totals), color="#c2410c", lw=2, ls="--", label="median N_c"
    )
    axes[0].axvline(stage0.rs_hat, color="#1d4ed8", lw=2, ls=":", label="exp(mu)=rS")
    axes[0].set_title("Dataset total UMI counts")
    axes[0].set_xlabel("N_c")
    axes[0].set_ylabel("Cell count")
    axes[0].legend(frameon=False)

    axes[1].hist(
        stage0.eta_posterior_mean, bins=80, density=True, color="#0f766e", alpha=0.35
    )
    axes[1].plot(
        stage0.eta_prior_grid, stage0.eta_prior_density, color="#1d4ed8", lw=2.2
    )
    axes[1].set_title("Poisson-LogNormal fit over eta_c")
    axes[1].set_xlabel("eta_c = rS * epsilon_c")
    axes[1].set_ylabel("Density")

    axes[2].scatter(
        totals[scatter_idx],
        stage0.eta_posterior_mean[scatter_idx],
        s=10,
        alpha=0.42,
        color="#c2410c",
        edgecolor="none",
    )
    lim = max(
        float(np.quantile(totals, 0.995)),
        float(np.quantile(stage0.eta_posterior_mean, 0.995)),
        1.0,
    )
    axes[2].plot([0, lim], [0, lim], color="#1f2430", lw=1.2, ls=":")
    axes[2].set_title("Observed N_c vs posterior mean eta_c")
    axes[2].set_xlabel("N_c")
    axes[2].set_ylabel("E[eta_c | N_c]")

    fig.tight_layout()
    return fig_to_uri(fig)


def plot_loss_trace(result):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
    its = np.arange(len(result.loss_history))

    axes[0].plot(its, result.loss_history, color="#1d4ed8", lw=2.2, label="total")
    axes[0].plot(its, result.nll_history, color="#0f766e", lw=1.8, label="NLL")
    axes[0].plot(its, result.align_history, color="#c2410c", lw=1.8, label="JSD")
    axes[0].set_title("Optimization trace")
    axes[0].set_xlabel("Iteration")
    axes[0].legend(frameon=False)

    axes[1].hist(result.confidence, bins=40, color="#155e75", alpha=0.84)
    axes[1].set_title("Posterior confidence distribution")
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Cell count")

    fig.tight_layout()
    return fig_to_uri(fig)


def plot_prior_fit(result):
    support = result.support
    prior_density = result.prior_weights / max(result.grid_step, 1e-12)
    q_density = result.q_hat / max(result.grid_step, 1e-12)

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
        result.x_eff,
        bins=50,
        density=True,
        color="#c2410c",
        alpha=0.28,
        label="X_eff = X/N * S_hat",
    )
    axes[2].hist(
        result.signal,
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


def plot_init_comparison(result):
    support = result.support
    init_q_density = result.init_q_hat / max(result.grid_step, 1e-12)
    init_prior_density = result.init_prior_weights / max(result.grid_step, 1e-12)
    final_prior_density = result.prior_weights / max(result.grid_step, 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))

    axes[0].plot(support, init_q_density, color="#c2410c", lw=2.0, label="init Q_hat")
    axes[0].plot(
        support,
        init_prior_density,
        color="#0f766e",
        lw=2.0,
        label="init F_g",
    )
    axes[0].set_title("Initialization bootstrap")
    axes[0].set_xlabel("mu")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)

    axes[1].plot(
        support,
        init_prior_density,
        color="#94a3b8",
        lw=2.0,
        ls="--",
        label="init F_g",
    )
    axes[1].plot(
        support,
        final_prior_density,
        color="#1d4ed8",
        lw=2.2,
        label="final F_g",
    )
    axes[1].set_title("Initialization vs final prior")
    axes[1].set_xlabel("mu")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    return fig_to_uri(fig)


def plot_signal_interface(result):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    x_eff = result.x_eff
    lim = max(
        float(np.max(x_eff)),
        float(np.max(result.signal)),
        1.0,
    )
    sc0 = axes[0, 0].scatter(
        x_eff,
        result.signal,
        c=result.confidence,
        cmap="viridis",
        s=14,
        alpha=0.72,
        edgecolor="none",
    )
    axes[0, 0].plot([0, lim], [0, lim], color="#1f2430", lw=1.1, ls=":")
    axes[0, 0].set_title("Signal vs X_eff")
    axes[0, 0].set_xlabel("X_eff = X_gc / N_c * S_hat")
    axes[0, 0].set_ylabel("Signal")
    cb0 = fig.colorbar(sc0, ax=axes[0, 0], shrink=0.88)
    cb0.set_label("Confidence")

    sc1 = axes[0, 1].scatter(
        x_eff,
        result.confidence,
        c=result.signal,
        cmap="cividis",
        s=14,
        alpha=0.72,
        edgecolor="none",
    )
    axes[0, 1].set_title("Confidence vs X_eff")
    axes[0, 1].set_xlabel("X_eff = X_gc / N_c * S_hat")
    axes[0, 1].set_ylabel("Confidence")
    cb1 = fig.colorbar(sc1, ax=axes[0, 1], shrink=0.88)
    cb1.set_label("Signal")

    sc2 = axes[1, 0].scatter(
        result.signal,
        result.surprisal,
        c=result.confidence,
        cmap="magma",
        s=14,
        alpha=0.72,
        edgecolor="none",
    )
    axes[1, 0].set_title("Surprisal vs signal")
    axes[1, 0].set_xlabel("Signal")
    axes[1, 0].set_ylabel("Surprisal")
    cb2 = fig.colorbar(sc2, ax=axes[1, 0], shrink=0.88)
    cb2.set_label("Confidence")

    sc3 = axes[1, 1].scatter(
        result.signal,
        result.sharpness,
        c=result.surprisal,
        cmap="plasma",
        s=14,
        alpha=0.72,
        edgecolor="none",
    )
    axes[1, 1].set_title("Sharpness vs signal")
    axes[1, 1].set_xlabel("Signal")
    axes[1, 1].set_ylabel("Sharpness")
    cb3 = fig.colorbar(sc3, ax=axes[1, 1], shrink=0.88)
    cb3.set_label("Surprisal")

    fig.tight_layout()
    return fig_to_uri(fig)


def _smoothed_surface(x, y, z, bins=40, smooth_sigma=1.2, min_density_rel=0.03):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if x.size == 0:
        raise ValueError("empty inputs for surface plot")

    x_lo, x_hi = np.quantile(x, [0.01, 0.99])
    y_lo, y_hi = np.quantile(y, [0.01, 0.99])
    if x_hi <= x_lo:
        x_hi = x_lo + 1.0
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0

    x_edges = np.linspace(x_lo, x_hi, bins + 1)
    y_edges = np.linspace(y_lo, y_hi, bins + 1)

    count_grid, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    value_grid, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=z)

    count_s = gaussian_filter(count_grid, sigma=smooth_sigma, mode="nearest")
    value_s = gaussian_filter(value_grid, sigma=smooth_sigma, mode="nearest")
    surface = np.divide(value_s, np.maximum(count_s, 1e-12))
    density_floor = max(float(np.max(count_s)) * float(min_density_rel), 1e-8)
    valid_mask = count_s >= density_floor
    surface = np.where(valid_mask, surface, np.nan)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    zz = surface.T
    return xx, yy, zz, valid_mask.T


def _surface_payload(xx, yy, zz, mask=None):
    zz_arr = np.asarray(zz, dtype=float)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        zz_arr = np.where(mask_arr, zz_arr, np.nan)
    return {
        "x": np.asarray(xx, dtype=float).tolist(),
        "y": np.asarray(yy, dtype=float).tolist(),
        "z": zz_arr.tolist(),
    }


def _clip_and_normalize(values):
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.quantile(values, 0.95))
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(values, lo, hi)
    normalized = (clipped - lo) / max(hi - lo, 1e-12)
    return normalized, lo, hi


def plot_signal_interface_3d_html(result):
    totals = np.asarray(result.totals, dtype=float)
    counts = np.asarray(result.counts, dtype=float)
    x_eff = np.asarray(result.x_eff, dtype=float)
    signal = np.asarray(result.signal, dtype=float)
    confidence = np.asarray(result.confidence, dtype=float)
    surprisal = np.asarray(result.surprisal, dtype=float)
    sharpness = np.asarray(result.sharpness, dtype=float)
    idx = _scatter_sample_idx(len(totals), max_points=1800, seed=7)

    xy_modes = {
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
            xx, yy, zz, surface_mask = _smoothed_surface(
                x_norm, y_norm, z_norm, bins=36, smooth_sigma=1.0
            )
            colors_payload = {}
            for metric_key, metric in color_metrics.items():
                c_norm, c_lo, c_hi = _clip_and_normalize(metric["values"])
                _, _, c_surface, c_mask = _smoothed_surface(
                    x_norm, y_norm, c_norm, bins=36, smooth_sigma=1.0
                )
                colors_payload[metric_key] = {
                    "surface": np.where(
                        c_mask, np.asarray(c_surface, dtype=float), np.nan
                    ).tolist(),
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
                "surface": _surface_payload(xx, yy, zz, surface_mask),
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
        return [range.lo.toFixed(2), mid.toFixed(2), range.hi.toFixed(2) + ' (p95)'];
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
              surfacecolor: color.surface,
              colorscale: color.palette,
              opacity: 0.84,
              showscale: true,
              visible: document.getElementById(prefix + '-surface').dataset.state !== 'off',
              colorbar: {{ title: color.label, len: 0.78, x: 1.02 }},
              hovertemplate: mode.x_label + ': %{{x:.3f}}<br>' + mode.y_label + ': %{{y:.3f}}<br>' + payload.z_label + ': %{{z:.3f}}<extra></extra>',
            }},
            {{
              type: 'scatter3d',
              mode: 'markers',
              x: mode.points.x,
              y: mode.points.y,
              z: mode.points.z,
              customdata: hover,
              marker: {{ size: 3, color: color.points, colorscale: color.palette, opacity: 0.32, showscale: false }},
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
              xaxis: {{ title: mode.x_label, range: [0, 1], tickvals: [0, 0.5, 1], ticktext: tickText(mode.ranges.x) }},
              yaxis: {{ title: mode.y_label, range: [0, 1], tickvals: [0, 0.5, 1], ticktext: tickText(mode.ranges.y) }},
              zaxis: {{ title: payload.z_label, range: [0, 1], tickvals: [0, 0.5, 1], ticktext: tickText(mode.ranges.z) }},
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


def plot_posterior_gallery(result):
    idx = _representative_indices(result.signal, n=12)
    support = result.support
    posterior = result.posterior

    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for ax, cell_idx in zip(axes.flat, idx):
        ax.plot(support, posterior[cell_idx], color="#1d4ed8", lw=1.8)
        ax.fill_between(support, posterior[cell_idx], color="#1d4ed8", alpha=0.18)
        ax.axvline(result.signal[cell_idx], color="#c2410c", ls="--", lw=1.4)
        ax.set_title(
            f"x={fmt_int(result.counts[cell_idx])}, N={fmt_int(result.totals[cell_idx])}\n"
            f"s={fmt_f(result.signal[cell_idx], 2)}, conf={fmt_f(result.confidence[cell_idx], 2)}",
            fontsize=10,
        )
        ax.set_xlabel("mu")
        ax.set_ylabel("Posterior")

    for ax in axes.flat[len(idx) :]:
        ax.set_axis_off()

    fig.tight_layout()
    return fig_to_uri(fig)


def page(title, body):
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>
:root {{ --bg:#f6f4ef;--panel:#fffdf8;--ink:#1f2430;--muted:#5d6675;--line:#d7d1c2;
  --accent:#0f766e;--accent2:#c2410c;--accent3:#1d4ed8;--shadow:0 18px 45px rgba(31,36,48,.08); }}
* {{ box-sizing:border-box; }} body {{ margin:0;color:var(--ink);background:var(--bg);
  font-family:Georgia,"Times New Roman",serif; }}
a {{ color:var(--accent3);text-decoration:none; }} a:hover {{ text-decoration:underline; }}
.wrap {{ max-width:1440px;margin:0 auto;padding:28px 20px 48px; }}
.hero {{ background:rgba(255,255,255,.82);border:1px solid rgba(215,209,194,.85);
  border-radius:28px;padding:28px;box-shadow:var(--shadow);margin-bottom:20px; }}
.hero h1 {{ margin:0 0 8px;font-size:32px; }}
.hero p {{ margin:0;color:var(--muted);font-size:16px;line-height:1.5; }}
.grid {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px;margin-top:18px; }}
.panel {{ background:rgba(255,253,248,.95);border:1px solid rgba(215,209,194,.9);
  border-radius:22px;padding:20px;box-shadow:var(--shadow);margin-bottom:18px; }}
.panel h2,.panel h3 {{ margin:0 0 12px; }}
.meta {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:14px 0 4px; }}
.stat {{ padding:10px 12px;border-radius:14px;background:#f2ede3;border:1px solid #e0d8c7; }}
.stat .k {{ display:block;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:4px; }}
.stat .v {{ font-size:19px;font-weight:700; }}
.toolbar {{ display:flex;flex-wrap:wrap;gap:8px;margin-top:14px; }}
.toolbar input,.toolbar select {{ min-height:38px;padding:8px 10px;border:1px solid var(--line);
  border-radius:10px;background:white;font:inherit; }}
.toolbar button,.btn {{ display:inline-block;min-height:38px;padding:8px 14px;border:0;
  border-radius:999px;background:linear-gradient(135deg,var(--accent),#155e75);
  color:white;cursor:pointer;font:inherit; }}
.btn.ghost {{ background:#ece7db;color:var(--ink); }}
.fit-form {{ margin-top:16px; }}
.fit-grid {{ display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px; }}
.fit-field {{ padding:12px 14px;border-radius:16px;background:#f7f3ea;border:1px solid #e6decd; }}
.fit-field label {{ display:block;margin-bottom:8px;font-size:12px;letter-spacing:.04em;color:var(--muted); }}
.fit-field input,.fit-field select {{ width:100%;min-height:42px;padding:8px 12px;border:1px solid var(--line);
  border-radius:12px;background:white;font:inherit; }}
.fit-footer {{ display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:16px;
  padding-top:14px;border-top:1px solid #ebe4d8;flex-wrap:wrap; }}
.fit-note {{ padding:10px 14px;border-radius:14px;background:#eef6f5;border:1px solid #cfe2df;color:#115e59; }}
.fit-submit {{ min-width:132px;justify-content:center; }}
.explain-list {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:12px; }}
.explain-item {{ padding:14px 15px;border-radius:16px;background:#f7f3ea;border:1px solid #e6decd; }}
.explain-item strong {{ display:block;margin-bottom:6px;font-size:13px; }}
.explain-item p {{ margin:0;color:var(--muted);font-size:13px;line-height:1.55; }}
.figure {{ margin-top:14px; }}
.figure img {{ width:100%;border-radius:16px;border:1px solid var(--line);background:white; }}
.interactive-3d-grid {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px; }}
.interactive-3d-card {{ background:white;border:1px solid var(--line);border-radius:16px;padding:12px; }}
.interactive-3d-card:fullscreen {{ width:100vw;height:100vh;padding:18px;border-radius:0;display:flex;flex-direction:column; }}
.interactive-3d-card:fullscreen .interactive-3d-plot {{ flex:1;height:calc(100vh - 140px); }}
.interactive-3d-header {{ font-size:14px;font-weight:700;margin:0 0 10px; }}
.interactive-3d-toolbar {{ display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 10px; }}
.interactive-3d-btn,.interactive-3d-select {{ min-height:32px;padding:6px 10px;border:1px solid var(--line);border-radius:999px;background:#f8fafc;font:inherit; }}
.interactive-3d-btn {{ cursor:pointer; }}
.interactive-3d-legend {{ width:88px;height:12px;border-radius:999px;border:1px solid #cbd5e1;display:inline-block;margin-left:auto; }}
.interactive-3d-legend.viridis {{ background:linear-gradient(90deg,#440154,#31688e,#35b779,#fde725); }}
.interactive-3d-legend.cividis {{ background:linear-gradient(90deg,#00224e,#575d6d,#a59c74,#fee838); }}
.interactive-3d-plot {{ width:100%;height:420px;border-radius:12px;background:white;display:block;overflow:hidden; }}
table {{ width:100%;border-collapse:collapse;font-size:14px; }}
th,td {{ padding:8px;border-bottom:1px solid #ebe4d8;text-align:left; }}
th {{ color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em; }}
.muted {{ color:var(--muted); }}
.mono {{ font-family:"Courier New",monospace; }}
@media (max-width: 980px) {{ .fit-grid {{ grid-template-columns:1fr; }} }}
@media (max-width: 700px) {{
  .wrap {{ padding:20px 14px 36px; }}
  .hero,.panel {{ padding:18px;border-radius:18px; }}
  .hero h1 {{ font-size:26px; }}
  .fit-submit {{ width:100%; }}
}}
</style></head><body><div class="wrap">{body}</div></body></html>"""


def nav_bar(ds_key, query=""):
    opts = []
    for item in list_dataset_summaries():
        selected = " selected" if item["dataset_key"] == ds_key else ""
        opts.append(
            f'<option value="{escape(item["dataset_key"])}"{selected}>{escape(item["dataset_key"])}</option>'
        )
    return f"""<div class="panel">
      <form class="toolbar" action="/gene" method="get">
        <select name="dataset">{"".join(opts)}</select>
        <input type="text" name="q" value="{escape(query)}" placeholder="输入基因 ID、名称或索引">
        <button type="submit">搜索</button>
        <a class="btn ghost" href="{qlink("/", dataset=ds_key)}">首页</a>
      </form></div>"""


def parameter_explanations():
    items = [
        (
            "Stage 0: S estimation",
            "先只用全数据集的 total UMI 拟合 Poisson-LogNormal。EM 输出 exp(mu)=rS，再结合给定 r 得到全局标尺 S_hat。",
        ),
        (
            "采样率 r",
            "r 不是在基因级别单独拟合，而是通过 S_hat = exp(mu) / r 改变 Binomial 分母大小，从而影响后验中先验和似然的相对权重。",
        ),
        (
            "离散网格 M",
            "每个基因的 F_g 在统一离散网格上表示。M 越大分辨率越高，但训练也更慢。",
        ),
        (
            "卷积 sigma_bins",
            "先做 softmax，再在网格上做高斯平滑。它控制最小可分辨结构宽度，是文档中的核心分辨率超参。当前默认值按你的新设置使用 1。",
        ),
        (
            "JSD 对齐",
            "训练目标固定为 NLL + JSD。JSD 比较当前先验 F_g 与由后验点决策汇总得到的 Q_hat。",
        ),
        (
            "后验生成分布",
            "可把完整 posterior average 作为 Q_hat 的消融版本。它更自然，但通常更宽，可能更容易淹没稀疏窄峰；当前默认改为 posterior generative。",
        ),
        (
            "初始化",
            "初始化策略现在是可选消融。可以直接用均匀初始化，也可以用 posterior bootstrap；当前默认使用最早的 bootstrap + raw observation。",
        ),
        (
            "初始化温度",
            "写回初始化 logits 时会除以一个 softmax 温度。默认值为标准 softmax 的 1.0；温度更高时，初始分布会更平。",
        ),
        (
            "决策温度 T",
            "T=0 使用 MAP，和文档默认一致。T>0 时用温度加权均值生成软决策，再用线性插值汇总到网格。",
        ),
        (
            "抽样细胞数",
            "Stage 0 始终用全数据集；这里只限制单基因先验拟合参与优化的细胞数。填 0 时默认使用全部细胞。",
        ),
        (
            "Signal interface",
            "最终输出固定为 Signal、Confidence、Surprisal 和 Sharpness。它们都由最终后验和 F_g 直接计算。",
        ),
    ]
    cards = []
    for title, desc in items:
        cards.append(
            f'<div class="explain-item"><strong>{title}</strong><p>{desc}</p></div>'
        )
    return f'<div class="panel"><h2>参数说明</h2><div class="explain-list">{"".join(cards)}</div></div>'


def fit_controls(ds_key, query, p):
    def inp(name, val, label, step="any", min_v="0", max_v=None):
        max_attr = f' max="{max_v}"' if max_v is not None else ""
        return (
            f'<div class="fit-field">'
            f'<label for="fit-{name}">{label}</label>'
            f'<input id="fit-{name}" type="number" step="{step}" min="{min_v}"{max_attr} '
            f'name="{name}" value="{val}">'
            f"</div>"
        )

    def sel(name, value, label, options):
        opts = []
        for opt_value, opt_label in options:
            selected = " selected" if value == opt_value else ""
            opts.append(
                f'<option value="{escape(opt_value)}"{selected}>{escape(opt_label)}</option>'
            )
        return (
            f'<div class="fit-field">'
            f'<label for="fit-{name}">{label}</label>'
            f'<select id="fit-{name}" name="{name}">{"".join(opts)}</select>'
            f"</div>"
        )

    return f"""<div class="panel">
      <h2>拟合面板</h2>
      <p class="muted">固定使用文档流程：估计 S_hat，再在 per-gene 网格上优化 F_g，最后输出完整 posterior interface。</p>
      <form class="fit-form" action="/gene" method="get">
        <input type="hidden" name="dataset" value="{escape(ds_key)}">
        <input type="hidden" name="q" value="{escape(query)}">
        <input type="hidden" name="fit" value="1">
        <div class="fit-grid">
          {inp("r", fmt_f(p["r"], 4), "采样率 r", min_v="0.0001", max_v="1")}
          {inp("max_cells", int(p["max_cells_fit"]), "拟合细胞数 (0=全部)", step="1", min_v="0")}
          {inp("n_iter", int(p["n_iter"]), "迭代次数", step="1", min_v="1")}
          {inp("lr", fmt_f(p["lr"], 4), "学习率")}
          {inp("grid", int(p["grid_size"]), "网格大小 M", step="1", min_v="64")}
          {inp("sigma", fmt_f(p["sigma_bins"], 2), "卷积 sigma_bins", min_v="0")}
          {sel("align_distribution", p["align_distribution"], "后验对齐分布", [("posterior_average", "posterior generative (default)"), ("map_histogram", "MAP histogram")])}
          {sel("warm_start_mode", p["warm_start_mode"], "warm start", [("posterior_average", "posterior average (recommended)"), ("map_histogram", "MAP histogram")])}
          {sel("init_strategy", p["init_strategy"], "初始化策略", [("bootstrap_raw_observation", "bootstrap + raw observation (default)"), ("bootstrap_zero_to_first_positive", "bootstrap + zero->first positive"), ("uniform", "uniform logits")])}
          {inp("init_temperature", fmt_f(p["init_temperature"], 2), "初始化 softmax 温度", min_v="1")}
          {inp("align", fmt_f(p["align_weight"], 3), "JSD 对齐权重", min_v="0")}
          {inp("temperature", fmt_f(p["decision_temperature"], 3), "决策温度 T", min_v="0")}
          {inp("seed", int(p["seed"]), "抽样随机种子", step="1", min_v="0")}
        </div>
        <div class="fit-footer">
          <div class="fit-note">优化器固定为 Adam，调度器固定为 cosine annealing</div>
          <button class="fit-submit" type="submit">开始拟合</button>
        </div>
      </form>
    </div>"""


def parse_fit_params(params):
    defaults = load_scprism_defaults().get("fit", {})
    align_distribution = params.get("align_distribution", ["posterior_average"])[0]
    if align_distribution not in {"map_histogram", "posterior_average"}:
        align_distribution = str(
            defaults.get("align_distribution", "posterior_average")
        )

    warm_start_mode = params.get(
        "warm_start_mode", [str(defaults.get("warm_start_mode", "posterior_average"))]
    )[0]
    if warm_start_mode not in {"map_histogram", "posterior_average"}:
        warm_start_mode = str(defaults.get("warm_start_mode", "posterior_average"))

    init_strategy = params.get(
        "init_strategy",
        [str(defaults.get("init_strategy", "bootstrap_raw_observation"))],
    )[0]
    if init_strategy not in {
        "bootstrap_zero_to_first_positive",
        "bootstrap_raw_observation",
        "uniform",
    }:
        init_strategy = str(defaults.get("init_strategy", "bootstrap_raw_observation"))

    return {
        "r": pf(params.get("r", [None])[0], float(defaults.get("r", 0.05)), 1e-6, 1.0),
        "max_cells_fit": pi(
            params.get("max_cells", [None])[0], int(defaults.get("max_cells_fit", 0)), 0
        ),
        "n_iter": pi(
            params.get("n_iter", [None])[0], int(defaults.get("n_iter", 60)), 1
        ),
        "lr": pf(params.get("lr", [None])[0], float(defaults.get("lr", 0.05)), 1e-6),
        "grid_size": pi(
            params.get("grid", [None])[0], int(defaults.get("grid_size", 512)), 64
        ),
        "sigma_bins": pf(
            params.get("sigma", [None])[0], float(defaults.get("sigma_bins", 1.0)), 0.0
        ),
        "align_distribution": align_distribution,
        "warm_start_mode": warm_start_mode,
        "init_strategy": init_strategy,
        "init_temperature": pf(
            params.get("init_temperature", [None])[0],
            float(defaults.get("init_temperature", 1.0)),
            1.0,
        ),
        "align_weight": pf(
            params.get("align", [None])[0],
            float(defaults.get("align_weight", 1.0)),
            0.0,
        ),
        "decision_temperature": pf(
            params.get("temperature", [None])[0],
            float(defaults.get("decision_temperature", 0.0)),
            0.0,
        ),
        "seed": pi(params.get("seed", [None])[0], int(defaults.get("seed", 0)), 0),
    }


def treatment_block(summary):
    rows = []
    for row in summary["treatment_table"]:
        rows.append(
            f"<tr><td>{escape(row['treatment'])}</td>"
            f"<td>{fmt_int(row['cells'])}</td>"
            f"<td>{fmt_f(row['mean_count'], 3)}</td>"
            f"<td>{fmt_f(row['detected_frac'], 3)}</td>"
            f"<td>{fmt_f(row['mean_total_umi'], 1)}</td></tr>"
        )
    if not rows:
        return ""
    return f"""<div class="panel">
      <h2>Top treatment groups</h2>
      <table><thead><tr><th>Treatment</th><th>Cells</th><th>Mean count</th><th>Detected frac</th><th>Mean total UMI</th></tr></thead>
      <tbody>{"".join(rows)}</tbody></table>
    </div>"""


def _terminal_progress_bar(current, total, width=28):
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    filled = int(round(width * current / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _make_fit_logger(gene_id, dataset_key, total_iter):
    total_iter = max(int(total_iter), 1)

    def _callback(iter_idx, stats):
        current = iter_idx + 1
        bar = _terminal_progress_bar(current, total_iter)
        print(
            f"[scPRISM] fit {dataset_key}/{gene_id} {bar} {current}/{total_iter} "
            f"loss={stats['loss']:.4f} nll={stats['nll']:.4f} align={stats['align']:.4f}",
            flush=True,
        )

    return _callback


def render_home(ds_key):
    datasets = discover_h5ad_datasets()
    h5ad = datasets[ds_key]
    summary = summarize_gene_expression(h5ad, 0)
    bundle_rows = []
    for gid in search_gene_candidates(h5ad, "", limit=15):
        link = qlink("/gene", dataset=ds_key, q=gid["gene_id"])
        bundle_rows.append(
            f"<tr><td><a href='{link}'>{escape(gid['gene_id'])}</a></td>"
            f"<td>{escape(gid['gene_name'])}</td>"
            f"<td>{fmt_int(gid['total_umi'])}</td>"
            f"<td>{fmt_int(gid['detected_cells'])}</td></tr>"
        )

    body = f"""
    <section class="hero">
      <h1>scPRISM 重构版面板</h1>
      <p>这一版直接遵循 `docs/scPRISM_re.md`：先从 total UMI 估计全局标尺 S_hat，再逐基因拟合 F_g，最后输出 Signal / Confidence / Surprisal / Sharpness。</p>
      <div class="meta">
        <div class="stat"><span class="k">数据集</span><span class="v">{escape(ds_key)}</span></div>
        <div class="stat"><span class="k">示例基因</span><span class="v">{escape(summary["gene_name"])}</span></div>
      </div>
    </section>
    {nav_bar(ds_key)}
    <div class="panel"><h2>高表达基因</h2>
      <table><thead><tr><th>基因 ID</th><th>名称</th><th>总 UMI</th><th>检出细胞</th></tr></thead>
      <tbody>{"".join(bundle_rows)}</tbody></table>
    </div>
    {parameter_explanations()}"""
    return page("scPRISM 重构版面板", body)


def render_gene(ds_key, query, do_fit, fp):
    datasets = discover_h5ad_datasets()
    h5ad = datasets[ds_key]
    summary = summarize_gene_expression(h5ad, query)
    gene_plot = plot_gene_overview(h5ad, query)

    fit_block = ""
    if do_fit:
        print(
            f"[scPRISM] start dataset={ds_key} gene={summary['gene_id']} r={fp['r']:.4f} cells={'all' if fp['max_cells_fit'] == 0 else fp['max_cells_fit']} grid={fp['grid_size']} sigma_bins={fp['sigma_bins']:.2f} align_q={fp['align_distribution']} warm_start={fp['warm_start_mode']} init={fp['init_strategy']} init_temp={fp['init_temperature']:.2f} iter={fp['n_iter']}",
            flush=True,
        )
        result = run_real_gene_fit(
            h5ad,
            query,
            max_cells_fit=fp["max_cells_fit"],
            r=fp["r"],
            grid_size=fp["grid_size"],
            sigma_bins=fp["sigma_bins"],
            decision_temperature=fp["decision_temperature"],
            align_distribution=fp["align_distribution"],
            warm_start_mode=fp["warm_start_mode"],
            init_strategy=fp["init_strategy"],
            init_temperature=fp["init_temperature"],
            align_weight=fp["align_weight"],
            lr=fp["lr"],
            n_iter=fp["n_iter"],
            seed=fp["seed"],
            callback=_make_fit_logger(summary["gene_id"], ds_key, fp["n_iter"]),
        )
        stage0 = result["stage0"]
        stage1 = result["stage1"]
        print(
            f"[scPRISM] completed dataset={ds_key} gene={summary['gene_id']} final_loss={stage1.loss_history[-1]:.4f} mean_conf={np.mean(stage1.confidence):.4f} s_hat={stage0.s_hat:.2f}",
            flush=True,
        )

        img_stage0 = plot_stage0(stage0, summary["totals"])
        img_trace = plot_loss_trace(stage1)
        img_init = plot_init_comparison(stage1)
        img_prior = plot_prior_fit(stage1)
        img_signal = plot_signal_interface(stage1)
        html_signal_3d = plot_signal_interface_3d_html(stage1)
        img_post = plot_posterior_gallery(stage1)

        fit_block = f"""
        <div class="panel">
          <h2>Stage 0: 采样池标尺估计</h2>
          <p class="muted">文档中的 Stage 0 只输出全局标尺 S_hat。这里同时展示 Poisson-LogNormal 的 eta_c 分布拟合，便于检查 exp(mu)=rS 的稳定性。</p>
          <div class="meta">
            <div class="stat"><span class="k">r</span><span class="v">{fmt_f(fp["r"], 4)}</span></div>
            <div class="stat"><span class="k">mu</span><span class="v">{fmt_f(stage0.mu, 3)}</span></div>
            <div class="stat"><span class="k">sigma</span><span class="v">{fmt_f(stage0.sigma, 3)}</span></div>
            <div class="stat"><span class="k">rS_hat</span><span class="v">{fmt_f(stage0.rs_hat, 1)}</span></div>
            <div class="stat"><span class="k">S_hat</span><span class="v">{fmt_f(stage0.s_hat, 1)}</span></div>
            <div class="stat"><span class="k">EM 迭代</span><span class="v">{fmt_int(stage0.n_iter)}</span></div>
            <div class="stat"><span class="k">LogLik</span><span class="v">{fmt_f(stage0.loglik, 1)}</span></div>
          </div>
          <div class="figure"><img src="{img_stage0}"></div>
        </div>

        <div class="panel">
          <h2>Stage 1: 基因先验估计</h2>
          <p class="muted">当前实现严格使用离散网格、softmax、Gaussian smoothing、以及 `NLL + JSD` 目标。默认使用 posterior generative distribution 做 JSD 对齐；初始化默认使用最早的 bootstrap + raw observation，同时保留其他初始化策略作为消融。</p>
          <div class="meta">
            <div class="stat"><span class="k">细胞数</span><span class="v">{fmt_int(len(stage1.counts))}</span></div>
            <div class="stat"><span class="k">迭代次数</span><span class="v">{fmt_int(stage1.config["n_iter"])}</span></div>
            <div class="stat"><span class="k">最终损失</span><span class="v">{fmt_f(stage1.loss_history[-1], 4)}</span></div>
            <div class="stat"><span class="k">最终 NLL</span><span class="v">{fmt_f(stage1.nll_history[-1], 4)}</span></div>
            <div class="stat"><span class="k">最终 JSD</span><span class="v">{fmt_f(stage1.align_history[-1], 4)}</span></div>
            <div class="stat"><span class="k">grid size</span><span class="v">{fmt_int(stage1.config["grid_size"])}</span></div>
            <div class="stat"><span class="k">sigma_bins</span><span class="v">{fmt_f(stage1.config["sigma_bins"], 2)}</span></div>
            <div class="stat"><span class="k">temperature</span><span class="v">{fmt_f(stage1.config["decision_temperature"], 3)}</span></div>
            <div class="stat"><span class="k">align q</span><span class="v">{escape(stage1.config["align_distribution"])}</span></div>
            <div class="stat"><span class="k">warm start</span><span class="v">{escape(stage1.config["warm_start_mode"])}</span></div>
            <div class="stat"><span class="k">init temp</span><span class="v">{fmt_f(stage1.config["init_temperature"], 2)}</span></div>
            <div class="stat"><span class="k">align weight</span><span class="v">{fmt_f(stage1.config["align_weight"], 3)}</span></div>
            <div class="stat"><span class="k">lr</span><span class="v">{fmt_f(stage1.config["lr"], 4)}</span></div>
            <div class="stat"><span class="k">init</span><span class="v">{escape(stage1.config["init_strategy"])}</span></div>
            <div class="stat"><span class="k">support max</span><span class="v">{fmt_f(stage1.support_max, 2)}</span></div>
            <div class="stat"><span class="k">mean confidence</span><span class="v">{fmt_f(np.mean(stage1.confidence), 3)}</span></div>
            <div class="stat"><span class="k">P95 surprisal</span><span class="v">{fmt_f(np.quantile(stage1.surprisal, 0.95), 3)}</span></div>
          </div>
          <div class="figure"><img src="{img_trace}"></div>
        </div>

        <div class="panel">
          <h2>Initialization ablation</h2>
          <p class="muted">展示初始化阶段聚合得到的 `Q_hat`、写回 logits 后的初始 `F_g`，以及它和最终 `F_g` 的差异，便于比较不同初始化消融设置。</p>
          <div class="figure"><img src="{img_init}"></div>
        </div>

        <div class="panel">
          <h2>Prior 与自洽结构</h2>
          <p class="muted">左图是估计得到的 F_g；中图比较 Q_hat 与 F_g 的对齐；右图比较 `X_eff = X_gc / N_c * S_hat`、Signal 和 F_g 的密度形状。</p>
          <div class="figure"><img src="{img_prior}"></div>
        </div>

        <div class="panel">
          <h2>Signal interface</h2>
          <p class="muted">Signal 是 MAP 或温度决策值；Signal 和 Confidence 图的横轴统一使用 `X_eff = X_gc / N_c * S_hat`；Confidence 来自后验熵；Surprisal 是 `-log F_g(signal)`；Sharpness 是 log-prior 的离散二阶曲率。</p>
          <div class="figure"><img src="{img_signal}"></div>
        </div>

        <div class="panel">
          <h2>3D signal space</h2>
          <p class="muted">支持自由旋转、缩放，并可在 `X_gc vs log1p(N_c)`、`X_gc / N_c vs log1p(N_c)`、`X_eff vs log1p(N_c)`、`X_eff vs N_c` 四种底面定义之间切换；同时可单独查看 surface、points 或两者叠加。</p>
          <div class="figure">{html_signal_3d}</div>
        </div>

        <div class="panel">
          <h2>Posterior gallery</h2>
          <p class="muted">按 signal 分位数挑代表细胞，直接查看最终 posterior 形状和点决策位置。</p>
          <div class="figure"><img src="{img_post}"></div>
        </div>
        """

    search_rows = []
    for row in search_gene_candidates(h5ad, query, limit=10):
        link = qlink("/gene", dataset=ds_key, q=row["gene_id"])
        search_rows.append(
            f"<tr><td><a href='{link}'>{escape(row['gene_id'])}</a></td>"
            f"<td>{escape(row['gene_name'])}</td>"
            f"<td>{fmt_int(row['total_umi'])}</td>"
            f"<td>{fmt_int(row['detected_cells'])}</td></tr>"
        )

    body = f"""
    <section class="hero">
      <h1>{escape(summary["gene_name"])} <span class="mono">({escape(summary["gene_id"])})</span></h1>
      <p>数据集：{escape(ds_key)}。当前页按重构文档展示 `S_hat` 估计、逐基因 `F_g` 拟合，以及完整 posterior signal interface。</p>
      <div class="meta">
        <div class="stat"><span class="k">索引</span><span class="v">{fmt_int(summary["gene_index"])}</span></div>
        <div class="stat"><span class="k">检出细胞</span><span class="v">{fmt_int(summary["detected_cells"])}</span></div>
        <div class="stat"><span class="k">检出比例</span><span class="v">{fmt_f(summary["detected_frac"], 4)}</span></div>
        <div class="stat"><span class="k">总 UMI</span><span class="v">{fmt_int(summary["total_counts"])}</span></div>
        <div class="stat"><span class="k">均值</span><span class="v">{fmt_f(summary["mean_count"], 3)}</span></div>
        <div class="stat"><span class="k">与 total 相关</span><span class="v">{fmt_f(summary["count_total_correlation"], 4)}</span></div>
      </div>
    </section>
    {nav_bar(ds_key, query)}
    <div class="grid">
      <div class="panel"><h2>搜索结果</h2>
        <table><thead><tr><th>基因</th><th>名称</th><th>UMI</th><th>检出</th></tr></thead>
          <tbody>{"".join(search_rows)}</tbody></table>
      </div>
      {fit_controls(ds_key, query, fp)}
    </div>
    {parameter_explanations()}
    <div class="panel">
      <h2>基因概览</h2>
      <div class="figure"><img src="{gene_plot}"></div>
    </div>
    {treatment_block(summary)}
    {fit_block}
    """
    return page(f"{summary['gene_id']} - scPRISM 重构版面板", body)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        datasets = discover_h5ad_datasets()
        default_ds = next(iter(datasets))
        ds_key = params.get("dataset", [default_ds])[0]
        if ds_key not in datasets:
            ds_key = default_ds
        fp = parse_fit_params(params)

        try:
            if parsed.path == "/":
                html = render_home(ds_key)
            elif parsed.path == "/gene":
                query = params.get("q", [""])[0].strip()
                if not query:
                    html = render_home(ds_key)
                else:
                    do_fit = params.get("fit", ["0"])[0] == "1"
                    html = render_gene(ds_key, query, do_fit, fp)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            payload = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            err = page(
                "错误",
                f'<div class="hero"><h1>出错了</h1><p>{escape(str(exc))}</p></div>{nav_bar(ds_key)}',
            )
            payload = err.encode("utf-8")
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, format, *args):
        print("[scPRISM]", format % args)


def main():
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"scPRISM 已启动：http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
