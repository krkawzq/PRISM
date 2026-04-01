from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import KBulkComparison, GeneAnalysis, GeneSummary
from .global_eval import GlobalEvaluationResult

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def fig_to_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def plot_gene_overview(
    summary: GeneSummary, counts: np.ndarray, totals: np.ndarray
) -> str:
    counts = np.asarray(counts, dtype=float)
    totals = np.asarray(totals, dtype=float)
    frac = counts / np.maximum(totals, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(counts, bins=60, color="#155e75", alpha=0.85)
    axes[0].set_title("Raw gene-count histogram")
    axes[0].set_xlabel("X_gc")
    axes[1].scatter(totals, counts, s=10, alpha=0.45, color="#0f766e", edgecolor="none")
    axes[1].set_title("Gene count vs dataset total")
    axes[1].set_xlabel("Total count")
    axes[1].set_ylabel("X_gc")
    axes[2].hist(frac, bins=60, color="#c2410c", alpha=0.8)
    axes[2].set_title("Observed ratio proxy")
    axes[2].set_xlabel("X_gc / total")
    fig.suptitle(f"{summary.gene_name} overview")
    fig.tight_layout()
    return fig_to_uri(fig)


def plot_prior_fit(analysis: GeneAnalysis) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(analysis.support_mu, analysis.prior_weights, color="#1d4ed8", lw=2.2)
    axes[0].fill_between(
        analysis.support_mu, analysis.prior_weights, color="#1d4ed8", alpha=0.18
    )
    axes[0].set_title("Prior mass over mu grid")
    axes[0].set_xlabel("mu = S p")
    axes[1].hist(
        analysis.observed_mu_proxy,
        bins=50,
        density=True,
        color="#c2410c",
        alpha=0.35,
        label="Observed mu proxy",
    )
    axes[1].hist(
        analysis.signal,
        bins=50,
        density=True,
        color="#0f766e",
        alpha=0.35,
        label="MAP mu",
    )
    axes[1].set_title("Observed proxy vs inferred signal")
    axes[1].set_xlabel("mu-scale")
    axes[1].legend(frameon=False)
    if np.any(np.isfinite(analysis.support_p)):
        axes[2].plot(
            analysis.support_p, analysis.prior_weights, color="#0f766e", lw=2.0
        )
        axes[2].set_title("Prior mass over p grid")
        axes[2].set_xlabel("p")
    else:
        axes[2].plot(
            analysis.support_mu, analysis.prior_weights, color="#0f766e", lw=2.0
        )
        axes[2].set_title("Prior mass over rate grid")
        axes[2].set_xlabel("rate")
    fig.tight_layout()
    return fig_to_uri(fig)


def plot_loss_trace(analysis: GeneAnalysis) -> str:
    if analysis.fit_result is None:
        raise ValueError("loss trace requires an on-demand fit result")
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
    iters = np.arange(len(analysis.fit_result.loss_history))
    axes[0].plot(
        iters, analysis.fit_result.loss_history, color="#1d4ed8", lw=2.0, label="total"
    )
    axes[0].plot(
        iters, analysis.fit_result.nll_history, color="#0f766e", lw=1.8, label="nll"
    )
    axes[0].plot(
        iters, analysis.fit_result.align_history, color="#c2410c", lw=1.8, label="align"
    )
    axes[0].set_title("On-demand fit trace")
    axes[0].set_xlabel("Iteration")
    axes[0].legend(frameon=False)
    axes[1].plot(
        analysis.support_mu,
        analysis.fit_result.initial_prior_weights[0],
        color="#94a3b8",
        lw=2.0,
        ls="--",
        label="initial",
    )
    axes[1].plot(
        analysis.support_mu,
        analysis.prior_weights,
        color="#1d4ed8",
        lw=2.2,
        label="final",
    )
    axes[1].set_title("Initial vs final prior")
    axes[1].set_xlabel("mu")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return fig_to_uri(fig)


def plot_signal_interface(analysis: GeneAnalysis) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    lim = max(
        float(np.max(analysis.observed_mu_proxy)), float(np.max(analysis.signal)), 1.0
    )
    sc0 = axes[0, 0].scatter(
        analysis.observed_mu_proxy,
        analysis.signal,
        c=analysis.posterior_entropy,
        cmap="viridis",
        s=14,
        alpha=0.72,
        edgecolor="none",
    )
    axes[0, 0].plot([0, lim], [0, lim], color="#1f2430", lw=1.1, ls=":")
    axes[0, 0].set_title("MAP mu vs observed mu proxy")
    axes[0, 0].set_xlabel("Observed mu proxy")
    axes[0, 0].set_ylabel("MAP mu")
    fig.colorbar(sc0, ax=axes[0, 0], shrink=0.88).set_label("Posterior entropy")
    axes[0, 1].scatter(
        analysis.observed_mu_proxy,
        analysis.posterior_entropy,
        c=analysis.signal,
        cmap="magma",
        s=14,
        alpha=0.72,
        edgecolor="none",
    )
    axes[0, 1].set_title("Entropy vs observed mu proxy")
    axes[0, 1].set_xlabel("Observed mu proxy")
    axes[0, 1].set_ylabel("Posterior entropy")
    axes[1, 0].scatter(
        analysis.signal,
        analysis.mutual_information,
        c=analysis.posterior_entropy,
        cmap="viridis",
        s=14,
        alpha=0.72,
        edgecolor="none",
    )
    axes[1, 0].set_title("Mutual information vs MAP mu")
    axes[1, 0].set_xlabel("MAP mu")
    axes[1, 0].set_ylabel("Mutual information")
    axes[1, 1].hist(
        analysis.posterior_entropy,
        bins=50,
        color="#1d4ed8",
        alpha=0.5,
        label="posterior entropy",
    )
    axes[1, 1].hist(
        analysis.prior_entropy,
        bins=50,
        color="#c2410c",
        alpha=0.4,
        label="prior entropy",
    )
    axes[1, 1].set_title("Entropy distributions")
    axes[1, 1].legend(frameon=False)
    fig.tight_layout()
    return fig_to_uri(fig)


def plot_posterior_gallery(analysis: GeneAnalysis) -> str:
    n = max(int(analysis.posterior_samples.shape[0]), 1)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx, posterior in zip(
        axes.ravel(),
        analysis.posterior_cell_indices.tolist(),
        analysis.posterior_samples,
        strict=False,
    ):
        ax.axis("on")
        ax.plot(analysis.support_mu, posterior, color="#1d4ed8", lw=2.0)
        ax.fill_between(analysis.support_mu, posterior, color="#1d4ed8", alpha=0.16)
        ax.set_title(f"Cell {idx}")
        ax.set_xlabel("mu")
        ax.set_ylabel("Posterior")
    fig.tight_layout()
    return fig_to_uri(fig)


def plot_global_overview(result: GlobalEvaluationResult) -> str:
    names = list(result.representation_metrics)
    silhouette = [result.representation_metrics[name].silhouette for name in names]
    nmi = [result.representation_metrics[name].nmi for name in names]
    neighbor = [
        result.representation_metrics[name].neighborhood_consistency for name in names
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(names, silhouette, color="#1d4ed8", alpha=0.8)
    axes[0].set_title("Silhouette")
    axes[1].bar(names, nmi, color="#0f766e", alpha=0.8)
    axes[1].set_title("NMI")
    axes[2].bar(names, neighbor, color="#c2410c", alpha=0.8)
    axes[2].set_title("Neighborhood consistency")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig_to_uri(fig)


def plot_kbulk_comparison(comparison: KBulkComparison) -> str:
    groups = comparison.groups
    cols = 2
    fig, axes = plt.subplots(1, cols, figsize=(18, 5.5))
    for group in groups:
        axes[0].plot(
            group.support_mu,
            group.prior_weights,
            lw=2.0,
            label=f"{group.label} (n={group.n_cells})",
        )
        axes[1].hist(
            group.sampled_map_mu,
            bins=min(16, max(group.sampled_map_mu.size, 4)),
            alpha=0.38,
            density=True,
            label=f"{group.label} mean={group.mean_map_mu:.2f}",
        )
    axes[0].set_title(f"Class-specific F_g fits ({comparison.label_key})")
    axes[0].set_xlabel("mu")
    axes[0].set_ylabel("Prior mass")
    axes[0].legend(frameon=False)
    axes[1].set_title(f"kBulk MAP mu distributions (k={comparison.k})")
    axes[1].set_xlabel("kBulk MAP mu")
    axes[1].set_ylabel("Density")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return fig_to_uri(fig)
