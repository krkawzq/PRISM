from __future__ import annotations

import base64
from io import BytesIO

import numpy as np

from prism.plotting import plt

from .analysis import GeneAnalysis, KBulkAnalysis


def figure_to_data_uri(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def plot_raw_overview(analysis: GeneAnalysis):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].hist(np.asarray(analysis.counts, dtype=np.float64), bins=40, color="#0f766e")
    axes[0].set_title("Raw Count Histogram")
    axes[0].set_xlabel("count")
    axes[0].set_ylabel("cells")

    axes[1].scatter(
        np.asarray(analysis.totals, dtype=np.float64),
        np.asarray(analysis.counts, dtype=np.float64),
        s=8,
        alpha=0.35,
        color="#1d4ed8",
    )
    axes[1].set_title("Raw Count vs Total Count")
    axes[1].set_xlabel("cell total count")
    axes[1].set_ylabel("gene count")

    axes[2].hist(np.asarray(analysis.raw_proxy, dtype=np.float64), bins=40, color="#c2410c")
    axes[2].set_title("Scaled Raw Proxy")
    axes[2].set_xlabel("proxy")
    axes[2].set_ylabel("cells")

    fig.tight_layout()
    return fig


def plot_prior_overlay(analysis: GeneAnalysis):
    if analysis.prior is None:
        raise ValueError("analysis does not have a prior to plot")
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    support = np.asarray(analysis.prior.scaled_support, dtype=np.float64).reshape(-1)
    probs = np.asarray(analysis.prior.prior_probabilities, dtype=np.float64).reshape(-1)
    ax.plot(support, probs, lw=2.2, color="#0f766e", label=f"{analysis.mode} prior")
    if analysis.checkpoint_prior is not None and analysis.mode == "fit":
        ckpt_support = np.asarray(
            analysis.checkpoint_prior.scaled_support,
            dtype=np.float64,
        ).reshape(-1)
        ckpt_probs = np.asarray(
            analysis.checkpoint_prior.prior_probabilities,
            dtype=np.float64,
        ).reshape(-1)
        ax.plot(
            ckpt_support,
            ckpt_probs,
            lw=2.0,
            ls="--",
            color="#1d4ed8",
            label="checkpoint prior",
        )
    ax.set_title("Prior Profile")
    ax.set_xlabel("support")
    ax.set_ylabel("prior mass")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_objective_trace(analysis: GeneAnalysis):
    if analysis.fit_result is None:
        raise ValueError("analysis does not have fit objective history")
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    history = np.asarray(analysis.fit_result.objective_history, dtype=np.float64)
    ax.plot(np.arange(1, history.size + 1), history, color="#c2410c", lw=2.0)
    ax.set_title("Fit Objective Trace")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("objective")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_signal_interface(analysis: GeneAnalysis):
    if analysis.posterior is None:
        raise ValueError("analysis does not have posterior outputs")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    signal = np.asarray(analysis.posterior.map_scaled_support[:, 0], dtype=np.float64)
    entropy = np.asarray(analysis.posterior.posterior_entropy[:, 0], dtype=np.float64)
    mi = np.asarray(analysis.posterior.mutual_information[:, 0], dtype=np.float64)
    axes[0].scatter(
        np.asarray(analysis.raw_proxy, dtype=np.float64),
        signal,
        s=10,
        alpha=0.35,
        color="#0f766e",
    )
    axes[0].set_title("Signal vs Raw Proxy")
    axes[0].set_xlabel("raw proxy")
    axes[0].set_ylabel("MAP signal")
    axes[1].scatter(entropy, mi, s=10, alpha=0.35, color="#7c3aed")
    axes[1].set_title("Entropy vs Mutual Information")
    axes[1].set_xlabel("posterior entropy")
    axes[1].set_ylabel("mutual information")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_posterior_gallery(analysis: GeneAnalysis):
    if analysis.posterior is None:
        raise ValueError("analysis does not have posterior outputs")
    posterior = np.asarray(
        analysis.posterior.posterior_probabilities,
        dtype=np.float64,
    )
    if posterior.ndim != 3 or posterior.shape[0] == 0:
        raise ValueError("posterior probabilities are missing")
    limit = min(12, posterior.shape[0])
    support = np.asarray(analysis.posterior.scaled_support, dtype=np.float64).reshape(-1)
    order = np.argsort(np.asarray(analysis.counts, dtype=np.float64))[::-1][:limit]
    cols = 3
    rows = (limit + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12.0, 3.1 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for panel_idx, row_idx in enumerate(order.tolist()):
        ax = axes.ravel()[panel_idx]
        ax.set_visible(True)
        ax.plot(support, posterior[row_idx, 0], color="#1d4ed8", lw=1.8)
        ax.set_title(f"cell {row_idx} | raw={analysis.counts[row_idx]:.0f}")
        ax.set_xlabel("support")
        ax.set_ylabel("posterior")
        ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig


def plot_kbulk_group_comparison(result: KBulkAnalysis):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    labels = [group.label for group in result.groups]
    signal_data = [np.asarray(group.signal_values, dtype=np.float64) for group in result.groups]
    entropy_data = [np.asarray(group.entropy_values, dtype=np.float64) for group in result.groups]
    axes[0].boxplot(signal_data, labels=labels, patch_artist=True)
    axes[0].set_title("kBulk MAP Signal")
    axes[0].set_xlabel(result.class_key)
    axes[0].set_ylabel("MAP signal")
    axes[1].boxplot(entropy_data, labels=labels, patch_artist=True)
    axes[1].set_title("kBulk Posterior Entropy")
    axes[1].set_xlabel(result.class_key)
    axes[1].set_ylabel("posterior entropy")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


__all__ = [
    "figure_to_data_uri",
    "plot_kbulk_group_comparison",
    "plot_objective_trace",
    "plot_posterior_gallery",
    "plot_prior_overlay",
    "plot_raw_overview",
    "plot_signal_interface",
]
