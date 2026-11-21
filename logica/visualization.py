"""Publication-ready Matplotlib visualizations for LogiCA readouts.

Provides clean, house-styled plotting functions for the three core native-logit
capabilities of LogiCA:
  1. :func:`plot_dependency_map`: 2D cross-modal (or intra-sequence) dependency heatmap
     with a marginal binding-interface / contact profile;
  2. :func:`plot_gibbs_design`: Multi-step Coordinate Gibbs optimization trajectories
     paired with the context-conditioned 20-AA site probability distribution;
  3. :func:`plot_variant_ranking`: Mutation-local ranking concordance & Top-k% enrichment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from .dependency import DependencyMapResult
from .gibbs import GibbsDesignResult
from .ranking import evaluate_ranking_metrics

# LogiCA manuscript house palette (deep violet + terracotta)
COLOR_LOGICA = "#C8521A"
COLOR_BASELINE = "#5B4397"
COLOR_NEUTRAL = "#7B7D85"
CMAP_DEPENDENCY = "Purples"
CMAP_GIBBS = "Oranges"


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.22, linewidth=0.7)


def plot_dependency_map(
    result: DependencyMapResult,
    *,
    contact_residues_1based: Sequence[int] | None = None,
    max_primary_display: int = 40,
    title: str = "Cross-Modal Native-Logit Dependency Map",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots a 2D token-by-token dependency map alongside the per-residue sensitivity profile."""
    mat = result.matrix
    p_toks = result.primary_tokens
    p_prof = result.primary_profile

    if len(p_toks) > max_primary_display:
        # Focus on the highest-dependency window for legible residue labels
        half = max_primary_display // 2
        peak = int(np.argmax(p_prof))
        st = max(0, min(len(p_toks) - max_primary_display, peak - half))
        en = st + max_primary_display
        mat_view = mat[st:en, :]
        p_labels = [f"{p_toks[i]}{i + 1}" for i in range(st, en)]
        prof_view = p_prof[st:en]
        idx_1based = [i + 1 for i in range(st, en)]
    else:
        mat_view = mat
        p_labels = [f"{tok}{i + 1}" for i, tok in enumerate(p_toks)]
        prof_view = p_prof
        idx_1based = [i + 1 for i in range(len(p_toks))]

    fig = plt.figure(figsize=(11.8, 4.4), dpi=200, layout="constrained")
    gs = GridSpec(1, 2, width_ratios=[1.45, 1.0], figure=fig)

    ax_hm = fig.add_subplot(gs[0, 0])
    im = ax_hm.imshow(mat_view.T, aspect="auto", cmap=CMAP_DEPENDENCY, interpolation="nearest")
    ax_hm.set_title(f"A. {title}", loc="left", fontweight="bold", fontsize=11)
    ax_hm.set_xlabel("Primary Sequence Position", fontsize=10)
    ax_hm.set_ylabel("Partner Token Index", fontsize=10)

    step_x = max(1, len(p_labels) // 12)
    ax_hm.set_xticks(range(0, len(p_labels), step_x))
    ax_hm.set_xticklabels([p_labels[i] for i in range(0, len(p_labels), step_x)], rotation=45, ha="right", fontsize=8)
    step_y = max(1, len(result.partner_tokens) // 10)
    ax_hm.set_yticks(range(0, len(result.partner_tokens), step_y))
    ax_hm.set_yticklabels(
        [f"{result.partner_tokens[j]} ({j + 1})" for j in range(0, len(result.partner_tokens), step_y)],
        fontsize=8,
    )
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label(f"Token Shift ({result.metric})", fontsize=9)

    ax_bar = fig.add_subplot(gs[0, 1])
    _style_axis(ax_bar)
    contact_set = set(int(c) for c in (contact_residues_1based or ()))
    colors = [COLOR_LOGICA if pos in contact_set else COLOR_BASELINE for pos in idx_1based]
    ax_bar.bar(range(len(prof_view)), prof_view, color=colors, width=0.82, edgecolor="none")
    ax_bar.set_xticks(range(0, len(p_labels), step_x))
    ax_bar.set_xticklabels([p_labels[i] for i in range(0, len(p_labels), step_x)], rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylabel("Aggregated Dependency Score", fontsize=10)
    ax_bar.set_xlabel("Primary Sequence Position", fontsize=10)
    ax_bar.set_title("B. Marginal Interface Sensitivity", loc="left", fontweight="bold", fontsize=11)
    if contact_set:
        from matplotlib.patches import Patch

        ax_bar.legend(
            handles=[
                Patch(facecolor=COLOR_LOGICA, label="<5 Å Contact Site"),
                Patch(facecolor=COLOR_BASELINE, label="Non-Contact Site"),
            ],
            frameon=True,
            fontsize=8.5,
        )

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_gibbs_design(
    result: GibbsDesignResult,
    *,
    baseline_result: GibbsDesignResult | None = None,
    title: str = "Context-Conditioned Coordinate Gibbs Design",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots multi-step Coordinate Gibbs score trajectories and the site-by-20AA probability map."""
    fig = plt.figure(figsize=(11.4, 4.3), dpi=200, layout="constrained")
    gs = GridSpec(1, 2, width_ratios=[1.15, 1.35], figure=fig)

    ax_tr = fig.add_subplot(gs[0, 0])
    _style_axis(ax_tr)
    steps = np.arange(result.score_trajectories.shape[1])
    mean_tr = result.score_trajectories.mean(axis=0)
    sem_tr = result.score_trajectories.std(axis=0) / max(1.0, np.sqrt(result.score_trajectories.shape[0]))
    ax_tr.plot(steps, mean_tr, color=COLOR_LOGICA, marker="D", lw=2.3, markersize=5, label=f"LogiCA (N_eff={result.effective_sample_size:.1f})")
    ax_tr.fill_between(steps, mean_tr - sem_tr, mean_tr + sem_tr, color=COLOR_LOGICA, alpha=0.18)

    if baseline_result is not None:
        b_steps = np.arange(baseline_result.score_trajectories.shape[1])
        b_mean = baseline_result.score_trajectories.mean(axis=0)
        b_sem = baseline_result.score_trajectories.std(axis=0) / max(
            1.0, np.sqrt(baseline_result.score_trajectories.shape[0])
        )
        ax_tr.plot(
            b_steps,
            b_mean,
            color=COLOR_BASELINE,
            marker="s",
            lw=2.0,
            markersize=4.5,
            linestyle="--",
            label=f"Unconditional BioFM (N_eff={baseline_result.effective_sample_size:.1f})",
        )
        ax_tr.fill_between(b_steps, b_mean - b_sem, b_mean + b_sem, color=COLOR_BASELINE, alpha=0.14)

    ax_tr.set_title(f"A. {title}", loc="left", fontweight="bold", fontsize=11)
    ax_tr.set_xlabel("Coordinate Gibbs Step (t)", fontsize=10)
    ax_tr.set_ylabel("Mean Log-Likelihood Advantage vs. Seed", fontsize=10)
    ax_tr.legend(frameon=True, fontsize=8.5, loc="lower right")

    ax_pwm = fig.add_subplot(gs[0, 1])
    im = ax_pwm.imshow(result.site_probability_matrix, aspect="auto", cmap=CMAP_GIBBS, vmin=0.0)
    ax_pwm.set_title("B. Native Token Proposal Probabilities", loc="left", fontweight="bold", fontsize=11)
    ax_pwm.set_xticks(range(len(result.alphabet)))
    ax_pwm.set_xticklabels(list(result.alphabet), fontsize=8.5)
    site_labels = [f"{result.seed_sequence[p]}{p + 1}" for p in result.design_positions_0based]
    ax_pwm.set_yticks(range(len(site_labels)))
    ax_pwm.set_yticklabels(site_labels, fontsize=8.5)
    ax_pwm.set_xlabel("Canonical Amino Acid Vocabulary", fontsize=10)
    ax_pwm.set_ylabel("Designed Site", fontsize=10)
    cbar = fig.colorbar(im, ax=ax_pwm, fraction=0.046, pad=0.04)
    cbar.set_label("Proposal Probability", fontsize=9)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_variant_ranking(
    experimental_fitness: Sequence[float] | np.ndarray,
    logica_scores: Sequence[float] | np.ndarray,
    *,
    baseline_scores: Sequence[float] | np.ndarray | None = None,
    baseline_name: str = "Unconditional BioFM",
    title: str = "Mutation-Local Variant Ranking",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots predicted LogiCA mutation scores vs. experimental fitness + Precision@k% curve."""
    y = np.asarray(experimental_fitness, dtype=np.float64)
    s_lg = np.asarray(logica_scores, dtype=np.float64)
    metrics_lg = evaluate_ranking_metrics(y, s_lg)

    fig = plt.figure(figsize=(11.4, 4.3), dpi=200, layout="constrained")
    gs = GridSpec(1, 2, width_ratios=[1.15, 1.0], figure=fig)

    ax_sc = fig.add_subplot(gs[0, 0])
    _style_axis(ax_sc)
    is_top = y >= np.quantile(y, 0.90)
    ax_sc.scatter(s_lg[~is_top], y[~is_top], s=18, alpha=0.45, color=COLOR_BASELINE, label="Variants (<P90)")
    ax_sc.scatter(s_lg[is_top], y[is_top], s=28, alpha=0.85, color=COLOR_LOGICA, label="Top-10% Hits")
    ax_sc.set_title(
        f"A. {title} (ρ = {metrics_lg['spearman_rho']:+.3f})",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    ax_sc.set_xlabel("LogiCA Mutation-Local Score s_M(x, y; x^wt)", fontsize=10)
    ax_sc.set_ylabel("Experimental Activity / Fitness", fontsize=10)
    ax_sc.legend(frameon=True, fontsize=8.5)

    ax_pk = fig.add_subplot(gs[0, 1])
    _style_axis(ax_pk)
    fracs = [0.01, 0.02, 0.05, 0.10, 0.20]
    labels_k = ["1%", "2%", "5%", "10%", "20%"]
    from .ranking import top_k_enrichment

    enr_lg = [top_k_enrichment(y, s_lg, f) for f in fracs]
    ax_pk.plot(range(len(fracs)), enr_lg, color=COLOR_LOGICA, marker="D", lw=2.3, label="LogiCA")

    if baseline_scores is not None:
        s_bs = np.asarray(baseline_scores, dtype=np.float64)
        enr_bs = [top_k_enrichment(y, s_bs, f) for f in fracs]
        ax_pk.plot(
            range(len(fracs)),
            enr_bs,
            color=COLOR_BASELINE,
            marker="s",
            lw=2.0,
            linestyle="--",
            label=baseline_name,
        )

    ax_pk.axhline(1.0, color=COLOR_NEUTRAL, linestyle=":", lw=1.2, label="Random (1.0×)")
    ax_pk.set_xticks(range(len(fracs)))
    ax_pk.set_xticklabels(labels_k)
    ax_pk.set_xlabel("Prioritized Library Fraction (Top-k%)", fontsize=10)
    ax_pk.set_ylabel("Hit Enrichment Factor (× Random)", fontsize=10)
    ax_pk.set_title("B. Top-k% Prioritization Enrichment", loc="left", fontweight="bold", fontsize=11)
    ax_pk.legend(frameon=True, fontsize=8.5)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fig
