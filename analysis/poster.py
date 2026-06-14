"""
Poster figure for the reward model results.

Panel A shows the mean reward gap (SAE - AAVE) per ReDial task, pooled across
reward models. Panel B shows the mean reward gap of one reward model lineage
trained on the base vs. the instruction-tuned backbone. Values are those reported
in Section 4.2 of the paper.

Usage:
    python analysis/poster.py
    python analysis/poster.py --output-dir /path/to/plots
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MultipleLocator

from rewards import DIALECT_COLORS as _REWARD_COLORS

plt.rcParams.update({
    "axes.labelsize": 25,
    "xtick.labelsize": 21,
    "axes.linewidth": 1.2,
    "pdf.fonttype": 42,
})

# Mean reward gaps (SAE - AAVE) reported in the paper
TASK_GAPS = {"algorithm": 0.43, "math": 0.04, "logic": -0.08, "planning": -0.14}
LINEAGE_GAPS = {"base": 0.41, "instruct": -0.19}

# Keep the AAVE orange from the reward plots; SAE takes the Dark2 purple instead of Set2 green
DIALECT_COLORS = {"sae": "#7570b3", "aave": _REWARD_COLORS["aave"]}

LINEAGE_LABELS = {"base": "Base", "instruct": "Instruction-tuned"}
LINEAGE_NOTES = {"left": "Ai2 reward model", "right": "Llama 3.1 8B"}

MUTED = "#555555"
TINT_ALPHA = 0.10
POINT_RADIUS = 18

TITLE_SIZE = 26
LABEL_SIZE = 24
VALUE_SIZE = 26
NOTE_SIZE = 20
CALLOUT_SIZE = 24


###########
# HELPERS #
###########

def _signed(value, decimals=2):
    """Format a gap with an explicit sign and a typographic minus."""
    return f"{value:+.{decimals}f}".replace("-", "−")


def _favored_color(gap):
    """Color of the dialect a gap (SAE - AAVE) favors."""
    return DIALECT_COLORS["sae"] if gap > 0 else DIALECT_COLORS["aave"]


def _style_gap_axis(ax, xmax, title, xlabel):
    """Apply the shared reward-gap axis: tinted sides, zero line, direction labels, titles."""
    ax.set_xlim(-xmax, xmax)
    ax.axvspan(-xmax, 0, color=DIALECT_COLORS["aave"], alpha=TINT_ALPHA, linewidth=0, zorder=0)
    ax.axvspan(0, xmax, color=DIALECT_COLORS["sae"], alpha=TINT_ALPHA, linewidth=0, zorder=0)
    ax.axvline(0, color="black", linewidth=2.5, zorder=3)

    for text, side in [("← Favors AAVE", -1), ("Favors SAE →", 1)]:
        ax.annotate(
            text, (0, 0.93), xycoords=ax.get_xaxis_transform(),
            xytext=(side * 14, 0), textcoords="offset points",
            ha="left" if side > 0 else "right", va="center",
            fontsize=LABEL_SIZE, color=MUTED,
        )

    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else _signed(v, 1)))
    ax.tick_params(axis="x", length=6, width=1.2)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)

    ax.set_title(title, fontsize=TITLE_SIZE, pad=14)
    sns.despine(ax=ax, left=True)


##########
# PANELS #
##########

def plot_task_gaps(ax, task_gaps):
    """Diverging bars of the mean reward gap per task, labeled at the zero line.

    Args:
        ax: Target axes.
        task_gaps: Mean gap (SAE - AAVE) keyed by task.
    """
    # Most SAE-favoring task at the bottom, so the bars read as a staircase from the top
    task_gaps = sorted(task_gaps.items(), key=lambda item: item[1], reverse=True)
    y = np.arange(len(task_gaps))
    ax.barh(
        y, [gap for _, gap in task_gaps], height=0.62, color=[_favored_color(gap) for _, gap in task_gaps],
        edgecolor="white", linewidth=0.5, zorder=2,
    )
    for yi, (task, gap) in zip(y, task_gaps):
        side = 1 if gap > 0 else -1
        ax.annotate(
            _signed(gap), (gap, yi), xytext=(side * 9, 0), textcoords="offset points",
            ha="left" if side > 0 else "right", va="center", fontsize=VALUE_SIZE, fontweight="bold",
        )
        ax.annotate(
            task.capitalize(), (0, yi), xytext=(-side * 12, 0), textcoords="offset points",
            ha="right" if side > 0 else "left", va="center", fontsize=LABEL_SIZE,
        )
    ax.set_ylim(-0.6, len(task_gaps) + 0.15)


def plot_lineage_flip(ax, base_gap, instruct_gap):
    """Two points on the reward-gap axis joined by an arrow from base to instruction-tuned.

    Args:
        ax: Target axes.
        base_gap: Mean gap (SAE - AAVE) of the reward model on the base backbone.
        instruct_gap: Mean gap of the reward model on the instruction-tuned backbone.
    """
    y = 0.48
    ax.set_ylim(0, 1)
    ax.annotate(
        "", xy=(instruct_gap, y), xytext=(base_gap, y), zorder=4,
        arrowprops={
            "arrowstyle": "-|>,head_length=0.6,head_width=0.32", "color": "black",
            "linewidth": 5, "mutation_scale": 34, "shrinkA": POINT_RADIUS + 3, "shrinkB": POINT_RADIUS + 3,
        },
    )
    gaps = {"base": base_gap, "instruct": instruct_gap}
    ax.scatter(
        list(gaps.values()), [y] * len(gaps), s=4 * POINT_RADIUS ** 2,
        c=[_favored_color(g) for g in gaps.values()], edgecolor="white", linewidth=2, zorder=5,
    )
    for key, gap in gaps.items():
        # Labels start at the point's inner edge and extend away from zero, clear of the zero line
        side = 1 if gap > 0 else -1
        align = {"textcoords": "offset points", "ha": "left" if side > 0 else "right", "va": "top"}
        ax.annotate(
            _signed(gap), (gap, y), xytext=(-side * POINT_RADIUS, -30),
            fontsize=VALUE_SIZE, fontweight="bold", **align,
        )
        ax.annotate(
            LINEAGE_LABELS[key], (gap, y), xytext=(-side * POINT_RADIUS, -64), fontsize=LABEL_SIZE, **align,
        )
    ax.annotate(
        "Preference flips after post-training", (0, 0.64), ha="center", va="center",
        fontsize=CALLOUT_SIZE, fontstyle="italic", zorder=6,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": MUTED, "linewidth": 1},
    )
    for (side, text), x in zip(LINEAGE_NOTES.items(), (0.02, 0.98)):
        ax.annotate(text, (x, 0.04), xycoords="axes fraction", ha=side, va="bottom", fontsize=NOTE_SIZE, color=MUTED)


##########
# FIGURE #
##########

def plot_poster(output_dir):
    """Two-panel poster figure: gap by task, and gap before/after instruction tuning."""
    # Leave room past the longest bar for its value label
    xmax = 1.6 * max(map(abs, [*TASK_GAPS.values(), *LINEAGE_GAPS.values()]))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(17, 8.4), sharex=True)
    fig.get_layout_engine().set(wspace=0.06, w_pad=0.15, h_pad=0.15)

    plot_task_gaps(ax_a, TASK_GAPS)
    _style_gap_axis(
        ax_a, xmax, "Reward preference depends on context",
        "Reward gap (pooled): SAE − AAVE",
    )
    plot_lineage_flip(ax_b, LINEAGE_GAPS["base"], LINEAGE_GAPS["instruct"])
    _style_gap_axis(
        ax_b, xmax, "Post-training can reverse token preferences",
        "Reward gap (mean): SAE − AAVE",
    )

    fig.savefig(os.path.join(output_dir, "poster_rewards.pdf"))
    fig.savefig(os.path.join(output_dir, "poster_rewards.png"), dpi=300)
    plt.close(fig)
    print("  Saved poster_rewards.pdf")


########
# MAIN #
########

def main():
    parser = argparse.ArgumentParser(description="Poster figure for reward model results")
    parser.add_argument(
        "--output-dir", default="analysis/plots/poster",
        help="Directory for plot outputs (default: analysis/plots/poster).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_poster(args.output_dir)


if __name__ == "__main__":
    main()
