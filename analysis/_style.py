"""Shared plot styling for analysis/*.py.

Matches the SC26 paper's publication figures (colours, fonts, layout) so the
figures a reviewer regenerates from their own run outputs read the same as the
ones in the paper. System colours mirror the paper's evaluation figures
(ServerlessLLM grey, ServerlessLoRA red, SwarmLoRA blue).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Canonical system order + colours (paper evaluation figures) ──────────────
SYSTEMS = ("ServerlessLLM", "ServerlessLoRA", "SwarmLoRA")

COLOR = {
    "SwarmLoRA": "#2166ac",       # blue  -- this artifact's own system
    "ServerlessLoRA": "#d6604d",  # red
    "ServerlessLLM": "#878787",   # grey
}

MARKER = {
    "SwarmLoRA": "o",
    "ServerlessLoRA": "s",
    "ServerlessLLM": "^",
}

# Per-traffic-category colours/markers (used by the SLO CDF). The required-path
# analysis only plots "normal", but the mapping is kept complete so extending
# to all five categories later needs no style change.
CAT_COLORS = {
    "steady_light": "#4daf4a", "bursty_light": "#377eb8", "normal": "#ff7f00",
    "steady_heavy": "#e41a1c", "bursty_heavy": "#984ea3",
}
CAT_MARKERS = {
    "steady_light": "o", "bursty_light": "s", "normal": "D",
    "steady_heavy": "^", "bursty_heavy": "v",
}

# Column widths (inches) matching the paper's single/double column.
COL1 = 3.5
COL2 = 7.16


def apply_style():
    """Publication rcParams -- called once at import."""
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "lines.linewidth": 1.0, "lines.markersize": 5,
        "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "legend.frameon": False,
        "legend.handlelength": 1.5, "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
    })


def apply_common_style(ax):
    """Per-axis grid + spine tidy (back-compat helper)."""
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


apply_style()
