"""Shared plotting style for journal figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_elsevier_figure_style(base_size: float = 8.0) -> None:
    """Use Elsevier-approved fonts and consistent vector output settings."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times"],
            "font.size": base_size,
            "axes.labelsize": base_size,
            "axes.titlesize": base_size,
            "legend.fontsize": max(base_size - 1.0, 6.0),
            "xtick.labelsize": max(base_size - 1.0, 6.0),
            "ytick.labelsize": max(base_size - 1.0, 6.0),
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.35,
            "savefig.bbox": "tight",
        }
    )
