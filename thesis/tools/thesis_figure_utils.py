#!/usr/bin/env python3
"""
Thesis Figure Utilities — Publication-Ready Figures for OU Thesis

Provides:
  - Publication style preset (Times serif, colorblind-safe palette)
  - Colorblind-friendly palettes (Okabe-Ito, Tol, Wong)
  - Multi-format export (PNG 300dpi + PDF vector)
  - Thesis-specific figure sizes (single-column, full-width)
  - Box plot, violin plot, grouped bar chart templates

Usage:
    from thesis_figure_utils import apply_thesis_style, save_thesis_figure
    from thesis_figure_utils import COLORS, thesis_box_plot, thesis_grouped_bar

    apply_thesis_style()
    fig, ax = plt.subplots()
    ax.bar(['A', 'B'], [10, 20], color=[COLORS['blue'], COLORS['vermillion']])
    save_thesis_figure(fig, 'my_figure')
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Colorblind-Friendly Palettes ──────────────────────────────────────────

# Okabe-Ito (recommended default)
OKABE_ITO = {
    'orange':         '#E69F00',
    'sky_blue':       '#56B4E9',
    'bluish_green':   '#009E73',
    'yellow':         '#F0E442',
    'blue':           '#0072B2',
    'vermillion':     '#D55E00',
    'reddish_purple': '#CC79A7',
    'black':          '#000000',
}
OKABE_ITO_LIST = list(OKABE_ITO.values())

# Thesis default order: blue first (BEV), vermillion second (image-space)
COLORS = {
    'blue':           '#0072B2',
    'vermillion':     '#D55E00',
    'green':          '#009E73',
    'orange':         '#E69F00',
    'purple':         '#CC79A7',
    'sky_blue':       '#56B4E9',
    'yellow':         '#F0E442',
    'black':          '#000000',
}

# For BEV vs Image-space comparisons
BEV_COLOR = '#0072B2'       # Blue
IMG_COLOR = '#D55E00'       # Vermillion
TEMPLATE_COLOR = '#009E73'  # Green
BASELINE_COLOR = '#999999'  # Gray

# Paul Tol palettes
TOL_BRIGHT = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
TOL_MUTED = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']

# Sequential colormaps (safe for colorblind)
SAFE_CMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']


# ── Style Application ─────────────────────────────────────────────────────

STYLE_FILE = Path(__file__).parent / 'publication_style.mplstyle'


def apply_thesis_style() -> None:
    """Apply the OU thesis publication style."""
    if STYLE_FILE.exists():
        plt.style.use(str(STYLE_FILE))
    else:
        # Fallback: apply inline
        plt.rcParams.update({
            'figure.figsize': (6.0, 4.0),
            'figure.constrained_layout.use': True,
            'font.size': 9,
            'font.family': 'serif',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'axes.prop_cycle': mpl.cycler(color=list(COLORS.values())),
            'legend.frameon': False,
            'legend.fontsize': 8,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        })


def thesis_figsize(width: str = 'full') -> Tuple[float, float]:
    """Return figure size for thesis layout.

    Parameters
    ----------
    width : str
        'full' (6.0 x 4.0), 'half' (3.0 x 2.5), 'wide' (6.5 x 3.0),
        'tall' (4.5 x 6.0), 'square' (4.0 x 4.0)
    """
    sizes = {
        'full':   (6.0, 4.0),
        'half':   (3.0, 2.5),
        'wide':   (6.5, 3.0),
        'tall':   (4.5, 6.0),
        'square': (4.0, 4.0),
    }
    return sizes.get(width, (6.0, 4.0))


# ── Figure Export ─────────────────────────────────────────────────────────

FIGURES_DIR = Path(__file__).parent.parent / 'figures'


def save_thesis_figure(
    fig: plt.Figure,
    name: str,
    formats: Sequence[str] = ('png', 'pdf'),
    dpi: int = 300,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Save figure in multiple formats for thesis.

    Parameters
    ----------
    fig : Figure
    name : str
        Base filename (no extension).
    formats : sequence of str
        File formats to save ('png', 'pdf', 'eps', 'svg').
    dpi : int
        Resolution for raster formats.
    output_dir : Path, optional
        Override output directory (default: thesis/figures/).

    Returns
    -------
    list of Path
        Saved file paths.
    """
    out = output_dir or FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        p = out / f'{name}.{fmt}'
        fig.savefig(p, dpi=dpi, bbox_inches='tight', pad_inches=0.05,
                    facecolor='white', transparent=False, format=fmt)
        saved.append(p)
        print(f'  saved: {p}')
    return saved


# ── Chart Templates ───────────────────────────────────────────────────────

def thesis_box_plot(
    data: Dict[str, np.ndarray],
    ylabel: str,
    title: str = '',
    figsize: str = 'full',
    colors: Optional[List[str]] = None,
    show_points: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a publication-quality box plot.

    Parameters
    ----------
    data : dict
        {label: array_of_values} for each group.
    ylabel : str
    title : str
    figsize : str
        Size preset ('full', 'half', 'wide').
    colors : list of str, optional
        Override colors per group.
    show_points : bool
        Overlay individual data points.

    Returns
    -------
    fig, ax
    """
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=thesis_figsize(figsize))

    labels = list(data.keys())
    values = [data[k] for k in labels]
    n = len(labels)
    if colors is None:
        palette = list(COLORS.values())
        colors = [palette[i % len(palette)] for i in range(n)]

    bp = ax.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        medianprops=dict(color='black', linewidth=1.5),
        meanprops=dict(marker='D', markerfacecolor='white',
                       markeredgecolor='black', markersize=5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')

    if show_points:
        for i, (vals, c) in enumerate(zip(values, colors)):
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(np.full_like(vals, i + 1, dtype=float) + jitter,
                       vals, alpha=0.35, s=12, color=c, edgecolors='none',
                       zorder=3)

    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def thesis_grouped_bar(
    groups: List[str],
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ylabel: str,
    title: str = '',
    figsize: str = 'full',
    colors: Optional[Dict[str, str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a grouped bar chart with error bars.

    Parameters
    ----------
    groups : list of str
        Group labels (x-axis categories).
    series : dict
        {series_name: (means, stds)} — one bar per series per group.
    ylabel : str
    title : str
    figsize : str
    colors : dict, optional
        {series_name: color_hex}.

    Returns
    -------
    fig, ax
    """
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=thesis_figsize(figsize))

    n_groups = len(groups)
    n_series = len(series)
    bar_width = 0.8 / n_series
    x = np.arange(n_groups)
    palette = list(COLORS.values())

    for i, (name, (means, stds)) in enumerate(series.items()):
        offset = (i - n_series / 2 + 0.5) * bar_width
        color = (colors or {}).get(name, palette[i % len(palette)])
        ax.bar(x + offset, means, bar_width, yerr=stds,
               label=name, color=color, edgecolor='black', linewidth=0.5,
               capsize=3, alpha=0.85, error_kw=dict(lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    return fig, ax


def thesis_violin_plot(
    data: Dict[str, np.ndarray],
    ylabel: str,
    title: str = '',
    figsize: str = 'full',
    colors: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a publication-quality violin plot.

    Parameters
    ----------
    data : dict
        {label: array_of_values}.
    ylabel : str
    title : str
    figsize : str
    colors : list of str, optional

    Returns
    -------
    fig, ax
    """
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=thesis_figsize(figsize))

    labels = list(data.keys())
    values = [data[k] for k in labels]
    n = len(labels)
    if colors is None:
        palette = list(COLORS.values())
        colors = [palette[i % len(palette)] for i in range(n)]

    parts = ax.violinplot(values, positions=range(1, n + 1),
                          showmeans=True, showmedians=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.65)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)

    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


if __name__ == '__main__':
    # Quick demo
    apply_thesis_style()

    # Box plot example
    rng = np.random.default_rng(42)
    demo_data = {
        'BEV Skeleton': rng.normal(76.6, 20, 32),
        'BEV DT Ridge': rng.normal(65.0, 18, 32),
        'Img Midpoint':  rng.normal(14.3, 5, 32),
        'Img DT':        rng.normal(60.4, 15, 32),
    }
    fig, ax = thesis_box_plot(
        demo_data,
        ylabel='Lateral Center Error (px)',
        title='Planner Comparison — 32 Hand-Annotated Frames',
        colors=[BEV_COLOR, BEV_COLOR, IMG_COLOR, IMG_COLOR],
    )
    plt.show()
