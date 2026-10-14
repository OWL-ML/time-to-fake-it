"""
Algorithm comparison table for one evaluated test set, mean ± std over seeds.

Columns:
    Init. F   : names[0] evaluated on ds right after training task ds
    names[i]  : method i evaluated on ds after task gen

Cell colours show the difference from names[0] after task gen (the reference column
itself is therefore white), matching the original function.

Input: fairness_group.csv written by fairness_from_predictions.py.
"""

from __future__ import annotations

import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairness_diff_table import DIFF_METRICS, GROUP_LABELS, _text_width_in
from loguru import logger
from matplotlib.patches import Rectangle

INIT_COL = "Init. F"
BORDER_COLOR, BORDER_WIDTH = "#cccccc", 0.8  # outer table border


def load_algs_table(
    fairness_group_csv: str,
    ds: int,
    gen: int,
    names: list[str],
    metrics: list[str] = DIFF_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_mean, df_std, df_n), each indexed by (Group, Metric) with columns
    ["Init. F", *names]. std is the sample std over seeds (ddof=1).
    """
    if len(set(names)) != len(names) or INIT_COL in names:
        raise ValueError(f"names must be unique and must not contain '{INIT_COL}'")

    fg = pd.read_csv(fairness_group_csv)
    fg = fg[fg["Group"].isin(GROUP_LABELS)].copy()
    fg["Group"] = fg["Group"].map(GROUP_LABELS)

    columns = [(INIT_COL, names[0], ds)] + [(name, name, gen) for name in names]
    means, stds, counts, seed_sets = {}, {}, {}, {}
    for col, method, task in columns:
        sel = fg[
            (fg["method"] == method) & (fg["task"] == task) & (fg["eval_task"] == ds)
        ]
        if sel.empty:
            raise ValueError(
                f"No rows for method={method}, task={task}, eval_task={ds}"
            )
        if sel.duplicated(["seed", "Group"]).any():
            raise ValueError(
                f"Duplicate (seed, Group) rows for method={method}, task={task}, "
                f"eval_task={ds}; was the CSV concatenated twice?"
            )
        long = sel.melt(
            id_vars=["seed", "Group"], value_vars=metrics, var_name="Metric"
        )
        grouped = long.groupby(["Group", "Metric"])["value"]
        means[col], stds[col], counts[col] = (
            grouped.mean(),
            grouped.std(),
            grouped.count(),
        )
        seed_sets[col] = set(sel["seed"])

    if len({frozenset(s) for s in seed_sets.values()}) > 1:
        logger.warning(
            "Columns are based on different seed sets, so mean differences between "
            f"columns mix seeds: { {c: sorted(s) for c, s in seed_sets.items()} }"
        )

    df_mean, df_std, df_n = (
        pd.DataFrame(d)[[c for c, _, _ in columns]] for d in (means, stds, counts)
    )
    return df_mean, df_std, df_n


def visualize_fairness_table_algs_matplotlib(
    df_mean,
    df_std,
    save_path=None,
    dpi=300,
    ref_col=None,
    decimals=3,
    col_width_in=None,
    fontsize=18,
    cbar_width_frac=0.9,
    cell_pad_in=0.08,
    std_fontsize=None,
    metric_fontsize=None,
    header_fontsize=None,
    cell_aspect=0.6,
    line_gap=0.39,
    row_pad_in=0.22,
    deadband=0.025,
    cbar_label=None,
    cbar_end_labels=None,
    cbar_neg_frac=0.0,
    gamma=0.6,
    cbar_even_ticks=False,
):
    """
    Publication-quality table comparing fairness metrics across algorithms at one
    timepoint, mean ± std over seeds, coloured by the difference from `ref_col`.

    Parameters
    ----------
    df_mean, df_std : pd.DataFrame
        From load_algs_table: index (Group, Metric), columns ["Init. F", *names].
    save_path : str, optional
        Path prefix for saving (e.g., 'figures/table_algs')
    dpi : int
        Resolution for output image (default: 300)
    ref_col : str, optional
        Column the colours are relative to; defaults to names[0] (second column).
    decimals : int
        Decimals shown for mean and std
    col_width_in : float, optional
        Width of the data columns in inches; by default just wide enough for the widest
        header or value. When given, it wins over cell_aspect.
    fontsize : int
        Font size of the mean, the largest text in a cell.
    cbar_width_frac : float
        Colour bar width as a fraction of the table width; it stays centred.
    cell_pad_in : float
        Horizontal padding added to the widest value in a data column, in inches.
    std_fontsize : int, optional
        Font size of the std line; defaults to fontsize - 4.
    metric_fontsize : int, optional
        Font size of the metric glyph ($F_{DP}$ etc.); defaults to std_fontsize - 2,
        so the glyph is the smallest text in the row.
    header_fontsize : int, optional
        Font size of the column headers and group labels; defaults to fontsize - 2.
    row_pad_in : float
        Vertical breathing room added to each row, in inches, on top of what the two
        text lines need. Raise it for a taller table.
    line_gap : float
        Vertical distance between the mean and std lines, as a fraction of the cell
        height. Raising it also makes the rows taller, so the text keeps its margins.
    cell_aspect : float
        Cell height divided by width. Below 1 the cells are landscape rectangles:
        the height is set by the two text lines, and the columns are widened to
        match. 1.0 gives square cells, 0 leaves the width to the text alone.
    deadband : float
        Cells within this distance of the reference are drawn white.
    cbar_label : str, optional
        Caption under the colour bar; by default it names the reference column and the
        white band. Pass "" for none.
    cbar_neg_frac : float
    gamma : float
        Exponent of the colour scale. Below 1 it stretches small differences, which
        is why evenly spaced tick VALUES are not evenly spaced on the bar; 1.0 makes
        the scale linear and the ticks evenly spaced.
    cbar_even_ticks : bool
        Place ticks evenly along the drawn bar instead of at round values; the
        labels are then uneven numbers. Ignored when gamma == 1.
        Width of a blue strip left of 0, as a fraction of the bar's DRAWN length
        (not of the value range — the colour scale is compressive, so a small
        negative value would otherwise take up a lot of room). The default 0 starts
        the bar at 0, so nothing negative is shown.
    cbar_end_labels : tuple[str, str] or None
        Optional words at the blue and red ends of the bar, e.g.
        ("fairer", "more unfair"); None (the default) leaves the ends bare.
    """

    if std_fontsize is None:
        std_fontsize = fontsize - 4
    if metric_fontsize is None:
        metric_fontsize = std_fontsize - 2
    if header_fontsize is None:
        header_fontsize = fontsize - 2
    if not metric_fontsize < std_fontsize < fontsize:
        logger.warning(
            f"font sizes usually run glyph < std < mean; got metric={metric_fontsize}, "
            f"std={std_fontsize}, mean={fontsize}"
        )
    if ref_col is None:
        ref_col = df_mean.columns[1]

    # ------------------ SELECT + ORDER ------------------

    desired_groups = ["Age", "Gender", "Skin Tone", "Intersection"]
    existing_groups = df_mean.index.get_level_values(0).unique()
    desired_groups = [g for g in desired_groups if g in existing_groups]
    metrics = df_mean.index.get_level_values(1).unique()

    new_index = pd.MultiIndex.from_product([desired_groups, metrics])
    df_merged = df_mean.reindex(new_index)
    df_sd = df_std.reindex(index=new_index, columns=df_merged.columns)
    df_merged.index.names = ["Demographic Group", "Metric"]

    # Colour and colour scale use the SAME reference.
    reference_for_color = df_merged[ref_col]
    df_diff = df_merged.sub(reference_for_color, axis=0)
    diff_values = df_diff.to_numpy(dtype=float)
    if np.isnan(diff_values).all():
        raise ValueError("No values to plot")
    max_abs = float(np.nanmax(np.abs(diff_values)))

    # ------------------ COLOR MAPPING ------------------

    vmax = max_abs if max_abs > 0 else 1.0
    norm = mcolors.FuncNorm(
        (
            lambda x: np.sign(x) * np.abs(x) ** gamma,
            lambda y: np.sign(y) * np.abs(y) ** (1 / gamma),
        ),
        vmin=-vmax,
        vmax=vmax,
    )
    colormap = plt.colormaps["RdBu_r"]

    def calculate_cell_color(diff):
        """Background and text colour for a cell from its difference to the reference."""
        if np.isnan(diff):
            return (0.9, 0.9, 0.9, 1), "black"
        if max_abs == 0 or abs(diff) < deadband:
            return (1, 1, 1, 1), "black"

        rgba = colormap(norm(diff))
        r, g, b = rgba[:3]
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "black" if brightness > 0.5 else "white"

        return rgba, text_color

    def cell_text(mean, sd):
        """(mean line, std line); the std line is empty when there is none."""
        if np.isnan(mean):
            return "n/a", ""
        if np.isnan(sd):  # a single seed: no std available
            return f"{mean:.{decimals}f}", ""
        return f"{mean:.{decimals}f}", f"±{sd:.{decimals}f}"

    # ------------------ FIGURE ------------------

    n_cols = len(df_merged.columns)

    header = ["", "Metric"] + list(df_merged.columns)
    table_data = [header]
    std_lines = {}  # (row, col) -> std string, drawn after layout in its own size

    prev_group = None
    for i, idx in enumerate(df_merged.index, start=1):
        group, metric = idx
        group_display = group if group != prev_group else ""
        means = []
        for j, c in enumerate(df_merged.columns, start=2):
            mean_line, std_line = cell_text(df_merged.loc[idx, c], df_sd.loc[idx, c])
            means.append(mean_line)
            std_lines[(i, j)] = std_line
        table_data.append([group_display, metric] + means)
        prev_group = group

    # Data columns only as wide as their widest line, plus padding
    GROUP_COL_IN, METRIC_COL_IN = 0.28, 0.95
    auto_width = col_width_in is None
    if auto_width:
        widest = max(
            max(
                _text_width_in(str(c), header_fontsize, "bold")
                for c in df_merged.columns
            ),
            max(
                _text_width_in(str(cell), fontsize)
                for row in table_data[1:]
                for cell in row[2:]
            ),
            max(
                [
                    _text_width_in(line, std_fontsize)
                    for line in std_lines.values()
                    if line
                ]
                or [0.0]
            ),
        )
        col_width_in = widest + cell_pad_in

    # Row height from the two stacked text lines, then widen the columns until each
    # cell has the requested aspect, so raising the fonts keeps the cells rectangular.
    LINE_SPACING, CELL_VPAD_IN = 1.35, row_pad_in
    row_height_in = (fontsize + std_fontsize) * LINE_SPACING / 72 + CELL_VPAD_IN
    # The two baselines sit line_gap apart, so the row must also fit half of each
    # line's own height plus the padding; grow it when line_gap alone would overflow.
    if line_gap < 1:
        row_height_in = max(
            row_height_in,
            ((fontsize + std_fontsize) / 144 + CELL_VPAD_IN) / (1 - line_gap),
        )
    header_height_in = row_height_in * 0.75
    if cell_aspect > 0 and auto_width:
        # only shape the columns when their width was not given explicitly
        col_width_in = max(col_width_in, row_height_in / cell_aspect)

    axes_width_in = GROUP_COL_IN + METRIC_COL_IN + n_cols * col_width_in
    fig_width = axes_width_in / (0.99 - 0.3)  # matches subplots_adjust below
    axes_height_in = header_height_in + len(df_merged) * row_height_in
    TABLE_FIG_HEIGHT = axes_height_in / (0.90 - 0.13)  # matches subplots_adjust below

    fig, ax = plt.subplots(figsize=(fig_width, TABLE_FIG_HEIGHT))
    ax.axis("off")

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 2.5)

    # Column widths and row heights (header height is no longer overwritten)
    HEADER_HEIGHT = header_height_in / axes_height_in
    BODY_HEIGHT = row_height_in / axes_height_in
    col_widths = [
        GROUP_COL_IN / axes_width_in,
        METRIC_COL_IN / axes_width_in,
    ] + [col_width_in / axes_width_in] * n_cols
    for i in range(len(table_data)):
        for j, width in enumerate(col_widths):
            cell = table[(i, j)]
            cell.set_width(width)
            cell.set_height(HEADER_HEIGHT if i == 0 else BODY_HEIGHT)
            cell.set_edgecolor("none")
            cell.set_linewidth(0)
            if i > 0:
                cell.PAD = 0.02

    # Header style
    for j in range(len(header)):
        cell = table[(0, j)]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold", fontsize=header_fontsize)

    # Body style
    for i, idx in enumerate(df_merged.index, start=1):
        cell = table[(i, 0)]
        cell.set_facecolor("#ecf0f1")
        cell.get_text().set_text("")

        cell = table[(i, 1)]
        cell.set_facecolor("#ecf0f1")
        cell.set_text_props(weight="bold", fontsize=metric_fontsize)
        metric_text = cell.get_text().get_text()
        cell.get_text().set_text(rf"${metric_text[0]}_{{{metric_text[1:]}}}$")

        for j, col in enumerate(df_merged.columns, start=2):
            bg_color, text_color = calculate_cell_color(df_diff.loc[idx, col])
            cell = table[(i, j)]
            cell.set_facecolor(bg_color)
            cell.set_text_props(color=text_color, fontsize=fontsize)

    # Both lines drawn here, not by the table: Table repositions its own cell text
    # on every draw, so a set_y on it is discarded and only the std would move.
    fig.canvas.draw()
    share = std_fontsize / (fontsize + std_fontsize)
    drawn = []  # (text artist, cell) pairs, re-centred once below
    probe = None  # a cell that has both lines, used to measure the residual
    for (i, j), std_line in std_lines.items():
        cell = table[(i, j)]
        x = cell.get_x() + cell.get_width() / 2
        y = cell.get_y() + cell.get_height() / 2
        h = cell.get_height()
        color = cell.get_text().get_color()
        mean_line = cell.get_text().get_text()
        cell.get_text().set_text("")

        # with no std the mean sits dead centre; otherwise the pair straddles it
        y_mean = y + h * line_gap * share if std_line else y
        mean_artist = ax.text(
            x,
            y_mean,
            mean_line,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            transform=ax.transData,
            zorder=25,
        )
        if std_line:
            drawn.append((mean_artist, cell))
            drawn.append(
                (
                    ax.text(
                        x,
                        y - h * line_gap * (1 - share),
                        std_line,
                        ha="center",
                        va="center",
                        fontsize=std_fontsize,
                        color=color,
                        transform=ax.transData,
                        zorder=25,
                    ),
                    cell,
                )
            )
            if probe is None:
                probe = (mean_artist, drawn[-1][0], cell)

    # The two font sizes give boxes of unequal height, so the pair lands a little
    # off centre. Measure that residual once and shift every line by it.
    if probe is not None:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        mean_t, std_t, cell = probe
        top = cell.get_window_extent(r).y1 - mean_t.get_window_extent(r).y1
        bottom = std_t.get_window_extent(r).y0 - cell.get_window_extent(r).y0
        inv = ax.transData.inverted()
        shift = inv.transform((0, (top - bottom) / 2))[1] - inv.transform((0, 0))[1]
        for artist, _ in drawn:
            artist.set_y(artist.get_position()[1] + shift)

    # Separators between groups
    group_changes = []
    prev_group = None
    for i, idx in enumerate(df_merged.index, start=1):
        group = idx[0]
        if prev_group is not None and group != prev_group:
            group_changes.append(i)
        prev_group = group

    fig.canvas.draw()
    for row_idx in group_changes:
        for j in range(len(header)):
            cell = table[(row_idx, j)]
            x = cell.get_x()
            y = cell.get_y() + cell.get_height()
            width = cell.get_width()
            line = plt.Line2D(
                [x, x + width],
                [y, y],
                color="black",
                linewidth=1.5,
                transform=ax.transData,
                clip_on=False,
                zorder=10,
            )
            ax.add_line(line)

    fig.canvas.draw()

    for group in desired_groups:
        rows = [i for i, idx in enumerate(df_merged.index, start=1) if idx[0] == group]
        if not rows:
            continue

        first_cell = table[(rows[0], 0)]
        last_cell = table[(rows[-1], 0)]

        x = first_cell.get_x() + first_cell.get_width() / 2
        y_top = first_cell.get_y() + first_cell.get_height()
        y_bottom = last_cell.get_y()
        y_center = (y_top + y_bottom) / 2

        ax.text(
            x,
            y_center,
            group,
            rotation=90,
            ha="center",
            va="center",
            fontsize=header_fontsize,
            fontweight="bold",
            transform=ax.transData,
            zorder=20,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.3, right=0.99, top=0.90, bottom=0.13)

    # Thin light-grey border around the cells only; the colour bar stays outside it
    cells = list(table.get_celld().values())
    x0 = min(c.get_x() for c in cells)
    y0 = min(c.get_y() for c in cells)
    x1 = max(c.get_x() + c.get_width() for c in cells)
    y1 = max(c.get_y() + c.get_height() for c in cells)
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=BORDER_COLOR,
            linewidth=BORDER_WIDTH,
            transform=ax.transData,
            clip_on=False,
            zorder=30,
        )
    )

    # ------------------ COLORBAR ------------------

    pad = 0.02
    height = 0.02

    def nice_cap_floor(max_abs):
        """Floor max_abs to the nearest value of the form m × 10^n, m in 1..9."""
        if max_abs <= 0:
            return 0.0
        exponent = math.floor(math.log10(max_abs))
        base = 10**exponent
        for m in [9, 8, 7, 6, 5, 4, 3, 2, 1]:
            cap = m * base
            if cap <= max_abs:
                return cap
        return base

    nice_cap = nice_cap_floor(max_abs)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = table.get_window_extent(renderer)
    bbox_fig = bbox.transformed(fig.transFigure.inverted())

    cbar_width = bbox_fig.width * cbar_width_frac
    cbar_ax = fig.add_axes(
        [
            bbox_fig.x0 + (bbox_fig.width - cbar_width) / 2,
            bbox_fig.y0 - pad - height,
            cbar_width,
            height,
        ]
    )
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.outline.set_edgecolor(BORDER_COLOR)  # matplotlib draws a black frame
    cbar.outline.set_linewidth(BORDER_WIDTH)

    # Show the whole positive range but only a sliver of blue, and label from 0 up:
    # negative differences are few and small, so full negative ticks would misread.
    # Invert the colour scale so the strip is cbar_neg_frac of the DRAWN bar:
    # forward(x) = |x|**gamma, so a visible fraction f sits at cap * f**(1/gamma).
    neg_extent = nice_cap * cbar_neg_frac ** (1 / gamma) if cbar_neg_frac > 0 else 0.0
    cbar_ax.set_xlim(-neg_extent, nice_cap)

    min_diff = float(np.nanmin(diff_values))
    if min_diff < -neg_extent:
        logger.info(
            f"colour bar shows blue down to {-neg_extent:.3g}; the most negative cell "
            f"is {min_diff:.3g} (raise cbar_neg_frac to cover it)"
        )

    # drop to 3 labels when 5 would not fit side by side on the bar
    def _ticks(n):
        if cbar_even_ticks and gamma != 1:
            # evenly spaced on the DRAWN bar: forward(t) = t**gamma is linear in i
            values = nice_cap * (np.linspace(0.0, 1.0, n) ** (1 / gamma))
        else:
            values = np.linspace(0.0, nice_cap, n)
        return values, ["0" if t == 0 else f"{t:.2g}" for t in values]

    ticks, labels = _ticks(5)
    needed = sum(_text_width_in(label, fontsize) for label in labels) * 2.2
    if needed > cbar_width * fig_width * (1 / (1 + cbar_neg_frac)):
        ticks, labels = _ticks(3)
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels(labels)
    cbar.ax.tick_params(labelsize=fontsize)

    # One caption line under the tick labels: ends say the direction, middle the reference
    if cbar_label is None:
        cbar_label = f"\u0394 vs {ref_col}"
    parts = [(0.5, "center", cbar_label)]
    if cbar_end_labels:
        blue_end, red_end = cbar_end_labels
        parts += [
            (0.0, "left", f"\u2190 {blue_end}"),
            (1.0, "right", f"{red_end} \u2192"),
        ]
    for x, ha, text in parts:
        if text:
            cbar_ax.text(
                x,
                -2.8,
                text,
                transform=cbar_ax.transAxes,
                ha=ha,
                va="top",
                fontsize=fontsize - 1,
            )

    if save_path:
        plt.savefig(f"{save_path}.pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        logger.info(f"Saved {save_path}.pdf")

    plt.close()
