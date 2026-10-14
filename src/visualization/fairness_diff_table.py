"""
Fairness-difference table with mean ± std over seeds.

For method `alg`, each seed and each evaluated test set ds < final_task:

    diff(ds) = fairness(eval ds, after final_task) - fairness(eval ds, after task ds)

The difference is taken WITHIN each seed and only then aggregated, because both terms
come from the same run and are not independent.

Input: fairness_group.csv written by fairness_from_predictions.py (columns
method, seed, task, eval_task, Group, n_subgroups, FEO, FDP, FOAE, FEOP).
"""

from __future__ import annotations

import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.patches import Rectangle

# Attribute column in the fairness CSV -> row label in the table.
# "Skin Tone" is the 10-level attribute; "Skin Tone Group" is deliberately excluded.
GROUP_LABELS = {
    "Predicted Age": "Age",
    "Predicted Gender": "Gender",
    "Skin Tone": "Skin Tone",
    "Intersection": "Intersection",
}
DIFF_METRICS = ["FEO", "FDP", "FOAE", "FEOP"]
CBAR_WIDTH_FRAC = 0.9  # colour bar width as a fraction of the table width
BORDER_COLOR, BORDER_WIDTH = "#cccccc", 0.8  # outer table border


def _text_width_in(text: str, fontsize: float, weight: str = "normal") -> float:
    """Rendered width of `text` in inches."""
    probe = plt.figure(figsize=(2, 2))
    artist = probe.text(0, 0, text, fontsize=fontsize, fontweight=weight)
    width = (
        artist.get_window_extent(renderer=probe.canvas.get_renderer()).width / probe.dpi
    )
    plt.close(probe)
    return width


def load_fairness_diff(
    fairness_group_csv: str,
    alg: str,
    final_task: int = 11,
    metrics: list[str] = DIFF_METRICS,
    fallback_method: str | None = "naive",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_mean, df_std, df_n), each indexed by (Group, Metric) with columns
    "DS {ds}" for ds = 1..final_task-1. std is the sample std over seeds (ddof=1);
    df_n is the number of seeds behind each cell.

    fallback_method: where a method has no checkpoint of its own for a dataset's
    diagonal (methods that start from the shared task-1 model), that diagonal is taken
    from this method's run with the same seed. None disables the substitution.
    """
    fg_all = pd.read_csv(fairness_group_csv)
    fg_all = fg_all[fg_all["Group"].isin(GROUP_LABELS)].copy()
    fg_all["Group"] = fg_all["Group"].map(GROUP_LABELS)

    fg = fg_all[fg_all["method"] == alg]
    if fg.empty:
        raise ValueError(f"No rows for method '{alg}' in {fairness_group_csv}")
    n_seeds = fg["seed"].nunique()

    keys = ["seed", "eval_task", "Group"]
    final = fg.loc[
        (fg["task"] == final_task) & (fg["eval_task"] < final_task), keys + metrics
    ]
    diag = fg.loc[
        (fg["task"] == fg["eval_task"]) & (fg["eval_task"] < final_task), keys + metrics
    ]

    if fallback_method is not None and fallback_method != alg:
        missing = set(
            map(tuple, final[["seed", "eval_task"]].drop_duplicates().to_numpy())
        ) - set(map(tuple, diag[["seed", "eval_task"]].drop_duplicates().to_numpy()))
        if missing:
            fb = fg_all[
                (fg_all["method"] == fallback_method)
                & (fg_all["task"] == fg_all["eval_task"])
            ]
            fb = fb[
                pd.MultiIndex.from_frame(fb[["seed", "eval_task"]]).isin(
                    sorted(missing)
                )
            ]
            if fb.empty:
                logger.warning(
                    f"{alg}: no diagonal for (seed, dataset) {sorted(missing)} and none in "
                    f"'{fallback_method}' either; those columns stay empty"
                )
            else:
                logger.info(
                    f"{alg}: diagonal for (seed, dataset) "
                    f"{sorted(map(tuple, fb[['seed', 'eval_task']].drop_duplicates().to_numpy()))} "
                    f"taken from '{fallback_method}'"
                )
                diag = pd.concat([diag, fb[keys + metrics]], ignore_index=True)

    d = final.merge(
        diag,
        on=keys,
        how="outer",
        suffixes=("_final", "_diag"),
        validate="one_to_one",
        indicator=True,
    )
    unmatched = d[d["_merge"] != "both"]
    if not unmatched.empty:
        logger.warning(
            f"{alg}: {len(unmatched)} (seed, eval_task, Group) rows lack a final or a "
            f"diagonal value and are excluded:\n{unmatched[keys + ['_merge']]}"
        )
    d = d[d["_merge"] == "both"]

    for m in metrics:
        d[m] = d[f"{m}_final"] - d[f"{m}_diag"]

    long = d.melt(
        id_vars=keys, value_vars=metrics, var_name="Metric", value_name="diff"
    )
    agg = long.groupby(["Group", "Metric", "eval_task"])["diff"].agg(
        ["mean", "std", "count"]
    )
    if (agg["count"] < n_seeds).any():
        logger.warning(
            f"{alg}: some cells use fewer than {n_seeds} seeds (see df_n); "
            "their std is less reliable"
        )

    all_datasets = list(range(1, final_task))

    def wide(stat: str) -> pd.DataFrame:
        # every dataset gets a column, empty where the difference is unavailable
        table = agg[stat].unstack("eval_task").reindex(columns=all_datasets)
        table.columns = [f"DS {c}" for c in table.columns]
        return table

    logger.info(f"{alg}: differences over {n_seeds} seed(s), final task {final_task}")
    return wide("mean"), wide("std"), wide("count")


def visualize_fairness_diff_by_generator_matplotlib(
    df_mean,
    df_std,
    alg,
    save_path=None,
    dpi=300,
    mean_decimals=3,
    std_decimals=3,
    col_width_in=None,
    cell_pad_in=0.08,
    mean_fontsize=16,
    std_fontsize=12,
    metric_fontsize=None,
    header_fontsize=None,
    cell_aspect=0.6,
    line_gap=0.39,
    row_pad_in=0.22,
    gamma=0.6,
    cbar_even_ticks=False,
):
    """
    Visualize fairness metric DIFFERENCES (mean ± std over seeds) for one method.

    Parameters
    ----------
    df_mean, df_std : pd.DataFrame
        From load_fairness_diff: index (Group, Metric), columns "DS {ds}".
    alg : str
        Algorithm name
    save_path : str, optional
        Path prefix for saving PDF
    dpi : int
        Figure resolution
    mean_decimals, std_decimals : int
        Decimals shown for mean and std
    col_width_in : float, optional
        Width of the data columns in inches; by default just wide enough for the widest
        header or value. When given, it wins over cell_aspect.
    cell_pad_in : float
        Horizontal padding added to the widest value in a data column, in inches.
    mean_fontsize, std_fontsize : int
        Font sizes of the two lines in a cell; the mean is drawn larger.
    metric_fontsize : int, optional
        Font size of the metric glyph ($\Delta F_{DP}$ etc.); defaults to
        std_fontsize - 2, so the glyph is the smallest text in the row.
    header_fontsize : int, optional
        Font size of the column headers and group labels; defaults to mean_fontsize.
    cell_aspect : float
        Cell height divided by width. Below 1 the cells are landscape rectangles:
        the height is set by the two text lines, and the columns are widened to
        match (a long header can still make them wider). 0 leaves the width to the
        text alone.
    line_gap : float
        Vertical distance between the mean and std lines, as a fraction of the cell
        height. Raising it also makes the rows taller, so the text keeps its margins.
    row_pad_in : float
        Vertical breathing room added to each row, in inches, on top of what the two
        text lines need. Raise it for a taller table.
    gamma : float
        Exponent of the colour scale. Below 1 it stretches small differences, which
        is why evenly spaced tick VALUES are not evenly spaced on the bar; 1.0 makes
        the scale linear and the ticks evenly spaced.
    cbar_even_ticks : bool
        Place ticks evenly along the drawn bar instead of at round values; the
        labels are then uneven numbers. Ignored when gamma == 1.
    """

    if metric_fontsize is None:
        metric_fontsize = std_fontsize - 2
    if header_fontsize is None:
        header_fontsize = mean_fontsize
    if not metric_fontsize < std_fontsize < mean_fontsize:
        logger.warning(
            f"font sizes usually run glyph < std < mean; got metric={metric_fontsize}, "
            f"std={std_fontsize}, mean={mean_fontsize}"
        )

    # ------------------ SELECT + ORDER ------------------

    df_std = df_std.reindex(index=df_mean.index, columns=df_mean.columns)

    desired_groups = ["Age", "Gender", "Skin Tone", "Intersection"]
    existing_groups = df_mean.index.get_level_values(0).unique()
    desired_groups = [g for g in desired_groups if g in existing_groups]
    metrics = df_mean.index.get_level_values(1).unique()

    full_index = pd.MultiIndex.from_product([desired_groups, metrics])
    df_diff = df_mean.reindex(full_index)
    df_sd = df_std.reindex(full_index)
    df_diff.index.names = ["", "Unfairness"]
    df_sd.index.names = ["", "Unfairness"]

    values = df_diff.to_numpy(dtype=float)
    if np.isnan(values).all():
        raise ValueError("df_mean contains no values to plot")
    max_abs = float(np.nanmax(np.abs(values)))

    # ------------------ COLOR MAPPING ------------------

    # Same sign-preserving power transform as before, expressed as a norm so the
    # colorbar uses exactly the mapping the cells use.
    vmax = max_abs if max_abs > 0 else 1.0
    norm = mcolors.FuncNorm(
        (
            lambda x: np.sign(x) * np.abs(x) ** gamma,
            lambda y: np.sign(y) * np.abs(y) ** (1 / gamma),
        ),
        vmin=-vmax,
        vmax=vmax,
    )
    cmap = plt.colormaps["RdBu_r"]

    def cell_color(diff):
        if np.isnan(diff):
            return (0.9, 0.9, 0.9, 1), "black"
        if max_abs == 0 or abs(diff) < 0.025:
            return (1, 1, 1, 1), "black"

        rgba = cmap(norm(diff))
        r, g, b = rgba[:3]
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "black" if brightness > 0.5 else "white"

        return rgba, text_color

    def cell_text(mean, sd):
        """(mean line, std line); the std line is empty when there is none."""
        if np.isnan(mean):
            return "n/a", ""
        if np.isnan(sd):  # a single seed: no std available
            return f"{mean:+.{mean_decimals}f}", ""
        return f"{mean:+.{mean_decimals}f}", f"±{sd:.{std_decimals}f}"

    # ------------------ FIGURE ------------------

    n_rows, n_cols = df_diff.shape

    header = ["", "Metric"] + list(df_diff.columns)
    table_data = [header]
    std_lines = {}  # (row, col) -> std string, drawn after layout in its own size
    prev_group = None
    for i, (group, metric) in enumerate(df_diff.index, start=1):
        g_disp = group if group != prev_group else ""
        means = []
        for j, c in enumerate(df_diff.columns, start=2):
            mean_line, std_line = cell_text(
                df_diff.loc[(group, metric), c], df_sd.loc[(group, metric), c]
            )
            means.append(mean_line)
            std_lines[(i, j)] = std_line
        table_data.append([g_disp, metric] + means)
        prev_group = group

    # Data columns only as wide as their widest line, plus padding
    GROUP_COL_IN, METRIC_COL_IN = 0.28, 0.95
    auto_width = col_width_in is None
    if auto_width:
        col_width_in = cell_pad_in + max(
            max(
                _text_width_in(str(c), header_fontsize, "bold") for c in df_diff.columns
            ),
            max(
                _text_width_in(str(cell), mean_fontsize)
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
    # Row height from the two stacked text lines, then widen the columns until each
    # cell has the requested aspect, so raising the fonts keeps the cells rectangular.
    LINE_SPACING = 1.35
    row_height_in = (mean_fontsize + std_fontsize) * LINE_SPACING / 72 + row_pad_in
    # The two baselines sit line_gap apart, so the row must also fit half of each
    # line's own height plus the padding; grow it when line_gap alone would overflow.
    if line_gap < 1:
        row_height_in = max(
            row_height_in,
            ((mean_fontsize + std_fontsize) / 144 + row_pad_in) / (1 - line_gap),
        )
    header_height_in = row_height_in * 0.75
    if cell_aspect > 0 and auto_width:
        # only shape the columns when their width was not given explicitly
        col_width_in = max(col_width_in, row_height_in / cell_aspect)

    axes_width_in = GROUP_COL_IN + METRIC_COL_IN + n_cols * col_width_in
    fig_width = axes_width_in / (0.99 - 0.3)  # matches subplots_adjust below
    axes_height_in = header_height_in + len(df_diff) * row_height_in
    TABLE_FIG_HEIGHT = axes_height_in / (0.90 - 0.13)  # matches subplots_adjust below

    fig, ax = plt.subplots(figsize=(fig_width, TABLE_FIG_HEIGHT))
    ax.axis("off")

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(mean_fontsize)
    table.scale(1, 2.5)

    for cell in table.get_celld().values():
        cell.set_edgecolor("none")
        cell.set_linewidth(0)

    # ------------------ HEADER STYLE ------------------

    for j in range(len(header)):
        cell = table[(0, j)]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold", fontsize=header_fontsize)
        cell.set_edgecolor("none")

    # ------------------ BODY STYLE ------------------

    for i, (group, metric) in enumerate(df_diff.index, start=1):
        table[(i, 0)].set_facecolor("#ecf0f1")
        table[(i, 0)].set_text_props(weight="bold")
        table[(i, 0)].get_text().set_text("")

        mcell = table[(i, 1)]
        mcell.set_facecolor("#ecf0f1")
        mcell.set_text_props(weight="bold", fontsize=metric_fontsize)
        txt = mcell.get_text().get_text()
        mcell.get_text().set_text(rf"$\Delta {txt[0]}_{{{txt[1:]}}}$")

        for j, col in enumerate(df_diff.columns, start=2):
            bg, tc = cell_color(df_diff.loc[(group, metric), col])
            cell = table[(i, j)]
            cell.set_facecolor(bg)
            cell.set_text_props(color=tc)

    # Uniform column widths; body rows taller to fit the two-line mean ± std
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

    # Both lines drawn here, not by the table: Table repositions its own cell text
    # on every draw, so a set_y on it is discarded and only the std would move.
    fig.canvas.draw()
    share = std_fontsize / (mean_fontsize + std_fontsize)
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
            fontsize=mean_fontsize,
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

    # Visual separators between groups
    group_changes = []
    prev_group = None
    for i, idx in enumerate(df_diff.index, start=1):
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
        rows = [i for i, idx in enumerate(df_diff.index, start=1) if idx[0] == group]
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

    # Thin light-grey border around the whole table
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

    fig.subplots_adjust(left=0.3, right=0.99, top=0.90, bottom=0.13)
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    bbox = table.get_window_extent(renderer)
    bbox_fig = bbox.transformed(fig.transFigure.inverted())

    pad = 0.02  # matches the algs table, so both figures end up the same height
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

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes(
        [
            bbox_fig.x0 + bbox_fig.width * (1 - CBAR_WIDTH_FRAC) / 2,
            bbox_fig.y0 - pad - height,
            bbox_fig.width * CBAR_WIDTH_FRAC,
            height,
        ]
    )
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.outline.set_edgecolor(BORDER_COLOR)  # matplotlib draws a black frame
    cbar.outline.set_linewidth(BORDER_WIDTH)

    if cbar_even_ticks and gamma != 1:
        # evenly spaced on the DRAWN bar: forward(t) = |t|**gamma is linear in i
        frac = np.linspace(-1.0, 1.0, 5)
        ticks = nice_cap * np.sign(frac) * np.abs(frac) ** (1 / gamma)
    else:
        ticks = np.linspace(-nice_cap, nice_cap, 5)
    ticks = np.where(np.isclose(ticks, 0.0), 0.0, ticks)
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels(["0" if t == 0 else f"{t:.2g}" for t in ticks])
    cbar.ax.tick_params(labelsize=mean_fontsize)

    if save_path:
        plt.savefig(f"{save_path}.pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        logger.info(f"Saved to {save_path}.pdf")

    plt.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fairness-csv", required=True, help="fairness_group.csv")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--final-task", type=int, default=11)
    p.add_argument("--out-prefix", default="fairness_diff")
    args = p.parse_args()

    for method in args.methods:
        mean, std, n = load_fairness_diff(args.fairness_csv, method, args.final_task)
        visualize_fairness_diff_by_generator_matplotlib(
            mean, std, method, save_path=f"{args.out_prefix}_{method}"
        )
