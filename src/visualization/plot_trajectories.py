"""Subgroup trajectories over tasks 1 -> 11, one panel per demographic attribute.

Layout and x-axis follow paper_fisher_cos_sim_plots_script.py (Fisher cosine
similarity trajectories): one row of four panels (Gender, Age, Skin tone,
Gender x tone), segments fading in from task 1, an arrow on the last segment, a
hollow circle at task 1, legends below each panel.

x (log scale): the subgroup's share of the training data of task t
   (annotated images of that task's dataset, real and fake together).
   --x cumulative: share of the training data of tasks 1..t
   --x overall:    share of all training data (constant; vertical trajectories)
   --x task:       the task index

Colours per attribute (--gender-colors, --age-colors, --tone-colors,
--intersection-colors): "reference" (default; as in the Fisher-similarity
figures: red/blue gender, plasma age, light-to-dark browns for skin tone with
solid/dotted/dashed for Light/Medium/Dark, gender x tone Female solid / Male
dashed), "categorical" (distinct hues; skin tone and gender x tone
coloured by Light/Medium/Dark group with dash patterns) or any matplotlib colour
map, e.g. plasma, viridis, cividis. Defaults: TRAJ_COLORS in
constants.py.
y: a quantity in [0, 1]:

  memory_share_{method}.pdf
      subgroup's buffer samples / total buffer size at the end of task t
      (all samples, including those without demographic annotation)
  memory_share_annotated_{method}.pdf
      subgroup's buffer samples / annotated buffer samples at the end of task t
      (unannotated samples left out, so the subgroups of an attribute sum to 1)
  routing_accuracy_{model}.pdf
      after training through task t: share of the subgroup's test samples of
      tasks 1..t routed to the correct prompt (pooled per seed, then averaged)
  routing_error_{model}.pdf
      1 - routing accuracy: share of those test samples sent to a wrong prompt

  fisher_similarity.pdf
      Fisher cosine similarity at the last checkpoint per evaluated dataset
      (results of fisher_evolution_pipeline.py via run_fisher.sh; --fisher-csv).
      As in paper_fisher_cos_sim_plots_script.py, x comes from the n column of the Fisher results (first n per
      subgroup and dataset; --fisher-x-counts metadata: training images instead),
      as a cumulative share over tasks 1..t by default (--fisher-x), and the
      y-axis is scaled to the data.
  trajectory_grid.pdf
      rows x 4 attributes, rows set by --grid (default: routing accuracy of SOYO,
      memory share of GradMatch, Fisher similarity); no titles or captions.

Memory-share panels run from 0 to --memory-ymax (default 0.6), raised to the
next 0.1 when a share is higher (annotated shares can be, as their whole is
smaller); Fisher panels are scaled to the data, all others 0-1.
For memory shares of annotated samples only (subgroup / annotated buffer
samples, as the x-axis and routing), use memory_annotated:<method> rows, e.g.
GRID_ROWS = ["routing:soyo", "memory_annotated:gradmatch", "fisher"].

Small subgroups (--min-n, default 0 = off: every subgroup is shown): points
whose n is below the threshold are faded (--min-n-mode fade) or left out
(drop). n per point: Fisher = the n column (images of the subgroup in dataset
t, mean over seeds); routing = test samples of tasks 1..t (mean over seeds);
memory = the subgroup's annotated training images in task t. --min-n-rows sets
which kinds of rows it applies to (default: all).
Each line starts with an open circle at task 1 and ends with an arrow at the
last task. Values are means over seeds. Rows or figures whose input is missing
are skipped with a message.

    python plot_trajectories.py
    python plot_trajectories.py --grid-only --grid routing:pina memory:derpp fisher
    python plot_trajectories.py --grid memory_annotated:gradmatch routing_error:soyo
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from plot_routing_heatmaps import save, with_names

from evaluations.buffer_analysis import load_metadata
from evaluations.routing_analysis import subgroup_key
from utils.constants import (
    CKPT,
    DEMOGRAPHICS,
    FAMILY_SIDE,
    FISHER_CSV,
    FISHER_X_COUNTS,
    FISHER_X_MODE,
    GRID_ROWS,
    LINE_PALETTE,
    LINE_STYLES,
    MEMORY_YMAX,
    METHOD_LABELS,
    MODEL_LABELS,
    RC_PARAMS,
    REF_AGE_CMAP,
    REF_AGE_RANGE,
    REF_AGE_STYLES,
    REF_GENDER_COLORS,
    REF_SKIN_RAMP,
    REF_TONE_STYLES,
    ROW_FAMILIES,
    SUBGROUP_NAMES,
    TASK_NAMES,
    TONE_GROUPS,
    TRAJ_CMAP_RANGE,
    TRAJ_COLORS,
)

SKIN_DARKEN = 0.75  # scales the RGB of the skin-tone browns; 1 = unchanged
# Small subgroups: points with n < MIN_N are faded or dropped (0 = keep all)
MIN_N = 0
MIN_N_MODE = "fade"  # "fade" (drawn with MIN_N_ALPHA) or "drop" (left out)
MIN_N_ALPHA = 0.15  # opacity of faded segments / start circles
MIN_N_ROWS = ("fisher", "routing", "memory")  # kinds of rows it applies to


def darken(color, factor: float = SKIN_DARKEN) -> tuple:
    return tuple(c * factor for c in to_rgb(color))


plt.rcParams.update(RC_PARAMS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--buffer-dir", type=Path, default=Path("buffer_analysis"))
    p.add_argument("--routing-dir", type=Path, default=Path("routing_analysis"))
    p.add_argument(
        "--metadata", type=Path, default=Path("metadata/train.csv")
    )
    p.add_argument("--out-dir", type=Path, default=Path("figures_trajectories"))
    p.add_argument(
        "--grid",
        nargs="+",
        default=GRID_ROWS,
        metavar="ROW",
        help="Rows of the grid figure, top to bottom: routing:<model>, "
        "routing_error:<model>, memory:<method>, "
        "memory_annotated:<method>, fisher (default: " + " ".join(GRID_ROWS) + ")",
    )
    p.add_argument(
        "--family-side",
        choices=["left", "right"],
        default=FAMILY_SIDE,
        help="Side of the grid for the method-family labels (Prompt based, ...)",
    )
    p.add_argument(
        "--grid-name",
        default="trajectory_grid",
        help="File name of the grid figure (without extension)",
    )
    p.add_argument("--grid-only", action="store_true", help="Only make the grid figure")
    p.add_argument(
        "--memory-ymax",
        type=float,
        default=MEMORY_YMAX,
        help="Top of the y-axis for memory-share panels",
    )
    p.add_argument(
        "--fisher-csv",
        type=Path,
        default=None,
        help="Fisher results (default: first existing path in FISHER_CSV)",
    )
    p.add_argument(
        "--fisher-x-counts",
        choices=["fisher", "metadata"],
        default=FISHER_X_COUNTS,
        help="Subgroup counts for the x-axis of the Fisher panels: the n column of "
        "the Fisher results (as paper_fisher_cos_sim_plots_script.py) or annotated "
        f"training images in --metadata (default: {FISHER_X_COUNTS})",
    )
    p.add_argument(
        "--fisher-x",
        choices=["cumulative", "dataset", "overall"],
        default=FISHER_X_MODE,
        help="x-axis of the Fisher panels: cumulative share over tasks 1..t "
        "(cumulative_share=True in paper_fisher_cos_sim_plots_script.py), share of "
        f"task t only, or of all tasks (default: {FISHER_X_MODE}); --x task "
        "overrides it",
    )
    p.add_argument(
        "--fisher-metric",
        choices=["cosine_similarity", "cosine_distance"],
        default="cosine_similarity",
    )
    for flags, attr in (
        (("--gender-colors", "--sex-colors"), "Predicted Gender"),
        (("--age-colors",), "Predicted Age"),
        (("--tone-colors",), "Skin Tone"),
        (("--intersection-colors",), "Intersection"),
    ):
        p.add_argument(
            *flags,
            default=TRAJ_COLORS[attr],
            metavar="NAME",
            help=f"'reference', 'categorical' or a matplotlib colour map, e.g. plasma, "
            f"viridis, cividis (default: {TRAJ_COLORS[attr]})",
        )
    p.add_argument(
        "--min-n",
        type=int,
        default=MIN_N,
        help="Fade or drop points whose subgroup n is below this (default: "
        f"{MIN_N} = show everything)",
    )
    p.add_argument(
        "--min-n-mode",
        choices=["fade", "drop"],
        default=MIN_N_MODE,
        help=f"What happens to points below --min-n (default: {MIN_N_MODE})",
    )
    p.add_argument(
        "--min-n-rows",
        nargs="+",
        choices=["fisher", "routing", "memory"],
        default=list(MIN_N_ROWS),
        help="Kinds of rows --min-n applies to (default: all)",
    )
    p.add_argument(
        "--x",
        choices=list(X_LABELS),
        default="dataset",
        help="x-axis: subgroup's share of the task's training dataset "
        "(default), cumulative share over tasks 1..t, share of the "
        "whole training set, or the task index",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# x-axis and styles
# ---------------------------------------------------------------------------
X_LABELS = {
    "dataset": "Share of training data",
    "cumulative": "Cumulative share of training data",
    "overall": "Share of all training data",
    "task": "Task",
}
# Panel titles (as in the Fisher-similarity figures)
FAMILY_TITLES = {
    "Predicted Gender": "Gender",
    "Predicted Age": "Age",
    "Skin Tone": "Skin tone",
    "Intersection": "Gender × tone",
}
# Clean percentage ticks for the log x-axis
CANDIDATE_TICKS = [
    0.001,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.025,
    0.03,
    0.05,
    0.08,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
]


def subgroup_share(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    """x per (attribute, subgroup, task) from annotated training-image counts.

    dataset:    subgroup's share of task t's training data
    cumulative: subgroup's share of the training data of tasks 1..t
    overall:    subgroup's share of all training data (same x for every task)
    """
    parts = []
    for attr in DEMOGRAPHICS:
        d = (
            data[data["demographic"] == attr]
            .pivot_table(
                index="task",
                columns="subgroup",
                values="n_data",
                aggfunc="sum",
                fill_value=0,
            )
            .sort_index()
        )
        if mode == "cumulative":
            d = d.cumsum()
        elif mode == "overall":
            d = pd.DataFrame([d.sum()] * len(d), index=d.index)
        share = d.div(d.sum(axis=1), axis=0)
        parts.append(share.stack().rename("x").reset_index().assign(demographic=attr))
    return pd.concat(parts, ignore_index=True)


def pct_label(t: float) -> str:
    """0.0025 -> '0.25%', 0.2 -> '20%' (as in the Fisher-similarity figures)."""
    txt = f"{t * 100:.2f}" if t < 0.01 else f"{t * 100:.1f}"
    return txt.rstrip("0").rstrip(".") + "%"


def tone_groups(path: Path) -> dict[str, str]:
    """Skin tone name -> Light/Medium/Dark, from the metadata's Skin Tone Group."""
    try:
        m = pd.read_csv(
            path, usecols=["Skin Tone", "Skin Tone Group"], low_memory=False
        )
    except (ValueError, FileNotFoundError):
        m = pd.DataFrame(columns=["Skin Tone", "Skin Tone Group"])
    m = m.apply(pd.to_numeric, errors="coerce")
    m = m[(m["Skin Tone"] > 0) & (m["Skin Tone Group"] >= 0)].dropna()
    tones = SUBGROUP_NAMES["Skin Tone"]
    if m.empty:  # fall back to three equal blocks
        logger.warning(
            "No Skin Tone Group in metadata; grouping tones 1-4 / 5-7 / 8-10"
        )
        return {
            name: TONE_GROUPS[0 if t <= 4 else 1 if t <= 7 else 2]
            for t, name in tones.items()
        }
    # Most common group per tone; groups ordered by their mean tone (lightest first)
    mode = m.groupby("Skin Tone")["Skin Tone Group"].agg(lambda s: s.mode().iloc[0])
    order = m.groupby("Skin Tone Group")["Skin Tone"].mean().sort_values().index
    names = {g: TONE_GROUPS[min(i, len(TONE_GROUPS) - 1)] for i, g in enumerate(order)}
    return {tones[int(t)]: names[g] for t, g in mode.items() if int(t) in tones}


def cmap_colors(name: str, n: int) -> list:
    """n colours spread over TRAJ_CMAP_RANGE of a matplotlib colour map."""
    lo, hi = TRAJ_CMAP_RANGE
    return [plt.get_cmap(name)(v) for v in np.linspace(lo, hi, max(n, 1))]


def check_colors(args) -> dict:
    choice = {
        "Predicted Gender": args.gender_colors,
        "Predicted Age": args.age_colors,
        "Skin Tone": args.tone_colors,
        "Intersection": args.intersection_colors,
    }
    for attr, name in choice.items():
        if name not in ("reference", "categorical") and name not in plt.colormaps():
            raise SystemExit(
                f"Unknown colour map {name!r} for {attr}; use 'reference', "
                "'categorical' or a matplotlib colour map such as plasma or viridis"
            )
    return choice


def reference_styles(attr: str, subgroups: list[str], tone_of: dict[str, str]) -> dict:
    """Colours of the Fisher-similarity figures (REF_* in constants.py)."""
    skin = LinearSegmentedColormap.from_list("skin", REF_SKIN_RAMP)
    if attr == "Predicted Gender":
        return {
            sg: dict(color=REF_GENDER_COLORS.get(sg, "black"), ls="-")
            for sg in subgroups
        }
    if attr == "Predicted Age":
        lo, hi = REF_AGE_RANGE
        cols = [
            plt.get_cmap(REF_AGE_CMAP)(v) for v in np.linspace(lo, hi, len(subgroups))
        ]
        return {
            sg: dict(color=c, ls=REF_AGE_STYLES[i % len(REF_AGE_STYLES)])
            for i, (sg, c) in enumerate(zip(subgroups, cols))
        }
    if attr == "Skin Tone":  # colour by tone 1 -> 10, dash pattern by tone group
        tones = SUBGROUP_NAMES["Skin Tone"]
        pos = {name: (t - 1) / max(len(tones) - 1, 1) for t, name in tones.items()}
        return {
            sg: dict(
                color=darken(skin(pos.get(sg, 1.0))),
                ls=REF_TONE_STYLES.get(tone_of.get(sg), "-"),
            )
            for sg in subgroups
        }
    if attr == "Intersection":  # tone group -> light/mid/dark brown, gender by dash
        shade = dict(zip(TONE_GROUPS, (darken(skin(v)) for v in (0.1, 0.5, 0.95))))
        return {
            sg: dict(
                color=shade.get(sg.partition(", ")[2], "black"),
                ls="-" if sg.startswith("Female") else "--",
            )
            for sg in subgroups
        }
    return {
        sg: dict(color=LINE_PALETTE[i % len(LINE_PALETTE)], ls="-")
        for i, sg in enumerate(subgroups)
    }


def styles(
    attr: str,
    subgroups: list[str],
    tone_of: dict[str, str],
    colors: str = "categorical",
) -> dict:
    """Colour and line style per subgroup: hue + dash pattern keep every line unique.

    colors: "reference", "categorical" or a matplotlib colour map name (ordered
    subgroups get colours along the map: Child -> Senior, Skin Tone 1 -> 10,
    Light -> Dark).
    """
    if colors == "reference":
        return reference_styles(attr, subgroups, tone_of)
    if colors != "categorical":
        if attr == "Intersection":  # tone level along the map, sex by dash
            shade = dict(zip(TONE_GROUPS, cmap_colors(colors, len(TONE_GROUPS))))
            return {
                sg: dict(
                    color=shade.get(sg.partition(", ")[2], "black"),
                    ls="-" if sg.startswith("Female") else "--",
                )
                for sg in subgroups
            }
        cols = cmap_colors(colors, len(subgroups))
        return {sg: dict(color=c, ls="-") for sg, c in zip(subgroups, cols)}
    hue = dict(zip(TONE_GROUPS, LINE_PALETTE))  # Light, Medium, Dark
    out = {}
    if attr == "Skin Tone":
        seen = {}
        for sg in subgroups:
            g = tone_of.get(sg, TONE_GROUPS[-1])
            k = seen.get(g, 0)
            seen[g] = k + 1
            out[sg] = dict(color=hue[g], ls=LINE_STYLES[k % len(LINE_STYLES)])
    elif attr == "Intersection":
        for sg in subgroups:
            sex, _, tone = sg.partition(", ")
            out[sg] = dict(
                color=hue.get(tone, LINE_PALETTE[3]),
                ls="-" if sex == "Female" else "--",
            )
    else:
        for i, sg in enumerate(subgroups):
            out[sg] = dict(color=LINE_PALETTE[i % len(LINE_PALETTE)], ls="-")
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
REF_RC = {  # style of the Fisher-similarity trajectory figures
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
}
ALPHA_START = 0.3  # first segment's opacity; rises to 1 at the last task

LINE_WIDTH = 1.8  # trajectories and legend handles


def with_x(values: pd.DataFrame, xs: pd.DataFrame, x_mode: str) -> pd.DataFrame:
    df = values.merge(xs, on=["demographic", "subgroup", "task"], how="left")
    if x_mode == "task":
        df["x"] = df["task"]
    else:
        df = df[df["x"] > 0]  # log axis: a subgroup absent from a task has no point
    return df


def fit_ymax(values, ymax: float) -> float:
    """ymax, raised to the next 0.1 if a value in values["y"] is higher (so no
    line is cut off, e.g. annotated memory shares)."""
    if values is None or values.empty:
        return ymax
    top = float(values["y"].max())
    return max(ymax, math.ceil(top * 10 - 1e-9) / 10) if top > ymax else ymax


def apply_min_n(values, min_n: int, mode: str, what: str = ""):
    """Mark (fade) or drop points whose n is below min_n. values needs an "n"
    column; without one, or with min_n <= 0, it is returned unchanged."""
    if values is None or min_n <= 0 or "n" not in values:
        return values
    small = values["n"] < min_n
    if small.any():
        logger.info(
            "{}: {} of {} points have n < {} ({})",
            what or "data",
            int(small.sum()),
            len(values),
            min_n,
            mode,
        )
    if mode == "drop":
        return values[~small]
    return values.assign(small=small)


def draw_panel(ax, sub: pd.DataFrame, attr: str, tone_of: dict, colors: dict) -> list:
    """Trajectories of one attribute's subgroups; returns legend handles.

    As in paper_fisher_cos_sim_plots_script.py: segments fade in from the first
    task, the last segment ends in an arrow, a hollow circle marks the first task.
    """
    tasks = sorted(sub["task"].unique())
    alpha = dict(zip(tasks, np.linspace(ALPHA_START, 1.0, len(tasks))))
    sgs = sorted(sub["subgroup"].unique(), key=subgroup_key)
    sty = styles(attr, sgs, tone_of, colors.get(attr, "categorical"))
    handles = []
    for sg in sgs:
        s = sub[sub["subgroup"] == sg].sort_values("task").dropna(subset=["x", "y"])
        if s.empty:
            continue
        st, x, y, t = (
            sty[sg],
            s["x"].to_numpy(),
            s["y"].to_numpy(),
            s["task"].to_numpy(),
        )
        small = (
            s["small"].to_numpy(dtype=bool)
            if "small" in s
            else np.zeros(len(s), dtype=bool)
        )
        for i in range(len(s) - 1):
            seg = dict(
                color=st["color"],
                linestyle=st["ls"],
                # a segment touching a small-n point (--min-n) is faded
                alpha=MIN_N_ALPHA if (small[i] or small[i + 1]) else alpha[t[i + 1]],
                lw=LINE_WIDTH,
            )
            if i == len(s) - 2:  # last segment: arrow to the latest checkpoint
                ax.annotate(
                    "",
                    xy=(x[i + 1], y[i + 1]),
                    xytext=(x[i], y[i]),
                    arrowprops=dict(
                        arrowstyle="-|>", mutation_scale=9, shrinkA=0, shrinkB=0, **seg
                    ),
                )
            else:
                ax.plot(x[i : i + 2], y[i : i + 2], **seg)
        ax.plot(
            x[0],
            y[0],
            "o",
            ms=4.5,
            mfc="white",
            mec=st["color"],
            mew=1.2,
            zorder=4,
            alpha=MIN_N_ALPHA if small[0] else 1.0,
        )
        handles.append(
            plt.Line2D(
                [],
                [],
                color=st["color"],
                linestyle=st["ls"],
                lw=1.4,
                label=sg.replace("Skin Tone", "ST"),
            )
        )
    return handles


def add_inner_xticks(ax, min_ticks: int = 3, inset: float = 0.12) -> None:
    """If fewer than min_ticks x ticks are visible, add ticks inside the current
    x range (it is not changed), near both ends: at `inset` of the log range
    from each edge, rounded to whole percent (0.1% for narrow ranges). Used for
    the leftmost column (Gender), whose narrow range shows only 50%. Call after
    all lines are drawn."""
    if ax.get_xscale() != "log":
        return
    lo, hi = ax.get_xlim()
    ticks = [t for t in ax.get_xticks() if lo <= t <= hi]
    if len(ticks) >= min_ticks:
        return
    step = 0.01 if hi / lo > 1.2 else 0.001  # round to 1% (0.1% if narrow)
    span = np.log(hi / lo)
    for frac in (inset, 1 - inset):
        t = round(lo * np.exp(frac * span) / step) * step
        # skip if outside or too close to an existing tick (8% of the range)
        if lo < t < hi and all(abs(np.log(t / u)) > 0.08 * span for u in ticks):
            ticks.append(t)
    ticks = sorted(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels([pct_label(v) for v in ticks])
    ax.set_xlim(lo, hi)  # keep the range exactly as it was


def format_x(ax, xvals: pd.Series, x_mode: str) -> None:
    if x_mode == "task":
        ax.set_xticks(sorted(xvals.unique()))
        return
    ax.set_xscale("log")
    lo, hi = xvals.min(), xvals.max()
    ticks = [c for c in CANDIDATE_TICKS if lo * 0.85 <= c <= hi * 1.15]
    if len(ticks) > 4:
        ticks = [ticks[i] for i in np.linspace(0, len(ticks) - 1, 4, dtype=int)]
    if not ticks:  # very narrow range: label its ends
        ticks = [lo, hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([pct_label(v) for v in ticks])
    ax.minorticks_off()


def format_y(ax, ymax: float | None) -> None:
    """Fixed 0..ymax axis; ymax None: scaled to the data (Fisher panels, as in
    paper_fisher_cos_sim_plots_script.py)."""
    if ymax is None:  # matplotlib's own limits and tick labels, as in that script
        ax.autoscale(axis="y")
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        return
    step = 0.1 if ymax <= 0.8 else 0.2
    ax.set_ylim(-0.02 * ymax, ymax * 1.02)
    ax.set_yticks(np.arange(0, ymax + 1e-9, step))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))


def legend_below(
    ax, handles, pad: float = -0.2, max_rows: int = 2, attr: str = ""
) -> None:
    """Legend under the panel, read row by row; gender centred, others stretched
    to the panel width."""
    n = len(handles)
    if attr == "Skin Tone":
        max_rows = 3
    ncol = n if n <= 3 else math.ceil(n / max_rows)  # short legends: one row
    rows = math.ceil(n / ncol)
    # matplotlib fills column by column; reorder so the legend reads row by row
    order = [
        handles[r * ncol + c]
        for c in range(ncol)
        for r in range(rows)
        if r * ncol + c < n
    ]
    if attr == "Intersection":  # long names: one column per gender (Female | Male)
        ncol, order = 2, handles
    common = dict(
        handles=order,
        ncol=ncol,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=0.6,
        borderaxespad=0,
    )
    if attr == "Predicted Gender":  # compact, centred under the panel
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, pad), **common)
    else:
        ax.legend(
            loc="upper left", bbox_to_anchor=(0, pad, 1, 0), mode="expand", **common
        )


def trajectory_figure(
    values: pd.DataFrame,
    xs: pd.DataFrame,
    x_mode: str,
    ylabel: str,
    title: str,
    path: Path,
    tone_of: dict,
    colors: dict | None = None,
    ymax: float | None = 1.0,
) -> None:
    """One row of four panels (one per attribute) for a single method."""
    df = with_x(values, xs, x_mode)
    tasks = sorted(df["task"].unique())
    attrs = [a for a in DEMOGRAPHICS if (df["demographic"] == a).any()]
    with plt.rc_context(REF_RC):
        fig, axes = plt.subplots(
            1, len(attrs), figsize=(3.0 * len(attrs), 3.2), sharey=True, squeeze=False
        )
        for ax, attr in zip(axes[0], attrs):
            sub = df[df["demographic"] == attr]
            handles = draw_panel(ax, sub, attr, tone_of, colors or TRAJ_COLORS)
            format_x(ax, sub["x"], x_mode)
            format_y(ax, ymax)
            ax.set_title(FAMILY_TITLES.get(attr, attr), fontsize=11, fontweight="bold")
            legend_below(ax, handles, attr=attr)
        add_inner_xticks(axes[0][0])  # leftmost panel: more than one tick
        axes[0][0].set_ylabel(ylabel, fontsize=10)
        for ax in axes[0]:
            ax.tick_params(axis="y", labelleft=True)
        for ax in axes[0][1:]:
            ax.tick_params(axis="y", labelleft=False)
        fig.suptitle(
            f"{title}\n{X_LABELS[x_mode]} ○ tasks {tasks[0]} → tasks {tasks[-1]}",
            fontsize=11,
            y=1.04,
        )
        fig.tight_layout()
        save(fig, path)


def grid_figure(
    rows: list[tuple[str, pd.DataFrame, float | None, str]],
    xs: list[pd.DataFrame],
    x_mode: str,
    path: Path,
    tone_of: dict,
    colors: dict,
    family_side: str = "left",
) -> None:
    """Rows = sources (e.g. routing, memory, Fisher), columns = attributes.

    Attribute names above the top row, y tick labels on every panel, the y label
    on each row's left panel, the method family (ROW_FAMILIES) vertically at the
    far left, x tick labels and one legend per column below the bottom row.

    xs: one x table per row. Rows sharing a table (memory, routing) share the
    column's x range, with tick labels under the lowest of them; a row with its
    own table (Fisher: n column of the Fisher results) gets its own x range.
    """
    data = [
        (label, with_x(v, x, x_mode), ymax) for (label, v, ymax, _), x in zip(rows, xs)
    ]
    groups: dict[int, list[int]] = {}  # rows that share an x table
    for r, x in enumerate(xs):
        groups.setdefault(id(x), []).append(r)
    families = [fam for *_, fam in rows]
    attrs = [
        a
        for a in DEMOGRAPHICS
        if any((d["demographic"] == a).any() for _, d, _ in data)
    ]
    n_r, n_c = len(data), len(attrs)
    with plt.rc_context(REF_RC):
        fig, axes = plt.subplots(
            n_r, n_c, figsize=(3.4 * n_c, 2.3 * n_r + 1.2), squeeze=False
        )
        for c, attr in enumerate(attrs):
            for members in groups.values():
                col_x = pd.concat(
                    [
                        data[r][1].loc[data[r][1]["demographic"] == attr, "x"]
                        for r in members
                    ]
                )
                if col_x.empty:
                    continue
                for r in members[:-1]:
                    axes[r][c].sharex(axes[members[-1]][c])
                    axes[r][c].tick_params(axis="x", labelbottom=False)
                format_x(axes[members[-1]][c], col_x, x_mode)
            for r, (label, d, ymax) in enumerate(data):
                ax = axes[r][c]
                handles = draw_panel(
                    ax, d[d["demographic"] == attr], attr, tone_of, colors
                )
                format_y(ax, ymax)
                if c == 0:  # method on its own line: "Routing accuracy\n(SOYO)"
                    ax.set_ylabel(label.replace(" (", "\n(", 1), fontsize=10)
                if r == 0:
                    ax.set_title(
                        FAMILY_TITLES.get(attr, attr), fontsize=9, fontweight="bold"
                    )
            legend_below(axes[-1][c], handles, pad=-0.15, attr=attr)
        for r, (_, _, ymax) in enumerate(data):  # data-scaled rows: one y range per
            if ymax is None:  # row (sharey=True in the Fisher script)
                lims = [ax.get_ylim() for ax in axes[r]]
                for ax in axes[r]:
                    ax.set_ylim(min(lo for lo, _ in lims), max(hi for _, hi in lims))
        add_inner_xticks(axes[-1][0])  # leftmost column: more than one tick
        for row in axes:  # y tick labels only on the left column
            for ax in row[1:]:
                ax.tick_params(axis="y", labelleft=False)
        fig.tight_layout(h_pad=0.6, w_pad=0.8, rect=family_rect(families, family_side))
        family_labels(fig, axes, families, family_side)
        save(fig, path)


def family_rect(families: list[str], side: str) -> tuple:
    """tight_layout area, leaving a strip for the family labels."""
    strip = 0.025 if any(families) else 0
    return (strip, 0, 1, 1) if side == "left" else (0, 0, 1 - strip, 1)


def family_labels(fig, axes, families: list[str], side: str = "left") -> None:
    """Bold vertical label at the far left, centred on its rows; consecutive
    rows of the same family share one label."""
    r = 0
    while r < len(families):
        end = r
        while end + 1 < len(families) and families[end + 1] == families[r]:
            end += 1
        if families[r]:
            top = axes[r][0].get_position().y1
            bottom = axes[end][0].get_position().y0
            left = side == "left"  # right side: text reads top to bottom
            fig.text(
                0.006 if left else 0.994,
                (top + bottom) / 2,
                families[r],
                rotation=90 if left else 270,
                ha="left" if left else "right",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
        r = end + 1


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def method_order(found: set, labels: dict) -> list:
    known = [m for k in labels for m in sorted(found) if canon(m) == canon(k)]
    return known + sorted(set(found) - set(known))


def load_memory(args) -> dict:
    """method -> {"memory": df, "memory_annotated": df} with task, y columns."""
    path = args.buffer_dir / "csv" / "representation.csv"
    if not path.exists():
        logger.warning("{} not found; no memory-share panels", path)
        return {}
    rep = pd.read_csv(path, dtype={"subgroup": str})
    rep = rep[rep["demographic"].isin(DEMOGRAPHICS)].rename(
        columns={"boundary": "task"}
    )
    return {
        m: {
            "memory": r.assign(y=r["buffer_ratio_mean"]),
            "memory_annotated": r.assign(y=r["actual_share_mean"]),
        }
        for m, r in rep.groupby("method")
    }


def load_routing(args) -> dict:
    """model -> {"routing": accuracy df, "routing_error": error df}."""
    path = args.routing_dir / "csv" / "routing_per_seed.csv"
    if not path.exists():
        logger.warning("{} not found; no routing panels", path)
        return {}
    c = with_names(pd.read_csv(path, dtype={"subgroup": str}))
    # after training through task k: pool test samples of tasks 1..k, per seed
    c = c[c["true_task"] <= c[CKPT]]
    pooled = (
        c.groupby(["model", "seed", CKPT, "demographic", "subgroup"])[["correct", "n"]]
        .sum()
        .reset_index()
    )
    pooled["acc"] = pooled["correct"] / pooled["n"]
    mean = (
        pooled.groupby(["model", CKPT, "demographic", "subgroup"])
        .agg(y=("acc", "mean"), n=("n", "mean"))  # n: pooled test samples
        .reset_index()
        .rename(columns={CKPT: "task"})
    )
    return {
        m: {"routing": d, "routing_error": d.assign(y=1 - d["y"])}
        for m, d in mean.groupby("model")
    }


def fisher_path(args) -> Path | None:
    path = args.fisher_csv or next(
        (Path(p) for p in FISHER_CSV if Path(p).exists()), None
    )
    return Path(path) if path is not None and Path(path).exists() else None


def fisher_counts(path: Path) -> pd.DataFrame:
    """Subgroup n per task from the Fisher results, as get_subgroup_n in
    paper_fisher_cos_sim_plots_script.py: per dataset, the first n of each
    subgroup (file order); codes without a subgroup name are left out.
    Columns: demographic, subgroup, task, n_data (input of subgroup_share)."""
    df = pd.read_csv(path)
    df = df[df["attr"].isin(DEMOGRAPHICS)].copy()
    df["subgroup"] = [
        SUBGROUP_NAMES[a].get(int(c)) if pd.notna(c) else None
        for a, c in zip(df["attr"], df["code"])
    ]
    task_of = {n.lower(): i + 1 for i, n in enumerate(TASK_NAMES)}
    df["task"] = df["dataset"].astype(str).str.lower().map(task_of)
    df = df.dropna(subset=["subgroup", "task"])
    n = (
        df.groupby(["dataset", "attr", "subgroup"], sort=False)
        .agg(task=("task", "first"), n_data=("n", "first"))
        .reset_index()
    )
    n["task"] = n["task"].astype(int)
    return n.rename(columns={"attr": "demographic"})[
        ["demographic", "subgroup", "task", "n_data"]
    ]


def load_fisher(args) -> pd.DataFrame | None:
    """Fisher cosine similarity at the last checkpoint per evaluated dataset
    (as in paper_fisher_cos_sim_plots_script.py), mean over seeds."""
    path = fisher_path(args)
    if path is None:
        logger.warning(
            "Fisher results not found (tried {}); no Fisher panels",
            args.fisher_csv or FISHER_CSV,
        )
        return None
    df = pd.read_csv(path)
    df = df[df["attr"].isin(DEMOGRAPHICS)].copy()
    task_of = {n.lower(): i + 1 for i, n in enumerate(TASK_NAMES)}
    df["task"] = df["dataset"].astype(str).str.lower().map(task_of)
    unknown = sorted(df.loc[df["task"].isna(), "dataset"].astype(str).unique())
    if unknown:
        logger.warning("Fisher datasets not in TASK_NAMES, ignored: {}", unknown)
    df = df.dropna(subset=["task"])
    df["subgroup"] = [
        SUBGROUP_NAMES.get(a, {}).get(int(c), f"{a} {c}")
        for a, c in zip(df["attr"], df["code"])
    ]
    out = (
        df.groupby(["attr", "subgroup", "task"])
        .agg(y=("cosine_similarity", "mean"), n=("n", "mean"))  # n: subgroup size
        .reset_index()
        .rename(columns={"attr": "demographic"})
    )
    out["task"] = out["task"].astype(int)
    if args.fisher_metric == "cosine_distance":
        out["y"] = 1 - out["y"]
    return out


def canon(name: str) -> str:
    """'Grad_Match', 'grad-match', 'GradMatch' -> 'gradmatch'."""
    return re.sub(r"[^0-9a-z]", "", str(name).lower())


def lookup(table: dict, name: str):
    """Entry of table whose key matches name, ignoring case, '_' and '-'."""
    return next((v for k, v in table.items() if canon(k) == canon(name)), {})


def label_for(labels: dict, name: str) -> str:
    return next((v for k, v in labels.items() if canon(k) == canon(name)), name)


def source(key: str, memory: dict, routing: dict, fisher, args):
    """'routing:soyo' -> (y label, data, y max, family), or None if unavailable.
    y max None: axis scaled to the data (Fisher)."""
    kind, _, method = key.partition(":")
    if kind in ("routing", "routing_error"):
        d = lookup(routing, method).get(kind)
        label = "Routing accuracy" if kind == "routing" else "Routing error"
        return (
            (
                f"{label} ({label_for(MODEL_LABELS, method)})",
                d,
                1.0,
                ROW_FAMILIES.get(kind, ""),
            )
            if d is not None
            else None
        )
    if kind in ("memory", "memory_annotated"):
        d = lookup(memory, method).get(kind)
        label = "Memory share" if kind == "memory" else "Annotated memory share"
        return (
            (
                f"{label} ({label_for(METHOD_LABELS, method)})",
                d,
                fit_ymax(d, args.memory_ymax),
                ROW_FAMILIES.get(kind, ""),
            )
            if d is not None
            else None
        )
    if kind == "fisher":
        label = (
            "Fisher cosine similarity (EWC)"
            if args.fisher_metric == "cosine_similarity"
            else "1 - Fisher cosine similarity"
        )
        return (
            (label, fisher, None, ROW_FAMILIES.get(kind, ""))  # y scaled to data
            if fisher is not None
            else None
        )
    raise SystemExit(
        f"Unknown grid row {key!r}; use routing:<model>, "
        "routing_error:<model>, memory:<method>, memory_annotated:<method> "
        "or fisher"
    )


def main() -> None:
    args = parse_args()
    args.colors = check_colors(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # x-axis: training metadata for memory and routing; for the Fisher panels the
    # n column of the Fisher results (as in paper_fisher_cos_sim_plots_script.py)
    # Fisher x mode: --fisher-x (default cumulative), unless --x task
    fisher_x = "task" if args.x == "task" else args.fisher_x
    xs = xs_fisher = pd.DataFrame(columns=["demographic", "subgroup", "task", "x"])
    if args.x != "task":
        train = load_metadata(args.metadata)["all"]
        xs = subgroup_share(train, args.x)
        path = fisher_path(args) if args.fisher_x_counts == "fisher" else None
        # xs_fisher = xs
        if path is not None:
            logger.info("Fisher x-axis: {} share, subgroup n from {}", fisher_x, path)
            xs_fisher = subgroup_share(fisher_counts(path), fisher_x)
        else:
            if args.fisher_x_counts == "fisher":
                logger.warning(
                    "Fisher results not found; Fisher x-axis from --metadata"
                )
            xs_fisher = subgroup_share(train, fisher_x)
    tone_of = tone_groups(args.metadata)
    logger.info(
        "Skin tone groups: {}",
        {g: [t for t, v in tone_of.items() if v == g] for g in TONE_GROUPS},
    )
    memory, routing, fisher = load_memory(args), load_routing(args), load_fisher(args)

    # --min-n: fade or drop small subgroups (default 0: everything is shown)
    if args.min_n > 0:
        rows_on = set(args.min_n_rows)
        logger.info(
            "Subgroups with n < {}: {} in {}",
            args.min_n,
            args.min_n_mode,
            sorted(rows_on),
        )
        if "fisher" in rows_on:
            fisher = apply_min_n(fisher, args.min_n, args.min_n_mode, "fisher")
        if "routing" in rows_on:
            routing = {
                m: {
                    k: apply_min_n(v, args.min_n, args.min_n_mode, f"{k} {m}")
                    for k, v in d.items()
                }
                for m, d in routing.items()
            }
        if "memory" in rows_on and memory:
            # n = the subgroup's annotated training images in task t
            n_train = (
                load_metadata(args.metadata)["all"]
                .groupby(["demographic", "subgroup", "task"])["n_data"]
                .sum()
                .rename("n")
                .reset_index()
            )
            memory = {
                m: {
                    k: apply_min_n(
                        v.drop(columns="n", errors="ignore").merge(
                            n_train, on=["demographic", "subgroup", "task"], how="left"
                        ),
                        args.min_n,
                        args.min_n_mode,
                        f"{k} {m}",
                    )
                    for k, v in d.items()
                }
                for m, d in memory.items()
            }

    if not args.grid_only:
        for m in method_order(set(memory), METHOD_LABELS):
            label = label_for(METHOD_LABELS, m)
            trajectory_figure(
                memory[m]["memory"],
                xs,
                args.x,
                "Share of buffer (all samples)",
                f"{label}: memory share = subgroup / total buffer size (incl. unannotated)",
                args.out_dir / f"memory_share_{m}",
                tone_of,
                args.colors,
                fit_ymax(memory[m]["memory"], args.memory_ymax),
            )
            trajectory_figure(
                memory[m]["memory_annotated"],
                xs,
                args.x,
                "Share of annotated buffer",
                f"{label}: memory share = subgroup / annotated buffer samples",
                args.out_dir / f"memory_share_annotated_{m}",
                tone_of,
                args.colors,
                fit_ymax(memory[m]["memory_annotated"], args.memory_ymax),
            )
        for m in method_order(set(routing), MODEL_LABELS):
            label = label_for(MODEL_LABELS, m)
            trajectory_figure(
                routing[m]["routing"],
                xs,
                args.x,
                "Routing accuracy",
                f"{label}: routing accuracy on test data of tasks 1..t",
                args.out_dir / f"routing_accuracy_{m}",
                tone_of,
                args.colors,
            )
            trajectory_figure(
                routing[m]["routing_error"],
                xs,
                args.x,
                "Routing error",
                f"{label}: routing error on test data of tasks 1..t",
                args.out_dir / f"routing_error_{m}",
                tone_of,
                args.colors,
            )
        if fisher is not None:
            ylabel = source("fisher", memory, routing, fisher, args)[0]
            trajectory_figure(
                fisher,
                xs_fisher,
                fisher_x,
                ylabel,
                "Fisher cosine similarity at the last checkpoint",
                args.out_dir / "fisher_similarity",
                tone_of,
                args.colors,
                ymax=None,  # scaled to the data
            )

    rows, row_xs = [], []
    for key in args.grid:
        src = source(key, memory, routing, fisher, args)
        if src is None:
            kind = key.partition(":")[0]
            have = (
                sorted(routing)
                if kind.startswith("routing")
                else sorted(memory)
                if kind.startswith("memory")
                else []
            )
            logger.warning(
                "Grid row {!r}: no data, left out{}",
                key,
                f" (available: {have})" if have else "",
            )
        else:
            rows.append(src)
            row_xs.append(xs_fisher if key.partition(":")[0] == "fisher" else xs)
    if rows:
        grid_figure(
            rows,
            row_xs,
            args.x,
            args.out_dir / args.grid_name,
            tone_of,
            args.colors,
            args.family_side,
        )
    logger.success("Figures written to {}", args.out_dir)


if __name__ == "__main__":
    main()
