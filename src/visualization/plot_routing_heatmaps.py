"""Heatmaps of task-routing error by demographic subgroup.

Reads the CSVs written by routing_analysis.py and produces (PDF + PNG).
By default only plot 1 is made; --all adds plots 2 and 3.

  1. error_by_task_{model}.pdf   (one figure per model)
       Rows: subgroups (grouped by attribute). Cols: true task.
       Value: routing error averaged over checkpoints k >= t and seeds.
       All models share one colour scale so the figures can be compared;
       --own-scale gives each figure its own range instead.

  2. error_by_checkpoint_task{t}.pdf   (one figure per true task; --all)
       Rows: subgroups. Cols: checkpoints k = t..n_tasks.
       Value: routing error on task t after training through task k.
       One panel per model, shared colour scale.

  3. destinations_{model}_{attribute}.pdf   (--all)
       One panel per subgroup (plus "All"). Rows: true task t. Cols: routed task r.
       Value: share of the subgroup's task-t samples sent to task r's prompt,
       averaged over checkpoints k >= t and seeds. The diagonal (correct
       routing) is masked so the colour scale shows where errors go.

     destinations_diff_{model}_{attribute}.pdf
       Same layout, value = subgroup share - "All" share (percentage points).
       Violet: the subgroup is sent to that wrong prompt more often than average;
       green: less often.

Style
  Cells show colour only. --annotate prints the values, formatted with
  --value-format (a Python format spec, default "{:.2f}") at --value-size pt.
  Magnitudes use a pastel single-hue ramp (--cmap violet | green); differences
  use a pastel green - warm grey - violet diverging map. Darker grey cells: no
  data (e.g. the masked diagonal).
  Cells with fewer than --min-n test samples are hatched (plots 1-2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import colors, transforms

from evaluations.routing_analysis import subgroup_key, subgroup_names
from utils.constants import (
    ATTR_LABEL_OFFSET,
    CKPT,
    COLORBAR_ASPECT,
    COLORBAR_PAD,
    DEMOGRAPHIC_LABELS,
    DEMOGRAPHICS,
    MODEL_LABELS,
    NO_DATA,
    PASTEL_DIVERGING,
    PASTEL_RAMPS,
    RC_PARAMS,
    STYLE_DEFAULTS,
)

STYLE = dict(STYLE_DEFAULTS)  # current style; flags update it
plt.rcParams.update(RC_PARAMS)


def add_style_args(p: argparse.ArgumentParser) -> None:
    """Heatmap style flags (shared with buffer_analysis.py)."""
    g = p.add_argument_group("heatmap style")
    g.add_argument(
        "--annotate",
        action="store_true",
        help="Print the value in every heatmap cell (default: colour only)",
    )
    g.add_argument(
        "--value-format",
        default=STYLE_DEFAULTS["fmt"],
        help='Format for cell values, e.g. "{:.1f}", "{:.0%}", "{:.2g}"',
    )
    g.add_argument(
        "--value-size",
        type=float,
        default=STYLE_DEFAULTS["size"],
        help="Cell value font size",
    )
    g.add_argument(
        "--cmap",
        choices=list(PASTEL_RAMPS),
        default=STYLE_DEFAULTS["cmap"],
        help="Hue of the magnitude colour ramp",
    )
    g.add_argument(
        "--hatch",
        default=STYLE_DEFAULTS["hatch"],
        help='Hatch for flagged cells, e.g. "//", "////", "\\\\", "xx", '
        '"..", or "none" to turn it off (more symbols = denser)',
    )
    g.add_argument(
        "--hatch-color", default=STYLE_DEFAULTS["hatch_color"], help="Hatch line colour"
    )
    g.add_argument(
        "--hatch-alpha",
        type=float,
        default=STYLE_DEFAULTS["hatch_alpha"],
        help="Hatch opacity, 0-1",
    )
    g.add_argument(
        "--hatch-width",
        type=float,
        default=STYLE_DEFAULTS["hatch_width"],
        help="Hatch line width in points",
    )


def configure_style(args) -> None:
    try:
        args.value_format.format(0.5)
    except (ValueError, IndexError, KeyError) as err:
        raise SystemExit(
            f"--value-format {args.value_format!r} is not a valid format "
            f"spec ({err}); use e.g. '{{:.2f}}'"
        )
    if not colors.is_color_like(args.hatch_color):
        raise SystemExit(f"--hatch-color {args.hatch_color!r} is not a colour")
    if not 0 <= args.hatch_alpha <= 1:
        raise SystemExit("--hatch-alpha must be between 0 and 1")
    STYLE.update(
        annotate=args.annotate,
        fmt=args.value_format,
        size=args.value_size,
        cmap=args.cmap,
        hatch=None if args.hatch.lower() == "none" else args.hatch,
        hatch_color=args.hatch_color,
        hatch_alpha=args.hatch_alpha,
        hatch_width=args.hatch_width,
    )
    plt.rcParams["hatch.linewidth"] = args.hatch_width


def seq_cmap(name: str | None = None) -> colors.Colormap:
    name = name or STYLE["cmap"]
    cm = colors.LinearSegmentedColormap.from_list(f"pastel_{name}", PASTEL_RAMPS[name])
    cm.set_bad(NO_DATA)
    return cm


def div_cmap() -> colors.Colormap:
    cm = colors.LinearSegmentedColormap.from_list(
        "pastel_green_violet", PASTEL_DIVERGING
    )
    cm.set_bad(NO_DATA)
    return cm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--analysis-dir", type=Path, default=Path("routing_analysis"))
    p.add_argument(
        "--out-dir", type=Path, default=None, help="Default: <analysis-dir>/figures"
    )
    p.add_argument(
        "--models", nargs="+", default=None, help="Default: all models in the CSVs"
    )
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument(
        "--all",
        action="store_true",
        help="Also make the per-checkpoint and destination heatmaps",
    )
    p.add_argument(
        "--own-scale",
        action="store_true",
        help="Colour scale per model for error_by_task (default: shared)",
    )
    add_style_args(p)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote {}", path.with_suffix(".pdf"))


def row_order(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(attribute, subgroup) rows in the fixed attribute order."""
    rows = []
    for demo in DEMOGRAPHICS:
        sgs = sorted(
            df.loc[df["demographic"] == demo, "subgroup"].unique(), key=subgroup_key
        )
        rows += [(demo, sg) for sg in sgs]
    return rows


def matrix(df, rows, cols, col_key, value) -> np.ndarray:
    idx = df.set_index(["demographic", "subgroup", col_key])[value]
    return np.array(
        [[idx.get((d, s, c), np.nan) for c in cols] for d, s in rows], dtype=float
    )


def annotate(ax, values, norm, cmap, suffix=None) -> None:
    """Cell values (only with --annotate), in dark or light ink by cell colour.

    suffix: optional boolean array; True cells get a trailing marker ("*").
    """
    if not STYLE["annotate"]:
        return
    for (i, j), v in np.ndenumerate(values):
        if not np.isfinite(v):
            continue
        r, g, b, _ = cmap(norm(v))
        dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
        label = STYLE["fmt"].format(v) + (
            "*" if suffix is not None and suffix[i, j] else ""
        )
        ax.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=STYLE["size"],
            color="white" if dark else "#2b2b2b",
        )


def add_colorbar(fig, im, axes, label: str):
    """Thin colour bar spanning the full height of the heatmap panel(s)."""
    cb = fig.colorbar(
        im, ax=axes, label=label, shrink=1.0, aspect=COLORBAR_ASPECT, pad=COLORBAR_PAD
    )
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    return cb


def hatch_cells(ax, mask) -> None:
    """Light diagonal hatching over flagged cells (e.g. too few samples)."""
    from matplotlib.patches import Rectangle

    if not STYLE["hatch"]:
        return
    for (i, j), flag in np.ndenumerate(mask):
        if flag:
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    lw=0,
                    hatch=STYLE["hatch"],
                    edgecolor=STYLE["hatch_color"],
                    alpha=STYLE["hatch_alpha"],
                )
            )


def attribute_axis(ax, rows, label_attributes: bool) -> None:
    """Subgroup tick labels, separators between attributes, attribute names."""
    ax.set_yticks(range(len(rows)))
    if label_attributes:
        ax.set_yticklabels([s for _, s in rows])
    else:
        ax.tick_params(labelleft=False)  # sharey: don't wipe the first panel
    blocks = []
    for i, (d, _) in enumerate(rows):
        if not blocks or blocks[-1][0] != d:
            blocks.append([d, i, i])
        blocks[-1][2] = i
    for d, start, end in blocks:
        if start > 0:
            ax.axhline(start - 0.5, color="black", lw=0.8)
    if not label_attributes:
        return
    # Vertical attribute names right next to the longest subgroup name
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    widest = (
        max(
            (t.get_window_extent(renderer).width for t in ax.get_yticklabels()),
            default=0,
        )
        * 72
        / fig.dpi
    )
    offset = widest + ax.yaxis.get_major_ticks()[0].get_pad() + ATTR_LABEL_OFFSET
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for d, start, end in blocks:
        label = DEMOGRAPHIC_LABELS.get(d, d)
        # names that don't fit the block height wrap onto two lines
        if " " in label and (end - start + 1) < 3:
            label = label.replace(" ", "\n", 1)
        ax.annotate(
            label,
            xy=(0, (start + end) / 2),
            xycoords=trans,
            xytext=(-offset, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            rotation_mode="anchor",
            fontsize=7,
            fontweight="bold",
            annotation_clip=False,
        )


def error_panels(
    frames, cols, col_key, xlabel, models, args, title, path, value_range=None
) -> None:
    """Shared layout for plots 1 and 2: one error heatmap per model."""
    rows = row_order(pd.concat(frames.values()))
    mats = {m: matrix(frames[m], rows, cols, col_key, "error") for m in models}
    ns = {m: matrix(frames[m], rows, cols, col_key, "n") for m in models}
    all_vals = np.concatenate([v[~np.isnan(v)] for v in mats.values()])
    lo, hi = value_range or (all_vals.min(), all_vals.max())
    norm = colors.Normalize(vmin=lo, vmax=hi)
    cmap = seq_cmap()

    width = max(1.6, 0.32 * len(cols) + 0.4)
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(width * len(models) + 1.6, 0.24 * len(rows) + 1.0),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for j, (ax, m) in enumerate(zip(axes[0], models)):
        im = ax.imshow(mats[m], aspect="auto", cmap=cmap, norm=norm)
        annotate(ax, mats[m], norm, cmap)
        hatch_cells(ax, np.nan_to_num(ns[m], nan=np.inf) < args.min_n)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([str(c) for c in cols])
        ax.set_xlabel(xlabel)
        ax.set_title(MODEL_LABELS.get(m, m))
        ax.tick_params(length=0)
        attribute_axis(ax, rows, label_attributes=(j == 0))
    add_colorbar(fig, im, axes[0].tolist(), "Routing error")
    notes = []
    all_n = np.concatenate([v.ravel() for v in ns.values()])
    if STYLE["hatch"] and (all_n < args.min_n).any():
        notes.append(f"Hatched: fewer than {args.min_n} test samples")
    # if any(np.isnan(v).any() for v in mats.values()):
    #     notes.append("Grey: no test samples of this subgroup in the task")
    if notes:
        fig.text(0.0, -0.01, ".  ".join(notes), fontsize=6, ha="left", va="top")
    if title:
        fig.suptitle(title, fontsize=8)
    save(fig, path)


# ---------------------------------------------------------------------------
# Plot 1 and 2
# ---------------------------------------------------------------------------
def plot_error_by_task(agg, models, args, out_dir) -> None:
    """One figure per model: error_by_task_{model}.pdf."""
    agg = agg.assign(error=1 - agg["accuracy_mean"])
    tasks = sorted(agg["true_task"].unique())
    shown = agg[agg["model"].isin(models)]["error"].dropna()
    shared = None if args.own_scale else (shown.min(), shown.max())
    # rows from all models, so every figure lists the same subgroups in the same order
    all_rows = agg[agg["model"].isin(models)]
    for m in models:
        frame = agg[agg["model"] == m]
        frame = pd.concat(
            [
                frame,
                all_rows[
                    ~all_rows.set_index(["demographic", "subgroup"]).index.isin(
                        frame.set_index(["demographic", "subgroup"]).index
                    )
                ]
                .assign(error=np.nan, n=np.nan)
                .drop_duplicates(["demographic", "subgroup"]),
            ]
        )
        error_panels(
            {m: frame},
            tasks,
            "true_task",
            "True task",
            [m],
            args,
            None,
            out_dir / f"error_by_task_{m}",
            value_range=shared,
        )


def plot_error_by_checkpoint(per_ckpt, models, args, out_dir) -> None:
    per_ckpt = per_ckpt.assign(error=1 - per_ckpt["accuracy_mean"])
    for t in sorted(per_ckpt["true_task"].unique()):
        sub = per_ckpt[per_ckpt["true_task"] == t]
        ckpts = sorted(sub[CKPT].unique())
        frames = {m: sub[sub["model"] == m] for m in models}
        error_panels(
            frames,
            ckpts,
            CKPT,
            "Trained through task",
            models,
            args,
            f"Routing error on test data of task {t}",
            out_dir / f"error_by_checkpoint_task{t}",
        )


# ---------------------------------------------------------------------------
# Plot 3
# ---------------------------------------------------------------------------
def destination_grid(dest, model, demo, n_tasks, diff: bool):
    sub = dest[(dest["model"] == model) & (dest["demographic"] == demo)]
    subgroups = ["All"] + sorted(
        (s for s in sub["subgroup"].unique() if s != "All"), key=subgroup_key
    )
    tasks = list(range(1, n_tasks + 1))
    mats = {}
    for sg in subgroups:
        m = (
            sub[sub["subgroup"] == sg]
            .pivot(index="true_task", columns="routed_task", values="share_mean")
            .reindex(index=tasks, columns=tasks)
            .to_numpy(dtype=float, copy=True)
        )
        np.fill_diagonal(m, np.nan)  # correct routing, shown separately
        mats[sg] = m
    if diff:
        base = mats.pop("All")
        subgroups = subgroups[1:]
        mats = {sg: 100 * (m - base) for sg, m in mats.items()}  # percentage points
    return subgroups, mats


def plot_destinations(dest, models, n_tasks, out_dir) -> None:
    ticks = list(range(1, n_tasks + 1))
    for model in models:
        for demo in DEMOGRAPHICS:
            if not (dest["demographic"] == demo).any():
                continue
            for diff in (False, True):
                subgroups, mats = destination_grid(dest, model, demo, n_tasks, diff)
                vals = np.concatenate([v[~np.isnan(v)] for v in mats.values()])
                if diff:
                    lim = np.abs(vals).max() or 1.0
                    norm = colors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
                    cm, label = div_cmap(), "Difference from all samples (pp)"
                else:
                    norm = colors.Normalize(vmin=0, vmax=vals.max())
                    cm, label = seq_cmap(), "Share of task-$t$ samples"

                ncols = min(4, len(subgroups))
                nrows = int(np.ceil(len(subgroups) / ncols))
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(1.75 * ncols + 0.8, 1.75 * nrows + 0.3),
                    squeeze=False,
                    constrained_layout=True,
                )
                for ax in axes.flat[len(subgroups) :]:
                    ax.axis("off")
                for ax, sg in zip(axes.flat, subgroups):
                    im = ax.imshow(mats[sg], cmap=cm, norm=norm)
                    annotate(ax, mats[sg], norm, cm)
                    ax.set_title(sg)
                    ax.set_xticks(range(n_tasks), ticks, fontsize=5)
                    ax.set_yticks(range(n_tasks), ticks, fontsize=5)
                    ax.set_xlabel("Routed to task", fontsize=6)
                    ax.set_ylabel("True task", fontsize=6)
                    ax.tick_params(length=0)
                add_colorbar(fig, im, axes.ravel().tolist(), label)
                what = "vs. all samples" if diff else "misrouting destinations"
                fig.suptitle(
                    f"{MODEL_LABELS.get(model, model)}: "
                    f"{DEMOGRAPHIC_LABELS.get(demo, demo)} {what}",
                    fontsize=8,
                )
                slug = demo.lower().replace(" ", "_")
                name = f"destinations_{'diff_' if diff else ''}{model}_{slug}"
                save(fig, out_dir / name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def with_names(df: pd.DataFrame) -> pd.DataFrame:
    """Subgroup codes -> names (CSVs from before names were written directly)."""
    df = df.copy()
    for demo, idx in df.groupby("demographic").groups.items():
        df.loc[idx, "subgroup"] = subgroup_names(df.loc[idx, "subgroup"], demo)
    return df


def main() -> None:
    args = parse_args()
    configure_style(args)
    csv_dir = args.analysis_dir / "csv"
    out_dir = args.out_dir or args.analysis_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = pd.read_csv(csv_dir / "routing_aggregated.csv", dtype={"subgroup": str})
    per_ckpt = pd.read_csv(
        csv_dir / "routing_per_checkpoint.csv", dtype={"subgroup": str}
    )
    dest = pd.read_csv(
        csv_dir / "routing_destinations_aggregated.csv", dtype={"subgroup": str}
    )
    agg, per_ckpt, dest = (with_names(df) for df in (agg, per_ckpt, dest))
    models = args.models or [m for m in MODEL_LABELS if m in set(agg["model"])]
    n_tasks = int(max(dest["routed_task"].max(), dest["true_task"].max()))

    plot_error_by_task(agg, models, args, out_dir)
    if args.all:
        plot_error_by_checkpoint(per_ckpt, models, args, out_dir)
        plot_destinations(dest, models, n_tasks, out_dir)
    else:
        logger.info(
            "Only error_by_task figures; pass --all for the per-checkpoint "
            "and destination heatmaps"
        )
    logger.success("Figures written to {}", out_dir)


if __name__ == "__main__":
    main()
