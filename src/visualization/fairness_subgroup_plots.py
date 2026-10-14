"""
Subgroup-level 3D bars: fairness after the final task, and its change since each dataset
was learned.

For method `alg`, metric m, subgroup s, evaluated dataset d and each seed:

    height(d, s) = m(eval d, after final_task)
    delta(d, s)  = m(eval d, after final_task) - m(eval d, after task d)

delta is computed WITHIN each seed and only then averaged, because both terms come from
the same run. The bars show mean values only; the loader still returns the standard
deviations for anything else that needs them.

Input: fairness_subgroup.csv written by fairness_from_predictions.py.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairness_diff_table import GROUP_LABELS
from loguru import logger
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

GROUP_ORDER = ["Gender", "Age", "Skin Tone", "Intersection"]

# Verify against the annotation pipeline: Skin Tone codes assumed 1..10,
# Intersection = gender * 3 + skin tone group with 0 = Female and 0 = Light.
SUBGROUP_LABELS = {
    ("Gender", 0): "Female",
    ("Gender", 1): "Male",
    ("Age", 0): "Child",
    ("Age", 1): "Youth",
    ("Age", 2): "Adult",
    ("Age", 3): "Middle-Aged",
    ("Age", 4): "Senior",
    **{("Skin Tone", k): f"Skin Tone {k}" for k in range(1, 11)},
    ("Intersection", 0): "Female, Light",
    ("Intersection", 1): "Female, Medium",
    ("Intersection", 2): "Female, Dark",
    ("Intersection", 3): "Male, Light",
    ("Intersection", 4): "Male, Medium",
    ("Intersection", 5): "Male, Dark",
}

MISSING_COLOR = (0.8, 0.8, 0.8, 1.0)


# --------------------------------------------------------------------------- data


def _as_code(value):
    """Subgroup codes as ints where possible (CSV may hold 0, 0.0 or '0')."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return int(f) if f.is_integer() else f


def _row_sort_key(group_subgroup):
    group, subgroup = group_subgroup
    numeric = isinstance(subgroup, (int, float))
    return (
        GROUP_ORDER.index(group),
        not numeric,
        subgroup if numeric else 0,
        str(subgroup),
    )


def load_subgroup_forgetting(
    fairness_subgroup_csv: str,
    alg: str,
    metric: str,
    final_task: int = 11,
    reference_method: str | None = "naive",
) -> dict[str, pd.DataFrame]:
    """
    Returns a dict of DataFrames, each indexed by (Group, Subgroup) with one column per
    evaluated dataset: height_mean, height_sd, delta_mean, delta_sd, n (seeds behind delta).
    Missing subgroups stay NaN (never 0).

    reference_method: whose diagonal (task == eval_task) delta is measured against,
    matched on seed — "naive" by default, as in the original reference_df. Pass None
    (or `alg`) to use the method's own diagonal. Datasets the reference lacks fall back
    to `alg`'s own diagonal.
    """
    fs_all = pd.read_csv(fairness_subgroup_csv)
    if metric not in fs_all.columns:
        raise ValueError(f"Metric '{metric}' not in {fairness_subgroup_csv}")
    fs_all = fs_all[fs_all["Group"].isin(GROUP_LABELS)].copy()
    fs_all["Group"] = fs_all["Group"].map(GROUP_LABELS)
    fs_all["Subgroup"] = fs_all["Subgroup"].map(_as_code)

    fs = fs_all[fs_all["method"] == alg]
    if fs.empty:
        raise ValueError(f"No rows for method '{alg}' in {fairness_subgroup_csv}")

    keys = ["seed", "eval_task", "Group", "Subgroup"]
    final = fs.loc[fs["task"] == final_task, keys + [metric]].rename(
        columns={metric: "height"}
    )
    if final.empty:
        raise ValueError(f"No rows for {alg} after task {final_task}")

    own_diag = fs.loc[fs["task"] == fs["eval_task"], keys + [metric]].rename(
        columns={metric: "diag"}
    )
    if reference_method is None or reference_method == alg:
        diag = own_diag
    else:
        ref = fs_all[
            (fs_all["method"] == reference_method)
            & (fs_all["task"] == fs_all["eval_task"])
        ]
        if ref.empty:
            raise ValueError(
                f"No diagonal rows for reference method '{reference_method}' in "
                f"{fairness_subgroup_csv}"
            )
        diag = ref[keys + [metric]].rename(columns={metric: "diag"})
        logger.info(
            f"{alg}/{metric}: delta measured against '{reference_method}' diagonals"
        )

        # datasets the reference lacks: use the method's own diagonal
        have = set(map(tuple, diag[["seed", "eval_task"]].drop_duplicates().to_numpy()))
        wanted = set(
            map(tuple, final[["seed", "eval_task"]].drop_duplicates().to_numpy())
        )
        gaps = wanted - have
        if gaps:
            fb = own_diag[
                pd.MultiIndex.from_frame(own_diag[["seed", "eval_task"]]).isin(
                    sorted(gaps)
                )
            ]
            if fb.empty:
                logger.warning(
                    f"{alg}/{metric}: no diagonal for (seed, dataset) {sorted(gaps)} in "
                    f"'{reference_method}' or in '{alg}'; their delta stays missing"
                )
            else:
                logger.info(
                    f"{alg}/{metric}: diagonal for (seed, dataset) "
                    f"{sorted(map(tuple, fb[['seed', 'eval_task']].drop_duplicates().to_numpy()))} "
                    f"taken from '{alg}' itself"
                )
                diag = pd.concat([diag, fb], ignore_index=True)

    d = final.merge(diag, on=keys, how="left", validate="one_to_one")
    n_missing = int(d["diag"].isna().sum())
    if n_missing:
        logger.warning(
            f"{alg}/{metric}: {n_missing} final values have no diagonal counterpart "
            "(subgroup skipped when the dataset was learned); their delta is missing"
        )
    d["delta"] = d["height"] - d["diag"]

    agg = d.groupby(["Group", "Subgroup", "eval_task"]).agg(
        height_mean=("height", "mean"),
        height_sd=("height", "std"),
        delta_mean=("delta", "mean"),
        delta_sd=("delta", "std"),
        n=("delta", "count"),
    )

    rows = sorted(agg.index.droplevel("eval_task").unique(), key=_row_sort_key)
    row_index = pd.MultiIndex.from_tuples(rows, names=["Group", "Subgroup"])
    columns = list(range(1, final_task + 1))
    out = {
        name: agg[name].unstack("eval_task").reindex(index=row_index, columns=columns)
        for name in agg.columns
    }

    n_seeds = fs["seed"].nunique()
    logger.info(
        f"{alg}/{metric}: {len(rows)} subgroups, {len(columns)} datasets, {n_seeds} seed(s)"
    )
    if n_seeds < 2:
        logger.warning(f"{alg}: only one seed, so the std columns stay empty")
    return out


def subgroup_labels(index: pd.MultiIndex) -> list[str]:
    labels, unmapped = [], []
    for group, subgroup in index:
        label = SUBGROUP_LABELS.get((group, subgroup))
        if label is None:
            unmapped.append((group, subgroup))
            label = f"{group} {subgroup}"
        labels.append(label)
    if unmapped:
        logger.warning(f"No label for subgroups {unmapped}; check SUBGROUP_LABELS")
    return labels


# --------------------------------------------------------------------------- colour


def nice_cap_floor(max_abs: float) -> float:
    """Floor max_abs to the nearest value of the form m × 10^n, m in 1..9."""
    if max_abs <= 0:
        return 0.0
    exponent = math.floor(math.log10(max_abs))
    base = 10**exponent
    for m in [9, 8, 7, 6, 5, 4, 3, 2, 1]:
        if m * base <= max_abs:
            return m * base
    return base


def _tick_label(t: float) -> str:
    return "0" if abs(t) < 1e-12 else f"{t:.2g}"


def delta_norm(delta_mean: pd.DataFrame) -> mcolors.Normalize:
    """Symmetric linear norm shared by the bars and the colour bar."""
    values = np.abs(delta_mean.to_numpy(dtype=float))
    vmax = float(np.nanmax(values)) if np.isfinite(values).any() else 0.0
    vmax = vmax if vmax > 0 else 1e-9
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _add_colorbar(fig, cax, norm, cmap, labelsize):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cap = nice_cap_floor(norm.vmax)
    ticks = np.linspace(-cap, cap, 5) if norm.vmin < 0 else np.linspace(0, cap, 5)
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([_tick_label(t) for t in ticks])
    cbar.ax.tick_params(labelsize=labelsize)
    return cbar


# --------------------------------------------------------------------------- 3D bars


def metrics_3d_bars_heatmap(
    data: dict[str, pd.DataFrame],
    metric: str,
    alg: str,
    generator: int = 11,
    save_dir: str = "plots",
    show: bool = False,
):
    """
    3D bars: height = mean metric after `generator`, colour = mean delta.
    Subgroups missing in a dataset are not drawn.
    Saved to <save_dir>/3DColorBar/<alg>/.
    """
    height = data["height_mean"]
    delta = data["delta_mean"]

    cmap = plt.colormaps["RdBu_r"]
    norm = delta_norm(delta)

    # X positions with group separation
    x_pos, current_x, gap, prev_group = [], 0.0, 0.8, None
    for group, _ in height.index:
        if prev_group is not None and group != prev_group:
            current_x += gap
        x_pos.append(current_x)
        current_x += 1
        prev_group = group
    x_pos = np.array(x_pos)

    fig = plt.figure(figsize=(18, 15))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=20, azim=-65)
    ax.set_box_aspect([5, 4, 1.2])
    ax.zaxis.set_pane_color((0.85, 0.85, 0.85, 1.0))
    ax.xaxis.set_pane_color((0.9, 0.9, 0.95, 1.0))
    ax.yaxis.set_pane_color((0.88, 0.88, 0.92, 1.0))
    fig.subplots_adjust(left=0.15, right=0.8, bottom=0.06, top=0.9)

    zmax = np.nanmax(height.to_numpy(dtype=float))
    if not np.isfinite(zmax):
        raise ValueError("No bar heights to plot")
    ax.set_zlim(0, max(zmax, 1e-6))

    datasets = list(height.columns)
    width, y_gap = 0.5, 4
    any_missing_delta = False

    # Draw from back to front
    for y_idx, d in reversed(list(enumerate(datasets))):
        heights = height[d].to_numpy(dtype=float)
        changes = delta[d].to_numpy(dtype=float)
        drawable = ~np.isnan(heights)
        if not drawable.any():
            continue

        colors = cmap(norm(np.nan_to_num(changes)))
        missing_delta = np.isnan(changes)
        colors[missing_delta] = MISSING_COLOR
        any_missing_delta |= bool((missing_delta & drawable).any())

        ax.bar(
            x_pos[drawable],
            heights[drawable],
            zs=y_idx * y_gap,
            zdir="y",
            color=colors[drawable],
            edgecolor="#777777",
            linewidth=0.05,
            width=width,
            alpha=1,
        )

    # Labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        subgroup_labels(height.index), rotation=45, ha="right", fontsize=18
    )
    ax.set_yticks(np.arange(len(datasets)) * y_gap)
    ax.set_yticklabels([f"{int(d)}" for d in datasets], fontsize=18)
    ax.set_ylabel("Dataset", labelpad=10, fontsize=16, rotation=45, fontweight="bold")
    ax.set_zlabel(metric, labelpad=10, fontsize=16, rotation=90, fontweight="bold")
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda t, pos: _tick_label(t)))
    ax.tick_params(axis="z", pad=10, labelsize=18)

    # Colour bar (same norm as the bars) and legend for the markings
    cax = fig.add_axes([0.15, 0.165, 0.35, 0.02])
    _add_colorbar(fig, cax, norm, cmap, labelsize=18)

    handles = []
    if any_missing_delta:
        handles.append(
            Patch(
                facecolor=MISSING_COLOR,
                edgecolor="#777777",
                label=r"$\Delta$ unavailable",
            )
        )
    if handles:
        fig.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.52, 0.145),
            fontsize=16,
            frameon=False,
        )

    out_dir = Path(save_dir) / "3DColorBar" / alg
    out_dir.mkdir(parents=True, exist_ok=True)
    file_base = f"{out_dir}/{alg}_gen_{generator}_{metric}_colorbar"
    fig.savefig(f"{file_base}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.2)
    logger.info(f"Saved {file_base}.pdf")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- CLI


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--fairness-subgroup-csv", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--metrics", nargs="+", default=["FEO", "FDP", "FOAE", "FEOP"])
    p.add_argument("--final-task", type=int, default=11)
    p.add_argument(
        "--reference-method",
        default="naive",
        help="whose diagonal delta is measured against; 'self' uses each method's own",
    )
    p.add_argument("--save-dir", default="plots")
    args = p.parse_args()

    for method in args.methods:
        for metric in args.metrics:
            reference = (
                None if args.reference_method == "self" else args.reference_method
            )
            data = load_subgroup_forgetting(
                args.fairness_subgroup_csv, method, metric, args.final_task, reference
            )
            metrics_3d_bars_heatmap(
                data, metric, method, args.final_task, args.save_dir
            )
