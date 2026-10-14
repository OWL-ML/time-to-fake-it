"""Figures for the replay-buffer analysis (DER++, GradMatch).

Reads the CSVs written by buffer_analysis.py and produces (PDF + PNG).
By default only the three heatmaps are made; --all adds the other figures.

  representation.pdf        Representation ratio per subgroup (rows) at the end of
                            each task (columns): buffer share / share expected
                            from the training data (1 = proportional). Diverging
                            map centred on 1; a dot marks cells outside the range
                            expected by chance. One panel per method.
  exposure.pdf              Exposure: subgroup's replay draws / all replay draws
                            during each task, from task 2 on (no replay while
                            task 1 is trained; --exposure-from-task to change).
  eviction_rate.pdf         Fraction of the subgroup's buffer samples evicted
                            during each task.

  With --all:
  retention_vs_replay.pdf   Share of buffer time vs share of replay draws.
  lifetime_km_{attribute}.pdf
                            Kaplan-Meier survival of buffer samples per subgroup.
  cohort_survival_diff_{method}_{attribute}.pdf
                            Cohort survival per subgroup minus all samples (pp);
                            violet = survives longer than average, green = shorter.

Style flags (--annotate, --value-format, --value-size, --cmap, --hatch...) are the
same as in plot_routing_heatmaps.py; defaults are in constants.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import colors
from plot_routing_heatmaps import (
    STYLE,
    add_colorbar,
    add_style_args,
    annotate,
    attribute_axis,
    configure_style,
    div_cmap,
    save,
    seq_cmap,
)

from evaluations.routing_analysis import subgroup_key
from utils.constants import DEMOGRAPHIC_LABELS, DEMOGRAPHICS, METHOD_LABELS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("buffer_analysis"),
        help="Output folder of buffer_analysis.py (reads <dir>/csv)",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None, help="Default: <analysis-dir>/figures"
    )
    p.add_argument(
        "--methods", nargs="+", default=None, help="Default: all methods in the CSVs"
    )
    p.add_argument(
        "--exposure-from-task",
        type=int,
        default=2,
        help="First task shown in the exposure heatmap (default 2: the "
        "buffer is empty while task 1 is trained, so nothing is "
        "replayed); 1 shows all tasks",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Also make the scatter, lifetime and cohort figures",
    )
    add_style_args(p)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def row_order(df):
    rows = []
    for attr in DEMOGRAPHICS:
        sgs = df.loc[df["demographic"] == attr, "subgroup"].dropna().unique()
        rows += [
            (attr, s) for s in sorted((s for s in sgs if s != "All"), key=subgroup_key)
        ]
    return rows


def attr_axis(ax, rows, first: bool):
    """Subgroup names, attribute separators and vertical attribute labels (shared)."""
    attribute_axis(ax, rows, first)


def heat_panels(
    tables,
    methods,
    col_key,
    value,
    cbar_label,
    center,
    path,
    mark_col=None,
    xlabel="Task",
    title=None,
    first_col=None,
):
    """One heatmap per method: rows = subgroups, cols = tasks."""
    rows = row_order(pd.concat(tables.values()))
    cols = sorted(pd.concat(tables.values())[col_key].unique())
    if first_col is not None:  # e.g. exposure: no replay during task 1
        cols = [c for c in cols if c >= first_col]

    def mat(df, col):
        s = df.set_index(["demographic", "subgroup", col_key])[col]
        return np.array(
            [[s.get((a, g, c), np.nan) for c in cols] for a, g in rows], dtype=float
        )

    mats = {m: mat(tables[m], value) for m in methods}
    vals = np.concatenate([v[np.isfinite(v)] for v in mats.values()])
    if center is None:
        norm, cm = colors.Normalize(vals.min(), vals.max()), seq_cmap()
    else:  # symmetric around the centre, so equal colour = equal distance
        span = max(abs(vals.max() - center), abs(center - vals.min()), 1e-6)
        norm = colors.TwoSlopeNorm(center, center - span, center + span)
        cm = div_cmap()
    fig, axes = plt.subplots(
        1,
        len(methods),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
        figsize=((0.34 * len(cols) + 0.4) * len(methods) + 1.6, 0.24 * len(rows) + 1.0),
    )
    for j, (ax, m) in enumerate(zip(axes[0], methods)):
        v = mats[m]
        im = ax.imshow(v, aspect="auto", cmap=cm, norm=norm)
        marks = (
            mat(tables[m], mark_col).astype(bool)
            if mark_col
            else np.zeros_like(v, bool)
        )
        if STYLE["annotate"]:
            annotate(ax, v, norm, cm, suffix=marks)
        else:  # colour only: flag marked cells with a small dot in the corner
            ii, kk = np.nonzero(marks & np.isfinite(v))
            ax.plot(kk + 0.3, ii - 0.28, "o", ms=1.8, color="#3d3a36", lw=0)
        ax.set_xticks(range(len(cols)), [str(int(c)) for c in cols])
        ax.set_xlabel(xlabel)
        ax.set_title(METHOD_LABELS.get(m, m))
        ax.tick_params(length=0)
        attr_axis(ax, rows, j == 0)
    add_colorbar(fig, im, axes[0].tolist(), cbar_label)
    notes = []
    if mark_col and any(mat(tables[m], mark_col).astype(bool).any() for m in methods):
        notes.append(
            f"{'*' if STYLE['annotate'] else 'Dot'}: outside the range "
            "expected by chance"
        )
    if any(np.isnan(v).any() for v in mats.values()):
        notes.append("Grey: no data (e.g. empty buffer or no replay in that task)")
    if notes:
        fig.text(0.0, -0.01, ".  ".join(notes), fontsize=6, ha="left", va="top")
    if title:
        fig.suptitle(title, fontsize=8)
    save(fig, path)


def plot_retention_vs_replay(rvr, methods, path):
    fig, axes = plt.subplots(
        1,
        len(methods),
        squeeze=False,
        constrained_layout=True,
        figsize=(3.0 * len(methods) + 0.6, 3.0),
        sharex=True,
        sharey=True,
    )
    palette = plt.get_cmap("tab10")
    for ax, m in zip(axes[0], methods):
        df = rvr[rvr["method"] == m]
        for i, attr in enumerate(DEMOGRAPHICS):
            a = df[df["demographic"] == attr]
            ax.errorbar(
                a["retention_share_mean"],
                a["replay_share_mean"],
                xerr=a["retention_share_std"],
                yerr=a["replay_share_std"],
                fmt="o",
                ms=3,
                lw=0.6,
                color=palette(i),
                label=DEMOGRAPHIC_LABELS.get(attr, attr),
            )
            for _, r in a.iterrows():
                ax.annotate(
                    r["subgroup"],
                    (r["retention_share_mean"], r["replay_share_mean"]),
                    fontsize=4,
                    xytext=(2, 2),
                    textcoords="offset points",
                )
        lim = [
            df[["retention_share_mean", "replay_share_mean"]].min().min() * 0.8,
            df[["retention_share_mean", "replay_share_mean"]].max().max() * 1.2,
        ]
        ax.plot(lim, lim, color="0.5", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Share of buffer time")
        ax.set_ylabel("Share of replay draws")
        ax.set_title(METHOD_LABELS.get(m, m))
    axes[0][-1].legend(fontsize=5, loc="lower right")
    save(fig, path)


def plot_km(km, methods, out_dir):
    for attr in DEMOGRAPHICS:
        fig, axes = plt.subplots(
            1,
            len(methods),
            squeeze=False,
            sharey=True,
            constrained_layout=True,
            figsize=(2.8 * len(methods) + 0.8, 2.4),
        )
        for ax, m in zip(axes[0], methods):
            df = km[(km["method"] == m) & (km["demographic"] == attr)]
            groups = sorted(
                (g for g in df["subgroup"].unique() if g != "All"), key=subgroup_key
            )
            cmap = plt.get_cmap("viridis", max(len(groups), 2))
            for i, g in enumerate(["All"] + groups):
                s = df[df["subgroup"] == g].sort_values("boundaries")
                kw = (
                    dict(color="black", lw=1.2, ls="--")
                    if g == "All"
                    else dict(color=cmap(i - 1), lw=0.9)
                )
                ax.step(
                    s["boundaries"], s["survival_mean"], where="post", label=g, **kw
                )
                if g != "All":
                    ax.fill_between(
                        s["boundaries"],
                        s["survival_mean"] - s["survival_std"].fillna(0),
                        s["survival_mean"] + s["survival_std"].fillna(0),
                        step="post",
                        color=cmap(i - 1),
                        alpha=0.12,
                        lw=0,
                    )
            ax.set_title(METHOD_LABELS.get(m, m))
            ax.set_xlabel("Task boundaries survived")
            ax.set_ylim(0, 1.02)
        axes[0][0].set_ylabel("Fraction still in buffer")
        axes[0][-1].legend(
            fontsize=5,
            ncol=2,
            title=DEMOGRAPHIC_LABELS.get(attr, attr),
            title_fontsize=6,
        )
        save(fig, out_dir / f"lifetime_km_{attr.lower().replace(' ', '_')}")


def plot_cohorts(coh, methods, n_tasks, out_dir):
    ticks = list(range(1, n_tasks + 1))
    for m in methods:
        for attr in DEMOGRAPHICS:
            df = coh[(coh["method"] == m) & (coh["demographic"] == attr)]
            if df.empty:
                continue
            groups = sorted(
                (g for g in df["subgroup"].unique() if g != "All"), key=subgroup_key
            )

            def grid(g):
                s = df[df["subgroup"] == g].pivot(
                    index="cohort_task", columns="boundary", values="survival_mean"
                )
                return s.reindex(index=ticks, columns=ticks).to_numpy(float, copy=True)

            base = grid("All")
            mats = {g: 100 * (grid(g) - base) for g in groups}
            vals = np.concatenate([v[np.isfinite(v)] for v in mats.values()])
            lim = np.abs(vals).max() if vals.size else 1.0
            norm = colors.TwoSlopeNorm(0, -lim - 1e-9, lim + 1e-9)
            cm = div_cmap()  # violet = survives longer than average, green = shorter
            ncols = min(4, len(groups))
            nrows = int(np.ceil(len(groups) / ncols))
            fig, axes = plt.subplots(
                nrows,
                ncols,
                squeeze=False,
                constrained_layout=True,
                figsize=(1.75 * ncols + 0.8, 1.75 * nrows + 0.3),
            )
            for ax in axes.flat[len(groups) :]:
                ax.axis("off")
            for ax, g in zip(axes.flat, groups):
                im = ax.imshow(mats[g], cmap=cm, norm=norm)
                annotate(ax, mats[g], norm, cm)
                ax.set_title(g)
                ax.set_xticks(range(n_tasks), ticks, fontsize=5)
                ax.set_yticks(range(n_tasks), ticks, fontsize=5)
                ax.set_xlabel("End of task", fontsize=6)
                ax.set_ylabel("Inserted in task", fontsize=6)
                ax.tick_params(length=0)
            add_colorbar(
                fig, im, axes.ravel().tolist(), "Survival vs. all samples (pp)"
            )
            fig.suptitle(
                f"{METHOD_LABELS.get(m, m)}: "
                f"{DEMOGRAPHIC_LABELS.get(attr, attr)} cohort survival",
                fontsize=8,
            )
            save(
                fig,
                out_dir / f"cohort_survival_diff_{m}_{attr.lower().replace(' ', '_')}",
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def read(csv_dir: Path, name: str) -> pd.DataFrame:
    path = csv_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run buffer_analysis.py first")
    return pd.read_csv(path, dtype={"subgroup": str})


def main() -> None:
    args = parse_args()
    configure_style(args)
    csv_dir = args.analysis_dir / "csv"
    fig_dir = args.out_dir or args.analysis_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rep_m = read(csv_dir, "representation")
    task_m = read(csv_dir, "replay_and_eviction")
    found = set(rep_m["method"])
    if args.methods:
        methods = [m.lower() for m in args.methods if m.lower() in found]
    else:
        methods = [m for m in METHOD_LABELS if m in found] + sorted(
            found - set(METHOD_LABELS)
        )
    if not methods:
        raise SystemExit(f"No such methods in {csv_dir}; found {sorted(found)}")
    by_m = lambda df: {m: df[df["method"] == m] for m in methods}  # noqa: E731

    heat_panels(
        by_m(rep_m.rename(columns={"boundary": "task"})),
        methods,
        "task",
        "ratio_mean",
        "Representation ratio (1 = proportional)",
        1.0,
        fig_dir / "representation",
        mark_col="outside_chance",
        xlabel="End of task",
    )
    heat_panels(
        by_m(task_m),
        methods,
        "task",
        "exposure_mean",
        "Exposure (share of all replay draws)",
        None,
        fig_dir / "exposure",
        first_col=args.exposure_from_task,
    )
    heat_panels(
        by_m(task_m),
        methods,
        "task",
        "eviction_rate_mean",
        "Eviction rate",
        None,
        fig_dir / "eviction_rate",
    )

    if args.all:
        coh = read(csv_dir, "cohort_survival")
        n_tasks = int(max(coh["cohort_task"].max(), coh["boundary"].max()))
        plot_retention_vs_replay(
            read(csv_dir, "retention_vs_replay"),
            methods,
            fig_dir / "retention_vs_replay",
        )
        plot_km(read(csv_dir, "lifetime_km"), methods, fig_dir)
        plot_cohorts(coh, methods, n_tasks, fig_dir)
    else:
        logger.info(
            "Only the three heatmaps; pass --all for the scatter, lifetime "
            "and cohort figures"
        )
    logger.success("Figures written to {}", fig_dir)


if __name__ == "__main__":
    main()
