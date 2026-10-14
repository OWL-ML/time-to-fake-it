"""
Merge the per-run fairness CSVs and build every figure.

    1) python fairness_from_predictions.py --root predictions            # per method/seed
    2) python make_figures.py                                            # merge + plot

Reads <fairness-dir>/<method>/seed_run_<seed>/*.csv, writes the merged tables to
<merged-dir> and the figures to <fig-dir>. A flat <fairness-dir> (from --flat) is used
as-is. Keep fairness_diff_table.py, fairness_algs_table.py and fairness_subgroup_plots.py
in the same folder as this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from fairness_algs_table import (
    load_algs_table,
    visualize_fairness_table_algs_matplotlib,
)
from fairness_diff_table import (
    load_fairness_diff,
    visualize_fairness_diff_by_generator_matplotlib,
)
from fairness_subgroup_plots import load_subgroup_forgetting, metrics_3d_bars_heatmap
from loguru import logger

# Labels shown in the figures. A method that isn't listed keeps its own name.
DISPLAY_NAMES = {
    "naive": "Naive",
    "accumulated": "Acm.",
    "ewc": "EWC",
    "derpp": "DER++",
    "grad_match": "GMatch",
    "soyo": "SOYO",
    "pina": "PINA",
    "sprompts": "SPrompts",
}
TABLES = ["subgroup_stats", "fairness_group", "fairness_subgroup"]
KEYS = ["method", "seed", "task", "eval_task", "Group", "Subgroup"]


def merge_runs(
    fairness_dir: Path,
    merged_dir: Path,
    exclude_methods: list[str] | None = None,
    exclude_seeds: list[int] | None = None,
) -> Path:
    """Concatenate <method>/seed_run_<seed>/*.csv into <merged-dir>, dropping excluded
    methods and seeds. Flat input is filtered the same way (or passed through untouched)."""
    flat = (fairness_dir / "fairness_group.csv").exists()
    if flat and not (exclude_methods or exclude_seeds):
        logger.info(f"{fairness_dir} is already flat; using it as-is")
        return fairness_dir

    merged_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        parts = (
            [fairness_dir / f"{name}.csv"]
            if flat
            else sorted(fairness_dir.glob(f"*/seed_run_*/{name}.csv"))
        )
        if not parts:
            raise FileNotFoundError(f"No {name}.csv under {fairness_dir}/*/seed_run_*/")
        merged = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

        for field, dropped in (("method", exclude_methods), ("seed", exclude_seeds)):
            if not dropped:
                continue
            keep = ~merged[field].isin(dropped)
            logger.info(
                f"{name}: dropping {int((~keep).sum())} rows for {field}(s) {dropped}"
            )
            merged = merged[keep]
        if merged.empty:
            raise ValueError(f"{name}: no rows left after excluding methods/seeds")

        keys = [k for k in KEYS if k in merged.columns]
        dupes = merged.duplicated(keys, keep=False)
        if dupes.any():  # same method+seed written by two runs
            raise ValueError(
                f"{name}: {int(dupes.sum())} duplicate rows across runs, e.g.\n"
                f"{merged.loc[dupes, keys].head()}"
            )

        out_path = merged_dir / f"{name}.csv"
        merged.sort_values(keys).to_csv(out_path, index=False)
        logger.info(f"Merged {len(parts)} file(s) -> {out_path} ({len(merged)} rows)")
    return merged_dir


def display_map(methods: list[str]) -> dict[str, str]:
    """{method: label}, falling back to the method's own name."""
    mapping = {m: DISPLAY_NAMES.get(m, m) for m in methods}
    labels = list(mapping.values())
    clashes = sorted({label for label in labels if labels.count(label) > 1})
    if clashes:
        raise ValueError(f"Display names must be unique; {clashes} used more than once")
    unmapped = [m for m in methods if m not in DISPLAY_NAMES]
    if unmapped:
        logger.info(f"No display name for {unmapped}; using the method name")
    return mapping


def discover_methods(merged_dir: Path) -> list[str]:
    return sorted(pd.read_csv(merged_dir / "fairness_group.csv")["method"].unique())


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--fairness-dir", type=Path, default=Path("fairness_metrics"))
    p.add_argument("--merged-dir", type=Path, default=Path("fairness_merged"))
    p.add_argument("--fig-dir", type=Path, default=Path("figures"))
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="default: every method found; the first one is the algs table's "
        "colour reference and supplies its Init. F column",
    )
    p.add_argument(
        "--exclude-methods",
        nargs="+",
        default=None,
        help="leave these methods out of the merged tables and the figures",
    )
    p.add_argument(
        "--exclude-seeds",
        nargs="+",
        type=int,
        default=None,
        help="leave these seeds out, e.g. a run you no longer trust",
    )
    p.add_argument("--metrics", nargs="+", default=["FEO", "FDP", "FOAE", "FEOP"])
    p.add_argument("--final-task", type=int, default=11)
    p.add_argument(
        "--reference-method",
        default="naive",
        help="3D bars: whose diagonal delta is measured against; "
        "'self' uses each method's own",
    )
    p.add_argument(
        "--figures",
        nargs="+",
        default=["all"],
        choices=["all", "diff", "algs", "bars"],
        help="diff = forgetting tables, algs = algorithm tables, "
        "bars = 3D subgroup bars, all = every type (default)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["1"],
        help="evaluated datasets for the algs tables: numbers, or 'all' "
        "for 1..final-task (default: 1)",
    )
    args = p.parse_args()

    figures = {"diff", "algs", "bars"} if "all" in args.figures else set(args.figures)
    logger.info(f"Building: {sorted(figures)}")

    data_dir = merge_runs(
        args.fairness_dir, args.merged_dir, args.exclude_methods, args.exclude_seeds
    )
    group_csv = data_dir / "fairness_group.csv"
    subgroup_csv = data_dir / "fairness_subgroup.csv"
    methods = args.methods or discover_methods(data_dir)
    labels = display_map(methods)
    logger.info(f"Methods: {methods} shown as {[labels[m] for m in methods]}")
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    SHARED = dict(
        std_fontsize=11,
        metric_fontsize=14,
        header_fontsize=14,
        row_pad_in=0.16,
        line_gap=0.42,
        cell_pad_in=0.05,
        cell_aspect=0,
    )

    # 1) Forgetting table per method: fairness after the final task minus right after learning
    if "diff" in figures:
        for method in methods:
            mean, std, _ = load_fairness_diff(
                group_csv, method, final_task=args.final_task
            )
            visualize_fairness_diff_by_generator_matplotlib(
                mean,
                std,
                method,
                save_path=str(args.fig_dir / f"table_diff_{method}"),
                mean_fontsize=16,
                col_width_in=1.034,
                **SHARED,
            )

    # 2) Algorithm comparison for dataset 1 by default (--datasets all for every dataset),
    #    after the final task. A different gen needs fairness_from_predictions.py --all-files.
    if "algs" in figures:
        datasets = (
            range(1, args.final_task + 1)
            if any(str(d).lower() == "all" for d in args.datasets)
            else [int(d) for d in args.datasets]
        )
        for ds in datasets:
            mean, std, _ = load_algs_table(
                group_csv, ds=ds, gen=args.final_task, names=methods
            )
            mean, std = mean.rename(columns=labels), std.rename(columns=labels)
            visualize_fairness_table_algs_matplotlib(
                mean,
                std,
                save_path=str(args.fig_dir / f"table_algs_ds{ds}"),
                fontsize=16,
                col_width_in=1.149,
                cbar_label="",
                **SHARED,
            )

    # 3) 3D subgroup bars per method and metric -> <fig-dir>/3DColorBar/<method>/
    if "bars" in figures:
        for method in methods:
            for metric in args.metrics:
                data = load_subgroup_forgetting(
                    subgroup_csv,
                    method,
                    metric,
                    final_task=args.final_task,
                    reference_method=(
                        None
                        if args.reference_method == "self"
                        else args.reference_method
                    ),
                )
                metrics_3d_bars_heatmap(
                    data,
                    metric,
                    method,
                    generator=args.final_task,
                    save_dir=str(args.fig_dir),
                )

    logger.info(f"Figures written to {args.fig_dir}")


if __name__ == "__main__":
    main()
