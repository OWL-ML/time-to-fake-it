"""Demographic analysis of task routing in prompt-based continual learning.

For every model, true task t, checkpoint k >= t (model trained through task k),
seed, demographic attribute and subgroup, this computes the routing accuracy:
the fraction of task-t test samples routed to task t. Routing error is exactly
1 - accuracy, so it is not reported separately.

Checkpoints k < t are skipped: task t has not been learned yet, so routing to
it is impossible and accuracy would be 0 by construction.

Task numbering: file names are 1-based (test_after_task_{k}_on_{t}), but the
trainers may store routed_task as a 0-based prompt index. Comparing the two
directly marks almost every sample as misrouted, i.e. accuracy and error come
out flipped. --task-base auto (default) detects the numbering per model: from the
parquet's own true_task column if present (t or t-1), otherwise from whether
routed_task ever takes the value 0. The detected base is logged; force it with
--task-base 0 or 1.

Outputs (under --out-dir):
  csv/routing_per_seed.csv          raw counts and accuracy per seed
  csv/routing_per_checkpoint.csv    mean/SD over seeds per checkpoint and subgroup
  csv/routing_aggregated.csv        per seed, averaged over checkpoints k >= t,
                                    then mean/SD over seeds
  csv/extremes_per_checkpoint.csv   best/worst subgroup per checkpoint
  csv/extremes_aggregated.csv       best/worst subgroup on aggregated accuracy
  csv/count_extremes_*.csv          subgroup with most/fewest correctly and
                                    incorrectly routed samples
  csv/routing_destinations_*.csv    share of each subgroup's task-t samples
                                    routed to each task r (for confusion maps)
  csv/gap_per_checkpoint.csv        max-min subgroup gap, computed per seed,
                                    then mean/SD over seeds
  tex/routing_extremes_{model}.tex        per task/attribute: highest/lowest accuracy,
                                          most/fewest (in)correctly routed subgroup
  tex/routing_gap_trajectory_{model}.tex  gap per task across checkpoints
  tex/routing_subgroups_{model}.tex       per-subgroup accuracy across tasks

LaTeX packages needed: booktabs, multirow, adjustbox, makecell.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from loguru import logger

from utils.constants import (
    CKPT,
    DEMOGRAPHIC_LABELS,
    DEMOGRAPHICS,
    MODEL_LABELS,
    SUBGROUP_NAMES,
    SUBGROUP_ORDER,
    TEX_SPECIAL,
)

load_dotenv()


def subgroup_key(s) -> tuple:
    """Sort key: known names in code order, then numbers, then other text."""
    s = str(s)
    if s in SUBGROUP_ORDER:
        return (0, SUBGROUP_ORDER[s], "")
    try:
        return (1, float(s), "")
    except ValueError:
        return (2, 0, s)


def subgroup_names(values: pd.Series, demo: str) -> pd.Series:
    """Codes (1, 1.0, "1") -> readable names; unknown values kept as text."""
    num = pd.to_numeric(values, errors="coerce")
    whole = num.notna() & (num % 1 == 0)
    names = num.where(whole).astype("Int64").map(SUBGROUP_NAMES.get(demo, {}))
    text = values.astype("string").where(values.notna())
    return names.astype("string").fillna(text)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--pred-dir",
        type=Path,
        default=Path(os.getenv("PREDICTIONS_DIR", "predictions")),
    )
    p.add_argument("--models", nargs="+", default=list(MODEL_LABELS))
    p.add_argument("--seeds", nargs="+", default=os.getenv("ALL_SEEDS", "42").split())
    p.add_argument("--n-tasks", type=int, default=11)
    p.add_argument(
        "--true-tasks",
        nargs="+",
        type=int,
        default=None,
        help="True tasks to analyse (default: all 1..n-tasks).",
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=30,
        help="Minimum test samples for a subgroup to count towards extremes/gaps.",
    )
    p.add_argument("--digits", type=int, default=3)
    p.add_argument(
        "--task-base",
        choices=["auto", "0", "1"],
        default="auto",
        help="Numbering of routed_task in the parquet files: 0 (prompt "
        "index) or 1 (as in the file names); auto detects it.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("routing_analysis"))
    return p.parse_args()


def mask_unannotated(df: pd.DataFrame) -> pd.DataFrame:
    """Real faces without annotations have 0 in every demographic field. Set their
    demographics to missing so they are not counted as code-0 subgroups
    (e.g. Female, Child). A sample is annotated if any field is > 0."""
    codes = df[DEMOGRAPHICS].apply(pd.to_numeric, errors="coerce")
    if codes.isna().all().all():  # names instead of codes: nothing to mask
        return df
    annotated = (codes > 0).any(axis=1)
    df = df.copy()
    df[DEMOGRAPHICS] = df[DEMOGRAPHICS].astype(object).where(annotated, None)
    return df


def count_routing(df: pd.DataFrame, true_task: int) -> pd.DataFrame:
    """Per-subgroup sample counts and correct routings for one prediction file."""
    correct = pd.to_numeric(df["routed_task"]).eq(true_task)
    parts = []
    for demo in DEMOGRAPHICS:
        part = (
            correct.groupby(subgroup_names(df[demo], demo), dropna=True)
            .agg(n="size", correct="sum")
            .rename_axis("subgroup")
            .reset_index()
        )
        part.insert(0, "demographic", demo)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def count_destinations(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subgroup counts of which task each sample was routed to."""
    routed = pd.to_numeric(df["routed_task"]).rename("routed_task")
    parts = []
    for demo in DEMOGRAPHICS:
        part = (
            routed.groupby([subgroup_names(df[demo], demo).rename("subgroup"), routed])
            .size()
            .rename("count")
            .reset_index()
        )
        part.insert(0, "demographic", demo)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def pred_path(args, model, seed, k, t) -> Path:
    return (
        args.pred_dir
        / model
        / f"seed_run_{seed}"
        / f"test_after_task_{k}_on_{t}.parquet"
    )


def detect_task_offset(args, model: str, true_tasks: list[int]) -> int:
    """Value to add to routed_task so it uses the file names' 1-based numbering."""
    if args.task_base != "auto":
        return 1 - int(args.task_base)
    votes, min_routed, max_routed = [], None, None
    for t in true_tasks:
        for k in range(t, args.n_tasks + 1):
            for seed in args.seeds:
                path = pred_path(args, model, seed, k, t)
                if not path.exists():
                    continue
                cols = [
                    c
                    for c in ("routed_task", "true_task")
                    if c in pq.read_schema(path).names
                ]
                df = pd.read_parquet(path, columns=cols)
                r = pd.to_numeric(df["routed_task"])
                min_routed = r.min() if min_routed is None else min(min_routed, r.min())
                max_routed = r.max() if max_routed is None else max(max_routed, r.max())
                if "true_task" in df:
                    tt = set(pd.to_numeric(df["true_task"]).dropna().astype(int))
                    if tt == {t}:
                        votes.append(0)
                    elif tt == {t - 1}:
                        votes.append(1)
                    else:
                        raise ValueError(
                            f"{path}: true_task values {sorted(tt)} match "
                            f"neither t={t} nor t-1"
                        )
    if votes:
        if len(set(votes)) > 1:
            raise ValueError(
                f"{model}: files disagree on task numbering "
                "(some 0-based, some 1-based); pass --task-base"
            )
        offset, how = votes[0], "the parquet true_task column"
    elif min_routed is not None:
        offset = 1 if min_routed == 0 else 0
        how = f"routed_task range {int(min_routed)}..{int(max_routed)}"
    else:
        return 0
    logger.info(
        "{}: routed_task is {}-based (from {}){}",
        model,
        1 - offset,
        how,
        "; shifted by +1 to match file names" if offset else "",
    )
    return offset


def load_counts(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-subgroup routing counts, per-subgroup routing destinations)."""
    true_tasks = args.true_tasks or list(range(1, args.n_tasks + 1))
    frames, dest_frames, missing = [], [], []

    for model in args.models:
        offset = detect_task_offset(args, model, true_tasks)
        for t in true_tasks:
            for k in range(t, args.n_tasks + 1):
                for seed in args.seeds:
                    path = pred_path(args, model, seed, k, t)
                    if not path.exists():
                        missing.append(path)
                        continue
                    df = mask_unannotated(
                        pd.read_parquet(path, columns=["routed_task", *DEMOGRAPHICS])
                    )
                    df["routed_task"] = pd.to_numeric(df["routed_task"]) + offset
                    if df["routed_task"].max() > k or df["routed_task"].min() < 1:
                        logger.warning(
                            "{}: routed tasks {}..{} outside 1..{} after numbering "
                            "fix; check --task-base",
                            path,
                            df["routed_task"].min(),
                            df["routed_task"].max(),
                            k,
                        )
                    meta = dict(model=model, seed=str(seed), true_task=t, **{CKPT: k})
                    frames.append(count_routing(df, t).assign(**meta))
                    dest_frames.append(count_destinations(df).assign(**meta))

    if missing:
        logger.warning(
            "{} prediction files missing (e.g. {})", len(missing), missing[0]
        )
    if not frames:
        raise FileNotFoundError(f"No prediction files found under {args.pred_dir}")

    counts = pd.concat(frames, ignore_index=True)
    counts["incorrect"] = counts["n"] - counts["correct"]
    counts["accuracy"] = counts["correct"] / counts["n"]
    counts = counts[
        [
            "model",
            "true_task",
            CKPT,
            "seed",
            "demographic",
            "subgroup",
            "n",
            "correct",
            "incorrect",
            "accuracy",
        ]
    ]
    pooled = (
        counts[counts["demographic"] == DEMOGRAPHICS[0]]
        .groupby("model")[["correct", "n"]]
        .sum()
    )
    for model, r in pooled.iterrows():
        logger.info(
            "{}: overall routing accuracy {:.3f} (error {:.3f})",
            model,
            r["correct"] / r["n"],
            1 - r["correct"] / r["n"],
        )
    logger.info(
        "Loaded {} files: models={}, true tasks={}, seeds={}",
        len(frames),
        sorted(counts["model"].unique()),
        sorted(counts["true_task"].unique().tolist()),
        sorted(counts["seed"].unique()),
    )
    dests = pd.concat(dest_frames, ignore_index=True)[
        [
            "model",
            "true_task",
            CKPT,
            "seed",
            "demographic",
            "subgroup",
            "routed_task",
            "count",
        ]
    ]
    return counts, dests


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def summarise_over_seeds(per_seed: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and SD (ddof=1) across seeds. SD is NaN when only one seed exists."""
    return (
        per_seed.groupby(keys, sort=True)
        .agg(
            n_seeds=("seed", "nunique"),
            n=("n", "mean"),
            correct_mean=("correct", "mean"),
            correct_std=("correct", "std"),
            incorrect_mean=("incorrect", "mean"),
            incorrect_std=("incorrect", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
        .reset_index()
    )


def aggregate_checkpoints(counts: pd.DataFrame) -> pd.DataFrame:
    """Per seed, average accuracy over all checkpoints k >= t (equal weight)."""
    return (
        counts.groupby(["model", "true_task", "demographic", "subgroup", "seed"])
        .agg(
            n=("n", "mean"),
            correct=("correct", "mean"),
            incorrect=("incorrect", "mean"),
            accuracy=("accuracy", "mean"),
            n_checkpoints=(CKPT, "nunique"),
        )
        .reset_index()
    )


def find_extremes(stats: pd.DataFrame, keys: list[str], min_n: int) -> pd.DataFrame:
    """Highest/lowest mean-accuracy subgroup per group, ties kept together."""
    eligible = stats[stats["n"] >= min_n]
    n_dropped = len(stats) - len(eligible)
    if n_dropped:
        logger.info(
            "Extremes: {} subgroup rows below min_n={} excluded", n_dropped, min_n
        )

    rows = []
    for key, g in eligible.groupby(keys, sort=True):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for label, target in (
            ("best", g["accuracy_mean"].max()),
            ("worst", g["accuracy_mean"].min()),
        ):
            tied = g[np.isclose(g["accuracy_mean"], target, rtol=0, atol=1e-12)]
            rec[f"{label}_subgroups"] = "; ".join(tied["subgroup"])
            rec[f"{label}_accuracy_mean"] = target
            # Largest SD among tied subgroups (conservative); all SDs kept below.
            rec[f"{label}_accuracy_std"] = tied["accuracy_std"].max()
            rec[f"{label}_accuracy_std_all"] = "; ".join(
                f"{x:.4f}" for x in tied["accuracy_std"]
            )
            rec[f"{label}_n"] = "; ".join(f"{x:g}" for x in tied["n"])
        rec["gap"] = rec["best_accuracy_mean"] - rec["worst_accuracy_mean"]
        rec["n_subgroups"] = len(g)
        rows.append(rec)
    return pd.DataFrame(rows)


def find_count_extremes(
    stats: pd.DataFrame, keys: list[str], min_n: int
) -> pd.DataFrame:
    """Subgroups with the most/fewest correctly and incorrectly routed samples.

    Counts scale with subgroup size, so each pick also records its n and
    routing accuracy for interpretation. Ties are kept together.
    """
    eligible = stats[stats["n"] >= min_n]
    rows = []
    for key, g in eligible.groupby(keys, sort=True):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for metric in ("correct", "incorrect"):
            col = f"{metric}_mean"
            for label, target in (
                (f"most_{metric}", g[col].max()),
                (f"fewest_{metric}", g[col].min()),
            ):
                tied = g[np.isclose(g[col], target, rtol=0, atol=1e-9)]
                rec[f"{label}_subgroups"] = "; ".join(tied["subgroup"])
                rec[f"{label}_mean"] = target
                rec[f"{label}_std"] = tied[f"{metric}_std"].max()
                rec[f"{label}_n"] = "; ".join(f"{x:g}" for x in tied["n"])
                rec[f"{label}_accuracy"] = "; ".join(
                    f"{x:.4f}" for x in tied["accuracy_mean"]
                )
        rec["n_subgroups"] = len(g)
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate_destinations(dests: pd.DataFrame, n_tasks: int) -> pd.DataFrame:
    """Share of a subgroup's task-t samples routed to each task r.

    Per seed and checkpoint the counts are divided by the subgroup size, then
    averaged over checkpoints k >= t (equal weight) and over seeds. Routed
    tasks that received no samples count as 0. Rows with subgroup "All" pool
    every sample with a known value of that attribute.
    """
    keys = ["model", "true_task", CKPT, "seed", "demographic", "subgroup"]
    pooled = (
        dests.groupby([k for k in keys if k != "subgroup"] + ["routed_task"])["count"]
        .sum()
        .reset_index()
        .assign(subgroup="All")
    )
    wide = (
        pd.concat([dests, pooled], ignore_index=True)
        .pivot_table(
            index=keys,
            columns="routed_task",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=range(1, n_tasks + 1), fill_value=0)
    )
    share = wide.div(wide.sum(axis=1), axis=0)
    long = share.stack().rename("share").reset_index()
    per_seed = (
        long.groupby(
            ["model", "true_task", "seed", "demographic", "subgroup", "routed_task"]
        )["share"]
        .mean()
        .reset_index()
    )
    return (
        per_seed.groupby(
            ["model", "true_task", "demographic", "subgroup", "routed_task"]
        )["share"]
        .agg(share_mean="mean", share_std="std")
        .reset_index()
    )


def gap_per_checkpoint(counts: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Max-min subgroup accuracy gap, computed within each seed, then mean/SD."""
    keys = ["model", "true_task", CKPT, "demographic"]
    eligible = counts[counts["n"] >= min_n]
    per_seed = (
        eligible.groupby(keys + ["seed"])["accuracy"]
        .agg(["max", "min", "size"])
        .reset_index()
    )
    per_seed["gap"] = np.where(
        per_seed["size"] >= 2, per_seed["max"] - per_seed["min"], np.nan
    )
    return (
        per_seed.groupby(keys)
        .agg(
            n_seeds=("seed", "nunique"),
            n_subgroups=("size", "max"),
            gap_mean=("gap", "mean"),
            gap_std=("gap", "std"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------
def tex(s) -> str:
    return "".join(TEX_SPECIAL.get(c, c) for c in str(s))


def fmt_stack(mean: float, std: float, d: int, mark=lambda s: s) -> str:
    """Mean on the first line of the cell, SD (smaller) on the line below.

    `mark` decorates the mean only (e.g. underline, dagger).
    """
    if pd.isna(mean):
        return "--"
    top = mark(f"${mean:.{d}f}$")
    if pd.isna(std):
        return top
    return f"\\makecell{{{top} \\\\ {{\\scriptsize $\\pm {std:.{d}f}$}}}}"


def fmt_subgroups(names: str, ns: str) -> str:
    """Subgroup name(s) on the first line, their n on the line below."""
    top = ", ".join(tex(a) for a in names.split("; "))
    bottom = ", ".join(f"{float(b):.0f}" for b in ns.split("; "))
    return f"\\makecell[l]{{{top} \\\\ {{\\scriptsize ($n={bottom}$)}}}}"


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by the fixed demographic order."""
    out = df.assign(_o=df["demographic"].map(DEMOGRAPHICS.index))
    if "subgroup" in df:
        out = out.assign(_s=df["subgroup"].map(subgroup_key))
    return out.sort_values([c for c in ("_o", "_s") if c in out]).drop(
        columns=[c for c in ("_o", "_s") if c in out]
    )


def wrap_table(header, rows, colspec, caption, label, wide=True) -> str:
    """wide=True uses table* (full text width in two-column layouts)."""
    env = "table*" if wide else "table"
    if rows and rows[-1] == r"\midrule":
        rows = rows[:-1]
    lines = [
        f"\\begin{{{env}}}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{adjustbox}{max totalsize={\linewidth}{0.9\textheight}, center}",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        *header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        f"\\end{{{env}}}",
    ]
    return "\n".join(lines) + "\n"


def sd_phrase(n_seeds: int) -> str:
    return (
        f"mean over {n_seeds} seeds, SD in smaller type below each value"
        if n_seeds > 1
        else "single seed"
    )


# ---------------------------------------------------------------------------
# LaTeX tables (one set per model; true tasks as row/column groups)
# ---------------------------------------------------------------------------
def model_label(model: str) -> str:
    return tex(MODEL_LABELS.get(model, model))


def demo_label(demo: str) -> str:
    return tex(DEMOGRAPHIC_LABELS.get(demo, demo))


def task_cell(t, n_rows: int) -> str:
    # Each data row holds two text lines (value + SD), hence 2 * n_rows.
    return f"\\multirow{{{2 * n_rows}}}{{*}}{{{t}}}"


def table_extremes(ext, cext, model, n_tasks, n_seeds, min_n, d) -> str:
    """Combined extremes table.

    Rows: true task (multirow) x attribute.
    Cols: highest/lowest routing accuracy + gap, most/fewest correctly routed,
    most/fewest incorrectly routed. Each subgroup is shown with its n.
    """
    keys = ["model", "true_task", "demographic"]
    sub = ext.merge(cext, on=keys, suffixes=("", "_count"))
    sub = sub[sub["model"] == model]
    header = [
        r" & & \multicolumn{5}{c}{Routing accuracy} "
        r"& \multicolumn{4}{c}{Correctly routed (count)} "
        r"& \multicolumn{4}{c}{Incorrectly routed (count)} \\",
        r"\cmidrule(lr){3-7}\cmidrule(lr){8-11}\cmidrule(lr){12-15}",
        r" & & \multicolumn{2}{c}{Highest} & \multicolumn{2}{c}{Lowest} & "
        r"& \multicolumn{2}{c}{Most} & \multicolumn{2}{c}{Fewest} "
        r"& \multicolumn{2}{c}{Most} & \multicolumn{2}{c}{Fewest} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){8-9}"
        r"\cmidrule(lr){10-11}\cmidrule(lr){12-13}\cmidrule(lr){14-15}",
        r"Task & Attribute & Subgroup & Acc. & Subgroup & Acc. & Gap "
        r"& Subgroup & Count & Subgroup & Count "
        r"& Subgroup & Count & Subgroup & Count \\",
    ]
    # (prefix of subgroup/n columns, prefix of mean/std columns, digits)
    acc_picks = [("best", "best_accuracy", d), ("worst", "worst_accuracy", d)]
    count_picks = [
        (p, p, 1)
        for p in (
            "most_correct",
            "fewest_correct",
            "most_incorrect",
            "fewest_incorrect",
        )
    ]

    def pick(r, who, val, digits):
        return (
            f"{fmt_subgroups(r[f'{who}_subgroups'], r[f'{who}_n'])} & "
            f"{fmt_stack(r[f'{val}_mean'], r[f'{val}_std'], digits)}"
        )

    rows = []
    for t in sorted(sub["true_task"].unique()):
        block = ordered(sub[sub["true_task"] == t])
        for i, (_, r) in enumerate(block.iterrows()):
            first = task_cell(t, len(block)) if i == 0 else ""
            cells = [pick(r, *p) for p in acc_picks]
            cells.append(f"${r['gap']:.{d}f}$")
            cells += [pick(r, *p) for p in count_picks]
            rows.append(
                f"{first} & {demo_label(r['demographic'])} & "
                + " & ".join(cells)
                + r" \\"
            )
            if i < len(block) - 1:
                rows.append(r"\addlinespace[2pt]")
        rows.append(r"\midrule")

    caption = (
        f"{model_label(model)}: demographic subgroups with the highest and lowest "
        f"task-routing accuracy, and with the most and fewest correctly and "
        f"incorrectly routed test samples, on the test data of each task $t$. Values "
        f"are averaged over checkpoints trained through tasks $t$--{n_tasks} "
        f"({sd_phrase(n_seeds)}). Gap is the difference between the highest and "
        f"lowest mean accuracy. $n$: number of test samples in the subgroup; counts "
        f"scale with $n$. Only subgroups with at least {min_n} test samples are "
        f"considered; tied subgroups are listed together with the largest SD among them."
    )
    return wrap_table(
        header,
        rows,
        "ll" + "lc" * 2 + "c" + "lc" * 4,
        caption,
        f"tab:routing-extremes-{model}",
    )


def table_gap_trajectory(gaps, model, n_seeds, min_n) -> str:
    """Rows: true task (multirow) x attribute. Cols: checkpoints (upper triangle)."""
    sub = gaps[gaps["model"] == model]
    ckpts = sorted(sub[CKPT].unique())
    k = len(ckpts)
    header = [
        rf" & & \multicolumn{{{k}}}{{c}}{{Trained through task}} \\",
        rf"\cmidrule(lr){{3-{k + 2}}}",
        "Task & Attribute & " + " & ".join(str(c) for c in ckpts) + r" \\",
    ]
    wide = sub.pivot_table(
        index=["true_task", "demographic"],
        columns=CKPT,
        values=["gap_mean", "gap_std"],
        aggfunc="first",
    )
    rows = []
    for t in sorted(sub["true_task"].unique()):
        demos = [dm for dm in DEMOGRAPHICS if (t, dm) in wide.index]
        for i, demo in enumerate(demos):
            first = task_cell(t, len(demos)) if i == 0 else ""
            cells = [
                ""  # task t not yet learned at checkpoint c < t
                if c < t
                else fmt_stack(
                    wide.loc[(t, demo), ("gap_mean", c)],
                    wide.loc[(t, demo), ("gap_std", c)],
                    2,
                )
                for c in ckpts
            ]
            rows.append(f"{first} & {demo_label(demo)} & " + " & ".join(cells) + r" \\")
            if i < len(demos) - 1:
                rows.append(r"\addlinespace[2pt]")
        rows.append(r"\midrule")

    caption = (
        f"{model_label(model)}: routing-accuracy gap between the best- and "
        f"worst-routed demographic subgroup on the test data of task $t$ (rows), "
        f"after training through each task (columns); computed within each seed "
        f"({sd_phrase(n_seeds)}). Subgroups with fewer than {min_n} test samples "
        f"are excluded."
    )
    return wrap_table(header, rows, "ll" + "c" * k, caption, f"tab:routing-gap-{model}")


def table_subgroups(agg, model, n_tasks, n_seeds, min_n, d) -> str:
    """Rows: attribute (multirow) x subgroup. Cols: true tasks."""
    sub = agg[agg["model"] == model]
    tasks = sorted(sub["true_task"].unique())
    m = len(tasks)
    header = [
        rf" & & \multicolumn{{{m}}}{{c}}{{Test data of task}} \\",
        rf"\cmidrule(lr){{3-{m + 2}}}",
        r"Attribute & Subgroup & " + " & ".join(str(t) for t in tasks) + r" \\",
    ]
    rows = []
    for demo in DEMOGRAPHICS:
        block = sub[sub["demographic"] == demo]
        if block.empty:
            continue
        mean = block.pivot(
            index="subgroup", columns="true_task", values="accuracy_mean"
        )
        std = block.pivot(index="subgroup", columns="true_task", values="accuracy_std")
        n = block.pivot(index="subgroup", columns="true_task", values="n")
        eligible = n >= min_n
        lowest = {
            t: mean.loc[eligible[t], t].idxmin()
            for t in tasks
            if t in mean and mean.loc[eligible[t], t].notna().any()
        }
        subgroups = sorted(mean.index, key=subgroup_key)
        for i, sg in enumerate(subgroups):
            first = (
                f"\\multirow{{{2 * len(subgroups)}}}{{*}}{{{demo_label(demo)}}}"
                if i == 0
                else ""
            )
            cells = []
            for t in tasks:
                if t not in mean or pd.isna(mean.at[sg, t]):
                    cells.append("--")
                    continue
                if not eligible.at[sg, t]:
                    mark = lambda s: s + r"$^\dagger$"
                elif lowest.get(t) == sg:
                    mark = lambda s: f"\\underline{{{s}}}"
                else:
                    mark = lambda s: s
                cells.append(fmt_stack(mean.at[sg, t], std.at[sg, t], d, mark))
            rows.append(f"{first} & {tex(sg)} & " + " & ".join(cells) + r" \\")
            if i < len(subgroups) - 1:
                rows.append(r"\addlinespace[2pt]")
        rows.append(r"\midrule")

    caption = (
        f"{model_label(model)}: task-routing accuracy per demographic subgroup on "
        f"the test data of each task $t$, averaged over checkpoints trained through "
        f"tasks $t$--{n_tasks} ({sd_phrase(n_seeds)}). Underlined: lowest-accuracy "
        f"eligible subgroup per attribute and task. $^\\dagger$: fewer than {min_n} "
        f"test samples; excluded from extremes and gaps. Sample sizes are listed in "
        f"the supplementary CSV."
    )
    return wrap_table(
        header, rows, "ll" + "c" * m, caption, f"tab:routing-subgroups-{model}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    csv_dir, tex_dir = args.out_dir / "csv", args.out_dir / "tex"
    csv_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    counts, dests = load_counts(args)
    models = [m for m in args.models if m in set(counts["model"])]
    n_seeds = counts["seed"].nunique()

    per_ckpt = summarise_over_seeds(
        counts, ["model", "true_task", CKPT, "demographic", "subgroup"]
    )
    agg = summarise_over_seeds(
        aggregate_checkpoints(counts), ["model", "true_task", "demographic", "subgroup"]
    )
    ext_ckpt = find_extremes(
        per_ckpt, ["model", "true_task", CKPT, "demographic"], args.min_n
    )
    ext_agg = find_extremes(agg, ["model", "true_task", "demographic"], args.min_n)
    gaps = gap_per_checkpoint(counts, args.min_n)
    cext_ckpt = find_count_extremes(
        per_ckpt, ["model", "true_task", CKPT, "demographic"], args.min_n
    )
    cext_agg = find_count_extremes(
        agg, ["model", "true_task", "demographic"], args.min_n
    )
    dest_agg = aggregate_destinations(dests, args.n_tasks)

    outputs = {
        "routing_per_seed.csv": counts,
        "routing_per_checkpoint.csv": per_ckpt,
        "routing_aggregated.csv": agg,
        "extremes_per_checkpoint.csv": ext_ckpt,
        "extremes_aggregated.csv": ext_agg,
        "gap_per_checkpoint.csv": gaps,
        "count_extremes_per_checkpoint.csv": cext_ckpt,
        "count_extremes_aggregated.csv": cext_agg,
        "routing_destinations_per_seed.csv": dests,
        "routing_destinations_aggregated.csv": dest_agg,
    }
    for name, df in outputs.items():
        df.to_csv(csv_dir / name, index=False)
        logger.info("Wrote {} ({} rows)", csv_dir / name, len(df))

    for model in models:
        tables = {
            f"routing_extremes_{model}.tex": table_extremes(
                ext_agg, cext_agg, model, args.n_tasks, n_seeds, args.min_n, args.digits
            ),
            f"routing_gap_trajectory_{model}.tex": table_gap_trajectory(
                gaps, model, n_seeds, args.min_n
            ),
            f"routing_subgroups_{model}.tex": table_subgroups(
                agg, model, args.n_tasks, n_seeds, args.min_n, args.digits
            ),
        }
        for name, body in tables.items():
            (tex_dir / name).write_text(body)
        logger.info("Wrote LaTeX tables for model {}", model)

    logger.success("Done. Preamble needs: booktabs, multirow, adjustbox, makecell")


if __name__ == "__main__":
    main()
