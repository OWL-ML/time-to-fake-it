#!/usr/bin/env python3
"""
Per-group and per-subgroup fairness from saved prediction parquet files.

Input layout
------------
Files are found recursively under --root and matched against --pattern, a regex over the
path relative to --root with named groups `method`, `seed`, `task` and optionally
`eval_task`. If the pattern has no `eval_task` group, the evaluated test set is read
from the column given by --eval-col instead.

Conventions
-----------
- Positive class = fake (1), negative = real (0). String labels "fake"/"real" are mapped
  directly; numeric labels/predictions require --numeric-labels {model,groundtruth}.
- task      : checkpoint index (model after training task t)   [evaluate.py: "Generator"]
- eval_task : evaluated generator / test set                    [evaluate.py: "Dataset"]
- The Overall row uses every sample in the (task, eval_task) slice. A subgroup row drops
  samples whose attribute is missing and is skipped unless both classes have at least
  --min-per-class samples.

Selection (defaults compute only what the tables and subgroup plots use)
-------------------------------------------------------------------------
- Files: only checkpoints after --final-task and diagonal checkpoints (task == eval_task);
  --all-files processes every file.
- Groups: Predicted Gender, Predicted Age, Skin Tone (10-level), Intersection; see --groups.
- Per-subgroup stats: only Count, Accuracy, TPR, FPR, Positive Rate (all the fairness metrics
  need); --full-stats adds AUC, EER, average precision and the probability deviations.
- Outputs: per-seed tables only; --seed-summaries adds the mean/std tables.

Output layout
-------------
<out-dir>/<method>/seed_run_<seed>/{subgroup_stats,fairness_group,fairness_subgroup}.csv,
so separate runs never overwrite each other; --flat writes one combined CSV per table.
Every row keeps its method and seed columns, so the files concatenate directly.

Metric definitions follow evaluate.py, with one deliberate change: AUC, EER and average
precision ("Precision") are computed from probability_fake, not from hard predictions.
"Error Rate" is the balanced error rate; "TP/TN Probability Deviation" is the variance of
the temperature-scaled P(fake) over true positives / true negatives.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ALL_GROUP_COLS = [
    "Predicted Gender",
    "Predicted Age",
    "Skin Tone",
    "Skin Tone Group",
    "Intersection",
]
DEFAULT_GROUP_COLS = ["Predicted Gender", "Predicted Age", "Skin Tone", "Intersection"]
# Trainer._write_records names it f"probability_t{temp:g}", e.g. probability_t5
TEMP_COL_RE = re.compile(r"^probability_t[0-9.eE+-]+$")
METRIC_COLS = [
    "Count",
    "Accuracy",
    "FPR",
    "TPR",
    "FNR",
    "TNR",
    "Error Rate",
    "Equal Error Rate",
    "AUC",
    "Precision",
    "Positive Rate",
    "Negative Rate",
    "TP Probability Deviation",
    "TN Probability Deviation",
]
FAIRNESS_COLS = ["FEO", "FDP", "FOAE", "FEOP"]
KEYS = ["method", "seed", "task", "eval_task"]

# e.g. accumulated/seed_run_123/test_after_task_1_on_1.parquet
DEFAULT_PATTERN = (
    r"(?P<method>[^/]+)/seed_run_(?P<seed>\d+)/"
    r"test_after_task_(?P<task>\d+)_on_(?P<eval_task>\d+)\.parquet$"
)


# --------------------------------------------------------------------------- loading


def _as_float(series: pd.Series) -> np.ndarray:
    """Floats, also when cells hold 1-element arrays (as the old evaluate.py wrote them)."""
    if series.dtype == object:
        return series.map(lambda v: float(np.asarray(v).reshape(-1)[0])).to_numpy(float)
    return series.to_numpy(dtype=float)


def _to_fake_is_one(
    series: pd.Series, name: str, numeric_labels: str | None
) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(series):
        values = series.astype(str).str.strip().str.lower()
        unknown = set(values.unique()) - {"fake", "real"}
        if unknown:
            raise ValueError(f"Unexpected values in '{name}': {sorted(unknown)}")
        return (values == "fake").to_numpy(dtype=np.int8)

    if numeric_labels is None:
        raise ValueError(
            f"'{name}' is numeric, so the class convention is ambiguous. Pass "
            "--numeric-labels model (0=fake, 1=real) or groundtruth (1=fake, 0=real)."
        )
    unknown = set(pd.unique(series.dropna())) - {0, 1}
    if unknown or series.isna().any():
        raise ValueError(
            f"'{name}' must contain only 0/1; found extra values {sorted(unknown)} "
            f"and {int(series.isna().sum())} missing"
        )
    codes = series.to_numpy(dtype=np.int8)
    return 1 - codes if numeric_labels == "model" else codes


def _resolve_temp_col(names: list[str], explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in names:
            raise ValueError(f"--temp-col '{explicit}' not found; columns: {names}")
        return explicit
    candidates = [c for c in names if TEMP_COL_RE.match(c)]
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot identify the temperature-scaled probability column "
            f"(candidates: {candidates}). Pass --temp-col."
        )
    return candidates[0]


def load_predictions(
    path: Path,
    group_cols: list[str],
    eval_col: str | None,
    temp_col: str | None,
    numeric_labels: str | None,
    full_stats: bool = False,
) -> pd.DataFrame:
    names = pq.read_schema(path).names
    tcol = _resolve_temp_col(names, temp_col) if full_stats else None
    cols = ["label", "prediction", "probability_fake", *group_cols]
    if tcol is not None:
        cols.append(tcol)
    if eval_col is not None:
        cols.append(eval_col)
    missing = [c for c in cols if c not in names]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    raw = pd.read_parquet(path, columns=cols)
    df = pd.DataFrame(
        {
            "y": _to_fake_is_one(raw["label"], "label", numeric_labels),
            "pred": _to_fake_is_one(raw["prediction"], "prediction", numeric_labels),
            "score": _as_float(raw["probability_fake"]),
        }
    )
    if tcol is not None:
        df["soft"] = _as_float(raw[tcol])
    for col in group_cols:
        values = raw[col]
        if pd.api.types.is_float_dtype(values) and np.allclose(
            values.dropna(), values.dropna().round()
        ):
            values = values.astype("Int64")  # codes stored as float because of NaNs
        df[col] = values.array
    if eval_col is not None:
        df["eval_task"] = raw[eval_col].to_numpy()

    for col, src in [("score", "probability_fake")] + (
        [("soft", tcol)] if tcol else []
    ):
        bad = ~np.isfinite(df[col].to_numpy())
        if bad.any():
            raise ValueError(f"{path}: {int(bad.sum())} non-finite values in '{src}'")

    # Guard against a flipped class convention between prediction and probability_fake.
    agreement = float(np.mean((df["score"] >= 0.5).astype(np.int8) == df["pred"]))
    if agreement < 0.5:
        raise ValueError(
            f"{path}: prediction agrees with probability_fake >= 0.5 for only "
            f"{agreement:.1%} of samples; the class convention is likely flipped."
        )
    if agreement < 0.99:
        logger.warning(f"{path.name}: prediction/probability agreement {agreement:.2%}")

    logger.debug(f"Loaded {len(df)} rows from {path} (temp col: '{tcol}')")
    return df


def discover(root: Path, pattern: str) -> list[tuple[Path, dict]]:
    rx = re.compile(pattern)
    found, seen = [], {}
    for path in sorted(root.rglob("*.parquet")):
        rel = path.relative_to(root).as_posix()
        match = rx.search(rel)
        if match is None:
            logger.warning(f"Skipping (pattern mismatch): {rel}")
            continue
        meta = {
            k: int(v) if v.isdigit() else v
            for k, v in match.groupdict().items()
            if v is not None
        }
        key = tuple(sorted(meta.items()))
        if key in seen:
            raise ValueError(f"{rel} and {seen[key]} map to the same run key {meta}")
        seen[key] = rel
        found.append((path, meta))
    return found


# --------------------------------------------------------------------------- metrics


def _div(num: float, den: float) -> float:
    return num / den if den > 0 else np.nan


def compute_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray | None = None,
    soft: np.ndarray | None = None,
    full: bool = False,
) -> dict:
    """Confusion-matrix rates the fairness metrics need; full=True adds the rest."""
    tp_mask = (y == 1) & (pred == 1)
    tn_mask = (y == 0) & (pred == 0)
    tp, tn = int(tp_mask.sum()), int(tn_mask.sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    n = len(y)

    tpr, fpr = _div(tp, tp + fn), _div(fp, fp + tn)
    if not full:
        return {
            "Count": n,
            "Accuracy": _div(tp + tn, n),
            "FPR": fpr,
            "TPR": tpr,
            "Positive Rate": _div(tp + fp, n),
        }

    auc = eer = ap = np.nan
    if tp + fn > 0 and tn + fp > 0:  # threshold-free metrics need both classes
        auc = roc_auc_score(y, score)
        ap = average_precision_score(y, score)
        roc_fpr, roc_tpr, _ = roc_curve(y, score, pos_label=1)
        eer = float(roc_fpr[np.argmin(np.abs((1 - roc_tpr) - roc_fpr))])

    return {
        "Count": n,
        "Accuracy": _div(tp + tn, n),
        "FPR": fpr,
        "TPR": tpr,
        "FNR": _div(fn, fn + tp),
        "TNR": _div(tn, tn + fp),
        "Error Rate": 1 - (tpr + (1 - fpr)) / 2,
        "Equal Error Rate": eer,
        "AUC": auc,
        "Precision": ap,
        "Positive Rate": _div(tp + fp, n),
        "Negative Rate": _div(tn + fn, n),
        "TP Probability Deviation": float(np.var(soft[tp_mask])) if tp else np.nan,
        "TN Probability Deviation": float(np.var(soft[tn_mask])) if tn else np.nan,
    }


def slice_stats(
    df: pd.DataFrame,
    group_cols: list[str],
    min_per_class: int,
    full_stats: bool = False,
) -> pd.DataFrame:
    def row(frame: pd.DataFrame, group: str, subgroup) -> dict:
        metrics = compute_metrics(
            frame["y"].to_numpy(),
            frame["pred"].to_numpy(),
            frame["score"].to_numpy(),
            frame["soft"].to_numpy() if full_stats else None,
            full=full_stats,
        )
        return {"Group": group, "Subgroup": subgroup, **metrics}

    rows = [row(df, "Overall", pd.NA)]
    if df["y"].nunique() < 2:
        logger.warning(
            f"Slice has only {'fake' if df['y'].iloc[0] else 'real'} samples; "
            "all subgroups are skipped and no fairness rows are produced"
        )
    for group in group_cols:
        annotated = df.dropna(subset=[group])
        if len(annotated) < len(df):
            logger.debug(
                f"{group}: {len(df) - len(annotated)} samples without annotation"
            )
        for subgroup, frame in annotated.groupby(group, sort=True, observed=True):
            n_fake = int(frame["y"].sum())
            n_real = len(frame) - n_fake
            if min(n_fake, n_real) < min_per_class:
                logger.debug(f"Skip {group}={subgroup} (real={n_real}, fake={n_fake})")
                continue
            rows.append(row(frame, group, subgroup))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- fairness


def _with_overall_deltas(stats: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    ref = {
        "TPR": "TPR_all",
        "FPR": "FPR_all",
        "Accuracy": "Accuracy_all",
        "Positive Rate": "PositiveRate_all",
    }
    overall = stats.loc[stats["Group"] == "Overall", keys + list(ref)].rename(
        columns=ref
    )
    sub = stats.loc[stats["Group"] != "Overall"].merge(
        overall, on=keys, how="left", validate="many_to_one"
    )
    sub["TPR_diff"] = (sub["TPR"] - sub["TPR_all"]).abs()
    sub["FPR_diff"] = (sub["FPR"] - sub["FPR_all"]).abs()
    sub["Acc_diff"] = (sub["Accuracy"] - sub["Accuracy_all"]).abs()
    sub["PosRate_diff"] = (sub["Positive Rate"] - sub["PositiveRate_all"]).abs()
    return sub


def group_fairness(stats: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Aggregate across subgroups of each attribute (as calculate_fairness_metrics)."""
    sub = _with_overall_deltas(stats, keys)
    sub["EO_diff"] = sub["TPR_diff"] + sub["FPR_diff"]
    g = sub.groupby(keys + ["Group"], sort=True)
    out = pd.DataFrame(
        {
            "n_subgroups": g.size(),
            "FEO": g["EO_diff"].mean(),
            "FDP": g["Positive Rate"].max() - g["Positive Rate"].min(),
            "FOAE": g["Accuracy"].max() - g["Accuracy"].min(),
            "FEOP": g["TPR_diff"].mean(),
        }
    ).reset_index()
    single = out.loc[out["n_subgroups"] < 2]
    if not single.empty:
        logger.warning(
            f"{len(single)} group rows have < 2 evaluable subgroups; their range-based "
            "metrics (FDP, FOAE) are 0 by construction."
        )
    return out


def subgroup_fairness(stats: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per-subgroup deltas to Overall (as calculate_fairness_metrics_by_subgroup)."""
    sub = _with_overall_deltas(stats, keys)
    out = sub[keys + ["Group", "Subgroup", "Count"]].copy()
    out["FEO"] = sub["TPR_diff"] + sub["FPR_diff"]
    out["FDP"] = sub["PosRate_diff"]
    out["FOAE"] = sub["Acc_diff"]
    out["FEOP"] = sub["TPR_diff"]
    return out


def aggregate_over_seeds(
    df: pd.DataFrame, by: list[str], value_cols: list[str]
) -> pd.DataFrame:
    g = df.groupby(by, dropna=False, sort=True)
    agg = g[value_cols].agg(["mean", "std", "count"])  # std: sample std (ddof=1)
    agg.columns = [
        f"{col}_{'n' if stat == 'count' else stat}" for col, stat in agg.columns
    ]
    agg.insert(0, "n_seeds", g["seed"].nunique())
    agg = agg.reset_index()
    if agg["n_seeds"].nunique() > 1:
        logger.warning(
            f"Unequal seed counts across rows: {sorted(agg['n_seeds'].unique())}"
        )
    return agg


# --------------------------------------------------------------------------- selection


def is_needed(task: int, eval_task: int, final_task: int) -> bool:
    """The plots use the checkpoint after the final task and the diagonal checkpoints."""
    return task == final_task or task == eval_task


def check_coverage(metas: list[dict], final_task: int) -> None:
    expected = {(final_task, e) for e in range(1, final_task + 1)} | {
        (t, t) for t in range(1, final_task + 1)
    }
    runs: dict[tuple, set] = {}
    for meta in metas:
        runs.setdefault((meta["method"], meta["seed"]), set()).add(
            (meta["task"], meta["eval_task"])
        )
    for (method, seed), found in sorted(runs.items()):
        missing = sorted(expected - found)
        if missing:
            logger.warning(
                f"{method} seed={seed}: missing (task, eval_task) files {missing}"
            )


# --------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="directory with prediction parquet files",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("fairness_metrics"),
        help="results go to <out-dir>/<method>/seed_run_<seed>/",
    )
    p.add_argument(
        "--flat",
        action="store_true",
        help="write one combined CSV per table directly in --out-dir instead",
    )
    p.add_argument(
        "--pattern", default=DEFAULT_PATTERN, help="regex over the relative path"
    )
    p.add_argument(
        "--eval-col", default=None, help="column holding the evaluated test set"
    )
    p.add_argument(
        "--max-eval-tasks", type=int, default=50, help="sanity cap for --eval-col"
    )
    p.add_argument(
        "--temp-col", default=None, help="temperature-scaled probability column"
    )
    p.add_argument(
        "--groups", nargs="+", default=DEFAULT_GROUP_COLS, choices=ALL_GROUP_COLS
    )
    p.add_argument("--final-task", type=int, default=11)
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="only these methods (default: every method found under --root)",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="only these seeds (default: every seed found)",
    )
    p.add_argument(
        "--all-files",
        action="store_true",
        help="process every checkpoint, not only final + diagonal",
    )
    p.add_argument(
        "--full-stats",
        action="store_true",
        help="also compute AUC, EER, average precision, probability deviations",
    )
    p.add_argument(
        "--seed-summaries",
        action="store_true",
        help="also write mean/std-over-seeds tables",
    )
    p.add_argument("--min-per-class", type=int, default=1)
    p.add_argument("--numeric-labels", choices=["model", "groundtruth"], default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    if args.min_per_class < 1:
        raise ValueError("--min-per-class must be >= 1")
    pattern_groups = set(re.compile(args.pattern).groupindex)
    missing_groups = {"method", "seed", "task"} - pattern_groups
    if missing_groups:
        raise ValueError(f"--pattern lacks named groups: {sorted(missing_groups)}")
    has_eval_group = "eval_task" in pattern_groups
    if not has_eval_group and args.eval_col is None:
        raise ValueError("Pattern has no 'eval_task' group; pass --eval-col.")
    if has_eval_group and args.eval_col is not None:
        logger.warning("--eval-col is ignored because --pattern captures eval_task")
    eval_col = None if has_eval_group else args.eval_col

    files = discover(args.root, args.pattern)
    if not files:
        raise FileNotFoundError(f"No parquet files under {args.root} match the pattern")
    for field, wanted in (("method", args.methods), ("seed", args.seeds)):
        if wanted is None:
            continue
        available = sorted({meta[field] for _, meta in files}, key=str)
        unknown = [w for w in wanted if w not in available]
        if unknown:
            raise ValueError(
                f"No files for {field}(s) {unknown}; available: {available}"
            )
        files = [(p, m) for p, m in files if m[field] in wanted]
        logger.info(
            f"Selected {field}(s) {sorted(set(wanted), key=str)}: {len(files)} files"
        )

    if has_eval_group:
        check_coverage([meta for _, meta in files], args.final_task)
        if not args.all_files:
            n_found = len(files)
            files = [
                (p, m)
                for p, m in files
                if is_needed(m["task"], m["eval_task"], args.final_task)
            ]
            logger.info(
                f"Using {len(files)} of {n_found} prediction files (final + diagonal)"
            )
    else:
        logger.info(f"Found {len(files)} prediction files")
        if not args.all_files:
            # eval_task is only known after loading, so non-final checkpoints are still read
            logger.info(
                "Selection applied per slice after loading (eval_task from --eval-col)"
            )

    frames = []
    for path, meta in files:
        df = load_predictions(
            path,
            args.groups,
            eval_col,
            args.temp_col,
            args.numeric_labels,
            args.full_stats,
        )
        if has_eval_group:
            slices = [(meta["eval_task"], df)]
        else:
            n_eval = df["eval_task"].nunique()
            if n_eval > args.max_eval_tasks:
                raise ValueError(
                    f"'{args.eval_col}' has {n_eval} unique values in {path.name}; "
                    "it looks like a per-sample index rather than a test-set id."
                )
            slices = df.groupby("eval_task", sort=True)

        for eval_task, sdf in slices:
            if not args.all_files and not is_needed(
                meta["task"], eval_task, args.final_task
            ):
                continue
            stats = slice_stats(sdf, args.groups, args.min_per_class, args.full_stats)
            for i, key in enumerate(KEYS):
                stats.insert(i, key, eval_task if key == "eval_task" else meta[key])
            frames.append(stats)
        logger.info(f"{meta['method']} seed={meta['seed']} task={meta['task']}: done")

    stats = pd.concat(frames, ignore_index=True).sort_values(KEYS, kind="stable")
    fair_group = group_fairness(stats, KEYS)
    fair_subgroup = subgroup_fairness(stats, KEYS)

    outputs = {
        "subgroup_stats": stats,
        "fairness_group": fair_group,
        "fairness_subgroup": fair_subgroup,
    }
    if args.seed_summaries:
        no_seed = [k for k in KEYS if k != "seed"]
        stat_cols = [c for c in METRIC_COLS if c in stats.columns]
        outputs.update(
            {
                "subgroup_stats_seed_mean_std": aggregate_over_seeds(
                    stats, no_seed + ["Group", "Subgroup"], stat_cols
                ),
                "fairness_group_seed_mean_std": aggregate_over_seeds(
                    fair_group, no_seed + ["Group"], ["n_subgroups", *FAIRNESS_COLS]
                ),
                "fairness_subgroup_seed_mean_std": aggregate_over_seeds(
                    fair_subgroup,
                    no_seed + ["Group", "Subgroup"],
                    ["Count", *FAIRNESS_COLS],
                ),
            }
        )

    def write(table: pd.DataFrame, name: str, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        out_path = directory / f"{name}.csv"
        table.to_csv(out_path, index=False)
        logger.info(f"Wrote {out_path} ({len(table)} rows)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        if args.flat:
            write(table, name, args.out_dir)
        elif name.endswith("_seed_mean_std"):  # aggregates seeds: one file per method
            for method, part in table.groupby("method", sort=True):
                write(part, name, args.out_dir / str(method))
        else:
            for (method, seed), part in table.groupby(["method", "seed"], sort=True):
                write(part, name, args.out_dir / str(method) / f"seed_run_{seed}")


if __name__ == "__main__":
    main()
