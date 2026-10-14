"""Replay-buffer retention analysis by demographic subgroup (DER++, GradMatch).

Five questions, each written as CSVs (per seed and mean/SD over seeds) and
LaTeX tables. Figures are made separately from these CSVs by
plot_buffer_heatmaps.py (as plot_routing_heatmaps.py does for routing):

  1. Representativeness  At the end of every task b, does each subgroup's share
                         of the buffer match what the data predicts?
                         ratio = actual share / expected share, where the
                         expected share mixes each origin task's training-data
                         distribution by how much of the buffer that task holds.
                         A chance band (random subsets of the same size from each
                         task's data) shows what pure random retention would give.
  2. Replay              Share of replay draws per subgroup during each task, and
                         exposure = subgroup's draws / all replay draws in the task.
  3. Eviction            Eviction rate (evicted / present during the task), hazard
                         (evictions per 1000 sample-steps), cohort survival and
                         Kaplan-Meier lifetime curves.
  4. Retention vs replay Share of total buffer time vs share of all replay draws.
  5. Link to outcomes    Spearman rho (95% cluster-bootstrap CI over tasks) between
                         the buffer metrics and (a) subgroup forgetting and
                         per-subgroup fairness, (b) task-level fairness change.

Inputs
  <log-root>/<METHOD>/seed_run_<seed>/{events,sampling}.csv          memlog output
  <metadata>                    training split; task read from image_path
  <fairness-root>/<method>/seed_run_<seed>/{subgroup_stats,fairness_group,
                                            fairness_subgroup}.csv

Conventions
  * Membership is rebuilt from events.csv: an accepted insert/replace opens a
    residency, the next evict closes it. lifecycle.csv keeps only the latest
    residency per sample, so it is used only as a cross-check.
  * Origin task of a sample = task of its first insertion.
  * Unannotated samples: images from UNANNOTATED_SOURCES (casia, celeba,
    metfaces; matched on the metadata `source` column or the image path), and
    any sample whose five demographic fields are all <= 0 (0 or -1). Their
    demographic fields are treated as missing even if a value was logged.
    Negative codes are always missing; 0 is otherwise a valid code (Female,
    Child, "Female, Light").
  * Demographic comparisons use annotated samples only, so tasks annotated on
    the fake class only are compared within that slice. Unannotated samples are
    analysed as a separate attribute, "Annotation" (Annotated / Unannotated),
    for questions 1-4. It appears in the CSVs only, not in the figures or LaTeX
    tables. Step 5 skips it, because the test sets have no unannotated samples.
  * The task of a metadata row is the folder right before the split folder in
    image_path: DCFACe/val/fake/DCFACe_val_00001.jpg -> DCFACe,
    STYLEGAN/val/real/FFHQ_val_001.png -> STYLEGAN. TASK_NAMES gives the order
    (task 1..11); names match ignoring case. metadata_task is not used.
  * Runs logged with 0-based tasks (0..n-1) are shifted to 1..n.
  * After a crash + resume, each task keeps only the rows of the last run_id
    that logged it.
  * Time inside a task is measured in logged steps. A residency's time in task t
    is its overlap with that task's step range, so the script works whether
    steps are global or restart every task.

Caching
  Per-run statistics are stored in <out-dir>/cache/<METHOD>/seed_run_<seed>.pkl
  and reused on the next run. A run is recomputed automatically when its
  events.csv / sampling.csv, the metadata file, or --n-tasks / --draws /
  --n-null / --seed changed. Force it with --recompute (all runs) or
  --recompute-runs GradMatch/43 (selected runs); --no-cache disables it.
  Fairness files and --min-buffer / --min-n / --n-boot / --digits are applied
  after the cache, so changing them never needs a recompute.
  The correlation bootstrap is also reused when the data it is built from is
  unchanged. CSVs and LaTeX tables are always rewritten.

LaTeX packages needed: booktabs, multirow, adjustbox, makecell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import rankdata

from utils.constants import (
    ALL_ATTRS,
    ANNOT_ATTR,
    ANNOTATION_FIELDS,
    CACHE_VERSION,
    CACHED_ARGS,
    CLASSES,
    DEMOGRAPHIC_LABELS,
    DEMOGRAPHICS,
    DYN_METRICS,
    FAIR_COLS,
    INSERT_EVENTS,
    LABEL_CODES,
    LOG_COLS,
    LOG_DEMO_FIELDS,
    METHOD_LABELS,
    MISSING,
    RUN_TABLES,
    SPLITS,
    SUBGROUP_LABELS,
    SUBGROUP_ORDER,
    TASK_NAMES,
    UNANNOTATED_SOURCES,
    UNIT_METRICS,
)

from .routing_analysis import (
    fmt_stack,
    fmt_subgroups,
    sd_phrase,
    tex,
    wrap_table,
)


# ---------------------------------------------------------------------------
# Arguments and small helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--log-root", type=Path, default=Path("replay_map"))
    p.add_argument(
        "--metadata", type=Path, default=Path("metadata/train.csv")
    )
    p.add_argument("--fairness-root", type=Path, default=Path("fairness_metrics"))
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Method folder names (default: all under --log-root)",
    )
    p.add_argument("--seeds", nargs="+", default=None)
    p.add_argument("--n-tasks", type=int, default=11)
    p.add_argument(
        "--draws",
        nargs="+",
        default=None,
        help="Only these sampling.csv 'draw' purposes (default: all)",
    )
    p.add_argument(
        "--min-buffer",
        type=int,
        default=10,
        help="Min buffer samples (or expected samples) for a subgroup "
        "to count in extremes and correlations",
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=30,
        help="Min test samples for a subgroup in the fairness files",
    )
    p.add_argument("--n-null", type=int, default=500, help="Chance-band draws")
    p.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples")
    p.add_argument("--digits", type=int, default=2)
    p.add_argument("--seed", type=int, default=0, help="RNG seed for band/bootstrap")
    p.add_argument("--out-dir", type=Path, default=Path("buffer_analysis"))
    p.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore cached per-run results and recompute everything",
    )
    p.add_argument(
        "--recompute-runs",
        nargs="+",
        default=None,
        metavar="METHOD/SEED",
        help="Recompute only these runs, e.g. GradMatch/43 (others reuse cache)",
    )
    p.add_argument(
        "--no-cache", action="store_true", help="Neither read nor write the cache"
    )
    return p.parse_args()


def natural_key(s: str):
    if s in SUBGROUP_ORDER:
        return (0, SUBGROUP_ORDER[s], "")
    return (1, int(s), "") if str(s).isdigit() else (2, 0, str(s))


def annotated_mask(
    df: pd.DataFrame, fields: list[str], source_cols: list[str] = ()
) -> pd.Series:
    """True where the sample is annotated.

    Unannotated = from an UNANNOTATED_SOURCES source (first available column in
    `source_cols`: a source name or an image path), or all fields <= 0.
    """
    present = [c for c in fields if c in df]
    if present:
        annotated = (df[present].apply(pd.to_numeric, errors="coerce") > 0).any(axis=1)
    else:
        annotated = pd.Series(False, index=df.index)
    for col in source_cols:
        if col in df:
            annotated &= ~from_unannotated_source(df[col])
            break
    return annotated


def from_unannotated_source(values: pd.Series) -> pd.Series:
    """Source name ('celeba') or path ('.../celeba/train/real/celeba_train_1.png')."""
    v = (
        values.astype("string")
        .fillna("")
        .str.replace("\\", "/", regex=False)
        .str.lower()
    )
    alt = "|".join(re.escape(s) for s in UNANNOTATED_SOURCES)
    by_name = v.str.strip().isin(UNANNOTATED_SOURCES)
    by_path = v.str.contains(rf"(?:^|/)(?:{alt})(?:/|_{SPLITS}_)", regex=True)
    return (by_name | by_path).astype(bool)


def class_of(
    df: pd.DataFrame, path_cols: list[str], label_cols: list[str]
) -> pd.Series:
    """'real' / 'fake' from the image path (.../<split>/<real|fake>/...), falling
    back to a label column ('real'/'fake' or LABEL_CODES)."""
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    for col in path_cols:
        if col in df:
            p = df[col].astype("string").str.replace("\\", "/", regex=False).str.lower()
            out = out.fillna(
                p.str.extract(rf"(?:^|/){SPLITS}/(real|fake)/", expand=False)
            )
    for col in label_cols:
        if col in df:
            v = norm_codes(df[col]).str.lower().replace(LABEL_CODES)
            out = out.fillna(v.where(v.isin(CLASSES)))
    return out


def annotation_status(annotated: pd.Series) -> pd.Series:
    return annotated.map({True: "Annotated", False: "Unannotated"}).astype("string")


def norm_codes(s: pd.Series, attr: str | None = None) -> pd.Series:
    """Codes as clean strings ('1', not '1.0'), then readable names; blank -> <NA>."""
    out = s.astype("string").str.strip()
    num = pd.to_numeric(out, errors="coerce")
    whole = num.notna() & np.isclose(num.fillna(0.5) % 1, 0)
    out = out.mask(whole, num.where(whole).round().astype("Int64").astype("string"))
    out = out.mask(
        out.str.lower().isin(MISSING) | (num < 0).fillna(False)
    )  # -1 = missing
    if attr in SUBGROUP_LABELS:
        out = out.replace(SUBGROUP_LABELS[attr])
    return out


def safe_div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)


def over_seeds(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and SD (ddof=1) of every numeric column across seeds."""
    vals = [c for c in df.select_dtypes("number").columns if c not in keys]
    g = df.groupby(keys, sort=True, dropna=False)
    out = g[vals].agg(["mean", "std"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    out.insert(0, "n_seeds", g["seed"].nunique())
    return out.reset_index()


def method_label(m: str) -> str:
    return tex(METHOD_LABELS.get(m, m))


def demo_label(d: str) -> str:
    return tex(DEMOGRAPHIC_LABELS.get(d, d))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_metadata(path: Path) -> dict[str, pd.DataFrame]:
    """Per task and subgroup: number of training samples and share, for all
    samples and separately for each class ('all', 'real', 'fake')."""
    header = pd.read_csv(path, nrows=0).columns
    src_cols = [c for c in ("source", "image_path", "path") if c in header]
    path_col = next((c for c in ("image_path", "path") if c in header), None)
    if path_col is None:
        raise ValueError(f"{path}: needs an image_path (or path) column")
    label_cols = [c for c in ("Target", "label") if c in header]
    meta = pd.read_csv(
        path, usecols=[*ANNOTATION_FIELDS, *src_cols, *label_cols], low_memory=False
    )
    meta["cls"] = class_of(meta, [path_col], label_cols)
    unknown = int(meta["cls"].isna().sum())
    if unknown:
        logger.warning(
            "Metadata: {} rows with no real/fake class; they count in 'all' only",
            unknown,
        )
    meta["task"] = task_from_path(meta[path_col], path)
    annotated = annotated_mask(meta, ANNOTATION_FIELDS, src_cols)
    logger.info("Metadata: {:.1%} of training samples annotated", annotated.mean())
    if src_cols:
        zero_rows = ~annotated_mask(meta, ANNOTATION_FIELDS)
        by_source = from_unannotated_source(meta[src_cols[0]])
        extra = int((zero_rows & ~by_source).sum())
        if extra:
            logger.warning(
                "Metadata: {} rows outside {} have no annotation (all "
                "fields <= 0); also treated as unannotated",
                extra,
                UNANNOTATED_SOURCES,
            )
    meta[ANNOT_ATTR] = annotation_status(annotated)

    for attr in ALL_ATTRS:
        meta[f"_sg_{attr}"] = (
            meta[ANNOT_ATTR]
            if attr == ANNOT_ATTR
            else norm_codes(meta[attr], attr).where(annotated)
        )

    def counts(m: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for attr in ALL_ATTRS:
            c = (
                m.assign(subgroup=m[f"_sg_{attr}"])
                .dropna(subset=["subgroup"])
                .groupby(["task", "subgroup"])
                .size()
                .rename("n_data")
                .reset_index()
            )
            parts.append(c.assign(demographic=attr))
        data = pd.concat(parts, ignore_index=True)
        data["data_share"] = data["n_data"] / data.groupby(["demographic", "task"])[
            "n_data"
        ].transform("sum")
        return data

    logger.info(
        "Metadata: {} rows, tasks {}", len(meta), sorted(meta["task"].unique().tolist())
    )
    out = {"all": counts(meta)}
    for cls in CLASSES:
        out[cls] = counts(meta[meta["cls"] == cls])
    return out


def task_from_path(paths: pd.Series, where: Path) -> pd.Series:
    """Task index from the folder right before the split folder.

    DCFACe/val/fake/DCFACe_val_00001.jpg -> DCFACe -> 9
    STYLEGAN/val/real/FFHQ_val_001.png   -> STYLEGAN -> 6 (not FFHQ)
    Works for relative and absolute paths.
    """
    index = {name.lower(): i + 1 for i, name in enumerate(TASK_NAMES)}
    p = paths.astype("string").str.replace("\\", "/", regex=False).str.strip()
    folder = p.str.extract(rf"(?:^|/)([^/]+)/{SPLITS}/", expand=False)
    task = folder.str.lower().map(index)

    no_split = folder.isna()
    if no_split.any():
        raise ValueError(
            f"{where}: {int(no_split.sum())} image paths have no <task>/<split>/ "
            f"folder (split = train/val/test), e.g. {p[no_split].iloc[0]!r}"
        )
    unknown = sorted(folder[task.isna()].unique())
    if unknown:
        raise ValueError(f"{where}: task folders not in TASK_NAMES: {unknown}")
    counts = folder.value_counts()
    logger.info(
        "Metadata tasks from image_path: {}",
        ", ".join(
            f"{n}={counts.get(n, 0)}"
            for n in sorted(counts.index, key=lambda n: index[n.lower()])
        ),
    )
    return task.astype(int)


def read_log(path: Path, wanted: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=wanted)
    header = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(
        path, usecols=[c for c in wanted if c in header], low_memory=False
    )


def keep_last_run(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """After crash + resume, keep each task's rows from the last run_id only."""
    if "run_id" not in df or df["run_id"].nunique() <= 1:
        return df
    order = {r: i for i, r in enumerate(pd.unique(df["run_id"]))}  # append order
    last = df.groupby("task")["run_id"].agg(lambda s: max(s.unique(), key=order.get))
    keep = df["run_id"].eq(df["task"].map(last))
    if (~keep).any():
        logger.warning(
            "{}: dropped {} rows from superseded runs (resume)",
            name,
            int((~keep).sum()),
        )
    return df[keep].copy()


def fix_task_offset(ev, sp, tag: str, n_tasks: int) -> None:
    """Shift runs logged as 0..n-1 to 1..n (in place)."""
    tasks = pd.concat([ev["task"], sp["task"]]).dropna()
    lo, hi = int(tasks.min()), int(tasks.max())
    if lo == 0:
        if hi > n_tasks - 1:
            raise ValueError(f"{tag}: tasks span {lo}..{hi}, cannot be 0-based")
        logger.warning("{}: tasks logged as 0..{}; shifted to 1..{}", tag, hi, hi + 1)
        ev["task"] += 1
        sp["task"] += 1
    elif lo < 1 or hi > n_tasks:
        raise ValueError(f"{tag}: unexpected task range {lo}..{hi}")


def load_run(run_dir: Path, tag: str, n_tasks: int, draws) -> tuple:
    demo = LOG_DEMO_FIELDS
    ev = read_log(
        run_dir / "events.csv",
        [
            "task",
            "global_step",
            "event",
            "accepted",
            "sample_uid",
            "image_path",
            "dataset",
            "label",
            *demo,
            "run_id",
        ],
    )
    sp = read_log(
        run_dir / "sampling.csv",
        ["task", "step", "sample_uid", "draw", "dataset", "label", *demo, "run_id"],
    )
    for df in (ev, sp):
        df["task"] = pd.to_numeric(df["task"]).astype(int)
    ev = keep_last_run(ev, f"{tag} events")
    sp = keep_last_run(sp, f"{tag} sampling")
    fix_task_offset(ev, sp, tag, n_tasks)

    for df in (ev, sp):
        # sample_uid is the absolute image path
        annotated = annotated_mask(df, LOG_DEMO_FIELDS, ["sample_uid", "image_path"])
        for log_col, attr in LOG_COLS.items():
            df[attr] = (
                norm_codes(df[log_col], attr).where(annotated)
                if log_col in df
                else pd.NA
            )
        df[ANNOT_ATTR] = annotation_status(annotated)
        df["cls"] = class_of(df, ["sample_uid", "image_path"], ["label"])
    ev["accepted"] = (
        ev["accepted"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    )
    if draws is not None and "draw" in sp:
        sp = sp[sp["draw"].isin(draws)].copy()
    return ev.reset_index(drop=True), sp.reset_index(drop=True)


def build_intervals(ev: pd.DataFrame, tag: str) -> pd.DataFrame:
    """One row per residency: start/end (task, step), origin task, demographics."""
    e = ev[ev["sample_uid"].notna()].copy()
    demo = e.groupby("sample_uid")[[*ALL_ATTRS, "cls"]].first()  # first non-null
    e["_ord"] = np.arange(len(e))
    is_ins = e["event"].isin(INSERT_EVENTS) & e["accepted"]
    is_ev = e["event"].eq("evict")
    e = e[is_ins | is_ev].assign(is_ins=is_ins[is_ins | is_ev])
    if not e["is_ins"].any():
        kinds = sorted(ev["event"].dropna().unique())
        raise ValueError(f"{tag}: no accepted insert/replace events (events: {kinds})")
    e = e.sort_values(["sample_uid", "task", "global_step", "_ord"], kind="stable")
    e["res"] = e.groupby("sample_uid")["is_ins"].cumsum()

    orphan = (~e["is_ins"]) & (e["res"] == 0)
    if orphan.any():
        logger.warning(
            "{}: {} evictions of samples never inserted in the log",
            tag,
            int(orphan.sum()),
        )

    cols = {"task": "start_task", "global_step": "start_step"}
    starts = e[e["is_ins"]].groupby(["sample_uid", "res"])[list(cols)].first()
    ends = (
        e[~e["is_ins"] & (e["res"] > 0)]
        .groupby(["sample_uid", "res"])[list(cols)]
        .first()
    )
    iv = (
        starts.rename(columns=cols)
        .join(ends.rename(columns={"task": "end_task", "global_step": "end_step"}))
        .reset_index()
    )

    # Re-inserted without a logged evict: close the earlier residency there.
    nxt = iv.groupby("sample_uid")[["start_task", "start_step"]].shift(-1)
    dangling = iv["end_task"].isna() & nxt["start_task"].notna()
    if dangling.any():
        logger.warning(
            "{}: {} residencies re-inserted without an evict; closed at the re-insert",
            tag,
            int(dangling.sum()),
        )
        iv.loc[dangling, ["end_task", "end_step"]] = nxt.loc[dangling].to_numpy()

    iv["evicted"] = iv["end_task"].notna()
    iv["origin_task"] = iv.groupby("sample_uid")["start_task"].transform("min")
    return iv.join(demo, on="sample_uid")


def task_bounds(ev, sp) -> tuple[dict, dict]:
    steps = pd.concat(
        [
            ev[["task", "global_step"]].rename(columns={"global_step": "step"}),
            sp[["task", "step"]],
        ]
    )
    b = steps.groupby("task")["step"].agg(["min", "max"])
    return b["min"].to_dict(), b["max"].to_dict()


def resident_at_end(iv: pd.DataFrame, b: int, last: dict) -> np.ndarray:
    z = last[b]
    started = (iv["start_task"] < b) | (
        (iv["start_task"] == b) & (iv["start_step"] <= z)
    )
    alive = (
        iv["end_task"].isna()
        | (iv["end_task"] > b)
        | ((iv["end_task"] == b) & (iv["end_step"] > z))
    )
    return (started & alive).to_numpy()


def task_overlap(iv: pd.DataFrame, t: int, first: dict, last: dict) -> np.ndarray:
    """Steps each residency spent in the buffer during task t."""
    a, z = first[t], last[t]
    st, ss = iv["start_task"].to_numpy(), iv["start_step"].to_numpy(float)
    et, es = iv["end_task"].to_numpy(float), iv["end_step"].to_numpy(float)
    lo = np.where(st < t, a, np.where(st == t, ss, np.inf))
    open_ = np.isnan(et) | (et > t)
    hi = np.where(open_, z, np.where(et == t, es, -np.inf))
    return np.clip(hi - lo, 0, None)


def load_fairness(root: Path, method: str, seed: str, final: int, min_n: int):
    """Subgroup-level (forgetting, per-subgroup fairness) and task-level tables."""
    d = root / method.lower() / f"seed_run_{seed}"
    need = ["subgroup_stats.csv", "fairness_subgroup.csv", "fairness_group.csv"]
    if not all((d / f).exists() for f in need):
        logger.warning("No fairness files in {}; step 5 skips this run", d)
        return None, None
    stats = pd.read_csv(d / "subgroup_stats.csv")
    fsub = pd.read_csv(d / "fairness_subgroup.csv")
    fgrp = pd.read_csv(d / "fairness_group.csv")

    def prep(df):
        df = df[df["Group"].isin(DEMOGRAPHICS)].copy()
        if "Subgroup" in df:
            df["subgroup"] = pd.concat(
                [norm_codes(g["Subgroup"], a) for a, g in df.groupby("Group")]
            )
        return df.rename(columns={"Group": "demographic", "eval_task": "origin_task"})

    stats, fsub, fgrp = prep(stats), prep(fsub), prep(fgrp)
    keys = ["origin_task", "demographic", "subgroup"]
    diag = stats[stats["task"] == stats["origin_task"]].set_index(keys)["Accuracy"]
    fin = stats[stats["task"] == final].set_index(keys)
    sub = pd.DataFrame(
        {
            "forgetting": diag - fin["Accuracy"],
            "test_n": fin["Count"],
        }
    )
    sub = sub.join(fsub[fsub["task"] == final].set_index(keys)[FAIR_COLS])
    sub = sub[sub["test_n"] >= min_n].reset_index()

    gk = ["origin_task", "demographic"]
    g_diag = fgrp[fgrp["task"] == fgrp["origin_task"]].set_index(gk)[FAIR_COLS]
    g_fin = fgrp[fgrp["task"] == final].set_index(gk)[FAIR_COLS]
    task = (g_fin - g_diag).add_prefix("delta_").reset_index()

    sub = sub[sub["origin_task"] < final]  # final task: no forgetting yet
    task = task[task["origin_task"] < final]
    return sub, task


# ---------------------------------------------------------------------------
# Per-run statistics
# ---------------------------------------------------------------------------
def representation(iv, data, last, tasks, rng, n_null):
    """Q1: buffer share vs data share at each task boundary, with chance band."""
    detail, pooled = [], []
    for b in tasks:
        res = iv[resident_at_end(iv, b, last)]
        for attr in ALL_ATTRS:
            r = res.dropna(subset=[attr])
            if r.empty:
                continue
            d = data[data["demographic"] == attr]
            origins = sorted(r["origin_task"].unique())
            ds = (
                d[d["task"].isin(origins)]
                .pivot(index="task", columns="subgroup", values="n_data")
                .fillna(0)
            )
            bc = r.groupby(["origin_task", attr]).size().unstack(fill_value=0)
            missing = set(bc.index) - set(ds.index)
            if missing:
                logger.warning(
                    "{}: origin tasks {} missing from metadata", attr, missing
                )
            groups = sorted(ds.columns.union(bc.columns), key=natural_key)
            ds = ds.reindex(columns=groups, fill_value=0)
            bc = bc.reindex(index=ds.index, columns=groups, fill_value=0)

            m = bc.sum(axis=1)
            M = m.sum()
            dshare = ds.div(ds.sum(axis=1), axis=0)
            expected = dshare.mul(m / M, axis=0).sum()
            actual = bc.sum() / M

            null = np.zeros((max(n_null, 1), len(groups)))
            for j in bc.index if n_null > 0 else []:
                pool, k = ds.loc[j].to_numpy().astype(np.int64), int(m[j])
                if k == 0:
                    continue
                if k <= pool.sum():
                    null += rng.multivariate_hypergeometric(pool, k, size=n_null)
                else:  # more buffer samples than annotated data: sample with replacement
                    null += rng.multinomial(k, pool / pool.sum(), size=n_null)
            if n_null > 0:
                null_ratio = safe_div(
                    null / M, np.broadcast_to(expected.to_numpy(), null.shape)
                )
                lo, hi = np.nanpercentile(null_ratio, [2.5, 97.5], axis=0)
            else:  # chance band not requested
                lo = hi = np.full(len(groups), np.nan)

            pooled.append(
                pd.DataFrame(
                    {
                        "boundary": b,
                        "demographic": attr,
                        "subgroup": groups,
                        "n": bc.sum().to_numpy(),
                        "expected_n": (expected * M).to_numpy(),
                        "actual_share": actual.to_numpy(),
                        "expected_share": expected.to_numpy(),
                        "ratio": safe_div(actual, expected),
                        "chance_lo": lo,
                        "chance_hi": hi,
                    }
                )
            )
            share = bc.div(m, axis=0)
            det = pd.DataFrame(
                {
                    "n": bc.stack(),
                    "buffer_share": share.stack(),
                    "data_share": dshare.stack(),
                }
            )
            det["ratio"] = safe_div(det["buffer_share"], det["data_share"])
            det.index.names = ["origin_task", "subgroup"]
            detail.append(det.reset_index().assign(boundary=b, demographic=attr))
    return pd.concat(pooled, ignore_index=True), pd.concat(detail, ignore_index=True)


def retention(iv, last, tasks):
    """Share of a subgroup's buffer samples at the end of task b-1 that are
    still in the buffer at the end of task b (not defined for the first task)."""
    rows = []
    for prev, b in zip(tasks[:-1], tasks[1:]):
        was = resident_at_end(iv, prev, last)
        kept = was & resident_at_end(iv, b, last)
        for attr in ALL_ATTRS:
            lab = iv[attr]
            ok = lab.notna().to_numpy()
            g = pd.DataFrame({"subgroup": lab[ok], "was": was[ok], "kept": kept[ok]})
            c = g.groupby("subgroup")[["was", "kept"]].sum()
            c = c[c["was"] > 0]
            rows.append(
                pd.DataFrame(
                    {
                        "boundary": b,
                        "demographic": attr,
                        "subgroup": c.index,
                        "n_prev": c["was"].to_numpy(),
                        "n_kept": c["kept"].to_numpy(),
                        "retention": (c["kept"] / c["was"]).to_numpy(),
                    }
                )
            )
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "boundary",
                "demographic",
                "subgroup",
                "n_prev",
                "n_kept",
                "retention",
            ]
        )
    )


def dynamics(iv, sp, first, last, tasks, tag):
    """Per (task, origin task, subgroup): buffer time, draws, evictions, at risk."""
    origin_of = iv.groupby("sample_uid")["origin_task"].first()
    sp = sp.assign(origin_task=sp["sample_uid"].map(origin_of))
    unknown = sp["origin_task"].isna().sum()
    if unknown:
        logger.warning(
            "{}: {} replay draws of samples never inserted in the log",
            tag,
            int(unknown),
        )
    rows = []
    for t in tasks:
        base = iv[["origin_task", *ALL_ATTRS]].assign(
            time=task_overlap(iv, t, first, last),
            evicted=(iv["end_task"] == t).to_numpy(),
            at_risk=(
                (iv["start_task"] <= t)
                & (iv["end_task"].isna() | (iv["end_task"] >= t))
            ).to_numpy(),
        )
        base = base[base["at_risk"]]
        drawn = sp[(sp["task"] == t) & sp["origin_task"].notna()]
        for attr in ALL_ATTRS:
            a = (
                base.dropna(subset=[attr])
                .groupby(["origin_task", attr])[["time", "evicted", "at_risk"]]
                .sum()
            )
            dr = drawn.dropna(subset=[attr]).groupby(["origin_task", attr]).size()
            m = a.join(dr.rename("draws"), how="outer").fillna(0)
            m.index.names = ["origin_task", "subgroup"]
            rows.append(m.reset_index().assign(task=t, demographic=attr))
    return pd.concat(rows, ignore_index=True)


def task_metrics(dyn: pd.DataFrame) -> pd.DataFrame:
    """Q2 + Q3 per task: replay share, exposure, eviction rate, hazard."""
    g = (
        dyn.groupby(["task", "demographic", "subgroup"])[
            ["time", "draws", "evicted", "at_risk"]
        ]
        .sum()
        .reset_index()
    )
    tot = g.groupby(["task", "demographic"])
    g["buffer_share"] = safe_div(g["time"], tot["time"].transform("sum"))
    g["replay_share"] = safe_div(g["draws"], tot["draws"].transform("sum"))
    # replay share relative to buffer occupancy (earlier definition, CSV only)
    g["exposure_vs_occupancy"] = safe_div(g["replay_share"], g["buffer_share"])
    g["eviction_rate"] = safe_div(g["evicted"], g["at_risk"])
    g["hazard_per_1k_steps"] = safe_div(1000 * g["evicted"], g["time"])
    return g


def retention_vs_replay(dyn: pd.DataFrame) -> pd.DataFrame:
    """Q4: share of all buffer time vs share of all replay draws (whole run)."""
    g = dyn.groupby(["demographic", "subgroup"])[["time", "draws"]].sum().reset_index()
    tot = g.groupby("demographic")
    g["retention_share"] = safe_div(g["time"], tot["time"].transform("sum"))
    g["replay_share"] = safe_div(g["draws"], tot["draws"].transform("sum"))
    g["replay_per_retention"] = safe_div(g["replay_share"], g["retention_share"])
    return g


def durations(iv: pd.DataFrame, final: int) -> tuple[np.ndarray, np.ndarray]:
    """Task boundaries survived; still-resident samples are censored."""
    ev = iv["evicted"].to_numpy()
    dur = np.where(ev, iv["end_task"] - iv["start_task"], final - iv["start_task"] + 1)
    return dur.astype(float), ev


def cohort_and_km(iv, last, tasks, final):
    """Q3: cohort survival (t_in, b) and Kaplan-Meier survival per subgroup."""
    dur, ev = durations(iv, final)
    coh_rows, km_rows = [], []
    for attr in ALL_ATTRS:
        lab = iv[attr].astype(object).where(iv[attr].notna(), None)
        annotated = lab.notna().to_numpy()
        groups = sorted(lab[annotated].unique(), key=natural_key) + ["All"]
        for g in groups:
            sel = annotated & ((lab == g).to_numpy() if g != "All" else True)
            # Cohort survival
            for t_in in tasks:
                c = sel & (iv["start_task"] == t_in).to_numpy()
                n = int(c.sum())
                if n == 0:
                    continue
                for b in [x for x in tasks if x >= t_in]:
                    alive = int((c & resident_at_end(iv, b, last)).sum())
                    coh_rows.append((attr, g, t_in, b, n, alive / n))
            # Kaplan-Meier
            d_, e_ = dur[sel], ev[sel]
            s = 1.0
            for k in range(0, len(tasks) + 1):
                at_risk = int(((e_) & (d_ >= k)).sum() + ((~e_) & (d_ > k)).sum())
                km_rows.append((attr, g, k, s, at_risk))
                deaths = int(((e_) & (d_ == k)).sum())
                if at_risk:
                    s *= 1 - deaths / at_risk
    coh = pd.DataFrame(
        coh_rows,
        columns=["demographic", "subgroup", "cohort_task", "boundary", "n", "survival"],
    )
    km = pd.DataFrame(
        km_rows,
        columns=["demographic", "subgroup", "boundaries", "survival", "at_risk"],
    )
    return coh, km


def unit_metrics(dyn, iv, detail, final, total_draws: dict):
    """Buffer metrics per (origin task, subgroup) for the step-5 correlations.

    exposure: the cohort's replay draws from its origin task onwards divided by
    all replay draws in those tasks.
    """
    keys = ["origin_task", "demographic", "subgroup"]
    o = dyn[dyn["task"] >= dyn["origin_task"]].groupby(keys)[["draws"]].sum()
    origins = o.index.get_level_values("origin_task")
    draws_from = {
        j: sum(n for t, n in total_draws.items() if t >= j) for j in set(origins)
    }
    out = pd.DataFrame(
        {
            "exposure": safe_div(o["draws"], origins.map(draws_from).to_numpy(float)),
        },
        index=o.index,
    )

    dur, ev = durations(iv, final)
    life = []
    for attr in DEMOGRAPHICS:
        f = iv.assign(dur=dur, ev=ev).dropna(subset=[attr])
        s = f.groupby(["origin_task", attr]).agg(
            n_buffer=("dur", "size"),
            eviction_rate=("ev", "mean"),
            lifetime=("dur", "mean"),
        )
        s.index.names = ["origin_task", "subgroup"]
        life.append(s.reset_index().assign(demographic=attr))
    life = pd.concat(life).set_index(keys)

    rep = (
        detail[detail["boundary"] >= detail["origin_task"]]
        .groupby(keys)["ratio"]
        .mean()
    )
    return (
        out.join(life, how="outer")
        .join(rep.rename("rep_ratio"), how="outer")
        .reset_index()
    )


def lifecycle_check(run_dir: Path, iv: pd.DataFrame, tag: str) -> None:
    path = run_dir / "lifecycle.csv"
    if not path.exists():
        return
    lc = pd.read_csv(path, usecols=["evicted"])
    in_lc = int((~lc["evicted"].astype(str).str.lower().isin(["true", "1"])).sum())
    in_ev = int((~iv["evicted"]).sum())
    if in_lc != in_ev:
        logger.warning(
            "{}: {} residents at end from events.csv vs {} in lifecycle.csv",
            tag,
            in_ev,
            in_lc,
        )


# ---------------------------------------------------------------------------
# Step 5: correlations
# ---------------------------------------------------------------------------
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return np.nan
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def correlate(units, unit_keys, xcols, ycols, n_boot, rng, level):
    """Spearman rho on seed-averaged units; CI by resampling origin tasks."""
    rows = []
    for (method, attr), u in units.groupby(["method", "demographic"]):
        per_seed = {s: g for s, g in u.groupby("seed")}
        avg = u.groupby(unit_keys)[xcols + ycols].mean().reset_index()
        clusters = avg["origin_task"].to_numpy()
        ids = np.unique(clusters)
        members = [np.flatnonzero(clusters == c) for c in ids]
        for x in xcols:
            for y in ycols:
                xv, yv = avg[x].to_numpy(float), avg[y].to_numpy(float)
                rho = spearman(xv, yv)
                boots = []
                for _ in range(n_boot):
                    pick = np.concatenate(
                        [members[i] for i in rng.integers(0, len(ids), len(ids))]
                    )
                    boots.append(spearman(xv[pick], yv[pick]))
                lo, hi = (
                    np.nanpercentile(boots, [2.5, 97.5])
                    if np.isfinite(boots).any()
                    else (np.nan, np.nan)
                )
                seed_rho = [
                    spearman(g[x].to_numpy(float), g[y].to_numpy(float))
                    for g in per_seed.values()
                ]
                rows.append(
                    {
                        "level": level,
                        "method": method,
                        "demographic": attr,
                        "x": x,
                        "y": y,
                        "n_units": int((~np.isnan(xv) & ~np.isnan(yv)).sum()),
                        "n_tasks": len(ids),
                        "rho": rho,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "rho_seed_mean": np.nanmean(seed_rho) if seed_rho else np.nan,
                        "rho_seed_std": np.nanstd(seed_rho, ddof=1)
                        if len(seed_rho) > 1
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def extremes(stats, keys, metric, count_col, min_count):
    """Highest/lowest seed-mean value of `metric` per group, ties kept."""
    mean, std = f"{metric}_mean", f"{metric}_std"
    elig = stats[(stats[count_col] >= min_count) & stats[mean].notna()]
    rows = []
    for key, g in elig.groupby(keys, sort=True):
        rec = dict(zip(keys, key))
        for side, target in (("hi", g[mean].max()), ("lo", g[mean].min())):
            tied = g[np.isclose(g[mean], target, rtol=0, atol=1e-12)]
            rec[f"{side}_subgroups"] = "; ".join(tied["subgroup"])
            rec[f"{side}_n"] = "; ".join(f"{x:.0f}" for x in tied["n_mean"])
            rec[f"{side}_mean"] = target
            rec[f"{side}_std"] = tied[std].max()
        rows.append(rec)
    return pd.DataFrame(
        rows,
        columns=list(keys)
        + [f"{s}_{c}" for s in ("hi", "lo") for c in ("subgroups", "n", "mean", "std")],
    )


def table_buffer_extremes(rep_m, task_m, method, n_seeds, min_buffer, d) -> str:
    rep = rep_m[rep_m["method"] == method].rename(columns={"boundary": "task"})
    tm = task_m[task_m["method"] == method].copy()
    tm["n_mean"] = tm["at_risk_mean"]
    parts = {
        "rep": extremes(
            rep, ["task", "demographic"], "ratio", "expected_n_mean", min_buffer
        ),
        "exp": extremes(
            tm, ["task", "demographic"], "exposure", "at_risk_mean", min_buffer
        ),
        "evi": extremes(
            tm, ["task", "demographic"], "eviction_rate", "at_risk_mean", min_buffer
        ),
    }
    # Percentage of the buffer without demographic annotation, per task boundary
    unann = rep[(rep["demographic"] == ANNOT_ATTR) & (rep["subgroup"] == "Unannotated")]
    unann = unann.set_index("task")[["actual_share_mean", "actual_share_std"]] * 100

    header = [
        r" & & & \multicolumn{4}{c}{Representation ratio} "
        r"& \multicolumn{4}{c}{Exposure} & \multicolumn{4}{c}{Eviction rate} \\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}\cmidrule(lr){12-15}",
        r" & & & \multicolumn{2}{c}{Most over} & \multicolumn{2}{c}{Most under} "
        r"& \multicolumn{2}{c}{Highest} & \multicolumn{2}{c}{Lowest} "
        r"& \multicolumn{2}{c}{Highest} & \multicolumn{2}{c}{Lowest} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}"
        r"\cmidrule(lr){10-11}\cmidrule(lr){12-13}\cmidrule(lr){14-15}",
        r"Task & \makecell{Unannot.\\(\%)} & Attribute"
        + r" & Subgroup & Value" * 6
        + r" \\",
    ]
    idx = {k: v.set_index(["task", "demographic"]) for k, v in parts.items()}
    tasks = sorted(
        set().union(*[set(v.index.get_level_values(0)) for v in idx.values()])
    )
    rows = []
    for t in tasks:
        demos = [
            a for a in DEMOGRAPHICS if any((t, a) in v.index for v in idx.values())
        ]
        for i, attr in enumerate(demos):
            span = 2 * len(demos)
            if i == 0:
                if t in unann.index:
                    pct = fmt_stack(
                        unann.at[t, "actual_share_mean"],
                        unann.at[t, "actual_share_std"],
                        1,
                    )
                else:
                    pct = "$0.0$"  # no unannotated samples in the data so far
                first = (
                    f"\\multirow{{{span}}}{{*}}{{{t}}} & "
                    f"\\multirow{{{span}}}{{*}}{{{pct}}}"
                )
            else:
                first = " & "
            cells = []
            for k in ("rep", "exp", "evi"):
                for side in ("hi", "lo"):
                    if (t, attr) not in idx[k].index:
                        cells.append("-- & --")
                        continue
                    r = idx[k].loc[(t, attr)]
                    cells.append(
                        f"{fmt_subgroups(r[f'{side}_subgroups'], r[f'{side}_n'])} & "
                        f"{fmt_stack(r[f'{side}_mean'], r[f'{side}_std'], d)}"
                    )
            rows.append(f"{first} & {demo_label(attr)} & " + " & ".join(cells) + r" \\")
            if i < len(demos) - 1:
                rows.append(r"\addlinespace[2pt]")
        rows.append(r"\midrule")
    caption = (
        f"{method_label(method)}: demographic subgroups at the extremes of buffer "
        f"retention ({sd_phrase(n_seeds)}). Representation ratio: subgroup's share "
        f"of the buffer at the end of task $t$ divided by the share expected from the "
        f"training data of the tasks the buffer holds (1 = proportional). Exposure: "
        f"number of replay draws of the subgroup during the training of task $t$ "
        f"divided by all replay draws in that task. "
        f"Eviction rate: fraction of the subgroup's buffer samples evicted during task "
        f"$t$. $n$: the subgroup's buffer samples (at the end of the task for "
        f"representation, present during the task otherwise). Only subgroups with at "
        f"least {min_buffer} (expected) buffer samples are considered. Unannot.: "
        f"percentage of buffer samples at the end of task $t$ without demographic "
        f"annotation (real faces from sources annotated on the fake class only); "
        f"all other columns use annotated samples only."
    )
    return wrap_table(
        header, rows, "lcl" + "lc" * 6, caption, f"tab:buffer-extremes-{method}"
    )


def fmt_inline(mean: float, std: float, d: int) -> str:
    """Mean with SD as a subscript on the same line."""
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"${mean:.{d}f}$"
    return f"${mean:.{d}f}_{{\\pm {std:.{d}f}}}$"


def table_buffer_dynamics(
    rep_m,
    ret_m,
    task_m,
    method,
    attrs,
    n_tasks,
    n_seeds,
    min_buffer,
    d,
    label_suffix="",
    subset=None,
) -> str:
    """subset: None (all images), 'real', 'fake' or 'unannotated'."""
    """Rows: attribute x subgroup x metric. Cols: tasks 1..n (task boundaries)."""
    src = {
        "rep": rep_m[rep_m["method"] == method].rename(columns={"boundary": "task"}),
        "ret": ret_m[ret_m["method"] == method].rename(columns={"boundary": "task"}),
        "task": task_m[task_m["method"] == method],
    }
    idx = {k: v.set_index(["demographic", "subgroup", "task"]) for k, v in src.items()}
    metrics = list(DYN_METRICS)
    if subset == "unannotated":  # also report how many such samples the buffer holds
        metrics.insert(0, ("Samples ($n$)", "rep", "n", None))
    tasks = list(range(1, n_tasks + 1))
    k = len(tasks)
    header = [
        rf" & & & \multicolumn{{{k}}}{{c}}{{Task}} \\",
        rf"\cmidrule(lr){{4-{k + 3}}}",
        "Attribute & Subgroup & Metric & " + " & ".join(map(str, tasks)) + r" \\",
    ]
    rows = []
    for attr in attrs:
        sgs = set()
        for v in src.values():
            sgs |= set(v.loc[v["demographic"] == attr, "subgroup"].dropna())
        sgs = sorted(sgs, key=natural_key)
        if subset == "unannotated":
            sgs = [g for g in sgs if g == "Unannotated"]
        if not sgs:
            continue
        n_rows = len(sgs) * len(metrics)
        for si, sg in enumerate(sgs):
            for mi, (label, key, col, cnt) in enumerate(metrics):
                cells = []
                for t in tasks:
                    pos = (attr, sg, t)
                    if pos not in idx[key].index:
                        cells.append("--")
                        continue
                    r = idx[key].loc[pos]
                    digits = d
                    if cnt is None:  # sample count: whole number for a single seed
                        digits = 0 if pd.isna(r[f"{col}_std"]) else 1
                    cell = fmt_inline(r[f"{col}_mean"], r[f"{col}_std"], digits)
                    if cell != "--" and cnt and r[f"{cnt}_mean"] < min_buffer:
                        cell += r"$^\dagger$"
                    cells.append(cell)
                a_cell = (
                    f"\\multirow{{{n_rows}}}{{*}}{{{demo_label(attr)}}}"
                    if si == 0 and mi == 0
                    else ""
                )
                s_cell = (
                    f"\\multirow{{{len(metrics)}}}{{*}}{{{tex(sg)}}}" if mi == 0 else ""
                )
                rows.append(
                    f"{a_cell} & {s_cell} & {label} & " + " & ".join(cells) + r" \\"
                )
            if si < len(sgs) - 1:
                rows.append(rf"\cmidrule(lr){{2-{k + 3}}}")
        rows.append(r"\midrule")

    seeds_phrase = (
        f"mean over {n_seeds} seeds, SD as subscript" if n_seeds > 1 else "single seed"
    )
    scope = (
        demo_label(attrs[0]) + " subgroups"
        if len(attrs) == 1
        else "demographic subgroups"
    )
    if subset in CLASSES:
        scope += f" among {subset} images"
    caption = (
        f"{method_label(method)}: replay-buffer dynamics of {scope} at each task "
        f"boundary ({seeds_phrase}). Representation: the "
        f"subgroup's number of buffer samples at the end of task $t$ divided by the "
        f"total buffer size (all samples, including real and fake and those without "
        f"annotation). Retention: fraction of the subgroup's buffer samples at "
        f"the end of task $t-1$ still in the buffer at the end of task $t$. Exposure: "
        f"number of replay draws of the subgroup's samples during the training of "
        f"task $t$ divided by all replay draws in that task (all samples). "
        f"Eviction: fraction of "
        f"the subgroup's buffer samples evicted during the training of task $t$. "
        f"$^\\dagger$: fewer than {min_buffer} buffer samples. Annotated samples only."
    )
    if subset in CLASSES:
        caption += (
            f" Only {subset} images are counted; representation is still "
            f"relative to the total buffer size, so the {subset} and "
            f"{'fake' if subset == 'real' else 'real'} tables add up to the "
            f"combined one; the same holds for exposure, which is relative to "
            f"all replay draws."
        )
    if subset == "unannotated":
        caption = (
            f"{method_label(method)}: replay-buffer dynamics of images without "
            f"demographic annotation (real faces from sources annotated on the fake "
            f"class only) at each task boundary ({seeds_phrase}). Samples: number of "
            f"unannotated samples in the buffer at the end of task $t$. Representation: "
            f"their number of buffer samples at the end of task $t$ divided by the "
            f"total buffer size. Retention: fraction of their buffer samples at the "
            f"end of task $t-1$ still in the buffer at the end of task $t$. Exposure: "
            f"number of replay draws of these samples during the training of task $t$ "
            f"divided by all replay draws in that task. Eviction: fraction "
            f"of their buffer samples evicted during the training of task $t$. "
            f"$^\\dagger$: fewer than {min_buffer} buffer samples. "
            f"--: no unannotated samples in the buffer yet."
        )
    return wrap_table(
        header,
        rows,
        "lll" + "c" * k,
        caption,
        f"tab:buffer-dynamics-{method}{label_suffix}",
    )


def training_composition(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Subgroup counts and shares per task and class, from the training metadata.

    Shares use the annotated images of that class and task as denominator.
    Annotation rows: annotated / unannotated counts and share of all images.
    """
    parts = []
    for cls in CLASSES:
        d = data[cls].copy()
        d["pct"] = 100 * d["data_share"]
        parts.append(d.assign(cls=cls))
    return pd.concat(parts, ignore_index=True)[
        ["cls", "task", "demographic", "subgroup", "n_data", "pct"]
    ]


def table_training_composition(
    comp: pd.DataFrame, attrs, n_tasks, d=1, label_suffix=""
) -> str:
    """Rows: attribute x subgroup x class (real/fake). Cols: tasks 1..n."""
    tasks = list(range(1, n_tasks + 1))
    k = len(tasks)
    idx = comp.set_index(["demographic", "subgroup", "cls", "task"])
    header = [
        rf" & & & \multicolumn{{{k}}}{{c}}{{Task}} \\",
        rf"\cmidrule(lr){{4-{k + 3}}}",
        "Attribute & Subgroup & Class & " + " & ".join(map(str, tasks)) + r" \\",
    ]

    def row_cells(dem, sg, cls, col, fmt):
        out = []
        for t in tasks:
            pos = (dem, sg, cls, t)
            out.append(fmt(idx.at[pos, col]) if pos in idx.index else "--")
        return out

    pct = lambda v: f"${v:.{d}f}$"  # noqa: E731
    rows = []
    for attr in attrs:
        sgs = sorted(
            comp.loc[comp["demographic"] == attr, "subgroup"].unique(), key=natural_key
        )
        n_rows = len(sgs) * len(CLASSES)
        for si, sg in enumerate(sgs):
            for ci, cls in enumerate(CLASSES):
                a_cell = (
                    f"\\multirow{{{n_rows}}}{{*}}{{{demo_label(attr)}}}"
                    if si == 0 and ci == 0
                    else ""
                )
                s_cell = (
                    f"\\multirow{{{len(CLASSES)}}}{{*}}{{{tex(sg)}}}" if ci == 0 else ""
                )
                rows.append(
                    f"{a_cell} & {s_cell} & {cls.capitalize()} & "
                    + " & ".join(row_cells(attr, sg, cls, "pct", pct))
                    + r" \\"
                )
            if si < len(sgs) - 1:
                rows.append(rf"\cmidrule(lr){{2-{k + 3}}}")
        rows.append(r"\midrule")

    # Footer: annotated images (n) and unannotated share (%) per class
    ann = comp[comp["demographic"] == ANNOT_ATTR].pivot_table(
        index=["cls", "task"],
        columns="subgroup",
        values="n_data",
        aggfunc="sum",
        fill_value=0,
    )
    for col in ("Annotated", "Unannotated"):
        if col not in ann:
            ann[col] = 0
    total = ann["Annotated"] + ann["Unannotated"]
    unann_pct = 100 * ann["Unannotated"] / total.where(total > 0)
    for ci, cls in enumerate(CLASSES):
        cells = [
            f"{int(ann.at[(cls, t), 'Annotated']):,}" if (cls, t) in ann.index else "--"
            for t in tasks
        ]
        first = (
            r"\multirow{2}{*}{\makecell[l]{Annotated\\images ($n$)}}" if ci == 0 else ""
        )
        rows.append(f"{first} & & {cls.capitalize()} & " + " & ".join(cells) + r" \\")
    rows.append(rf"\cmidrule(lr){{1-{k + 3}}}")
    for ci, cls in enumerate(CLASSES):
        cells = [
            pct(unann_pct.at[(cls, t)])
            if (cls, t) in unann_pct.index and pd.notna(unann_pct.at[(cls, t)])
            else "--"
            for t in tasks
        ]
        first = (
            r"\multirow{2}{*}{\makecell[l]{Unannotated\\images (\%)}}"
            if ci == 0
            else ""
        )
        rows.append(f"{first} & & {cls.capitalize()} & " + " & ".join(cells) + r" \\")

    scope = (
        demo_label(attrs[0]) + " subgroups"
        if len(attrs) == 1
        else "demographic subgroups"
    )
    caption = (
        f"Composition of the training data: share (\\%) of {scope} among the "
        f"annotated real and fake training images of each task. Shares sum to 100 "
        f"within each attribute, class and task. Real images from sources without "
        f"demographic annotation are excluded from the shares; the bottom rows give "
        f"the number of annotated images and the percentage of images without "
        f"annotation. --: no annotated images of that class in the task."
    )
    return wrap_table(
        header,
        rows,
        "lll" + "c" * k,
        caption,
        f"tab:training-composition{label_suffix}",
    )


def table_correlations(corr, level, ycols, ylabels, n_seeds, d) -> str:
    c = corr[corr["level"] == level]
    methods = list(dict.fromkeys(c["method"]))
    xcols = list(dict.fromkeys(c["x"]))
    header = [
        rf" & & & \multicolumn{{{len(ycols)}}}{{c}}{{Spearman $\rho$ [95\% CI]}} \\",
        rf"\cmidrule(lr){{4-{len(ycols) + 3}}}",
        "Method & Attribute & Buffer metric & " + " & ".join(ylabels) + r" \\",
    ]
    idx = c.set_index(["method", "demographic", "x", "y"])
    rows = []
    for m in methods:
        attrs = [a for a in DEMOGRAPHICS if (m, a) in idx.index.droplevel([2, 3])]
        for ai, attr in enumerate(attrs):
            for xi, x in enumerate(xcols):
                cells = []
                for y in ycols:
                    if (m, attr, x, y) not in idx.index:
                        cells.append("--")
                        continue
                    r = idx.loc[(m, attr, x, y)]
                    if np.isnan(r["rho"]):
                        cells.append("--")
                        continue
                    sig = r["ci_lo"] > 0 or r["ci_hi"] < 0
                    val = (
                        f"$\\mathbf{{{r['rho']:.{d}f}}}$"
                        if sig
                        else f"${r['rho']:.{d}f}$"
                    )  # \\textbf does not bold math
                    cells.append(
                        f"\\makecell{{{val} \\\\ {{\\scriptsize "
                        f"$[{r['ci_lo']:.{d}f}, {r['ci_hi']:.{d}f}]$}}}}"
                    )
                m_cell = (
                    f"\\multirow{{{2 * len(attrs) * len(xcols)}}}{{*}}"
                    f"{{{method_label(m)}}}"
                    if ai == 0 and xi == 0
                    else ""
                )
                a_cell = (
                    f"\\multirow{{{2 * len(xcols)}}}{{*}}{{{demo_label(attr)}}}"
                    if xi == 0
                    else ""
                )
                rows.append(
                    f"{m_cell} & {a_cell} & {tex(UNIT_METRICS.get(x, x))} & "
                    + " & ".join(cells)
                    + r" \\"
                )
            rows.append(r"\cmidrule(lr){2-" + str(len(ycols) + 3) + "}")
        rows[-1] = r"\midrule"
    if level == "subgroup":
        what = (
            "units are (task, subgroup) pairs; outcomes are the subgroup's "
            "forgetting on that task (accuracy after learning it minus final "
            "accuracy) and its final per-subgroup fairness deviation"
        )
        label = "tab:buffer-corr-subgroup"
    else:
        what = (
            "units are tasks; buffer metrics are the range (max $-$ min) across "
            "subgroups, outcomes the change of each fairness metric on that task "
            "from right after learning it to the end of training"
        )
        label = "tab:buffer-corr-task"
    caption = (
        f"Spearman correlation between buffer retention and outcomes; {what}. "
        f"Metrics are averaged over {n_seeds} seeds before correlating; 95\\% CIs "
        f"resample tasks. Bold: CI excludes 0. The final task is excluded."
    )
    return wrap_table(header, rows, "lll" + "c" * len(ycols), caption, label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Per-run cache
# ---------------------------------------------------------------------------


def file_stamp(path: Path) -> list | None:
    """Size and modification time: changes whenever the file is rewritten."""
    if not path.exists():
        return None
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def run_fingerprint(sdir: Path, args) -> str:
    key = {
        "version": CACHE_VERSION,
        "args": {a: getattr(args, a) for a in CACHED_ARGS},
        "inputs": {
            name: file_stamp(sdir / name) for name in ("events.csv", "sampling.csv")
        },
        "metadata": [str(args.metadata.resolve()), file_stamp(args.metadata)],
        "task_names": TASK_NAMES,
        "unannotated_sources": UNANNOTATED_SOURCES,
    }
    blob = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def cache_paths(cache_dir: Path, folder: str, seed: str) -> tuple[Path, Path]:
    base = cache_dir / folder / f"seed_run_{seed}"
    return base.with_suffix(".pkl"), base.with_suffix(".json")


def load_cached(cache_dir, folder, seed, fingerprint) -> dict | None:
    pkl, meta = cache_paths(cache_dir, folder, seed)
    if not (pkl.exists() and meta.exists()):
        return None
    try:
        if json.loads(meta.read_text()).get("fingerprint") != fingerprint:
            return "stale"
        return pd.read_pickle(pkl)
    except Exception as err:  # corrupt or unreadable cache: recompute
        logger.warning("{}: cache unreadable ({}); recomputing", pkl, err)
        return None


def save_cached(cache_dir, folder, seed, fingerprint, results: dict) -> None:
    pkl, meta = cache_paths(cache_dir, folder, seed)
    pkl.parent.mkdir(parents=True, exist_ok=True)
    tmp = pkl.with_suffix(".pkl.tmp")
    pd.to_pickle(results, tmp)
    tmp.replace(pkl)  # atomic: never leaves a half-written cache
    meta.write_text(json.dumps({"fingerprint": fingerprint}, indent=2))


def data_hash(*parts) -> str:
    """Stable hash of DataFrames and plain values (for downstream caches)."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, pd.DataFrame):
            h.update(json.dumps(list(map(str, part.columns))).encode())
            h.update(pd.util.hash_pandas_object(part, index=False).to_numpy().tobytes())
        else:
            h.update(json.dumps(part, sort_keys=True, default=str).encode())
    return h.hexdigest()


class StepCache:
    """Reuse a downstream step (correlations, figures) when its inputs match."""

    def __init__(self, cache_dir: Path, name: str, key: str, enabled: bool):
        self.meta = cache_dir / f"{name}.json"
        self.pkl = cache_dir / f"{name}.pkl"
        self.key, self.enabled = key, enabled

    def valid(self, outputs: list[Path] = ()) -> bool:
        if not (self.enabled and self.meta.exists()):
            return False
        try:
            same = json.loads(self.meta.read_text()).get("key") == self.key
        except Exception:
            return False
        return same and all(p.exists() for p in outputs)

    def load(self):
        return pd.read_pickle(self.pkl)

    def store(self, value=None) -> None:
        if not self.enabled:
            return
        self.meta.parent.mkdir(parents=True, exist_ok=True)
        if value is not None:
            tmp = self.pkl.with_suffix(".pkl.tmp")
            pd.to_pickle(value, tmp)
            tmp.replace(self.pkl)
        self.meta.write_text(json.dumps({"key": self.key}, indent=2))


def load_metadata_cached(args, cache_dir: Path) -> dict[str, pd.DataFrame]:
    """Metadata counts, reused while the metadata file and settings are unchanged."""
    step = StepCache(
        cache_dir,
        "metadata",
        data_hash(
            CACHE_VERSION,
            str(args.metadata.resolve()),
            file_stamp(args.metadata),
            TASK_NAMES,
            UNANNOTATED_SOURCES,
            LABEL_CODES,
        ),
        enabled=not (args.no_cache or args.recompute),
    )
    if step.valid([step.pkl]):
        logger.info("Metadata: reusing cached counts")
        return step.load()
    data = load_metadata(args.metadata)
    step.store(data)
    return data


def add_buffer_ratio(rep: pd.DataFrame, buffer_size: dict) -> pd.DataFrame:
    """Subgroup's buffer samples / total buffer size at the boundary."""
    rep = rep.copy()
    rep["buffer_size"] = rep["boundary"].map(buffer_size)
    rep["buffer_ratio"] = safe_div(rep["n"], rep["buffer_size"])
    return rep


def add_draw_share(tm: pd.DataFrame, total_draws: dict) -> pd.DataFrame:
    """Subgroup's replay draws / all replay draws during the task."""
    tm = tm.copy()
    tm["total_draws"] = tm["task"].map(total_draws).fillna(0)
    tm["draw_share"] = safe_div(tm["draws"], tm["total_draws"])
    tm["exposure"] = tm["draw_share"]  # exposure = draws / all draws in the task
    return tm


def analyse_run(sdir, folder, seed, args, data) -> dict:
    """All per-run buffer statistics (the expensive part)."""
    tag = f"{folder}/seed_run_{seed}"
    # Per-run RNG: results do not depend on which other runs are analysed
    rng = np.random.default_rng([args.seed, zlib.crc32(tag.encode())])
    ev, sp = load_run(sdir, tag, args.n_tasks, args.draws)
    iv = build_intervals(ev, tag)
    lifecycle_check(sdir, iv, tag)
    first, last = task_bounds(ev, sp)
    tasks = sorted(first)
    final = args.n_tasks
    logger.info(
        "{}: {} residencies, {} replay draws, tasks {}..{}",
        tag,
        len(iv),
        len(sp),
        tasks[0],
        tasks[-1],
    )

    rep, detail = representation(iv, data["all"], last, tasks, rng, args.n_null)
    # Total buffer size at each task boundary (all samples: any class, annotated
    # or not); denominator of the buffer representation ratio
    buffer_size = {b: int(resident_at_end(iv, b, last).sum()) for b in tasks}
    rep = add_buffer_ratio(rep, buffer_size)
    # All replay draws per task (any class, annotated or not); denominator of the
    # replay share
    total_draws = sp.groupby("task").size().to_dict()
    dyn = dynamics(iv, sp, first, last, tasks, tag)
    coh, km = cohort_and_km(iv, last, tasks, final)
    res = {
        "rep": rep,
        "rep_detail": detail,
        "task": add_draw_share(task_metrics(dyn), total_draws),
        "ret": retention(iv, last, tasks),
        "rvr": retention_vs_replay(dyn),
        "coh": coh,
        "km": km,
        "units": unit_metrics(dyn, iv, detail, final, total_draws),
    }
    # Same statistics within each class: shares are relative to that class's
    # buffer samples, replay draws and training data (no chance band).
    unknown = int(iv["cls"].isna().sum())
    if unknown:
        logger.warning(
            "{}: {} residencies with no real/fake class (path or label); "
            "left out of the class tables",
            tag,
            unknown,
        )
    for cls in CLASSES:
        iv_c = iv[iv["cls"] == cls].reset_index(drop=True)
        sp_c = sp[sp["cls"] == cls]
        if iv_c.empty:
            logger.warning("{}: no {} samples in the buffer", tag, cls)
            continue
        rep_c, _ = representation(iv_c, data[cls], last, tasks, rng, 0)
        res[f"rep_{cls}"] = add_buffer_ratio(rep_c, buffer_size)
        res[f"task_{cls}"] = add_draw_share(
            task_metrics(dynamics(iv_c, sp_c, first, last, tasks, tag)), total_draws
        )
        res[f"ret_{cls}"] = retention(iv_c, last, tasks)
    return res


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    csv_dir, tex_dir = args.out_dir / "csv", args.out_dir / "tex"
    for d in (csv_dir, tex_dir):
        d.mkdir(parents=True, exist_ok=True)

    cache_dir = args.out_dir / "cache"
    force = {r.lower() for r in (args.recompute_runs or [])}
    data = load_metadata_cached(args, cache_dir)
    wanted = {m.lower() for m in args.methods} if args.methods else None
    runs = []
    for mdir in sorted(p for p in args.log_root.iterdir() if p.is_dir()):
        if wanted and mdir.name.lower() not in wanted:
            continue
        for sdir in sorted(mdir.glob("seed_run_*")):
            seed = sdir.name.removeprefix("seed_run_")
            if args.seeds and seed not in args.seeds:
                continue
            if (sdir / "events.csv").exists():
                runs.append((mdir.name, seed, sdir))
    if not runs:
        raise FileNotFoundError(f"No events.csv under {args.log_root}")

    out = {k: [] for k in (*RUN_TABLES, "tunits")}
    n_cached = 0
    for folder, seed, sdir in runs:
        method, tag = folder.lower(), f"{folder}/seed_run_{seed}"
        final = args.n_tasks
        fingerprint = run_fingerprint(sdir, args)
        forced = args.recompute or f"{folder}/{seed}".lower() in force
        res = None
        if not (forced or args.no_cache):
            res = load_cached(cache_dir, folder, seed, fingerprint)
            if isinstance(res, str):
                logger.info(
                    "{}: inputs or settings changed since the cached run; recomputing",
                    tag,
                )
                res = None
            elif res is not None:
                logger.info("{}: reusing cached results (--recompute to refresh)", tag)
                n_cached += 1
        if res is None:
            res = analyse_run(sdir, folder, seed, args, data)
            if not args.no_cache:
                save_cached(cache_dir, folder, seed, fingerprint, res)

        meta = dict(method=method, seed=seed)
        for k in RUN_TABLES:
            if k != "units" and k in res:
                out[k].append(res[k].assign(**meta))
        units = res["units"]

        sub, task_lvl = load_fairness(
            args.fairness_root, method, seed, final, args.min_n
        )
        if sub is not None:
            u = units[units["n_buffer"] >= args.min_buffer]
            su = u.merge(sub, on=["origin_task", "demographic", "subgroup"])
            if su.empty:
                logger.warning(
                    "{}: no subgroup overlap between buffer and fairness "
                    "files (check subgroup codes)",
                    tag,
                )
            out["units"].append(su.assign(**meta))
            rng_ = u.groupby(["origin_task", "demographic"])[list(UNIT_METRICS)].agg(
                lambda s: s.max() - s.min()
            )
            out["tunits"].append(
                rng_.reset_index()
                .merge(task_lvl, on=["origin_task", "demographic"])
                .assign(**meta)
            )

    logger.info("{} of {} runs reused from cache", n_cached, len(runs))
    tables = {k: pd.concat(v, ignore_index=True) for k, v in out.items() if v}
    methods = [m for m in ("derpp", "gradmatch") if m in set(tables["rep"]["method"])]
    methods += sorted(set(tables["rep"]["method"]) - set(methods))
    n_seeds = int(tables["rep"].groupby("method")["seed"].nunique().min())

    agg = {
        "representation": over_seeds(
            tables["rep"], ["method", "boundary", "demographic", "subgroup"]
        ),
        "representation_by_origin": over_seeds(
            tables["rep_detail"],
            ["method", "boundary", "origin_task", "demographic", "subgroup"],
        ),
        "replay_and_eviction": over_seeds(
            tables["task"], ["method", "task", "demographic", "subgroup"]
        ),
        "retention": over_seeds(
            tables["ret"], ["method", "boundary", "demographic", "subgroup"]
        ),
        "retention_vs_replay": over_seeds(
            tables["rvr"], ["method", "demographic", "subgroup"]
        ),
        "cohort_survival": over_seeds(
            tables["coh"],
            ["method", "demographic", "subgroup", "cohort_task", "boundary"],
        ),
        "lifetime_km": over_seeds(
            tables["km"], ["method", "demographic", "subgroup", "boundaries"]
        ),
    }
    for cls in CLASSES:  # per-class dynamics
        if f"rep_{cls}" not in tables:
            continue
        agg[f"representation_{cls}"] = over_seeds(
            tables[f"rep_{cls}"], ["method", "boundary", "demographic", "subgroup"]
        )
        agg[f"replay_and_eviction_{cls}"] = over_seeds(
            tables[f"task_{cls}"], ["method", "task", "demographic", "subgroup"]
        )
        agg[f"retention_{cls}"] = over_seeds(
            tables[f"ret_{cls}"], ["method", "boundary", "demographic", "subgroup"]
        )
    rep_m = agg["representation"]
    rep_m["outside_chance"] = (rep_m["ratio_mean"] < rep_m["chance_lo_mean"]) | (
        rep_m["ratio_mean"] > rep_m["chance_hi_mean"]
    )

    # CSVs
    per_seed = {
        "representation": "rep",
        "representation_by_origin": "rep_detail",
        "replay_and_eviction": "task",
        "retention": "ret",
        "retention_vs_replay": "rvr",
        **{
            f"{name}_{c}": f"{key}_{c}"
            for c in CLASSES
            for name, key in (
                ("representation", "rep"),
                ("replay_and_eviction", "task"),
                ("retention", "ret"),
            )
            if f"{key}_{c}" in tables
        },
        "cohort_survival": "coh",
        "lifetime_km": "km",
    }
    for name, key in per_seed.items():
        tables[key].to_csv(csv_dir / f"{name}_per_seed.csv", index=False)
        agg[name].to_csv(csv_dir / f"{name}.csv", index=False)

    # Step 5 (bootstrap is slow: reused when its inputs are unchanged)
    corr_cache = StepCache(
        cache_dir,
        "correlations",
        data_hash(
            CACHE_VERSION,
            args.n_boot,
            args.seed,
            tables.get("units", pd.DataFrame()),
            tables.get("tunits", pd.DataFrame()),
        ),
        enabled=not (args.no_cache or args.recompute),
    )
    if corr_cache.valid([corr_cache.pkl]):
        logger.info("Correlations: reusing cached results")
        corr = corr_cache.load()
    else:
        corr = []
        if "units" in tables:
            ycols = ["forgetting", *FAIR_COLS]
            corr.append(
                correlate(
                    tables["units"],
                    ["origin_task", "subgroup"],
                    list(UNIT_METRICS),
                    ycols,
                    args.n_boot,
                    rng,
                    "subgroup",
                )
            )
        if "tunits" in tables:
            ycols_t = [f"delta_{c}" for c in FAIR_COLS]
            corr.append(
                correlate(
                    tables["tunits"],
                    ["origin_task"],
                    list(UNIT_METRICS),
                    ycols_t,
                    args.n_boot,
                    rng,
                    "task",
                )
            )
        corr = pd.concat(corr, ignore_index=True) if corr else pd.DataFrame()
        corr_cache.store(corr)
    if "units" in tables:
        tables["units"].to_csv(csv_dir / "correlation_units_subgroup.csv", index=False)
    if "tunits" in tables:
        tables["tunits"].to_csv(csv_dir / "correlation_units_task.csv", index=False)
    if not corr.empty:
        corr.to_csv(csv_dir / "correlations.csv", index=False)
    logger.info("Wrote CSVs to {}", csv_dir)

    # LaTeX
    for m in methods:
        (tex_dir / f"buffer_extremes_{m}.tex").write_text(
            table_buffer_extremes(
                rep_m,
                agg["replay_and_eviction"],
                m,
                n_seeds,
                args.min_buffer,
                args.digits,
            )
        )
        dyn_args = (rep_m, agg["retention"], agg["replay_and_eviction"], m)
        (tex_dir / f"buffer_dynamics_{m}.tex").write_text(
            table_buffer_dynamics(
                *dyn_args,
                DEMOGRAPHICS,
                args.n_tasks,
                n_seeds,
                args.min_buffer,
                args.digits,
            )
        )
        for attr in DEMOGRAPHICS:  # split version: one table per attribute
            slug = attr.lower().replace(" ", "_")
            (tex_dir / f"buffer_dynamics_{m}_{slug}.tex").write_text(
                table_buffer_dynamics(
                    *dyn_args,
                    [attr],
                    args.n_tasks,
                    n_seeds,
                    args.min_buffer,
                    args.digits,
                    label_suffix=f"-{slug.replace('_', '-')}",
                )
            )
        for cls in CLASSES:  # same table within real / fake images
            if f"representation_{cls}" not in agg:
                continue
            cls_args = (
                agg[f"representation_{cls}"],
                agg[f"retention_{cls}"],
                agg[f"replay_and_eviction_{cls}"],
                m,
            )
            if not (cls_args[0]["method"] == m).any():
                continue
            (tex_dir / f"buffer_dynamics_{m}_{cls}.tex").write_text(
                table_buffer_dynamics(
                    *cls_args,
                    DEMOGRAPHICS,
                    args.n_tasks,
                    n_seeds,
                    args.min_buffer,
                    args.digits,
                    label_suffix=f"-{cls}",
                    subset=cls,
                )
            )
        # Images without demographic annotation (one block, all samples)
        (tex_dir / f"buffer_dynamics_{m}_unannotated.tex").write_text(
            table_buffer_dynamics(
                *dyn_args,
                [ANNOT_ATTR],
                args.n_tasks,
                n_seeds,
                args.min_buffer,
                args.digits,
                label_suffix="-unannotated",
                subset="unannotated",
            )
        )

    # Training-data composition (metadata only; same for every method and seed)
    comp = training_composition(data)
    comp.to_csv(csv_dir / "training_composition.csv", index=False)
    (tex_dir / "training_composition.tex").write_text(
        table_training_composition(comp, DEMOGRAPHICS, args.n_tasks)
    )
    for attr in DEMOGRAPHICS:  # split version: one table per attribute
        slug = attr.lower().replace(" ", "_")
        (tex_dir / f"training_composition_{slug}.tex").write_text(
            table_training_composition(
                comp, [attr], args.n_tasks, label_suffix=f"-{slug.replace('_', '-')}"
            )
        )
    if not corr.empty:
        if (corr["level"] == "subgroup").any():
            (tex_dir / "buffer_correlations_subgroup.tex").write_text(
                table_correlations(
                    corr,
                    "subgroup",
                    ["forgetting", *FAIR_COLS],
                    ["Forgetting", *FAIR_COLS],
                    n_seeds,
                    args.digits,
                )
            )
        if (corr["level"] == "task").any():
            (tex_dir / "buffer_correlations_task.tex").write_text(
                table_correlations(
                    corr,
                    "task",
                    [f"delta_{c}" for c in FAIR_COLS],
                    [f"$\\Delta${c}" for c in FAIR_COLS],
                    n_seeds,
                    args.digits,
                )
            )
    logger.info("Wrote LaTeX tables to {}", tex_dir)

    logger.success(
        "Done: {}. Figures: python plot_buffer_heatmaps.py --analysis-dir {}",
        args.out_dir,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
