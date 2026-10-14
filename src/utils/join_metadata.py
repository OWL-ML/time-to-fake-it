#!/usr/bin/env python3
"""
Consolidate per-task metadata CSVs into one CSV per split.

Reads metadata/<split>/<task>.csv and writes metadata/<split>.csv, the files
the training and analysis scripts read (utils.constants.METADATA_CSV).
run_training.sh and run_fisher.sh call this automatically when they are
missing.

Real images from sources without a demographic annotation
(utils.constants.UNANNOTATED_SOURCES: casia, celeba, metfaces) carry
placeholder 0s in the per-task CSVs; their annotation fields are set to -1
(missing) here so they are never counted as a real subgroup.

Changes vs. the previous version:
  - `path` is now a normalized SUFFIX key (last --key-depth components of
    the image path, forward-slashed), not a bare basename. Basenames
    collide across generators and across real/fake; suffixes don't.
  - Duplicate keys are NEVER silently dropped. Collisions are reported
    per (task, key) and written to a collisions CSV; --strict makes them
    fatal. A dropped row is worse than a duplicate: it either becomes a
    missing-demographics sample or, worse, a WRONG match to another
    task's surviving row.
  - Skipped input files (no path-like column) are collected and printed
    as a summary at the end, so a missing task is impossible to overlook
    in scrollback.

Example:
  uv run python src/utils/join_metadata.py \
    --metadata-root metadata --output-dir metadata
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from utils.constants import ANNOTATION_FIELDS, UNANNOTATED_SOURCES

# Annotation code for "no demographic annotation"
MISSING_CODE = -1

PATH_CANDIDATES = ["path", "image_path", "filepath", "file_path"]


def _choose_path_column(df: pd.DataFrame) -> str | None:
    for c in PATH_CANDIDATES:
        if c in df.columns:
            return c
    return None


def suffix_key(v: object, depth: int) -> str:
    s = "" if pd.isna(v) else str(v)
    parts = [p for p in s.replace("\\", "/").split("/") if p]
    return "/".join(parts[-depth:])


def consolidate_split(
    split_dir: Path, split_name: str, key_depth: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    csv_files = sorted(split_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {split_dir}")

    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        path_col = _choose_path_column(df)
        if path_col is None:
            skipped.append(str(fp))
            print(
                f"[WARN] {fp}: no path-like column found, SKIPPING "
                f"(this task will have NO demographics)"
            )
            continue
        out = df.copy()
        # Preserve the original full path under a stable name so the
        # matcher can build its own suffix tables from it.
        out["source_image_path"] = out[path_col].astype(str)
        out["path"] = out[path_col].map(lambda v: suffix_key(v, key_depth))
        out["basename"] = out[path_col].map(lambda v: suffix_key(v, 1))
        out["metadata_split"] = split_name
        out["metadata_task"] = fp.stem
        out["metadata_source_file"] = str(fp)
        frames.append(out)

    if not frames:
        raise RuntimeError(f"No usable CSVs for split={split_name} in {split_dir}")

    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    n_empty = int((merged["path"].astype(str).str.len() == 0).sum())
    if n_empty:
        print(f"[WARN] split={split_name}: {n_empty} rows have empty `path` key")

    # Collision report — kept, not dropped.
    dup_keys = merged.loc[merged.duplicated(subset=["path"], keep=False)]
    collisions = pd.DataFrame()
    if len(dup_keys):
        collisions = (
            dup_keys.groupby("path")
            .agg(
                n=("path", "size"),
                tasks=("metadata_task", lambda s: ";".join(sorted(set(s)))),
            )
            .reset_index()
            .sort_values("n", ascending=False)
        )
        by_task = dup_keys["metadata_task"].value_counts()
        print(
            f"[WARN] split={split_name}: {len(dup_keys)} rows share a "
            f"non-unique key at depth {key_depth} "
            f"({collisions['n'].gt(0).sum()} distinct keys). "
            f"Rows involved per task: {by_task.to_dict()}. "
            f"Consider a larger --key-depth."
        )

    return merged, collisions, skipped


def mark_unannotated(merged: pd.DataFrame) -> int:
    """Set the annotation fields of UNANNOTATED_SOURCES rows to MISSING_CODE.
    Returns the number of rows marked."""
    unannotated = merged["source"].astype(str).str.lower().isin(UNANNOTATED_SOURCES)
    cols = [c for c in ANNOTATION_FIELDS if c in merged.columns]
    merged.loc[unannotated, cols] = MISSING_CODE
    return int(unannotated.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata-root", type=Path, default=Path("metadata"))
    p.add_argument("--output-dir", type=Path, default=Path("metadata"))
    p.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    p.add_argument(
        "--key-depth",
        type=int,
        default=4,
        help="path components in the join key, e.g. 4 -> "
        "'StableDiffusion/train/fake/0001.png'",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on key collisions or skipped files",
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    any_problem = False
    for split in args.splits:
        merged, collisions, skipped = consolidate_split(
            args.metadata_root / split, split, args.key_depth
        )
        n_marked = mark_unannotated(merged)
        out_path = args.output_dir / f"{split}.csv"
        merged.to_csv(out_path, index=False)
        print(f"[OK] {split}: {n_marked:,} unannotated rows marked {MISSING_CODE}")
        print(f"[OK] {split}: wrote {len(merged):,} rows -> {out_path}")
        print(
            f"[OK] {split}: rows per task: "
            f"{merged['metadata_task'].value_counts().to_dict()}"
        )

        if len(collisions):
            cpath = args.output_dir / f"{split}_key_collisions.csv"
            collisions.to_csv(cpath, index=False)
            print(f"[WARN] {split}: collision report -> {cpath}")
            any_problem = True
        if skipped:
            print(f"[WARN] {split}: SKIPPED files (no path column): {skipped}")
            any_problem = True

    if args.strict and any_problem:
        sys.exit(1)


if __name__ == "__main__":
    main()
