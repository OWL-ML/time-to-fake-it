from collections import Counter, defaultdict

import numpy as np
import pandas as pd

_AMBIGUOUS = object()  # sentinel: key exists in CSV but maps to >1 row


def _infer_task_from_sample_path(sample_path: str) -> str:
    """Best-effort task name extraction from a sample path."""
    parts = sample_path.replace("\\", "/").split("/")
    for split_name in ("train", "val", "test", "analysis"):
        if split_name in parts:
            i = parts.index(split_name)
            if i > 0:
                return parts[i - 1]
    for p in parts:
        if p:
            return p
    return "<unknown>"


def _suffix(path: str, k: int) -> str:
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    return "/".join(parts[-k:])


def demographics_from_csv(
    samples: list[tuple],
    csv_path: str,
    path_col: str = None,
    demo_cols: list[str] = None,
    max_suffix_depth: int = 4,
) -> dict:
    """
    Load demographics from a CSV and align row-for-row with samples.

    Matching is hierarchical suffix matching on path components: for each
    sample, try the last `max_suffix_depth` components, then depth-1, ...
    down to the bare basename. A depth only matches if the key is UNIQUE
    in the CSV at that depth; ambiguous keys fall through to nothing
    (never to an arbitrary row). Deeper matches win, so
    'StableDiffusion/train/fake/0001.png' can never be satisfied by
    'StyleGAN2/train/fake/0001.png'.

    Path column preference: source_image_path (written by the fixed
    consolidation script, holds the ORIGINAL full path), then path,
    image_path, filepath, file_path.

    Failure accounting distinguishes:
      absent    — no key at any depth exists in the CSV
                  (task missing from consolidation, or path format drift)
      ambiguous — keys exist but are non-unique even at max depth
                  (raise --key-depth / max_suffix_depth, or the CSV
                  genuinely contains duplicate rows)
    """
    meta_df = pd.read_csv(csv_path)

    candidates = (
        [path_col]
        if path_col
        else ["source_image_path", "path", "image_path", "filepath", "file_path"]
    )
    for cand in candidates:
        if cand and cand in meta_df.columns:
            path_col = cand
            break
    else:
        raise KeyError(
            f"No path column found in {csv_path}. Expected one of: {candidates}"
        )

    reserved = {
        path_col,
        "path",
        "basename",
        "source_image_path",
        "metadata_split",
        "metadata_task",
        "metadata_source_file",
    }
    if demo_cols is None:
        demo_cols = [c for c in meta_df.columns if c not in reserved]
    missing_cols = [c for c in demo_cols if c not in meta_df.columns]
    if missing_cols:
        raise KeyError(
            f"Requested demographic columns not found in {csv_path}: {missing_cols}"
        )

    # Build one lookup table per suffix depth: key -> row position,
    # or _AMBIGUOUS if the key maps to multiple rows at that depth.
    csv_paths = meta_df[path_col].astype(str).tolist()
    tables: list[dict] = []
    for k in range(max_suffix_depth, 0, -1):
        table: dict = {}
        for pos, p in enumerate(csv_paths):
            key = _suffix(p, k)
            if key in table:
                table[key] = _AMBIGUOUS
            else:
                table[key] = pos
        tables.append(table)

    col_values = {c: meta_df[c].to_numpy() for c in demo_cols}
    result = {col: [] for col in demo_cols}

    matched_at_depth = Counter()
    absent = ambiguous = 0
    absent_by_task = defaultdict(int)
    ambiguous_by_task = defaultdict(int)

    for path, _ in samples:
        sample_path = str(path)
        pos = None
        saw_ambiguous = False
        for depth_idx, k in enumerate(range(max_suffix_depth, 0, -1)):
            hit = tables[depth_idx].get(_suffix(sample_path, k))
            if hit is _AMBIGUOUS:
                saw_ambiguous = True
                continue
            if hit is not None:
                pos = hit
                matched_at_depth[k] += 1
                break

        if pos is not None:
            for col in demo_cols:
                result[col].append(col_values[col][pos])
        else:
            task_name = _infer_task_from_sample_path(sample_path)
            if saw_ambiguous:
                ambiguous += 1
                ambiguous_by_task[task_name] += 1
            else:
                absent += 1
                absent_by_task[task_name] += 1
            for col in demo_cols:
                result[col].append(None)

    n = len(samples)
    if absent or ambiguous:
        print(
            f"[WARNING] in {csv_path}: {absent + ambiguous}/{n} samples "
            f"missing demographics ({absent} absent from CSV, "
            f"{ambiguous} ambiguous keys)"
        )
        if absent_by_task:
            s = ", ".join(
                f"{t}={c}"
                for t, c in sorted(absent_by_task.items(), key=lambda kv: -kv[1])
            )
            print(
                f"[WARNING] absent by task (task missing from "
                f"consolidation or path drift): {s}"
            )
        if ambiguous_by_task:
            s = ", ".join(
                f"{t}={c}"
                for t, c in sorted(ambiguous_by_task.items(), key=lambda kv: -kv[1])
            )
            print(f"[WARNING] ambiguous by task (raise key depth): {s}")
    if matched_at_depth:
        print(
            f"[INFO] demographic matches by suffix depth: "
            f"{dict(sorted(matched_at_depth.items(), reverse=True))}"
        )

    return {k: np.array(v, dtype=object) for k, v in result.items()}


class DemographicsMixin:
    """
    Shared demographic + path lookup for ImageFolder-based datasets.
    Requires self.samples (list of (path, target)) and optionally
    self.demographics (dict of {attr: np.ndarray} parallel to samples).
    """

    demographics: dict = None

    def get_demographics(self, idx: int) -> dict:
        if self.demographics is None:
            return {}
        return {k: v[idx] for k, v in self.demographics.items()}

    @property
    def path_at(self):
        """Access sample path by index: dataset.path_at(idx)"""
        return lambda idx: self.samples[idx][0]

    def attach_demographics(self, demo_source, demo_attrs=None):
        """
        Attach demographics from either:
          - a callable path -> dict (path-encoded demographics)
          - a CSV path (str) with a path column
        """
        if callable(demo_source):
            records = [demo_source(p) for p, _ in self.samples]
            attrs = records[0].keys()
            self.demographics = {
                attr: np.array([r[attr] for r in records]) for attr in attrs
            }
        elif isinstance(demo_source, str):
            self.demographics = demographics_from_csv(
                self.samples, demo_source, demo_cols=demo_attrs
            )
        else:
            raise TypeError(
                f"demo_source must be callable or CSV path, got {type(demo_source)}"
            )
