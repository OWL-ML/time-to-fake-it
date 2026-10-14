"""Demographic schema, label maps and ordering helpers.

Single source of truth for the demographic columns, how their integer
codes map to labels, and how groups are ordered in tables and figures.
Import this from the trainers and the analysis scripts rather than
re-declaring anything.

The dataset stores FIVE columns, and two of them are about skin tone —
keeping them apart matters:

    Predicted Gender    2 levels
    Predicted Age       5 levels (ordinal)
    Skin Tone          10 levels (ordinal)   <- the fine scale
    Skin Tone Group     3 levels (ordinal)   <- Light / Medium / Dark bins
    Intersection        6 levels             = gender x Skin Tone GROUP

So `skin_tone` and `skin_tone_group` are DIFFERENT attributes, not
aliases. A lookup that treats "Skin Tone" as a synonym for
skin_tone_group silently records the 10-level value under the 3-level
name, which reads as a plausible number and corrupts every downstream
grouping. ATTR_CANDIDATES therefore lists exact names only.

Values arrive as INTEGER CODES (e.g. {'Predicted Gender': 0,
'Skin Tone': 5, 'Skin Tone Group': 1, 'Intersection': 1}), so grouping
works on the codes and labels are applied at analysis time via
apply_labels().
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

# Canonical attribute names used throughout the codebase.
ATTRIBUTES = (
    "predicted_gender",
    "predicted_age",
    "skin_tone",
    "skin_tone_group",
    "intersection",
)

SUBGROUP_LABELS: Dict[str, Dict[int, str]] = {
    "predicted_gender": {0: "Female", 1: "Male"},
    "predicted_age": {
        0: "Child",
        1: "Youth",
        2: "Adult",
        3: "Middle-Aged",
        4: "Senior",
    },
    "skin_tone": {i: f"Skin Tone {i}" for i in range(1, 11)},
    "skin_tone_group": {0: "Light", 1: "Medium", 2: "Dark"},
    "intersection": {
        0: "Female, Light",
        1: "Female, Medium",
        2: "Female, Dark",
        3: "Male, Light",
        4: "Male, Medium",
        5: "Male, Dark",
    },
}

# Exact column names only — no cross-attribute fallbacks. Order is
# most-specific first; the first key present in the dict wins.
ATTR_CANDIDATES: Dict[str, List[str]] = {
    "predicted_gender": ["predicted_gender", "Predicted Gender", "Gender", "gender"],
    "predicted_age": ["predicted_age", "Predicted Age", "Age", "age"],
    # NB: "Skin Tone" belongs to skin_tone ONLY. It must never appear as a
    # candidate for skin_tone_group.
    "skin_tone": ["skin_tone", "Skin Tone"],
    "skin_tone_group": ["skin_tone_group", "Skin Tone Group"],
    "intersection": ["intersection", "Intersection"],
}

# Dataset column name -> canonical attribute (for DataFrame renaming).
COLUMN_ALIASES = {
    orig: attr for attr, names in ATTR_CANDIDATES.items() for orig in names
}

# Display order per attribute (ordinal where meaningful).
GROUP_ORDER: Dict[str, List[str]] = {
    attr: [labels[k] for k in sorted(labels)]
    for attr, labels in SUBGROUP_LABELS.items()
}

# Attributes with a meaningful ordering. Enables trend tests (Spearman of
# group index vs metric) rather than only across-group rank correlation —
# skin_tone with 10 ordered levels is the best-powered of these.
ORDINAL_ATTRS = ("skin_tone", "skin_tone_group", "predicted_age")

# Intersection is a pure function of the other two: gender x 3-bin group.
INTERSECTION_PARTS = ("predicted_gender", "skin_tone_group")
_N_TONE_GROUPS = 3


def demo_value(demo: Dict, attribute: str):
    """Read one attribute from a demographics dict.

    Matches only that attribute's own column names, so the two skin-tone
    columns can never stand in for one another. Returns None when no
    variant is present — check for that rather than letting every sample
    collapse into a single None group.
    """
    for key in ATTR_CANDIDATES.get(attribute, [attribute]):
        if key in demo:
            return demo[key]
    return None


def label_of(attribute: str, value) -> Optional[str]:
    """Map an integer code to its label; pass through already-labelled
    values and unknown codes unchanged."""
    labels = SUBGROUP_LABELS.get(attribute, {})
    if value in labels:
        return labels[value]
    try:
        return labels.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def group_index(attribute: str, label) -> Optional[int]:
    """Position of a group in its attribute's order (for trend tests)."""
    order = GROUP_ORDER.get(attribute)
    if not order:
        return None
    s = str(label)
    return order.index(s) if s in order else None


def expected_intersection(gender_code: int, tone_group_code: int) -> int:
    """Intersection code implied by gender and the 3-bin tone group.

    Derived from the label maps: codes 0-2 are Female x Light/Medium/Dark
    and 3-5 are Male x the same, i.e. gender * 3 + tone_group.
    """
    return int(gender_code) * _N_TONE_GROUPS + int(tone_group_code)


def check_intersection_consistency(demo: Dict) -> Optional[str]:
    """Verify Intersection agrees with Gender x Skin Tone Group.

    Returns None when consistent (or when a part is missing), else a
    description of the mismatch. Worth running over a sample of each
    split: a disagreement means the annotation pipeline's binning does
    not match the label map assumed here, which would silently mislabel
    every intersectional result.
    """
    g = demo_value(demo, "predicted_gender")
    t = demo_value(demo, "skin_tone_group")
    i = demo_value(demo, "intersection")
    if g is None or t is None or i is None:
        return None
    try:
        exp = expected_intersection(g, t)
    except (TypeError, ValueError):
        return None
    if int(i) != exp:
        return (
            f"Intersection={i} but gender={g} x tone_group={t} implies "
            f"{exp} ({label_of('intersection', exp)})"
        )
    return None


def apply_labels(
    df: pd.DataFrame, attr_col: str = "attribute", group_col: str = "group"
) -> pd.DataFrame:
    """Label the group column and make it an ordered Categorical.

    Works on the long-format tables the analysis scripts emit
    ([task, attribute, group, ...]). Groups are labelled per attribute
    and ordered by GROUP_ORDER, so groupby/plot output follows the
    ordinal scale rather than sorting "Skin Tone 10" before
    "Skin Tone 2". Also adds ``group_index`` for ordinal trend tests.
    """
    out = df.copy()
    out[group_col] = [label_of(a, g) for a, g in zip(out[attr_col], out[group_col])]
    out["group_index"] = [
        group_index(a, g) for a, g in zip(out[attr_col], out[group_col])
    ]
    pieces = []
    for attr, chunk in out.groupby(attr_col, sort=False):
        order = [g for g in GROUP_ORDER.get(attr, []) if g in set(chunk[group_col])]
        if order:
            chunk = chunk.copy()
            chunk[group_col] = pd.Categorical(
                chunk[group_col], categories=order, ordered=True
            )
        pieces.append(chunk)
    return pd.concat(pieces).loc[out.index]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename dataset columns to the canonical attribute names."""
    return df.rename(
        columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}
    )
