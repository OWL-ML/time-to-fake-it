"""Per-sample lifecycle tracking with demographic columns.

Each record stores the sample's demographic attributes directly
(copied once at insert time), so ``lifecycle.csv`` supports subgroup
analyses without an external uid-to-subgroup join.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LifecycleRecord:
    """Life story of one replay sample or one prompt."""

    uid: str
    insert_task: int = -1
    insert_step: int = -1
    current_slot: Optional[int] = None
    slot_history: List[int] = field(default_factory=list)
    times_sampled: int = 0
    # Number of separate residencies: >1 means the sample re-entered the
    # buffer after being evicted. ``lifetime`` below describes only the
    # most recent residency.
    times_inserted: int = 0
    last_seen_task: int = -1
    last_seen_step: int = -1
    evicted: bool = False
    eviction_task: Optional[int] = None
    # demographics, copied from the sample at insert time
    predicted_gender: Optional[str] = None
    predicted_age: Optional[str] = None
    skin_tone: Optional[str] = None
    skin_tone_group: Optional[str] = None
    intersection: Optional[str] = None

    @property
    def lifetime(self) -> int:
        """Number of tasks the sample survived in memory (current residency)."""
        end = self.eviction_task if self.evicted else self.last_seen_task
        return max(0, (end if end is not None else self.insert_task) - self.insert_task)


class LifecycleTracker:
    """Dictionary of LifecycleRecords with CSV export."""

    def __init__(self) -> None:
        self.records: Dict[str, LifecycleRecord] = {}

    def on_insert(
        self,
        uid: str,
        task: int,
        step: int,
        slot: Optional[int],
        demographics: Optional[Dict[str, Optional[str]]] = None,
    ) -> None:
        """Register a sample entering memory (with its demographics).

        A uid that was previously evicted and is now re-entering starts a
        fresh residency: the eviction state is cleared so ``lifetime`` is
        measured from this insert rather than yielding a negative span.
        """
        rec = self.records.setdefault(uid, LifecycleRecord(uid=uid))
        rec.insert_task, rec.insert_step = task, step
        rec.last_seen_task, rec.last_seen_step = task, step
        rec.times_inserted += 1
        rec.evicted = False
        rec.eviction_task = None
        if slot is not None:
            rec.current_slot = slot
            rec.slot_history.append(slot)
        if demographics:
            for key, value in demographics.items():
                if value is not None and hasattr(rec, key):
                    setattr(rec, key, value)

    def on_sampled(self, uid: str, task: int, step: int) -> None:
        """Count one replay draw / prompt activation."""
        rec = self.records.setdefault(uid, LifecycleRecord(uid=uid))
        rec.times_sampled += 1
        rec.last_seen_task, rec.last_seen_step = task, step

    def on_evict(self, uid: str, task: int) -> None:
        """Mark a sample as removed/replaced."""
        rec = self.records.setdefault(uid, LifecycleRecord(uid=uid))
        rec.evicted = True
        rec.eviction_task = task
        rec.current_slot = None

    def to_csv(self, path: Path) -> None:
        """Write lifecycle.csv (one row per uid, incl. demographics).

        An empty tracker still writes a header-only file, so a stale
        lifecycle.csv from a previous run is never mistaken for output
        of the current one.
        """
        fieldnames = [f.name for f in LifecycleRecord.__dataclass_fields__.values()] + [
            "lifetime"
        ]

        rows = []
        for rec in self.records.values():
            row = asdict(rec)
            row["slot_history"] = "|".join(map(str, rec.slot_history))
            row["lifetime"] = rec.lifetime
            rows.append(row)

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
