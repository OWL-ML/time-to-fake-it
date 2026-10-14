"""Buffered CSV logging for continual-learning memory dynamics.

Records replay-buffer decisions and sampling draws for failure-mode
analysis. Purely observational: when ``enabled`` is False every method
is a no-op and training behaviour is unchanged. Writes are batched to
keep overhead negligible.

CSV outputs are opened in append mode, so a resumed run adds to the
existing ``events.csv`` / ``sampling.csv`` / ``prompt_events.csv``.
``lifecycle.csv`` is derived state and is rewritten in full on every
flush; pass ``resume=True`` so it is rebuilt from ``events.csv`` first
rather than truncated to the current session.
"""

from __future__ import annotations

import csv
import datetime
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger as log

from .lifecycle import LifecycleTracker

#: Demographic attributes logged per sample. ``intersection`` is the
#: pre-computed gender x skin-tone group from the annotation pipeline.
#: Every name here must exist as a field on ReplayEvent, SamplingEvent
#: and LifecycleRecord — both are populated by iterating this tuple.
DEMOGRAPHIC_FIELDS = (
    "predicted_gender",
    "predicted_age",
    "skin_tone",
    "skin_tone_group",
    "intersection",
)


@dataclass
class ReplayEvent:
    """One replay-buffer decision (insert / replace / reject / evict)."""

    timestamp: str
    task: int
    global_step: int
    event: str  # insert | replace | reject | evict | select
    slot: Optional[int]
    accepted: bool
    reservoir_index: Optional[int] = None
    acceptance_probability: Optional[float] = None
    sample_uid: Optional[str] = None
    image_path: Optional[str] = None
    dataset: Optional[str] = None
    generator: Optional[str] = None
    predicted_gender: Optional[str] = None
    predicted_age: Optional[str] = None
    skin_tone: Optional[str] = None
    skin_tone_group: Optional[str] = None
    intersection: Optional[str] = None
    label: Optional[Any] = None
    loss: Optional[float] = None
    confidence: Optional[float] = None
    entropy: Optional[float] = None
    #: Set by MemoryLogger at log time; identifies the process that wrote
    #: the row, so re-run tasks can be separated after a crash+resume.
    run_id: Optional[str] = None


@dataclass
class PromptEvent:
    """One prompt-pool event (created / selected / retired).

    Written to ``prompt_events.csv``. Lifecycle-derived fields
    (``activation_count``, ``created``, ``last_used``, ``retired``,
    ``lifetime``) are filled by the logger from its lifecycle tracker
    at log time — they reflect the prompt's state at that moment and
    do not need to be supplied by the caller.
    """

    # Governance: when / which prompt / where it came from
    timestamp: str
    task: int
    step: int
    prompt_id: str
    creation_task: Optional[int] = None
    # Transparency: what happened and how strongly
    event: str = "selected"  # created | selected | retired
    selected: bool = True  # False for scored-but-not-chosen rows
    activation_count: Optional[int] = None
    activation_score: Optional[float] = None
    # Explainability: why the router chose it
    similarity: Optional[float] = None
    embedding_norm: Optional[float] = None
    gradient_norm: Optional[float] = None
    # Performance: model state for the routed batch/sample
    confidence: Optional[float] = None
    loss: Optional[float] = None
    prediction: Optional[Any] = None
    # Fairness: demographics of the routed sample/batch
    predicted_gender: Optional[str] = None
    predicted_age: Optional[str] = None
    skin_tone: Optional[str] = None
    skin_tone_group: Optional[str] = None
    intersection: Optional[str] = None
    # Provenance: origin of the routed sample
    dataset: Optional[str] = None
    generator: Optional[str] = None
    subject_id: Optional[str] = None
    # Lifecycle: prompt state at event time (auto-filled)
    created: Optional[str] = None  # timestamp of the creation event
    last_used: Optional[int] = None  # last task the prompt was selected in
    retired: bool = False
    lifetime: Optional[int] = None  # tasks since creation
    run_id: Optional[str] = None


@dataclass
class SamplingEvent:
    """One replay draw (or prompt selection) during training."""

    task: int
    step: int
    slot: Optional[int]
    sample_uid: str
    # Purpose of the draw within the training step, e.g. "distillation"
    # (DER++ logit-matching draw) vs "classification" (CE draw). Optional.
    draw: Optional[str] = None
    # Ground-truth task of the routed sample (prompt-routing logs);
    # lets routing correctness per subgroup be computed directly.
    true_task: Optional[int] = None
    dataset: Optional[str] = None
    generator: Optional[str] = None
    predicted_gender: Optional[str] = None
    predicted_age: Optional[str] = None
    skin_tone: Optional[str] = None
    skin_tone_group: Optional[str] = None
    intersection: Optional[str] = None
    label: Optional[Any] = None
    run_id: Optional[str] = None


def _as_bool(value: Any) -> bool:
    """Parse a CSV cell back into a bool."""
    return str(value).strip().lower() in ("true", "1", "yes")


def _as_int(value: Any) -> Optional[int]:
    """Parse a CSV cell back into an int, tolerating blanks."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class MemoryLogger:
    """Single sink for all memory logs.

    Produces ``events.csv``, ``sampling.csv``, ``prompt_events.csv``,
    ``lifecycle.csv`` and ``runs.jsonl`` in ``out_dir``. Rows are
    buffered in memory and flushed every ``flush_every`` rows and at
    task boundaries.

    Args:
        out_dir: Directory for all outputs.
        enabled: When False every method is a no-op.
        flush_every: Rows buffered before a flush. Note that up to this
            many rows are lost if the process dies between flushes.
        resume: Rebuild the lifecycle tracker from an existing
            ``events.csv`` before logging. Required when continuing a
            run, otherwise ``lifecycle.csv`` is overwritten with only
            the current session's records.
    """

    def __init__(
        self,
        out_dir: str | Path,
        enabled: bool = True,
        flush_every: int = 1000,
        resume: bool = False,
    ) -> None:
        self.enabled = enabled
        self.out_dir = Path(out_dir)
        self.flush_every = flush_every
        self.run_id = uuid.uuid4().hex[:12]
        self._events: List[ReplayEvent] = []
        self._sampling: List[SamplingEvent] = []
        self._prompt_events: List[PromptEvent] = []
        self._prompt_created: Dict[str, str] = {}  # prompt_id -> timestamp
        self.lifecycle = LifecycleTracker()
        if self.enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            if resume:
                self.rehydrate_lifecycle()

    # ------------------------------------------------------------------
    # resume support
    # ------------------------------------------------------------------

    def rehydrate_lifecycle(self, include_sampling: bool = True) -> int:
        """Replay ``events.csv`` (and ``sampling.csv``) into the tracker.

        ``lifecycle.csv`` is rewritten in full on every flush, so without
        this a resumed run would truncate it to the current session and
        lose every insert/evict pair from before the interruption.

        Args:
            include_sampling: Also replay ``sampling.csv``. Required for
                ``times_sampled`` and ``last_seen_task`` to survive a
                resume — without it, exposure counts restart at zero and
                ``lifetime`` is understated for still-resident samples.
                Skip it only if that file has grown too large to scan.

        Returns:
            Number of rows replayed across both files.
        """
        replayed = 0

        events_path = self.out_dir / "events.csv"
        if events_path.exists():
            with events_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    uid = row.get("sample_uid")
                    if not uid:
                        continue
                    task = _as_int(row.get("task"))
                    step = _as_int(row.get("global_step"))
                    if task is None or step is None:
                        continue
                    kind = row.get("event")
                    if kind in ("insert", "replace") and _as_bool(row.get("accepted")):
                        demographics = {
                            f: (row.get(f) or None) for f in DEMOGRAPHIC_FIELDS
                        }
                        self.lifecycle.on_insert(
                            uid, task, step, _as_int(row.get("slot")), demographics
                        )
                    elif kind == "evict":
                        self.lifecycle.on_evict(uid, task)
                    replayed += 1

        sampling_path = self.out_dir / "sampling.csv"
        if include_sampling and sampling_path.exists():
            with sampling_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    uid = row.get("sample_uid")
                    task = _as_int(row.get("task"))
                    step = _as_int(row.get("step"))
                    if not uid or task is None or step is None:
                        continue
                    self.lifecycle.on_sampled(uid, task, step)
                    replayed += 1

        log.info(
            "Rehydrated lifecycle from {} rows ({} uids tracked)",
            replayed,
            len(self.lifecycle.records),
        )
        return replayed

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------

    def log_event(self, event: ReplayEvent) -> None:
        """Record one buffer decision; updates the lifecycle tracker."""
        if not self.enabled:
            return
        event.run_id = self.run_id
        self._events.append(event)
        uid = event.sample_uid
        if uid is not None:
            if event.event in ("insert", "replace") and event.accepted:
                demographics = {f: getattr(event, f) for f in DEMOGRAPHIC_FIELDS}
                self.lifecycle.on_insert(
                    uid, event.task, event.global_step, event.slot, demographics
                )
            elif event.event == "evict":
                self.lifecycle.on_evict(uid, event.task)
        if len(self._events) >= self.flush_every:
            self._flush(self.out_dir / "events.csv", self._events)
            self._events = []

    def log_prompt_event(self, event: PromptEvent) -> None:
        """Record one prompt event; auto-fills lifecycle-derived fields.

        ``created`` events register the prompt in the lifecycle tracker;
        ``selected`` events (with ``selected=True``) count as
        activations; ``retired`` events mark it evicted.
        """
        if not self.enabled:
            return
        event.run_id = self.run_id
        pid = event.prompt_id
        if event.event == "created":
            self.lifecycle.on_insert(pid, event.task, event.step, None)
            self._prompt_created[pid] = event.timestamp
        elif event.event == "selected" and event.selected:
            self.lifecycle.on_sampled(pid, event.task, event.step)
        elif event.event == "retired":
            self.lifecycle.on_evict(pid, event.task)

        rec = self.lifecycle.records.get(pid)
        if rec is not None:
            if event.creation_task is None:
                event.creation_task = rec.insert_task
            if event.activation_count is None:
                event.activation_count = rec.times_sampled
            event.created = self._prompt_created.get(pid, event.created)
            event.last_used = rec.last_seen_task
            event.retired = rec.evicted
            if event.lifetime is None:
                event.lifetime = rec.lifetime

        self._prompt_events.append(event)
        if len(self._prompt_events) >= self.flush_every:
            self._flush(self.out_dir / "prompt_events.csv", self._prompt_events)
            self._prompt_events = []

    def log_sampling(self, event: SamplingEvent, update_lifecycle: bool = True) -> None:
        """Record one replay draw / prompt selection.

        ``update_lifecycle=False`` writes the row without touching the
        lifecycle tracker — used when the activation was already counted
        by a companion ``PromptEvent``.
        """
        if not self.enabled:
            return
        event.run_id = self.run_id
        self._sampling.append(event)
        if update_lifecycle:
            self.lifecycle.on_sampled(event.sample_uid, event.task, event.step)
        if len(self._sampling) >= self.flush_every:
            self._flush(self.out_dir / "sampling.csv", self._sampling)
            self._sampling = []

    def log_experiment_metadata(self, extra: Optional[Dict[str, Any]] = None) -> None:
        """Append this run's provenance to ``runs.jsonl``.

        Appended rather than overwritten so a resumed run does not erase
        the git commit and timestamp of the session that produced the
        earlier rows in the CSVs. ``experiment.json`` is still written
        for the first run in a directory, for backwards compatibility.
        """
        if not self.enabled:
            return
        meta: Dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "git_commit": self._git_commit(),
        }
        try:
            import torch

            meta["torch_version"] = torch.__version__
            meta["cuda"] = torch.version.cuda
        except ImportError:
            pass
        if extra:
            meta.update(extra)

        with (self.out_dir / "runs.jsonl").open("a") as f:
            f.write(json.dumps(meta) + "\n")

        legacy = self.out_dir / "experiment.json"
        if not legacy.exists():
            legacy.write_text(json.dumps(meta, indent=2))

    # ------------------------------------------------------------------
    # flushing
    # ------------------------------------------------------------------

    def end_task(self) -> None:
        """Flush all buffers at a task boundary."""
        if self.enabled:
            self.close()

    def close(self) -> None:
        """Final flush; safe to call repeatedly."""
        if not self.enabled:
            return
        if self._events:
            self._flush(self.out_dir / "events.csv", self._events)
            self._events = []
        if self._sampling:
            self._flush(self.out_dir / "sampling.csv", self._sampling)
            self._sampling = []
        if self._prompt_events:
            self._flush(self.out_dir / "prompt_events.csv", self._prompt_events)
            self._prompt_events = []
        self.lifecycle.to_csv(self.out_dir / "lifecycle.csv")

    @staticmethod
    def _flush(path: Path, rows: List[Any]) -> None:
        if not rows:
            return
        dicts = [asdict(r) for r in rows]
        fieldnames = list(dicts[0].keys())

        write_header = not path.exists()
        if not write_header:
            with path.open(newline="") as f:
                existing = next(csv.reader(f), None)
            if existing is not None and existing != fieldnames:
                raise ValueError(
                    f"{path.name} was written with a different schema and cannot be "
                    f"appended to. Existing columns: {existing}. Current columns: "
                    f"{fieldnames}. Move the old file aside before rerunning."
                )

        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(dicts)
        log.debug("flushed {} rows to {}", len(rows), path.name)

    @staticmethod
    def _git_commit() -> Optional[str]:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:
            return None
