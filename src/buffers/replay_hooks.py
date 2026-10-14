"""Instrumentation hooks for a replay buffer.

One call per buffer decision; the buffer itself never writes files.
All methods are no-ops when the underlying logger is disabled.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from .memory_logger import DEMOGRAPHIC_FIELDS, MemoryLogger, ReplayEvent, SamplingEvent


def _demo(sample: Dict[str, Any], field: str) -> Optional[Any]:
    """Read a demographic attribute from a sample-metadata dict.

    Accepts snake_case keys (``predicted_gender``) or the original
    annotation column names (``Predicted Gender``, ``Skin Tone Group``,
    ``Intersection``).
    """
    if field in sample:
        return sample[field]
    return sample.get(field.replace("_", " ").title())


class ReplayHooks:
    """Passive hooks a replay trainer calls at each buffer decision."""

    def __init__(self, logger: MemoryLogger) -> None:
        self.logger = logger

    def _event(
        self,
        kind: str,
        task: int,
        step: int,
        accepted: bool,
        slot: Optional[int],
        sample: Dict[str, Any],
        reservoir_index: Optional[int] = None,
        acceptance_probability: Optional[float] = None,
    ) -> None:
        self.logger.log_event(
            ReplayEvent(
                timestamp=datetime.datetime.now().isoformat(),
                task=task,
                global_step=step,
                event=kind,
                slot=slot,
                accepted=accepted,
                reservoir_index=reservoir_index,
                acceptance_probability=acceptance_probability,
                sample_uid=sample.get("sample_uid"),
                image_path=sample.get("image_path"),
                dataset=sample.get("dataset"),
                generator=sample.get("generator"),
                predicted_gender=_demo(sample, "predicted_gender"),
                predicted_age=_demo(sample, "predicted_age"),
                skin_tone=_demo(sample, "skin_tone"),
                skin_tone_group=_demo(sample, "skin_tone_group"),
                intersection=_demo(sample, "intersection"),
                label=sample.get("label"),
                loss=sample.get("loss"),
                confidence=sample.get("confidence"),
                entropy=sample.get("entropy"),
            )
        )

    def log_insert(
        self, task: int, step: int, slot: int, sample: Dict[str, Any], **kw: Any
    ) -> None:
        """A new sample entered an empty slot."""
        self._event("insert", task, step, True, slot, sample, **kw)

    def log_replace(
        self,
        task: int,
        step: int,
        slot: int,
        new_sample: Dict[str, Any],
        old_sample: Optional[Dict[str, Any]] = None,
        **kw: Any,
    ) -> None:
        """A reservoir replacement: evict the old sample, insert the new."""
        if old_sample is not None:
            self._event("evict", task, step, True, slot, old_sample)
        self._event("replace", task, step, True, slot, new_sample, **kw)

    def log_reject(
        self, task: int, step: int, sample: Dict[str, Any], **kw: Any
    ) -> None:
        """A candidate was not admitted to the buffer."""
        self._event("reject", task, step, False, None, sample, **kw)

    def log_evict(
        self, task: int, step: int, slot: int, sample: Dict[str, Any]
    ) -> None:
        """A sample was removed without replacement."""
        self._event("evict", task, step, True, slot, sample)

    def log_sampling_batch(
        self,
        task: int,
        step: int,
        samples: List[Dict[str, Any]],
        draw: Optional[str] = None,
    ) -> None:
        """Record every sample drawn for one replay step.

        ``draw`` labels the purpose of the draw within the step
        (e.g. ``"distillation"`` vs ``"classification"`` for DER++'s two draws).
        """
        for s in samples:
            self.logger.log_sampling(
                SamplingEvent(
                    task=task,
                    step=step,
                    slot=s.get("slot"),
                    sample_uid=s["sample_uid"],
                    draw=draw,
                    dataset=s.get("dataset"),
                    generator=s.get("generator"),
                    **{f: _demo(s, f) for f in DEMOGRAPHIC_FIELDS},
                    label=s.get("label"),
                )
            )

    def end_task(self) -> None:
        """Flush logs at a task boundary."""
        self.logger.end_task()
