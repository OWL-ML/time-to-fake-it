"""Replay buffers plus memlog — raw memory logging for continual learning.

memlog records replay-buffer events, sampling draws and per-sample
lifecycles for failure-mode analysis, without modifying training behaviour.

Minimal usage::

    from buffers import MemoryLogger, ReplayHooks

    logger = MemoryLogger("runs/exp1/logs", enabled=cfg.log_memory)
    logger.log_experiment_metadata({"seed": cfg.seed})
    hooks = ReplayHooks(logger)

    hooks.log_insert(task, step, slot, sample_meta)
    hooks.log_replace(task, step, slot, new_meta, old_meta)
    hooks.log_sampling_batch(task, step, drawn_metas)
    hooks.end_task()
"""

from .der_buffer import DERReplayBuffer
from .lifecycle import LifecycleRecord, LifecycleTracker
from .memory_logger import DEMOGRAPHIC_FIELDS, MemoryLogger, ReplayEvent, SamplingEvent
from .replay_buffer import ReplayBuffer
from .replay_hooks import ReplayHooks

__all__ = [
    "DERReplayBuffer",
    "ReplayBuffer",
    "DEMOGRAPHIC_FIELDS",
    "MemoryLogger",
    "ReplayEvent",
    "SamplingEvent",
    "LifecycleRecord",
    "LifecycleTracker",
    "ReplayHooks",
]
