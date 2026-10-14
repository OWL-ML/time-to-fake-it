"""DER++ replay buffer instrumented with memlog hooks.
Single-process training assumed (no rank handling).
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Sequence

import torch
from loguru import logger

from .memory_logger import MemoryLogger
from .replay_hooks import ReplayHooks


class DERReplayBuffer:
    """Minimal DER++ buffer with reservoir sampling and memlog logging.

    Args:
        capacity: Maximum number of stored samples.
        store_logits: Whether to store teacher logits (DER++).
        seed: Seed for the buffer's private RNG stream.
        model_name: Used in the log directory and experiment metadata.
        log_path: Base directory for memlog output.
        enable_logging: ``False`` disables all logging (original behaviour
            minus its CSV writes).
    """

    def __init__(
        self,
        capacity: int,
        store_logits: bool = True,
        seed: int = 42,
        model_name: str = "DERPP",
        log_path: Optional[str] = "replay_map",
        enable_logging: bool = True,
    ) -> None:
        self.capacity = capacity
        self.store_logits = store_logits
        self.seed = seed
        self.model_name = model_name

        # Private RNG stream: sampling draws can no longer disturb the
        # reservoir draws in ``add`` (and vice versa), and the two DER++
        # draws per step are genuinely independent.
        self._rng = random.Random(seed)

        # Data storage (parallel lists, as in the original)
        self.examples: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.image_paths: List[str] = []
        self.task_labels: List[torch.Tensor] = []
        self.logits: List[torch.Tensor] = []
        # Per-sample metadata dicts (uid + demographics), parallel to
        # ``examples``; supplied by the caller of ``add``.
        self.metas: List[Dict[str, Any]] = []

        # Latched on the first ``add``: whether logits / task labels are
        # supplied at all. Prevents the optional lists from silently
        # desynchronising from ``examples`` if a batch omits them.
        self._has_logits: Optional[bool] = None
        self._has_task_labels: Optional[bool] = None

        # Reservoir tracking
        self.n_seen = 0
        self._step = 0
        self.task_id = 0
        self._task_started = False

        base = "replay_map" if log_path is None else log_path
        self.log_path = os.path.join(base, self.model_name, f"seed_run_{seed}")
        self.hooks: Optional[ReplayHooks] = None
        self._set_hooks(enable_logging)

    def _set_hooks(self, enabled: bool) -> None:
        mem_logger = MemoryLogger(
            self.log_path,
            enabled=enabled,
            flush_every=50_000,  # good default at your 1.2M scale
        )
        mem_logger.log_experiment_metadata(
            {
                "seed": self.seed,
                "memory_size": self.capacity,
                "model_name": self.model_name,
            }
        )
        self.hooks = ReplayHooks(mem_logger)

    def __len__(self) -> int:
        return len(self.examples)

    # ---------------- task lifecycle ----------------

    def set_task_id(self, task_id: int) -> None:
        """Switch to a new task; flushes logs for the finished one."""
        if self.hooks is not None and self._task_started:
            self.hooks.end_task()
        self.task_id = task_id
        self._task_started = True
        logger.info("Replay buffer task id set to {}", self.task_id)

    def finalize(self) -> None:
        """Flush logs at the end of training."""
        if self.hooks is not None and self._task_started:
            self.hooks.end_task()
            self._task_started = False

    def tick(self) -> None:
        """Advance the step counter without adding data.

        Call once per training batch when ``add()`` is skipped (e.g. the
        buffer was preloaded for the current task) — otherwise the step
        counter freezes and all sampling logs of the task share one step.
        """
        self._step += 1

    # ---------------- core ----------------

    def _build_meta(
        self, path: str, label: int, extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Identical schema to the path-backed buffer's meta rows."""
        meta = {
            "sample_uid": path,
            "image_path": path,
            "label": label,
            "generator": self.task_id,
        }
        if extra:
            for k, v in extra.items():
                meta.setdefault(k, v)
        return meta

    def _latch(self, attr: str, present: bool, what: str) -> bool:
        current = getattr(self, attr)
        if current is None:
            setattr(self, attr, present)
            return present
        if current != present:
            raise ValueError(
                f"{what} were {'supplied' if current else 'omitted'} on earlier "
                f"add() calls but {'omitted' if current else 'supplied'} now; "
                "the parallel lists would desynchronise from examples"
            )
        return current

    def add(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        image_paths: Sequence[str],
        metas: Optional[Dict[str, torch.Tensor]] = None,
        task_labels: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a batch using reservoir sampling (unchanged algorithm).

        Args:
            examples: Tensor ``[B, C, H, W]``.
            labels: Tensor ``[B]``.
            image_paths: Length-``B`` sequence of image paths.
            metas: Optional dict of demographic columns, each a length-``B``
                tensor, as yielded by the dataloader. Must be supplied
                consistently across calls.
            task_labels: Optional tensor ``[B]``.
            logits: Optional tensor ``[B, num_classes]``.
        """
        examples = examples.cpu()
        labels = labels.cpu()
        if metas is not None:
            metas = {k: v.detach().cpu() for k, v in metas.items()}
        if task_labels is not None:
            task_labels = task_labels.cpu()
        if logits is not None:
            logits = logits.cpu()

        keep_logits = self._latch(
            "_has_logits", self.store_logits and logits is not None, "logits"
        )
        keep_task_labels = self._latch(
            "_has_task_labels", task_labels is not None, "task labels"
        )

        B = examples.size(0)
        for i in range(B):
            extra = (
                {k: v[i].item() for k, v in metas.items()}
                if metas is not None
                else None
            )
            sample_meta = self._build_meta(image_paths[i], labels[i].item(), extra)

            if self.n_seen < self.capacity:
                self.examples.append(examples[i].clone())
                self.labels.append(labels[i].clone())
                if keep_task_labels:
                    self.task_labels.append(task_labels[i].clone())
                if keep_logits:
                    self.logits.append(logits[i].clone())
                self.image_paths.append(image_paths[i])
                self.metas.append(sample_meta)
                if self.hooks is not None:
                    self.hooks.log_insert(
                        self.task_id,
                        self._step,
                        len(self.examples) - 1,
                        sample_meta,
                        reservoir_index=len(self.examples) - 1,
                    )
            else:
                # Reservoir sampling: identical draw order to the original.
                j = self._rng.randint(0, self.n_seen)
                if j < self.capacity:
                    old_meta = self.metas[j]
                    # clone(): examples[i] is a view into the caller's batch
                    # tensor, and storing the view pins the whole batch alive.
                    self.examples[j] = examples[i].clone()
                    self.labels[j] = labels[i].clone()
                    if keep_task_labels:
                        self.task_labels[j] = task_labels[i].clone()
                    if keep_logits:
                        self.logits[j] = logits[i].clone()
                    self.image_paths[j] = image_paths[i]
                    self.metas[j] = sample_meta
                    if self.hooks is not None:
                        self.hooks.log_replace(
                            self.task_id,
                            self._step,
                            j,
                            sample_meta,
                            old_meta,
                            reservoir_index=j,
                        )
                # Rejected candidates (j >= capacity) are not logged.
            self.n_seen += 1
        self.tick()

    # ---------------- sampling ----------------

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        return_logits: bool = True,
        draw: Optional[str] = None,
    ):
        """Sample a batch from the buffer (unchanged tensor logic).

        Every drawn sample is logged so replay exposure per subgroup can
        be measured. ``draw`` labels the purpose of the draw (e.g.
        ``"distillation"`` / ``"classification"`` for DER++'s two draws per step).
        """
        bs = min(batch_size, len(self.examples))
        if bs == 0:
            raise ValueError("cannot sample from an empty buffer")
        indices = self._rng.sample(range(len(self.examples)), bs)

        if self.hooks is not None:
            self.hooks.log_sampling_batch(
                self.task_id,
                self._step,
                [dict(self.metas[i], slot=i) for i in indices],
                draw=draw,
            )

        x = torch.stack([self.examples[i] for i in indices]).to(device)
        y = torch.stack([self.labels[i] for i in indices]).to(device)
        out = [x, y]

        if self.store_logits and len(self.logits) > 0 and return_logits:
            lg = torch.stack([self.logits[i] for i in indices]).to(device)
            out.append(lg)

        if len(self.task_labels) > 0:
            tlabels = torch.stack([self.task_labels[i] for i in indices]).to(device)
            out.append(tlabels)

        return tuple(out)

    # ---------------- checkpointing ----------------

    def state_dict(self, example_dtype: Optional[torch.dtype] = None) -> Dict[str, Any]:
        """Serializable snapshot. Hooks/logger are deliberately excluded."""
        ex = None
        if self.examples:
            ex = torch.stack(self.examples)
            if example_dtype is not None:
                ex = ex.to(example_dtype)
        return {
            "format_version": 1,
            "capacity": self.capacity,
            "store_logits": self.store_logits,
            "seed": self.seed,
            "model_name": self.model_name,
            "n_seen": self.n_seen,
            "step": self._step,
            "task_id": self.task_id,
            "examples": ex,
            "example_dtype": str(ex.dtype) if ex is not None else None,
            "labels": torch.stack(self.labels) if self.labels else None,
            "logits": torch.stack(self.logits) if self.logits else None,
            "task_labels": torch.stack(self.task_labels) if self.task_labels else None,
            "image_paths": list(self.image_paths),
            "metas": list(self.metas),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(
        self,
        sd: Dict[str, Any],
        strict: bool = True,
        restore_rng: bool = True,
        target_dtype: torch.dtype = torch.float32,
    ) -> "DERReplayBuffer":
        if strict:
            if sd["capacity"] != self.capacity:
                raise ValueError(
                    f"capacity mismatch: checkpoint {sd['capacity']} vs buffer {self.capacity}"
                )
            if sd["store_logits"] != self.store_logits:
                raise ValueError("store_logits mismatch between checkpoint and buffer")

        ex = sd["examples"]
        stored_dtype = sd.get("example_dtype")
        if (
            ex is not None
            and stored_dtype is not None
            and str(ex.dtype) != stored_dtype
        ):
            logger.warning(
                "checkpoint example_dtype {} disagrees with the stored tensor ({})",
                stored_dtype,
                ex.dtype,
            )
        if ex is not None and str(ex.dtype) != str(target_dtype):
            logger.debug("upcasting buffer examples {} -> {}", ex.dtype, target_dtype)

        self.examples = list(ex.to(target_dtype).unbind(0)) if ex is not None else []
        self.labels = list(sd["labels"].unbind(0)) if sd["labels"] is not None else []
        self.logits = list(sd["logits"].unbind(0)) if sd["logits"] is not None else []
        self.task_labels = (
            list(sd["task_labels"].unbind(0)) if sd["task_labels"] is not None else []
        )
        self.image_paths = list(sd["image_paths"])
        self.metas = list(sd["metas"])
        self.n_seen = sd["n_seen"]
        self._step = sd["step"]
        self.task_id = sd["task_id"]

        n = len(self.examples)
        if not (len(self.labels) == n == len(self.image_paths) == len(self.metas)):
            raise ValueError("checkpoint arrays are not aligned")
        if self.logits and len(self.logits) != n:
            raise ValueError("checkpoint logits are not aligned with examples")
        if self.task_labels and len(self.task_labels) != n:
            raise ValueError("checkpoint task_labels are not aligned with examples")

        # Re-derive the latches so a resumed run enforces the same contract.
        self._has_logits = bool(self.logits) if n else None
        self._has_task_labels = bool(self.task_labels) if n else None

        if restore_rng and sd.get("rng_state") is not None:
            # Old checkpoints stored the global stream's state; the tuple
            # format is identical, so it loads into the private Random too.
            self._rng.setstate(sd["rng_state"])
        return self

    def save_checkpoint(
        self, path: Optional[str] = None, example_dtype: Optional[torch.dtype] = None
    ) -> str:
        """Atomic rolling snapshot: write to .tmp, then replace."""
        path = path or os.path.join(self.log_path, f"{self.model_name}_buffer.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(self.state_dict(example_dtype=example_dtype), tmp)
        os.replace(tmp, path)
        logger.info(
            "Buffer snapshot after task {} -> {} ({} samples, n_seen={})",
            self.task_id,
            path,
            len(self),
            self.n_seen,
        )
        return path

    def load_checkpoint(
        self,
        path: Optional[str] = None,
        strict: bool = True,
        restore_rng: bool = True,
        target_dtype: torch.dtype = torch.float32,
    ) -> "DERReplayBuffer":
        path = path or os.path.join(self.log_path, f"{self.model_name}_buffer.pt")
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(
            state,
            strict=strict,
            restore_rng=restore_rng,
            target_dtype=target_dtype,
        )
        logger.info(
            "Restored buffer from end of task {} ({} samples, n_seen={})",
            self.task_id,
            len(self),
            self.n_seen,
        )
        return self
