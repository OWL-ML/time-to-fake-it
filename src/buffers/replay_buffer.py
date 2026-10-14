"""Plain reservoir replay buffer (no logits) instrumented with memlog hooks.

Single-process training assumed (no rank handling). Tensor-backed: images are
stored in the buffer, so there is no loader, cache or path re-reading. Paths
are still kept because the memlog rows are keyed on them.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Sequence

import torch
from loguru import logger

from .memory_logger import MemoryLogger
from .replay_hooks import ReplayHooks


class ReplayBuffer:
    """Reservoir buffer with memlog logging. DER++ minus the logits."""

    def __init__(
        self,
        capacity: int,
        model_name: str = "replay",
        seed: int = 42,
        log_path: Optional[str] = "replay_map",
        enable_logging: bool = True,
    ) -> None:
        self.capacity = capacity
        self.seed = seed
        self.model_name = model_name

        # Private RNG stream: sampling draws cannot disturb the reservoir
        # draws in ``add`` (and vice versa).
        self._rng = random.Random(seed)

        # Data storage (parallel lists)
        self.examples: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.image_paths: List[str] = []
        self.task_labels: List[torch.Tensor] = []
        # Parallel to ``examples``; holds the row written to the memlog for
        # whatever currently occupies each slot, so evictions can report it.
        self.metas: List[Dict[str, Any]] = []

        # Latched on the first ``add``: whether task labels are supplied at
        # all, so the optional list cannot desynchronise from ``examples``.
        self._has_task_labels: Optional[bool] = None

        self.n_seen = 0
        self._step = 0
        self.task_id = 0
        self._task_started = False

        base = "replay_map" if log_path is None else log_path
        self.log_path = os.path.join(base, self.model_name, f"seed_run_{seed}")
        self.hooks: Optional[ReplayHooks] = None
        self._set_hooks(enable_logging)

    def _set_hooks(self, enabled: bool) -> None:
        mem_logger = MemoryLogger(self.log_path, enabled=enabled, flush_every=50_000)
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
        """Advance the step counter for a batch where ``add()`` is skipped."""
        self._step += 1

    # ---------------- core ----------------

    def _build_meta(
        self, path: str, label: int, extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Identical schema to the DER++ buffer's meta rows."""
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
    ) -> None:
        """Add a batch using reservoir sampling.

        Args:
            examples: Tensor ``[B, C, H, W]``.
            labels: Tensor ``[B]``.
            image_paths: Length-``B`` sequence of image paths.
            metas: Optional dict of demographic columns, each a length-``B``
                tensor, as yielded by the dataloader. Must be supplied
                consistently across calls.
            task_labels: Optional tensor ``[B]``.
        """
        examples = examples.cpu()
        labels = labels.cpu()
        if metas is not None:
            metas = {k: v.detach().cpu() for k, v in metas.items()}
        if task_labels is not None:
            task_labels = task_labels.cpu()

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
                # clone(): examples[i] is a view into the caller's batch
                # tensor, and storing the view pins the whole batch alive.
                self.examples.append(examples[i].clone())
                self.labels.append(labels[i].clone())
                if keep_task_labels:
                    self.task_labels.append(task_labels[i].clone())
                self.image_paths.append(image_paths[i])
                self.metas.append(sample_meta)
                slot = len(self.examples) - 1
                if self.hooks is not None:
                    self.hooks.log_insert(
                        self.task_id,
                        self._step,
                        slot,
                        sample_meta,
                        reservoir_index=slot,
                    )
            else:
                j = self._rng.randint(0, self.n_seen)
                if j < self.capacity:
                    old_meta = self.metas[j]
                    self.examples[j] = examples[i].clone()
                    self.labels[j] = labels[i].clone()
                    if keep_task_labels:
                        self.task_labels[j] = task_labels[i].clone()
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
        draw: Optional[str] = None,
    ):
        """Sample a batch from the buffer.

        Every drawn sample is logged so replay exposure per subgroup can be
        measured. ``draw`` labels the purpose of the draw.
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

        if len(self.task_labels) > 0:
            tlabels = torch.stack([self.task_labels[i] for i in indices]).to(device)
            out.append(tlabels)

        return tuple(out)

    # ---------------- checkpointing ----------------

    def state_dict(self, example_dtype: Optional[torch.dtype] = None) -> Dict[str, Any]:
        """Serializable snapshot. Hooks and logger are deliberately excluded.

        ``example_dtype=torch.float16`` halves the snapshot size.
        """
        ex = None
        if self.examples:
            ex = torch.stack(self.examples)
            if example_dtype is not None:
                ex = ex.to(example_dtype)
        return {
            "format_version": 2,
            "capacity": self.capacity,
            "seed": self.seed,
            "model_name": self.model_name,
            "task_id": self.task_id,
            "n_seen": self.n_seen,
            "step": self._step,
            "examples": ex,
            "example_dtype": str(ex.dtype) if ex is not None else None,
            "labels": torch.stack(self.labels) if self.labels else None,
            "task_labels": torch.stack(self.task_labels) if self.task_labels else None,
            "image_paths": list(self.image_paths),
            "metas": list(self.metas),
            "py_random_state": self._rng.getstate(),
        }

    def load_state_dict(
        self,
        state: Dict[str, Any],
        restore_rng: bool = True,
        target_dtype: torch.dtype = torch.float32,
    ) -> "ReplayBuffer":
        if state["capacity"] != self.capacity:
            raise ValueError(
                f"capacity mismatch: checkpoint {state['capacity']} vs buffer {self.capacity}"
            )

        ex = state.get("examples")
        if ex is None and state.get("format_version", 1) < 2:
            raise ValueError(
                "checkpoint predates tensor-backed storage and holds paths only; "
                "it cannot be restored into this buffer"
            )
        self.examples = list(ex.to(target_dtype).unbind(0)) if ex is not None else []
        self.labels = (
            list(state["labels"].unbind(0)) if state["labels"] is not None else []
        )
        self.task_labels = (
            list(state["task_labels"].unbind(0))
            if state.get("task_labels") is not None
            else []
        )
        self.image_paths = list(state["image_paths"])
        self.metas = list(state.get("metas") or [])
        self.n_seen = state["n_seen"]
        self._step = state["step"]
        self.task_id = state["task_id"]

        n = len(self.examples)
        if not (len(self.labels) == n == len(self.image_paths) == len(self.metas)):
            raise ValueError("checkpoint arrays are not aligned")
        if self.task_labels and len(self.task_labels) != n:
            raise ValueError("checkpoint task_labels are not aligned with examples")

        self._has_task_labels = bool(self.task_labels) if n else None

        if restore_rng and state.get("py_random_state") is not None:
            # Old checkpoints stored the global stream's state; the tuple
            # format is identical, so it loads into the private Random too.
            self._rng.setstate(state["py_random_state"])
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
        restore_rng: bool = True,
        target_dtype: torch.dtype = torch.float32,
    ) -> "ReplayBuffer":
        path = path or os.path.join(self.log_path, f"{self.model_name}_buffer.pt")
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state, restore_rng=restore_rng, target_dtype=target_dtype)
        logger.info(
            "Restored buffer from end of task {} ({} samples, n_seen={})",
            self.task_id,
            len(self),
            self.n_seen,
        )
        return self
