import os
from pathlib import Path

import torch
from loguru import logger

from buffers import ReplayBuffer

from .trainer import Trainer


class ReplayTrainer(Trainer):
    """Base for trainers that keep a replay buffer in ``self.buffer``:
    snapshots the buffer after each task and restores it on resume."""

    def __init__(
        self,
        model,
        model_name,
        config,
        device,
        criterion=None,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__(
            model,
            model_name,
            config,
            device,
            criterion,
            optimizer,
            scheduler,
        )
        self.use_replay = config.get("use_replay", True)
        self.beta = config.get("beta", 0.5)
        self.buffer = self._make_buffer()
        self.preloaded_from_task = -1

    def _make_buffer(self):
        """Create the empty replay buffer. Override for a different buffer."""
        capacity = self.config.get("capacity", self.config.get("memory_size", 1000))
        seed = self.config.get("base_seed", 42)
        buffer = ReplayBuffer(
            capacity,
            model_name=self.model_name,
            seed=seed,  # keeps per-seed memlogs in separate directories
            log_path=self.config.get("log_path", "replay_map"),
            enable_logging=self.config.get("enable_buffer_logging", True),
        )
        logger.info(
            "Replay buffer: capacity={}, seed={}, log_path={}",
            capacity,
            seed,
            buffer.log_path,
        )
        return buffer

    def buffer_checkpoint_path(self) -> Path:
        root = Path(
            self.config.get(
                "buffer_dir", self.config.get("checkpoint_dir", "checkpoints")
            )
        )
        return root / f"{self.model_name}_{self.config.get('base_seed', 42)}_buffer.pt"

    def save_buffer(self, path=None) -> Path:
        if self.buffer is None:
            logger.error("No buffer to save for {}.", self.model_name)
            raise ValueError("No buffer to save")
        path = Path(path) if path is not None else self.buffer_checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        dtype = torch.float16 if self.config.get("buffer_fp16", False) else None
        payload = {
            "buffer": self.buffer.state_dict(example_dtype=dtype),
            "current_task": self.current_task,
            "preloaded_from_task": self.preloaded_from_task,
            "task_resource_profile": self.task_resource_profile,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic overwrite of the previous task's snapshot
        logger.info(
            "Buffer snapshot after task {} -> {} ({} samples, n_seen={}, {:.1f} MB)",
            self.current_task + 1,
            path,
            len(self.buffer),
            self.buffer.n_seen,
            path.stat().st_size / 1024**2,
        )
        return path

    def load_buffer(self, path=None) -> int:
        """Restore the buffer. Returns the index of the last COMPLETED task."""
        path = Path(path) if path is not None else self.buffer_checkpoint_path()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.buffer.load_state_dict(payload["buffer"])
        self.preloaded_from_task = payload["preloaded_from_task"]
        self.task_resource_profile = payload.get("task_resource_profile", {})
        logger.info(
            "Restored buffer from end of task {} ({} samples, n_seen={})",
            payload["current_task"] + 1,
            len(self.buffer),
            self.buffer.n_seen,
        )
        return payload["current_task"]
    buffer = None
