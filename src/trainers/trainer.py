import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch import optim
from torch.utils.data import SequentialSampler

# Demographic columns copied from dataset.get_demographics() into every
# prediction record.
RECORD_DEMOGRAPHICS = (
    "Predicted Gender",
    "Predicted Age",
    "Skin Tone",
    "Skin Tone Group",
    "Intersection",
)


def _git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def _atomic_json(path, obj):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _atomic_torch_save(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


class Trainer:
    """Encapsulates training state and utilities"""

    # Method-specific record columns, written between "logits" and the
    # demographics (e.g. routing columns for prompt-based trainers).
    _EXTRA_RECORD_COLUMNS: tuple = ()

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
        self.model = model
        self.model_name = model_name
        self.checkpoint_path = None
        self.metrics_file = os.path.join(
            "metrics",
            f"{self.model_name}_seed_{config.get('base_seed', 42)}_metrics.json",
        )
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.scheduler = scheduler
        self.current_epoch = 0
        self.best_val_acc = 0.0
        # Lower is better for loss; must start at +inf so the first
        # observed val loss registers as an improvement.
        self.min_val_loss = float("inf")
        self.no_improve_epochs = 0
        self.task_metrics = {}  # Track per-task performance
        # Per-task timing/memory, keyed 0-indexed 'task_{id}'
        self.task_resource_profile: Dict[str, Dict[str, Any]] = {}
        self.current_task = 0
        self.min_epochs = 20
        self.save_eval = False
        self.temperature = 5.0
        self.eval_dir = os.path.join(
            os.environ.get("EVALUATION_DIR", "evaluations"), self.model_name
        )
        self.__set_checkpoint_dir()
        self._init_checkpointing()

    def __set_checkpoint_dir(self):
        self.checkpoint_path = os.path.join(
            "checkpoints",
            f"{self.model_name}",
            f"seed_run_{self.config.get('base_seed', 42)}",
        )
        if not os.path.isdir(self.checkpoint_path):
            os.makedirs(self.checkpoint_path, exist_ok=True)
        print(f"Checkpoint path set to: {self.checkpoint_path}")

    def _init_checkpointing(self):
        """Initialize atomic checkpoint manifest (resume-safe)."""
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self._manifest_path = os.path.join(self.checkpoint_path, "manifest.json")

        existing = None
        if os.path.exists(self._manifest_path):
            try:
                with open(self._manifest_path) as f:
                    existing = json.load(f)
            except Exception:
                # Corrupt/partial manifest should not kill training; rebuild.
                existing = None

        created_at = (
            existing.get("created_at")
            if isinstance(existing, dict) and existing.get("created_at")
            else datetime.now(timezone.utc).isoformat()
        )
        checkpoints = (
            existing.get("checkpoints", {}) if isinstance(existing, dict) else {}
        )

        self._manifest = {
            "model_name": self.model_name,
            "checkpoint_dir": self.checkpoint_path,
            "git_sha": _git_sha(),
            "torch_version": torch.__version__,
            "batch_size": self.config.get("batch_size"),
            "epochs_per_task": self.config.get("epochs"),
            "created_at": created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoints": checkpoints,
        }

        # Keep an immutable copy of the training config for provenance.
        if isinstance(existing, dict) and "config" in existing:
            self._manifest["config"] = existing["config"]
        else:
            self._manifest["config"] = dict(self.config)

        self._write_manifest()

    def _write_manifest(self):
        self._manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self._manifest_path, self._manifest)

    def _checkpoint_manifest_key(self, filepath):
        """Stable manifest key for a checkpoint path."""
        filename = os.path.basename(filepath)

        # Canonical task checkpoints: {model_name}_{task+1}.pt
        if filename.startswith(self.model_name + "_") and filename.endswith(".pt"):
            try:
                n = int(filename.rsplit("_", 1)[-1].split(".")[0])
                return str(n - 1)
            except ValueError:
                pass

        return filename

    def reset_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["epochs"],
            eta_min=self.config["eta_min"],
        )

    def train_epoch(self, train_loader):
        """Train the model for one epoch.

        NOTE on batch convention: this base implementation unpacks
        (data, target, _) — it is NOT compatible with the AI_FACE
        IndxImageFolder(return_mode='index') convention of
        (idx, image, label). PromptCLTrainer overrides train_epoch with the
        index convention; only use this base version with loaders that yield
        (data, target, extra).

        Args:
            train_loader (torch.utils.data.DataLoader): The training data loader.

        Returns:
            dict: A dictionary containing the average loss and accuracy.
        """
        if self.criterion is None:
            raise ValueError("Criterion is not set")
        if self.optimizer is None:
            raise ValueError("Optimizer is not set")
        if self.scheduler is None:
            raise ValueError("Scheduler is not set")

        self.model.train()
        metrics = {"loss": 0.0, "correct": 0, "total": 0}
        for data, target, _ in train_loader:
            data, target = (
                data.to(self.device, non_blocking=True),
                target.to(self.device),
            )
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()
            batch_size = data.size(0)
            metrics["loss"] += loss.item() * batch_size  # scale loss by batch size
            metrics["total"] += batch_size
            pred = output.argmax(dim=1, keepdim=True)
            metrics["correct"] += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = metrics["loss"] / metrics["total"]
        accuracy = 100.0 * metrics["correct"] / metrics["total"]
        return {"loss": avg_loss, "accuracy": accuracy}

    def update_task_metrics(self, metrics, task_id=None):
        if task_id is None:
            task_id = self.current_task
        # 0-indexed, matching _activation_log_context / task_resource_profile.
        # (Checkpoint FILENAMES remain 1-indexed for backward compatibility
        # with existing checkpoints and get_start_task; only metrics keys
        # changed. Downstream parsers of the metrics JSON must be updated.)
        self.task_metrics[f"task_{task_id}"] = metrics

    def _task_checkpoint_path(
        self, task_id: int, checkpoint_dir: str | None = None
    ) -> str:
        """Path of the checkpoint saved after training ``task_id``.

        Filenames stay 1-indexed ('{model_name}_{t+1}.pt') while task ids
        are 0-indexed.
        """
        return os.path.join(
            checkpoint_dir or self.checkpoint_path,
            f"{self.model_name}_{task_id + 1}.pt",
        )

    def save_checkpoint(self, filepath=None, **kwargs):
        """Save model checkpoint atomically and update manifest."""
        if filepath is None:
            filepath = self._task_checkpoint_path(self.current_task)

        state_dict = self.model.state_dict()
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer is not None
            else None,
            "best_val_acc": self.best_val_acc,
            "config": self.config,
            **kwargs,
        }

        _atomic_torch_save(checkpoint, filepath)

        key = self._checkpoint_manifest_key(filepath)
        self._manifest["checkpoints"][key] = {
            "path": filepath,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "task_id": int(self.current_task),
            "epoch": int(self.current_epoch),
            "best_val_acc": float(self.best_val_acc),
            "extra_keys": sorted(kwargs.keys()),
        }
        self._write_manifest()
        print(f"[{type(self).__name__}] Checkpoint saved -> {filepath}")

        self.save_metrics()

    def save_task_checkpoint(self, task_id, **extra):
        """Mixin-style helper used by baseline trainers: one checkpoint per task."""
        path = os.path.join(self.checkpoint_path, f"task_{task_id}.pth")
        payload = {
            "task_id": task_id,
            "model_state": self.model.state_dict(),
            **extra,
        }
        _atomic_torch_save(payload, path)
        self._manifest["checkpoints"][str(task_id)] = {
            "path": path,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        self._write_manifest()
        print(f"[{type(self).__name__}] Task checkpoint saved -> {path}")
        return path

    def load_checkpoint(self, filepath):
        """Load model checkpoint and return the full checkpoint dict, so
        subclasses can read their extra keys (all_keys, soyo_network_state,
        domain_compression, ...) without a second torch.load."""
        if filepath is None:
            filepath = self._task_checkpoint_path(self.current_task)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found at {filepath}")
        print(f"Loading checkpoint from {filepath}")

        # weights_only=False is required: checkpoints contain the config dict
        # and subclass state (numpy GMM params etc.), which the torch>=2.6
        # weights_only=True default refuses to deserialize.
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # strict=False tolerates architecture growth across tasks (prompt
        # pools expand via update_fc), but silent key mismatches have bitten
        # this codebase before — always surface what was skipped.
        result = self.model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if result.missing_keys:
            print(
                f"  [load_checkpoint] {len(result.missing_keys)} missing keys "
                f"(model params not in checkpoint), e.g. {result.missing_keys[:5]}"
            )
        if result.unexpected_keys:
            print(
                f"  [load_checkpoint] {len(result.unexpected_keys)} unexpected keys "
                f"(checkpoint params not in model), e.g. {result.unexpected_keys[:5]}"
            )
        return checkpoint

    def save_metrics(self, **kwargs):
        """Save metrics"""
        metrics = {}

        task_metrics = {
            "task_wise_performance": self.task_metrics,
        }
        metrics = {
            "model_name": self.model_name,
            **kwargs,  # extra info like lambda for EWC
            **task_metrics,
        }
        with open(self.metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

    def early_stopping_with_accuracy(self, val_acc, min_delta=0.2, patience=5):
        """Early stopping with accuracy. Higher is better; an epoch counts as
        'no improvement' only if it did NOT beat the best by min_delta."""
        if val_acc > self.best_val_acc + min_delta:
            self.best_val_acc = val_acc
            self.no_improve_epochs = 0
        elif self.current_epoch >= self.min_epochs:
            self.no_improve_epochs += 1
        return (
            self.no_improve_epochs >= patience and self.current_epoch >= self.min_epochs
        )

    def early_stopping_with_loss(self, val_loss, min_delta=0.1, patience=5):
        """Early stopping with loss. Lower is better; an epoch counts as
        'no improvement' only if it did NOT undercut the best by min_delta."""
        if val_loss < self.min_val_loss - min_delta:
            self.min_val_loss = val_loss
            self.no_improve_epochs = 0
        else:
            self.no_improve_epochs += 1
        return self.no_improve_epochs >= patience

    def reset_early_stopping(self):
        self.no_improve_epochs = 0
        self.best_val_acc = 0.0
        self.min_val_loss = float("inf")

    def check_test_results(self, task_id=None):
        if task_id is None:
            task_id = self.current_task
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, "r") as f:
                metrics = json.load(f)
                tasks_available = metrics.get("task_wise_performance", {})
                # Key format matches update_task_metrics (0-indexed 'task_N').
                if f"task_{task_id}" in tasks_available:
                    return True
        return False

    def get_start_task(self):
        """0-indexed task id to resume from, based on which per-task
        checkpoint files exist. Filenames are '{model_name}_{t+1}.pt', so a
        run that completed tasks 0..k-1 has files numbered 1..k and this
        returns k.

        Sorting is NUMERIC on the parsed task number — lexicographic sorting
        of filenames puts '_10.pt' before '_2.pt' and silently breaks the
        consecutive-completion check on runs with 10+ tasks (the 11-task
        AI_FACE sequence hits this)."""
        task_nums = []
        for checkpoint in os.listdir(self.checkpoint_path):
            file_name = os.path.basename(checkpoint)
            if not (
                file_name.startswith(self.model_name) and file_name.endswith(".pt")
            ):
                continue
            try:
                task_nums.append(int(file_name.rsplit("_", 1)[-1].split(".")[0]))
            except ValueError:
                continue  # non-conforming filename; ignore

        completed = 0
        for n in sorted(set(task_nums)):
            if n == completed + 1:
                completed = n
            else:
                break
        return completed

    def _is_cuda_device(self) -> bool:
        if not torch.cuda.is_available():
            return False
        if isinstance(self.device, torch.device):
            return self.device.type == "cuda"
        return str(self.device).startswith("cuda")

    def _sync_cuda(self):
        if self._is_cuda_device():
            torch.cuda.synchronize(self.device)

    def _cuda_memory_profile(self) -> Dict[str, float]:
        """Peak and current CUDA memory (MB); empty off-CUDA."""
        if not self._is_cuda_device():
            return {}
        mb = 1024**2
        return {
            "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(self.device) / mb,
            "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved(self.device) / mb,
            "end_cuda_allocated_mb": torch.cuda.memory_allocated(self.device) / mb,
            "end_cuda_reserved_mb": torch.cuda.memory_reserved(self.device) / mb,
        }

    def _start_resource_profile(self) -> bool:
        """Reset CUDA peak-memory stats if profiling is on; returns the flag."""
        profile_resources = bool(self.config.get("profile_resources", True))
        if profile_resources and self._is_cuda_device():
            self._sync_cuda()
            torch.cuda.reset_peak_memory_stats(self.device)
        return profile_resources

    def _record_resource_profile(
        self, profile: Dict[str, Any], log: bool = True
    ) -> Dict[str, Any]:
        """Add CUDA memory stats to ``profile`` and store it for the current
        task. Logs a one-line summary unless ``log`` is False."""
        profile = {**profile, **self._cuda_memory_profile()}
        self.task_resource_profile[f"task_{self.current_task}"] = profile
        if log:
            logger.info(
                "Task {} resources | train {:.2f}s (phase1 {:.2f}s, phase2 {:.2f}s) "
                "| peak alloc {:.1f} MB | peak reserved {:.1f} MB",
                self.current_task,
                profile["task_train_sec"],
                profile["phase1_train_sec"],
                profile["phase2_after_task_sec"],
                profile.get("peak_cuda_allocated_mb", 0.0),
                profile.get("peak_cuda_reserved_mb", 0.0),
            )
        return profile

    def _train_epochs(self, train_loader, val_loader, wandb_run=None):
        """Epoch loop with per-epoch validation and accuracy-based early
        stopping."""
        for epoch in range(self.config["epochs"]):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(train_loader)
            self.scheduler.step()

            val_metrics = self.evaluate(
                val_loader, task_id=self.current_task, split_name="val"
            )
            per_epoch_metric = {
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            logger.info(
                "Task {} | epoch {}/{} | train_acc={:.2f}% | val_acc={:.2f}%",
                self.current_task + 1,
                epoch + 1,
                self.config["epochs"],
                train_metrics["accuracy"],
                val_metrics["accuracy"],
            )
            if wandb_run:
                wandb_run.log(
                    {
                        f"task_{self.current_task + 1}": {
                            "epoch": epoch,
                            **per_epoch_metric,
                        }
                    }
                )

            if self.early_stopping_with_accuracy(val_metrics["accuracy"]):
                logger.info(
                    "Early stopping at epoch {} (val acc {:.2f}, best {:.2f})",
                    epoch,
                    val_metrics["accuracy"],
                    self.best_val_acc,
                )
                break

    def _after_epochs(self, train_loader, val_loader):
        """Hook run after the epoch loop, timed as part of phase 1. No-op."""

    def train(self, train_loader, val_loader, wandb_run=None):
        """
        Train the model for multiple epochs.

        Args:
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader): The validation data loader.
            wandb_run (WandbRun): The Weights & Biases run object to log metrics.

        Returns:
            None
        """
        profile_resources = self._start_resource_profile()
        train_t0 = time.perf_counter()

        self._train_epochs(train_loader, val_loader, wandb_run)
        self._after_epochs(train_loader, val_loader)

        if profile_resources:
            self._sync_cuda()
            phase1_train_sec = time.perf_counter() - train_t0
            self._record_resource_profile(
                {
                    "before_task_sec": 0.0,
                    "phase1_train_sec": phase1_train_sec,
                    "phase2_after_task_sec": 0.0,
                    "task_train_sec": time.perf_counter() - train_t0,
                }
            )

    def evaluate(
        self,
        data_loader,
        task_id: int,
        save_predictions: bool = False,
        split_name: Optional[str] = "test",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate on ``data_loader``.

        Loss and accuracy are computed identically whether or not predictions
        are saved, so validation calls are unaffected by this path.

        Args:
            data_loader: Evaluation loader.
            task_id: Task ID for the evaluation.
            save_predictions: Write per-sample records to parquet.
            split_name: Tag for the output filename, e.g. ``"test_task_3"``.
                Required when ``save_predictions`` is True.
            temperature: Temperature for the scaled-probability column.
                Defaults to ``config["eval_temperature"]`` or 5.0.

        Returns:
            ``{"loss", "accuracy"}``, plus ``{"records_path", "n_records"}``
            when predictions were saved.
        """
        if self.criterion is None:
            raise ValueError("Criterion is not set")

        logger.info(f"Evaluating task {task_id}")

        dataset = getattr(data_loader, "dataset", None)
        idx_to_class = None
        path_to_idx = None

        if save_predictions:
            idx_to_class, fake_col = self._record_label_maps(dataset, split_name)

            # Prefer resolving demographics by path so the result is
            # independent of loader order; fall back to a running counter,
            # which is only valid for a sequential sampler.
            samples = getattr(dataset, "samples", None)
            if samples is not None:
                path_to_idx = {str(p): i for i, (p, _) in enumerate(samples)}
            else:
                sampler = getattr(data_loader, "sampler", None)
                if sampler is not None and not isinstance(sampler, SequentialSampler):
                    raise ValueError(
                        "dataset exposes no .samples, so demographics are resolved "
                        "by running counter — requires a sequential loader."
                    )

        temp, temp_col = self._temperature_column(temperature)

        self.model.eval()
        metrics = {"loss": 0.0, "correct": 0, "total": 0}
        rows: List[Dict[str, Any]] = []
        seen = 0
        missing_demographics = 0

        with torch.no_grad():
            for batch in data_loader:
                data, target = batch[0], batch[1]
                paths = batch[2] if len(batch) > 2 else None

                if save_predictions and paths is None:
                    raise ValueError(
                        "loader yields no image paths; cannot save prediction records"
                    )

                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device)

                output = self.model(data)
                output = output["logits"] if isinstance(output, dict) else output
                loss = self.criterion(output, target)

                batch_size = data.size(0)
                metrics["loss"] += loss.item() * batch_size
                metrics["total"] += batch_size
                pred = output.argmax(dim=1, keepdim=True)
                metrics["correct"] += pred.eq(target.view_as(pred)).sum().item()

                if save_predictions:
                    arrays = self._record_arrays(output, target, temp, fake_col)

                    for b in range(batch_size):
                        idx = (
                            path_to_idx[str(paths[b])]
                            if path_to_idx is not None
                            else seen + b
                        )
                        dem = dataset.get_demographics(idx)
                        if dem is None:
                            missing_demographics += 1
                            continue
                        rows.append(
                            self._record_row(
                                arrays,
                                b,
                                self._relative_path(paths[b]),
                                idx,
                                dem,
                                idx_to_class,
                                temp_col,
                            )
                        )
                    seen += batch_size

            avg_loss = metrics["loss"] / metrics["total"]
            overall_accuracy = 100.0 * metrics["correct"] / metrics["total"]
            results = {"loss": avg_loss, "accuracy": overall_accuracy}

            if save_predictions:
                self._warn_missing_demographics(missing_demographics, seen, split_name)
                results["records_path"] = str(
                    self._write_records(rows, task_id, split_name, temp_col)
                )
                results["n_records"] = len(rows)

            return results

    # ------------------------------------------------------------------
    # Prediction records (shared by evaluate() overrides)
    # ------------------------------------------------------------------

    def _relative_path(self, path) -> str:
        root = self.config.get("dataset_dir") or os.environ.get("DATASET_DIR", "")
        p = str(path)
        return p.replace(f"{root}/", "") if root else p

    @staticmethod
    def _record_label_maps(dataset, split_name):
        """Check ``dataset`` can produce prediction records and return
        ``(idx_to_class, fake_col)``."""
        if split_name is None:
            raise ValueError("split_name is required when save_predictions=True")
        if dataset is None:
            raise ValueError("data_loader has no .dataset; cannot save predictions")
        if not hasattr(dataset, "get_demographics"):
            raise ValueError("dataset does not expose get_demographics()")
        class_to_idx = getattr(dataset, "class_to_idx", None)
        if not class_to_idx:
            raise ValueError("dataset has no class_to_idx; cannot write string labels")
        if "fake" not in class_to_idx:
            raise ValueError(
                f"expected a 'fake' class in class_to_idx, got {class_to_idx}"
            )
        return {v: k for k, v in class_to_idx.items()}, class_to_idx["fake"]

    def _temperature_column(self, temperature):
        """Resolve the eval temperature and its record column name."""
        temp = float(
            temperature
            if temperature is not None
            else self.config.get("eval_temperature", 5.0)
        )
        return temp, f"probability_t{temp:g}"

    @staticmethod
    def _record_arrays(logits, target, temp, fake_col):
        """Per-batch numpy views used to build prediction records."""
        return {
            "logits": logits.float().cpu().numpy(),
            "prob_fake": torch.softmax(logits, dim=1)[:, fake_col].float().cpu().numpy(),
            "prob_t_fake": F.softmax(logits / temp, dim=1)[:, fake_col]
            .float()
            .cpu()
            .numpy(),
            "pred": logits.argmax(dim=1).cpu().numpy(),
            "label": target.cpu().numpy(),
        }

    @staticmethod
    def _record_row(
        arrays, b, image_path, idx, dem, idx_to_class, temp_col, **extra
    ) -> Dict[str, Any]:
        return {
            "image_path": image_path,
            "dataset_index": idx,
            "label": idx_to_class[int(arrays["label"][b])],
            "prediction": idx_to_class[int(arrays["pred"][b])],
            "probability_fake": float(arrays["prob_fake"][b]),
            temp_col: float(arrays["prob_t_fake"][b]),
            "logits": arrays["logits"][b].tolist(),
            **extra,
            **{c: dem[c] for c in RECORD_DEMOGRAPHICS},
        }

    @staticmethod
    def _warn_missing_demographics(missing, total, split_name):
        if missing:
            logger.warning(
                "{}/{} samples in {} had no demographics and are absent from "
                "the records; metrics still include them.",
                missing,
                total,
                split_name,
            )

    def _write_records(
        self, rows, task_id: int, split_name: str, temp_col: str
    ) -> Path:
        cols = [
            "image_path",
            "dataset_index",
            "label",
            "prediction",
            "probability_fake",
            temp_col,
            "logits",
            *self._EXTRA_RECORD_COLUMNS,
            *RECORD_DEMOGRAPHICS,
        ]
        df = pd.DataFrame(rows, columns=cols)
        df["model_name"] = self.model_name
        df["seed"] = self.config.get("base_seed")
        df["trained_through_task"] = self.current_task + 1

        out_dir = (
            Path(self.config.get("predictions_dir", "predictions"))
            / self.model_name
            / f"seed_run_{self.config.get('base_seed')}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = (
            out_dir
            / f"{split_name}_after_task_{self.current_task + 1}_on_{task_id + 1}.parquet"
        )
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        logger.info("Wrote {} prediction records -> {}", len(df), path)
        return path

    def test_all_tasks(self, data_test, save_predictions=True):
        """
        Test all tasks and return the test accuracy for each task.
        """
        test_acc = []
        for i, test_loader in enumerate(data_test.values()):
            test_results_for_task = self.evaluate(
                test_loader, task_id=i, save_predictions=save_predictions
            )
            test_acc.append(test_results_for_task["accuracy"])
            print(
                f"Test accuracy on task {i + 1} at task {self.current_task + 1}: {test_acc[i]:.2f}%"
            )
        return test_acc
