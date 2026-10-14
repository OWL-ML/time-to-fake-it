"""
PromptCLTrainer: base class for prompt-based CL trainers (S-Prompts, SOYO, PINA)
on per-task-directory datasets (AI_FACE layout).

each task's IndxImageFolder yields local targets in [0, class_num), and task
identity comes from the task index, never from the label.

Progress output goes through loguru.

Metrics JSON layout (written via Trainer.save_metrics):
  task_wise_performance: rows keyed by the 1-INDEXED task number as a
    string ("1".."11"). Each row holds ``task_id`` (1-indexed),
    ``average_accuracy`` (mean over SEEN tasks only), and one
    ``task_{t}_acc_after_task_{J}`` column per evaluated task, both
    1-indexed. Rows cover unseen tasks too when eval_future_tasks is on.
  cl_metrics: top-level {"ACC", "BWT"} — not nested inside
    task_wise_performance.
  task_resource_profile: per-task timing/memory, keyed 0-indexed
    'task_{id}' (unchanged).
Task ids stay 0-indexed everywhere in memory (_all_task_accs,
task_resource_profile); the 1-indexing exists only in the serialized
rows. restore_task_accs_from_metrics parses that format back and
rejects the older 0-indexed 'task_{id}' rows rather than shifting them.

Method-specific trainers override:
  _before_task         (param freezing, optimizer setup, set self._task_epochs)
  _after_task          (routing setup: KMeans / soyo_network; checkpoint)
  _knn_select          (per-sample routing at inference)
  train_epoch          (only if the forward/loss differs from the default)
  evaluate             (only if interface's return type differs, e.g. SiNet)
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from .trainer import Trainer


class PromptCLTrainer(Trainer):
    # Serialized accuracy column: task_{t+1}_acc_after_task_{J+1}
    _ROW_KEY_RE = re.compile(r"^task_(\d+)_acc_after_task_(\d+)$")
    _EXTRA_RECORD_COLUMNS = ("routed_task", "true_task", "routing_confidence")

    def __init__(
        self,
        model,
        model_name,
        config,
        device,
        train_datasets: dict,  # {task_name: IndxImageFolder(return_mode="index")}
        val_datasets: dict,
        test_datasets: dict,
        analysis_datasets: dict = None,  # train data w/ test transforms (optional)
        **kwargs,
    ):
        super().__init__(model, model_name, config, device, **kwargs)

        self.task_names = list(train_datasets.keys())
        self.train_datasets = list(train_datasets.values())
        self.val_datasets = list(val_datasets.values())
        self.test_datasets = list(test_datasets.values())
        self.analysis_datasets = (
            list(analysis_datasets.values()) if analysis_datasets else None
        )

        self.class_num = config["class_num"]

        # Per-task runtime/resource profiling
        self.profile_resources = self.config.get("profile_resources", True)
        self.profile_include_eval = self.config.get("profile_include_eval", False)

        # _all_task_accs[t] = {task_id: acc} evaluated after training task t
        self._all_task_accs: list[dict[int, float]] = []
        # Latest ACC/BWT; written as a TOP-LEVEL key by save_metrics (not
        # nested in task_wise_performance).
        self._cl_metrics: dict[str, float] | None = None
        # Sanity: consistent lengths, correct dataset convention.
        n = len(self.train_datasets)
        assert len(self.val_datasets) == n and len(self.test_datasets) == n, (
            "train/val/test dataset dicts must cover the same tasks"
        )
        if self.analysis_datasets is not None:
            assert len(self.analysis_datasets) == n, (
                "analysis dataset dict must cover the same tasks as train/val/test"
            )
        for ds in (
            self.train_datasets
            + self.val_datasets
            + self.test_datasets
            + (self.analysis_datasets or [])
        ):
            mode = getattr(ds, "return_mode", None)
            assert mode == "index", (
                "All datasets must be IndxImageFolder(return_mode='index') "
                f"yielding (idx, image, label); got return_mode={mode!r}"
            )

    # ======================================================================
    # Task bookkeeping
    # ======================================================================

    @property
    def nb_tasks(self) -> int:
        return len(self.train_datasets)

    # ======================================================================
    # Core CL loop
    # ======================================================================

    def incremental_train(self, task_id: int):
        """Called once per task, in order."""
        self.current_task = task_id

        logger.info("{}", "=" * 60)
        logger.info("Task {} ({})", task_id, self.task_names[task_id])
        logger.info("{}", "=" * 60)

        self._start_resource_profile()

        train_t0 = time.perf_counter()

        t0 = time.perf_counter()
        self._before_task(task_id)
        if self.profile_resources:
            self._sync_cuda()
        before_task_sec = time.perf_counter() - t0

        train_loader = self._make_loader(
            self.train_datasets[task_id], shuffle=True, seed_offset=task_id
        )

        t0 = time.perf_counter()
        self._train_task(task_id, train_loader)
        if self.profile_resources:
            self._sync_cuda()
        phase1_train_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._after_task(task_id)
        if self.profile_resources:
            self._sync_cuda()
        phase2_after_task_sec = time.perf_counter() - t0

        task_train_sec = time.perf_counter() - train_t0

        # 0-indexed, matching _all_task_accs / eval prints.
        task_key = f"task_{task_id}"
        if self.profile_resources:
            profile = {
                "task_name": self.task_names[task_id]
                if task_id < len(self.task_names)
                else f"task_{task_id}",
                "before_task_sec": before_task_sec,
                "phase1_train_sec": phase1_train_sec,
                "phase2_after_task_sec": phase2_after_task_sec,
                "task_train_sec": task_train_sec,
            }
            profile = self._record_resource_profile(profile, log=False)

            peak_alloc = profile.get("peak_cuda_allocated_mb", 0.0)
            peak_res = profile.get("peak_cuda_reserved_mb", 0.0)
            logger.info(
                "Task {} resources | train {:.2f}s "
                "(before {:.2f}s, phase1 {:.2f}s, phase2 {:.2f}s) | "
                "peak alloc {:.1f} MB | peak reserved {:.1f} MB",
                task_id,
                task_train_sec,
                before_task_sec,
                phase1_train_sec,
                phase2_after_task_sec,
                peak_alloc,
                peak_res,
            )

        if self.profile_resources and self.profile_include_eval:
            t0 = time.perf_counter()
            self._evaluate_all_tasks(task_id)
            self._sync_cuda()
            eval_sec = time.perf_counter() - t0
            self.task_resource_profile.setdefault(task_key, {})["eval_sec"] = eval_sec
            logger.info("Task {} eval time | {:.2f}s", task_id, eval_sec)
        else:
            self._evaluate_all_tasks(task_id)

    def _before_task(self, task_id: int):
        """Override: freeze/unfreeze params, build optimizer + scheduler,
        set self._task_epochs.
        Reference convention: init_lr/init_epochs for task 0, lr/epochs
        afterwards."""
        raise NotImplementedError

    def _train_task(self, task_id: int, train_loader):
        """Fixed epoch count, no early stopping (cosine annealing T_max equals
        the full epoch count in the reference implementations)."""
        epochs = getattr(self, "_task_epochs", self.config["epochs"])

        for epoch in range(epochs):
            self.current_epoch = epoch
            metrics = self.train_epoch(train_loader)
            self.scheduler.step()

            logger.info(
                "Epoch {:3d}/{} | loss {:.4f} | acc {:.2f}%",
                epoch + 1,
                epochs,
                metrics["loss"],
                metrics["accuracy"],
            )

    def train_epoch(self, train_loader):
        """Default epoch: local targets, current-task forward.
        Model stays in eval() per the reference implementations (frozen
        backbone, dropout off; gradients still flow to requires_grad params).
        Override if the method needs a different forward/loss."""
        self.model.eval()
        metrics = {"loss": 0.0, "correct": 0, "total": 0}

        for batch in train_loader:
            _, data, target = batch  # target already in [0, class_num)
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(data)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = F.cross_entropy(logits, target)
            loss.backward()
            self.optimizer.step()

            metrics["loss"] += loss.item() * data.size(0)
            metrics["total"] += data.size(0)
            metrics["correct"] += logits.argmax(1).eq(target).sum().item()

        assert metrics["total"] > 0, (
            "train_epoch received an empty loader; check the task dataset."
        )
        return {
            "loss": metrics["loss"] / metrics["total"],
            "accuracy": 100.0 * metrics["correct"] / metrics["total"],
        }

    def _after_task(self, task_id: int):
        """Override: routing setup (KMeans / soyo_network training),
        checkpointing."""
        raise NotImplementedError

    # ======================================================================
    # Dataset / loader access
    # ======================================================================

    def _get_analysis_dataset(self, task_id: int):
        """Task's train data with test transforms — for post-task passes
        (routing key building) where augmentation noise would pollute the
        result. Falls back to the val set (same
        domain, test transforms) if none was provided."""
        if self.analysis_datasets is not None:
            return self.analysis_datasets[task_id]
        return self.val_datasets[task_id]

    def _make_loader(self, dataset, shuffle: bool, seed_offset: int = 0):
        """Loader factory. When shuffle=True, batch order (and worker base
        seeds) are drawn from a dedicated generator seeded by
        base_seed + seed_offset, so shuffle order is reproducible and
        independent of how much of the GLOBAL torch RNG was consumed earlier
        (e.g. by SOYO phase-2 randperm calls)."""
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(
                int(self.config.get("base_seed", 0)) + int(seed_offset)
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            generator=generator,
        )

    # ======================================================================
    # Routing (method-specific)
    # ======================================================================

    def _knn_select(self, feat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Each method implements its own routing: "
            "KMeans centers with L1 distance for S-Prompts and PINA "
            "(reference-implementation metric; the S-Prompts paper "
            "describes L2); learned soyo_network domain classifier for SOYO."
        )

    def _routing_confidence(self, scores: torch.Tensor) -> torch.Tensor:
        """Top-1 softmax probability over per-task routing scores.

        ``scores`` is ``[B, n_seen_tasks]`` and must be oriented so that
        HIGHER means better (negate distances before passing them in).
        """
        return torch.softmax(scores.float(), dim=1).max(dim=1).values

    # ======================================================================
    # Evaluation
    # ======================================================================

    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
        task_id,
        save_predictions: bool = False,
        split_name: Optional[str] = "test",
        temperature: Optional[float] = None,
    ) -> dict:
        """Full routed evaluation: extract_vector -> _knn_select -> interface.

        Only valid AFTER _after_task has built routing state for all seen
        tasks. Targets are local, so plain equality is correct here.

        Loss is not computed on this path; it is reported as None (never 0.0)
        so any downstream aggregation fails loudly instead of averaging in a
        fabricated number.

        Note: SiNet.interface returns a raw tensor rather than a dict;
        the isinstance check below covers that.

        Args:
            loader: Evaluation loader yielding ``(index, data, target)``.
            save_predictions: Write per-sample records to parquet.
            split_name: Tag for the output filename, e.g. ``"test_task_3"``.
                Required when ``save_predictions`` is True.
            temperature: Temperature for the scaled-probability column.
                Defaults to ``config["eval_temperature"]`` or 5.0.

        Returns:
            ``{"accuracy", "loss"}``, plus ``{"records_path", "n_records"}``
            when predictions were saved.
        """
        dataset = getattr(data_loader, "dataset", None)
        idx_to_class = None
        samples = None

        if save_predictions:
            idx_to_class, fake_col = self._record_label_maps(dataset, split_name)

            # Indices arrive in the batch, so no order assumptions are needed;
            # samples[] is only used to recover the path for each index.
            samples = getattr(dataset, "samples", None)

        temp, temp_col = self._temperature_column(temperature)

        self.model.eval()
        correct, total = 0, 0
        rows: List[Dict[str, Any]] = []
        missing_demographics = 0

        for batch in data_loader:
            indices, data, target = batch
            data, target = data.to(self.device), target.to(self.device)

            feat = self.model.extract_vector(data)
            sel_out = self._knn_select(feat)
            if isinstance(sel_out, tuple):
                selection, route_scores = sel_out
            else:
                selection, route_scores = sel_out, None
            out = self.model.interface(data, selection)
            logits = out["logits"] if isinstance(out, dict) else out

            correct += logits.argmax(1).eq(target).sum().item()
            total += data.size(0)

            if save_predictions:
                arrays = self._record_arrays(logits, target, temp, fake_col)
                idx_np = (
                    indices.cpu().numpy()
                    if torch.is_tensor(indices)
                    else np.asarray(indices)
                )
                routed_np = (
                    selection.cpu().numpy()
                    if torch.is_tensor(selection)
                    else np.asarray(selection)
                )
                if route_scores is not None:
                    route_conf = self._routing_confidence(route_scores).cpu().numpy()
                elif hasattr(self, "_warned_no_route_conf"):
                    if not self._warned_no_route_conf:
                        logger.warning(
                            "_knn_select returned no scores; routing_confidence will be "
                            "null for {}.",
                            split_name,
                        )
                        self._warned_no_route_conf = True
                    route_conf = None
                else:
                    route_conf = None

                for b in range(data.size(0)):
                    idx = int(idx_np[b])
                    dem = dataset.get_demographics(idx)
                    if dem is None:
                        missing_demographics += 1
                        continue
                    rows.append(
                        self._record_row(
                            arrays,
                            b,
                            (
                                self._relative_path(samples[idx][0])
                                if samples is not None
                                else None
                            ),
                            idx,
                            dem,
                            idx_to_class,
                            temp_col,
                            routed_task=int(routed_np[b]),
                            true_task=int(task_id),
                            routing_confidence=float(route_conf[b])
                            if route_conf is not None
                            else None,
                        )
                    )

        assert total > 0, "evaluate received an empty loader."
        results = {"accuracy": 100.0 * correct / total, "loss": None}

        if save_predictions:
            self._warn_missing_demographics(missing_demographics, total, split_name)
            results["n_records"] = len(rows)
            results["records_path"] = str(
                self._write_records(rows, task_id, split_name, temp_col)
            )

        return results

    @torch.no_grad()
    def evaluate_current_task(self, loader) -> dict:
        """Mid-task evaluation via the model's own forward() (current-task
        prompt, no routing). Safe to call during training. Loss is reported
        as None (not computed on this path)."""
        self.model.eval()
        correct, total = 0, 0

        for batch in loader:
            _, data, target = batch
            data = data.to(self.device)
            out = self.model(data)
            logits = out["logits"] if isinstance(out, dict) else out
            correct += logits.argmax(1).cpu().eq(target).sum().item()
            total += data.size(0)

        assert total > 0, "evaluate_current_task received an empty loader."
        return {"accuracy": 100.0 * correct / total, "loss": None}

    def _evaluate_all_tasks(self, current_task_id: int):
        """After training task t, evaluate on each task 0..t separately.
        Populates the accuracy matrix used for ACC/BWT."""
        # Positional index into _all_task_accs IS the task id; the ACC/BWT
        # computation (and the BWT anchor _all_task_accs[t][t]) depends on
        # exactly one row per task, appended in order. Guard against resume /
        # re-run misuse that would silently skew the metrics.
        assert len(self._all_task_accs) == current_task_id, (
            f"_evaluate_all_tasks called out of order: have "
            f"{len(self._all_task_accs)} rows but current_task_id="
            f"{current_task_id}. One ordered incremental_train call per task "
            "is required. When resuming a run, call "
            "restore_task_accs_from_metrics() first to rebuild the accuracy "
            "matrix; re-running an already-evaluated task corrupts ACC/BWT."
        )

        # Seen tasks fill the lower triangle; with eval_future_tasks the
        # not-yet-trained tasks are evaluated too, completing the matrix
        # (zero-shot entries, routed through the prompts that exist now).
        eval_tasks = range(self.nb_tasks)

        row: dict[int, float] = {}
        for t in eval_tasks:
            ds = self.test_datasets[t]
            loader = self._make_loader(ds, shuffle=False)
            metrics = self.evaluate(
                loader, task_id=t, split_name="test", save_predictions=True
            )
            row[t] = metrics["accuracy"]

            logger.info(
                "Eval task {} ({}){}: {:.2f}%",
                t,
                self.task_names[t],
                "" if t <= current_task_id else " [unseen]",
                metrics["accuracy"],
            )

        self._all_task_accs.append(row)
        # Serialized row: 1-indexed string key, 1-indexed columns. NOT
        # update_task_metrics — that writes the older 0-indexed
        # 'task_{id}' key.
        self.task_metrics[str(current_task_id + 1)] = self._format_metrics_row(
            current_task_id, row
        )
        self._compute_cl_metrics()

    def _format_metrics_row(self, current_task_id: int, row: dict[int, float]) -> dict:
        """Serialize one accuracy-matrix row in the persisted 1-indexed format.

        Columns cover every evaluated task (including not-yet-seen ones
        when eval_future_tasks is on); ``average_accuracy`` covers only
        tasks 0..current_task_id, matching the ACC definition in
        _compute_cl_metrics.
        """
        j = current_task_id + 1
        seen = [row[t] for t in range(current_task_id + 1) if t in row]
        entry: dict[str, Any] = {
            "task_id": j,
            "average_accuracy": float(np.mean(seen)) if seen else 0.0,
        }
        for t in sorted(row):
            entry[f"task_{t + 1}_acc_after_task_{j}"] = row[t]
        return entry

    def restore_task_accs_from_metrics(self) -> int:
        """Rebuild the accuracy matrix from the persisted metrics JSON when
        resuming a run, so the _evaluate_all_tasks ordering guard passes and
        ACC/BWT are computed against the true post-task accuracies (the BWT
        anchor _all_task_accs[t][t] in particular).

        Also restores self.task_metrics, because Trainer.save_metrics
        rewrites the whole JSON from that dict — resuming with it empty
        would wipe earlier tasks' rows on the first save.

        Returns the number of restored rows, i.e. the 0-indexed task id to
        resume from (same semantics as Trainer.get_start_task). Returns 0
        and restores nothing if the metrics file doesn't exist.

        Intended call order for a resume:
            start = trainer.restore_task_accs_from_metrics()
            assert start == trainer.get_start_task()  # metrics/ckpts agree
            trainer.load_checkpoint(<checkpoint for task start-1>)
            for task_id in range(start, trainer.nb_tasks):
                trainer.incremental_train(task_id)

        NOTE: only rows written under the 1-indexed string key convention
        ('1', '2', ... with 'task_{t}_acc_after_task_{J}' columns) are
        restored. Metrics files from the earlier 0-indexed 'task_{id}'
        convention are detected and rejected rather than silently shifted
        by one.
        """
        assert len(self._all_task_accs) == 0, (
            "restore_task_accs_from_metrics must be called before any "
            "training/evaluation, not after rows already exist."
        )

        if not os.path.exists(self.metrics_file):
            logger.info(
                "[restore] no metrics file at {}; starting fresh (task 0).",
                self.metrics_file,
            )
            return 0

        with open(self.metrics_file, "r") as f:
            payload = json.load(f)
        stored = payload.get("task_wise_performance", {})

        rows_by_id: dict[int, dict[int, float]] = {}
        for key, value in stored.items():
            if not str(key).isdigit():
                if str(key).startswith("task_"):
                    raise ValueError(
                        f"Metrics row {key!r} uses the old 0-indexed "
                        "'task_{id}' key convention. Refusing to restore — "
                        "migrate the file or rerun rather than risk an "
                        "off-by-one accuracy matrix."
                    )
                continue  # e.g. a stale inline 'cl_metrics'
            tid = int(key) - 1
            if tid < 0:
                raise ValueError(
                    f"Metrics row key {key!r} is not 1-indexed; rows start at '1'."
                )
            if not isinstance(value, dict):
                raise ValueError(
                    f"Metrics row {key!r} is not a per-task accuracy dict "
                    f"(got {type(value).__name__}); this metrics file was "
                    "not written by PromptCLTrainer._evaluate_all_tasks."
                )
            row: dict[int, float] = {}
            for k, v in value.items():
                m = self._ROW_KEY_RE.match(str(k))
                if m is None:
                    continue  # 'task_id' / 'average_accuracy'
                t, j = int(m.group(1)) - 1, int(m.group(2)) - 1
                if j != tid:
                    raise ValueError(
                        f"Metrics row {key!r} contains column {k!r}, which "
                        f"reports an accuracy after task {j + 1}; rows must "
                        "be internally consistent."
                    )
                row[t] = float(v)
            if not row:
                raise ValueError(
                    f"Metrics row {key!r} has no "
                    "'task_{t}_acc_after_task_{J}' columns; metrics file is "
                    "not in the expected format."
                )
            rows_by_id[tid] = row

        if not rows_by_id:
            logger.info("[restore] metrics file has no task rows; starting fresh.")
            return 0

        # Require a contiguous 0..k-1 prefix; the positional list can't
        # represent gaps.
        n = len(rows_by_id)
        missing = [t for t in range(n) if t not in rows_by_id]
        if missing:
            raise ValueError(
                f"Metrics file has non-contiguous task rows (missing "
                f"{[m + 1 for m in missing]} out of 1..{n}); cannot rebuild "
                "the accuracy matrix."
            )

        # Row t was evaluated after training task t, so it must at least
        # cover tasks 0..t. Later columns (unseen tasks, written when
        # eval_future_tasks is on) are optional.
        for t in range(n):
            gaps = [c for c in range(t + 1) if c not in rows_by_id[t]]
            if gaps:
                raise ValueError(
                    f"Row {t + 1} is missing accuracies for tasks "
                    f"{[g + 1 for g in gaps]}; metrics file is inconsistent."
                )

        self._all_task_accs = [rows_by_id[t] for t in range(n)]
        # Keep the serialized rows verbatim so save_metrics preserves
        # history; they are already in the persisted format.
        self.task_metrics = dict(stored)
        # cl_metrics is top-level and recomputed on the next
        # _evaluate_all_tasks anyway; carry it so an interim save doesn't
        # drop the key.
        restored_cl = payload.get("cl_metrics")
        self._cl_metrics = dict(restored_cl) if isinstance(restored_cl, dict) else None

        logger.info(
            "[restore] rebuilt accuracy matrix for {} task(s) from {}; "
            "resume at task {}.",
            n,
            self.metrics_file,
            n,
        )
        return n

    def save_metrics(self, **kwargs):
        payload = {"task_resource_profile": self.task_resource_profile}
        # Top-level, alongside task_resource_profile — Trainer.save_metrics
        # spreads **kwargs at the root and nests only self.task_metrics
        # under 'task_wise_performance'.
        if self._cl_metrics is not None:
            payload["cl_metrics"] = self._cl_metrics
        payload.update(kwargs)
        super().save_metrics(**payload)

    def _compute_cl_metrics(self):
        """ACC: mean accuracy over all tasks after the latest one.
        BWT: mean(acc_now(t) - acc_right_after_training(t)) over past tasks."""
        n = len(self._all_task_accs)
        if n == 0:
            return

        latest = self._all_task_accs[-1]
        acc = float(np.mean([latest.get(t, 0.0) for t in range(n)]))

        bwt_terms = []
        for t in range(n - 1):
            acc_after_training = self._all_task_accs[t].get(t, 0.0)
            acc_now = latest.get(t, 0.0)
            bwt_terms.append(acc_now - acc_after_training)
        bwt = float(np.mean(bwt_terms)) if bwt_terms else 0.0

        self._cl_metrics = {"ACC": acc, "BWT": bwt}
        self.save_metrics()
        logger.info("ACC: {:.2f}% | BWT: {:+.2f}%", acc, bwt)

    def _assert_routing_restored(self, task_id: int):
        """Override: fail loudly if load_checkpoint did not restore the
        routing state for tasks 0..task_id. No-op here.

        Without this, _knn_select runs on stale or empty routing state and
        silently produces plausible-but-wrong accuracies rather than crashing.
        """

    def evaluate_saved_checkpoints(
        self,
        checkpoint_dir: str | None = None,
        predictions_dir: str | None = None,
        metrics_file: str | None = None,
    ) -> list[dict[int, float]]:
        """Post-hoc evaluation of a finished run: for each task t, load the
        checkpoint saved after training t and evaluate on all tasks,
        rebuilding the same accuracy matrix (and ACC/BWT) a live run produces.

        No training, no _before_task/_after_task — routing state
        comes from the checkpoints.

        ``predictions_dir`` and ``metrics_file`` redirect the outputs. Leave
        them None ONLY if you intend to overwrite the training run's
        parquet records and metrics JSON: the filenames are identical.
        """
        assert len(self._all_task_accs) == 0, (
            "evaluate_saved_checkpoints must run on a fresh trainer; "
            "rows already exist in _all_task_accs."
        )

        paths = [
            self._task_checkpoint_path(t, checkpoint_dir) for t in range(self.nb_tasks)
        ]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {self.nb_tasks} task checkpoints are missing, "
                f"e.g. {missing[:3]}. _evaluate_all_tasks needs a contiguous "
                "0..n-1 sequence to keep the accuracy matrix aligned."
            )

        if predictions_dir is not None:
            self.config["predictions_dir"] = predictions_dir
        if metrics_file is not None:
            os.makedirs(os.path.dirname(metrics_file) or ".", exist_ok=True)
            self.metrics_file = metrics_file

        for task_id, path in enumerate(paths):
            # Set BEFORE loading/evaluating: _write_records reads
            # self.current_task for both the filename and the
            # 'trained_through_task' column.
            self.current_task = task_id
            self.load_checkpoint(path)
            self.model.to(self.device)
            self._assert_routing_restored(task_id)

            logger.info("{}", "=" * 60)
            logger.info(
                "Post-hoc eval | checkpoint after task {} ({})",
                task_id,
                self.task_names[task_id],
            )
            logger.info("{}", "=" * 60)

            self._evaluate_all_tasks(task_id)

        return self._all_task_accs
