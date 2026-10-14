import os
import time

import torch
from loguru import logger

from .trainer import Trainer


class EWCTrainer(Trainer):
    def __init__(
        self,
        model,
        model_name,
        config,
        device,
        criterion=None,
        optimizer=None,
        scheduler=None,
        exclude_activation_params=True,
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
        self.is_ewc = self.config.get("is_ewc", False)
        self.lambda_ewc = self.config.get("lambda_ewc", 0)
        self.model_name = f"{model_name}"
        self.penalty_centers = []
        self.exclude_activation_params = exclude_activation_params
        self.store_dir = os.path.join(
            self.checkpoint_path, self.config.get("ewc_dir", "ewc_state")
        )
        os.makedirs(self.store_dir, exist_ok=True)
        logger.info(f"EWC state will be stored in {self.store_dir}")

        # Fisher entries are non-negative relative weights, so fp16 storage
        # costs nothing that matters and cuts the state file ~2x. Anchors are
        # subtracted from fp32 parameters and stay fp32.
        self.fisher_store_dtype = (
            torch.float16 if self.config.get("ewc_fisher_fp16", True) else torch.float32
        )

        if self.is_ewc and not self.lambda_ewc:
            logger.warning(
                "is_ewc=True but lambda_ewc={} — the penalty term is inert",
                self.lambda_ewc,
            )

    def is_excluded_param(self, name):
        if self.exclude_activation_params:
            return "activation" in name
        else:
            return False

    def save_metrics(self, **kwargs):
        return super().save_metrics(ewc_lambda=self.lambda_ewc, **kwargs)

    # ---------------- training ----------------

    def train_epoch(self, train_loader):
        """Train the model for one epoch.

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

            if self.current_task > 0 and self.is_ewc and self.lambda_ewc:
                if self.penalty_centers:
                    loss = loss + self.lambda_ewc * self.penalty()

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

    def train(self, train_loader, val_loader, wandb_run=None):
        """Train the model on the current task, then consolidate.

        Phase 1 is the epoch loop; phase 2 computes the Fisher information
        for the finished task and appends a penalty center. Both phases are
        timed separately so EWC's consolidation cost is comparable to the
        replay methods' buffer cost. Serializing the state is deliberately
        left outside the phase-2 timer: it is I/O whose cost grows with the
        number of stored centers and is not part of the method's compute.

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
        if profile_resources:
            self._sync_cuda()
        phase1_train_sec = time.perf_counter() - train_t0

        # ---- phase 2: consolidation (compute only; the save is timed apart) ----
        phase2_t0 = time.perf_counter()
        if self.is_ewc:
            self.compute_fisher_and_store(train_loader)
        if profile_resources:
            self._sync_cuda()
        phase2_after_task_sec = time.perf_counter() - phase2_t0

        save_t0 = time.perf_counter()
        self.save_ewc_state()
        ewc_save_sec = time.perf_counter() - save_t0

        task_train_sec = time.perf_counter() - train_t0

        if profile_resources:
            profile = self._record_resource_profile(
                {
                    "before_task_sec": 0.0,
                    "phase1_train_sec": phase1_train_sec,
                    "phase2_after_task_sec": phase2_after_task_sec,
                    "ewc_save_sec": ewc_save_sec,
                    "task_train_sec": task_train_sec,
                    "n_penalty_centers": len(self.penalty_centers),
                },
                log=False,
            )
            logger.info(
                "Task {} resources | train {:.2f}s (phase1 {:.2f}s, fisher {:.2f}s, "
                "save {:.2f}s) | {} penalty centers | peak alloc {:.1f} MB "
                "| peak reserved {:.1f} MB",
                self.current_task,
                task_train_sec,
                phase1_train_sec,
                phase2_after_task_sec,
                ewc_save_sec,
                len(self.penalty_centers),
                profile.get("peak_cuda_allocated_mb", 0.0),
                profile.get("peak_cuda_reserved_mb", 0.0),
            )

    def compute_fisher_and_store(self, dataloader):
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized.")
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized.")
        if self.criterion is None:
            raise ValueError("Criterion is not initialized.")

        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad and not self.is_excluded_param(n)
        }

        was_training = self.model.training
        self.model.eval()

        n_batches = 0
        for data, target, _ in dataloader:
            self.optimizer.zero_grad(set_to_none=True)
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.detach() ** 2
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("Fisher dataloader yielded no batches")
        for n in fisher:
            fisher[n] /= n_batches

        # Leave no stale gradients behind for the next task's first step.
        self.optimizer.zero_grad(set_to_none=True)
        if was_training:
            self.model.train()
        logger.info("Fisher computed over {} batches", n_batches)

        current_params = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad and not self.is_excluded_param(n)
        }

        theta_tilde = self._compute_tilde_theta(current_params, fisher)
        logger.info("theta_tilde computed")
        self.penalty_centers.append({"fisher": fisher, "theta_tilde": theta_tilde})

    def _compute_tilde_theta(self, theta_star, fisher_current):
        # If this is the first task, tilde_theta = theta_star
        if not self.penalty_centers:
            return theta_star

        # Sum of all past fishers + current
        F_total = {n: fisher_current[n].clone() for n in fisher_current}
        for penalty in self.penalty_centers:
            for n in F_total:
                F_total[n] += penalty["fisher"][n]

        # Compute RHS of the formula
        rhs = {}
        for n in F_total:
            rhs[n] = F_total[n] * theta_star[n]
            for penalty in self.penalty_centers:
                rhs[n] -= penalty["fisher"][n] * penalty["theta_tilde"][n]

        # Invert current Fisher (elementwise since diagonal)
        theta_tilde = {
            n: rhs[n] / (fisher_current[n] + 1e-8)  # add epsilon to avoid div-by-zero
            for n in rhs
        }

        return theta_tilde

    def penalty(self):
        loss = 0
        for penalty in self.penalty_centers:
            fisher = penalty["fisher"]
            theta_tilde = penalty["theta_tilde"]
            for n, p in self.model.named_parameters():
                if p.requires_grad and not self.is_excluded_param(n):
                    _loss = fisher[n] * (p - theta_tilde[n]).pow(2)
                    loss += _loss.sum()

        return loss

    # ---------------- state ----------------

    def ewc_state_path(self):
        """Single rolling snapshot.

        Penalty centers are cumulative — the state written after task k
        already contains every earlier center — so per-task files are pure
        redundancy when the only purpose is resuming training.
        """
        return os.path.join(self.store_dir, "ewc_state.pth")

    def save_ewc_state(self, filepath=None):
        """Save all Fisher matrices + theta_tilde anchors to disk (atomically)."""
        filepath = filepath or self.ewc_state_path()
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        state = {
            "penalty_centers": [
                {
                    "fisher": {
                        n: t.detach().cpu().to(self.fisher_store_dtype)
                        for n, t in c["fisher"].items()
                    },
                    "theta_tilde": {
                        n: t.detach().cpu().float() for n, t in c["theta_tilde"].items()
                    },
                }
                for c in self.penalty_centers
            ],
            "current_task": self.current_task,
            "fisher_store_dtype": str(self.fisher_store_dtype),
        }
        tmp = filepath + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, filepath)
        logger.info(
            "[EWC] Saved {} penalty centers -> {}", len(self.penalty_centers), filepath
        )
        return filepath

    def load_ewc_state(self, filepath=None):
        """Load Fisher matrices + theta_tilde anchors, moved to the device.

        Fisher tensors are cast back to fp32 on load: the penalty multiplies
        them against fp32 parameters, and mixing dtypes there would either
        error or silently downcast the parameter difference.
        """
        filepath = filepath or self.ewc_state_path()
        logger.info(
            "[EWC] Loading penalty centers from {} for task {}",
            filepath,
            self.current_task + 1,
        )
        try:
            state = torch.load(filepath, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            logger.warning("[EWC] No state file at {}", filepath)
            return False
        except Exception as e:
            logger.error("[EWC] State could not be loaded: {}", e)
            return False

        if not state or "penalty_centers" not in state:
            logger.error("[EWC] State file at {} has no penalty_centers", filepath)
            return False

        restored_centers = []
        for entry in state["penalty_centers"]:
            restored_centers.append(
                {
                    "fisher": {
                        n: t.to(self.device, torch.float32)
                        for n, t in entry["fisher"].items()
                    },
                    "theta_tilde": {
                        n: t.to(self.device, torch.float32)
                        for n, t in entry["theta_tilde"].items()
                    },
                }
            )

        self.penalty_centers = restored_centers
        saved_task = state.get("current_task")
        if saved_task is not None and saved_task != self.current_task - 1:
            logger.warning(
                "[EWC] State was saved after task {} but current_task is {} — "
                "expected {} centers for a clean resume, loaded {}",
                saved_task,
                self.current_task,
                self.current_task,
                len(restored_centers),
            )
        logger.info(
            "[EWC] Loaded {} penalty centers from {}",
            len(self.penalty_centers),
            filepath,
        )
        return True
