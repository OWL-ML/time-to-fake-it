"""Accumulated (growing-union joint) baseline trainer.

Standalone counterpart of the AccumulatedTrainer, adapted to the current
base Trainer (checkpoint manifest + atomic save live in the base).

Design
------
- Stage t trains on union(tasks 0..t).
- One model is carried across all stages (continued fine-tuning on the
  growing union). This is the continued-fine-tuning variant, NOT the true
  joint upper bound: because stage t starts from stage t-1's weights it
  carries task-order effects. Report the residual per-group disparity as
  "continued-FT accumulated", not as the pure data-inherent floor.

Loader convention for train_epoch: batches are (data, target, meta).
"""

from __future__ import annotations

from .trainer import Trainer


class AccumulatedTrainer(Trainer):
    """Growing-union accumulated baseline (continued fine-tuning).

    Stage t trains on the union of tasks 0..t, carrying one model across all
    stages. Because each stage continues from the previous stage's weights,
    this is the continued-fine-tuning variant — it carries task-order effects
    and is not the true joint upper bound. Report its residual per-group
    disparity accordingly.
    """

    def __init__(
        self,
        model,
        model_name,
        config,
        device,
        optimizer=None,
        scheduler=None,
        criterion=None,
    ):
        """Continued-fine-tuning accumulated baseline: one model trained
        across all stages on the growing union(0..t).

        Note: this is NOT the true joint upper bound — stage t continues
        from stage t-1's weights, so it carries task-order effects and sits
        between naive and joint. Report it as such, not as the
        data-inherent floor.
        """
        model = model.to(device)

        super().__init__(
            model=model,
            model_name=model_name,
            config=config,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        # The base owns self.model/optimizer/scheduler after this — do not
        # reassign them.

        # Persist baseline provenance in the base Trainer manifest.
        self._manifest["trainer_class"] = type(self).__name__
        self._manifest["accumulated_variant"] = "continued_finetuning"
        self._write_manifest()

    # ------------------------------------------------------------------ #
    # Stage lifecycle
    # ------------------------------------------------------------------ #

    def begin_stage(self, stage_id: int):
        """Call before training stage `stage_id`. One model is carried
        across all stages, so this only advances the stage index; weights,
        optimizer, and scheduler state persist into the next stage."""
        self.current_task = int(stage_id)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train_epoch(self, train_loader, epoch=None):
        if self.criterion is None or self.optimizer is None:
            raise ValueError("Criterion/optimizer not set")

        self.model.train()
        metrics = {"loss": 0.0, "correct": 0, "total": 0}

        for data, target, _ in train_loader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            bs = data.size(0)
            metrics["loss"] += loss.item() * bs
            metrics["total"] += bs
            pred = output.argmax(dim=1, keepdim=True)
            metrics["correct"] += pred.eq(target.view_as(pred)).sum().item()

        if metrics["total"] == 0:
            raise ValueError("train_epoch received an empty loader")

        return {
            "loss": metrics["loss"] / metrics["total"],
            "accuracy": 100.0 * metrics["correct"] / metrics["total"],
        }

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def save_stage_checkpoint(self, stage_id=None, **extra):
        """Compatibility helper: stage checkpoint under task_{id}.pth."""
        if stage_id is None:
            stage_id = self.current_task
        return self.save_task_checkpoint(int(stage_id), **extra)
