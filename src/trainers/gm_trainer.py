import logging

import torch
from loguru import logger
from tqdm import tqdm

from coreset.cords.utils.data.dataloader.SL.adaptive import GradMatchDataLoader

from .replay_trainer import ReplayTrainer

# Demographic columns carried into the memlog. Integer codes, in the order
# the annotation pipeline defines them.
DEMOGRAPHIC_COLUMNS = (
    "Predicted Gender",
    "Predicted Age",
    "Skin Tone",
    "Skin Tone Group",
    "Intersection",
)


class GradMatchTrainer(ReplayTrainer):
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
        self.gm_config = {
            "model": dict(
                architecture="ViT-B_16",  # vit_base_patch16_224
                numclasses=2,
                pretrained=True,
            ),
            "shuffle": True,
            "device": self.device,
            "loss": self.criterion,
            "num_classes": self.config["num_classes"],
            "fraction": 0.01,
            "select_every": 1,
            "lam": 0.5,
            "selection_type": "PerBatch",
            "v1": False,
            "valid": False,
            "kappa": 0,
            "eps": 1e-100,
            "linear_layer": False,
            "reduction": "none",
            "eta": 3e-2,
        }

    def save_metrics(self, **kwargs):
        return super().save_metrics(beta=self.beta, **kwargs)

    # ---------------- training ----------------

    def train_epoch(self, train_loader):
        if self.criterion is None:
            raise ValueError("Criterion is not set")
        if self.optimizer is None:
            raise ValueError("Optimizer is not set")
        if self.scheduler is None:
            raise ValueError("Scheduler is not set")
        self.model.train()

        metrics = {"loss": 0.0, "correct": 0, "total": 0}
        for data, target, _, _ in train_loader:
            data, target = (
                data.to(self.device, non_blocking=True),
                target.to(self.device),
            )

            output = self.model(data)
            loss = self.criterion(output, target)

            if self.use_replay and len(self.buffer) > 0 and self.current_task > 0:
                bData, bTarget = self.buffer.sample(
                    self.config["replay_size"],
                    self.device,
                    draw="classification",
                )
                bOutput = self.model(bData)
                loss = loss + self.beta * self.criterion(bOutput, bTarget)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # The buffer is only filled once per task (at coreset selection),
            # so add() never advances its step counter during training. Tick
            # per batch, otherwise every sampling row in the task collapses
            # onto a single step and exposure-over-time is unreadable.
            self.buffer.tick()

            batch_size = data.size(0)
            metrics["loss"] += loss.item() * batch_size  # scale loss by batch size
            metrics["total"] += batch_size

            pred = output.argmax(dim=1, keepdim=True)
            metrics["correct"] += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = metrics["loss"] / metrics["total"]
        accuracy = 100.0 * metrics["correct"] / metrics["total"]
        return {"loss": avg_loss, "accuracy": accuracy}

    # ---------------- coreset ----------------

    def select_coreset(self, train_loader, val_loader):
        gm_logger = logging.getLogger(__name__)
        dataloader = GradMatchDataLoader(
            self.model,
            train_loader,
            val_loader,
            self.gm_config,
            gm_logger,
            batch_size=self.config["batch_size"],
            shuffle=self.gm_config["shuffle"],
            pin_memory=True,
        )

        logger.info("Selecting coreset using GradMatch...")
        idx, w = dataloader._resample_subset_indices()
        # Membership test runs once per dataset row; a list here makes the
        # loop O(N*M) and dominates the whole task at 1.2M rows.
        selected = set(int(i) for i in idx)

        images = []
        target = []
        img_path = []
        demographic_lists = {col: [] for col in DEMOGRAPHIC_COLUMNS}

        for i_ in tqdm(range(len(train_loader.dataset)), desc="Gathering coreset"):
            if i_ not in selected:
                continue
            image, label, image_path, demographics = train_loader.dataset[i_]
            images.append(image)
            target.append(int(label))
            img_path.append(image_path)
            for col in DEMOGRAPHIC_COLUMNS:
                if col not in demographics:
                    raise KeyError(
                        f"demographic column {col!r} missing for {image_path}"
                    )
                demographic_lists[col].append(int(demographics[col]))

        if not images:
            raise RuntimeError("GradMatch returned an empty coreset")

        images_stack = torch.stack(images)
        # torch.tensor(..., dtype=torch.long), NOT torch.Tensor(): the legacy
        # constructor always yields float32, and cross_entropy has no float
        # target kernel. The demographic codes must stay integral too, or the
        # memlog writes 1.0 where the annotation pipeline wrote 1.
        target_stack = torch.tensor(target, dtype=torch.long)
        demographics_stack = {
            col: torch.tensor(vals, dtype=torch.long)
            for col, vals in demographic_lists.items()
        }

        if target_stack.shape[0] != len(img_path):
            raise ValueError(
                f"Mismatch in number of examples {target_stack.shape[0]} "
                f"and paths {len(img_path)}"
            )
        logger.info("Coreset selection completed: {} samples", len(img_path))

        return images_stack, target_stack, img_path, demographics_stack

    # ---------------- task loop ----------------

    def _after_epochs(self, train_loader, val_loader):
        if self.config.get("use_coreset", False):
            images, targets, image_paths, demographics = self.select_coreset(
                train_loader, val_loader
            )
            self.buffer.add(
                images.detach().cpu(),
                targets.detach().cpu(),
                image_paths,
                metas=demographics,
            )

    def finalize(self) -> None:
        """Flush the last task's memlog. Call after the final task."""
        self.buffer.finalize()
