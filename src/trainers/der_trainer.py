import torch

from buffers import DERReplayBuffer

from .replay_trainer import ReplayTrainer


class DERPPTrainer(ReplayTrainer):
    def __init__(
        self,
        model,
        model_name,
        config,
        device,
        criterion=None,
        optimizer=None,
        scheduler=None,
        store_logits=False,
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
        self.alpha = config.get("alpha", 0.5)

    def _make_buffer(self):
        return DERReplayBuffer(
            self.config.get("capacity", 1000),
            store_logits=True,
            seed=self.config.get("base_seed", 42),
        )

    def save_metrics(self, **kwargs):
        return super().save_metrics(alpha=self.alpha, beta=self.beta, **kwargs)

    # resuming from task 1 and populating buffer before starting training from task 2 onwards
    def preload_task_into_buffer(self, task_data_loader):
        self.model.eval()
        with torch.no_grad():
            for data, target, image_paths, demographics in task_data_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                # compute logits with the pretrained Task1 model
                logits = self.model(data)
                # Store all three (DER++)
                self.buffer.add(
                    examples=data.detach().cpu(),
                    labels=target.detach().cpu(),
                    logits=logits.detach().cpu(),
                    image_paths=image_paths,
                    metas=demographics,
                )
        self.preloaded_from_task = 0

    def train_epoch(self, train_loader):
        if self.criterion is None:
            raise ValueError("Criterion is not set")
        if self.optimizer is None:
            raise ValueError("Optimizer is not set")
        if self.scheduler is None:
            raise ValueError("Scheduler is not set")

        self.model.train()
        metrics = {"loss": 0.0, "correct": 0, "total": 0}
        for data, target, image_paths, demographics in train_loader:
            data, target = (
                data.to(self.device, non_blocking=True),
                target.to(self.device),
            )
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            if len(self.buffer) > 0 and self.current_task > 0:
                bData1, _, bLogits1 = self.buffer.sample(
                    self.config["replay_size"], self.device, draw="distillation"
                )
                bOutput1 = self.model(bData1)
                distill_loss = self.alpha * torch.nn.functional.mse_loss(
                    bOutput1, bLogits1
                )
                loss += distill_loss
                bData2, bTarget2, _ = self.buffer.sample(
                    self.config["replay_size"], self.device, draw="classification"
                )
                bOutput2 = self.model(bData2)
                replay_loss = self.beta * self.criterion(bOutput2, bTarget2)
                loss += replay_loss

            loss.backward()
            self.optimizer.step()
            # skip adding if the samples were already preloaded in buffer
            if self.preloaded_from_task != self.current_task:
                self.buffer.add(
                    examples=data.detach().cpu(),
                    labels=target.detach().cpu(),
                    logits=output.detach().cpu(),
                    image_paths=image_paths,
                    metas=demographics,
                )
            else:
                self.buffer.tick()

            batch_size = data.size(0)
            metrics["loss"] += loss.item() * batch_size  # scale loss by batch size
            metrics["total"] += batch_size

            pred = output.argmax(dim=1, keepdim=True)
            metrics["correct"] += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = metrics["loss"] / metrics["total"]
        accuracy = 100.0 * metrics["correct"] / metrics["total"]
        return {"loss": avg_loss, "accuracy": accuracy}
