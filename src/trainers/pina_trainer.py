"""
PINATrainer for the AI_FACE per-task-directory pipeline.

Adapted from the original PINATrainer: the PINAPromptLogger machinery
(set_router / record_activations) is removed and all ``print`` lines are
converted to loguru. Training behaviour (KMeans key building, L1
routing, checkpointing) is unchanged.

Freezing schedule (matches the reference PINA._train): the unified
classifier trains on task 0 only, together with prompt_pool[0]; from
task 1 onward only the current task's prompt and adapters train, and
the shared head stays frozen. PINA's forgetting is therefore purely
routing-mediated. Earlier runs of this trainer trained the head on every
task — treat those results as the "shared head trained throughout"
ablation, not as PINA.
"""

import torch
from loguru import logger
from sklearn.cluster import KMeans

from .prompt_cl_trainer import PromptCLTrainer

class PINATrainer(PromptCLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One tensor [K, D] per seen task (KMeans centers for routing)
        self.all_keys: list[torch.Tensor] = []
        self.n_clusters = int(self.config.get("pina_kmeans_clusters", 5))

    def _before_task(self, task_id: int):
        # Pre-allocated pools: update_fc just advances the task counter.
        self.model.update_fc(nb_classes=None)
        self.model.to(self.device)

        current = self.model.numtask - 1

        # Freeze everything, then re-enable exactly what this task trains.
        for p in self.model.parameters():
            p.requires_grad_(False)

        trainable_modules = [self.model.prompt_pool[current]]
        # Task 0 adapters are zero-init and treated as a frozen identity;
        # ca_mode does not change WHICH modules train, only their structure,
        # and .parameters() recurses into deep per-layer ModuleLists.
        if current > 0:
            trainable_modules.append(self.model.down_pool[current])
            trainable_modules.append(self.model.up_pool[current])
        # Unified classifier trains on task 0 only, then stays frozen
        # (reference PINA._train). Later tasks adapt only through their
        # own prompt + adapters, so the shared head never drifts.
        if current == 0:
            trainable_modules.append(self.model.unified_classifier)

        # Task-0 vs later-task schedule (same convention as SOYO/SPrompts)
        if task_id == 0:
            lr = self.config.get("init_lr", self.config["lr"])
            wd = self.config.get("init_weight_decay", self.config["weight_decay"])
            epochs = self.config.get("init_epochs", self.config["epochs"])
        else:
            lr = self.config["lr"]
            wd = self.config["weight_decay"]
            epochs = self.config["epochs"]

        self._task_epochs = epochs

        param_groups = []
        for module in trainable_modules:
            params = list(module.parameters())
            for p in params:
                p.requires_grad_(True)
            param_groups.append({"params": params, "lr": lr})

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        # Run-identity output, matching the SOYO trainer for uniform logs.
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable)
        train_ds = self.train_datasets[task_id]
        task_path = getattr(train_ds, "root", None)
        logger.info("Task name: {}", self.task_names[task_id])
        logger.info("Task dataset path: {}", task_path if task_path else "<unknown>")
        logger.info("Trainable parameters this task: {:,}", n_trainable)
        logger.info(
            "Unified classifier: {}",
            "trainable (task 0)" if current == 0 else "frozen",
        )

    def _after_task(self, task_id: int):
        # Eval-transform dataset for key building: keys must live in the
        # same feature distribution as the eval-time queries.
        analysis_ds = self._get_analysis_dataset(task_id)
        loader = self._make_loader(analysis_ds, shuffle=False)

        self._build_task_key(task_id, loader)

        self.save_checkpoint()

    @torch.no_grad()
    def _build_task_key(self, task_id: int, loader):
        """Build KMeans routing centers for the just-trained task."""
        self.model.eval()
        feats = []

        for _, data, _ in loader:
            data = data.to(self.device)
            feat = self.model.extract_vector(data).to(torch.float32)
            # Match reference behavior: normalize before clustering.
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            feats.append(feat.cpu())

        if len(feats) == 0:
            raise ValueError("No features available to build PINA task key.")

        feat_np = torch.cat(feats, dim=0).numpy()
        n_clusters_eff = min(self.n_clusters, feat_np.shape[0])
        if n_clusters_eff < 1:
            raise ValueError("Invalid number of clusters for PINA task key building.")

        # Seed KMeans from base_seed so routing-key initialization varies
        # across seed-variance runs along with everything else (a fixed
        # random_state=0 would silently hold it constant across seeds);
        # + task_id keeps per-task inits distinct within a run.
        kmeans_seed = int(self.config.get("base_seed", 0)) + int(task_id)
        clustering = KMeans(n_clusters=n_clusters_eff, random_state=kmeans_seed).fit(
            feat_np
        )
        centers = torch.tensor(clustering.cluster_centers_, dtype=torch.float32).to(
            self.device
        )

        if task_id < len(self.all_keys):
            self.all_keys[task_id] = centers
        else:
            self.all_keys.append(centers)

    def _routing_scores(self, feat: torch.Tensor) -> torch.Tensor:
        """Return routing scores [B, n_seen] for PINA task selection.

        Score convention: higher is preferred. We therefore use negative
        min L1 distance-to-centers per task.
        """
        n_seen = min(self.model.numtask, len(self.all_keys))
        if n_seen == 0:
            raise RuntimeError(
                "PINA routing called before any task keys were built; "
                "evaluate is only valid after _after_task (or after loading "
                "a checkpoint containing all_keys)."
            )

        feat = feat.to(self.device, dtype=torch.float32)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        scores = []
        for task_centers in self.all_keys[:n_seen]:
            centers = task_centers.to(self.device)
            # Reference metric: min L1 distance to this task's centers.
            min_l1 = torch.cdist(feat, centers, p=1).min(1)[0]
            scores.append(-min_l1)

        return torch.stack(scores, dim=1)

    def _knn_select(self, feat: torch.Tensor) -> torch.Tensor:
        """Task routing by argmax over _routing_scores(feat)."""
        return self._routing_scores(feat).argmax(dim=1)

    def save_checkpoint(self, filepath=None, **kwargs):
        super().save_checkpoint(
            filepath=filepath,
            all_keys=[k.detach().cpu() for k in self.all_keys],
            numtask=self.model.numtask,
            **kwargs,
        )

    def load_checkpoint(self, filepath):
        # If the base implementation returns the loaded state dict, reuse it
        # to avoid deserializing the checkpoint twice; otherwise fall back to
        # a second torch.load (reference behavior).
        checkpoint = super().load_checkpoint(filepath)
        if not isinstance(checkpoint, dict):
            checkpoint = torch.load(
                filepath, map_location=self.device, weights_only=False
            )
        if "all_keys" in checkpoint:
            self.all_keys = [k.to(self.device) for k in checkpoint["all_keys"]]
        if "numtask" in checkpoint:
            self.model.numtask = checkpoint["numtask"]

    def _assert_routing_restored(self, task_id: int):
        if len(self.all_keys) < task_id + 1:
            raise RuntimeError(
                f"Checkpoint for task {task_id} restored {len(self.all_keys)} "
                f"KMeans key set(s); need {task_id + 1}. Routing would fall back "
                "to a subset of tasks and quietly skew the accuracy matrix."
            )
        if self.model.numtask < task_id + 1:
            raise RuntimeError(
                f"model.numtask={self.model.numtask} after loading the task "
                f"{task_id} checkpoint; expected >= {task_id + 1}. "
                "_routing_scores clamps n_seen to numtask, so routing would "
                "ignore later prompts."
            )
