"""
SPromptsTrainer for the AI_FACE per-task-directory pipeline.

Adapted from the original SPromptsTrainer: the SPromptsLogger machinery
(record_activations) is removed and all ``print`` lines are converted to
loguru. Training behaviour (freezing schedule, KMeans key building, L1
routing, checkpointing) is unchanged — ``_knn_select`` is kept
byte-for-byte; ``_routing_scores`` is argmax-equivalent to it and is
used by utils/run_inference_prompt.py.
"""

import torch
from loguru import logger
from sklearn.cluster import KMeans

from .prompt_cl_trainer import PromptCLTrainer

class SPromptsTrainer(PromptCLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One tensor [K, D] per seen task (KMeans centers for routing)
        self.all_keys: list[torch.Tensor] = []
        self.n_clusters = int(self.config.get("sprompts_kmeans_clusters", 5))

    def _before_task(self, task_id: int):
        self.model.update_fc(nb_classes=None)
        self.model.to(self.device)

        current = self.model.numtask - 1

        # Freeze everything, then unfreeze current task prompt + classifier head.
        for p in self.model.parameters():
            p.requires_grad_(False)

        trainable = []
        for module in (
            self.model.prompt_pool[current],
            self.model.classifier_pool[current],
        ):
            params = list(module.parameters())
            for p in params:
                p.requires_grad_(True)
            trainable.extend(params)

        # Task 0 uses init schedule; later tasks use regular schedule.
        if task_id == 0:
            lr = self.config["init_lr"]
            wd = self.config["init_weight_decay"]
            epochs = self.config["init_epochs"]
        else:
            lr = self.config["lr"]
            wd = self.config["weight_decay"]
            epochs = self.config["epochs"]

        self._task_epochs = epochs
        self.optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        # Run-identity output, matching the SOYO/PINA trainers for uniform logs.
        n_trainable = sum(p.numel() for p in trainable)
        train_ds = self.train_datasets[task_id]
        task_path = getattr(train_ds, "root", None)
        logger.info("Task name: {}", self.task_names[task_id])
        logger.info("Task dataset path: {}", task_path if task_path else "<unknown>")
        logger.info("Trainable parameters this task: {:,}", n_trainable)

    def _after_task(self, task_id: int):
        # Build routing keys on eval-transform data to match inference distribution.
        analysis_ds = self._get_analysis_dataset(task_id)
        loader = self._make_loader(analysis_ds, shuffle=False)

        self._build_task_key(task_id, loader)

        self.save_checkpoint()

    @torch.no_grad()
    def _build_task_key(self, task_id: int, loader):
        """KMeans over extract_vector embeddings for current task."""
        self.model.eval()
        feats = []

        for _, data, _ in loader:
            data = data.to(self.device)
            feat = self.model.extract_vector(data).to(torch.float32)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            feats.append(feat.cpu())

        if len(feats) == 0:
            raise ValueError("No features available to build SPrompts task key.")

        feat_np = torch.cat(feats, dim=0).numpy()
        n_clusters_eff = min(self.n_clusters, feat_np.shape[0])
        if n_clusters_eff < 1:
            raise ValueError(
                "Invalid number of clusters for SPrompts task key building."
            )

        # Seed KMeans from base_seed so routing-key initialization varies
        # across seed-variance runs along with everything else (a fixed
        # random_state=0 would silently hold it constant across seeds);
        # + task_id keeps per-task inits distinct within a run.
        kmeans_seed = int(self.config.get("base_seed", 0)) + int(task_id)
        km = KMeans(n_clusters=n_clusters_eff, random_state=kmeans_seed).fit(feat_np)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32).to(self.device)

        if task_id < len(self.all_keys):
            self.all_keys[task_id] = centers
        else:
            self.all_keys.append(centers)

    def _routing_scores(self, feat: torch.Tensor) -> torch.Tensor:
        """Routing scores [B, n_seen] for LOGGING only; higher is better
        (negative min L1 distance per task). argmax over these is exactly
        argmin over the distances _knn_select computes, so the logged
        winner always matches eval-time routing; _knn_select itself is
        kept byte-for-byte from the reference."""
        n_seen = min(self.model.numtask, len(self.all_keys))
        if n_seen == 0:
            raise RuntimeError(
                "S-Prompts routing called before any task keys were built; "
                "evaluate is only valid after _after_task (or after loading "
                "a checkpoint containing all_keys)."
            )

        feat = feat.to(self.device, dtype=torch.float32)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        scores = []
        for task_centers in self.all_keys[:n_seen]:
            centers = task_centers.to(self.device)
            scores.append(-torch.cdist(feat, centers, p=1).min(1)[0])

        return torch.stack(scores, dim=1)

    def _knn_select(self, feat: torch.Tensor) -> torch.Tensor:
        """
        L1-over-dim distance to KMeans centers (same metric as backup SPrompts;
        note this is L1, not the L2 nearest-centroid distance the original
        S-Prompts paper describes — documented as a reference-implementation
        divergence in the methods notes). Min over centers per task, then
        argmin over seen tasks.
        """
        n_seen = min(self.model.numtask, len(self.all_keys))
        if n_seen == 0:
            raise RuntimeError(
                "S-Prompts routing called before any task keys were built; "
                "evaluate is only valid after _after_task (or after loading "
                "a checkpoint containing all_keys)."
            )

        feat = feat.to(self.device, dtype=torch.float32)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        taskselection = []
        for task_centers in self.all_keys[:n_seen]:
            centers = task_centers.to(self.device)
            # Reference metric kept exactly: L1 over absolute differences
            # (cdist p=1 is the vectorized equivalent of the per-center loop).
            taskselection.append(torch.cdist(feat, centers, p=1).min(1)[0])

        return torch.vstack(taskselection).min(0)[1]

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
