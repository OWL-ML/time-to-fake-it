"""
SOYOTrainer for the AI_FACE per-task-directory pipeline.

Adapted from the original SOYOTrainer: the SOYOPromptLogger machinery
(set_true_task / set_router / record_activations and _clamped_router)
is removed and all ``print`` progress lines are converted to loguru.
Training behaviour (both phases, GMM feature replay, routing,
checkpointing) is unchanged.

Two phases per task:
  Phase 1 - train the current task's prompt + adapters + head
            (base PromptCLTrainer.train_epoch: local targets, model.eval()).
  Phase 2 - _compress_train_resample:
            (a) train soyo_network (domain classifier) on real current-task
                features mixed with synthetic features sampled from stored
                GMMs of past tasks (generative feature replay),
            (b) fit and store a GMM of the current task's features for
                future replay.

Routing at inference: soyo_network(extract_vector(x)).argmax per sample.

Target semantics (AI_FACE): every dataset yields LOCAL targets in
[0, class_num). No masking, no remapping, no `target // class_num` —
the domain label for phase 2 is simply the current task index.

Numerical-error policy: sklearn's GMM internals and scipy's logsumexp
compute exp(log-probabilities); underflow to zero is the mathematically
correct result there, not an error. Some audit/logging modules in this
codebase may set np.seterr to raise-mode globally, so every GMM
fit/sample/score path in this file is wrapped in
np.errstate(under="ignore"). The real fix is to scope any
np.seterr(all="raise") at its source (grep for `seterr`) so it raises on
over/invalid/divide but ignores `under`.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from torch.nn import functional as F

from .prompt_cl_trainer import PromptCLTrainer

# ----------------------------------------------------------------------
# GMM feature compression / generative replay
# (Note: this is a Gaussian-mixture generative model of the feature
# distribution, not PCA — only the GMM parameters are stored per task.)
# ----------------------------------------------------------------------


def feature_compression(features, n_components):
    """Fit a GMM robustly on extracted features.

    Guards against non-finite rows, covariance singularities, and
    raise-mode floating-point underflow that can occur on low-diversity
    batches/domains.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Expected features shape [N, D], got {features.shape}")

    finite_mask = np.isfinite(features).all(axis=1)
    n_bad = int((~finite_mask).sum())
    if n_bad > 0:
        logger.warning(
            "[SOYO net] dropping {} non-finite feature rows before GMM fit", n_bad
        )
        features = features[finite_mask]

    n_samples = features.shape[0]
    if n_samples == 0:
        raise ValueError("No valid features available for GMM compression.")

    n_components_eff = min(int(n_components), n_samples)

    # Primary attempt: full covariance (reference behavior) with small
    # regularization.
    gmm = GaussianMixture(
        n_components=n_components_eff,
        covariance_type="full",
        init_params="kmeans",
        reg_covar=1e-6,
    )
    try:
        with np.errstate(under="ignore"):
            gmm.fit(features)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as e:
        # Fallback for ill-conditioned covariance on low-diversity features,
        # or for raise-mode FP errors escaping through numpy paths not
        # covered by the errstate context (e.g. non-underflow categories).
        logger.warning(
            "[SOYO net] full-cov GMM fit failed ({}); retrying with diag covariance",
            e,
        )
        gmm = GaussianMixture(
            n_components=n_components_eff,
            covariance_type="diag",
            init_params="kmeans",
            reg_covar=1e-5,
        )
        with np.errstate(under="ignore"):
            gmm.fit(features)

    return (gmm.means_, gmm.covariances_, gmm.weights_)


def random_sampling(n_samples, compression, n_components):
    # n_components argument kept for API compatibility; use stored compression
    # shape as source of truth to support adaptive component counts.
    gmm_means, gmm_covariances, gmm_weights = compression
    n_components_eff = int(np.asarray(gmm_weights).shape[0])

    # Infer covariance type from stored covariance tensor shape.
    cov = np.asarray(gmm_covariances)
    covariance_type = "full" if cov.ndim == 3 else "diag"

    gmm = GaussianMixture(
        n_components=n_components_eff, covariance_type=covariance_type
    )
    gmm.means_ = gmm_means
    gmm.covariances_ = gmm_covariances
    gmm.weights_ = gmm_weights

    # Keep precision factors consistent with sklearn internals; useful for
    # downstream score/predict paths when parameters are set manually.
    # Sampling and any score/predict on a manually-assembled GMM go through
    # the same exp(log-prob) machinery as fit, so guard underflow here too.
    with np.errstate(under="ignore"):
        gmm.precisions_cholesky_ = _compute_precision_cholesky(cov, covariance_type)
        reconstructed_features = gmm.sample(n_samples)[0]
    return reconstructed_features


class soyo_network(nn.Module):
    """Linear domain classifier over frozen-backbone CLS features."""

    def __init__(self, args):
        super().__init__()
        self.soyo_clf = nn.Linear(args["soyo_dim"], args["total_sessions"])

    def forward(self, image_features):
        return self.soyo_clf(image_features)


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------


class SOYOTrainer(PromptCLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert "soyo_dim" in self.config and "total_sessions" in self.config, (
            "SOYOTrainer requires config['soyo_dim'] (=768 for ViT-B) and "
            "config['total_sessions'] (=number of tasks)"
        )
        self.soyo_network = soyo_network(self.config).to(self.device)
        self.domain_compression = []  # stored GMM params, one per task
        self.n_components = self.config.get("n_components", 2)
        self.soyo_epoch = self.config["soyo_epoch"]
        self.soyo_lr = self.config["soyo_lr"]
        self.soyo_weight_decay = self.config["soyo_weight_decay"]

        # Optional phase-2 speedup: extract features once and reuse across
        # soyo epochs instead of re-running the frozen backbone every epoch.
        # Default False to preserve reference behavior — the phase-2 loader
        # is the *augmented* train loader, so caching freezes one augmentation
        # draw per sample and changes the effective training distribution.
        # Safe to enable when the phase-2 transform is deterministic
        # (eval-style transform, no random augmentation).
        self.soyo_cache_features = bool(self.config.get("soyo_cache_features", False))

    # ------------------------------------------------------------------
    # Phase 1: standard task training
    # ------------------------------------------------------------------

    def _before_task(self, task_id: int):
        self.model.update_fc()
        self.model.to(self.device)

        t = self.model.numtask - 1
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if (
                f"prompt_pool.{t}." in name
                or name.endswith(f"prompt_pool.{t}.weight")
                or f"classifier.{t}." in name
                or f"down_pool.{t}." in name
                or f"up_pool.{t}." in name
            ):
                param.requires_grad_(True)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable)

        train_ds = self.train_datasets[task_id]
        task_path = getattr(train_ds, "root", None)
        task_name = (
            self.task_names[task_id]
            if task_id < len(self.task_names)
            else f"task_{task_id}"
        )
        if task_path:
            inferred_name = os.path.basename(os.path.dirname(task_path.rstrip("/")))
            if inferred_name:
                task_name = inferred_name

        logger.info("Task name: {}", task_name)
        logger.info("Task dataset path: {}", task_path if task_path else "<unknown>")
        logger.info("Trainable parameters this task: {:,}", n_trainable)

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

    # train_epoch: inherited from PromptCLTrainer.
    # Local targets, current-task forward, model.eval() — nothing to override.

    # ------------------------------------------------------------------
    # Phase 2: soyo_network training + GMM feature compression
    # ------------------------------------------------------------------

    def _after_task(self, task_id: int):
        # Phase 2 uses the augmented train loader (matches the reference)
        train_loader = self._make_loader(self.train_datasets[task_id], shuffle=True)
        self._compress_train_resample(train_loader, task_id)

        self.save_checkpoint()

    def _extract_all_features(self, dataloader) -> torch.Tensor:
        """Frozen-backbone CLS features for every sample in the loader."""
        self.model.eval()
        feats = []
        with torch.no_grad():
            for batch_idx, (_, data, _) in enumerate(dataloader):
                data = data.to(self.device)
                f = self.model.extract_vector(data).to(torch.float32)
                finite_mask = torch.isfinite(f)
                if not finite_mask.all():
                    n_total = f.numel()
                    n_bad = int((~finite_mask).sum().item())
                    n_nan = int(torch.isnan(f).sum().item())
                    n_inf = int(torch.isinf(f).sum().item())

                    finite_vals = f[finite_mask]
                    if finite_vals.numel() > 0:
                        fmin = float(finite_vals.min().item())
                        fmax = float(finite_vals.max().item())
                        stats_msg = f"finite_min={fmin:.4e}, finite_max={fmax:.4e}"
                    else:
                        stats_msg = "no finite values in feature tensor"

                    logger.error(
                        "[SOYO net] non-finite features at batch {}: "
                        "bad={}/{}, nan={}, inf={}, {}",
                        batch_idx,
                        n_bad,
                        n_total,
                        n_nan,
                        n_inf,
                        stats_msg,
                    )
                    raise ValueError(
                        "Non-finite features detected in _extract_all_features; "
                        "check backbone outputs upstream."
                    )
                feats.append(f)
        return torch.cat(feats, dim=0)

    def _iter_feature_batches(self, dataloader, cached_features=None):
        """Yield float32 feature batches on-device for phase-2 training.

        With cached_features set, iterate over precomputed features in
        loader-sized chunks (one backbone pass total). Otherwise run the
        frozen backbone per batch (reference behavior; re-draws any
        train-time augmentation each epoch).
        """
        if cached_features is not None:
            bs = dataloader.batch_size
            for start in range(0, len(cached_features), bs):
                yield cached_features[start : start + bs].to(self.device)
            return

        for _, data, _ in dataloader:
            data = data.to(self.device)
            # IMPORTANT: compute under no_grad, but yield *after* exiting the
            # context. Yielding inside the context keeps grad mode disabled in
            # the caller until the generator resumes, which would break
            # soyo_network training (loss has no grad_fn).
            with torch.no_grad():
                feature = self.model.extract_vector(data).to(torch.float32)
            yield feature

    def _compress_train_resample(self, dataloader, task_id: int):
        # --- Resample synthetic features from past tasks' stored GMMs ---
        rs_features, rs_domain_targets = None, None
        if len(self.domain_compression) > 0:
            rs_feat_list, rs_tgt_list = [], []
            n_samples = len(dataloader.dataset)
            for domain_idx, compression in enumerate(self.domain_compression):
                rs_feat = random_sampling(
                    n_samples=n_samples,
                    compression=compression,
                    n_components=self.n_components,
                )
                rs_feat_list.append(rs_feat)
                rs_tgt_list.append(np.full((n_samples,), domain_idx))
            rs_features = torch.tensor(np.vstack(rs_feat_list), dtype=torch.float32)
            rs_domain_targets = torch.tensor(np.hstack(rs_tgt_list))

        # max(1, ...): with small (e.g. debug) batch sizes and many past
        # tasks, integer division can hit 0 — the current task would then be
        # entirely absent from phase-2 batches and the router would silently
        # never learn the newest domain.
        num_new = max(1, dataloader.batch_size // (1 + len(self.domain_compression)))
        num_rs = dataloader.batch_size - num_new
        if rs_features is not None:
            logger.info("[SOYO net] {} replayed + {} new per batch", num_rs, num_new)

        # --- Optional one-shot feature extraction (see __init__ caveat) ---
        cached_features = None
        if self.soyo_cache_features:
            logger.info(
                "[SOYO net] soyo_cache_features=True: extracting features once "
                "and reusing across soyo epochs (assumes deterministic transform)"
            )
            cached_features = self._extract_all_features(dataloader).cpu()

        # --- Train only the soyo_clf weights ---
        for name, param in self.soyo_network.named_parameters():
            param.requires_grad_("soyo_clf" in name)

        optimizer_soyo = torch.optim.Adam(
            self.soyo_network.parameters(),
            lr=self.soyo_lr,
            weight_decay=self.soyo_weight_decay,
        )
        scheduler_soyo = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_soyo, T_max=self.soyo_epoch
        )

        self.soyo_network.train()
        for epoch in range(self.soyo_epoch):
            correct, total, losses, n_batches = 0, 0, 0.0, 0
            # Defensive guard: phase-2 needs gradients on soyo_network params
            # even if some upstream code accidentally disabled global grad mode.
            # NOTE: this does NOT protect against a generator suspended inside
            # no_grad (the flag is set once at block entry, not per statement);
            # the hoisted yield in _iter_feature_batches is the real fix there.
            with torch.enable_grad():
                for feature in self._iter_feature_batches(dataloader, cached_features):
                    # Domain label = current task index (targets are local
                    # per-task labels and carry NO domain information)
                    domain_targets = torch.full(
                        (feature.shape[0],),
                        task_id,
                        dtype=torch.long,
                        device=self.device,
                    )

                    # Mix real current-task + replayed past-task features
                    if rs_features is not None:
                        sel_new = torch.randperm(len(feature))[:num_new]
                        sel_rs = torch.randperm(len(rs_features))[:num_rs]
                        final_feat = torch.cat(
                            [feature[sel_new], rs_features[sel_rs].to(self.device)]
                        )
                        final_tgt = torch.cat(
                            [
                                domain_targets[sel_new],
                                rs_domain_targets[sel_rs].to(self.device),
                            ]
                        )
                    else:
                        final_feat = feature
                        final_tgt = domain_targets

                    domain_logits = self.soyo_network(final_feat)
                    loss = F.cross_entropy(domain_logits, final_tgt)
                    losses += loss.item()
                    n_batches += 1
                    optimizer_soyo.zero_grad()
                    loss.backward()
                    optimizer_soyo.step()

                    correct += domain_logits.argmax(1).eq(final_tgt).sum().item()
                    total += len(final_tgt)

            scheduler_soyo.step()
            if total > 0:
                logger.info(
                    "[SOYO net] epoch {} | loss {:.4f} | acc {:.2f}%",
                    epoch,
                    losses / max(n_batches, 1),
                    100.0 * correct / total,
                )

        # --- Fit and store a GMM of this task's feature distribution ---
        if cached_features is not None:
            features = cached_features.numpy()
        else:
            features = self._extract_all_features(dataloader).cpu().numpy()
        compression = feature_compression(features, n_components=self.n_components)
        self.domain_compression.append(compression)
        logger.info(
            "[SOYO net] stored GMM for task {} ({} total)",
            task_id,
            len(self.domain_compression),
        )

    # ------------------------------------------------------------------
    # Inference: routing via soyo_network
    # ------------------------------------------------------------------

    def _route_tasks(self, feat: torch.Tensor) -> torch.Tensor:
        """Per-sample task selection via the learned linear domain
        classifier (soyo_network), clamped to tasks seen so far (the
        classifier head covers all total_sessions outputs from the start).

        Despite the base-class hook name `_knn_select`, this is NOT a
        kNN — it is a learned linear router over frozen CLS features.
        """
        self.soyo_network.eval()
        with torch.no_grad():
            domain_logits = self.soyo_network(feat.to(torch.float32))
            n_seen = self.model.numtask
            domain_logits = domain_logits[:, :n_seen]
        return domain_logits.argmax(dim=1)

    def _knn_select(self, feat: torch.Tensor) -> torch.Tensor:
        """Base-class routing hook; delegates to _route_tasks. Kept under
        this name because PromptCLTrainer.evaluate calls _knn_select."""
        return self._route_tasks(feat)

    # evaluate: inherited from PromptCLTrainer.
    # Local targets => plain equality is correct; interface returns a dict,
    # which the base handles.

    # ------------------------------------------------------------------
    # Checkpointing: persist routing state
    # ------------------------------------------------------------------

    def save_checkpoint(self, filepath=None, **kwargs):
        super().save_checkpoint(
            filepath=filepath,
            soyo_network_state=self.soyo_network.state_dict(),
            domain_compression=self.domain_compression,
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
        if "soyo_network_state" in checkpoint:
            self.soyo_network.load_state_dict(checkpoint["soyo_network_state"])
        if "domain_compression" in checkpoint:
            self.domain_compression = checkpoint["domain_compression"]
        if "numtask" in checkpoint:
            self.model.numtask = checkpoint["numtask"]
