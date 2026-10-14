# models/soyo_model.py
"""
SOYOModel on top of models.vit_backbone.ViTBackbone (torchvision ViT-B/16).

Same architecture and API as the timm-based version:
  - per-task prompt tokens (nn.Linear(embed_dim, prompt_length).weight)
  - per-task deep adapters (n_layers x (down, up) bottleneck pairs)
  - per-task classifier heads
  - extract_vector / forward / interface / update_fc / copy / freeze

The trainer (SOYOTrainer) is unchanged: it only touches this API.

Expected args keys:
  total_sessions, prompt_length, image_dim (=768), hidden_dim, class_num
"""

import copy

import torch
import torch.nn as nn

from models.vit_backbone_timm import ViTBackbone


class SOYOModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.image_encoder = ViTBackbone(pretrained=True, freeze=True)

        self.total_sessions = args["total_sessions"]
        if "class_num" in args:
            self.class_num = args["class_num"]
        elif args.get("dataset") == "ai_face":
            self.class_num = 2
        else:
            raise KeyError(
                "SOYOModel requires args['class_num'] unless dataset='ai_face' "
                "(default class_num=2)."
            )
        image_dim = args["image_dim"]
        hidden_dim = args["hidden_dim"]
        prompt_length = args["prompt_length"]
        n_layers = self.image_encoder.n_layers  # 12 for ViT-B/16

        assert image_dim == self.image_encoder.embed_dim, (
            f"image_dim={image_dim} but backbone embed_dim="
            f"{self.image_encoder.embed_dim}"
        )

        # Per-task classifier heads (pre-allocated for all sessions)
        self.classifier = nn.ModuleList(
            [
                nn.Linear(image_dim, self.class_num, bias=True)
                for _ in range(self.total_sessions)
            ]
        )

        # Per-task prompts: weight of a bias-free Linear, shape [L, D]
        self.prompt_pool = nn.ModuleList(
            [
                nn.Linear(image_dim, prompt_length, bias=False)
                for _ in range(self.total_sessions)
            ]
        )

        # Per-task deep adapters: (down, up) pair per transformer block
        self.down_pool = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(image_dim, hidden_dim) for _ in range(n_layers)]
                )
                for _ in range(self.total_sessions)
            ]
        )
        self.up_pool = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(hidden_dim, image_dim) for _ in range(n_layers)]
                )
                for _ in range(self.total_sessions)
            ]
        )
        for layer in range(n_layers):
            for i in range(self.total_sessions):
                nn.init.xavier_uniform_(self.down_pool[i][layer].weight)
                nn.init.zeros_(self.down_pool[i][layer].bias)
                nn.init.xavier_uniform_(self.up_pool[i][layer].weight)
                nn.init.zeros_(self.up_pool[i][layer].bias)

        # numtask = tasks started so far; current task index is numtask - 1.
        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.embed_dim

    def extract_vector(self, image):
        """Raw frozen-backbone CLS features (no prompt, no adapters),
        L2-normalized. Input to soyo_network routing and GMM compression."""
        feats = self.image_encoder(image)
        # Guard against exact/near-zero norms only (does not sanitize NaNs
        # coming from upstream backbone outputs).
        denom = feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats / denom

    def forward(self, image):
        """Training-time forward with the CURRENT task's modules."""
        t = self.numtask - 1
        feats = self.image_encoder.forward_with_prompts(
            image,
            prompts=self.prompt_pool[t].weight,
            up_pool=self.up_pool[t],
            down_pool=self.down_pool[t],
        )
        return {"logits": self.classifier[t](feats)}

    def interface(self, image, selection):
        """Inference-time forward with per-sample routing.

        selection: LongTensor [B] of task indices (from soyo_network).
        Runs the backbone once per SEEN task (adapters are task-wide) and
        gathers per-sample results.
        """
        n_seen = self.numtask
        B = image.shape[0]
        arange_b = torch.arange(B, device=image.device)

        # Per-sample prompt selection: [B, L, D]
        instance_batch = torch.stack(
            [self.prompt_pool[i].weight for i in range(n_seen)], dim=0
        )[selection, :, :]

        feat_list = [
            self.image_encoder.forward_with_prompts(
                image,
                prompts=instance_batch,
                up_pool=self.up_pool[i],
                down_pool=self.down_pool[i],
            )
            for i in range(n_seen)
        ]
        image_features = torch.stack(feat_list, dim=0)[selection, arange_b]

        logits_list = [self.classifier[i](image_features) for i in range(n_seen)]
        logits = torch.stack(logits_list, dim=0)[selection, arange_b]

        return {"logits": logits}

    def update_fc(self, nb_classes=None):
        """Advance the task pointer (all modules pre-allocated)."""
        assert self.numtask < self.total_sessions, (
            f"update_fc called {self.numtask + 1} times but total_sessions="
            f"{self.total_sessions}"
        )
        self.numtask += 1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()
        return self
