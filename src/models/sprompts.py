import copy

import torch
import torch.nn as nn

from models.vit_backbone_timm import ViTBackbone


class SPromptsModel(nn.Module):
    """
    S-Prompts / SiNet-style prompt pool with per-task classifier heads.

    Adapted for this repo's local-label PromptCL pipeline:
      - pre-allocated prompt pool and classifier pool
      - update_fc() only advances task pointer
      - extract_vector() returns normalized backbone CLS features for KMeans routing
    """

    def __init__(self, args):
        super().__init__()

        self.image_encoder = ViTBackbone(
            pretrained=args.get("pretrained", True), freeze=True
        )

        self.total_sessions = args["total_sessions"]
        if "class_num" in args:
            self.class_num = args["class_num"]
        elif args.get("dataset") == "ai_face":
            self.class_num = 2
        else:
            raise KeyError(
                "SPromptsModel requires args['class_num'] unless dataset='ai_face' "
                "(default class_num=2)."
            )

        self.prompt_length = args["prompt_length"]
        self.embed_dim = self.image_encoder.embed_dim

        self.prompt_pool = nn.ModuleList(
            [
                nn.Linear(self.embed_dim, self.prompt_length, bias=False)
                for _ in range(self.total_sessions)
            ]
        )

        self.classifier_pool = nn.ModuleList(
            [
                nn.Linear(self.embed_dim, self.class_num, bias=True)
                for _ in range(self.total_sessions)
            ]
        )

        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.embed_dim

    @torch.no_grad()
    def extract_vector(self, image):
        feat = self.image_encoder(image)
        return feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(self, image):
        t = self.numtask - 1
        feat = self.image_encoder.forward_with_prompts(
            image,
            prompts=self.prompt_pool[t].weight,
        )
        return {"logits": self.classifier_pool[t](feat), "features": feat}

    @torch.no_grad()
    def interface(self, image, selection):
        """
        Inference-time forward with per-sample task routing.
        Returns logits tensor [B, class_num] (SiNet-style).
        """
        n_seen = self.numtask
        if n_seen <= 0:
            raise RuntimeError("SPromptsModel.interface called before first task.")

        B = image.shape[0]
        selection = selection.to(image.device, dtype=torch.long)

        instance_batch = torch.stack(
            [self.prompt_pool[i].weight for i in range(n_seen)], dim=0
        )[selection, :, :]

        feat = self.image_encoder.forward_with_prompts(image, prompts=instance_batch)

        logits = image.new_zeros(B, self.class_num)
        for t in selection.unique().tolist():
            idx = (selection == t).nonzero(as_tuple=False).view(-1)
            logits[idx] = self.classifier_pool[t](feat.index_select(0, idx))

        return logits

    def update_fc(self, nb_classes=None):
        self.numtask += 1
        if self.numtask > self.total_sessions:
            raise RuntimeError(
                f"numtask={self.numtask} exceeds total_sessions={self.total_sessions}"
            )

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self
