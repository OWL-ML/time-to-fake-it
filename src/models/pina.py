import torch
import torch.nn as nn

from models.vit_backbone_timm import ViTBackbone


class PINAModel(nn.Module):
    """
    PINA — prompt + adapter continual learner (SOYO family).

    Matches the reference implementation and the interface expected by
    PINATrainer / PINAPromptLogger:

      - Pre-allocated per-task prompts: nn.Linear(embd_dim, prompt_length,
        bias=False); the .weight matrix [prompt_length, embd_dim] IS the
        prompt (same storage trick as SiNet/SOYO — the logger reads
        prompt_pool[t].weight directly).
      - Per-task adapters controlled by ca_mode:
          "shallow": one down/up pair per task, applied post-hoc on the
                     CLS token as a residual offset.
          "deep":    per-layer down/up pairs, injected inside each
                     transformer block by the encoder.
      - Task-0 adapters zero-initialised (identity at start; never trained).
      - Single unified classifier, fixed at init — it never expands.
      - update_fc() only increments numtask (everything pre-allocated).

    Encoder API (models.vit_backbone_timm.ViTBackbone):
      self.backbone(x)                                  -> [B, D] raw CLS
      self.backbone.forward_with_prompts(x, P)          -> [B, D] prompted CLS
      self.backbone.forward_with_prompts(
          x, P, up_pool=..., down_pool=...)             -> [B, D] deep-adapter CLS
      Prompt P may be [L, D]; ViT_Prompts broadcasts it over the batch and
      inserts it between CLS and patch tokens. Deep adapters are applied by
      ViT_Prompts after every block as x = x + up_pool[i](down_pool[i](x))
      on the full token sequence [B, N, D].
    """

    def __init__(self, args):
        super().__init__()
        self.backbone = ViTBackbone(
            pretrained=args.get("pretrained", True), freeze=True
        )
        self.embd_dim = self.backbone.embed_dim  # 768 for ViT-B/16
        self.class_num = args["class_num"]  # 2 for AI_FACE
        self.prompt_length = args["prompt_length"]
        self.total_sessions = args["total_sessions"]  # 11
        self.ca_mode = args["ca_mode"]  # "shallow" | "deep"
        self.bottleneck = args.get("adapter_bottleneck", 64)
        self.n_layers = self.backbone.n_layers  # 12; deep pools MUST
        # match this exactly — ViT_Prompts indexes up_pool[i]/down_pool[i]
        # for every block i with no length check.
        self.numtask = 0

        # ---- prompts -------------------------------------------------
        self.prompt_pool = nn.ModuleList(
            [
                nn.Linear(self.embd_dim, self.prompt_length, bias=False)
                for _ in range(self.total_sessions)
            ]
        )

        # ---- adapters ------------------------------------------------
        if self.ca_mode == "shallow":
            self.down_pool = nn.ModuleList(
                [
                    nn.Linear(self.embd_dim, self.bottleneck, bias=False)
                    for _ in range(self.total_sessions)
                ]
            )
            self.up_pool = nn.ModuleList(
                [
                    nn.Linear(self.bottleneck, self.embd_dim, bias=False)
                    for _ in range(self.total_sessions)
                ]
            )
        elif self.ca_mode == "deep":
            self.down_pool = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            nn.Linear(self.embd_dim, self.bottleneck, bias=False)
                            for _ in range(self.n_layers)
                        ]
                    )
                    for _ in range(self.total_sessions)
                ]
            )
            self.up_pool = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            nn.Linear(self.bottleneck, self.embd_dim, bias=False)
                            for _ in range(self.n_layers)
                        ]
                    )
                    for _ in range(self.total_sessions)
                ]
            )
        else:
            raise ValueError(f"Unknown ca_mode: {self.ca_mode!r}")

        self._init_adapters()

        # ---- unified classifier (fixed, never expands) ----------------
        self.unified_classifier = nn.Linear(self.embd_dim, self.class_num, bias=True)

        # Backbone params are already frozen via ViTBackbone(freeze=True).

    # ------------------------------------------------------------------
    def _init_adapters(self):
        """Task 0: zeros (identity residual). Later tasks: xavier.

        The reference snippet showed this rule for down_pool; it is applied
        to up_pool as well — with down zero-initialised the residual is
        exactly zero at task 0 regardless, so this is safe either way.
        Verify against the reference if up_pool init differs.
        """
        for t in range(self.total_sessions):
            downs = (
                [self.down_pool[t]]
                if self.ca_mode == "shallow"
                else list(self.down_pool[t])
            )
            ups = (
                [self.up_pool[t]]
                if self.ca_mode == "shallow"
                else list(self.up_pool[t])
            )
            for m in downs + ups:
                if t == 0:
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.xavier_uniform_(m.weight)

    # ------------------------------------------------------------------
    def update_fc(self, nb_classes=None):
        """Everything is pre-allocated — just advance the task counter."""
        self.numtask += 1
        if self.numtask > self.total_sessions:
            raise RuntimeError(
                f"numtask={self.numtask} exceeds total_sessions={self.total_sessions}"
            )

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor, task: int) -> torch.Tensor:
        """Prompted (and adapted) CLS features for a single task index."""
        img_prompt = self.prompt_pool[task].weight  # [prompt_length, D]

        if self.ca_mode == "shallow":
            feat = self.backbone.forward_with_prompts(x, img_prompt)
            # Task-0 pair is zero-init and untrained -> residual is exactly 0,
            # so applying it uniformly keeps the code (and logger hooks) simple.
            feat = feat + self.up_pool[task](self.down_pool[task](feat))
        else:  # deep
            feat = self.backbone.forward_with_prompts(
                x,
                img_prompt,
                up_pool=self.up_pool[task],
                down_pool=self.down_pool[task],
            )
        return feat

    def forward(self, x: torch.Tensor) -> dict:
        task = self.numtask - 1
        feat = self._encode(x, task)
        return {"logits": self.unified_classifier(feat), "features": feat}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Raw CLS embedding, no prompt, no adapters — used for KNN task keys."""
        return self.backbone(x)

    @torch.no_grad()
    def interface(self, x: torch.Tensor, selection: torch.Tensor) -> dict:
        """Per-sample inference with the selected task's prompt + adapters.

        Adapters are modules, so unlike SiNet we cannot stack per-sample
        prompt weights into one pass; instead we sub-batch by unique task id
        (at most `numtask` encoder passes, usually 1-2 in practice).
        """
        B = x.size(0)
        logits = x.new_zeros(B, self.class_num)
        features = x.new_zeros(B, self.embd_dim)

        for t in selection.unique().tolist():
            idx = (selection == t).nonzero(as_tuple=False).view(-1)
            feat = self._encode(x.index_select(0, idx), t)
            logits[idx] = self.unified_classifier(feat)
            features[idx] = feat

        return {"logits": logits, "features": features}
