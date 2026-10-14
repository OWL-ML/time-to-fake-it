# models/vit_backbone.py
"""
Frozen ViT-B/16 feature extractor (torchvision weights) with prompt-token
injection and per-block adapter residuals, as needed by SOYO / PINA /
S-Prompts.

Fixes over the previous version:
  1. torchvision's Encoder.forward adds pos_embedding internally, so the old
     forward() added it TWICE. We now bypass Encoder.forward entirely and
     iterate encoder.layers manually (adding pos_embedding exactly once).
  2. The old forward_with_prompts() crashed: after concatenating prompts the
     sequence length no longer matched pos_embedding inside Encoder.forward.
     Manual iteration fixes this too.
  3. Prompts are now inserted BETWEEN the CLS token and patch tokens (the
     SOYO / S-Prompts convention), so CLS stays at index 0.
  4. Optional up_pool/down_pool arguments apply a bottleneck adapter residual
     x + up[i](down[i](x)) after every transformer block (SOYO deep mode).
"""

import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ViTBackbone(nn.Module):
    """
    Frozen ViT-B/16. Returns CLS token embedding [B, 768].

    Note on weights: torchvision's IMAGENET1K_V1 is ImageNet-1k-trained,
    whereas the reference SOYO/S-Prompts implementations use timm's
    ImageNet-21k-pretrained ViT-B/16. Functionally interchangeable here,
    but absolute accuracies will differ from the papers'.
    """

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        vit = vit_b_16(weights=weights)

        self.patch_embedding = vit.conv_proj
        self.class_token = vit.class_token
        self.pos_embedding = vit.encoder.pos_embedding  # [1, N+1, D]
        # Unpack the encoder so we control pos-embed and can hook blocks:
        self.dropout = vit.encoder.dropout
        self.layers = vit.encoder.layers  # 12 EncoderBlocks
        self.ln = vit.encoder.ln  # final LayerNorm
        self.embed_dim = 768
        self.n_layers = len(self.layers)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify, prepend CLS, add positional embedding (exactly once).
        Returns [B, N+1, D]."""
        B = x.shape[0]
        x = self.patch_embedding(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        cls = self.class_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)  # [B, N+1, D]
        return x + self.pos_embedding

    def _run_blocks(self, x, up_pool=None, down_pool=None) -> torch.Tensor:
        """Iterate encoder blocks, optionally applying an adapter residual
        after each block (SOYO deep adapters)."""
        x = self.dropout(x)
        for i, blk in enumerate(self.layers):
            x = blk(x)
            if up_pool is not None:
                assert down_pool is not None
                x = x + up_pool[i](down_pool[i](x))
        return self.ln(x)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Plain frozen-backbone CLS features: [B, 768]."""
        x = self._embed(x)
        x = self._run_blocks(x)
        return x[:, 0]

    def forward_with_prompts(
        self,
        x: torch.Tensor,
        prompts: torch.Tensor,  # [L, D] or [B, L, D]
        up_pool: nn.ModuleList = None,  # n_layers x Linear(hidden, D)
        down_pool: nn.ModuleList = None,  # n_layers x Linear(D, hidden)
    ) -> torch.Tensor:
        """Prompted (and optionally adapted) forward. Prompts are inserted
        between CLS and patch tokens after positional embedding; prompts
        receive no positional embedding. CLS remains at index 0.
        Returns CLS token [B, 768]."""
        x = self._embed(x)  # [B, N+1, D]
        if prompts.dim() == 2:
            prompts = prompts.unsqueeze(0).expand(x.shape[0], -1, -1)
        prompts = prompts.to(x.dtype)
        x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)  # [B, 1+L+N, D]
        x = self._run_blocks(x, up_pool=up_pool, down_pool=down_pool)
        return x[:, 0]
