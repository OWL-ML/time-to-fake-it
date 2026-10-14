"""
Timm-based ViT-B/16 backbone for SOYO / PINA / S-Prompts.

This file is designed for two goals:
1) Source-level parity with backup SOYO's ViT path (`ViT_Prompts` +
   `_create_vision_transformer` style usage), and
2) Drop-in compatibility with the current project backbone API
   (`ViTBackbone.forward` / `forward_with_prompts`, `.embed_dim`, `.n_layers`).
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

try:
    import timm
except Exception as e:  # pragma: no cover - dependency guard
    raise ImportError(
        "timm is required for models.vit_backbone_timm. "
        "Install it with `pip install timm`."
    ) from e


class ViT_Prompts(nn.Module):
    """
    Prompt-capable timm ViT wrapper with backup-compatible forward signature.

    Notes:
    - Prompt insertion follows SOYO convention: prompts are inserted between
      CLS and patch tokens, so CLS remains at index 0.
    - Adapter residual (deep mode): x = x + up_pool[i](down_pool[i](x)).
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        init_values=None,
        embed_layer=None,
        norm_layer=None,
        act_layer=None,
        block_fn=None,
        *,
        variant: str = "vit_base_patch16_224",
        pretrained: bool = False,
    ):
        super().__init__()

        # "token" pooling in old timm-style code corresponds to no explicit
        # global pooling on top of CLS token.
        timm_global_pool = "" if global_pool == "token" else global_pool

        # Keep kwargs explicit and stable across timm versions.
        model_kwargs = dict(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=timm_global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Optional args retained for signature parity; only pass through if
        # not None to avoid compatibility issues on older/newer timm versions.
        if representation_size is not None:
            model_kwargs["representation_size"] = representation_size
        if init_values is not None:
            model_kwargs["init_values"] = init_values
        if norm_layer is not None:
            model_kwargs["norm_layer"] = norm_layer
        if act_layer is not None:
            model_kwargs["act_layer"] = act_layer
        if weight_init:
            model_kwargs["weight_init"] = weight_init

        # `embed_layer` and `block_fn` are intentionally not forwarded because
        # they are uncommon in external usage and can be version-fragile.

        vit = timm.create_model(variant, pretrained=pretrained, **model_kwargs)

        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm

        self.embed_dim = getattr(vit, "embed_dim", vit.num_features)
        self.out_dim = self.embed_dim

        # Cache whether blocks support attn_mask to avoid runtime introspection.
        self._blocks_support_attn_mask = False
        if len(self.blocks) > 0:
            try:
                sig = inspect.signature(self.blocks[0].forward)
                self._blocks_support_attn_mask = "attn_mask" in sig.parameters
            except (TypeError, ValueError):
                self._blocks_support_attn_mask = False

    def forward(
        self,
        x,
        img_prompt=None,
        up_pool=None,
        down_pool=None,
        align_bias=None,
        get_attn_mask=False,
        attn_mask=None,
        **kwargs,
    ):
        # Parity behavior from backup SOYO implementation.
        if get_attn_mask:
            assert img_prompt is None

        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if img_prompt is not None:
            # Accept [L, D] and [B, L, D], matching practical usage.
            if img_prompt.dim() == 2:
                img_prompt = img_prompt.unsqueeze(0).expand(x.shape[0], -1, -1)
            elif img_prompt.dim() != 3:
                raise ValueError(
                    "img_prompt must have shape [L, D] or [B, L, D], "
                    f"got {tuple(img_prompt.shape)}"
                )
            if img_prompt.shape[0] != x.shape[0]:
                raise ValueError(
                    "img_prompt batch mismatch: "
                    f"{img_prompt.shape[0]} vs image batch {x.shape[0]}"
                )

            # Equivalent to backup's broadcast trick.
            img_prompt = img_prompt.to(device=x.device, dtype=x.dtype)

        x = x + self.pos_embed[:, : x.shape[1], :].to(x.dtype)
        if img_prompt is not None:
            x = torch.cat([x[:, :1, :], img_prompt, x[:, 1:, :]], dim=1)

        x = self.pos_drop(x)

        for i in range(len(self.blocks)):
            if attn_mask is not None and self._blocks_support_attn_mask:
                x = self.blocks[i](x, attn_mask=attn_mask)
            else:
                x = self.blocks[i](x)

            if up_pool is not None:
                assert down_pool is not None
                assert isinstance(up_pool[0], nn.Linear)
                assert isinstance(down_pool[0], nn.Linear)
                x = x + up_pool[i](down_pool[i](x))

        # Backup behavior applies final norm on CLS directly.
        x = self.norm(x[:, 0])
        return x


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    """
    Backup-compatible model factory.

    Mirrors the backup helper's `features_only` guard and returns ViT_Prompts.
    """
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )
    kwargs.pop("features_only", None)

    return ViT_Prompts(variant=variant, pretrained=pretrained, **kwargs)


class ViTBackbone(nn.Module):
    """
    Drop-in backbone API used by the current codebase.

    Internally builds the prompt-capable model via
    `_create_vision_transformer(...)` for strict parity with backup SOYO
    construction style, while exposing the simplified project API.
    """

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        self.encoder = _create_vision_transformer(
            "vit_base_patch16_224",
            pretrained=pretrained,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
        )

        self.embed_dim = self.encoder.embed_dim
        self.out_dim = self.encoder.out_dim
        self.n_layers = len(self.encoder.blocks)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_with_prompts(
        self,
        x: torch.Tensor,
        prompts: torch.Tensor,
        up_pool: nn.ModuleList = None,
        down_pool: nn.ModuleList = None,
    ) -> torch.Tensor:
        return self.encoder(
            x,
            img_prompt=prompts,
            up_pool=up_pool,
            down_pool=down_pool,
        )
