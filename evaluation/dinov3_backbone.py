from pathlib import Path

import torch
from torch import nn


class DINOv3PatchEmbed(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.eval()
        self.backbone.requires_grad_(False)

    def forward(self, images):
        features = self.backbone.forward_features(images)
        patch_tokens = features["x_norm_patchtokens"]
        if patch_tokens.ndim != 3 or patch_tokens.shape[-1] != 1024:
            raise ValueError(
                f"Expected DINOv3 patch tokens shaped [B, P, 1024], got {tuple(patch_tokens.shape)}"
            )
        return patch_tokens


def load_dinov3_vitl16(repo_dir, weights):
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 repository not found: {repo_dir}")
    if not weights:
        raise ValueError("DINOv3 weights URL or path is required")

    return torch.hub.load(str(repo_dir), "dinov3_vitl16", source="local", weights=str(weights))
