import torch
from torch import nn


DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_HIDDEN_SIZE = 1024
DINOV3_PATCH_SIZE = 16
DINOV3_PAPER_LAYERS = (4, 11, 17, 23)
DINOV3_PAPER_FEATURE_SIZE = DINOV3_HIDDEN_SIZE * len(DINOV3_PAPER_LAYERS)


class DINOv3PatchEmbed(nn.Module):
    def __init__(self, backbone, adapter=None, feature_mode="last"):
        super().__init__()
        if feature_mode not in {"last", "paper4"}:
            raise ValueError(f"Unsupported DINOv3 feature mode: {feature_mode}")
        if feature_mode == "paper4" and adapter is not None:
            raise ValueError("The existing 1024->1024 adapters are not compatible with paper4 4096-D features")

        self.backbone = backbone.eval()
        self.backbone.requires_grad_(False)
        self.adapter = adapter
        self.feature_mode = feature_mode
        if self.adapter is not None:
            self.adapter = self.adapter.eval()
            self.adapter.requires_grad_(False)

    def _last_layer_tokens(self, images, num_register_tokens):
        outputs = self.backbone(pixel_values=images)
        return outputs.last_hidden_state[:, 1 + num_register_tokens:]

    def _paper_four_layer_tokens(self, images, num_register_tokens):
        outputs = self.backbone(pixel_values=images, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= max(DINOV3_PAPER_LAYERS) + 1:
            raise ValueError("DINOv3 did not return all intermediate hidden states required by paper4 mode")

        patch_layers = []
        for block_index in DINOV3_PAPER_LAYERS:
            normalized = self.backbone.norm(hidden_states[block_index + 1])
            patch_layers.append(normalized[:, 1 + num_register_tokens:])
        return torch.cat(patch_layers, dim=-1)

    def forward(self, images):
        num_register_tokens = int(self.backbone.config.num_register_tokens)
        if self.feature_mode == "paper4":
            patch_tokens = self._paper_four_layer_tokens(images, num_register_tokens)
            expected_dim = DINOV3_PAPER_FEATURE_SIZE
        else:
            patch_tokens = self._last_layer_tokens(images, num_register_tokens)
            expected_dim = DINOV3_HIDDEN_SIZE

        expected_patches = (images.shape[-2] // DINOV3_PATCH_SIZE) * (images.shape[-1] // DINOV3_PATCH_SIZE)
        if patch_tokens.ndim != 3 or patch_tokens.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected DINOv3 patch tokens shaped [B, P, {expected_dim}], got {tuple(patch_tokens.shape)}"
            )
        if patch_tokens.shape[1] != expected_patches:
            raise ValueError(
                f"Expected {expected_patches} DINOv3 patch tokens for input {tuple(images.shape[-2:])}, "
                f"got {patch_tokens.shape[1]}"
            )
        if self.adapter is not None:
            patch_tokens = self.adapter(patch_tokens)
        if self.feature_mode == "last" and patch_tokens.shape[-1] != DINOV3_HIDDEN_SIZE:
            raise ValueError(f"DINOv3 adapter must preserve feature size {DINOV3_HIDDEN_SIZE}")
        return patch_tokens


def load_dinov3_vitl16():
    from transformers import AutoModel

    return AutoModel.from_pretrained(DINOV3_MODEL_ID)
