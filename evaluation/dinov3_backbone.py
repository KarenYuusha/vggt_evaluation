import torch
from torch import nn


DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_HIDDEN_SIZE = 1024
DINOV3_PATCH_SIZE = 16
DINOV3_MULTILAYER_INDICES = (4, 11, 17, 23)
DINOV3_MULTILAYER_DIM = DINOV3_HIDDEN_SIZE * len(DINOV3_MULTILAYER_INDICES)
DINOV3_FEATURE_MODES = ("final", "multilayer")


def _strip_special_tokens(backbone, state):
    num_register_tokens = int(backbone.config.num_register_tokens)
    return state[:, 1 + num_register_tokens :]


def extract_dinov3_patch_tokens(backbone, images, feature_mode="final"):
    if feature_mode not in DINOV3_FEATURE_MODES:
        raise ValueError(f"Unsupported DINOv3 feature mode: {feature_mode}")

    if feature_mode == "final":
        outputs = backbone(pixel_values=images)
        patch_tokens = _strip_special_tokens(backbone, outputs.last_hidden_state)
        expected_dim = DINOV3_HIDDEN_SIZE
    else:
        outputs = backbone(pixel_values=images, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("DINOv3 multi-layer extraction requires hidden states")
        selected = []
        for block_idx in DINOV3_MULTILAYER_INDICES:
            state_idx = block_idx + 1  # HF hidden_states[0] is the embedding output.
            if state_idx >= len(hidden_states):
                raise ValueError(
                    f"DINOv3 hidden states have length {len(hidden_states)}; cannot read block {block_idx}"
                )
            state = hidden_states[state_idx]
            # Official DINOv3 get_intermediate_layers(..., norm=True) normalizes
            # every selected block output before returning patch tokens.
            norm = getattr(backbone, "norm", None)
            if norm is not None:
                state = norm(state)
            patches = _strip_special_tokens(backbone, state)
            if patches.ndim != 3 or patches.shape[-1] != DINOV3_HIDDEN_SIZE:
                raise ValueError(
                    f"Expected DINOv3 block {block_idx} patch tokens shaped [B, P, {DINOV3_HIDDEN_SIZE}], "
                    f"got {tuple(patches.shape)}"
                )
            selected.append(patches)
        patch_counts = {tokens.shape[1] for tokens in selected}
        if len(patch_counts) != 1:
            raise ValueError("Selected DINOv3 intermediate layers have mismatched patch counts")
        patch_tokens = torch.cat(selected, dim=-1)
        expected_dim = DINOV3_MULTILAYER_DIM

    if patch_tokens.ndim != 3 or patch_tokens.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected DINOv3 {feature_mode} patch tokens shaped [B, P, {expected_dim}], "
            f"got {tuple(patch_tokens.shape)}"
        )
    return patch_tokens


class DINOv3PatchEmbed(nn.Module):
    def __init__(self, backbone, adapter=None, feature_mode="final"):
        super().__init__()
        if feature_mode not in DINOV3_FEATURE_MODES:
            raise ValueError(f"Unsupported DINOv3 feature mode: {feature_mode}")
        self.backbone = backbone.eval()
        self.backbone.requires_grad_(False)
        self.feature_mode = feature_mode
        self.adapter = adapter
        if self.adapter is not None:
            self.adapter = self.adapter.eval()
            self.adapter.requires_grad_(False)

    def forward(self, images):
        patch_tokens = extract_dinov3_patch_tokens(self.backbone, images, feature_mode=self.feature_mode)
        expected_patches = (images.shape[-2] // DINOV3_PATCH_SIZE) * (images.shape[-1] // DINOV3_PATCH_SIZE)
        if patch_tokens.shape[1] != expected_patches:
            raise ValueError(
                f"Expected {expected_patches} DINOv3 patch tokens for input {tuple(images.shape[-2:])}, "
                f"got {patch_tokens.shape[1]}"
            )
        if self.adapter is not None:
            patch_tokens = self.adapter(patch_tokens)
        if patch_tokens.shape[-1] != DINOV3_HIDDEN_SIZE:
            raise ValueError(
                f"DINOv3 features entering VGGT must have size {DINOV3_HIDDEN_SIZE}; got {patch_tokens.shape[-1]}"
            )
        return patch_tokens


def load_dinov3_vitl16():
    from transformers import AutoModel

    return AutoModel.from_pretrained(DINOV3_MODEL_ID)
