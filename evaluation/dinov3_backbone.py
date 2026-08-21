from torch import nn


DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_HIDDEN_SIZE = 1024
DINOV3_PATCH_SIZE = 16


class DINOv3PatchEmbed(nn.Module):
    def __init__(self, backbone, adapter=None):
        super().__init__()
        self.backbone = backbone.eval()
        self.backbone.requires_grad_(False)
        self.adapter = adapter
        if self.adapter is not None:
            self.adapter = self.adapter.eval()
            self.adapter.requires_grad_(False)

    def forward(self, images):
        outputs = self.backbone(pixel_values=images)
        num_register_tokens = int(self.backbone.config.num_register_tokens)
        patch_tokens = outputs.last_hidden_state[:, 1 + num_register_tokens:]
        expected_patches = (images.shape[-2] // DINOV3_PATCH_SIZE) * (images.shape[-1] // DINOV3_PATCH_SIZE)

        if patch_tokens.ndim != 3 or patch_tokens.shape[-1] != DINOV3_HIDDEN_SIZE:
            raise ValueError(
                f"Expected DINOv3 patch tokens shaped [B, P, {DINOV3_HIDDEN_SIZE}], "
                f"got {tuple(patch_tokens.shape)}"
            )
        if patch_tokens.shape[1] != expected_patches:
            raise ValueError(
                f"Expected {expected_patches} DINOv3 patch tokens for input {tuple(images.shape[-2:])}, "
                f"got {patch_tokens.shape[1]}"
            )
        if self.adapter is not None:
            patch_tokens = self.adapter(patch_tokens)
        if patch_tokens.shape[-1] != DINOV3_HIDDEN_SIZE:
            raise ValueError(f"DINOv3 adapter must preserve feature size {DINOV3_HIDDEN_SIZE}")
        return patch_tokens


def load_dinov3_vitl16():
    from transformers import AutoModel

    return AutoModel.from_pretrained(DINOV3_MODEL_ID)
