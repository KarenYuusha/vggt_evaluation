import torch

from .data import validate_feature_pair


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_imagenet(images):
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected [N, 3, H, W] images, got {tuple(images.shape)}")
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def extract_dinov2_patch_tokens(vggt_model, images):
    normalized = normalize_imagenet(images)
    with torch.inference_mode():
        output = vggt_model.aggregator.patch_embed(normalized)
    if isinstance(output, dict):
        output = output["x_norm_patchtokens"]
    if output.ndim != 3 or output.shape[-1] != 1024:
        raise ValueError(f"Expected DINOv2 patch tokens shaped [N, P, 1024], got {tuple(output.shape)}")
    return output


def extract_dinov3_patch_tokens(dinov3_model, images):
    normalized = normalize_imagenet(images)
    with torch.inference_mode():
        output = dinov3_model(pixel_values=normalized)
    num_register_tokens = int(dinov3_model.config.num_register_tokens)
    patches = output.last_hidden_state[:, 1 + num_register_tokens:]
    if patches.ndim != 3 or patches.shape[-1] != 1024:
        raise ValueError(f"Expected DINOv3 patch tokens shaped [N, P, 1024], got {tuple(patches.shape)}")
    return patches


def extract_feature_pair(vggt_model, dinov3_model, dino2_images, dino3_images):
    dino2 = extract_dinov2_patch_tokens(vggt_model, dino2_images)
    dino3 = extract_dinov3_patch_tokens(dinov3_model, dino3_images)
    validate_feature_pair(dino2, dino3)
    return dino2, dino3
