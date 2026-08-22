from dataclasses import dataclass
from functools import partial
from pathlib import Path
import time

import numpy as np
import torch

from adapter.model import load_adapter_checkpoint
from .dinov3_backbone import (
    DINOV3_HIDDEN_SIZE,
    DINOV3_MULTILAYER_DIM,
    DINOv3PatchEmbed,
    load_dinov3_vitl16,
)


@dataclass
class Prediction:
    extrinsics: np.ndarray
    preprocess_ms: float
    inference_ms: float


def _dinov3_feature_mode(backbone):
    if backbone in ("dinov3", "dinov3-final"):
        return "final"
    if backbone == "dinov3-multilayer":
        return "multilayer"
    raise ValueError(f"Unsupported backbone: {backbone}")


def configure_backbone(model, backbone, preprocess_fn, adapter_checkpoint=None,
                       dinov3_loader=load_dinov3_vitl16, adapter_loader=load_adapter_checkpoint):
    if backbone == "dinov2":
        if adapter_checkpoint is not None:
            raise ValueError("Adapter checkpoints are only valid with the DINOv3 backbone")
        return preprocess_fn, 518, 14

    feature_mode = _dinov3_feature_mode(backbone)
    if feature_mode == "multilayer" and adapter_checkpoint is None:
        raise ValueError("dinov3-multilayer requires a trained 4096 -> 1024 adapter checkpoint")

    dinov3 = dinov3_loader()
    adapter = None
    if adapter_checkpoint is not None:
        adapter, _ = adapter_loader(adapter_checkpoint)
        expected_input_dim = DINOV3_MULTILAYER_DIM if feature_mode == "multilayer" else DINOV3_HIDDEN_SIZE
        input_dim = getattr(adapter, "input_dim", getattr(adapter, "dim", None))
        output_dim = getattr(adapter, "output_dim", getattr(adapter, "dim", None))
        if input_dim != expected_input_dim or output_dim != DINOV3_HIDDEN_SIZE:
            raise ValueError(
                f"DINOv3 {feature_mode} adapter must be {expected_input_dim} -> {DINOV3_HIDDEN_SIZE}; "
                f"got {input_dim} -> {output_dim}"
            )
        adapter.eval()
        adapter.requires_grad_(False)

    model.aggregator.patch_embed = DINOv3PatchEmbed(
        dinov3, adapter=adapter, feature_mode=feature_mode
    )
    model.aggregator.patch_size = 16
    preprocess = partial(preprocess_fn, target_size=592, patch_size=16)
    return preprocess, 592, 16


class VGGTModelRunner:
    def __init__(self, model, device, dtype, preprocess_fn, pose_decode_fn, backbone="dinov2",
                 input_target_size=518, patch_size=14):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model.eval().to(self.device)
        if hasattr(self.model, "requires_grad_"):
            self.model.requires_grad_(False)
        self.preprocess_fn = preprocess_fn
        self.pose_decode_fn = pose_decode_fn
        self.backbone = backbone
        self.input_target_size = input_target_size
        self.patch_size = patch_size

    @classmethod
    def from_pretrained(cls, model_source="facebook/VGGT-1B", device=None, backbone="dinov2",
                        adapter_checkpoint=None):
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        source_path = Path(model_source)
        if source_path.exists():
            model = VGGT()
            state = torch.load(source_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
        else:
            model = VGGT.from_pretrained(model_source)

        preprocess_fn, input_target_size, patch_size = configure_backbone(
            model, backbone, load_and_preprocess_images, adapter_checkpoint=adapter_checkpoint
        )

        if device.type == "cuda":
            major = torch.cuda.get_device_capability(device)[0]
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            dtype = torch.float32

        return cls(
            model, device, dtype, preprocess_fn, pose_encoding_to_extri_intri, backbone=backbone,
            input_target_size=input_target_size, patch_size=patch_size,
        )

    def _camera_forward(self, images):
        images = images.unsqueeze(0)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    aggregated_tokens, _ = self.model.aggregator(images)
                with torch.autocast(device_type="cuda", enabled=False):
                    pose_enc = self.model.camera_head(aggregated_tokens)[-1]
                    extrinsic, _ = self.pose_decode_fn(pose_enc, images.shape[-2:])
            else:
                aggregated_tokens, _ = self.model.aggregator(images)
                pose_enc = self.model.camera_head(aggregated_tokens)[-1]
                extrinsic, _ = self.pose_decode_fn(pose_enc, images.shape[-2:])
        return extrinsic

    def predict(self, image_paths):
        preprocess_start = time.perf_counter()
        images = self.preprocess_fn(image_paths).to(self.device)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            extrinsic = self._camera_forward(images)
            end.record()
            torch.cuda.synchronize(self.device)
            inference_ms = float(start.elapsed_time(end))
        else:
            inference_start = time.perf_counter()
            extrinsic = self._camera_forward(images)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

        extrinsics = extrinsic[0].detach().float().cpu().numpy()
        return Prediction(extrinsics=extrinsics, preprocess_ms=preprocess_ms, inference_ms=inference_ms)

    def warmup(self, image_paths):
        self.predict(image_paths)
