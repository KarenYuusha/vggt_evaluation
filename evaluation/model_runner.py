from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch


@dataclass
class Prediction:
    extrinsics: np.ndarray
    preprocess_ms: float
    inference_ms: float


class VGGTModelRunner:
    def __init__(self, model, device, dtype, preprocess_fn, pose_decode_fn):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model.eval().to(self.device)
        self.preprocess_fn = preprocess_fn
        self.pose_decode_fn = pose_decode_fn

    @classmethod
    def from_pretrained(cls, model_source="facebook/VGGT-1B", device=None):
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

        if device.type == "cuda":
            major = torch.cuda.get_device_capability(device)[0]
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            dtype = torch.float32

        return cls(model, device, dtype, load_and_preprocess_images, pose_encoding_to_extri_intri)

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
