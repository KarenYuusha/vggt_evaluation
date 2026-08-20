import numpy as np
import torch

from evaluation.model_runner import VGGTModelRunner


class FakeAggregator:
    def __call__(self, images):
        return [images], 0


class FakeCameraHead:
    def __call__(self, aggregated):
        images = aggregated[-1]
        batch, frames = images.shape[:2]
        return [torch.zeros(batch, frames, 9)]


class FakeModel:
    def __init__(self):
        self.aggregator = FakeAggregator()
        self.camera_head = FakeCameraHead()

    def eval(self):
        return self

    def to(self, device):
        return self


def fake_preprocess(paths):
    return torch.zeros(len(paths), 3, 14, 14)


def fake_decode(pose_enc, image_size):
    batch, frames = pose_enc.shape[:2]
    extri = torch.zeros(batch, frames, 3, 4)
    extri[..., 0, 0] = 1
    extri[..., 1, 1] = 1
    extri[..., 2, 2] = 1
    intri = torch.eye(3).expand(batch, frames, 3, 3)
    return extri, intri


def test_runner_executes_camera_only_path_and_returns_timings():
    runner = VGGTModelRunner(
        model=FakeModel(), device="cpu", dtype=torch.float32,
        preprocess_fn=fake_preprocess, pose_decode_fn=fake_decode,
    )

    prediction = runner.predict(["a.jpg", "b.jpg"])

    assert prediction.extrinsics.shape == (2, 3, 4)
    assert np.allclose(prediction.extrinsics[:, :3, :3], np.eye(3)[None])
    assert prediction.preprocess_ms >= 0
    assert prediction.inference_ms >= 0
