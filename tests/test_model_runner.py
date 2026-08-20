import numpy as np
import torch

from evaluation.model_runner import VGGTModelRunner, configure_backbone


class FakeAggregator:
    def __init__(self):
        self.patch_embed = object()
        self.patch_size = 14

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


class FakeDINOv3(torch.nn.Module):
    def forward_features(self, images):
        return {"x_norm_patchtokens": torch.zeros(images.shape[0], 4, 1024)}


def fake_preprocess(paths, **kwargs):
    fake_preprocess.kwargs = kwargs
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


def test_configure_backbone_keeps_dinov2_defaults():
    model = FakeModel()
    original_patch_embed = model.aggregator.patch_embed
    preprocess, target_size, patch_size = configure_backbone(model, "dinov2", fake_preprocess)

    preprocess(["x.jpg"])
    assert model.aggregator.patch_embed is original_patch_embed
    assert model.aggregator.patch_size == 14
    assert fake_preprocess.kwargs == {}
    assert (target_size, patch_size) == (518, 14)


def test_configure_backbone_replaces_only_patch_encoder_for_dinov3():
    model = FakeModel()
    camera_head = model.camera_head
    preprocess, target_size, patch_size = configure_backbone(
        model, "dinov3", fake_preprocess, dinov3_repo="repo", dinov3_weights="weights",
        dinov3_loader=lambda repo, weights: FakeDINOv3(),
    )

    preprocess(["x.jpg"])
    assert model.camera_head is camera_head
    assert model.aggregator.patch_size == 16
    assert fake_preprocess.kwargs == {"target_size": 592, "patch_size": 16}
    assert (target_size, patch_size) == (592, 16)
