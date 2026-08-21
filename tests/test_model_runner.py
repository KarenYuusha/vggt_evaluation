from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adapter.model import build_identity_adapter
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
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_register_tokens=4)

    def forward(self, pixel_values):
        return SimpleNamespace(last_hidden_state=torch.zeros(pixel_values.shape[0], 9, 1024))


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


def test_configure_backbone_loads_huggingface_dinov3_without_repo_or_weights():
    model = FakeModel()
    camera_head = model.camera_head
    calls = []
    preprocess, target_size, patch_size = configure_backbone(
        model, "dinov3", fake_preprocess,
        dinov3_loader=lambda: calls.append(True) or FakeDINOv3(),
    )

    preprocess(["x.jpg"])
    assert calls == [True]
    assert model.camera_head is camera_head
    assert model.aggregator.patch_size == 16
    assert fake_preprocess.kwargs == {"target_size": 592, "patch_size": 16}
    assert (target_size, patch_size) == (592, 16)


def test_configure_backbone_loads_and_freezes_adapter_for_dinov3():
    model = FakeModel()
    adapter = build_identity_adapter()
    calls = []
    configure_backbone(
        model, "dinov3", fake_preprocess, adapter_checkpoint="adapter.pt",
        dinov3_loader=lambda: FakeDINOv3(),
        adapter_loader=lambda path: calls.append(path) or (adapter, {"epoch": 1}),
    )

    assert calls == ["adapter.pt"]
    assert model.aggregator.patch_embed.adapter is adapter
    assert all(not parameter.requires_grad for parameter in adapter.parameters())


def test_configure_backbone_rejects_adapter_with_dinov2():
    with pytest.raises(ValueError, match="DINOv3"):
        configure_backbone(FakeModel(), "dinov2", fake_preprocess, adapter_checkpoint="adapter.pt")
