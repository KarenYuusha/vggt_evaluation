from types import SimpleNamespace

import torch

from adapter.features import (
    extract_dinov2_patch_tokens,
    extract_dinov3_patch_tokens,
    extract_feature_pair,
    normalize_imagenet,
)


class FakeDINOv2Patch(torch.nn.Module):
    def forward(self, images):
        batch = images.shape[0]
        return {"x_norm_patchtokens": torch.ones(batch, 6, 1024, device=images.device)}


class FakeVGGT:
    def __init__(self):
        self.aggregator = SimpleNamespace(patch_embed=FakeDINOv2Patch())


class FakeHFBackbone(torch.nn.Module):
    def __init__(self, num_register_tokens=4, patch_count=6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(num_register_tokens=num_register_tokens)
        self.patch_count = patch_count

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        total = 1 + self.config.num_register_tokens + self.patch_count
        tokens = torch.arange(total, dtype=pixel_values.dtype, device=pixel_values.device)
        tokens = tokens.view(1, total, 1).expand(batch, total, 1024)
        return SimpleNamespace(last_hidden_state=tokens)


def test_imagenet_normalization_matches_expected_constants():
    x = torch.zeros(1, 3, 2, 2)
    normalized = normalize_imagenet(x)
    expected = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    assert torch.allclose(normalized[0, :, 0, 0], expected)


def test_dinov2_extractor_returns_final_patch_tokens_without_gradients():
    patches = extract_dinov2_patch_tokens(FakeVGGT(), torch.zeros(2, 3, 32, 48))
    assert patches.shape == (2, 6, 1024)
    assert not patches.requires_grad


def test_dinov3_extractor_strips_cls_and_register_tokens():
    model = FakeHFBackbone(num_register_tokens=4, patch_count=6)
    patches = extract_dinov3_patch_tokens(model, torch.zeros(2, 3, 32, 48))
    assert patches.shape == (2, 6, 1024)
    assert torch.all(patches[:, 0] == 5)
    assert not patches.requires_grad


def test_feature_pair_rejects_mismatched_patch_counts():
    dino2 = FakeVGGT()
    dino3 = FakeHFBackbone(patch_count=7)
    try:
        extract_feature_pair(dino2, dino3, torch.zeros(2, 3, 32, 48), torch.zeros(2, 3, 32, 48))
    except ValueError as exc:
        assert "matching" in str(exc)
    else:
        raise AssertionError("Expected mismatched patch features to fail")
