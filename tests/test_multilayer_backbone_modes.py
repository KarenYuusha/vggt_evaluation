from types import SimpleNamespace

import pytest
import torch

from adapter.model import build_identity_adapter, build_linear_adapter
from evaluation.model_runner import configure_backbone


class FakeAggregator:
    def __init__(self):
        self.patch_embed = object()
        self.patch_size = 14


class FakeModel:
    def __init__(self):
        self.aggregator = FakeAggregator()


class FakeDINOv3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_register_tokens=4)
        self.norm = torch.nn.Identity()

    def forward(self, pixel_values, output_hidden_states=False):
        total = 1 + 4 + ((pixel_values.shape[-2] // 16) * (pixel_values.shape[-1] // 16))
        states = tuple(torch.zeros(pixel_values.shape[0], total, 1024) for _ in range(25))
        return SimpleNamespace(
            last_hidden_state=states[-1],
            hidden_states=states if output_hidden_states else None,
        )


def preprocess(paths, **kwargs):
    target = kwargs.get("target_size", 518)
    return torch.zeros(len(paths), 3, target, target)


def test_dinov3_alias_and_explicit_final_both_use_final_features():
    for mode in ("dinov3", "dinov3-final"):
        model = FakeModel()
        configure_backbone(model, mode, preprocess, dinov3_loader=FakeDINOv3)
        assert model.aggregator.patch_embed.feature_mode == "final"
        assert model.aggregator.patch_size == 16


def test_multilayer_mode_requires_trained_projection():
    with pytest.raises(ValueError, match="requires.*adapter"):
        configure_backbone(
            FakeModel(), "dinov3-multilayer", preprocess, dinov3_loader=FakeDINOv3
        )


def test_multilayer_mode_accepts_only_4096_to_1024_projection():
    good = build_linear_adapter(input_dim=4096, output_dim=1024, identity_if_possible=False)
    model = FakeModel()
    configure_backbone(
        model,
        "dinov3-multilayer",
        preprocess,
        adapter_checkpoint="good.pt",
        dinov3_loader=FakeDINOv3,
        adapter_loader=lambda _: (good, {}),
    )
    assert model.aggregator.patch_embed.feature_mode == "multilayer"
    assert model.aggregator.patch_embed.adapter is good
    assert all(not parameter.requires_grad for parameter in good.parameters())

    bad = build_identity_adapter(dim=1024)
    with pytest.raises(ValueError, match="4096.*1024"):
        configure_backbone(
            FakeModel(),
            "dinov3-multilayer",
            preprocess,
            adapter_checkpoint="bad.pt",
            dinov3_loader=FakeDINOv3,
            adapter_loader=lambda _: (bad, {}),
        )
