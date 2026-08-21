import sys
from types import SimpleNamespace

import torch

from adapter.model import build_residual_mlp_adapter, load_adapter_checkpoint, save_adapter_checkpoint
from evaluation.dinov3_backbone import DINOv3PatchEmbed, load_dinov3_vitl16


class FakeHFDINOv3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(num_register_tokens=4)

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        special = torch.zeros(batch, 5, 1024)
        patches = torch.ones(batch, 6, 1024)
        return SimpleNamespace(last_hidden_state=torch.cat([special, patches], dim=1))


class FakePaperHFDINOv3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(num_register_tokens=4)
        self.norm = AddConstantNorm(100.0)
        self.output_hidden_states_calls = []

    def forward(self, pixel_values, output_hidden_states=False):
        self.output_hidden_states_calls.append(output_hidden_states)
        batch = pixel_values.shape[0]
        hidden_states = tuple(torch.full((batch, 11, 1024), float(i)) for i in range(25))
        return SimpleNamespace(last_hidden_state=hidden_states[-1], hidden_states=hidden_states)


class AddConstantNorm(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, features):
        return features + self.value


class ScaleAdapter(torch.nn.Module):
    def forward(self, features):
        return features * 2


def test_adapter_strips_cls_and_register_tokens_and_freezes_weights():
    wrapper = DINOv3PatchEmbed(FakeHFDINOv3())
    output = wrapper(torch.zeros(2, 3, 32, 48))

    assert output.shape == (2, 6, 1024)
    assert torch.all(output == 1)
    assert all(not parameter.requires_grad for parameter in wrapper.parameters())


def test_paper4_concatenates_normalized_layers_4_11_17_23():
    backbone = FakePaperHFDINOv3()
    wrapper = DINOv3PatchEmbed(backbone, feature_mode="paper4")
    output = wrapper(torch.zeros(2, 3, 32, 48))

    assert output.shape == (2, 6, 4096)
    assert backbone.output_hidden_states_calls == [True]
    expected_values = [105.0, 112.0, 118.0, 124.0]
    for index, expected in enumerate(expected_values):
        chunk = output[..., index * 1024:(index + 1) * 1024]
        assert torch.all(chunk == expected)


def test_wrapper_applies_and_freezes_feature_adapter():
    adapter = ScaleAdapter()
    wrapper = DINOv3PatchEmbed(FakeHFDINOv3(), adapter=adapter)
    output = wrapper(torch.zeros(2, 3, 32, 48))

    assert output.shape == (2, 6, 1024)
    assert torch.all(output == 2)
    assert wrapper.adapter is adapter
    assert all(not parameter.requires_grad for parameter in wrapper.parameters())


def test_wrapper_accepts_loaded_residual_mlp_checkpoint(tmp_path):
    adapter = build_residual_mlp_adapter(dim=1024, hidden_dim=2048)
    with torch.no_grad():
        adapter.fc2.bias.fill_(0.25)
    checkpoint = tmp_path / "mlp.pt"
    save_adapter_checkpoint(checkpoint, adapter)
    loaded, _ = load_adapter_checkpoint(checkpoint)

    wrapper = DINOv3PatchEmbed(FakeHFDINOv3(), adapter=loaded)
    output = wrapper(torch.zeros(1, 3, 32, 48))

    assert output.shape == (1, 6, 1024)
    assert torch.allclose(output, torch.full_like(output, 1.25))
    assert all(not parameter.requires_grad for parameter in wrapper.parameters())


def test_loader_uses_huggingface_auto_model(monkeypatch):
    calls = {}
    expected = FakeHFDINOv3()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id):
            calls["model_id"] = model_id
            return expected

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=FakeAutoModel))
    loaded = load_dinov3_vitl16()

    assert loaded is expected
    assert calls == {"model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m"}
