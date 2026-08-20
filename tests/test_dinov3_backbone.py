import torch

from evaluation.dinov3_backbone import DINOv3PatchEmbed, load_dinov3_vitl16


class FakeDINOv3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward_features(self, images):
        batch = images.shape[0]
        return {"x_norm_patchtokens": torch.ones(batch, 6, 1024)}


def test_adapter_returns_final_normalized_patch_tokens_and_freezes_weights():
    backbone = FakeDINOv3()
    adapter = DINOv3PatchEmbed(backbone)
    output = adapter(torch.zeros(2, 3, 32, 48))

    assert output.shape == (2, 6, 1024)
    assert all(not parameter.requires_grad for parameter in adapter.parameters())


def test_loader_uses_official_local_torch_hub_api(monkeypatch, tmp_path):
    calls = {}
    expected = FakeDINOv3()

    def fake_load(repo_dir, model_name, source, weights):
        calls.update(repo_dir=repo_dir, model_name=model_name, source=source, weights=weights)
        return expected

    monkeypatch.setattr(torch.hub, "load", fake_load)
    loaded = load_dinov3_vitl16(tmp_path, "weights-url")

    assert loaded is expected
    assert calls == {
        "repo_dir": str(tmp_path),
        "model_name": "dinov3_vitl16",
        "source": "local",
        "weights": "weights-url",
    }
