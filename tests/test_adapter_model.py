import torch

from adapter.model import FeatureAdapter, build_identity_adapter, load_adapter_checkpoint, save_adapter_checkpoint


def test_identity_adapter_preserves_features():
    adapter = build_identity_adapter()
    x = torch.randn(3, 7, 1024)
    assert torch.allclose(adapter(x), x)
    assert adapter.linear.weight.shape == (1024, 1024)
    assert adapter.linear.bias.shape == (1024,)


def test_adapter_checkpoint_round_trip(tmp_path):
    adapter = build_identity_adapter()
    path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(path, adapter, {"epoch": 4, "val_loss": 0.25})

    loaded, metadata = load_adapter_checkpoint(path)

    assert isinstance(loaded, FeatureAdapter)
    assert torch.allclose(loaded.linear.weight, adapter.linear.weight)
    assert torch.allclose(loaded.linear.bias, adapter.linear.bias)
    assert metadata["epoch"] == 4
    assert metadata["val_loss"] == 0.25
