import torch

from adapter.model import (
    FeatureAdapter,
    ResidualMLPAdapter,
    build_identity_adapter,
    build_residual_mlp_adapter,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)


def test_identity_adapter_preserves_features():
    adapter = build_identity_adapter()
    x = torch.randn(3, 7, 1024)
    assert torch.allclose(adapter(x), x)
    assert adapter.linear.weight.shape == (1024, 1024)
    assert adapter.linear.bias.shape == (1024,)


def test_residual_mlp_adapter_starts_as_identity():
    adapter = build_residual_mlp_adapter(dim=1024, hidden_dim=2048)
    x = torch.randn(3, 7, 1024)

    assert torch.allclose(adapter(x), x)
    assert adapter.hidden_dim == 2048
    assert adapter.fc1.weight.shape == (2048, 1024)
    assert adapter.fc2.weight.shape == (1024, 2048)
    assert torch.count_nonzero(adapter.fc2.weight) == 0
    assert torch.count_nonzero(adapter.fc2.bias) == 0


def test_linear_adapter_checkpoint_round_trip(tmp_path):
    adapter = build_identity_adapter()
    path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(path, adapter, {"epoch": 4, "val_loss": 0.25})

    loaded, metadata = load_adapter_checkpoint(path)

    assert isinstance(loaded, FeatureAdapter)
    assert torch.allclose(loaded.linear.weight, adapter.linear.weight)
    assert torch.allclose(loaded.linear.bias, adapter.linear.bias)
    assert metadata["epoch"] == 4
    assert metadata["val_loss"] == 0.25


def test_legacy_linear_checkpoint_without_adapter_type_still_loads(tmp_path):
    adapter = build_identity_adapter(dim=4)
    path = tmp_path / "legacy.pt"
    torch.save({
        "state_dict": adapter.state_dict(),
        "dim": 4,
        "metadata": {"epoch": 2},
    }, path)

    loaded, metadata = load_adapter_checkpoint(path)

    assert isinstance(loaded, FeatureAdapter)
    assert torch.allclose(loaded.linear.weight, adapter.linear.weight)
    assert metadata["epoch"] == 2


def test_mlp_adapter_checkpoint_round_trip(tmp_path):
    adapter = build_residual_mlp_adapter(dim=1024, hidden_dim=2048)
    with torch.no_grad():
        adapter.fc2.bias.fill_(0.125)
    path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(path, adapter, {"epoch": 3})

    loaded, metadata = load_adapter_checkpoint(path)

    assert isinstance(loaded, ResidualMLPAdapter)
    assert loaded.dim == 1024
    assert loaded.hidden_dim == 2048
    assert torch.allclose(loaded.fc2.bias, adapter.fc2.bias)
    assert metadata["epoch"] == 3
