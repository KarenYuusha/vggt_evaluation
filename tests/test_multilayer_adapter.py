import torch

from adapter.data import validate_feature_pair
from adapter.model import (
    FeatureAdapter,
    build_identity_adapter,
    build_linear_adapter,
    build_residual_mlp_adapter,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from adapter.training import iter_patch_batches, train_epoch


def write_cache(path, dino3, dino2):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dino3": dino3.half(), "dino2": dino2.half()}, path)


def test_feature_pair_allows_matching_patch_axes_with_4096_to_1024_dims():
    validate_feature_pair(
        torch.zeros(20, 100, 1024),
        torch.zeros(20, 100, 4096),
        expected_dino2_dim=1024,
        expected_dino3_dim=4096,
    )


def test_linear_adapter_supports_4096_to_1024():
    adapter = build_linear_adapter(input_dim=4096, output_dim=1024)
    out = adapter(torch.randn(2, 3, 4096))
    assert out.shape == (2, 3, 1024)
    assert adapter.input_dim == 4096
    assert adapter.output_dim == 1024


def test_mlp_adapter_supports_4096_to_1024_without_invalid_residual():
    adapter = build_residual_mlp_adapter(input_dim=4096, output_dim=1024, hidden_dim=64)
    out = adapter(torch.randn(2, 3, 4096))
    assert out.shape == (2, 3, 1024)
    assert not adapter.use_residual


def test_equal_dim_mlp_still_initializes_as_identity():
    adapter = build_residual_mlp_adapter(dim=4, hidden_dim=8)
    x = torch.randn(2, 3, 4)
    assert adapter.use_residual
    assert torch.allclose(adapter(x), x)


def test_new_dimension_metadata_round_trips(tmp_path):
    adapter = FeatureAdapter(input_dim=8, output_dim=4)
    path = tmp_path / "new.pt"
    save_adapter_checkpoint(path, adapter, {"kind": "multi"})
    loaded, metadata = load_adapter_checkpoint(path)
    assert loaded.input_dim == 8
    assert loaded.output_dim == 4
    assert loaded.linear.weight.shape == (4, 8)
    assert metadata["kind"] == "multi"


def test_legacy_dim_checkpoint_still_loads(tmp_path):
    adapter = build_identity_adapter(dim=4)
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": adapter.state_dict(), "dim": 4, "metadata": {}}, path)
    loaded, _ = load_adapter_checkpoint(path)
    assert loaded.input_dim == 4
    assert loaded.output_dim == 4
    assert torch.allclose(loaded.linear.weight, torch.eye(4))


def test_patch_batches_preserve_different_source_target_dims(tmp_path):
    path = tmp_path / "a.pt"
    write_cache(path, torch.randn(2, 3, 8), torch.randn(2, 3, 4))
    batches = list(iter_patch_batches([path], batch_size=4, shuffle=False))
    assert batches[0][0].shape == (4, 8)
    assert batches[0][1].shape == (4, 4)


def test_asymmetric_projection_can_train_against_smaller_target(tmp_path):
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(4, 6, 8, generator=generator)
    mapping = torch.randn(4, 8, generator=generator)
    y = x @ mapping.T
    path = tmp_path / "train.pt"
    write_cache(path, x, y)

    adapter = build_linear_adapter(input_dim=8, output_dim=4, identity_if_possible=False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.05, weight_decay=0)
    for epoch in range(8):
        train_epoch(adapter, [path], optimizer, "cpu", batch_size=12, seed=epoch)

    assert adapter(torch.randn(2, 8)).shape == (2, 4)
