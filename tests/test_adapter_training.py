import torch

from adapter.model import build_identity_adapter, build_residual_mlp_adapter, load_adapter_checkpoint
from adapter.training import alignment_loss, fit_adapter, iter_patch_batches, train_epoch


def write_cache(path, dino3, dino2):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "clip": path.stem,
        "images": ["x.jpg"],
        "dino3": dino3.to(torch.float16),
        "dino2": dino2.to(torch.float16),
    }, path)


def test_alignment_loss_is_zero_for_identical_features():
    x = torch.randn(8, 4)
    loss = alignment_loss(x, x)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_patch_batches_flatten_clip_and_are_float32(tmp_path):
    dino3 = torch.randn(2, 3, 4)
    dino2 = torch.randn(2, 3, 4)
    cache = tmp_path / "clip.pt"
    write_cache(cache, dino3, dino2)

    batches = list(iter_patch_batches([cache], batch_size=4, seed=0, shuffle=False))
    assert [batch[0].shape[0] for batch in batches] == [4, 2]
    assert batches[0][0].dtype == torch.float32
    assert batches[0][1].dtype == torch.float32


def test_training_decreases_loss_on_small_linear_mapping(tmp_path):
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(4, 8, 4, generator=generator)
    transform = torch.tensor([
        [1.2, 0.1, 0.0, 0.0],
        [0.0, 0.8, 0.2, 0.0],
        [0.0, 0.0, 1.1, 0.1],
        [0.1, 0.0, 0.0, 0.9],
    ])
    y = x @ transform.T
    cache = tmp_path / "train" / "clip.pt"
    write_cache(cache, x, y)

    adapter = build_identity_adapter(dim=4)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.05, weight_decay=0.0)
    before = float(alignment_loss(adapter(x.float()), y.float()).detach())
    for epoch in range(12):
        train_epoch(adapter, [cache], optimizer, "cpu", batch_size=16, seed=epoch)
    after = float(alignment_loss(adapter(x.float()), y.float()).detach())

    assert after < before * 0.35
    assert all(parameter.grad is not None for parameter in adapter.parameters())


def test_residual_mlp_training_decreases_nonlinear_alignment_loss(tmp_path):
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(8, 8, 4, generator=generator)
    y = x + 0.2 * torch.tanh(x @ torch.tensor([
        [0.8, -0.2, 0.1, 0.0],
        [0.1, 0.7, -0.2, 0.1],
        [0.0, 0.2, 0.9, -0.1],
        [-0.1, 0.0, 0.2, 0.8],
    ]).T)
    cache = tmp_path / "train" / "mlp.pt"
    write_cache(cache, x, y)

    adapter = build_residual_mlp_adapter(dim=4, hidden_dim=8)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.03, weight_decay=0.0)
    before = float(alignment_loss(adapter(x.float()), y.float()).detach())
    for epoch in range(20):
        train_epoch(adapter, [cache], optimizer, "cpu", batch_size=32, seed=epoch)
    after = float(alignment_loss(adapter(x.float()), y.float()).detach())

    assert after < before * 0.25
    assert torch.count_nonzero(adapter.fc2.weight) > 0


def test_fit_adapter_saves_best_validation_checkpoint(tmp_path):
    x = torch.randn(2, 4, 4)
    y = x * 1.1
    train_cache = tmp_path / "cache" / "train" / "a.pt"
    val_cache = tmp_path / "cache" / "val" / "b.pt"
    write_cache(train_cache, x, y)
    write_cache(val_cache, x, y)

    adapter = build_identity_adapter(dim=4)
    output_dir = tmp_path / "out"
    history = fit_adapter(
        adapter, [train_cache], [val_cache], output_dir, device="cpu", epochs=3,
        batch_size=8, lr=0.02, weight_decay=0.0, seed=0, metadata={"source": "test"},
    )

    best_path = output_dir / "best_adapter.pt"
    last_path = output_dir / "last_adapter.pt"
    assert best_path.exists()
    assert last_path.exists()
    _, metadata = load_adapter_checkpoint(best_path)
    assert metadata["source"] == "test"
    assert metadata["val_loss"] == min(item["val_loss"] for item in history)
