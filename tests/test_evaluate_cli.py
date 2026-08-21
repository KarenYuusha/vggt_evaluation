import pytest

from evaluate import parse_args


def test_cli_defaults_to_dinov2():
    args = parse_args(["--dataset", "co3d", "--data-dir", "data"])
    assert args.backbone == "dinov2"
    assert args.adapter_checkpoint is None


def test_cli_dinov3_defaults_to_paper4_features():
    args = parse_args(["--dataset", "co3d", "--data-dir", "data", "--backbone", "dinov3"])
    assert args.backbone == "dinov3"
    assert args.dinov3_feature_mode == "paper4"
    assert not hasattr(args, "dinov3_repo")
    assert not hasattr(args, "dinov3_weights")


def test_cli_accepts_legacy_last_layer_dinov3():
    args = parse_args([
        "--dataset", "co3d", "--data-dir", "data", "--backbone", "dinov3",
        "--dinov3-feature-mode", "last",
    ])
    assert args.dinov3_feature_mode == "last"


def test_cli_accepts_adapter_checkpoint_only_with_last_layer_dinov3():
    args = parse_args([
        "--dataset", "realestate10k", "--data-dir", "data", "--backbone", "dinov3",
        "--dinov3-feature-mode", "last", "--adapter-checkpoint", "adapter.pt",
    ])
    assert str(args.adapter_checkpoint) == "adapter.pt"


def test_cli_rejects_adapter_checkpoint_with_paper4():
    with pytest.raises(SystemExit):
        parse_args([
            "--dataset", "realestate10k", "--data-dir", "data", "--backbone", "dinov3",
            "--adapter-checkpoint", "adapter.pt",
        ])


def test_cli_rejects_adapter_checkpoint_with_dinov2():
    with pytest.raises(SystemExit):
        parse_args([
            "--dataset", "realestate10k", "--data-dir", "data",
            "--adapter-checkpoint", "adapter.pt",
        ])
