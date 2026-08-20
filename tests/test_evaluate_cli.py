import pytest

from evaluate import parse_args


def test_cli_defaults_to_dinov2_without_dinov3_arguments():
    args = parse_args(["--dataset", "co3d", "--data-dir", "data"])
    assert args.backbone == "dinov2"
    assert args.dinov3_repo is None
    assert args.dinov3_weights is None


def test_cli_requires_dinov3_repo_and_weights_for_dinov3():
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "co3d", "--data-dir", "data", "--backbone", "dinov3"])


def test_cli_accepts_complete_dinov3_configuration():
    args = parse_args([
        "--dataset", "realestate10k", "--data-dir", "data", "--backbone", "dinov3",
        "--dinov3-repo", "dinov3", "--dinov3-weights", "weights-url",
    ])
    assert args.backbone == "dinov3"
    assert str(args.dinov3_repo) == "dinov3"
    assert args.dinov3_weights == "weights-url"
