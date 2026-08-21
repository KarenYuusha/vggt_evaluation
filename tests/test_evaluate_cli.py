from evaluate import parse_args


def test_cli_defaults_to_dinov2():
    args = parse_args(["--dataset", "co3d", "--data-dir", "data"])
    assert args.backbone == "dinov2"


def test_cli_accepts_dinov3_without_repo_or_weight_arguments():
    args = parse_args(["--dataset", "co3d", "--data-dir", "data", "--backbone", "dinov3"])
    assert args.backbone == "dinov3"
    assert not hasattr(args, "dinov3_repo")
    assert not hasattr(args, "dinov3_weights")
