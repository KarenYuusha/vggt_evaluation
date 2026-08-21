from train_adapter import parse_args


def test_train_cli_defaults_to_linear_adapter():
    args = parse_args([])
    assert args.adapter_type == "linear"
    assert args.hidden_dim == 2048


def test_train_cli_accepts_residual_mlp_adapter():
    args = parse_args(["--adapter-type", "mlp", "--hidden-dim", "3072"])
    assert args.adapter_type == "mlp"
    assert args.hidden_dim == 3072
