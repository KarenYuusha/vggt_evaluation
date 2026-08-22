import pytest

from evaluate import parse_args


def base(mode):
    return ["--dataset", "co3d", "--data-dir", "data", "--backbone", mode]


def test_cli_accepts_explicit_final_and_multilayer_modes():
    assert parse_args(base("dinov3-final")).backbone == "dinov3-final"
    args = parse_args(
        base("dinov3-multilayer") + ["--adapter-checkpoint", "adapter.pt"]
    )
    assert args.backbone == "dinov3-multilayer"


def test_multilayer_cli_requires_adapter_checkpoint():
    with pytest.raises(SystemExit):
        parse_args(base("dinov3-multilayer"))


def test_legacy_dinov3_alias_still_works():
    assert parse_args(base("dinov3")).backbone == "dinov3"
