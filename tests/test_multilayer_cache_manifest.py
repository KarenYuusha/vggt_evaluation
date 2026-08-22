import torch

from extract_adapter_features import build_manifest, parse_args, save_clip_cache
from train_adapter import build_adapter_from_manifest


def test_feature_extraction_defaults_to_multilayer_mode():
    args = parse_args([])
    assert args.feature_mode == "multilayer"


def test_multilayer_manifest_records_source_target_dimensions_and_indices():
    manifest = build_manifest(["a"], ["b"], feature_mode="multilayer")
    assert manifest["feature_mode"] == "multilayer"
    assert manifest["dino3_layer_indices"] == [4, 11, 17, 23]
    assert manifest["dino3_feature_dim"] == 4096
    assert manifest["dino2_feature_dim"] == 1024
    assert manifest["adapter_input_dim"] == 4096
    assert manifest["adapter_output_dim"] == 1024


def test_cache_accepts_multilayer_source_and_dinov2_target(tmp_path):
    path = tmp_path / "clip.pt"
    save_clip_cache(
        path,
        "clip",
        ["a.jpg"],
        torch.zeros(1, 6, 1024),
        torch.zeros(1, 6, 4096),
    )
    record = torch.load(path)
    assert record["dino3"].shape[-1] == 4096
    assert record["dino2"].shape[-1] == 1024


def test_multilayer_manifest_builds_4096_to_1024_linear_adapter():
    manifest = {"adapter_input_dim": 4096, "adapter_output_dim": 1024}
    adapter = build_adapter_from_manifest(manifest, "linear", hidden_dim=2048)
    assert adapter.input_dim == 4096
    assert adapter.output_dim == 1024


def test_legacy_feature_dim_manifest_still_builds_equal_dim_adapter():
    manifest = {"feature_dim": 8}
    adapter = build_adapter_from_manifest(manifest, "mlp", hidden_dim=16)
    assert adapter.input_dim == 8
    assert adapter.output_dim == 8
    assert adapter.use_residual
