import pytest
import torch

from extract_adapter_features import build_manifest, parse_args, prepare_clip_split, save_clip_cache


def make_dataset(root, count, frames=20):
    for clip_idx in range(count):
        image_dir = root / f"clip_{clip_idx:03d}" / "images"
        image_dir.mkdir(parents=True)
        for frame_idx in range(frames):
            (image_dir / f"{frame_idx:06d}.jpg").touch()


def test_cli_defaults_match_proof_of_concept():
    args = parse_args([])
    assert str(args.data_dir) == "RealEstate10k_finetune"
    assert str(args.evaluation_dir) == "dataset/RealEstate10k"
    assert str(args.output_dir) == "adapter_data"
    assert args.expected_clips == 100
    assert args.train_count == 90
    assert args.frames_per_clip == 20
    assert args.image_batch_size == 2


def test_prepare_split_has_90_train_10_val_and_rejects_overlap(tmp_path):
    finetune = tmp_path / "finetune"
    evaluation = tmp_path / "evaluation"
    make_dataset(finetune, 100)
    evaluation.mkdir()

    train, val = prepare_clip_split(finetune, evaluation, expected_clips=100, train_count=90, seed=0)
    assert len(train) == 90
    assert len(val) == 10

    overlap = evaluation / train[0].name
    (overlap / "images").mkdir(parents=True)
    with pytest.raises(ValueError, match="overlap"):
        prepare_clip_split(finetune, evaluation, expected_clips=100, train_count=90, seed=0)


def test_save_cache_record_uses_fp16_and_manifest_records_split(tmp_path):
    path = tmp_path / "clip.pt"
    dino2 = torch.randn(20, 5, 1024)
    dino3 = torch.randn(20, 5, 1024)
    images = [f"{i}.jpg" for i in range(20)]
    save_clip_cache(path, "clip_a", images, dino2, dino3)

    record = torch.load(path)
    assert record["dino2"].dtype == torch.float16
    assert record["dino3"].dtype == torch.float16
    assert record["dino2"].device.type == "cpu"
    assert record["images"] == images

    manifest = build_manifest(["a", "b"], ["c"], seed=7, frames_per_clip=3)
    assert manifest["train_clips"] == ["a", "b"]
    assert manifest["val_clips"] == ["c"]
    assert manifest["seed"] == 7
    assert manifest["num_images"] == 9
