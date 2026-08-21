from pathlib import Path

import pytest
import torch

from adapter.data import assert_no_clip_overlap, list_clip_dirs, split_clips, validate_feature_pair


def make_clip(root, name, frames=20):
    image_dir = root / name / "images"
    image_dir.mkdir(parents=True)
    for i in range(frames):
        (image_dir / f"{i:06d}.jpg").touch()
    return root / name


def test_split_clips_is_deterministic_and_disjoint(tmp_path):
    clips = [make_clip(tmp_path, f"clip_{i:03d}") for i in range(100)]
    listed = list_clip_dirs(tmp_path)
    train_a, val_a = split_clips(listed, train_count=90, seed=0)
    train_b, val_b = split_clips(listed, train_count=90, seed=0)

    assert len(train_a) == 90
    assert len(val_a) == 10
    assert [p.name for p in train_a] == [p.name for p in train_b]
    assert [p.name for p in val_a] == [p.name for p in val_b]
    assert set(train_a).isdisjoint(val_a)


def test_list_clip_dirs_rejects_wrong_frame_count(tmp_path):
    make_clip(tmp_path, "bad", frames=19)
    with pytest.raises(ValueError, match="20"):
        list_clip_dirs(tmp_path)


def test_overlap_guard_rejects_evaluation_clip_id(tmp_path):
    finetune = tmp_path / "finetune"
    evaluation = tmp_path / "evaluation"
    make_clip(finetune, "same")
    make_clip(evaluation, "same")

    with pytest.raises(ValueError, match="overlap"):
        assert_no_clip_overlap(finetune, evaluation)


def test_feature_pair_validation_requires_exact_matching_shape():
    validate_feature_pair(torch.zeros(20, 100, 1024), torch.zeros(20, 100, 1024))

    with pytest.raises(ValueError, match="matching"):
        validate_feature_pair(torch.zeros(20, 100, 1024), torch.zeros(20, 101, 1024))
    with pytest.raises(ValueError, match="1024"):
        validate_feature_pair(torch.zeros(20, 100, 512), torch.zeros(20, 100, 512))
