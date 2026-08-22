from pathlib import Path
import random


def list_clip_dirs(root, expected_frames=20):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    clips = sorted(path for path in root.iterdir() if path.is_dir())
    for clip in clips:
        image_dir = clip / "images"
        if not image_dir.is_dir():
            raise ValueError(f"{clip.name}: missing images directory")
        images = sorted(path for path in image_dir.iterdir() if path.is_file())
        if len(images) != expected_frames:
            raise ValueError(f"{clip.name}: expected {expected_frames} image files, got {len(images)}")
    return clips


def split_clips(clips, train_count=90, seed=0):
    clips = sorted(Path(path) for path in clips)
    if len(clips) <= train_count:
        raise ValueError(f"Need more than {train_count} clips, got {len(clips)}")
    rng = random.Random(seed)
    shuffled = clips.copy()
    rng.shuffle(shuffled)
    return shuffled[:train_count], shuffled[train_count:]


def assert_no_clip_overlap(finetune_root, evaluation_root):
    finetune_root = Path(finetune_root)
    evaluation_root = Path(evaluation_root)
    finetune_ids = {path.name for path in finetune_root.iterdir() if path.is_dir()}
    evaluation_ids = {path.name for path in evaluation_root.iterdir() if path.is_dir()}
    overlap = sorted(finetune_ids & evaluation_ids)
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(f"Finetune/evaluation clip overlap detected: {preview}")


def validate_feature_pair(dino2, dino3, expected_dino2_dim=1024, expected_dino3_dim=None):
    if dino2.ndim != 3 or dino3.ndim != 3:
        raise ValueError(
            f"Expected feature tensors shaped [frames, patches, dim], got {tuple(dino2.shape)} and {tuple(dino3.shape)}"
        )
    if tuple(dino2.shape[:2]) != tuple(dino3.shape[:2]):
        raise ValueError(
            f"DINOv2 and DINOv3 must have matching frame/patch axes, got {tuple(dino2.shape[:2])} and {tuple(dino3.shape[:2])}"
        )
    if expected_dino2_dim is not None and dino2.shape[-1] != expected_dino2_dim:
        raise ValueError(f"Expected DINOv2 feature dim {expected_dino2_dim}, got {dino2.shape[-1]}")
    if expected_dino3_dim is not None and dino3.shape[-1] != expected_dino3_dim:
        raise ValueError(f"Expected DINOv3 feature dim {expected_dino3_dim}, got {dino3.shape[-1]}")
