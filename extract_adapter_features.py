import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from adapter.data import assert_no_clip_overlap, list_clip_dirs, split_clips, validate_feature_pair
from adapter.features import extract_feature_pair


VGGT_MODEL_ID = "facebook/VGGT-1B"
DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Cache frozen DINOv2/DINOv3 features for adapter training")
    parser.add_argument("--data-dir", type=Path, default=Path("RealEstate10k_finetune"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("dataset/RealEstate10k"))
    parser.add_argument("--output-dir", type=Path, default=Path("adapter_data"))
    parser.add_argument("--model", default=VGGT_MODEL_ID)
    parser.add_argument("--expected-clips", type=int, default=100)
    parser.add_argument("--train-count", type=int, default=90)
    parser.add_argument("--frames-per-clip", type=int, default=20)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def prepare_clip_split(data_dir, evaluation_dir, expected_clips=100, train_count=90, seed=0, frames_per_clip=20):
    assert_no_clip_overlap(data_dir, evaluation_dir)
    clips = list_clip_dirs(data_dir, expected_frames=frames_per_clip)
    if len(clips) != expected_clips:
        raise ValueError(f"Expected exactly {expected_clips} finetune clips, got {len(clips)}")
    return split_clips(clips, train_count=train_count, seed=seed)


def save_clip_cache(path, clip, images, dino2, dino3):
    validate_feature_pair(dino2, dino3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "clip": str(clip),
        "images": list(images),
        "dino2": dino2.detach().to(device="cpu", dtype=torch.float16),
        "dino3": dino3.detach().to(device="cpu", dtype=torch.float16),
    }
    torch.save(record, path)


def build_manifest(train_clips, val_clips, seed=0, model_id=VGGT_MODEL_ID, dinov3_model_id=DINOV3_MODEL_ID):
    train_names = [Path(clip).name for clip in train_clips]
    val_names = [Path(clip).name for clip in val_clips]
    return {
        "seed": seed,
        "train_clips": train_names,
        "val_clips": val_names,
        "num_train_clips": len(train_names),
        "num_val_clips": len(val_names),
        "num_images": 20 * (len(train_names) + len(val_names)),
        "vggt_model": model_id,
        "dinov3_model": dinov3_model_id,
        "dino2_target_size": 518,
        "dino2_patch_size": 14,
        "dino3_target_size": 592,
        "dino3_patch_size": 16,
        "feature_dim": 1024,
    }


def choose_device_and_dtype(device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        major = torch.cuda.get_device_capability(device)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def load_feature_models(model_id, device, dtype):
    from transformers import AutoModel
    from vggt.models.vggt import VGGT

    vggt = VGGT.from_pretrained(model_id)
    teacher_patch = vggt.aggregator.patch_embed.eval().to(device=device, dtype=dtype)
    teacher_patch.requires_grad_(False)
    teacher = SimpleNamespace(aggregator=SimpleNamespace(patch_embed=teacher_patch))
    del vggt

    dinov3 = AutoModel.from_pretrained(DINOV3_MODEL_ID).eval().to(device=device, dtype=dtype)
    dinov3.requires_grad_(False)
    return teacher, dinov3


def list_clip_images(clip_dir):
    image_dir = Path(clip_dir) / "images"
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def extract_clip_features(clip_dir, teacher, dinov3, preprocess_fn, device, dtype, image_batch_size=2):
    image_paths = list_clip_images(clip_dir)
    if not image_paths:
        raise ValueError(f"{Path(clip_dir).name}: no images found")

    dino2_batches = []
    dino3_batches = []
    for start in range(0, len(image_paths), image_batch_size):
        batch_paths = image_paths[start:start + image_batch_size]
        path_strings = [str(path) for path in batch_paths]
        dino2_images = preprocess_fn(path_strings, target_size=518, patch_size=14).to(device)
        dino3_images = preprocess_fn(path_strings, target_size=592, patch_size=16).to(device)
        autocast = torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
        with autocast:
            dino2, dino3 = extract_feature_pair(teacher, dinov3, dino2_images, dino3_images)
        dino2_batches.append(dino2.detach().cpu())
        dino3_batches.append(dino3.detach().cpu())

    dino2 = torch.cat(dino2_batches, dim=0)
    dino3 = torch.cat(dino3_batches, dim=0)
    validate_feature_pair(dino2, dino3)
    return [path.name for path in image_paths], dino2, dino3


def main(argv=None):
    args = parse_args(argv)
    if args.image_batch_size < 1:
        raise ValueError("--image-batch-size must be at least 1")

    train_clips, val_clips = prepare_clip_split(
        args.data_dir, args.evaluation_dir, expected_clips=args.expected_clips,
        train_count=args.train_count, seed=args.seed, frames_per_clip=args.frames_per_clip,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = choose_device_and_dtype(args.device)
    teacher, dinov3 = load_feature_models(args.model, device, dtype)
    from vggt.utils.load_fn import load_and_preprocess_images

    manifest = build_manifest(train_clips, val_clips, seed=args.seed, model_id=args.model)
    manifest["frames_per_clip"] = args.frames_per_clip
    manifest["image_batch_size"] = args.image_batch_size
    manifest["device"] = str(device)

    for split_name, clips in (("train", train_clips), ("val", val_clips)):
        split_dir = args.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for index, clip in enumerate(clips, start=1):
            images, dino2, dino3 = extract_clip_features(
                clip, teacher, dinov3, load_and_preprocess_images, device, dtype,
                image_batch_size=args.image_batch_size,
            )
            save_clip_cache(split_dir / f"{clip.name}.pt", clip.name, images, dino2, dino3)
            print(f"[{split_name} {index}/{len(clips)}] {clip.name}: {tuple(dino2.shape)}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
