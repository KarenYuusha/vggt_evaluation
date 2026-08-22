# VGGT Evaluation

Evaluate feed-forward VGGT camera pose estimation on the included CO3Dv2 and RealEstate10K subsets using AUC@30 and runtime.

The repository now exposes three main backbone paths:

```text
dinov2              original pretrained VGGT + DINOv2 ViT-L/14
dinov3 / dinov3-final
                    frozen zero-shot DINOv3 ViT-L/16 final-layer substitution
dinov3-multilayer   frozen DINOv3 four-layer features + trained 4096->1024 adapter + frozen VGGT
```

## Important: relation to the DINOv3 paper

The DINOv3 paper's VGGT experiment does more than replace DINOv2 at inference time. It uses DINOv3 ViT-L/16, changes the image target from 518 to 592, concatenates four intermediate DINOv3 layers, and trains/fine-tunes the modified VGGT pipeline including the image backbone.

This repository intentionally keeps DINOv3 and the released pretrained VGGT frozen. Therefore `dinov3-multilayer` is a **paper-aligned frozen-backbone approximation**, not a reproduction of the paper's reported DINOv3-VGGT numbers.

The multi-layer path uses the DINOv3 ViT-L block indices `[4, 11, 17, 23]`, matching the four evenly spaced ViT-L indices identified as used in the official DINOv3 code. Each selected intermediate state is normalized with the DINOv3 model norm, CLS/register tokens are removed, and the four 1024-dimensional patch features are concatenated:

```text
block 4   [P, 1024] --\
block 11  [P, 1024] ---\
block 17  [P, 1024] ----> concatenate -> [P, 4096] -> learned adapter -> [P, 1024] -> frozen VGGT
block 23  [P, 1024] ---/
```

A trained adapter is mandatory for `dinov3-multilayer`; the evaluator refuses to insert a random projection.

## Setup

Install dependencies:

```powershell
uv pip install -r requirements.txt
```

The evaluator loads `facebook/VGGT-1B` by default. For DINOv3, obtain access to the gated Hugging Face model `facebook/dinov3-vitl16-pretrain-lvd1689m`, then authenticate:

```powershell
uv run hf auth login
uv run hf auth whoami
```

`transformers>=4.56.0` is required for the `dinov3_vit` architecture.

## DINOv2 baseline

DINOv2 preserves the original VGGT preprocessing: target size 518, patch size 14.

```powershell
uv run python evaluate.py --dataset co3d --data-dir dataset\CO3DV2 --num-frames 10 --seed 0 --backbone dinov2 --output results\vggt_dinov2_co3d.json

uv run python evaluate.py --dataset realestate10k --data-dir dataset\RealEstate10k --num-frames 10 --backbone dinov2 --output results\vggt_dinov2_realestate10k.json
```

## Zero-shot DINOv3 final-layer baseline

This mode replaces only `model.aggregator.patch_embed`. DINOv3 and VGGT are frozen. It removes the DINOv3 CLS/register tokens and sends the normalized final-layer 1024-D patch tokens directly to the pretrained VGGT transformer.

DINOv3 uses target size 592 and patch size 16:

```text
518 / 14 = 37
592 / 16 = 37
```

so both square preprocessing paths keep a 37 x 37 = 1369 patch grid.

`dinov3` remains an alias of `dinov3-final` for backward compatibility.

```powershell
uv run python evaluate.py --dataset co3d --data-dir dataset\CO3DV2 --num-frames 10 --seed 0 --backbone dinov3-final --output results\vggt_dinov3_final_co3d.json

uv run python evaluate.py --dataset realestate10k --data-dir dataset\RealEstate10k --num-frames 10 --backbone dinov3-final --output results\vggt_dinov3_final_realestate10k.json
```

This is a **zero-shot backbone substitution experiment**, not the paper's trained DINOv3-VGGT model.

## Paper-aligned frozen multi-layer adapter

The closest supported setup to the paper while keeping the large models frozen is:

```text
image @ 592
 -> frozen DINOv3 ViT-L/16
 -> normalized blocks [4, 11, 17, 23]
 -> concatenate [P, 4096]
 -> trained adapter 4096 -> 1024
 -> frozen pretrained VGGT frame/global transformer
 -> frozen CameraHead
 -> camera pose
```

Only the adapter is trained.

### Training data layout

The proof-of-concept feature-alignment dataset is expected at:

```text
RealEstate10k_finetune/
├── clip_001/
│   └── images/        # 20 frames
├── clip_002/
│   └── images/        # 20 frames
└── ...                # exactly 100 clips
```

The script deterministically splits the clips into 90 training and 10 validation clips. It aborts if a clip ID overlaps with the final RealEstate10K evaluation set.

### 1. Cache frozen features

Multi-layer extraction is the default:

```powershell
uv run python extract_adapter_features.py --data-dir RealEstate10k_finetune --evaluation-dir dataset\RealEstate10k --output-dir adapter_data_multilayer --feature-mode multilayer --seed 0 --image-batch-size 2
```

For each image the cache contains:

```text
VGGT DINOv2 ViT-L/14 @ 518px  -> target [P, 1024]
DINOv3 blocks 4/11/17/23 @ 592px -> source [P, 4096]
```

The DINOv3 intermediate states are normalized before their CLS/register tokens are removed and before concatenation. Cached tensors are stored as FP16 on CPU/disk.

The manifest records the feature mode, selected block indices, source/target dimensions, model IDs, patch sizes, and train/validation split.

### 2. Train a linear 4096 -> 1024 projection

This is the simplest adapter and the recommended first multi-layer experiment:

```powershell
uv run python train_adapter.py --cache-dir adapter_data_multilayer --output-dir adapter_checkpoints_multilayer_linear --adapter-type linear --epochs 20 --batch-size 4096 --lr 1e-3 --weight-decay 1e-4 --seed 0
```

### 3. Train an MLP 4096 -> 1024 projection

```powershell
uv run python train_adapter.py --cache-dir adapter_data_multilayer --output-dir adapter_checkpoints_multilayer_mlp --adapter-type mlp --hidden-dim 2048 --epochs 20 --batch-size 4096 --lr 1e-3 --weight-decay 1e-4 --seed 0
```

For equal-dimensional legacy adapters the MLP remains residual. For the 4096 -> 1024 multi-layer adapter an identity residual is mathematically impossible, so the MLP is a normal projection:

```text
4096 -> Linear -> GELU -> Linear -> 1024
```

Both adapter types use the existing feature-alignment loss:

```text
MSE(predicted_dino2, target_dino2)
+ mean(1 - cosine_similarity(predicted_dino2, target_dino2))
```

DINOv2, DINOv3, and VGGT are not loaded during adapter training because training reads the cached feature tensors only.

### 4. Evaluate the multi-layer adapter

RealEstate10K:

```powershell
uv run python evaluate.py --dataset realestate10k --data-dir dataset\RealEstate10k --num-frames 10 --backbone dinov3-multilayer --adapter-checkpoint adapter_checkpoints_multilayer_linear\best_adapter.pt --output results\vggt_dinov3_multilayer_realestate10k.json
```

CO3Dv2:

```powershell
uv run python evaluate.py --dataset co3d --data-dir dataset\CO3DV2 --num-frames 10 --seed 0 --backbone dinov3-multilayer --adapter-checkpoint adapter_checkpoints_multilayer_linear\best_adapter.pt --output results\vggt_dinov3_multilayer_co3d.json
```

The evaluator verifies that the checkpoint is exactly 4096 -> 1024. A final-layer 1024 -> 1024 adapter cannot accidentally be used in this mode.

## Legacy final-layer feature adapter

The previous 1024 -> 1024 feature-alignment experiment is still supported. Generate its cache explicitly:

```powershell
uv run python extract_adapter_features.py --data-dir RealEstate10k_finetune --evaluation-dir dataset\RealEstate10k --output-dir adapter_data_final --feature-mode final --seed 0 --image-batch-size 2
```

Then train normally with `train_adapter.py` and evaluate with `--backbone dinov3-final --adapter-checkpoint ...`.

Old adapter checkpoints containing only the legacy `dim` field remain loadable.

## Recommended comparison

Use the same dataset, seed, frame count, and metric settings for all runs:

```text
1. DINOv2 baseline
2. DINOv3 final layer, zero-shot
3. DINOv3 final layer + trained 1024 -> 1024 adapter
4. DINOv3 [4, 11, 17, 23] + trained 4096 -> 1024 adapter
```

Experiment 4 is the closest configuration in this repository to the paper's four-intermediate-layer feature design. The remaining major difference is that the paper fine-tunes the modified VGGT pipeline whereas this repository freezes DINOv3 and VGGT.

## CO3Dv2 subset

The loader prefers `set_lists_fewview_dev.json` when available. For the official CO3D single-sequence subset, where that file is absent, it falls back to available `set_lists_manyview_dev_*.json` files. Sequences must have a finite `viewpoint_quality_score > 0.5`, camera poses must be finite, and frames are sampled deterministically using the requested seed.

Raw CO3D PyTorch3D poses are converted to OpenCV world-to-camera poses using the same conversion as the public VGGT evaluator. Scores from this single-sequence subset are custom subset results, not the official full CO3Dv2 benchmark.

## RealEstate10K

Each evaluation scene must contain `images/` and `selected_frames.txt`. The evaluator uses the exact timestamped frames selected for the scene and reads the final 12 metadata values as the 3 x 4 world-to-camera pose.

## Timing and outputs

Model loading and warm-up are excluded from timing. `inference_ms` covers:

```text
selected image backbone -> optional adapter -> VGGT aggregator -> CameraHead -> pose decoding
```

Image loading/preprocessing is reported separately. CUDA timing uses synchronized CUDA events; CPU timing uses `time.perf_counter()`.

Results default to `results/vggt_<dataset>.json`. Each output records `backbone`, `adapter_checkpoint`, `input_target_size`, `patch_size`, AUC@30, runtime statistics, and per-scene metrics. CO3D output also includes per-category AUC@30.
