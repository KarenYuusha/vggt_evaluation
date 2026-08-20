# VGGT Evaluation

Evaluate feed-forward VGGT camera pose estimation on the included CO3Dv2 subset and RealEstate10K subset using AUC@30 and runtime. The evaluator supports the original DINOv2 backbone and a zero-shot DINOv3 ViT-L/16 backbone substitution.

## Setup

Install the project dependencies in the same Python environment. The evaluator loads `facebook/VGGT-1B` by default; you can also pass a local VGGT checkpoint with `--model`.

For DINOv3, clone the official repository once:

```powershell
git clone https://github.com/facebookresearch/dinov3.git
```

Use the ViT-L/16 weight URL from the Meta acceptance email directly with `--dinov3-weights`, or download that checkpoint locally and pass its path. Do not commit the private weight URL to this repository.

## DINOv2 baseline

CO3Dv2:

```powershell
uv run python evaluate.py --dataset co3d --data-dir dataset\CO3DV2 --num-frames 10 --seed 0 --backbone dinov2
```

RealEstate10K:

```powershell
uv run python evaluate.py --dataset realestate10k --data-dir dataset\RealEstate10k --num-frames 10 --backbone dinov2
```

The DINOv2 path preserves the original VGGT preprocessing: target size 518 and patch size 14.

## Zero-shot DINOv3 ViT-L/16

The DINOv3 path first loads the pretrained VGGT checkpoint and then replaces only `model.aggregator.patch_embed` with the official pretrained DINOv3 ViT-L/16 backbone. It uses final-layer normalized patch tokens (`x_norm_patchtokens`) with no learned adapter, no training, and no fine-tuning.

DINOv3 uses target size 592 and patch size 16. This preserves the patch-grid scale used by VGGT because `518 / 14 = 592 / 16 = 37`.

CO3Dv2:

```powershell
uv run python evaluate.py --dataset co3d --data-dir dataset\CO3DV2 --num-frames 10 --seed 0 --backbone dinov3 --dinov3-repo .\dinov3 --dinov3-weights "<URL_OR_LOCAL_PATH_FROM_META_EMAIL>"
```

RealEstate10K:

```powershell
uv run python evaluate.py --dataset realestate10k --data-dir dataset\RealEstate10k --num-frames 10 --backbone dinov3 --dinov3-repo .\dinov3 --dinov3-weights "<URL_OR_LOCAL_PATH_FROM_META_EMAIL>"
```

This is a **zero-shot backbone substitution experiment**. The DINOv3 paper's reported VGGT results used a trained modified VGGT pipeline, so these results should not be described as reproducing the paper's DINOv3-VGGT numbers.

## CO3Dv2 subset

The loader prefers `set_lists_fewview_dev.json` when it is available. For the official CO3D single-sequence subset, where that file is absent, it falls back to available `set_lists_manyview_dev_*.json` files. Sequences must have a finite `viewpoint_quality_score > 0.5`, camera poses must be finite, and 10 frames are sampled deterministically from the selected dev sequence using the requested seed.

Raw CO3D PyTorch3D poses are converted to OpenCV world-to-camera poses using the same conversion as the public VGGT evaluator. The dataset-level score is the mean AUC@30 across available categories. Scores from the single-sequence subset are custom subset results, not the official full CO3Dv2 benchmark.

## RealEstate10K

Each scene must contain `images/` and `selected_frames.txt`. The evaluator uses the exact timestamped frames already selected in each scene and reads the final 12 numbers of each metadata row as the 3x4 world-to-camera pose.

## Timing

Model loading and the warm-up pass are excluded from timing. The reported `inference_ms` covers:

```text
selected image backbone -> VGGT frame/global aggregator -> camera head -> pose decoding
```

Image loading/preprocessing is reported separately. CUDA timing uses synchronized CUDA events; CPU timing uses `time.perf_counter()`.

## Outputs

Results are written to `results/vggt_<dataset>.json` unless `--output` is supplied. Each JSON file records `backbone`, `input_target_size`, and `patch_size` in addition to dataset AUC@30, average/std inference time, preprocessing time, total scene time, and per-scene metrics. CO3D output also includes per-category AUC@30.

Use `--max-scenes N` for a quick smoke test. When comparing DINOv2 and DINOv3, use the same dataset, seed, frame count, and output metric settings.
