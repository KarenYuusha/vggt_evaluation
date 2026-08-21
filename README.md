# VGGT Evaluation

Evaluate feed-forward VGGT camera pose estimation on the included CO3Dv2 subset and RealEstate10K subset using AUC@30 and runtime. The evaluator supports the original DINOv2 backbone and a zero-shot DINOv3 ViT-L/16 backbone substitution.

## Setup

Install the project dependencies:

```powershell
uv pip install -r requirements.txt
```

The evaluator loads `facebook/VGGT-1B` by default; you can also pass a local VGGT checkpoint with `--model`.

For DINOv3, access is provided through the gated Hugging Face repository `facebook/dinov3-vitl16-pretrain-lvd1689m`. Log in once using the same Hugging Face account that was approved:

```powershell
uv run hf auth login
uv run hf auth whoami
```

`transformers>=4.56.0` is required for the `dinov3_vit` architecture. Hugging Face downloads and caches the checkpoint automatically on the first DINOv3 run.

## DINOv2 baseline

CO3Dv2:

```powershell
uv run python evaluate.py --dataset co3d --data-dir dataset\CO3DV2 --num-frames 10 --seed 0 --backbone dinov2 --output results\vggt_dinov2_co3d.json
```

RealEstate10K:

```powershell
uv run python evaluate.py --dataset realestate10k --data-dir dataset\RealEstate10k --num-frames 10 --backbone dinov2 --output results\vggt_dinov2_realestate10k.json
```

The DINOv2 path preserves the original VGGT preprocessing: target size 518 and patch size 14.

## Zero-shot DINOv3 ViT-L/16

The DINOv3 path loads `facebook/dinov3-vitl16-pretrain-lvd1689m` with Hugging Face Transformers and replaces only `model.aggregator.patch_embed`. It removes the DINOv3 CLS and register tokens, then passes only final-layer 1024-dimensional patch tokens into the pretrained VGGT frame/global transformer. There is no learned adapter, training, or fine-tuning.

DINOv3 uses target size 592 and patch size 16. This preserves VGGT's 37x37 patch grid because `518 / 14 = 592 / 16 = 37`.

CO3Dv2:

```powershell
uv run python evaluate.py --dataset co3d --data-dir dataset\CO3DV2 --num-frames 10 --seed 0 --backbone dinov3 --output results\vggt_dinov3_co3d.json
```

RealEstate10K:

```powershell
uv run python evaluate.py --dataset realestate10k --data-dir dataset\RealEstate10k --num-frames 10 --backbone dinov3 --output results\vggt_dinov3_realestate10k.json
```

This is a **zero-shot backbone substitution experiment**. The DINOv3 paper's reported VGGT results used a trained modified VGGT pipeline, so these results should not be described as reproducing the paper's DINOv3-VGGT numbers.

## CO3Dv2 subset

The loader prefers `set_lists_fewview_dev.json` when it is available. For the official CO3D single-sequence subset, where that file is absent, it falls back to available `set_lists_manyview_dev_*.json` files. Sequences must have a finite `viewpoint_quality_score > 0.5`, camera poses must be finite, and 10 frames are sampled deterministically from the selected dev sequence using the requested seed.

Raw CO3D PyTorch3D poses are converted to OpenCV world-to-camera poses using the same conversion as the public VGGT evaluator. The dataset-level score is the mean AUC@30 across available categories. Scores from the single-sequence subset are custom subset results, not the official full CO3Dv2 benchmark.

## RealEstate10K

Each RealEstate10K scene must contain `images/` and `selected_frames.txt`. The evaluator uses the exact timestamped frames already selected in each scene and reads the final 12 numbers of each metadata row as the 3x4 world-to-camera pose.

## Timing

Model loading and the warm-up pass are excluded from timing. The reported `inference_ms` covers:

```text
selected image backbone -> VGGT frame/global aggregator -> camera head -> pose decoding
```

Image loading/preprocessing is reported separately. CUDA timing uses synchronized CUDA events; CPU timing uses `time.perf_counter()`.

## Outputs

Results are written to `results/vggt_<dataset>.json` unless `--output` is supplied. Each JSON file records `backbone`, `input_target_size`, and `patch_size` in addition to dataset AUC@30, average/std inference time, preprocessing time, total scene time, and per-scene metrics. CO3D output also includes per-category AUC@30.

Use `--max-scenes N` for a quick smoke test. When comparing DINOv2 and DINOv3, use the same dataset, seed, frame count, and output metric settings.
