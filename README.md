# VGGT Evaluation

Evaluate feed-forward VGGT camera pose estimation on the included CO3Dv2 single-sequence subset and RealEstate10K subset using AUC@30 and runtime.

## Setup

Install the public VGGT repository in the same Python environment. With `uv` on Windows, for example:

```powershell
uv pip install -e ..\vggt
```

The evaluator uses VGGT's pretrained model from Hugging Face by default. You can also pass a local `model.pt` checkpoint with `--model`.

## CO3Dv2

The included CO3Dv2 sample is the many-view single-sequence subset. Evaluation uses the `val` entries in `set_lists_manyview_test_0.json`, deterministically samples 10 frames per sequence, and converts raw PyTorch3D poses to OpenCV world-to-camera poses using the same conversion as the public VGGT evaluator.

```powershell
uv run python evaluate.py --dataset co3d --data-dir dataset\CO3DV2 --num-frames 10 --seed 0
```

The dataset-level CO3D score is the mean AUC@30 across available categories. This is a custom subset score, not the official full CO3Dv2 benchmark result.

## RealEstate10K

Each RealEstate10K scene must contain `images/` and `selected_frames.txt`. The evaluator uses the exact timestamped frames already selected in each scene and reads the final 12 numbers of each metadata row as the 3x4 world-to-camera pose.

```powershell
uv run python evaluate.py --dataset realestate10k --data-dir dataset\RealEstate10k --num-frames 10
```

## Timing

Model loading and the warm-up pass are excluded from timing. The reported `inference_ms` covers only the camera-pose path:

```text
VGGT aggregator -> camera head -> pose decoding
```

Image loading/preprocessing is reported separately. CUDA timing uses synchronized CUDA events; CPU timing uses `time.perf_counter()`.

## Outputs

Results are written to `results/vggt_<dataset>.json` unless `--output` is supplied. Each result includes dataset AUC@30, average/std inference time, preprocessing time, total scene time, and per-scene metrics. CO3D output also includes per-category AUC@30.

Use `--max-scenes N` for a quick smoke test.
