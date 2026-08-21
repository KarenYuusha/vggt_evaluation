# VGGT Evaluation

Evaluate VGGT camera pose estimation on the included CO3Dv2 and RealEstate10K subsets using AUC@30 and runtime.

## Setup

```powershell
uv pip install -r requirements.txt
uv run hf auth login
```

The default VGGT checkpoint is `facebook/VGGT-1B`. DINOv3 uses the gated Hugging Face checkpoint `facebook/dinov3-vitl16-pretrain-lvd1689m`.

## DINOv2 baseline

CO3Dv2:

```powershell
uv run python evaluate.py --dataset co3d --data-dir dataset\CO3DV2 --num-frames 10 --seed 0 --backbone dinov2 --output results\vggt_dinov2_co3d.json
```

RealEstate10K:

```powershell
uv run python evaluate.py --dataset realestate10k --data-dir dataset\RealEstate10k --num-frames 10 --backbone dinov2 --output results\vggt_dinov2_realestate10k.json
```

The original VGGT path uses DINOv2 ViT-L/14 at target size 518.

## DINOv3 paper configuration

The DINOv3 paper's VGGT experiment changes the image backbone to DINOv3 ViT-L/16, uses target size 592, and uses the concatenation of four intermediate ViT-L layers rather than only the final layer. The official DINOv3 implementation uses ViT-L block indices `[4, 11, 17, 23]` for this four-level representation.

`paper4` mode implements the published feature extraction exactly:

```text
image @ 592px
 -> frozen DINOv3 ViT-L/16
 -> blocks [4, 11, 17, 23]
 -> apply DINOv3 final LayerNorm to each block
 -> remove CLS + register tokens
 -> concatenate channels
 -> [B, P, 4096]
```

The DINOv3 paper also runs the VGGT training pipeline and fine-tunes the image backbone. The public frozen `facebook/VGGT-1B` checkpoint expects 1024-D patch tokens, while `paper4` produces 4096-D tokens. The paper/released DINOv3 repository does not publish the trained 4096-to-1024 VGGT interface or its weights. Therefore, this repository intentionally does **not** invent a projection and label it as the paper implementation.

`--backbone dinov3` now defaults to `--dinov3-feature-mode paper4`. Attempting to run `paper4` directly through the frozen public VGGT produces a clear error explaining this missing interface instead of silently falling back to the final DINOv3 layer.

## Legacy frozen DINOv3 final-layer experiment

The previous zero-shot experiment used only the final normalized DINOv3 patch features. It remains available explicitly for reproducibility, but it is **not** the main DINOv3 paper configuration.

CO3Dv2:

```powershell
uv run python evaluate.py --dataset co3d --data-dir dataset\CO3DV2 --num-frames 10 --seed 0 --backbone dinov3 --dinov3-feature-mode last --output results\vggt_dinov3_last_co3d.json
```

RealEstate10K:

```powershell
uv run python evaluate.py --dataset realestate10k --data-dir dataset\RealEstate10k --num-frames 10 --backbone dinov3 --dinov3-feature-mode last --output results\vggt_dinov3_last_realestate10k.json
```

This path uses 592px input, patch size 16, removes CLS/register tokens, and returns `[B, P, 1024]` final-layer features to the frozen VGGT transformer.

## Feature-alignment adapter experiments

The linear and residual-MLP adapters developed in this repository are separate custom experiments. They operate on the legacy 1024-D final-layer DINOv3 representation, not on the paper's 4096-D four-layer representation.

Cache features:

```powershell
uv run python extract_adapter_features.py --data-dir RealEstate10k_finetune --evaluation-dir dataset\RealEstate10k --output-dir adapter_data --seed 0 --image-batch-size 2
```

Train the linear adapter:

```powershell
uv run python train_adapter.py --cache-dir adapter_data --output-dir adapter_checkpoints --adapter-type linear --epochs 20 --batch-size 4096 --lr 1e-3 --weight-decay 1e-4 --seed 0
```

Train the residual MLP adapter:

```powershell
uv run python train_adapter.py --cache-dir adapter_data --output-dir adapter_checkpoints_mlp --adapter-type mlp --hidden-dim 2048 --epochs 20 --batch-size 4096 --lr 1e-3 --weight-decay 1e-4 --seed 0
```

Evaluate an adapter by explicitly selecting the legacy final-layer representation:

```powershell
uv run python evaluate.py --dataset co3d --data-dir dataset\CO3DV2 --num-frames 10 --seed 0 --backbone dinov3 --dinov3-feature-mode last --adapter-checkpoint adapter_checkpoints\best_adapter.pt --output results\vggt_dinov3_linear_adapter_co3d.json
```

## Evaluation notes

CO3Dv2 poses are converted to OpenCV world-to-camera convention before relative-pose evaluation. Each 10-frame scene produces 45 camera pairs. The reported dataset score is AUC@30. Model loading and warm-up are excluded from inference timing.

When reporting results, distinguish the following configurations explicitly:

```text
DINOv2 VGGT baseline
DINOv3 paper4 feature extraction (paper configuration; downstream bridge/training unavailable for frozen public VGGT)
DINOv3 final-layer frozen swap (legacy zero-shot experiment)
DINOv3 final-layer + custom feature adapter (our custom experiment)
```
