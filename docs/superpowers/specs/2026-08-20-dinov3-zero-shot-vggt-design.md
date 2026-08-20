# Zero-Shot DINOv3 Backbone in VGGT

## Goal

Add a DINOv3 ViT-L/16 evaluation path to the existing VGGT camera-pose evaluator without any training or fine-tuning. The DINOv2 baseline must remain unchanged so DINOv2 and DINOv3 can be compared on the same datasets, frames, AUC@30 implementation, and timing methodology.

## Experiment Definition

The DINOv3 path is a zero-shot backbone substitution, not a reproduction of the trained DINOv3-VGGT model from the DINOv3 paper.

- Start from the pretrained `facebook/VGGT-1B` checkpoint.
- Keep the pretrained VGGT frame-attention blocks, global-attention blocks, camera/register tokens, and CameraHead unchanged.
- Replace only `model.aggregator.patch_embed` with a pretrained DINOv3 ViT-L/16 backbone.
- Use only final-layer normalized DINOv3 patch tokens (`x_norm_patchtokens`).
- Do not add a learned projection layer.
- Do not train or fine-tune DINOv3 or VGGT.

## DINOv3 Loading

Use the official DINOv3 PyTorch Hub interface from a locally cloned DINOv3 repository:

```python
torch.hub.load(
    dinov3_repo,
    "dinov3_vitl16",
    source="local",
    weights=dinov3_weights,
)
```

`dinov3_weights` may be either a local checkpoint path or the private weight URL supplied by Meta after access approval. No weight URL is stored in this repository.

## Input and Token Geometry

The original VGGT DINOv2 ViT-L/14 path uses a 518-pixel target width and patch size 14. The DINOv3 path uses a 592-pixel target width and patch size 16.

This preserves the patch-grid scale:

- `518 / 14 = 37`
- `592 / 16 = 37`

For non-square images, the existing aspect-ratio-preserving crop preprocessing remains the same except that rounding uses the selected patch size. This preserves the corresponding patch-grid dimensions between the DINOv2 and DINOv3 paths for the same aspect ratio.

Both backbones use ImageNet mean/std normalization already performed inside VGGT's Aggregator.

## Architecture

DINOv2 baseline:

```text
images -> existing VGGT preprocessing (518, patch 14)
       -> DINOv2 ViT-L/14 patch encoder
       -> VGGT alternating frame/global attention
       -> existing CameraHead
       -> pose decoding
       -> AUC@30
```

DINOv3 zero-shot:

```text
images -> same preprocessing policy (592, patch 16)
       -> DINOv3 ViT-L/16 final normalized patch tokens
       -> same pretrained VGGT alternating frame/global attention
       -> same pretrained CameraHead
       -> same pose decoding
       -> same AUC@30
```

The DINOv3 patch-token width is 1024, matching the 1024-dimensional patch tokens expected by VGGT's fusion transformer. The camera head still consumes the existing 2048-dimensional concatenated frame/global token representation.

## Code Boundaries

- `evaluation/dinov3_backbone.py`: load and wrap the official DINOv3 ViT-L/16 model and expose normalized patch tokens.
- `evaluation/model_runner.py`: select DINOv2 or DINOv3, replace the patch encoder only for DINOv3, set the aggregator patch size, and select preprocessing parameters.
- `vggt/utils/load_fn.py`: parameterize target size and patch size while preserving defaults (`518`, `14`).
- `evaluate.py`: add `--backbone`, `--dinov3-repo`, and `--dinov3-weights` CLI options and validate the DINOv3-only arguments.
- Tests: verify token extraction, frozen weights, 518/14 vs 592/16 grid equivalence, baseline compatibility, and CLI argument validation.

## Timing

Reuse the existing timing code without modification to the definition of inference time. Inference time includes:

- selected image backbone,
- VGGT frame/global transformer,
- CameraHead,
- pose decoding.

Preprocessing remains reported separately. Warm-up remains excluded.

## Result Metadata

Saved JSON results must identify the backbone and input configuration so results cannot be confused later:

```json
{
  "backbone": "dinov3",
  "input_target_size": 592,
  "patch_size": 16
}
```

The DINOv2 baseline records `518` and `14` respectively.

## Non-Goals

- No VGGT fine-tuning.
- No DINOv3 fine-tuning.
- No four-layer DINOv3 feature concatenation.
- No learned adapter/projection.
- No bundle adjustment changes.
- No changes to dataset selection or pose/AUC metrics.
