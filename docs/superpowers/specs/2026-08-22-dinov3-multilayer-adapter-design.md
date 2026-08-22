# DINOv3 Multi-Layer Adapter Design

## Goal

Make the repository's DINOv3→VGGT path as close as practical to the DINOv3 paper while preserving the project constraint that the released DINOv3 backbone and pretrained VGGT remain frozen. Only a lightweight adapter is trained.

## Paper alignment

The DINOv3 paper's VGGT experiment replaces the original DINOv2 ViT-L/14 backbone with DINOv3 ViT-L/16, changes the image target from 518 to 592 so the patch grid remains 37×37, concatenates four intermediate DINOv3 ViT-L layers, and fine-tunes the resulting VGGT pipeline.

The official DINOv3 code identifies ViT-L blocks `[4, 11, 17, 23]` as the four evenly spaced intermediate layers used in the paper. This project will use those zero-based indices.

This repository intentionally differs from the paper in one major way: DINOv3 and VGGT stay frozen. Because four 1024-dimensional DINOv3 features concatenate to 4096 dimensions while pretrained VGGT expects 1024-dimensional patch tokens, a learned projection is required.

## Architecture

The new paper-aligned frozen-backbone path is:

```text
image @ 592px
  -> frozen DINOv3 ViT-L/16
  -> intermediate blocks [4, 11, 17, 23]
  -> remove CLS/register tokens from each layer
  -> concatenate along feature dimension
  -> [batch, patches, 4096]
  -> trainable adapter 4096 -> 1024
  -> frozen pretrained VGGT frame/global transformer
  -> frozen CameraHead
  -> camera pose
```

The original DINOv2 baseline and existing DINOv3 final-layer path remain available for ablation and backward compatibility.

## Feature extraction

`evaluation/dinov3_backbone.py` and `adapter/features.py` will expose a shared DINOv3 extraction path supporting:

- `final`: final hidden-state patch tokens, shape `[N, P, 1024]`.
- `multilayer`: concatenated blocks `[4, 11, 17, 23]`, shape `[N, P, 4096]`.

For Hugging Face DINOv3, intermediate states will be requested with `output_hidden_states=True`. Hidden-state numbering must be mapped carefully: Hugging Face `hidden_states[0]` is the embedding output, so transformer block `k` corresponds to `hidden_states[k + 1]`.

Every selected state must contain the same patch count. CLS and DINOv3 register tokens are removed before concatenation.

## Adapter interface

The adapter API will be generalized from one `dim` to explicit `input_dim` and `output_dim`.

Supported adapters:

```text
linear:
  Linear(input_dim, output_dim)

mlp:
  Linear(input_dim, hidden_dim)
  GELU
  Linear(hidden_dim, output_dim)
```

For the existing 1024→1024 residual MLP, the residual connection remains valid. For 4096→1024, there is no identity residual because input and output dimensions differ.

New multi-layer checkpoints use:

```text
input_dim  = 4096
output_dim = 1024
```

Old checkpoints containing only `dim` must continue to load as `input_dim=output_dim=dim`.

## Adapter training objective

DINOv3 and DINOv2 features are cached while both backbones are frozen.

For each corresponding patch:

```text
DINOv3 multi-layer feature [4096]
  -> adapter
  -> predicted DINOv2-like feature [1024]
```

The existing alignment objective is retained:

```text
MSE(predicted_dino2, target_dino2)
+ mean(1 - cosine_similarity(predicted_dino2, target_dino2))
```

The training cache therefore no longer requires DINOv2 and DINOv3 feature tensors to have identical final dimensions. It requires matching frame and patch dimensions only.

## Cache format

The manifest records:

```text
feature_mode: "multilayer"
dino3_layer_indices: [4, 11, 17, 23]
dino3_feature_dim: 4096
dino2_feature_dim: 1024
adapter_input_dim: 4096
adapter_output_dim: 1024
```

Cached tensors remain FP16 on disk.

## Inference modes

The evaluator keeps existing behavior and adds an explicit multi-layer path.

Recommended names:

```text
dinov2                 original VGGT baseline
dinov3-final           frozen final-layer DINOv3 substitution
dinov3-multilayer      four-layer DINOv3 + trained 4096->1024 adapter
```

For backward compatibility, the current `dinov3` spelling may continue to mean `dinov3-final`.

`dinov3-multilayer` requires an adapter checkpoint whose input/output dimensions are 4096/1024. Running the multi-layer path without such an adapter is an error rather than silently inserting a random projection.

## Preprocessing

DINOv2 remains at 518 target size with patch size 14.

DINOv3 remains at 592 target size with patch size 16:

```text
518 / 14 = 37
592 / 16 = 37
```

Both paths therefore produce a 37×37 = 1369 patch grid for square inputs.

VGGT continues to perform ImageNet normalization immediately before its configured patch-embedding module. Standalone adapter feature extraction applies the same normalization exactly once.

## Freezing behavior

For the new path:

- DINOv3: frozen, eval mode.
- VGGT aggregator: frozen, eval mode.
- CameraHead: frozen, eval mode.
- Adapter: trainable only during adapter training; frozen during evaluation.

This must be explicit in code and covered by tests.

## Backward compatibility

Existing final-layer zero-shot evaluation remains unchanged.

Existing 1024→1024 adapter checkpoints continue to load.

Existing cache manifests that use `feature_dim` remain readable for final-layer adapter training.

New multi-layer caches use explicit source/target dimensions.

## Tests

Tests must cover:

1. DINOv3 block indices are exactly `[4, 11, 17, 23]`.
2. Hugging Face hidden-state indexing selects transformer block `k` via state `k+1`.
3. CLS and all DINOv3 register tokens are removed from every selected layer.
4. Four `[B, P, 1024]` states concatenate to `[B, P, 4096]`.
5. Final-layer extraction remains `[B, P, 1024]`.
6. Multi-layer inference rejects a missing or dimensionally incompatible adapter.
7. Linear 4096→1024 adapter works.
8. MLP 4096→1024 adapter works without an invalid residual connection.
9. Legacy 1024→1024 adapter checkpoints still load.
10. Cache validation allows `[frames, patches, 4096]` DINOv3 paired with `[frames, patches, 1024]` DINOv2 when frame/patch axes match.
11. DINOv3/VGGT remain frozen in evaluation.
12. DINOv3 preprocessing remains 592/16 and DINOv2 remains 518/14.

## Documentation

README must clearly distinguish:

- the paper's actual end-to-end fine-tuned VGGT experiment;
- this repository's frozen-backbone approximation;
- the zero-shot final-layer substitution;
- the trained multi-layer feature projection.

The README must not describe the multi-layer adapter experiment as a reproduction of the paper's reported numbers.