# DINOv3-to-DINOv2 Feature Alignment Adapter

## Goal

Train a lightweight feature adapter that maps frozen DINOv3 ViT-L/16 patch features into the DINOv2 patch-feature space expected by the pretrained VGGT transformer. DINOv2, DINOv3, the VGGT aggregator, and the VGGT camera head remain frozen. Only the adapter is trainable.

This is a proof-of-concept using a local `RealEstate10k_finetune/` dataset containing 100 clips with 20 frames per clip (2,000 images total). The existing `dataset/RealEstate10k/` evaluation subset is never used for adapter training or validation.

## Data split

The 100 finetuning clips are sorted by clip directory name, shuffled with a deterministic seed, and split at the clip level:

- 90 clips for training.
- 10 clips for validation.

All 20 frames from one clip stay in the same split. This prevents near-duplicate frames from one trajectory appearing in both train and validation.

The final camera-pose evaluation continues to use the separate existing RealEstate10K evaluation subset.

## Phase 1: Offline feature extraction

A feature-extraction script accepts `--data-dir RealEstate10k_finetune` and an output directory. It validates that each selected clip contains an `images/` directory with 20 readable frames.

For every image, two frozen feature tensors are produced from the same source frame:

1. **DINOv2 teacher feature**
   - Load `facebook/VGGT-1B`.
   - Reuse the exact pretrained `model.aggregator.patch_embed` DINOv2 ViT-L/14 from VGGT rather than loading a separate DINOv2 checkpoint.
   - Use VGGT preprocessing at target size 518 and patch size 14.
   - Apply the same ImageNet normalization performed in `Aggregator.forward` before calling the patch encoder.
   - Save final normalized patch tokens only.

2. **DINOv3 student-side feature**
   - Load `facebook/dinov3-vitl16-pretrain-lvd1689m` through Hugging Face Transformers.
   - Use target size 592 and patch size 16.
   - Apply the same ImageNet normalization before the DINOv3 model.
   - Remove the CLS token and all DINOv3 register tokens from `last_hidden_state`.
   - Save final normalized patch tokens only.

Because `518 / 14 = 592 / 16 = 37`, both preprocessing paths produce the same patch-grid width. The existing preprocessing rounds the patch-grid height from the source aspect ratio using the same 37-wide grid, so corresponding DINOv2 and DINOv3 patch positions must have identical patch counts. The extractor fails immediately if the two feature tensors for an image do not have the same `[P, 1024]` shape.

Features are converted to FP16 before being written to disk. Each clip is stored as one `.pt` file containing:

- `dino2`: `[20, P, 1024]` FP16 tensor.
- `dino3`: `[20, P, 1024]` FP16 tensor.
- `images`: ordered source image names.
- `clip`: clip identifier.

If a clip cannot be stacked because source images unexpectedly produce different patch counts within that clip, extraction fails with a clear error rather than silently padding or truncating features.

The output directory also contains a JSON manifest recording the deterministic split, seed, source clip names, number of images, feature shapes, model identifiers, and preprocessing sizes. Cached feature tensors and trained checkpoints are generated artifacts and are not committed to Git.

## Phase 2: Adapter training

The adapter is exactly:

```python
nn.Linear(1024, 1024)
```

Its weight is initialized to the identity matrix and its bias to zero, so training starts from the current direct DINOv3 substitution behavior.

No backbone or VGGT parameter participates in this phase; DINOv2 and DINOv3 do not need to be loaded at all. Training reads only the cached feature files.

For each training clip, `[20, P, 1024]` features are flattened into patch pairs `[20 * P, 1024]`. Patch pairs are shuffled and optimized in mini-batches so memory use stays low.

The training objective combines feature-value reconstruction and directional alignment:

```text
loss = mse_loss + cosine_loss
cosine_loss = mean(1 - cosine_similarity(predicted_dino2, target_dino2))
```

Initial training defaults:

- optimizer: AdamW.
- learning rate: `1e-3`.
- weight decay: `1e-4`.
- epochs: `20`.
- patch batch size: `4096`.
- deterministic seed: `0`.

Validation runs after every epoch over all validation clip feature pairs without updating weights. The checkpoint with the lowest mean validation loss is saved as `best_adapter.pt`; the final epoch checkpoint may also be saved separately for diagnostics.

Each checkpoint stores the adapter state dict plus metadata including input/output dimension, model identifiers, training seed, epoch, train loss, validation loss, and loss weights.

## Phase 3: VGGT integration and evaluation

The existing DINOv3 patch wrapper gains optional adapter support. When an adapter checkpoint is supplied, the path becomes:

```text
image
  -> DINOv3 ViT-L/16 (frozen)
  -> final patch tokens [P, 1024]
  -> Linear(1024, 1024) adapter (frozen at evaluation)
  -> pretrained VGGT frame/global transformer (frozen)
  -> pretrained CameraHead (frozen)
  -> camera pose
```

The evaluation CLI gains `--adapter-checkpoint PATH`. It is valid only with `--backbone dinov3`. Without this flag, DINOv3 behavior remains the existing zero-shot direct substitution, so current experiments remain reproducible.

The intended comparison on the unchanged RealEstate10K evaluation subset is:

1. Original pretrained VGGT with DINOv2.
2. Pretrained VGGT with direct frozen DINOv3 substitution.
3. Pretrained VGGT with frozen DINOv3 plus the trained feature adapter.

## Files and interfaces

Expected new modules/scripts:

- `adapter/__init__.py`
- `adapter/model.py` — adapter definition, identity initialization, checkpoint loading.
- `adapter/features.py` — DINOv2/DINOv3 patch feature extraction helpers and cache validation.
- `extract_adapter_features.py` — deterministic 90/10 clip split and offline cache generation.
- `train_adapter.py` — cached-feature training and validation loop.
- adapter-specific tests under `tests/`.

Expected existing files to change:

- `evaluation/dinov3_backbone.py` — optionally apply a loaded adapter to DINOv3 patch tokens.
- `evaluation/model_runner.py` — pass optional adapter checkpoint into DINOv3 configuration.
- `evaluate.py` — add `--adapter-checkpoint` and validate its use.
- `.gitignore` — ignore generated feature caches/checkpoints.
- `README.md` — document extraction, training, and evaluation commands.

## Testing

Tests must cover:

- deterministic clip-level 90/10 split with no overlap.
- exact DINOv2/DINOv3 patch-shape validation.
- identity adapter initialization.
- only adapter parameters are trainable in the training module.
- training loss decreases on a small synthetic linear-mapping problem.
- best-validation checkpoint selection.
- DINOv3 wrapper applies the adapter when configured and preserves direct behavior when not configured.
- CLI rejects `--adapter-checkpoint` with DINOv2 and accepts it with DINOv3.
- existing DINOv2, DINOv3 direct-swap, dataset, metric, and preprocessing tests remain green.

## Success criteria

The implementation is complete when:

1. Feature extraction can process the local 100-clip/20-frame dataset without loading any evaluation clips.
2. The generated manifest contains exactly 90 training clips and 10 validation clips with no overlap.
3. Every cached DINOv2/DINOv3 pair has matching `[20, P, 1024]` shapes.
4. Adapter training can run using only cached features and produces a best-validation checkpoint.
5. The trained adapter can be loaded by `evaluate.py --backbone dinov3 --adapter-checkpoint ...` without modifying VGGT weights.
6. The existing direct DINOv3 and DINOv2 evaluation modes continue to work unchanged.
