# DINOv3 Adapter Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline feature-alignment pipeline that trains only a `Linear(1024, 1024)` adapter from frozen DINOv3 patch features to the frozen DINOv2 feature space expected by pretrained VGGT, then optionally applies that adapter during DINOv3 camera-pose evaluation.

**Architecture:** Phase 1 deterministically splits `RealEstate10k_finetune/` at clip level, verifies no clip overlaps with the existing evaluation set, extracts frozen DINOv2/DINOv3 patch features, and caches them as FP16 tensors. Phase 2 trains only a linear adapter from cached tensors using MSE plus cosine loss and selects the checkpoint with the best validation loss. Phase 3 loads that checkpoint into the existing DINOv3 patch wrapper before frozen VGGT attention and CameraHead.

**Tech Stack:** Python 3.10+, PyTorch 2.3.1, torchvision 0.18.1, Transformers >=4.56,<5, NumPy, pytest, Hugging Face gated `facebook/dinov3-vitl16-pretrain-lvd1689m`, pretrained `facebook/VGGT-1B`.

**Spec:** `docs/superpowers/specs/2026-08-21-dinov3-adapter-alignment-design.md`

## Global Constraints

- Finetune source contains exactly 100 clips with 20 frames per clip for the proof-of-concept.
- Split clips deterministically into 90 train / 10 validation clips with seed 0; frames from one clip never cross splits.
- Reject clip-ID overlap between `RealEstate10k_finetune/` and `dataset/RealEstate10k/`.
- DINOv2, DINOv3, VGGT attention blocks, and CameraHead remain frozen; only the adapter is trainable.
- DINOv2 teacher features come from `facebook/VGGT-1B`'s own `aggregator.patch_embed`.
- DINOv2 preprocessing is target 518 / patch 14; DINOv3 preprocessing is target 592 / patch 16.
- Cached DINOv2 and DINOv3 features must match exactly in `[frames, patches, 1024]` shape.
- Cache features in FP16; train adapter in FP32 by default.
- Adapter is exactly `nn.Linear(1024, 1024)`, identity-initialized with zero bias.
- Default training objective is `MSE + mean(1 - cosine_similarity)`.
- Default optimizer is AdamW with lr `1e-3`, weight decay `1e-4`, 20 epochs, patch batch size 4096, seed 0.
- Existing DINOv2 and direct DINOv3 evaluation behavior must remain unchanged when no adapter checkpoint is supplied.

---

### Task 1: Adapter model and checkpoint contract

**Files:**
- Create: `adapter/__init__.py`
- Create: `adapter/model.py`
- Create: `tests/test_adapter_model.py`

**Interfaces:**
- Produces: `FeatureAdapter(nn.Module)` with `forward(features: Tensor) -> Tensor`.
- Produces: `build_identity_adapter(dim: int = 1024) -> FeatureAdapter`.
- Produces: `save_adapter_checkpoint(path, adapter, metadata)` and `load_adapter_checkpoint(path, map_location="cpu") -> tuple[FeatureAdapter, dict]`.

- [ ] **Step 1: Write failing tests** for exact identity behavior, parameter count/shape, checkpoint round-trip, and metadata preservation.

```python

def test_identity_adapter_preserves_features():
    adapter = build_identity_adapter()
    x = torch.randn(3, 7, 1024)
    assert torch.allclose(adapter(x), x)


def test_adapter_checkpoint_round_trip(tmp_path):
    adapter = build_identity_adapter()
    path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(path, adapter, {"epoch": 4, "val_loss": 0.25})
    loaded, metadata = load_adapter_checkpoint(path)
    assert torch.allclose(loaded.linear.weight, adapter.linear.weight)
    assert metadata["epoch"] == 4
```

- [ ] **Step 2: Run** `pytest tests/test_adapter_model.py -v` and verify it fails because `adapter.model` does not exist.
- [ ] **Step 3: Implement minimal adapter/checkpoint code**. `FeatureAdapter` owns only `self.linear = nn.Linear(dim, dim)`. Identity initialization uses `torch.nn.init.eye_` and zeros the bias. Checkpoint stores `state_dict`, `dim`, and a plain metadata dictionary.
- [ ] **Step 4: Run** `pytest tests/test_adapter_model.py -v`; expected all PASS.
- [ ] **Step 5: Commit** with `feat: add feature alignment adapter`.

### Task 2: Dataset split, leakage guard, and cache validation

**Files:**
- Create: `adapter/data.py`
- Create: `tests/test_adapter_data.py`

**Interfaces:**
- Produces: `list_clip_dirs(root: Path, expected_frames: int = 20) -> list[Path]`.
- Produces: `split_clips(clips: list[Path], train_count: int = 90, seed: int = 0) -> tuple[list[Path], list[Path]]`.
- Produces: `assert_no_clip_overlap(finetune_root: Path, evaluation_root: Path) -> None`.
- Produces: `validate_feature_pair(dino2: Tensor, dino3: Tensor, expected_dim: int = 1024) -> None`.

- [ ] **Step 1: Write failing tests** that create 100 fake clip directories with `images/` and 20 fake `.jpg` files, assert deterministic 90/10 split, no overlap, and failure for an overlapping clip ID or mismatched feature shapes.
- [ ] **Step 2: Run** `pytest tests/test_adapter_data.py -v`; expected import failure.
- [ ] **Step 3: Implement minimal data helpers**. Sort by clip name before seeded shuffle. `list_clip_dirs` rejects malformed clips rather than silently skipping them. `assert_no_clip_overlap` compares directory names only.
- [ ] **Step 4: Run** `pytest tests/test_adapter_data.py -v`; expected PASS.
- [ ] **Step 5: Commit** with `feat: add adapter dataset validation`.

### Task 3: Frozen DINOv2/DINOv3 feature extraction helpers

**Files:**
- Create: `adapter/features.py`
- Create: `tests/test_adapter_features.py`

**Interfaces:**
- Produces: `normalize_imagenet(images: Tensor) -> Tensor` for `[N,3,H,W]` tensors in `[0,1]`.
- Produces: `extract_dinov2_patch_tokens(vggt_model, images: Tensor) -> Tensor`.
- Produces: `extract_dinov3_patch_tokens(dinov3_model, images: Tensor) -> Tensor`.
- Produces: `extract_feature_pair(vggt_model, dinov3_model, dino2_images, dino3_images) -> tuple[Tensor, Tensor]`.

- [ ] **Step 1: Write failing tests** using fake backbones. Verify ImageNet normalization, DINOv2 dictionary output handling, DINOv3 CLS/register stripping, no gradients, and exact shape validation.

```python

def test_dinov3_extractor_strips_cls_and_register_tokens():
    model = FakeHFBackbone(num_register_tokens=4, patch_count=6)
    patches = extract_dinov3_patch_tokens(model, torch.zeros(2, 3, 32, 48))
    assert patches.shape == (2, 6, 1024)
```

- [ ] **Step 2: Run** `pytest tests/test_adapter_features.py -v`; expected import failure.
- [ ] **Step 3: Implement helpers** under `torch.inference_mode()`. Both helpers accept already resized `[0,1]` images, normalize once with ImageNet mean/std, and return final normalized patch tokens. DINOv2 handles either direct tensor or `{"x_norm_patchtokens": ...}` output. DINOv3 slices `last_hidden_state[:, 1 + config.num_register_tokens:]`.
- [ ] **Step 4: Run** `pytest tests/test_adapter_features.py -v`; expected PASS.
- [ ] **Step 5: Commit** with `feat: add frozen feature extractors`.

### Task 4: Offline cache extraction CLI

**Files:**
- Create: `extract_adapter_features.py`
- Create: `tests/test_extract_adapter_features.py`
- Modify: `.gitignore`

**Interfaces:**
- CLI: `uv run python extract_adapter_features.py --data-dir RealEstate10k_finetune --evaluation-dir dataset\RealEstate10k --output-dir adapter_data --seed 0`.
- Produces: `adapter_data/manifest.json`, `adapter_data/train/<clip>.pt`, `adapter_data/val/<clip>.pt`.
- Cache record: `{clip: str, images: list[str], dino2: Tensor, dino3: Tensor}`.

- [ ] **Step 1: Write failing tests** around pure orchestration helpers: CLI defaults, manifest split counts, cache record dtype/shape, and rejection of evaluation overlap. Inject fake model loaders/extractors so tests never download models.
- [ ] **Step 2: Run** `pytest tests/test_extract_adapter_features.py -v`; expected failure because script/helpers do not exist.
- [ ] **Step 3: Implement CLI and extraction flow**. Load VGGT once, keep only its DINOv2 patch encoder, load Hugging Face DINOv3 once, freeze/eval both, process one clip at a time, resize through existing `load_and_preprocess_images` separately at 518/14 and 592/16, validate `[20,P,1024]`, save CPU FP16 tensors, and write the manifest atomically after successful extraction.
- [ ] **Step 4: Add `.gitignore` entries** for `adapter_data/`, `adapter_checkpoints/`, and `*.adapter.pt`.
- [ ] **Step 5: Run** `pytest tests/test_extract_adapter_features.py -v` plus `pytest tests/test_adapter_data.py tests/test_adapter_features.py -v`; expected PASS.
- [ ] **Step 6: Commit** with `feat: add offline adapter feature cache`.

### Task 5: Cached-feature adapter training

**Files:**
- Create: `adapter/training.py`
- Create: `train_adapter.py`
- Create: `tests/test_adapter_training.py`

**Interfaces:**
- Produces: `alignment_loss(pred: Tensor, target: Tensor, mse_weight: float = 1.0, cosine_weight: float = 1.0) -> Tensor`.
- Produces: `iter_patch_batches(cache_files, batch_size, seed, shuffle) -> Iterator[tuple[Tensor, Tensor]]`.
- Produces: `train_epoch(adapter, cache_files, optimizer, device, batch_size, seed) -> float`.
- Produces: `validate_epoch(adapter, cache_files, device, batch_size) -> float`.
- CLI: `uv run python train_adapter.py --cache-dir adapter_data --output-dir adapter_checkpoints --epochs 20 --batch-size 4096 --lr 1e-3 --weight-decay 1e-4 --seed 0`.

- [ ] **Step 1: Write failing tests** for loss value/gradient, patch batching, only adapter parameters receiving gradients, synthetic recoverable mapping where training loss decreases, and best-validation checkpoint selection.
- [ ] **Step 2: Run** `pytest tests/test_adapter_training.py -v`; expected import failure.
- [ ] **Step 3: Implement training helpers**. Load one cache file at a time, convert FP16 features to FP32, flatten `[F,P,D] -> [F*P,D]`, shuffle patch indices deterministically for training, and avoid loading DINO models.
- [ ] **Step 4: Implement CLI** with AdamW, identity initialization, per-epoch train/validation reporting, `best_adapter.pt` on strictly lower validation loss, and `last_adapter.pt` after final epoch. Checkpoint metadata includes epoch, train/val loss, loss weights, optimizer settings, model IDs, split seed, and cache manifest path.
- [ ] **Step 5: Run** `pytest tests/test_adapter_training.py -v`; expected PASS.
- [ ] **Step 6: Commit** with `feat: train cached feature adapter`.

### Task 6: Apply adapter in DINOv3 VGGT evaluation

**Files:**
- Modify: `evaluation/dinov3_backbone.py`
- Modify: `evaluation/model_runner.py`
- Modify: `evaluate.py`
- Modify: `tests/test_dinov3_backbone.py`
- Modify: `tests/test_model_runner.py`
- Modify: `tests/test_evaluate_cli.py`

**Interfaces:**
- `DINOv3PatchEmbed(backbone, adapter=None)` optionally applies the adapter after patch extraction.
- `configure_backbone(..., adapter_checkpoint=None)` loads and freezes the adapter only for DINOv3.
- `VGGTModelRunner.from_pretrained(..., adapter_checkpoint=None)` forwards the option.
- CLI: `--adapter-checkpoint PATH`, valid only with `--backbone dinov3`.

- [ ] **Step 1: Write failing tests** verifying direct DINOv3 output is unchanged without adapter, adapter transforms tokens when present, adapter parameters are frozen at evaluation, CLI rejects DINOv2+adapter, and runner wires the checkpoint into DINOv3 only.
- [ ] **Step 2: Run focused tests** and verify expected failures.
- [ ] **Step 3: Implement minimal integration** using `load_adapter_checkpoint`. Preserve 592/16 preprocessing and all existing direct-swap defaults.
- [ ] **Step 4: Run** `pytest tests/test_dinov3_backbone.py tests/test_model_runner.py tests/test_evaluate_cli.py -v`; expected PASS.
- [ ] **Step 5: Commit** with `feat: support aligned DINOv3 evaluation`.

### Task 7: Dependencies, documentation, and full regression verification

**Files:**
- Modify: `README.md`
- Modify if needed: `requirements.txt`
- Modify if needed: `pyproject.toml`

**Interfaces:**
- Documents the exact three-stage workflow and comparison commands.

- [ ] **Step 1: Update README** with commands:

```powershell
uv run python extract_adapter_features.py --data-dir RealEstate10k_finetune --evaluation-dir dataset\RealEstate10k --output-dir adapter_data --seed 0
uv run python train_adapter.py --cache-dir adapter_data --output-dir adapter_checkpoints --epochs 20 --batch-size 4096
uv run python evaluate.py --dataset realestate10k --data-dir dataset\RealEstate10k --backbone dinov3 --adapter-checkpoint adapter_checkpoints\best_adapter.pt --output results\vggt_dinov3_adapter_realestate10k.json
```

Also document baseline DINOv2 and direct DINOv3 commands and explicitly state that only the adapter was trained.

- [ ] **Step 2: Ensure dependency declarations include Transformers >=4.56,<5 and pytest where project convention expects them.** Do not change PyTorch/CUDA pins.
- [ ] **Step 3: Run full suite** with `pytest -q`; expected all tests PASS.
- [ ] **Step 4: Run syntax verification** with `python -m compileall adapter extract_adapter_features.py train_adapter.py evaluation evaluate.py`; expected success.
- [ ] **Step 5: Compare branch against `main`** and confirm no dataset files, pose metrics, CO3D loader, or RealEstate10K evaluator logic changed unexpectedly.
- [ ] **Step 6: Commit** with `docs: document adapter alignment workflow`.

### Task 8: Completion review

**Files:** none unless verification finds a defect.

- [ ] **Step 1: Re-read the approved spec** and map every success criterion to verified implementation/test evidence.
- [ ] **Step 2: Verify generated data/checkpoint directories are ignored and no model weights/features are committed.**
- [ ] **Step 3: Record the exact branch head and full-test result.**
- [ ] **Step 4: Leave the feature branch ready for the user's local real-model extraction/training run; do not claim the 2,000-image extraction itself was executed unless it was run against the user's local `RealEstate10k_finetune/` directory and gated model access.
