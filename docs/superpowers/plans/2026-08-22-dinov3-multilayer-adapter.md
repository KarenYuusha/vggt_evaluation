# DINOv3 Multi-Layer Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-aligned four-intermediate-layer DINOv3 feature path with a learned 4096→1024 adapter while keeping DINOv3 and VGGT frozen.

**Architecture:** DINOv3 ViT-L/16 extracts transformer blocks `[4, 11, 17, 23]`; patch tokens are stripped of CLS/register tokens and concatenated to 4096 dimensions. A trained adapter maps those features to the 1024-dimensional feature interface expected by the released pretrained VGGT. Existing DINOv2 and final-layer DINOv3 modes remain available.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, pytest, VGGT.

**Spec:** `docs/superpowers/specs/2026-08-22-dinov3-multilayer-adapter-design.md`

## Global Constraints

- DINOv3 model is `facebook/dinov3-vitl16-pretrain-lvd1689m`.
- DINOv3 input target is 592 with patch size 16.
- DINOv2 input target is 518 with patch size 14.
- DINOv3 multi-layer block indices are exactly `[4, 11, 17, 23]`.
- DINOv3 and VGGT remain frozen; only the adapter is trained.
- Multi-layer DINOv3 features are 4096-dimensional and VGGT-facing features are 1024-dimensional.
- Legacy final-layer and 1024→1024 adapter checkpoints remain compatible.

---

### Task 1: Multi-layer DINOv3 feature extraction

**Files:**
- Modify: `evaluation/dinov3_backbone.py`
- Modify: `adapter/features.py`
- Test: `tests/test_dinov3_backbone.py`
- Test: `tests/test_adapter_features.py`

**Interfaces:**
- Produces: `DINOV3_MULTILAYER_INDICES = (4, 11, 17, 23)`.
- Produces: a shared helper that returns DINOv3 patch tokens in either `final` or `multilayer` mode.
- `multilayer` returns `[B, P, 4096]`; `final` returns `[B, P, 1024]`.

- [ ] Write tests proving hidden-state block indexing, special-token removal, and 4096-D concatenation.
- [ ] Run the focused tests and confirm RED because multi-layer extraction is missing.
- [ ] Implement the minimal shared extraction helper and wire both runtime and adapter feature extraction to it.
- [ ] Run focused tests and confirm GREEN.

### Task 2: Generalize adapters to input/output dimensions

**Files:**
- Modify: `adapter/model.py`
- Test: `tests/test_adapter_model.py`

**Interfaces:**
- `FeatureAdapter(input_dim=1024, output_dim=1024)`.
- `ResidualMLPAdapter(input_dim=1024, output_dim=1024, hidden_dim=2048)`.
- Legacy checkpoints with `dim` load as equal input/output dimensions.

- [ ] Write failing tests for 4096→1024 linear/MLP adapters and legacy checkpoint loading.
- [ ] Run focused tests and confirm RED.
- [ ] Generalize adapter modules and checkpoint metadata; only use residual addition when input and output dimensions match.
- [ ] Run focused tests and confirm GREEN.

### Task 3: Allow asymmetric cached feature dimensions

**Files:**
- Modify: `adapter/data.py`
- Modify: `adapter/training.py`
- Modify: `extract_adapter_features.py`
- Modify: `train_adapter.py`
- Test: `tests/test_adapter_data.py`
- Test: `tests/test_adapter_training.py`
- Test: `tests/test_extract_adapter_features.py`
- Test: `tests/test_train_adapter_cli.py`

**Interfaces:**
- Cache validation requires matching frame/patch axes, not identical final dimensions.
- New manifests expose `dino3_feature_dim=4096`, `dino2_feature_dim=1024`, `feature_mode="multilayer"`, and selected layer indices.
- Training constructs adapter input/output dimensions from the manifest.

- [ ] Write failing tests for asymmetric `[F,P,4096]` / `[F,P,1024]` caches and manifest-driven adapter construction.
- [ ] Run focused tests and confirm RED.
- [ ] Implement asymmetric cache validation, multi-layer feature caching, manifest fields, and dimension-aware training.
- [ ] Run focused tests and confirm GREEN.

### Task 4: Add paper-aligned evaluation mode

**Files:**
- Modify: `evaluation/model_runner.py`
- Modify: `evaluate.py`
- Test: `tests/test_model_runner.py`
- Test: `tests/test_evaluate_cli.py`

**Interfaces:**
- Existing `dinov3` remains an alias for final-layer DINOv3.
- Add explicit `dinov3-final` and `dinov3-multilayer` modes.
- `dinov3-multilayer` requires a 4096→1024 adapter checkpoint.

- [ ] Write failing tests for mode selection and rejection of missing/incompatible multi-layer adapters.
- [ ] Run focused tests and confirm RED.
- [ ] Wire multi-layer extraction and dimension validation into `configure_backbone`; update CLI choices and result metadata.
- [ ] Run focused tests and confirm GREEN.

### Task 5: Documentation and full regression verification

**Files:**
- Modify: `README.md`
- Modify tests only if a documentation-linked command needs correction.

- [ ] Document the exact distinction between the paper's fine-tuned pipeline, zero-shot final-layer substitution, and frozen multi-layer learned projection.
- [ ] Document cache regeneration and training/evaluation commands for the new 4096→1024 adapter.
- [ ] Run the full pytest suite.
- [ ] Inspect the branch diff for accidental unrelated changes.
- [ ] Verify legacy DINOv2/final-layer behavior remains covered by tests.