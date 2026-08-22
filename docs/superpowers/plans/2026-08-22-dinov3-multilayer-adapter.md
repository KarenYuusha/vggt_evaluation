# DINOv3 Multi-Layer Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-aligned four-intermediate-layer DINOv3 feature path with a learned 4096→1024 adapter while keeping DINOv3 and VGGT frozen.

**Architecture:** DINOv3 ViT-L/16 extracts transformer blocks `[4, 11, 17, 23]`. Each Hugging Face intermediate state is passed through the DINOv3 model norm to match official `get_intermediate_layers(..., norm=True)` semantics, then CLS/register tokens are removed and the four 1024-D patch features are concatenated to 4096 dimensions. A trained adapter maps those features to the 1024-dimensional interface expected by the released pretrained VGGT. Existing DINOv2 and final-layer DINOv3 modes remain available.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, pytest, VGGT.

**Spec:** `docs/superpowers/specs/2026-08-22-dinov3-multilayer-adapter-design.md`

## Global Constraints

- DINOv3 model is `facebook/dinov3-vitl16-pretrain-lvd1689m`.
- DINOv3 input target is 592 with patch size 16.
- DINOv2 input target is 518 with patch size 14.
- DINOv3 multi-layer block indices are exactly `[4, 11, 17, 23]`.
- Selected DINOv3 intermediate states are normalized individually before special-token removal and concatenation.
- DINOv3 and VGGT remain frozen; only the adapter is trained.
- Multi-layer DINOv3 features are 4096-dimensional and VGGT-facing features are 1024-dimensional.
- Legacy final-layer and 1024→1024 adapter checkpoints remain compatible.

---

### Task 1: Multi-layer DINOv3 feature extraction

**Files:**
- Modify: `evaluation/dinov3_backbone.py`
- Modify: `adapter/features.py`
- Test: `tests/test_dinov3_multilayer.py`
- Preserve: `tests/test_dinov3_backbone.py`, `tests/test_adapter_features.py`

**Interfaces:**
- Produces: `DINOV3_MULTILAYER_INDICES = (4, 11, 17, 23)`.
- Produces: a shared helper that returns DINOv3 patch tokens in either `final` or `multilayer` mode.
- `multilayer` returns `[B, P, 4096]`; `final` returns `[B, P, 1024]`.
- Hugging Face block `k` is read from `hidden_states[k + 1]`, normalized with `backbone.norm`, then stripped of CLS/register tokens.

- [x] Write tests proving hidden-state block indexing, per-layer normalization, special-token removal, and 4096-D concatenation.
- [x] Run the focused tests and confirm RED because multi-layer extraction is missing.
- [x] Implement the shared extraction helper and wire both runtime and adapter feature extraction to it.
- [x] Run focused tests and confirm GREEN.

### Task 2: Generalize adapters to input/output dimensions

**Files:**
- Modify: `adapter/model.py`
- Test: `tests/test_multilayer_adapter.py`
- Preserve: `tests/test_adapter_model.py`

**Interfaces:**
- `FeatureAdapter` supports explicit `input_dim` and `output_dim` while retaining the legacy `dim` form.
- `ResidualMLPAdapter` supports explicit `input_dim` and `output_dim`; residual addition is used only when dimensions match.
- Legacy checkpoints with `dim` load as equal input/output dimensions.

- [x] Write failing tests for 4096→1024 linear/MLP adapters and legacy checkpoint loading.
- [x] Run focused tests and confirm RED.
- [x] Generalize adapter modules and checkpoint metadata; only use residual addition when input and output dimensions match.
- [x] Run focused tests and confirm GREEN.

### Task 3: Allow asymmetric cached feature dimensions

**Files:**
- Modify: `adapter/data.py`
- Modify: `adapter/training.py`
- Modify: `extract_adapter_features.py`
- Modify: `train_adapter.py`
- Test: `tests/test_multilayer_cache_manifest.py`
- Test: `tests/test_multilayer_adapter.py`
- Preserve existing adapter/cache tests.

**Interfaces:**
- Cache validation requires matching frame/patch axes, not identical final dimensions.
- New manifests expose `dino3_feature_dim=4096`, `dino2_feature_dim=1024`, `feature_mode="multilayer"`, and selected layer indices.
- Training constructs adapter input/output dimensions from the manifest.

- [x] Write failing tests for asymmetric `[F,P,4096]` / `[F,P,1024]` caches and manifest-driven adapter construction.
- [x] Run focused tests and confirm RED.
- [x] Implement asymmetric cache validation, multi-layer feature caching, manifest fields, and dimension-aware training.
- [x] Run focused tests and confirm GREEN.

### Task 4: Add paper-aligned evaluation mode

**Files:**
- Modify: `evaluation/model_runner.py`
- Modify: `evaluate.py`
- Test: `tests/test_multilayer_backbone_modes.py`
- Test: `tests/test_multilayer_evaluate_cli.py`
- Preserve: `tests/test_model_runner.py`, `tests/test_evaluate_cli.py`

**Interfaces:**
- Existing `dinov3` remains an alias for final-layer DINOv3.
- Add explicit `dinov3-final` and `dinov3-multilayer` modes.
- `dinov3-multilayer` requires a 4096→1024 adapter checkpoint.

- [x] Write failing tests for mode selection and rejection of missing/incompatible multi-layer adapters.
- [x] Run focused tests and confirm RED.
- [x] Wire multi-layer extraction and dimension validation into `configure_backbone`; update CLI choices and result metadata.
- [x] Run focused tests and confirm GREEN.

### Task 5: Documentation and regression verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-22-dinov3-multilayer-adapter-design.md`

- [x] Document the exact distinction between the paper's fine-tuned pipeline, zero-shot final-layer substitution, and frozen multi-layer learned projection.
- [x] Document per-layer DINOv3 normalization and the reason it is required with Hugging Face hidden states.
- [x] Document cache regeneration and training/evaluation commands for the new 4096→1024 adapter.
- [x] Run focused and backward-compatibility pytest coverage in the available local reconstruction.
- [ ] Run the repository's complete pytest suite in an environment that can check out this GitHub branch.
- [ ] Inspect the final GitHub branch diff and integration choice.
