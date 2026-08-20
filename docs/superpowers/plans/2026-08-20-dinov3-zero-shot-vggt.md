# Zero-Shot DINOv3 VGGT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fully frozen DINOv3 ViT-L/16 backbone option to the existing VGGT camera-pose evaluator while preserving the DINOv2 baseline and all existing AUC/timing behavior.

**Architecture:** The evaluator still loads the pretrained VGGT model first. For `--backbone dinov3`, only `model.aggregator.patch_embed` is replaced by a wrapper around official DINOv3 ViT-L/16 final normalized patch tokens; the aggregator patch size becomes 16 and preprocessing becomes 592/16. DINOv2 keeps the current 518/14 path unchanged.

**Tech Stack:** Python, PyTorch, official DINOv3 PyTorch Hub interface, existing VGGT implementation, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-dinov3-zero-shot-vggt-design.md`

## Global Constraints

- No VGGT fine-tuning.
- No DINOv3 fine-tuning.
- No learned adapter or projection layer.
- DINOv3 backbone is ViT-L/16 and uses final-layer `x_norm_patchtokens` only.
- DINOv3 preprocessing uses target size 592 and patch size 16.
- DINOv2 preprocessing remains target size 518 and patch size 14.
- Dataset selection, pose metrics, AUC@30, and timing definitions remain unchanged.
- Do not store private DINOv3 weight URLs in the repository.

---

### Task 1: Parameterize VGGT image preprocessing

**Files:**
- Modify: `vggt/utils/load_fn.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Produces: `load_and_preprocess_images(image_path_list, mode="crop", target_size=518, patch_size=14)`.
- Preserves: all existing callers that omit the new arguments.

- [ ] **Step 1: Write failing preprocessing tests**

Create temporary 160x90 RGB images and assert that the default path produces a grid of `37 x 21` patches at patch size 14 while the DINOv3 path produces the same `37 x 21` grid at patch size 16. Also assert the default target width remains 518.

```python
from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images


def test_preprocess_preserves_patch_grid_between_dinov2_and_dinov3(tmp_path):
    path = tmp_path / "image.jpg"
    Image.new("RGB", (160, 90)).save(path)

    dino2 = load_and_preprocess_images([str(path)])
    dino3 = load_and_preprocess_images([str(path)], target_size=592, patch_size=16)

    assert dino2.shape[-1] // 14 == dino3.shape[-1] // 16 == 37
    assert dino2.shape[-2] // 14 == dino3.shape[-2] // 16 == 21
    assert dino2.shape[-1] == 518
    assert dino3.shape[-1] == 592
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest tests/test_preprocess.py -v`

Expected: FAIL because `load_and_preprocess_images` does not accept `target_size` or `patch_size`.

- [ ] **Step 3: Implement configurable target/patch sizes**

Change the function signature and replace hard-coded `518` and `/ 14` rounding with `target_size` and `patch_size`. Keep default values `518` and `14`.

- [ ] **Step 4: Run preprocessing tests**

Run: `pytest tests/test_preprocess.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vggt/utils/load_fn.py tests/test_preprocess.py
git commit -m "feat: parameterize VGGT image preprocessing"
```

### Task 2: Add the DINOv3 patch-token adapter

**Files:**
- Create: `evaluation/dinov3_backbone.py`
- Create: `tests/test_dinov3_backbone.py`

**Interfaces:**
- Produces: `DINOv3PatchEmbed(nn.Module)` whose `forward(images)` returns normalized patch tokens shaped `[B, P, 1024]`.
- Produces: `load_dinov3_vitl16(repo_dir, weights)` which invokes official `torch.hub.load` with model name `dinov3_vitl16`, `source="local"`, and the supplied weights URL/path.

- [ ] **Step 1: Write failing adapter tests**

Use a fake DINOv3 model whose `forward_features()` returns `{"x_norm_patchtokens": tensor}`. Assert the adapter returns that tensor unchanged and all wrapped parameters are frozen. Monkeypatch `torch.hub.load` and assert the loader passes the exact official hub arguments.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/test_dinov3_backbone.py -v`

Expected: FAIL because `evaluation.dinov3_backbone` does not exist.

- [ ] **Step 3: Implement adapter and official loader**

```python
class DINOv3PatchEmbed(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.eval()
        self.backbone.requires_grad_(False)

    def forward(self, images):
        features = self.backbone.forward_features(images)
        patch_tokens = features["x_norm_patchtokens"]
        if patch_tokens.ndim != 3 or patch_tokens.shape[-1] != 1024:
            raise ValueError(...)
        return patch_tokens
```

`load_dinov3_vitl16` validates that `repo_dir` exists and `weights` is non-empty before calling `torch.hub.load`.

- [ ] **Step 4: Run adapter tests**

Run: `pytest tests/test_dinov3_backbone.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/dinov3_backbone.py tests/test_dinov3_backbone.py
git commit -m "feat: add DINOv3 patch-token adapter"
```

### Task 3: Select the backbone in the model runner

**Files:**
- Modify: `evaluation/model_runner.py`
- Modify: `tests/test_model_runner.py`

**Interfaces:**
- `VGGTModelRunner.from_pretrained(model_source, device=None, backbone="dinov2", dinov3_repo=None, dinov3_weights=None)`.
- Runner exposes `backbone`, `input_target_size`, and `patch_size` metadata.
- DINOv3 mode replaces only `model.aggregator.patch_embed` and sets `model.aggregator.patch_size = 16`.

- [ ] **Step 1: Write failing runner tests**

Test a helper-level DINOv3 configuration path with fake VGGT and fake DINOv3 modules. Assert DINOv3 sets 592/16, replaces only patch embed, preserves camera head identity, and uses a preprocessing callable that passes `target_size=592, patch_size=16`. Assert DINOv2 remains 518/14.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/test_model_runner.py -v`

Expected: FAIL because the runner has no backbone selection.

- [ ] **Step 3: Implement runner backbone configuration**

Add a small configuration helper rather than duplicating inference logic. In DINOv3 mode load the official backbone, wrap it with `DINOv3PatchEmbed`, replace `aggregator.patch_embed`, set `aggregator.patch_size = 16`, and use `functools.partial(load_and_preprocess_images, target_size=592, patch_size=16)`. Freeze the complete final model with `model.requires_grad_(False)`.

- [ ] **Step 4: Run runner tests**

Run: `pytest tests/test_model_runner.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluation/model_runner.py tests/test_model_runner.py
git commit -m "feat: support DINOv3 backbone in VGGT runner"
```

### Task 4: Add CLI and result metadata

**Files:**
- Modify: `evaluate.py`
- Create: `tests/test_evaluate_cli.py`

**Interfaces:**
- CLI: `--backbone {dinov2,dinov3}` default `dinov2`.
- CLI: `--dinov3-repo PATH` and `--dinov3-weights VALUE` required only when `--backbone dinov3`.
- Saved result metadata: `backbone`, `input_target_size`, `patch_size`.

- [ ] **Step 1: Write failing CLI validation tests**

Parse explicit argument lists through a testable `parse_args(argv=None)` function. Assert DINOv2 needs no DINOv3 arguments, DINOv3 without either required value raises a parser error, and DINOv3 with both values parses successfully.

- [ ] **Step 2: Run focused CLI tests and verify failure**

Run: `pytest tests/test_evaluate_cli.py -v`

Expected: FAIL because these arguments and conditional validation do not exist.

- [ ] **Step 3: Implement CLI and metadata**

Pass backbone arguments into `VGGTModelRunner.from_pretrained`. Save runner metadata in the JSON result and print the selected backbone/input configuration before AUC/timing output.

- [ ] **Step 4: Run CLI tests**

Run: `pytest tests/test_evaluate_cli.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluate.py tests/test_evaluate_cli.py
git commit -m "feat: expose DINOv3 evaluation CLI"
```

### Task 5: Document execution and verify the complete repository

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents official DINOv3 local-repo + approved-weight URL/path workflow.
- Documents DINOv2 and DINOv3 evaluation commands for both datasets.
- Explicitly labels DINOv3 results as zero-shot backbone substitution rather than reproduction of the paper's trained DINOv3-VGGT experiment.

- [ ] **Step 1: Update README**

Include:

```powershell
git clone https://github.com/facebookresearch/dinov3.git
uv run python evaluate.py --dataset realestate10k --data-dir dataset\RealEstate10k --backbone dinov3 --dinov3-repo .\dinov3 --dinov3-weights "<URL_OR_PATH_FROM_META_EMAIL>"
```

and the equivalent CO3D command. Do not include an actual private weight URL.

- [ ] **Step 2: Run the full test suite**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 3: Compile Python files**

Run: `python -m compileall -q evaluation evaluate.py vggt/utils/load_fn.py`

Expected: exit code 0.

- [ ] **Step 4: Review branch diff**

Verify that dataset files and pose/AUC logic are unchanged and no private DINOv3 URL was committed.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md
git commit -m "docs: document zero-shot DINOv3 evaluation"
```
