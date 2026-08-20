# VGGT Pose Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable VGGT camera-pose evaluator for the repository's CO3Dv2 and RealEstate10K samples with AUC@30 and runtime reporting.

**Architecture:** Dataset-specific adapters produce a shared scene structure. A model runner performs only the VGGT camera path, and a shared evaluator computes pairwise pose errors/AUC and timing summaries.

**Tech Stack:** Python, NumPy, PyTorch, public `facebookresearch/vggt` package, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-vggt-pose-evaluation-design.md`

## Global Constraints

- Feed-forward camera pose only; no bundle adjustment.
- Default to 10 frames and seed 0.
- CO3D uses `val` from `set_lists_manyview_test_0.json` and PyTorch3D-to-OpenCV conversion.
- RealEstate10K uses exact timestamped images and the stored 3x4 world-to-camera poses.
- Model loading and warm-up are excluded from inference timing.
- Keep the model runner independent from dataset and metric code for a future DINOv3 swap.

---

### Task 1: Pose metrics and shared scene type

**Files:** create `evaluation/types.py`, `evaluation/pose_metrics.py`; test `tests/test_pose_metrics.py`.

- [x] Write a failing perfect-pose AUC test.
- [x] Implement homogeneous pose conversion, pairwise relative pose errors, and public-style AUC@30.
- [x] Run the test suite and verify the metric test passes.

### Task 2: Dataset adapters

**Files:** create `evaluation/co3d.py`, `evaluation/realestate10k.py`; tests `tests/test_co3d.py`, `tests/test_realestate10k.py`.

- [x] Write failing loader/convention tests with synthetic dataset fixtures.
- [x] Implement deterministic CO3D `val` sampling and Meta's pose conversion.
- [x] Implement RealEstate10K timestamp/image matching and 3x4 pose parsing.
- [x] Run the test suite and verify loader tests pass.

### Task 3: Camera-only VGGT runner

**Files:** create `evaluation/model_runner.py`; test `tests/test_model_runner.py`.

- [x] Write a failing test using an injected fake aggregator/camera head/decoder.
- [x] Implement pretrained/local checkpoint loading and camera-only inference.
- [x] Use mixed precision for the aggregator, disable autocast for the camera head, and use CUDA events for GPU inference timing.
- [x] Run the test suite and verify runner tests pass.

### Task 4: Evaluation/reporting CLI

**Files:** create `evaluation/evaluator.py`, `evaluate.py`; modify `README.md`; test `tests/test_evaluator.py`.

- [x] Write a failing end-to-end evaluator test with a perfect fake runner.
- [x] Implement warm-up, per-scene metrics, dataset aggregation, timing summaries, JSON output, and CLI options.
- [x] Document Windows/uv commands for both datasets.
- [x] Run full verification (`python -m pytest -q`, `python -m compileall evaluation evaluate.py`).
