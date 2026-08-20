# VGGT Pose Evaluation Design

## Goal

Evaluate the existing VGGT camera-pose model on the repository's CO3Dv2 single-sequence sample and RealEstate10K sample with AUC@30 and reproducible runtime measurements, while isolating the model runner so a future DINOv3-backed VGGT can reuse the same datasets and metrics.

## Architecture

Dataset adapters emit a common `SceneSample` containing ordered image paths and OpenCV world-to-camera 3x4 ground-truth extrinsics. CO3D reads the many-view `val` split, samples 10 frames deterministically, and converts PyTorch3D cameras using Meta's public conversion. RealEstate10K reads the exact timestamp rows already stored in `selected_frames.txt` and uses the final 12 values as the world-to-camera pose.

The VGGT runner executes only `aggregator -> camera_head -> pose_encoding_to_extri_intri`. Model loading and warm-up are excluded from timing. Preprocessing is timed separately; GPU inference uses CUDA events and synchronization.

A shared metric module forms all pairwise relative poses, computes rotation and translation-direction angular errors, uses `max(r_error, t_error)` per pair, and reproduces the public evaluator's histogram AUC@30 definition. CO3D reports per-category AUC@30 and the mean across categories; RealEstate10K reports pooled AUC@30 across scenes.

## Outputs

JSON results contain dataset/model metadata, AUC@30, average/std inference time, average preprocessing and total-scene time, and per-scene metrics. The evaluator is feed-forward only; no bundle adjustment is included.
