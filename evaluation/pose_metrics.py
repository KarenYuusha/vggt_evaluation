from itertools import combinations

import numpy as np


def to_homogeneous(extrinsics):
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected (N, 3, 4) or (N, 4, 4), got {extrinsics.shape}")
    if extrinsics.shape[1:] == (4, 4):
        return extrinsics

    bottom = np.zeros((len(extrinsics), 1, 4), dtype=extrinsics.dtype)
    bottom[:, 0, 3] = 1.0
    return np.concatenate([extrinsics, bottom], axis=1)


def rotation_angle_deg(rot_gt, rot_pred):
    delta = rot_gt @ np.swapaxes(rot_pred, -1, -2)
    cos_angle = (np.trace(delta, axis1=1, axis2=2) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def translation_angle_deg(t_gt, t_pred, eps=1e-15):
    gt_norm = np.linalg.norm(t_gt, axis=1, keepdims=True)
    pred_norm = np.linalg.norm(t_pred, axis=1, keepdims=True)
    gt_unit = np.divide(t_gt, gt_norm, out=np.zeros_like(t_gt), where=gt_norm > eps)
    pred_unit = np.divide(t_pred, pred_norm, out=np.zeros_like(t_pred), where=pred_norm > eps)
    cosine = np.sum(gt_unit * pred_unit, axis=1)
    return np.degrees(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0)))


def relative_pose_errors(pred_extrinsics, gt_extrinsics):
    pred = to_homogeneous(pred_extrinsics)
    gt = to_homogeneous(gt_extrinsics)
    if len(pred) != len(gt):
        raise ValueError("Predicted and ground-truth poses must have the same length")
    if len(pred) < 2:
        raise ValueError("At least two poses are required")

    pairs = list(combinations(range(len(pred)), 2))
    pred_rel = np.stack([pred[i] @ np.linalg.inv(pred[j]) for i, j in pairs])
    gt_rel = np.stack([gt[i] @ np.linalg.inv(gt[j]) for i, j in pairs])

    r_error = rotation_angle_deg(gt_rel[:, :3, :3], pred_rel[:, :3, :3])
    t_error = translation_angle_deg(gt_rel[:, :3, 3], pred_rel[:, :3, 3])
    return r_error, t_error


def calculate_auc_np(r_error, t_error, max_threshold=30):
    r_error = np.asarray(r_error, dtype=np.float64)
    t_error = np.asarray(t_error, dtype=np.float64)
    if len(r_error) == 0 or len(r_error) != len(t_error):
        raise ValueError("Rotation and translation errors must be non-empty and equally sized")
    if not np.isfinite(r_error).all() or not np.isfinite(t_error).all():
        raise ValueError("Rotation and translation errors must be finite")

    max_errors = np.maximum(r_error, t_error)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    normalized_histogram = histogram.astype(float) / float(len(max_errors))
    return float(np.mean(np.cumsum(normalized_histogram))), normalized_histogram
