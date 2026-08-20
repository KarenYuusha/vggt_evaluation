import numpy as np
import pytest

from evaluation.pose_metrics import calculate_auc_np, relative_pose_errors


def make_pose(tx=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = tx
    return pose


def test_perfect_predictions_have_zero_error_and_auc_one():
    gt = np.stack([make_pose(0.0), make_pose(1.0), make_pose(2.0)])
    r_error, t_error = relative_pose_errors(gt, gt)

    assert np.allclose(r_error, 0.0, atol=1e-6)
    assert np.allclose(t_error, 0.0, atol=1e-6)
    auc30, _ = calculate_auc_np(r_error, t_error, max_threshold=30)
    assert auc30 == 1.0


def test_auc_rejects_nonfinite_errors():
    with pytest.raises(ValueError, match="finite"):
        calculate_auc_np(np.array([0.0, np.nan]), np.array([0.0, 1.0]))

    with pytest.raises(ValueError, match="finite"):
        calculate_auc_np(np.array([0.0, 1.0]), np.array([0.0, np.inf]))
