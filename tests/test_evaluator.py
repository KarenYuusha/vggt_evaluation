import numpy as np

from evaluation.evaluator import evaluate_scenes
from evaluation.model_runner import Prediction
from evaluation.types import SceneSample


class PerfectRunner:
    def warmup(self, image_paths):
        pass

    def predict(self, image_paths):
        n = len(image_paths)
        extri = np.repeat(np.eye(4, dtype=np.float64)[None, :3], n, axis=0)
        for i in range(n):
            extri[i, 0, 3] = i
        return Prediction(extrinsics=extri, preprocess_ms=2.0, inference_ms=5.0)


def test_evaluator_reports_auc_and_average_timing():
    gt = np.repeat(np.eye(4, dtype=np.float64)[None, :3], 3, axis=0)
    for i in range(3):
        gt[i, 0, 3] = i
    scene = SceneSample(name="s1", image_paths=["1", "2", "3"], gt_extrinsics=gt)

    result = evaluate_scenes(PerfectRunner(), [scene], dataset_name="RealEstate10k")

    assert result["auc30"] == 1.0
    assert result["avg_inference_ms"] == 5.0
    assert result["avg_preprocess_ms"] == 2.0
    assert result["num_scenes"] == 1
