from collections import defaultdict
import time

import numpy as np

from .pose_metrics import calculate_auc_np, relative_pose_errors


def evaluate_scenes(runner, scenes, dataset_name):
    if not scenes:
        raise ValueError("No scenes to evaluate")

    runner.warmup(scenes[0].image_paths)
    evaluation_start = time.perf_counter()

    all_r_error = []
    all_t_error = []
    category_errors = defaultdict(lambda: {"r": [], "t": []})
    scene_results = []

    for scene in scenes:
        scene_start = time.perf_counter()
        prediction = runner.predict(scene.image_paths)
        r_error, t_error = relative_pose_errors(prediction.extrinsics, scene.gt_extrinsics)
        auc30, _ = calculate_auc_np(r_error, t_error, max_threshold=30)
        total_scene_ms = (time.perf_counter() - scene_start) * 1000.0

        all_r_error.extend(r_error.tolist())
        all_t_error.extend(t_error.tolist())
        if scene.category:
            category_errors[scene.category]["r"].extend(r_error.tolist())
            category_errors[scene.category]["t"].extend(t_error.tolist())

        scene_result = {
            "scene": scene.name,
            "images": [path.split("/")[-1].split("\\")[-1] for path in scene.image_paths],
            "auc30": auc30,
            "mean_rotation_error": float(np.mean(r_error)),
            "mean_translation_error": float(np.mean(t_error)),
            "preprocess_ms": prediction.preprocess_ms,
            "inference_ms": prediction.inference_ms,
            "total_scene_ms": total_scene_ms,
        }
        if scene.category:
            scene_result["category"] = scene.category
        scene_results.append(scene_result)

    category_auc30 = {}
    if category_errors:
        for category, errors in sorted(category_errors.items()):
            auc30, _ = calculate_auc_np(errors["r"], errors["t"], max_threshold=30)
            category_auc30[category] = auc30
        dataset_auc30 = float(np.mean(list(category_auc30.values())))
    else:
        dataset_auc30, _ = calculate_auc_np(all_r_error, all_t_error, max_threshold=30)

    inference_times = np.asarray([item["inference_ms"] for item in scene_results], dtype=np.float64)
    preprocess_times = np.asarray([item["preprocess_ms"] for item in scene_results], dtype=np.float64)
    total_scene_times = np.asarray([item["total_scene_ms"] for item in scene_results], dtype=np.float64)

    result = {
        "dataset": dataset_name,
        "num_scenes": len(scene_results),
        "auc30": dataset_auc30,
        "avg_inference_ms": float(np.mean(inference_times)),
        "std_inference_ms": float(np.std(inference_times)),
        "avg_preprocess_ms": float(np.mean(preprocess_times)),
        "avg_total_scene_ms": float(np.mean(total_scene_times)),
        "total_evaluation_s": time.perf_counter() - evaluation_start,
        "scenes": scene_results,
    }
    if category_auc30:
        result["category_auc30"] = category_auc30
    return result
