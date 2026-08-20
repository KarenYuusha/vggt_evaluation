import gzip
import json
from pathlib import Path

import numpy as np

from evaluation.co3d import convert_pt3d_RT_to_opencv, load_co3d_scenes


def _write_category(root, category_name, split_files, sequences, frames):
    category = root / category_name
    (category / "set_lists").mkdir(parents=True)

    for filename, content in split_files.items():
        (category / "set_lists" / filename).write_text(json.dumps(content), encoding="utf-8")

    with gzip.open(category / "sequence_annotations.jgz", "wt", encoding="utf-8") as f:
        json.dump(sequences, f)
    with gzip.open(category / "frame_annotations.jgz", "wt", encoding="utf-8") as f:
        json.dump(frames, f)


def _make_frames(root, category, seq_name, count, bad_frame=None):
    rows = []
    frames = []
    rot = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]

    for frame_number in range(1, count + 1):
        rel = f"{category}/{seq_name}/images/frame{frame_number:06d}.jpg"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        rows.append([seq_name, frame_number, rel])
        trans = [float(frame_number), 0.0, 0.0]
        if frame_number == bad_frame:
            trans[0] = float("nan")
        frames.append({
            "sequence_name": seq_name,
            "frame_number": frame_number,
            "viewpoint": {"R": rot, "T": trans},
        })

    return rows, frames


def test_pt3d_to_opencv_conversion_matches_vggt_evaluator():
    rot = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    trans = [0, 0, 0]
    extri = convert_pt3d_RT_to_opencv(rot, trans)
    assert np.allclose(extri, np.eye(4)[:3])


def test_co3d_loader_prefers_dev_and_filters_invalid_quality(tmp_path):
    test_rows, test_frames = _make_frames(tmp_path, "mouse", "bad_test", 5)
    dev_rows, dev_frames = _make_frames(tmp_path, "mouse", "good_dev", 5)
    split_files = {
        "set_lists_manyview_test_0.json": {"train": [], "val": test_rows},
        "set_lists_manyview_dev_0.json": {"train": [], "val": dev_rows},
    }
    sequences = [
        {"sequence_name": "bad_test", "viewpoint_quality_score": float("nan")},
        {"sequence_name": "good_dev", "viewpoint_quality_score": 1.5},
    ]
    _write_category(tmp_path, "mouse", split_files, sequences, test_frames + dev_frames)

    first = load_co3d_scenes(tmp_path, num_frames=3, seed=7)
    second = load_co3d_scenes(tmp_path, num_frames=3, seed=7)

    assert len(first) == 1
    assert first[0].name == "mouse/good_dev"
    assert first[0].image_paths == second[0].image_paths
    assert len(first[0].image_paths) == 3


def test_co3d_loader_prefers_fewview_dev_when_available(tmp_path):
    many_rows, many_frames = _make_frames(tmp_path, "mouse", "many_dev", 5)
    few_rows, few_frames = _make_frames(tmp_path, "mouse", "few_dev", 5)
    split_files = {
        "set_lists_manyview_dev_0.json": {"train": [], "val": many_rows},
        "set_lists_fewview_dev.json": {"train": [], "test": few_rows},
    }
    sequences = [
        {"sequence_name": "many_dev", "viewpoint_quality_score": 1.0},
        {"sequence_name": "few_dev", "viewpoint_quality_score": 1.0},
    ]
    _write_category(tmp_path, "mouse", split_files, sequences, many_frames + few_frames)

    scenes = load_co3d_scenes(tmp_path, num_frames=3, seed=0)

    assert [scene.name for scene in scenes] == ["mouse/few_dev"]


def test_co3d_loader_drops_nonfinite_frames_before_sampling(tmp_path):
    rows, frames = _make_frames(tmp_path, "mouse", "good_dev", 5, bad_frame=2)
    split_files = {"set_lists_manyview_dev_0.json": {"train": [], "val": rows}}
    sequences = [{"sequence_name": "good_dev", "viewpoint_quality_score": 1.0}]
    _write_category(tmp_path, "mouse", split_files, sequences, frames)

    scenes = load_co3d_scenes(tmp_path, num_frames=4, seed=0)

    assert len(scenes) == 1
    assert np.isfinite(scenes[0].gt_extrinsics).all()
    assert not any(Path(path).name == "frame000002.jpg" for path in scenes[0].image_paths)
