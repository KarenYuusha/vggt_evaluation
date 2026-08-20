import gzip
import json
from pathlib import Path

import numpy as np

from evaluation.co3d import convert_pt3d_RT_to_opencv, load_co3d_scenes


def test_pt3d_to_opencv_conversion_matches_vggt_evaluator():
    rot = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    trans = [0, 0, 0]
    extri = convert_pt3d_RT_to_opencv(rot, trans)
    assert np.allclose(extri, np.eye(4)[:3])


def test_co3d_loader_uses_val_split_and_is_deterministic(tmp_path):
    category = tmp_path / "keyboard"
    seq = category / "seq1" / "images"
    set_lists = category / "set_lists"
    seq.mkdir(parents=True)
    set_lists.mkdir(parents=True)

    split_rows = []
    frames = []
    rot = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    for frame_number in range(1, 6):
        rel = f"keyboard/seq1/images/frame{frame_number:06d}.jpg"
        (tmp_path / rel).touch()
        split_rows.append(["seq1", frame_number, rel])
        frames.append({
            "sequence_name": "seq1",
            "frame_number": frame_number,
            "viewpoint": {"R": rot, "T": [frame_number, 0, 0]},
        })

    (set_lists / "set_lists_manyview_test_0.json").write_text(
        json.dumps({"train": [], "val": split_rows}), encoding="utf-8"
    )
    with gzip.open(category / "frame_annotations.jgz", "wt", encoding="utf-8") as f:
        json.dump(frames, f)

    first = load_co3d_scenes(tmp_path, num_frames=3, seed=7)
    second = load_co3d_scenes(tmp_path, num_frames=3, seed=7)

    assert len(first) == 1
    assert first[0].category == "keyboard"
    assert first[0].image_paths == second[0].image_paths
    assert len(first[0].image_paths) == 3
    assert all("frame" in Path(p).name for p in first[0].image_paths)
