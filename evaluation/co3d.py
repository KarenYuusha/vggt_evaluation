from collections import defaultdict
from pathlib import Path
import gzip
import json
import random

import numpy as np

from .types import SceneSample


def convert_pt3d_RT_to_opencv(rot, trans):
    rot_pt3d = np.asarray(rot, dtype=np.float64).copy()
    trans_pt3d = np.asarray(trans, dtype=np.float64).copy()
    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    return np.hstack((rot_pt3d, trans_pt3d[:, None]))


def _load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _find_dev_splits(category_dir):
    set_lists_dir = category_dir / "set_lists"
    fewview_dev = set_lists_dir / "set_lists_fewview_dev.json"
    if fewview_dev.exists():
        return [(fewview_dev, "test")]

    return [(path, "val") for path in sorted(set_lists_dir.glob("set_lists_manyview_dev_*.json"))]


def _good_quality_sequences(sequence_annotations, min_quality):
    good_sequences = set()

    for item in sequence_annotations:
        score = item.get("viewpoint_quality_score")
        if score is None or not np.isfinite(score) or score <= min_quality:
            continue
        good_sequences.add(item["sequence_name"])

    return good_sequences


def load_co3d_scenes(root, num_frames=10, seed=0, min_quality=0.5):
    root = Path(root)
    scenes = []

    for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        frame_annotation_path = category_dir / "frame_annotations.jgz"
        sequence_annotation_path = category_dir / "sequence_annotations.jgz"
        dev_splits = _find_dev_splits(category_dir)
        if not dev_splits or not frame_annotation_path.exists() or not sequence_annotation_path.exists():
            continue

        sequence_annotations = _load_json_gz(sequence_annotation_path)
        good_sequences = _good_quality_sequences(sequence_annotations, min_quality)
        if not good_sequences:
            continue

        grouped_rows = defaultdict(dict)
        for set_list_path, split_name in dev_splits:
            set_lists = json.loads(set_list_path.read_text(encoding="utf-8"))
            rows = set_lists.get(split_name, [])
            if not rows and split_name == "val":
                rows = set_lists.get("test", [])

            for seq_name, frame_number, filepath in rows:
                if seq_name in good_sequences:
                    grouped_rows[seq_name][int(frame_number)] = filepath

        frame_annotations = _load_json_gz(frame_annotation_path)
        frame_map = {
            (item["sequence_name"], int(item["frame_number"])): item
            for item in frame_annotations
        }

        for seq_name, row_map in sorted(grouped_rows.items()):
            valid_frames = []

            for frame_number, filepath in sorted(row_map.items()):
                annotation = frame_map.get((seq_name, frame_number))
                if annotation is None or annotation.get("viewpoint") is None:
                    continue

                viewpoint = annotation["viewpoint"]
                rot = np.asarray(viewpoint.get("R"), dtype=np.float64)
                trans = np.asarray(viewpoint.get("T"), dtype=np.float64)
                if rot.shape != (3, 3) or trans.shape != (3,) or not np.isfinite(rot).all() or not np.isfinite(trans).all():
                    continue

                image_path = root / filepath
                if not image_path.exists():
                    continue

                extrinsic = convert_pt3d_RT_to_opencv(rot, trans)
                valid_frames.append((frame_number, str(image_path), extrinsic))

            if len(valid_frames) < num_frames:
                continue

            rng = random.Random(f"{seed}:{category_dir.name}:{seq_name}")
            selected = rng.sample(valid_frames, num_frames)
            selected.sort(key=lambda item: item[0])

            scenes.append(SceneSample(
                name=f"{category_dir.name}/{seq_name}",
                image_paths=[item[1] for item in selected],
                gt_extrinsics=np.stack([item[2] for item in selected]),
                category=category_dir.name,
            ))

    return scenes
