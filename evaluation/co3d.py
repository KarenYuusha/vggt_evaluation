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


def load_co3d_scenes(root, num_frames=10, seed=0, split="val"):
    root = Path(root)
    scenes = []

    for category_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        set_list_path = category_dir / "set_lists" / "set_lists_manyview_test_0.json"
        frame_annotation_path = category_dir / "frame_annotations.jgz"
        if not set_list_path.exists() or not frame_annotation_path.exists():
            continue

        set_lists = json.loads(set_list_path.read_text(encoding="utf-8"))
        rows = set_lists.get(split, [])
        grouped_rows = defaultdict(list)
        for seq_name, frame_number, filepath in rows:
            grouped_rows[seq_name].append((int(frame_number), filepath))

        frame_annotations = _load_json_gz(frame_annotation_path)
        frame_map = {
            (item["sequence_name"], int(item["frame_number"])): item
            for item in frame_annotations
        }

        for seq_name, seq_rows in sorted(grouped_rows.items()):
            if len(seq_rows) < num_frames:
                continue

            rng = random.Random(f"{seed}:{category_dir.name}:{seq_name}")
            selected = rng.sample(seq_rows, num_frames)
            selected.sort(key=lambda item: item[0])

            image_paths = []
            extrinsics = []
            for frame_number, filepath in selected:
                annotation = frame_map[(seq_name, frame_number)]
                viewpoint = annotation["viewpoint"]
                image_paths.append(str(root / filepath))
                extrinsics.append(convert_pt3d_RT_to_opencv(viewpoint["R"], viewpoint["T"]))

            scenes.append(SceneSample(
                name=f"{category_dir.name}/{seq_name}",
                image_paths=image_paths,
                gt_extrinsics=np.stack(extrinsics),
                category=category_dir.name,
            ))

    return scenes
