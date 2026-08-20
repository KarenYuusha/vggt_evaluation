from pathlib import Path

import numpy as np

from .types import SceneSample


def _find_timestamp_image(image_dir, timestamp):
    preferred = image_dir / f"{timestamp}.jpg"
    if preferred.exists():
        return preferred
    matches = sorted(image_dir.glob(f"{timestamp}.*"))
    if not matches:
        raise FileNotFoundError(f"No image found for timestamp {timestamp} in {image_dir}")
    return matches[0]


def load_realestate10k_scenes(root, num_frames=10):
    root = Path(root)
    scenes = []

    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata_path = scene_dir / "selected_frames.txt"
        image_dir = scene_dir / "images"
        if not metadata_path.exists() or not image_dir.is_dir():
            continue

        lines = [line.strip() for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        frame_lines = lines[1:]
        if len(frame_lines) != num_frames:
            raise ValueError(
                f"{scene_dir.name}: expected {num_frames} frame rows in selected_frames.txt, got {len(frame_lines)}"
            )

        image_paths = []
        extrinsics = []
        for line in frame_lines:
            values = line.split()
            timestamp = values[0]
            pose_values = np.asarray(values[-12:], dtype=np.float64)
            if pose_values.size != 12:
                raise ValueError(f"{scene_dir.name}: malformed pose row for timestamp {timestamp}")
            image_paths.append(str(_find_timestamp_image(image_dir, timestamp)))
            extrinsics.append(pose_values.reshape(3, 4))

        scenes.append(SceneSample(
            name=scene_dir.name,
            image_paths=image_paths,
            gt_extrinsics=np.stack(extrinsics),
        ))

    return scenes
