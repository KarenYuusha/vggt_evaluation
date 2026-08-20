from pathlib import Path

import numpy as np

from evaluation.realestate10k import load_realestate10k_scenes


def test_realestate_loader_matches_timestamped_images_and_reads_w2c_pose(tmp_path):
    scene_dir = tmp_path / "scene_a"
    image_dir = scene_dir / "images"
    image_dir.mkdir(parents=True)
    for ts in (1000000, 2000000):
        (image_dir / f"{ts}.jpg").touch()

    identity_3x4 = "1 0 0 0 0 1 0 0 0 0 1 0"
    meta = "\n".join([
        "https://youtube.example/video",
        f"1000000 0 0 0 0 0 0 {identity_3x4}",
        f"2000000 0 0 0 0 0 0 {identity_3x4}",
    ])
    (scene_dir / "selected_frames.txt").write_text(meta, encoding="utf-8")

    scenes = load_realestate10k_scenes(tmp_path, num_frames=2)

    assert len(scenes) == 1
    scene = scenes[0]
    assert scene.name == "scene_a"
    assert [Path(p).name for p in scene.image_paths] == ["1000000.jpg", "2000000.jpg"]
    assert scene.gt_extrinsics.shape == (2, 3, 4)
    assert np.allclose(scene.gt_extrinsics[0], np.eye(4)[:3])
