from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images


def test_preprocess_preserves_patch_grid_between_dinov2_and_dinov3(tmp_path):
    path = tmp_path / "image.jpg"
    Image.new("RGB", (160, 90)).save(path)

    dino2 = load_and_preprocess_images([str(path)])
    dino3 = load_and_preprocess_images([str(path)], target_size=592, patch_size=16)

    assert dino2.shape[-1] // 14 == dino3.shape[-1] // 16 == 37
    assert dino2.shape[-2] // 14 == dino3.shape[-2] // 16 == 21
    assert dino2.shape[-1] == 518
    assert dino3.shape[-1] == 592
