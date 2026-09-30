"""Small synthetic fixtures for checking the Blender data format loader.

These four PNGs are test inputs only. They are not research observations and
must never be used to report reconstruction accuracy.
"""

import json
import math

import pytest
import torch
from PIL import Image

from nerf_step1.dataset import DatasetFormatError, load_blender_scene


IDENTITY_POSE = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

TRANSLATED_POSE = [
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 2.0],
    [0.0, 0.0, 1.0, 3.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _write_split(scene_root, split, poses, camera_angle_x=math.pi / 3):
    image_dir = scene_root / split
    image_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, pose in enumerate(poses):
        # Standard Blender NeRF image paths omit the .png suffix.
        relative_path = f"./{split}/r_{index}"
        image = Image.new("RGBA", (4, 4), (64, 128, 192, 255))
        if split == "train" and index == 0:
            image.putpixel((0, 0), (255, 0, 0, 128))
        image.save(image_dir / f"r_{index}.png")
        frames.append({"file_path": relative_path, "transform_matrix": pose})
    (scene_root / f"transforms_{split}.json").write_text(
        json.dumps({"camera_angle_x": camera_angle_x, "frames": frames}),
        encoding="utf-8",
    )


@pytest.fixture
def tiny_blender_root(tmp_path):
    scene_root = tmp_path / "lego"
    scene_root.mkdir()
    _write_split(scene_root, "train", [IDENTITY_POSE, TRANSLATED_POSE])
    _write_split(scene_root, "val", [IDENTITY_POSE])
    _write_split(scene_root, "test", [TRANSLATED_POSE])
    return tmp_path


def _rewrite_frame_pose(scene_root, split, pose):
    metadata_path = scene_root / f"transforms_{split}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["frames"][0]["transform_matrix"] = pose
    # 1e400 is a valid JSON number that overflows to infinity on float decode.
    # It exercises finite-value validation rather than malformed-JSON handling.
    metadata_path.write_text(
        json.dumps(metadata).replace("Infinity", "1e400"), encoding="utf-8"
    )


def test_splits_images_and_camera_counts(tiny_blender_root):
    scene = load_blender_scene(tiny_blender_root, "lego")
    assert (len(scene.train), len(scene.val), len(scene.test)) == (2, 1, 1)

    for split in (scene.train, scene.val, scene.test):
        assert split.camera_to_world.shape == (len(split), 4, 4)
        assert split.camera_to_world.dtype == torch.float64
        for index in range(len(split)):
            view = split.get_view(index)
            assert view.image.shape == (4, 4, 3)
            assert view.image.dtype == torch.float32
            torch.testing.assert_close(
                view.camera_to_world, split.camera_to_world[index]
            )


def test_intrinsics_and_original_poses_are_preserved(tiny_blender_root):
    scene = load_blender_scene(tiny_blender_root, "lego")
    intrinsics = scene.train.intrinsics
    assert (intrinsics.width, intrinsics.height) == (4, 4)
    assert intrinsics.fx == pytest.approx(2.0 / math.tan(math.pi / 6))
    assert intrinsics.fy == pytest.approx(intrinsics.fx)
    assert intrinsics.cx == pytest.approx(2.0)
    assert intrinsics.cy == pytest.approx(2.0)
    torch.testing.assert_close(
        scene.train.camera_to_world[1],
        torch.tensor(TRANSLATED_POSE, dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_rgba_composites_against_white_background(tiny_blender_root):
    scene = load_blender_scene(tiny_blender_root, "lego")
    image = scene.train.get_view(0).image
    alpha = 128.0 / 255.0
    torch.testing.assert_close(
        image[0, 0],
        torch.tensor([1.0, 1.0 - alpha, 1.0 - alpha]),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        image[0, 1],
        torch.tensor([64.0, 128.0, 192.0]) / 255.0,
        rtol=0,
        atol=1e-6,
    )


def test_inconsistent_intrinsics_across_splits_are_rejected(tiny_blender_root):
    scene_root = tiny_blender_root / "lego"
    val_path = scene_root / "transforms_val.json"
    metadata = json.loads(val_path.read_text(encoding="utf-8"))
    metadata["camera_angle_x"] = math.pi / 4
    val_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(DatasetFormatError, match="[Ii]ntrinsics"):
        load_blender_scene(tiny_blender_root, "lego")


def test_missing_frame_image_is_reported(tiny_blender_root):
    (tiny_blender_root / "lego" / "train" / "r_1.png").unlink()
    with pytest.raises((DatasetFormatError, FileNotFoundError)):
        load_blender_scene(tiny_blender_root, "lego")


@pytest.mark.parametrize(
    "invalid_pose",
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [row[:] for row in IDENTITY_POSE[:3]],
        [
            [1.0, 0.0, 0.0, float("inf")],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ],
    ids=["two_by_two", "three_by_four", "non_finite"],
)
def test_invalid_camera_matrices_are_rejected(tiny_blender_root, invalid_pose):
    _rewrite_frame_pose(tiny_blender_root / "lego", "train", invalid_pose)
    with pytest.raises(DatasetFormatError):
        load_blender_scene(tiny_blender_root, "lego")
