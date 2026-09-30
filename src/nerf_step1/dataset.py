"""Read the original NeRF Blender scene layout without changing camera poses.

Expected layout: ``root/lego/transforms_{train,val,test}.json`` and the PNG
files referenced by each frame's ``file_path``. Images are loaded on demand
so the full-resolution scene does not need to fit in RAM.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image


class DatasetFormatError(ValueError):
    """The supplied scene differs from the supported Blender data format."""


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pixel dimensions and pinhole intrinsics in original image coordinates."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image width and height must be positive")
        if any(not math.isfinite(value) for value in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")

    def matrix(self, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """Return the 3×3 intrinsic matrix without a coordinate conversion."""
        return torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )


@dataclass(frozen=True)
class View:
    """One RGB image paired with its unchanged camera-to-world matrix."""

    image: torch.Tensor  # [height, width, 3], float32 in [0, 1], CPU
    camera_to_world: torch.Tensor  # [4, 4], float64, CPU
    intrinsics: CameraIntrinsics
    split: str
    index: int
    image_path: Path


@dataclass(frozen=True)
class BlenderSplit:
    """A split with lazy images and all poses in the metadata's frame order."""

    name: Literal["train", "val", "test"]
    image_paths: tuple[Path, ...]
    camera_to_world: torch.Tensor  # [N, 4, 4], float64, CPU
    intrinsics: CameraIntrinsics
    camera_angle_x: float
    background_color: tuple[float, float, float]

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_view(self, index: int) -> View:
        """Load a view as RGB; composite source RGBA over the configured color.

        The original Blender PNGs are RGBA. The RGB conversion is explicit:
        ``rgb = rgb * alpha + background * (1 - alpha)`` after division by 255,
        matching the original NeRF white-background branch when background is
        (1, 1, 1). This does not alter the stored camera matrix.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"{self.name} view index {index} is out of range")

        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            if image.mode not in ("RGB", "RGBA"):
                raise DatasetFormatError(
                    f"{image_path}: expected RGB or RGBA PNG, got {image.mode}"
                )
            pixels = np.array(image, dtype=np.float32, copy=True) / 255.0

        if pixels.shape[:2] != (self.intrinsics.height, self.intrinsics.width):
            raise DatasetFormatError(f"{image_path}: image dimensions changed after loading")
        if pixels.shape[2] == 4:
            alpha = pixels[..., 3:4]
            background = np.asarray(self.background_color, dtype=np.float32)
            rgb = pixels[..., :3] * alpha + background * (1.0 - alpha)
        else:
            rgb = pixels

        return View(
            image=torch.from_numpy(np.ascontiguousarray(rgb)),
            camera_to_world=self.camera_to_world[index],
            intrinsics=self.intrinsics,
            split=self.name,
            index=index,
            image_path=image_path,
        )


@dataclass(frozen=True)
class BlenderScene:
    """The separate train, validation, and test views of one Blender scene."""

    name: str
    directory: Path
    train: BlenderSplit
    val: BlenderSplit
    test: BlenderSplit


def _camera_pose(raw_matrix: object, context: str) -> torch.Tensor:
    try:
        pose = torch.tensor(raw_matrix, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise DatasetFormatError(f"{context}: transform_matrix must be numeric 4×4") from exc
    if pose.shape != (4, 4) or not bool(torch.isfinite(pose).all()):
        raise DatasetFormatError(f"{context}: transform_matrix must be finite 4×4")
    if not torch.allclose(
        pose[3], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-6
    ):
        raise DatasetFormatError(f"{context}: transform_matrix has an invalid last row")
    rotation = pose[:3, :3]
    if not torch.allclose(rotation.T @ rotation, torch.eye(3, dtype=torch.float64), atol=1e-3):
        raise DatasetFormatError(f"{context}: camera rotation is not orthonormal")
    if not math.isclose(float(torch.det(rotation)), 1.0, abs_tol=1e-3):
        raise DatasetFormatError(f"{context}: camera rotation must have determinant +1")
    return pose


def _image_path(scene_directory: Path, raw_path: object, context: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise DatasetFormatError(f"{context}: file_path must be a nonempty string")
    relative_path = Path(raw_path)
    if relative_path.suffix == "":
        relative_path = relative_path.with_suffix(".png")
    if relative_path.suffix.lower() != ".png":
        raise DatasetFormatError(f"{context}: only PNG images are supported")
    path = (scene_directory / relative_path).resolve()
    if not path.is_relative_to(scene_directory.resolve()):
        raise DatasetFormatError(f"{context}: image path escapes scene directory")
    if not path.is_file():
        raise DatasetFormatError(f"{context}: image not found: {path}")
    return path


def _load_split(
    scene_directory: Path,
    name: Literal["train", "val", "test"],
    background_color: tuple[float, float, float],
) -> BlenderSplit:
    metadata_path = scene_directory / f"transforms_{name}.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Blender metadata: {metadata_path}")
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetFormatError(f"Cannot read {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise DatasetFormatError(f"{metadata_path}: expected a JSON object")
    try:
        camera_angle_x = float(metadata["camera_angle_x"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetFormatError(f"{metadata_path}: missing/invalid camera_angle_x") from exc
    if not math.isfinite(camera_angle_x) or not 0.0 < camera_angle_x < math.pi:
        raise DatasetFormatError(f"{metadata_path}: camera_angle_x must be radians in (0, π)")
    frames = metadata.get("frames")
    if not isinstance(frames, list) or len(frames) == 0:
        raise DatasetFormatError(f"{metadata_path}: frames must be a nonempty list")

    paths: list[Path] = []
    poses: list[torch.Tensor] = []
    size: tuple[int, int] | None = None
    for index, frame in enumerate(frames):
        context = f"{metadata_path} frame {index}"
        if not isinstance(frame, dict):
            raise DatasetFormatError(f"{context}: expected a JSON object")
        path = _image_path(scene_directory, frame.get("file_path"), context)
        pose = _camera_pose(frame.get("transform_matrix"), context)
        try:
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode not in ("RGB", "RGBA"):
                    raise DatasetFormatError(f"{path}: expected RGB or RGBA PNG")
                current_size = image.size  # (width, height); no full image decode yet
        except OSError as exc:
            raise DatasetFormatError(f"Cannot open {path}: {exc}") from exc
        if size is not None and current_size != size:
            raise DatasetFormatError(f"{path}: image size {current_size} differs from {size}")
        size = current_size
        paths.append(path)
        poses.append(pose)

    assert size is not None  # frames is nonempty
    width, height = size
    focal = 0.5 * width / math.tan(0.5 * camera_angle_x)
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=focal,
        fy=focal,  # original Blender loader assumes square pixels
        cx=width / 2.0,
        cy=height / 2.0,
    )
    return BlenderSplit(
        name=name,
        image_paths=tuple(paths),
        camera_to_world=torch.stack(poses),
        intrinsics=intrinsics,
        camera_angle_x=camera_angle_x,
        background_color=background_color,
    )


def load_blender_scene(
    data_root: Path | str,
    scene: str,
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> BlenderScene:
    """Validate and expose one standard NeRF Blender scene and all three splits.

    ``data_root`` is the parent of the scene directory, for example
    ``data/nerf_synthetic``. Frames retain JSON order and original c2w values.
    There is no axis flip, camera recentering, resizing, or pose normalization.
    """
    if not scene or Path(scene).name != scene or scene in (".", ".."):
        raise ValueError("scene must be one directory name, such as 'lego'")
    if len(background_color) != 3 or any(
        not math.isfinite(float(channel)) or not 0.0 <= float(channel) <= 1.0
        for channel in background_color
    ):
        raise ValueError("background_color must contain three finite values in [0, 1]")
    color = tuple(float(channel) for channel in background_color)
    scene_directory = (Path(data_root) / scene).resolve()
    if not scene_directory.is_dir():
        raise FileNotFoundError(
            f"Blender scene not found: {scene_directory}. "
            "Place the official Lego data at data/nerf_synthetic/lego/."
        )

    train = _load_split(scene_directory, "train", color)
    val = _load_split(scene_directory, "val", color)
    test = _load_split(scene_directory, "test", color)
    for split in (val, test):
        if split.intrinsics != train.intrinsics or not math.isclose(
            split.camera_angle_x, train.camera_angle_x, rel_tol=1e-10
        ):
            raise DatasetFormatError(
                f"{split.name}: camera intrinsics differ from train; mixed intrinsics "
                "need an explicit per-view implementation"
            )
    return BlenderScene(scene, scene_directory, train, val, test)
