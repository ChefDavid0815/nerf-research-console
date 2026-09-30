"""Saved visual checks made only from loaded, real Blender scene views."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a desktop GUI.
import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import BlenderScene, BlenderSplit
from .rays import generate_rays


def _sample_indices(count: int, wanted: int) -> list[int]:
    return np.linspace(0, count - 1, num=min(count, wanted), dtype=int).tolist()


def _equal_3d_axes(axis: plt.Axes, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float((maximum - minimum).max()) / 2.0, 1.0) * 1.15
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def save_camera_ray_plot(
    scene: BlenderScene, output_path: Path, *, camera_count: int = 6
) -> list[float]:
    """Plot real train cameras, their forward axes, and five rays each.

    Returns the cosine of each center ray with the vector toward world origin.
    The origin is a *reference point* for Blender's object-centered scenes;
    positive cosines show orientation, not proof of pixel/image alignment.
    """
    split = scene.train
    indices = _sample_indices(len(split), camera_count)
    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, len(indices)))
    all_points = [np.zeros(3)]
    cosines: list[float] = []
    center_x = split.intrinsics.width // 2
    center_y = split.intrinsics.height // 2
    step_x = max(1, split.intrinsics.width // 8)
    step_y = max(1, split.intrinsics.height // 8)
    pixels = torch.tensor(
        [
            [center_x, center_y],
            [max(0, center_x - step_x), center_y],
            [min(split.intrinsics.width - 1, center_x + step_x), center_y],
            [center_x, max(0, center_y - step_y)],
            [center_x, min(split.intrinsics.height - 1, center_y + step_y)],
        ],
        dtype=torch.int64,
    )

    for color, index in zip(colors, indices):
        pose = split.camera_to_world[index]
        origins, directions = generate_rays(
            pose, split.intrinsics, pixels, normalize_directions=True
        )
        origin = origins[0].numpy()
        rays = directions.numpy()
        toward_origin = -origin
        distance = float(np.linalg.norm(toward_origin))
        if distance == 0.0:
            raise ValueError(f"train camera {index} is at the reference origin")
        cosine = float(np.dot(rays[0], toward_origin / distance))
        cosines.append(cosine)
        all_points.append(origin)
        axis.scatter(*origin, color=color, s=50)
        axis.text(*(origin + 0.07), f"train {index}", fontsize=8)
        axis.quiver(*origin, *(rays[0] * 0.85), color=color, linewidth=2.1)
        for direction in rays:
            end = origin + direction * 1.35
            axis.plot([origin[0], end[0]], [origin[1], end[1]],
                      [origin[2], end[2]], color=color, alpha=0.43, linewidth=1)
            all_points.append(end)

    axis.scatter(0, 0, 0, marker="*", color="black", s=180, label="world origin")
    axis.set(xlabel="world X", ylabel="world Y", zlabel="world Z")
    axis.set_title(f"{scene.name}: train camera positions, forward axes and rays")
    axis.legend(loc="upper left")
    _equal_3d_axes(axis, np.stack(all_points))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return cosines


def save_view_contact_sheet(scene: BlenderScene, output_path: Path) -> None:
    """Show actual RGB views with split/index, camera position and forward axis."""
    selected: list[tuple[BlenderSplit, int]] = []
    for split in (scene.train, scene.val, scene.test):
        selected.extend((split, index) for index in _sample_indices(len(split), 2))

    columns = 3
    rows = (len(selected) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(13, 4.6 * rows), squeeze=False)
    for axis, (split, index) in zip(axes.flat, selected):
        view = split.get_view(index)
        position = view.camera_to_world[:3, 3]
        forward = -view.camera_to_world[:3, 2]
        axis.imshow(view.image.numpy())
        axis.set_title(
            f"{split.name} [{index}]  {view.intrinsics.width}×{view.intrinsics.height}\n"
            f"position=({position[0]:+.2f}, {position[1]:+.2f}, {position[2]:+.2f})  "
            f"forward=({forward[0]:+.2f}, {forward[1]:+.2f}, {forward[2]:+.2f})",
            fontsize=9,
        )
        axis.axis("off")
    for axis in list(axes.flat)[len(selected):]:
        axis.axis("off")
    fig.suptitle(f"{scene.name}: source views and unmodified camera poses", fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
