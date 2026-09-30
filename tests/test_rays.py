"""Checks for camera geometry using the original NeRF OpenGL convention."""

import pytest
import torch

from nerf_step1.dataset import CameraIntrinsics
from nerf_step1.rays import generate_rays


@pytest.fixture
def intrinsics():
    return CameraIntrinsics(width=4, height=4, fx=2.0, fy=2.0, cx=2.0, cy=2.0)


def test_ray_shapes_and_pixel_axis_convention(intrinsics):
    pose = torch.eye(4, dtype=torch.float64)
    pixels_xy = torch.tensor([[2, 2], [3, 1], [0, 0]], dtype=torch.int64)
    origins, directions = generate_rays(pose, intrinsics, pixels_xy)

    assert origins.shape == directions.shape == (3, 3)
    torch.testing.assert_close(origins, torch.zeros_like(origins))
    torch.testing.assert_close(
        directions,
        torch.tensor(
            [[0.0, 0.0, -1.0], [0.5, 0.5, -1.0], [-1.0, 1.0, -1.0]],
            dtype=directions.dtype,
        ),
        rtol=0,
        atol=1e-8,
    )


def test_center_pixel_uses_camera_negative_z_and_pose_rotation(intrinsics):
    # A +90 degree world Y rotation sends camera forward (-Z) to world -X.
    pose = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [-1.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    origins, directions = generate_rays(
        pose, intrinsics, torch.tensor([[2, 2]], dtype=torch.int64)
    )
    torch.testing.assert_close(
        origins[0], torch.tensor([1.0, 2.0, 3.0], dtype=origins.dtype)
    )
    torch.testing.assert_close(
        directions[0], torch.tensor([-1.0, 0.0, 0.0], dtype=directions.dtype)
    )


def test_normalized_directions_are_unit_length(intrinsics):
    pixels_xy = torch.tensor([[0, 0], [2, 2], [3, 1]], dtype=torch.int64)
    _, directions = generate_rays(
        torch.eye(4), intrinsics, pixels_xy, normalize_directions=True
    )
    torch.testing.assert_close(
        torch.linalg.norm(directions, dim=-1), torch.ones(3), rtol=0, atol=1e-6
    )


def test_fixed_inputs_are_deterministic(intrinsics):
    pose = torch.eye(4, dtype=torch.float64)
    pixels_xy = torch.tensor([[0, 1], [3, 2]], dtype=torch.int64)
    first = generate_rays(pose, intrinsics, pixels_xy)
    second = generate_rays(pose, intrinsics, pixels_xy)
    for first_tensor, second_tensor in zip(first, second):
        assert torch.equal(first_tensor, second_tensor)


def test_explicit_cpu_device_and_dtype(intrinsics):
    pose = torch.eye(4, dtype=torch.float64)
    pixels_xy = torch.tensor([[2, 2]], dtype=torch.int64)
    origins, directions = generate_rays(
        pose, intrinsics, pixels_xy, device=torch.device("cpu"), dtype=torch.float32
    )
    assert origins.device.type == directions.device.type == "cpu"
    assert origins.dtype == directions.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_explicit_cuda_device_and_dtype(intrinsics):
    pose = torch.eye(4, dtype=torch.float64)
    pixels_xy = torch.tensor([[2, 2]], dtype=torch.int64)
    origins, directions = generate_rays(
        pose, intrinsics, pixels_xy, device=torch.device("cuda"), dtype=torch.float32
    )
    assert origins.device.type == directions.device.type == "cuda"
    assert origins.dtype == directions.dtype == torch.float32
    torch.testing.assert_close(directions.cpu()[0], torch.tensor([0.0, 0.0, -1.0]))
