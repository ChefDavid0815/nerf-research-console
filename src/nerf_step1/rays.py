"""Generate pinhole camera rays using the original NeRF Blender convention.

The JSON ``transform_matrix`` is camera-to-world in OpenGL coordinates: camera
+X points right, +Y up, and viewing forward is -Z. Pixel indices are integer
image coordinates with (0, 0) at the top left. As in the original NeRF
``get_rays``, the pixel center is represented by its integer index; do not
silently add 0.5. The principal point is (width/2, height/2).

The default world directions are deliberately *not* unit length, matching the
original baseline's ray parameterization. Set ``normalize_directions=True``
only where unit directions are needed (for example, a direction-only plot).
Future volume rendering must use bounds consistent with this choice.
"""

from __future__ import annotations

import torch

from .dataset import CameraIntrinsics


def generate_rays(
    camera_to_world: torch.Tensor,
    intrinsics: CameraIntrinsics,
    pixels_xy: torch.Tensor | list[tuple[int, int]] | tuple[int, int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    normalize_directions: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(origins, directions)`` with shape ``pixels_xy.shape[:-1]+(3,)``.

    Each pixel is ``(x, y)``. A single pair gives a 3-vector; an ``[N,2]``
    array gives ``[N,3]``. All rays start at the camera translation. The
    requested device/dtype apply to both outputs and are never inferred from
    the pixel index tensor; by default they follow ``camera_to_world``.
    """
    if not isinstance(camera_to_world, torch.Tensor):
        raise TypeError("camera_to_world must be a torch.Tensor")
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must have shape [4, 4]")
    target_device = torch.device(device) if device is not None else camera_to_world.device
    target_dtype = dtype if dtype is not None else camera_to_world.dtype
    if target_dtype not in (torch.float32, torch.float64):
        raise TypeError("ray dtype must be torch.float32 or torch.float64")

    pose = camera_to_world.to(device=target_device, dtype=target_dtype)
    pixels = torch.as_tensor(pixels_xy, device=target_device)
    if pixels.ndim < 1 or pixels.shape[-1] != 2:
        raise ValueError("pixels_xy must end with an (x, y) coordinate pair")
    pixels = pixels.to(dtype=target_dtype)
    if not bool(torch.isfinite(pose).all()) or not bool(torch.isfinite(pixels).all()):
        raise ValueError("pose and pixel coordinates must be finite")
    if not bool(torch.equal(pixels, pixels.round())):
        raise ValueError("pixels_xy must contain integer pixel indices")

    x, y = pixels[..., 0], pixels[..., 1]
    if bool(((x < 0) | (x >= intrinsics.width) | (y < 0) | (y >= intrinsics.height)).any()):
        raise ValueError("pixel index is outside the image")
    camera_directions = torch.stack(
        ((x - intrinsics.cx) / intrinsics.fx,
         -(y - intrinsics.cy) / intrinsics.fy,
         -torch.ones_like(x)),
        dim=-1,
    )
    directions = camera_directions @ pose[:3, :3].T
    if normalize_directions:
        directions = torch.nn.functional.normalize(directions, dim=-1)
    origins = pose[:3, 3].expand_as(directions)
    return origins, directions
