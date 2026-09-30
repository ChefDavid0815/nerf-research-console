"""The view-dependent vanilla NeRF MLP used by the bandwidth experiment.

``VanillaNeRF.forward(x, d)`` maps raw positions and viewing directions to
``(rgb, sigma)``. The two Fourier encoders are constructed with the model so
the independent variable, ``position_encoding_L``, cannot change during a
run without constructing a new network. The remaining architecture settings
are controlled variables and can be saved using ``architecture_config``.

The original NeRF implementation normalizes viewing directions before it
calls the network. This module encodes directions as supplied, leaving ray
generation and normalization decisions with the caller. Its raw density and
RGB heads match the released view-dependent MLP; ReLU and sigmoid are applied
here to provide the constrained outputs requested by this project's forward
contract. The released TensorFlow code applies those constraints in rendering.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .positional_encoding import SinusoidalPositionalEncoding


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class VanillaNeRF(nn.Module):
    """Original NeRF's 8-by-256 position trunk and one-layer view branch.

    The skip index is zero-based and means *after* that position layer's
    activation. At the default index 4, the sixth fully connected position
    layer receives the fifth layer's output concatenated with encoded xyz.
    This follows ``bmild/nerf/run_nerf_helpers.py`` rather than the sometimes
    ambiguous description "skip at layer 4".
    """

    def __init__(
        self,
        position_encoding_L: int = 10,
        direction_encoding_L: int = 4,
        network_depth: int = 8,
        network_width: int = 256,
        skip_connection_layer: int = 4,
        view_width: int = 128,
    ) -> None:
        super().__init__()
        self._position_encoding_L = _nonnegative_int("position_encoding_L", position_encoding_L)
        self._direction_encoding_L = _nonnegative_int("direction_encoding_L", direction_encoding_L)
        self._network_depth = _positive_int("network_depth", network_depth)
        self._network_width = _positive_int("network_width", network_width)
        self._skip_connection_layer = _nonnegative_int(
            "skip_connection_layer", skip_connection_layer
        )
        self._view_width = _positive_int("view_width", view_width)
        if self._skip_connection_layer >= self._network_depth - 1:
            raise ValueError(
                "skip_connection_layer must precede a position layer "
                "(must be less than network_depth - 1)"
            )

        self.position_encoder = SinusoidalPositionalEncoding(
            input_dim=3, num_frequencies=self._position_encoding_L, include_input=True
        )
        self.direction_encoder = SinusoidalPositionalEncoding(
            input_dim=3, num_frequencies=self._direction_encoding_L, include_input=True
        )
        position_dim = self.position_encoder.output_dim
        direction_dim = self.direction_encoder.output_dim

        position_layers: list[nn.Linear] = []
        for layer_index in range(self._network_depth):
            if layer_index == 0:
                input_width = position_dim
            elif layer_index == self._skip_connection_layer + 1:
                input_width = self._network_width + position_dim
            else:
                input_width = self._network_width
            position_layers.append(nn.Linear(input_width, self._network_width))
        self.position_layers = nn.ModuleList(position_layers)
        self.density_layer = nn.Linear(self._network_width, 1)
        self.feature_layer = nn.Linear(self._network_width, self._network_width)
        self.view_layer = nn.Linear(self._network_width + direction_dim, self._view_width)
        self.rgb_layer = nn.Linear(self._view_width, 3)

    @property
    def position_encoding_L(self) -> int:
        return self._position_encoding_L

    @property
    def direction_encoding_L(self) -> int:
        return self._direction_encoding_L

    @property
    def network_depth(self) -> int:
        return self._network_depth

    @property
    def network_width(self) -> int:
        return self._network_width

    @property
    def skip_connection_layer(self) -> int:
        return self._skip_connection_layer

    @property
    def view_width(self) -> int:
        return self._view_width

    @property
    def architecture_config(self) -> dict[str, int]:
        """Return a serializable snapshot of the network's controlled settings."""
        return {
            "position_encoding_L": self.position_encoding_L,
            "direction_encoding_L": self.direction_encoding_L,
            "network_depth": self.network_depth,
            "network_width": self.network_width,
            "skip_connection_layer": self.skip_connection_layer,
            "view_width": self.view_width,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> VanillaNeRF:
        """Construct from a complete, explicit architecture config snapshot.

        A full experiment mapping may include unrelated dataset and training
        keys, but none of the six architecture settings may be omitted. This
        prevents an experiment run from silently reverting a controlled value
        to a constructor default.
        """
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        required_keys = (
            "position_encoding_L",
            "direction_encoding_L",
            "network_depth",
            "network_width",
            "skip_connection_layer",
            "view_width",
        )
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(f"config is missing required architecture keys: {', '.join(missing)}")
        return cls(**{key: config[key] for key in required_keys})

    def forward(
        self, positions: torch.Tensor, directions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``F_theta(x, d) -> (rgb, sigma)`` for raw ``[N, 3]`` input.

        ``rgb`` has shape ``[N, 3]`` and lies in ``[0, 1]``; ``sigma`` has
        shape ``[N, 1]`` and is nonnegative. The two inputs must share the
        model's floating dtype and device. To reproduce original NeRF viewing
        behavior, callers should supply unit-length directions.
        """
        for name, tensor in (("positions", positions), ("directions", directions)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.ndim != 2 or tensor.shape[1] != 3:
                raise ValueError(f"{name} must have shape [N, 3]")
            if tensor.dtype not in (torch.float32, torch.float64):
                raise TypeError(f"{name} must have dtype torch.float32 or torch.float64")
        if positions.shape[0] != directions.shape[0]:
            raise ValueError("positions and directions must have the same batch size")
        if positions.device != directions.device or positions.dtype != directions.dtype:
            raise ValueError("positions and directions must share a device and dtype")
        model_weight = self.position_layers[0].weight
        if positions.device != model_weight.device or positions.dtype != model_weight.dtype:
            raise ValueError("input device and dtype must match the model parameters")

        encoded_position = self.position_encoder(positions)
        encoded_direction = self.direction_encoder(directions)
        hidden = encoded_position
        for layer_index, layer in enumerate(self.position_layers):
            hidden = F.relu(layer(hidden))
            if layer_index == self._skip_connection_layer:
                hidden = torch.cat((encoded_position, hidden), dim=-1)

        sigma = F.relu(self.density_layer(hidden))
        features = self.feature_layer(hidden)
        view_hidden = F.relu(self.view_layer(torch.cat((features, encoded_direction), dim=-1)))
        rgb = torch.sigmoid(self.rgb_layer(view_hidden))
        return rgb, sigma
