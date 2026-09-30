"""Explicit sinusoidal positional encoding for the NeRF bandwidth experiment.

For each input vector, features are ordered as the original coordinates,
then ``sin(2**k * pi * x)`` and ``cos(2**k * pi * x)`` for all vector
components and each frequency ``k = 0, ..., L - 1``. The bandwidth ``L`` is
fixed when the module is constructed; it is never a forward-pass argument.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Encode vectors with fixed powers-of-two sinusoidal frequency bands.

    For a 3D input, ``output_dim = 3 + 6 * L``. The raw input is mandatory
    for every bandwidth, and ``L=0`` is exactly the raw-coordinate baseline.
    Frequency tensors follow each input's dtype and device.
    """

    def __init__(
        self, input_dim: int, num_frequencies: int, include_input: bool = True
    ) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if (
            isinstance(num_frequencies, bool)
            or not isinstance(num_frequencies, int)
            or num_frequencies < 0
        ):
            raise ValueError("num_frequencies must be a non-negative integer")
        if not isinstance(include_input, bool):
            raise TypeError("include_input must be a bool")
        if not include_input:
            raise ValueError("include_input must be True for the NeRF research baseline")

        self._input_dim = input_dim
        self._num_frequencies = num_frequencies
        self._include_input = include_input

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def num_frequencies(self) -> int:
        return self._num_frequencies

    @property
    def include_input(self) -> bool:
        return self._include_input

    @property
    def output_dim(self) -> int:
        return self._input_dim * (1 + 2 * self._num_frequencies)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return features with shape ``inputs.shape[:-1] + (output_dim,)``."""
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim == 0 or inputs.shape[-1] != self._input_dim:
            raise ValueError(f"inputs must have a final dimension of {self._input_dim}")
        if not inputs.is_floating_point():
            raise TypeError("inputs must use a floating-point dtype")
        if self._num_frequencies == 0:
            return inputs

        # Construct on the input device and in its dtype so CPU/CUDA and
        # float32/float64 calls use the same mathematical definition.
        frequencies = torch.exp2(
            torch.arange(
                self._num_frequencies, device=inputs.device, dtype=inputs.dtype
            )
        ) * math.pi
        phases = inputs.unsqueeze(-2) * frequencies.unsqueeze(-1)
        # Shape [..., L, 2, input_dim] gives sin(all axes), cos(all axes)
        # within each band before the final flatten.
        encoded = torch.stack((torch.sin(phases), torch.cos(phases)), dim=-2)
        encoded = encoded.flatten(start_dim=-3)
        return torch.cat((inputs, encoded), dim=-1)
