# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Basic implementation of Quant and Dequant modules.
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Union


class Quant(nn.Module):
    """
    Fake-quant module that applies a "saturating" approach using scale, zero_point, bit_width,
    and signedness parameters extracted from a Brevitas parameter dictionary.

    This module simulates quantization effects on tensors by scaling, shifting, rounding,
    and clamping their values.
    """

    def __init__(
        self,
        original_module: nn.Module,
        scale: float,
        zero_point: float,
        bit_width: float,
        signed: Optional[bool] = True,
    ) -> None:
        """
        Initialize the Quant module.

        Args:
            original_module: The original Brevitas quant module (kept for reference).
            scale: Scale factor used for quantization.
            zero_point: Zero-point used for quantization.
            bit_width: Bit width for the quantized representation (e.g., 8.0, 32.0).
            signed: Boolean flag indicating if quantization is signed.
        """
        super().__init__()
        self.original_module = original_module
        self.scale = scale
        self.zero_point = zero_point
        self.bit_width = bit_width
        self.signed = signed

        if self.bit_width is not None:
            bw_int = int(self.bit_width)
            if self.signed:
                self.min_val = -(2 ** (bw_int - 1))
                self.max_val = (2 ** (bw_int - 1)) - 1
            else:
                self.min_val = 0
                self.max_val = (2**bw_int) - 1
        else:
            self.min_val = None
            self.max_val = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fake quantization to the input tensor.

        The quantization process is as follows:
          1) Scale the input tensor by 1/scale.
          2) Shift the scaled tensor by the zero_point.
          3) Round the shifted tensor to the nearest integer.
          4) Clamp the rounded tensor to the representable range based on bit_width
             and signedness.

        Args:
            x: Input tensor.

        Returns:
            The fake quantized tensor.
        """
        if self.scale is None or self.zero_point is None:
            return x

        x_scaled = x / self.scale
        x_shifted = x_scaled + self.zero_point
        x_rounded = torch.round(x_shifted)
        if self.bit_width is not None:
            x_rounded = torch.clamp(x_rounded, self.min_val, self.max_val)
        return x_rounded


class Dequant(nn.Module):
    """
    Dequant module that re-applies scale and zero_point to invert the quantization effect.
    """

    def __init__(
        self,
        original_module: nn.Module,
        scale: float,
        zero_point: float,
        bit_width: float,
        signed: Optional[bool] = True,
    ) -> None:
        """
        Initialize the Dequant module.

        Args:
            original_module: The original Brevitas quant module.
            scale: Scale factor from extracted parameters.
            zero_point: Zero-point from extracted parameters.
            bit_width: Bit width from extracted parameters.
            signed: Boolean flag indicating if quantization is signed.
        """
        super().__init__()
        self.original_module = original_module
        self.scale = scale
        self.zero_point = zero_point
        self.bit_width = bit_width
        self.signed = signed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Undo the fake quantization by reversing the shift and scale.

        Args:
            x: Input tensor.

        Returns:
            The dequantized tensor.
        """
        if self.scale is None or self.zero_point is None:
            return x
        x_dequant = (x - self.zero_point) * self.scale
        return x_dequant
