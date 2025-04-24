# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import torch
import torch.nn as nn
from typing import Optional


class Quant(nn.Module):
    """Quantization module that applies scale, zero-point, and bit-width constraints."""

    def __init__(
        self,
        originalModule: nn.Module,
        scale: float,
        zeroPoint: float,
        bitWidth: float,
        signed: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.originalModule = originalModule
        self.scale = scale
        self.zeroPoint = zeroPoint
        self.bitWidth = bitWidth
        self.signed = signed

        if self.bitWidth is not None:
            bwInt = int(self.bitWidth)
            if self.signed:
                self.minVal = -(2 ** (bwInt - 1))
                self.maxVal = (2 ** (bwInt - 1)) - 1
            else:
                self.minVal = 0
                self.maxVal = (2**bwInt) - 1
        else:
            self.minVal = None
            self.maxVal = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize the input tensor."""
        if self.scale is None or self.zeroPoint is None:
            return x

        xScaled = x / self.scale
        xShifted = xScaled + self.zeroPoint
        xRounded = torch.round(xShifted)
        if self.bitWidth is not None:
            xRounded = torch.clamp(xRounded, self.minVal, self.maxVal)
        return xRounded


class Dequant(nn.Module):
    """Dequantization module that applies inverse scale and zero-point transformations."""

    def __init__(
        self,
        originalModule: nn.Module,
        scale: float,
        zeroPoint: float,
        bitWidth: float,
        signed: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.originalModule = originalModule
        self.scale = scale
        self.zeroPoint = zeroPoint
        self.bitWidth = bitWidth
        self.signed = signed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize the input tensor."""
        if self.scale is None or self.zeroPoint is None:
            return x
        xDequant = (x - self.zeroPoint) * self.scale
        return xDequant
