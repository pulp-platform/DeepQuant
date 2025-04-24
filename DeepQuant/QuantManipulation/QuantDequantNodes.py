# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import torch
import torch.nn as nn
from typing import Optional


class Quant(nn.Module):
    def __init__(
        self,
        original_module: nn.Module,
        scale: float,
        zero_point: float,
        bit_width: float,
        signed: Optional[bool] = True,
    ) -> None:
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
        if self.scale is None or self.zero_point is None:
            return x

        x_scaled = x / self.scale
        x_shifted = x_scaled + self.zero_point
        x_rounded = torch.round(x_shifted)
        if self.bit_width is not None:
            x_rounded = torch.clamp(x_rounded, self.min_val, self.max_val)
        return x_rounded


class Dequant(nn.Module):
    def __init__(
        self,
        original_module: nn.Module,
        scale: float,
        zero_point: float,
        bit_width: float,
        signed: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.original_module = original_module
        self.scale = scale
        self.zero_point = zero_point
        self.bit_width = bit_width
        self.signed = signed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale is None or self.zero_point is None:
            return x
        x_dequant = (x - self.zero_point) * self.scale
        return x_dequant
