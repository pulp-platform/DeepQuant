# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import brevitas.nn as qnn
import pytest
import torch
import torch.nn as nn
from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
)

from DeepQuant import brevitasToTrueQuant


class QuantLinearNet(nn.Module):
    """Simple quantized network with a single linear layer."""

    def __init__(self, inFeatures: int = 16, hiddenFeatures: int = 32) -> None:
        super().__init__()
        self.inputQuant = qnn.QuantIdentity(return_quant_tensor=True)
        self.linear1 = qnn.QuantLinear(
            in_features=inFeatures,
            out_features=hiddenFeatures,
            bias=True,
            weight_bit_width=4,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            return_quant_tensor=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inputQuant(x)
        x = self.linear1(x)
        return x


@pytest.mark.SingleLayerTests
def deepQuantTestLinear() -> None:
    torch.manual_seed(42)
    model = QuantLinearNet().eval()
    sampleInput = torch.randn(1, 4, 16)
    brevitasToTrueQuant(model, sampleInput, debug=True, checkEquivalence=True)
