# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Victor Jung <jungvi@iis.ee.ethz.ch>
# Federico Brancasi <fbrancasi@ethz.ch>


import pytest
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int32Bias,
    Int8WeightPerTensorFloat,
)
from DeepQuant.ExportBrevitas import exportBrevitas


class QuantConvNet(nn.Module):

    convAndLinQuantParams = {
        "bias": True,
        "weight_bit_width": 4,
        "bias_quant": Int32Bias,
        "input_quant": Int8ActPerTensorFloat,
        "weight_quant": Int8WeightPerTensorFloat,
        "output_quant": Int8ActPerTensorFloat,
        "return_quant_tensor": True,
    }

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.inputQuant = qnn.QuantIdentity(return_quant_tensor=True)

        self.conv1 = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            padding=1,
            **QuantConvNet.convAndLinQuantParams
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.inputQuant(x)
        x = self.conv1(x)
        
        return x


@pytest.mark.SingleLayerTests
def deepQuantTestConv() -> None:
    
    torch.manual_seed(42)

    model = QuantConvNet().eval()
    sampleInput = torch.randn(1, 1, 28, 28)
    exportBrevitas(model, sampleInput, debug=True)
