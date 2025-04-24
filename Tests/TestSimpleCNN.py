# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
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
from DeepQuant import exportQuantModel


class SimpleQuantCNN(nn.Module):
    """A simple quantized CNN with two conv layers and a linear layer."""

    convQuantParams = {
        "bias": True,
        "weight_bit_width": 4,
        "bias_quant": Int32Bias,
        "input_quant": Int8ActPerTensorFloat,
        "weight_quant": Int8WeightPerTensorFloat,
        "output_quant": Int8ActPerTensorFloat,
        "return_quant_tensor": True,
    }

    def __init__(self, inChannels: int = 1, numClasses: int = 10) -> None:
        super().__init__()
        self.inputQuant = qnn.QuantIdentity(return_quant_tensor=True)

        self.conv1 = qnn.QuantConv2d(
            in_channels=inChannels,
            out_channels=16,
            kernel_size=3,
            padding=1,
            **SimpleQuantCNN.convQuantParams,
        )
        self.relu1 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = qnn.QuantConv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
            **SimpleQuantCNN.convQuantParams,
        )
        self.relu2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc = qnn.QuantLinear(
            in_features=32 * 7 * 7,
            out_features=numClasses,
            **SimpleQuantCNN.convQuantParams,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inputQuant(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x


@pytest.mark.ModelTests
def deepQuantTestSimpleCNN() -> None:
    torch.manual_seed(42)
    model = SimpleQuantCNN().eval()
    sampleInput = torch.randn(1, 1, 28, 28)
    exportQuantModel(model, sampleInput, debug=True)
