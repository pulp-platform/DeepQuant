# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Test file demonstrating an example of a small CNN model with quantized layers
and exporting it via the exportBrevitas function.
"""

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int32Bias,
    Int8WeightPerTensorFloat,
)
from DeepQuant.export_brevitas import exportBrevitas


class SimpleQuantCNN(nn.Module):
    """
    A simple quantized CNN that includes:
      - Input quantization
      - Two QuantConv2d layers with ReLU
      - MaxPool2d
      - A final QuantLinear layer
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        """
        Args:
            in_channels: Number of input channels (e.g., 1 for grayscale).
            num_classes: Number of output classes for the final linear layer.
        """
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)

        # First convolutional block
        self.conv1 = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            padding=1,
            bias=True,
            weight_bit_width=4,
            bias_quant=Int32Bias,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.relu1 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Second convolutional block
        self.conv2 = qnn.QuantConv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=True,
            weight_bit_width=4,
            bias_quant=Int32Bias,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.relu2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Final linear layer
        self.flatten = nn.Flatten()
        self.fc = qnn.QuantLinear(
            in_features=32 * 7 * 7,  # If input is 28x28, shape after pooling is 7x7
            out_features=num_classes,
            bias=True,
            weight_bit_width=4,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SimpleQuantCNN.

        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width].

        Returns:
            A quantized output tensor (batch_size, num_classes).
        """
        x = self.input_quant(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x


def test_simple_quant_cnn() -> None:
    """
    Test function for the SimpleQuantCNN using exportBrevitas.
    Ensures both the forward pass and the FX tracing work properly.

    Returns:
        None
    """
    torch.manual_seed(42)

    model = SimpleQuantCNN().eval()
    sample_input = torch.randn(1, 1, 28, 28)

    fx_model = exportBrevitas(model, sample_input, debug=True)
