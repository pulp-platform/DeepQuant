# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import warnings
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias
from DeepQuant.export_brevitas import exportBrevitas

# Filter warnings for better output
warnings.filterwarnings("ignore", category=UserWarning, module="torch._tensor")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="brevitas.backport.fx._symbolic_trace"
)
warnings.filterwarnings("ignore", message="Defining your `__torch_function__`")


class SimpleQuantModel(nn.Module):
    """
    A simple quantized model with one QuantIdentity and one QuantConv2d,
    used to demonstrate the export process.
    """

    def __init__(self) -> None:
        """
        Initializes the SimpleQuantModel with:
          - A QuantIdentity (input_quant)
          - A QuantConv2d layer (conv)
        """
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.conv = qnn.QuantConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            bias=True,
            weight_bit_width=4,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying quantization, then a quantized convolution.

        Args:
            x: Input tensor of shape [N, 3, H, W].

        Returns:
            A torch.Tensor with shape [N, 16, H-2, W-2] (depending on padding/stride).
        """
        x = self.input_quant(x)
        x = self.conv(x)
        return x


def main() -> None:
    """
    Demonstrates usage of exportBrevitas on a simple model.

    Returns:
        None
    """
    model = SimpleQuantModel().eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    fx_model = exportBrevitas(model, dummy_input, debug=True)


if __name__ == "__main__":
    main()
