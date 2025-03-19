# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Custom forward implementations for Brevitas QuantActivation layers.
"""

import torch
import torch.nn as nn
from torch import Tensor
from brevitas.nn.quant_layer import QuantNonLinearActLayer


class InnerForwardImplWrapperActivation(nn.Module):
    """
    A small wrapper around the activation function of a Brevitas QuantActivation layer.

    This wrapper exposes the original activation function as a standalone submodule
    so that FX tracing can display it as a separate node.
    """

    def __init__(self, act_impl: nn.Module) -> None:
        """
        Args:
            act_impl: The original activation function module (e.g. an instance of nn.ReLU).
        """
        super().__init__()
        self.act_impl = act_impl

    def forward(self, quant_input: Tensor) -> Tensor:
        """
        Applies the wrapped activation function.

        Args:
            quant_input: Input tensor after input quantization.

        Returns:
            Output tensor after applying the activation.
        """
        return self.act_impl(quant_input)


def quant_activation_forward(self: QuantNonLinearActLayer, inp: Tensor) -> Tensor:
    """
    Unrolled forward pass for a Brevitas QuantActivation layer.

    Steps:
      1) Apply self.input_quant to the input.
      2) Apply the activation function via the wrapped activation implementation.
      3) Apply self.act_quant to the activation output.

    Args:
        self: The QuantNonLinearActLayer instance.
        inp: The input tensor.

    Returns:
        Output tensor after applying activation and output quantization.
    """
    quant_input = self.input_quant(inp) if self.input_quant is not None else inp
    # Use the wrapped activation if available; otherwise pass through.
    if hasattr(self, "wrapped_act_impl"):
        output = self.wrapped_act_impl(quant_input)
    else:
        output = quant_input
    quant_output = self.act_quant(output) if self.act_quant is not None else output
    return quant_output
