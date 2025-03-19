# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Custom forward implementations for Brevitas QuantLinear layers.
"""

import torch
import torch.nn as nn
from torch import Tensor
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer


class InnerForwardImplWrapperLinear(nn.Module):
    """
    A small wrapper around the 'inner_forward_impl' of a Brevitas QuantLinear
    (QuantWeightBiasInputOutputLayer).

    We want to expose the logic within 'inner_forward_impl' as a standalone
    submodule, so that FX tracing can see it as a leaf.
    """

    def __init__(self, inner_forward_impl: nn.Module) -> None:
        """
        Args:
            inner_forward_impl: The original function that processes
                                (quant_input, quant_weight, quant_bias).
        """
        super().__init__()
        self.inner_forward_impl = inner_forward_impl

    def forward(
        self, quant_input: Tensor, quant_weight: Tensor, quant_bias: Tensor
    ) -> Tensor:
        """
        Applies the wrapped inner_forward_impl.

        Args:
            quant_input: Input after input_quant.
            quant_weight: Weight after weight_quant.
            quant_bias: Bias after bias_quant (or None).

        Returns:
            A torch.Tensor with the linear operation applied.
        """
        return self.inner_forward_impl(quant_input, quant_weight, quant_bias)


def quantWBIOL_forward(self: QuantWeightBiasInputOutputLayer, inp: Tensor) -> Tensor:
    """
    Unrolled forward pass for a Brevitas QuantLinear:

    Steps:
      1) self.input_quant
      2) self.weight_quant
      3) self.bias_quant (if bias is present)
      4) inner_forward_impl (wrapped)
      5) self.output_quant

    Args:
        self: The QuantWeightBiasInputOutputLayer instance.
        inp: The input Tensor to be processed.

    Returns:
        Output Tensor after the unrolled quantized linear steps.
    """
    quant_input = self.input_quant(inp)
    quant_weight = self.weight_quant(self.weight)

    quant_bias = None
    if self.bias is not None:
        quant_bias = self.bias_quant(self.bias, quant_input, quant_weight)

    output = self.wrapped_inner_forward_impl(quant_input, quant_weight, quant_bias)
    quant_output = self.output_quant(output)
    return quant_output
