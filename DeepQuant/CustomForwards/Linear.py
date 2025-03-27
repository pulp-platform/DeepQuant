# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>


import torch.nn as nn
from torch import Tensor
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer


class InnerForwardImplWrapperLinear(nn.Module):
    """
    A small wrapper around the 'innerForwardImpl' of a Brevitas QuantLinear
    (QuantWeightBiasInputOutputLayer).

    We want to expose the logic within 'innerForwardImpl' as a standalone
    submodule, so that FX tracing can see it as a leaf.
    """

    def __init__(self, innerForwardImpl: nn.Module) -> None:
        """
        Args:
            innerForwardImpl: The original function that processes
                                (quant_input, quant_weight, quant_bias).
        """
        super().__init__()
        self.innerForwardImpl = innerForwardImpl

    def forward(
        self, quantInput: Tensor, quantWeight: Tensor, quantBias: Tensor
    ) -> Tensor:
        """
        Applies the wrapped innerForwardImpl.

        Args:
            quant_input: Input after input_quant.
            quant_weight: Weight after weight_quant.
            quant_bias: Bias after bias_quant (or None).

        Returns:
            A torch.Tensor with the linear operation applied.
        """
        return self.innerForwardImpl(quantInput, quantWeight, quantBias)


def quantWBIOLForward(self: QuantWeightBiasInputOutputLayer, inp: Tensor) -> Tensor:
    """
    Unrolled forward pass for a Brevitas QuantLinear:

    Steps:
      1) self.input_quant
      2) self.weight_quant
      3) self.bias_quant (if bias is present)
      4) innerForwardImpl (wrapped)
      5) self.output_quant

    Args:
        self: The QuantWeightBiasInputOutputLayer instance.
        inp: The input Tensor to be processed.

    Returns:
        Output Tensor after the unrolled quantized linear steps.
    """
    quantInput = self.input_quant(inp)
    quantWeight = self.weight_quant(self.weight)

    quantBias = None
    if self.bias is not None:
        quantBias = self.bias_quant(self.bias, quantInput, quantWeight)

    output = self.wrappedInnerForwardImpl(quantInput, quantWeight, quantBias)
    quantOutput = self.output_quant(output)
    return quantOutput
