# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import torch.nn as nn
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer
from torch import Tensor


class WrapperLinear(nn.Module):
    """Expose `inner_forward_impl` as a standalone submodule."""

    def __init__(self, innerForwardImpl: nn.Module) -> None:
        super().__init__()
        self.innerForwardImpl = innerForwardImpl

    def forward(
        self, quantInput: Tensor, quantWeight: Tensor, quantBias: Tensor
    ) -> Tensor:
        return self.innerForwardImpl(quantInput, quantWeight, quantBias)


def linearForward(self: QuantWeightBiasInputOutputLayer, inp: Tensor) -> Tensor:
    """Quant-in → quant-weight/bias → matmul → quant-out."""
    quantInput = self.input_quant(inp)
    quantWeight = self.weight_quant(self.weight)

    quantBias = None
    if self.bias is not None:
        quantBias = self.bias_quant(self.bias, quantInput, quantWeight)

    output = self.wrappedInnerForwardImpl(quantInput, quantWeight, quantBias)
    quantOutput = self.output_quant(output)
    return quantOutput
