# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>


import torch.nn as nn
from torch import Tensor
from brevitas.nn.quant_layer import QuantNonLinearActLayer


class WrapperActivation(nn.Module):
    """Expose inner activation so FX sees it as a leaf."""

    def __init__(self, actImpl: nn.Module) -> None:
        super().__init__()
        self.actImpl = actImpl

    def forward(self, quantInput: Tensor) -> Tensor:
        return self.actImpl(quantInput)


def activationForward(self: QuantNonLinearActLayer, inp: Tensor) -> Tensor:
    """Unroll input→act→output quant steps."""
    quantInput = self.input_quant(inp) if self.input_quant is not None else inp
    if hasattr(self, "wrappedActImpl"):
        output = self.wrappedActImpl(quantInput)
    else:
        output = quantInput
    quantOutput = self.act_quant(output) if self.act_quant is not None else output
    return quantOutput
