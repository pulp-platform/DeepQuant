# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>


import torch.nn as nn
from torch import Tensor
from brevitas.nn.quant_layer import QuantNonLinearActLayer


class InnerForwardImplWrapperActivation(nn.Module):
    """
    A small wrapper around the activation function of a Brevitas QuantActivation layer.

    This wrapper exposes the original activation function as a standalone submodule
    so that FX tracing can display it as a separate node.
    """

    def __init__(self, actImpl: nn.Module) -> None:
        """
        Args:
            act_impl: The original activation function module (e.g. an instance of nn.ReLU).
        """
        super().__init__()
        self.actImpl = actImpl

    def forward(self, quantInput: Tensor) -> Tensor:
        """
        Applies the wrapped activation function.

        Args:
            quant_input: Input tensor after input quantization.

        Returns:
            Output tensor after applying the activation.
        """
        return self.actImpl(quantInput)


def quantActivationForward(self: QuantNonLinearActLayer, inp: Tensor) -> Tensor:
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
    quantInput = self.input_quant(inp) if self.input_quant is not None else inp
    # Use the wrapped activation if available; otherwise pass through.
    if hasattr(self, "wrappedActImpl"):
        output = self.wrappedActImpl(quantInput)
    else:
        output = quantInput
        import IPython; IPython.embed()
    quantOutput = self.act_quant(output) if self.act_quant is not None else output
    return quantOutput
