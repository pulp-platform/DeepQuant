# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import pytest
import torch
import torch.nn as nn
import brevitas.nn as qnn
from torch import Tensor
from DeepQuant import exportQuantModel

from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int32Bias,
    Int8WeightPerTensorFloat,
    Uint8ActPerTensorFloat,
)


class QuantMHSANet(nn.Module):
    """Simple quantized network with multi-head self-attention."""

    def __init__(self, embedDim: int, numHeads: int) -> None:
        super().__init__()
        self.inputQuant = qnn.QuantIdentity(return_quant_tensor=True)
        self.mha = qnn.QuantMultiheadAttention(
            embed_dim=embedDim,
            num_heads=numHeads,
            dropout=0.0,
            bias=True,
            packed_in_proj=False,  # FBRANCASI: separate Q, K, V
            batch_first=False,  # FBRANCASI: expects (sequence, batch, embed_dim)
            in_proj_input_quant=Int8ActPerTensorFloat,
            in_proj_weight_quant=Int8WeightPerTensorFloat,
            in_proj_bias_quant=Int32Bias,
            attn_output_weights_quant=Uint8ActPerTensorFloat,
            q_scaled_quant=Int8ActPerTensorFloat,
            k_transposed_quant=Int8ActPerTensorFloat,
            v_quant=Int8ActPerTensorFloat,
            out_proj_input_quant=Int8ActPerTensorFloat,
            out_proj_weight_quant=Int8WeightPerTensorFloat,
            out_proj_bias_quant=Int32Bias,
            out_proj_output_quant=Int8ActPerTensorFloat,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.inputQuant(x)
        out = self.mha(x, x, x)
        return out


@pytest.mark.SingleLayerTests
def deepQuantTestMHSA() -> None:
    torch.manual_seed(42)
    model = QuantMHSANet(embedDim=16, numHeads=4).eval()
    sampleInput = torch.randn(10, 2, 16)
    exportQuantModel(model, sampleInput)
