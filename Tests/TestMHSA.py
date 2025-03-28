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
from DeepQuant.ExportBrevitas import exportBrevitas

from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int32Bias,
    Int8WeightPerTensorFloat,
    Uint8ActPerTensorFloat,
)


class QuantMHSANet(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """
        Args:
            embed_dim: The dimension of each embedding vector.
            num_heads: The number of attention heads.
        """
        super().__init__()
        self.inputQuant = qnn.QuantIdentity(return_quant_tensor=True)
        self.mha = qnn.QuantMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=True,
            packed_in_proj=False,  # separate Q, K, V
            batch_first=False,  # expects (sequence, batch, embed_dim)
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
        """
        Forward pass that first quantizes the input, then applies multi-head attention.

        Args:
            x: Input tensor of shape [sequence_len, batch_size, embed_dim].

        Returns:
            A tuple (output, None) as per the Brevitas MHA API, where output has shape
            [sequence_len, batch_size, embed_dim].
        """
        x = self.inputQuant(x)
        out = self.mha(x, x, x)
        return out


@pytest.mark.SingleLayerTests
def deepQuantTestMHSA() -> None:

    torch.manual_seed(42)

    model = QuantMHSANet(embed_dim=16, num_heads=4).eval()
    sampleInput = torch.randn(10, 2, 16)

    exportBrevitas(model, sampleInput, debug=True)
