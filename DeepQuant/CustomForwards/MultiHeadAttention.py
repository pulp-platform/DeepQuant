# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from brevitas.nn.quant_mha import QuantMultiheadAttention


def mhaForward(
    self: QuantMultiheadAttention, query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    """Explicit, export-friendly MHA forward."""
    qOut = self.q_proj(query)
    kOut = self.k_proj(key)
    vOut = self.v_proj(value)

    seqLen, batchSize, embedDim = qOut.shape
    headDim = embedDim // self.num_heads

    qOut = (
        qOut.view(seqLen, batchSize, self.num_heads, headDim)
        .permute(1, 2, 0, 3)
        .reshape(batchSize * self.num_heads, seqLen, headDim)
    )
    kOut = (
        kOut.view(seqLen, batchSize, self.num_heads, headDim)
        .permute(1, 2, 0, 3)
        .reshape(batchSize * self.num_heads, seqLen, headDim)
    )
    vOut = (
        vOut.view(seqLen, batchSize, self.num_heads, headDim)
        .permute(1, 2, 0, 3)
        .reshape(batchSize * self.num_heads, seqLen, headDim)
    )

    qScaled = qOut / math.sqrt(headDim)
    qScaled = self.q_scaled_quant(qScaled)

    k_t = kOut.transpose(-2, -1)
    k_t = self.k_transposed_quant(k_t)

    attnWeights = torch.bmm(qScaled, k_t)
    attnWeights = self.softmax_input_quant(attnWeights)
    attnWeights = F.softmax(attnWeights, dim=-1)
    attnWeights = self.attn_output_weights_quant(attnWeights)

    vOut = self.v_quant(vOut)
    attnOutput = torch.bmm(attnWeights, vOut)

    attnOutput = (
        attnOutput.view(batchSize, self.num_heads, seqLen, headDim)
        .permute(2, 0, 1, 3)
        .reshape(seqLen, batchSize, embedDim)
    )

    attnOutput = self.out_proj(attnOutput)
    return attnOutput