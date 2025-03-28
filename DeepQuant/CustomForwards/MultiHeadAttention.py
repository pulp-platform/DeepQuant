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


def unrolledQuantMhaForward(
    self: QuantMultiheadAttention, query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    """
    Export-friendly forward that explicitly unrolls the multi-head logic.

    Steps:
      1) Q, K, V projections
      2) Reshapes & permutes for multi-head
      3) Scales queries
      4) Applies softmax and intermediate quantizations
      5) Out projection

    Args:
        self: The QuantMultiheadAttention instance.
        query: The query tensor of shape [sequence_len, batch_size, embed_dim].
        key: The key tensor, same shape as query.
        value: The value tensor, same shape as query.

    Returns:
        A torch.Tensor of shape [sequence_len, batch_size, embed_dim]
        after the unrolled MHA steps.
    """
    # 1) Q, K, V projections
    qOut = self.q_proj(query)
    kOut = self.k_proj(key)
    vOut = self.v_proj(value)

    # 2) Multi-head reshape
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

    # 3) Scale queries, then quantize
    qScaled = qOut / math.sqrt(headDim)
    qScaled = self.q_scaled_quant(qScaled)

    # 4) Transpose + quantize K, compute attention weights
    k_t = kOut.transpose(-2, -1)
    k_t = self.k_transposed_quant(k_t)

    attnWeights = torch.bmm(qScaled, k_t)
    attnWeights = self.softmax_input_quant(attnWeights)
    attnWeights = F.softmax(attnWeights, dim=-1)
    attnWeights = self.attn_output_weights_quant(attnWeights)

    # 5) Quantize V, multiply, reshape back, and final out projection
    vOut = self.v_quant(vOut)
    attnOutput = torch.bmm(attnWeights, vOut)

    attnOutput = (
        attnOutput.view(batchSize, self.num_heads, seqLen, headDim)
        .permute(2, 0, 1, 3)
        .reshape(seqLen, batchSize, embedDim)
    )

    attnOutput = self.out_proj(attnOutput)
    return attnOutput
