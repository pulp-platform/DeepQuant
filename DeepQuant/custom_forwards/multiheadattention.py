# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Custom forward implementation for Brevitas QuantMultiheadAttention.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from brevitas.nn.quant_mha import QuantMultiheadAttention


def unrolled_quant_mha_forward(
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
    q_out = self.q_proj(query)
    k_out = self.k_proj(key)
    v_out = self.v_proj(value)

    # 2) Multi-head reshape
    seq_len, batch_size, embed_dim = q_out.shape
    head_dim = embed_dim // self.num_heads

    q_out = (
        q_out.view(seq_len, batch_size, self.num_heads, head_dim)
        .permute(1, 2, 0, 3)
        .reshape(batch_size * self.num_heads, seq_len, head_dim)
    )
    k_out = (
        k_out.view(seq_len, batch_size, self.num_heads, head_dim)
        .permute(1, 2, 0, 3)
        .reshape(batch_size * self.num_heads, seq_len, head_dim)
    )
    v_out = (
        v_out.view(seq_len, batch_size, self.num_heads, head_dim)
        .permute(1, 2, 0, 3)
        .reshape(batch_size * self.num_heads, seq_len, head_dim)
    )

    # 3) Scale queries, then quantize
    q_scaled = q_out / math.sqrt(head_dim)
    q_scaled = self.q_scaled_quant(q_scaled)

    # 4) Transpose + quantize K, compute attention weights
    k_t = k_out.transpose(-2, -1)
    k_t = self.k_transposed_quant(k_t)

    attn_weights = torch.bmm(q_scaled, k_t)
    attn_weights = self.softmax_input_quant(attn_weights)
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = self.attn_output_weights_quant(attn_weights)

    # 5) Quantize V, multiply, reshape back, and final out projection
    v_out = self.v_quant(v_out)
    attn_output = torch.bmm(attn_weights, v_out)

    attn_output = (
        attn_output.view(batch_size, self.num_heads, seq_len, head_dim)
        .permute(2, 0, 1, 3)
        .reshape(seq_len, batch_size, embed_dim)
    )

    attn_output = self.out_proj(attn_output)
    return attn_output
