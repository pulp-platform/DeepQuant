# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from brevitas.nn.quant_mha import QuantMultiheadAttention
from torch import Tensor


def _mhaForwardImpl(
    self: QuantMultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    need_transpose_in: bool,
    need_transpose_out: bool,
) -> Tensor:
    """Core MHA forward implementation."""
    # FBRANCASI: Handle batch_first by transposing if needed
    if need_transpose_in:
        if key is value:
            if query is key:
                query = key = value = query.transpose(1, 0)
            else:
                query, key = [x.transpose(1, 0) for x in (query, key)]
                value = key
        else:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

    if self.in_proj is not None:
        # FBRANCASI: Handle packed projections (default case for models like ViT)
        # Only support self-attention where query == key == value
        if not (query is key and key is value):
            raise RuntimeError(
                "Packed in_proj is supported only for self-attention with k is v is q. Set packed_in_proj=False."
            )
        qkv = self.in_proj(query)
        qkv_tensor = qkv.value if hasattr(qkv, "value") else qkv
        qOut, kOut, vOut = qkv_tensor.chunk(3, dim=-1)
    else:
        q_result = self.q_proj(query)
        k_result = self.k_proj(key)
        v_result = self.v_proj(value)

        qOut = q_result.value if hasattr(q_result, "value") else q_result
        kOut = k_result.value if hasattr(k_result, "value") else k_result
        vOut = v_result.value if hasattr(v_result, "value") else v_result

    seqLen, batchSize, embedDim = qOut.shape
    headDim = embedDim // self.num_heads

    qOut = (
        qOut.contiguous()
        .view(seqLen, batchSize * self.num_heads, headDim)
        .transpose(0, 1)
    )
    kOut = (
        kOut.contiguous()
        .view(seqLen, batchSize * self.num_heads, headDim)
        .transpose(0, 1)
    )
    vOut = (
        vOut.contiguous()
        .view(seqLen, batchSize * self.num_heads, headDim)
        .transpose(0, 1)
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
        attnOutput.transpose(0, 1).contiguous().view(seqLen, batchSize, embedDim)
    )

    out_result = self.out_proj(attnOutput)
    attnOutput = out_result.value if hasattr(out_result, "value") else out_result

    if need_transpose_out:
        attnOutput = attnOutput.transpose(1, 0)

    return attnOutput


def mhaForwardBatchFirst(
    self: QuantMultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    need_weights: bool = True,
    **kwargs,
) -> Tuple[Tensor, Optional[Tensor]]:
    """MHA forward for batch_first=True."""
    attn_output = _mhaForwardImpl(
        self, query, key, value, need_transpose_in=True, need_transpose_out=True
    )
    return (attn_output, None)


def mhaForwardSeqFirst(
    self: QuantMultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    need_weights: bool = True,
    **kwargs,
) -> Tuple[Tensor, Optional[Tensor]]:
    """MHA forward for batch_first=False."""
    attn_output = _mhaForwardImpl(
        self, query, key, value, need_transpose_in=False, need_transpose_out=False
    )
    return (attn_output, None)


def mhaForward(
    self: QuantMultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    need_weights: bool = True,
    **kwargs,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Explicit, export-friendly MHA forward.

    This function will be replaced with the appropriate batch_first or seq_first version
    during module transformation based on the module's batch_first attribute.
    """
    # FBRANCASI: Appropriate version before tracing
    if self.batch_first:
        return mhaForwardBatchFirst(self, query, key, value, need_weights, **kwargs)
    else:
        return mhaForwardSeqFirst(self, query, key, value, need_weights, **kwargs)
