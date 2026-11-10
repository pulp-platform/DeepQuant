# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Lionnus Kesting <lkesting@iis.ee.ethz.ch>

"""
The PACE LayerNorm: the BF16 schedule the sequencer runs on each row.

    mean = sum(x) * (1/C)
    d    = x - mean
    var  = sum(d*d) * (1/C)
    rstd = rsqrt(var + eps)
    y    = (x*rstd - mean*rstd) * gamma

Each step is one FMA with one rounding to BF16. The hardware has no beta, so
beta is an Add behind the node.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from onnx import TensorProto
from torch import Tensor

from .Kernels import SURYA_DOMAIN
from .Operators import SuryaOperator, fmaBf16, roundToBf16
from .Pace import SuryaPwpa, buildTable, floatToBf16

# Both sums run on this many interleaved partial sums, one for each FMA stage.
LN_PARTIALS = 3


class SuryaLayerNormFunction(torch.autograd.Function):
    """Name the schedule as one node in the `surya` domain. `out` passes through."""

    @staticmethod
    def forward(ctx, out, x, gamma, epsilon):
        return out

    @staticmethod
    def symbolic(g, out, x, gamma, epsilon):
        node = g.op(
            f"{SURYA_DOMAIN}::LayerNorm",
            g.op("Cast", x, to_i=TensorProto.BFLOAT16),
            g.op("Cast", gamma, to_i=TensorProto.BFLOAT16),
            epsilon_f=epsilon,
        )
        node.setType(out.type().with_dtype(torch.bfloat16))
        return g.op("Cast", node, to_i=TensorProto.FLOAT)


class SuryaLayerNorm(SuryaOperator):
    """LayerNorm over the last dimension, BF16 in and BF16 out."""

    approximation = "layernorm"

    def __init__(
        self, gamma: Tensor, beta: Optional[Tensor] = None, eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.rsqrt = SuryaPwpa(buildTable("rsqrt", 1.0, 4.0))
        self.register_buffer(
            "gamma", gamma.detach().to(torch.bfloat16).to(torch.float32)
        )
        self.register_buffer("beta", None if beta is None else beta.detach().clone())

    @staticmethod
    def _sum(a: Tensor, b: Tensor) -> Tensor:
        """`sum(a * b)` over the last dimension, in the order the sequencer adds."""
        pad = -a.shape[-1] % LN_PARTIALS
        a = F.pad(a, (0, pad)).unflatten(-1, (-1, LN_PARTIALS))
        b = F.pad(b, (0, pad)).unflatten(-1, (-1, LN_PARTIALS))
        partials = torch.zeros_like(a[..., 0, :])
        for step in range(a.shape[-2]):
            partials = fmaBf16(a[..., step, :], b[..., step, :], partials)
        total = partials[..., 0]
        for index in range(1, LN_PARTIALS):
            total = fmaBf16(torch.ones_like(total), partials[..., index], total)
        return total.unsqueeze(-1)

    def _rsqrt(self, value: Tensor) -> Tensor:
        """PACE on [1, 4): the exponent parity moves into the reduced operand."""
        mantissa, exponent = torch.frexp(value)
        exponent = exponent - 1
        parity = exponent & 1
        approx = self.rsqrt(mantissa * torch.exp2(parity + 1.0)).to(torch.float64)
        # `ldexp` is a `pow(2, n)`, which CUDA rounds for some n.
        return approx * torch.exp2(-((exponent - parity) >> 1).to(torch.float64))

    def _schedule(self, values: Tensor) -> Tensor:
        x = roundToBf16(values.to(torch.float64))
        ones = torch.ones_like(x)
        recip = float(floatToBf16(1.0 / x.shape[-1]))
        eps = float(floatToBf16(self.eps))
        mean = roundToBf16(self._sum(ones, x) * recip)
        d = fmaBf16(ones, x, -mean)
        var = roundToBf16(self._sum(d, d) * recip)
        rstd = self._rsqrt(fmaBf16(ones[..., :1], var, torch.full_like(var, eps)))
        xhat = fmaBf16(x, rstd, roundToBf16(-mean * rstd))
        return roundToBf16(xhat * self.gamma).to(values.dtype)

    def exactForward(self, x: Tensor) -> Tensor:
        out = self._schedule(x)
        return out if self.beta is None else out + self.beta

    def exportForward(self, x: Tensor) -> Tensor:
        # The schedule has no opset 13 form, so the tracer sees the node only.
        out = x if torch.jit.is_tracing() else self._schedule(x)
        out = SuryaLayerNormFunction.apply(out, x, self.gamma, self.eps)
        return out if self.beta is None else out + self.beta

    def idealForward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x, x.shape[-1:], self.gamma.to(x.dtype), self.beta, self.eps
        )
