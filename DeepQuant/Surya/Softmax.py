# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Lionnus Kesting <lkesting@iis.ee.ethz.ch>

"""
The softmax job of the PACE sequencer, over the last dimension.

    xmax = max(x)         the software passes it in BF16
    e    = exp(x - xmax)  PACE, zero at and below xMin
    s    = sum(e)         three interleaved partial sums, as the LayerNorm
    r    = 1 / s          PACE on the mantissa in [1, 2), the exponent negated
    y    = e * r          the BF16 result, NORM_MODE 2

Each step rounds to BF16. NORM_MODE 0 and 4 store INT8: the decast
`round(y * 2**decShift)` to INT32, then the NormQuant of the row. NORM_MODE 4 is
the NormQuant with S16 1, B32 0 and shift 0.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .LayerNorm import SuryaLayerNorm
from .Operators import (
    INT32_MAX,
    SuryaNormQuant,
    SuryaOperator,
    channelShape,
    roundToBf16,
)
from .Pace import SuryaPwpa, buildTable


class SuryaSoftmax(SuryaOperator):
    """Softmax over the last dimension, BF16 in and BF16 out.

    With `normQuant` the output is INT8. `decShift` and `normQuant` hold one
    entry for each row of a [rows, cols] tile.
    """

    approximation = "softmax"

    def __init__(
        self,
        xMin: float = -11.0,
        decShift: Optional[Tensor] = None,
        normQuant: Optional[SuryaNormQuant] = None,
    ) -> None:
        super().__init__()
        self.exp = SuryaPwpa(buildTable("exp", xMin, 0.0))
        self.inv = SuryaPwpa(buildTable("inv", 1.0, 2.0))
        self.normQuant = normQuant
        if normQuant is not None:
            shift = torch.as_tensor(decShift).flatten().to(torch.float64)
            self.register_buffer("decScale", channelShape(torch.exp2(shift), 2, 0))

    def _reciprocal(self, s: Tensor) -> Tensor:
        mantissa, exponent = torch.frexp(s)
        approx = self.inv(2.0 * mantissa).to(torch.float64)
        return approx * torch.exp2(1.0 - exponent.to(torch.float64))

    def _store(self, y: Tensor, dtype: torch.dtype) -> Tensor:
        if self.normQuant is None:
            return y.to(dtype)
        return self.normQuant(
            torch.round(y * self.decScale).clamp(-INT32_MAX, INT32_MAX)
        )

    def exactForward(self, x: Tensor) -> Tensor:
        logits = roundToBf16(x.to(torch.float64))
        shifted = roundToBf16(logits - logits.amax(-1, keepdim=True))
        e = self.exp(shifted).to(torch.float64)
        s = SuryaLayerNorm._sum(torch.ones_like(e), e)
        return self._store(roundToBf16(e * self._reciprocal(s)), x.dtype)

    def idealForward(self, x: Tensor) -> Tensor:
        return self._store(F.softmax(x, dim=-1), x.dtype)
