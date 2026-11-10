# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Lionnus Kesting <lkesting@iis.ee.ethz.ch>

"""
The PACE unit: a piecewise polynomial in BF16, and the fit that fills its memory.

The PACE memory holds the breakpoints in breadth-first tree order, then the
coefficients of each partition, constant term first. Every value is BF16.

The decast after PACE is `round(g * 2**dec_fp_shift)`, so a table that holds
`gain * f` gives the output scale `2**-dec_fp_shift / gain`.
"""

import collections
import dataclasses
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .Operators import SuryaOperator, fmaBf16

# For each name: the function the fit samples, and its float form in torch.
FUNCTIONS = {
    "silu": (lambda x: x / (1.0 + np.exp(-x)), F.silu),
    "gelu": (
        lambda x: 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0))),
        F.gelu,
    ),
    "exp": (np.exp, torch.exp),
    "inv": (lambda x: 1.0 / x, torch.reciprocal),
    "rsqrt": (lambda x: 1.0 / np.sqrt(x), torch.rsqrt),
    "sigmoid": (lambda x: 1.0 / (1.0 + np.exp(-x)), torch.sigmoid),
    "tanh": (np.tanh, torch.tanh),
}

# These vanish at -inf, so partition 0 is a zero guard below xMin.
ZERO_BELOW_FUNCTIONS = ("silu", "gelu", "exp", "sigmoid")

PACE_PARTITIONS = 32
PACE_DEGREE = 1

PACE_NAMES = {
    "SiLU": "silu",
    "GELU": "gelu",
    "GELUActivation": "gelu",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
}


def floatToBf16(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 and back, nearest with ties to even."""
    fp32 = np.asarray(values, dtype=np.float32)
    bits = fp32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    bf16 = (rounded & np.uint32(0xFFFF0000)).view(np.float32)
    return np.where(np.isnan(fp32), fp32, bf16).astype(np.float32)


def bstOrder(count: int) -> np.ndarray:
    """The sorted index each memory slot holds, breadth first over the search tree."""
    out: list = []
    queue: collections.deque = collections.deque([(0, count - 1)])
    while queue:
        lo, hi = queue.popleft()
        if lo > hi:
            continue
        mid = (lo + hi) // 2
        out.append(mid)
        queue.append((lo, mid - 1))
        queue.append((mid + 1, hi))
    return np.asarray(out, dtype=np.int64)


@dataclasses.dataclass
class PaceTable:
    """One image of the PACE memory, in real numbers."""

    functionName: str
    xMin: float
    xMax: float
    breakpoints: np.ndarray  # sorted, [partitions - 1]
    coefficients: np.ndarray  # [partitions, degree + 1]
    gain: float = 1.0

    @property
    def partitions(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def degree(self) -> int:
        return int(self.coefficients.shape[1]) - 1

    def packed(self) -> np.ndarray:
        """The values in memory order."""
        order = bstOrder(self.breakpoints.size)
        return np.concatenate((self.breakpoints[order], self.coefficients.reshape(-1)))

    @classmethod
    def fromPacked(
        cls,
        values: np.ndarray,
        functionName: str,
        xMin: float,
        xMax: float,
        gain: float = 1.0,
        degree: int = PACE_DEGREE,
    ) -> "PaceTable":
        """The inverse of `packed`."""
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        partitions, rest = divmod(flat.size + 1, degree + 2)
        if rest or partitions < 2:
            raise ValueError(f"{flat.size} values are no table of degree {degree}")
        breakpoints = np.empty(partitions - 1, dtype=np.float32)
        breakpoints[bstOrder(partitions - 1)] = flat[: partitions - 1]
        return cls(
            functionName=functionName,
            xMin=float(xMin),
            xMax=float(xMax),
            breakpoints=breakpoints,
            coefficients=flat[partitions - 1 :].reshape(partitions, degree + 1).copy(),
            gain=float(gain),
        )


def buildTable(
    functionName: str,
    xMin: float = -8.0,
    xMax: float = 8.0,
    zeroBelow: Optional[bool] = None,
) -> PaceTable:
    """Least-squares fit of one polynomial for each partition, rounded to BF16.

    `zeroBelow` overrides the zero guard, which otherwise follows
    `ZERO_BELOW_FUNCTIONS`. A range too narrow to reach the tail of the function
    wants it off.
    """
    if not xMin < xMax:
        raise ValueError("xMin must be smaller than xMax")

    fit = FUNCTIONS[functionName][0]
    guard = functionName in ZERO_BELOW_FUNCTIONS if zeroBelow is None else zeroBelow
    edges = np.linspace(xMin, xMax, PACE_PARTITIONS - guard + 1, dtype=np.float64)
    if guard:
        edges = np.concatenate(([xMin - (edges[1] - edges[0])], edges))
    coefficients = np.zeros((PACE_PARTITIONS, PACE_DEGREE + 1), dtype=np.float32)
    for part in range(int(guard), PACE_PARTITIONS):
        xs = np.linspace(edges[part], edges[part + 1], 256, dtype=np.float64)
        coefficients[part] = np.polyfit(xs, fit(xs), PACE_DEGREE)[::-1]
    return PaceTable(
        functionName=functionName,
        xMin=xMin,
        xMax=xMax,
        breakpoints=floatToBf16(edges[1:-1]),
        coefficients=floatToBf16(coefficients),
    )


def scaleTable(table: PaceTable, gain: float) -> PaceTable:
    """Fold the gain into the coefficients."""
    coefficients = floatToBf16(table.coefficients.astype(np.float64) * float(gain))
    return dataclasses.replace(table, coefficients=coefficients, gain=float(gain))


class SuryaPwpa(SuryaOperator):
    """The PACE unit, evaluating a `PaceTable`: Horner with one FMA for each step."""

    approximation = "pace"

    def __init__(self, table: PaceTable) -> None:
        super().__init__()
        self.table = table
        self.functionName = table.functionName
        self.gain = float(table.gain)
        self.register_buffer(
            "breakpoints", torch.as_tensor(table.breakpoints, dtype=torch.float64)
        )
        self.register_buffer(
            "coefficients", torch.as_tensor(table.coefficients, dtype=torch.float64)
        )

    def _evaluate(self, flat: Tensor, parts: Tensor) -> Tensor:
        picked = self.coefficients.index_select(0, parts)
        acc = picked[:, -1]
        for index in range(self.table.degree - 1, -1, -1):
            acc = fmaBf16(acc, flat, picked[:, index], self.exportMode)
        return acc.to(torch.float32)

    def exactForward(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1).to(torch.float64)
        # A value on a breakpoint takes the partition below it.
        parts = torch.searchsorted(self.breakpoints, flat)
        return self._evaluate(flat, parts).reshape(x.shape)

    def exportForward(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1).to(torch.float64)
        # Opset 13 has no SearchSorted, so count the breakpoints below the value.
        parts = (flat > self.breakpoints[0]).to(torch.int64)
        for index in range(1, self.breakpoints.numel()):
            parts = parts + (flat > self.breakpoints[index]).to(torch.int64)
        return self._evaluate(flat, parts).reshape(x.shape)

    def idealForward(self, x: Tensor) -> Tensor:
        return (FUNCTIONS[self.functionName][1](x) * self.gain).to(torch.float32)


def buildPwpa(
    fn: nn.Module,
    xMin: float = -8.0,
    xMax: float = 8.0,
    gain: float = 1.0,
    zeroBelow: Optional[bool] = None,
) -> Optional[SuryaPwpa]:
    """A PACE unit for a PyTorch activation, or None when PACE has no fit for it."""
    name = PACE_NAMES.get(type(fn).__name__)
    if name is None:
        return None
    return SuryaPwpa(
        scaleTable(buildTable(name, xMin, xMax, zeroBelow=zeroBelow), gain)
    )
