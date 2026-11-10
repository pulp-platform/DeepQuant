# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Lionnus Kesting <lkesting@iis.ee.ethz.ch>

"""
Operators of the Surya accelerator datapath, as FX-traceable leaf modules.

The NormQuant requantizes the INT32 accumulator of the CIM array to INT8.
NORM_MODE 4 keeps the result in INT32 and sends it to the tap, the path of
every non-linearity:

    acc32 -> NormQuant (INT32) -> Cast -> Pace -> Decast -> INT8
"""

from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

# NormQuant field limits.
NQ_SCALE_MAX = 2**15 - 1
NQ_SHIFT_MAX = 31
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1

# The approximations `setIdeal` switches off:
#   normquant   S16 * 2**-shift in place of the float multiplier
#   cast        the truncation to BF16 on the way into PACE
#   store       the truncation to BF16 of a tensor NORM_MODE 2 writes to memory
#   pace        the piecewise polynomial in place of the function
#   layernorm   the BF16 schedule in place of a float LayerNorm
#   softmax     the BF16 schedule in place of a float softmax
APPROXIMATIONS = ("normquant", "cast", "store", "pace", "layernorm", "softmax")

# BF16 has 1 implicit and 7 explicit mantissa bits. The datapath has no subnormals.
BF16_SIGNIFICAND_BITS = 8
BF16_MIN_NORMAL = 2.0**-126


def channelShape(values: Tensor, rank: int, channelDim: int) -> Tensor:
    """Reshape a per-channel vector so it broadcasts over a rank-N tensor."""
    shape = [1] * rank
    shape[channelDim] = -1
    return values.reshape(shape)


def _asAccumulator(acc: Tensor) -> Tensor:
    """Round to int64: a float32 convolution misses the integer by a few ULP."""
    if acc.dtype.is_floating_point:
        return torch.round(acc).to(torch.int64)
    return acc.to(torch.int64)


def roundToBf16(value: Tensor, export: bool = False) -> Tensor:
    """Round a float64 tensor to BF16, nearest with ties to even."""
    mag = value.abs()
    if export:
        # Opset 13 has no frexp. A log2 that is one octave off is next to a
        # power of two, which both grids hold.
        exponent = torch.floor(torch.log2(torch.where(mag > 0, mag, mag.new_ones(()))))
    else:
        exponent = (torch.frexp(mag)[1] - 1).to(torch.float64)
    exponent = exponent.clamp(min=-126.0) - (BF16_SIGNIFICAND_BITS - 1)
    # CUDA rounds `pow(2, n)` for some n, and ONNX has no Exp2.
    step = torch.pow(2.0, exponent) if export else torch.exp2(exponent)
    return torch.round(value / step) * step


def fmaBf16(a: Tensor, b: Tensor, c: Tensor, export: bool = False) -> Tensor:
    """`a * b + c` on BF16 operands in float64, rounded once, as the fpnew FMA does."""
    product = a * b
    total = product + c
    # An addend below the float64 resolution of the product still breaks a tie.
    lost = (total == product) & (c != 0)
    nudged = product * (1.0 + 2.0**-52 * torch.sign(c) * torch.sign(product))
    return roundToBf16(torch.where(lost, nudged, total), export)


class SuryaOperator(nn.Module):
    """One function in up to three forms.

    `exactForward` is the arithmetic the RTL performs. `exportMode` selects
    `exportForward`, the same function in ONNX opset-13 operators. `ideal`
    selects `idealForward`, the float function the arithmetic approximates.
    `approximation` is the name of the operator in `APPROXIMATIONS`, empty for
    an operator that approximates nothing.
    """

    approximation = ""

    def __init__(self) -> None:
        super().__init__()
        self.exportMode = False
        self.ideal = False

    def exactForward(self, *args: Tensor) -> Tensor:
        raise NotImplementedError

    def exportForward(self, *args: Tensor) -> Tensor:
        return self.exactForward(*args)

    def idealForward(self, *args: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, *args: Tensor) -> Tensor:
        if self.ideal:
            return self.idealForward(*args)
        if self.exportMode:
            return self.exportForward(*args)
        return self.exactForward(*args)


def setExportMode(module: nn.Module, enabled: bool = True) -> None:
    """Switch every Surya operator in a model between the two forward paths."""
    for sub in module.modules():
        if isinstance(sub, SuryaOperator):
            sub.exportMode = enabled


def setIdeal(module: nn.Module, names: Iterable[str] = ()) -> int:
    """Run the operators of the named approximations in their ideal form.

    Every other operator goes back to the RTL arithmetic. Returns how many
    operators run ideal.
    """
    unknown = sorted(set(names) - set(APPROXIMATIONS))
    if unknown:
        raise ValueError(f"unknown approximation {unknown}, known: {APPROXIMATIONS}")
    count = 0
    for sub in module.modules():
        if isinstance(sub, SuryaOperator):
            sub.ideal = sub.approximation in names
            count += sub.ideal
    return count


@torch.no_grad()
def assertExportMatchesExact(
    operator: SuryaOperator, *args: Tensor, name: Optional[str] = None
) -> None:
    """Raise unless the two forward paths of one operator agree bit for bit."""
    was = operator.exportMode
    try:
        operator.exportMode = False
        exact = operator(*args)
        operator.exportMode = True
        # An ONNX runtime computes on the CPU, where `pow(2, n)` is exact.
        exported = operator.cpu()(*(arg.cpu() for arg in args)).to(exact.device)
    finally:
        operator.exportMode = was
        operator.to(args[0].device)
    if not torch.equal(exact, exported):
        differing = int((exact != exported).sum())
        worst = float((exact - exported).abs().max())
        raise AssertionError(
            f"{name or type(operator).__name__}: the export arithmetic differs "
            f"from the RTL arithmetic on {differing} of {exact.numel()} values, "
            f"largest difference {worst}"
        )


class SuryaNormQuant(SuryaOperator):
    """`clamp((acc32 * S16 + B32) >>> shift)`, the requantization of the NQ unit.

    The shift is arithmetic, so it floors a negative value where a division
    truncates toward zero. `outputBits` is 8 on the normal path. NORM_MODE 4
    sets it to 32, which saturates to +/-(2**31 - 1). `multiplier` is the float
    M that S16 and shift encode.
    """

    approximation = "normquant"

    def __init__(
        self,
        s16: Tensor,
        b32: Tensor,
        shift: Tensor,
        signed: bool = True,
        outputBits: int = 8,
        rank: int = 4,
        channelDim: int = 1,
        cInit: Optional[Tensor] = None,
        multiplier: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        s16 = torch.as_tensor(s16).flatten().to(torch.int64)
        b32 = torch.as_tensor(b32).flatten().to(torch.int64)
        shift = torch.as_tensor(shift).flatten().to(torch.int64)
        if not (s16.shape == b32.shape == shift.shape):
            raise ValueError(
                f"S16, B32 and shift must have one entry per channel, got "
                f"{tuple(s16.shape)}, {tuple(b32.shape)}, {tuple(shift.shape)}"
            )
        if int(s16.abs().max()) > NQ_SCALE_MAX:
            raise ValueError(f"S16 does not fit 16 bits: max {int(s16.abs().max())}")
        if int(b32.min()) < INT32_MIN or int(b32.max()) > INT32_MAX:
            raise ValueError("B32 does not fit int32")
        if int(shift.min()) < 0 or int(shift.max()) > NQ_SHIFT_MAX:
            raise ValueError(f"shift must lie in 0..{NQ_SHIFT_MAX}")

        self.signed = signed
        self.outputBits = outputBits
        self.register_buffer("s16", channelShape(s16, rank, channelDim))
        self.register_buffer("b32", channelShape(b32, rank, channelDim))
        self.register_buffer("shift", channelShape(shift, rank, channelDim))
        self.register_buffer(
            "twoPowShift",
            channelShape(torch.exp2(shift.to(torch.float64)), rank, channelDim),
        )
        # The accumulator start value, which the array loads before the reduction.
        if cInit is not None:
            cInit = torch.as_tensor(cInit).flatten().to(torch.int64)
            if int(cInit.abs().max()) > INT32_MAX:
                raise ValueError("the accumulator start value does not fit int32")
            cInit = channelShape(cInit, rank, channelDim)
        self.register_buffer("cInit", cInit)
        if multiplier is None:
            realised = self.s16 / self.twoPowShift
        else:
            values = torch.as_tensor(multiplier).flatten().to(torch.float64)
            realised = channelShape(values, rank, channelDim)
        self.register_buffer("multiplier", realised)

    @property
    def outputDtype(self) -> torch.dtype:
        """float64 on the INT32 path: float32 cannot hold a 31-bit accumulator."""
        return torch.float64 if self.outputBits == 32 else torch.float32

    @property
    def limits(self) -> tuple:
        if self.outputBits == 32:
            return (-INT32_MAX, INT32_MAX)
        return (-128, 127) if self.signed else (0, 255)

    def _accumulator(self, acc: Tensor) -> Tensor:
        value = _asAccumulator(acc)
        return value if self.cInit is None else value + self.cInit

    def exactForward(self, acc: Tensor) -> Tensor:
        lo, hi = self.limits
        # int64 holds the 49-bit intermediate exactly. The shift is arithmetic.
        product = self._accumulator(acc) * self.s16 + self.b32
        return torch.clamp(product >> self.shift, lo, hi).to(self.outputDtype)

    def exportForward(self, acc: Tensor) -> Tensor:
        lo, hi = self.limits
        product = self._accumulator(acc) * self.s16 + self.b32
        # floor(x / 2**shift) is the arithmetic shift. A double holds the 49 bits.
        shifted = torch.floor(product.to(torch.float64) / self.twoPowShift)
        return torch.clamp(shifted, lo, hi).to(self.outputDtype)

    def idealForward(self, acc: Tensor) -> Tensor:
        lo, hi = self.limits
        # The offset as lowered, so the multiplier is the one difference.
        start = 0 if self.cInit is None else self.cInit * self.s16
        offset = (start + self.b32) / self.twoPowShift
        value = _asAccumulator(acc).to(torch.float64) * self.multiplier + offset
        return torch.clamp(torch.floor(value), lo, hi).to(self.outputDtype)


class SuryaCast(SuryaOperator):
    """INT32 -> BF16, truncating, representing `x * 2**-fpShift`.

    The hardware normalises with a leading zero count and has no round bit, so
    the magnitude only shrinks. `fpShift` is one byte for each channel and folds
    into the exponent.
    """

    approximation = "cast"

    def __init__(
        self, fpShift: Union[int, Tensor] = 0, rank: int = 4, channelDim: int = 1
    ) -> None:
        super().__init__()
        shift = torch.as_tensor(fpShift).flatten().to(torch.float64)
        self.register_buffer(
            "scale", channelShape(torch.exp2(-shift), rank, channelDim)
        )

    def exactForward(self, x: Tensor) -> Tensor:
        # The leading zero count of the RTL as a bit smear: `smeared` is 2**n - 1
        # for an n-bit magnitude, so the mask clears the bits below the mantissa.
        xi = x.to(torch.int64)
        mag = xi.abs() & INT32_MAX
        smeared = mag
        for step in (1, 2, 4, 8, 16, 32):
            smeared = smeared | (smeared >> step)
        mag = mag & ~(smeared >> BF16_SIGNIFICAND_BITS)
        value = torch.where(xi < 0, -mag, mag).to(torch.float64) * self.scale
        # The hardware has no subnormals and flushes below the smallest normal.
        value = torch.where(value.abs() < BF16_MIN_NORMAL, value.new_zeros(()), value)
        return value.to(torch.float32)

    def exportForward(self, x: Tensor) -> Tensor:
        # Opset 13 has no leading zero count, so the exponent comes from log2.
        # The constants are 0-dim: `ones_like` would store a full tensor in the file.
        v = x.to(torch.float64)
        mag = torch.abs(v)
        safe = torch.where(mag > 0, mag, mag.new_ones(()))
        # ONNX has no Exp2 operator at any opset.
        step = torch.pow(
            2.0, torch.floor(torch.log2(safe)) - float(BF16_SIGNIFICAND_BITS - 1)
        )
        truncated = torch.floor(safe / step) * step
        value = torch.sign(v) * torch.where(mag > 0, truncated, mag.new_zeros(()))
        value = value * self.scale
        value = torch.where(value.abs() < BF16_MIN_NORMAL, value.new_zeros(()), value)
        return value.to(torch.float32)

    def idealForward(self, x: Tensor) -> Tensor:
        return (x.to(torch.float64) * self.scale).to(torch.float32)


class SuryaDecast(SuryaOperator):
    """BF16 -> INT8: `round(x * 2**decShift)` with ties to even, then saturate."""

    def __init__(
        self,
        decShift: Union[int, Tensor],
        signed: bool = True,
        rank: int = 4,
        channelDim: int = 1,
    ) -> None:
        super().__init__()
        self.signed = signed
        shift = torch.as_tensor(decShift).flatten().to(torch.float64)
        self.register_buffer("scale", channelShape(torch.exp2(shift), rank, channelDim))

    def exactForward(self, x: Tensor) -> Tensor:
        lo, hi = (-128, 127) if self.signed else (0, 255)
        scaled = torch.round(x.to(torch.float64) * self.scale)
        return torch.clamp(scaled, lo, hi).to(torch.float32)


class SuryaResidual(SuryaOperator):
    """The accumulator start value, loaded from a BF16 tensor in memory.

    `ACCUM_INIT_MODE` 2 loads it with a decast of shift zero, so the writer
    stores the skip at the scale `a * s_x * s_w` of the reader's accumulator.
    The depthwise unit takes no residual.
    """

    def exactForward(self, acc: Tensor, skip: Tensor) -> Tensor:
        started = torch.round(skip.to(torch.float64)).clamp(-INT32_MAX, INT32_MAX)
        total = _asAccumulator(acc).to(torch.float64) + started
        return torch.clamp(total, -INT32_MAX, INT32_MAX)
