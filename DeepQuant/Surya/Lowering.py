# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Lionnus Kesting <lkesting@iis.ee.ethz.ch>

"""
Turn float quantization scales into the integer fields of the Surya NormQuant.

For each output channel c, with the input scale s_x, the weight scale s_w[c],
the output scale s_y, the bias, and the affine (a, b) that the path applies
before the next quantizer:

    M      = a[c] * s_x * s_w[c] / s_y
    shift  = clamp(floor(log2(32767 / M)), 0, 31)
    S16    = clamp(round(M * 2**shift), 1, 32767)
    B32    = round((a[c]*bias[c] + b[c]) / s_y * 2**shift)
    y      = clamp((conv(x, w) * S16 + B32) >>> shift)

The array loads the start value C_init into the accumulator before the
reduction. C_init takes the offset in accumulator LSBs and B32 the residue, so
B32 fits at the shift S16 wants:

    C_init = round(b / (a * s_x * s_w))       the offset, coarsely
    r      = b/s_y - C_init * M               the residue, |r| <= M/2
    B32    = round(r * 2**shift)              so |B32| <= S16/2 <= 16384
    y      = ((acc + C_init) * S16 + B32) >>> shift

With `useAccumulatorBias=False`, B32 carries the whole offset and limits the
shift.
"""

import dataclasses
from typing import Optional, Tuple

import numpy as np

from .Operators import (
    NQ_BIAS_MAX,
    NQ_BIAS_MIN,
    NQ_SCALE_MAX,
    NQ_SHIFT_MAX,
)


@dataclasses.dataclass
class RequantParams:
    """The NormQuant fields of one layer, one entry per output channel."""

    s16: np.ndarray  # int16
    b32: np.ndarray  # int32
    shift: np.ndarray  # uint8
    multiplier: np.ndarray  # the float M this approximates
    relError: float  # |S16*2**-shift - M| / M, worst channel
    cInit: Optional[np.ndarray] = None  # int32, the accumulator start value
    clampedChannels: int = 0  # channels whose multiplier has no encoding


def lowerRequant(
    sX: float,
    sW: np.ndarray,
    sY: float,
    bias: Optional[np.ndarray] = None,
    affine: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    useAccumulatorBias: bool = True,
    strict: bool = False,
) -> RequantParams:
    """Float scales -> the integer fields of the NormQuant.

    A multiplier with no encoding is clamped to the nearest one and counted in
    `clampedChannels`. `strict` raises on such a channel.
    """
    sW = np.asarray(sW, dtype=np.float64).reshape(-1)
    n = sW.size
    a = (
        np.ones(n)
        if affine is None
        else np.asarray(affine[0], dtype=np.float64).reshape(-1)
    )
    b = (
        np.zeros(n)
        if affine is None
        else np.asarray(affine[1], dtype=np.float64).reshape(-1)
    )
    if bias is not None:
        b = b + a * np.asarray(bias, dtype=np.float64).reshape(-1)

    m = a * sX * sW / sY
    if np.any(m <= 0):
        raise ValueError(
            "non-positive requant multiplier: S16 is positive, so a channel with "
            "a negative gain has to move its sign into the weight row first"
        )

    offsetOut = b / sY

    # 1. The shift S16 wants, and the multiplier it gives.
    with np.errstate(divide="ignore"):
        roomS16 = np.floor(np.log2(NQ_SCALE_MAX / m))
    shift = np.clip(np.nan_to_num(roomS16, neginf=0.0), 0, NQ_SHIFT_MAX)
    s16 = np.clip(np.round(m * np.exp2(shift)), 1, NQ_SCALE_MAX)

    # 2. The accumulator start value takes the offset.
    cInit = None
    residue = offsetOut
    if useAccumulatorBias:
        cInit = np.clip(
            np.round(offsetOut / (s16 * np.exp2(-shift))), NQ_BIAS_MIN + 1, NQ_BIAS_MAX
        )
        residue = offsetOut - cInit * (s16 * np.exp2(-shift))

    # 3. A channel with a clipped start value gets a smaller shift.
    need = np.abs(residue)
    roomB32 = np.full(n, float(NQ_SHIFT_MAX))
    live = need > 0
    roomB32[live] = np.floor(np.log2(NQ_BIAS_MAX / need[live]))
    reduced = np.clip(np.minimum(shift, roomB32), 0, NQ_SHIFT_MAX)
    if np.any(reduced != shift):
        shift = reduced
        s16 = np.clip(np.round(m * np.exp2(shift)), 1, NQ_SCALE_MAX)
        if useAccumulatorBias:
            cInit = np.clip(
                np.round(offsetOut / (s16 * np.exp2(-shift))),
                NQ_BIAS_MIN + 1,
                NQ_BIAS_MAX,
            )
            residue = offsetOut - cInit * (s16 * np.exp2(-shift))
    twoP = np.exp2(shift)
    b32 = np.clip(np.round(residue * twoP), NQ_BIAS_MIN, NQ_BIAS_MAX)

    # 4. The multipliers outside [1, 32767]/2**shift, which have no encoding.
    wanted = m * twoP
    clamped = (wanted < 1) | (wanted > NQ_SCALE_MAX)
    if strict and np.any(clamped):
        low = int((wanted < 1).sum())
        high = int((wanted > NQ_SCALE_MAX).sum())
        raise ValueError(
            f"the multiplier has no encoding on {int(clamped.sum())} channel(s): "
            f"{low} below the floor of 1 and {high} above the ceiling of "
            f"{NQ_SCALE_MAX}. Those channels are far outside the range the rest "
            "of the layer uses"
        )

    realised = s16 * np.exp2(-shift)
    return RequantParams(
        s16=s16.astype(np.int16),
        b32=b32.astype(np.int32),
        shift=shift.astype(np.uint8),
        multiplier=m,
        # Over the channels that have an encoding.
        relError=(
            0.0
            if bool(clamped.all())
            else float(np.max((np.abs(realised - m) / m)[~clamped]))
        ),
        cInit=None if cInit is None else cInit.astype(np.int32),
        clampedChannels=int(clamped.sum()),
    )


def moveSignIntoWeights(
    weightInt8: np.ndarray, a: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Make every per-channel gain positive by negating the weight rows.

    S16 is positive. The move is exact: a*acc == |a| * (-acc), and a
    narrow-range weight code negates without saturating.
    """
    negative = a < 0
    if not negative.any():
        return weightInt8, a
    if (weightInt8[negative] == -128).any():
        raise ValueError(
            "a weight code is -128 on a channel with a negative multiplier, so "
            "the sign cannot move into the weight row without saturating"
        )
    weightInt8 = weightInt8.copy()
    weightInt8[negative] = -weightInt8[negative]
    return weightInt8, np.abs(a)


def applyRequant(
    acc: np.ndarray,
    params: RequantParams,
    signed: bool = True,
    channelAxis: int = 1,
) -> np.ndarray:
    """Run the NormQuant chain on an INT32 accumulator, in numpy.

    `(acc * S16 + B32) >>> shift` with saturation. The shift is arithmetic, so it
    floors, and int64 holds the 49-bit intermediate.
    """
    shape = [1] * acc.ndim
    shape[channelAxis] = -1
    value = acc.astype(np.int64)
    if params.cInit is not None:
        value = value + params.cInit.astype(np.int64).reshape(shape)
    out = (
        value * params.s16.astype(np.int64).reshape(shape)
        + params.b32.astype(np.int64).reshape(shape)
    ) >> params.shift.astype(np.int64).reshape(shape)
    lo, hi = (-128, 127) if signed else (0, 255)
    return np.clip(out, lo, hi).astype(np.int8 if signed else np.uint8)
