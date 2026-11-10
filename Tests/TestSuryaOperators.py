# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Lionnus Kesting <lkesting@iis.ee.ethz.ch>

"""
Tests for the Surya operators.

Each operator has two forward paths: the arithmetic the RTL performs, and the
same function in ONNX opset-13 operators. The two must agree bit for bit.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from DeepQuant.Surya import (
    NQ_SCALE_MAX,
    SuryaCast,
    SuryaDecast,
    SuryaNormQuant,
    SuryaPwpa,
    SuryaResidual,
    SuryaSoftmax,
    SuryaTap,
    applyRequant,
    assertExportMatchesExact,
    buildPwpa,
    buildTable,
    lowerRequant,
    moveSignIntoWeights,
    scaleTable,
    setIdeal,
)
from DeepQuant.Surya.Operators import roundToBf16
from DeepQuant.Surya.Pace import floatToBf16

CHANNELS = 16


def makeParams(seed: int = 0):
    rng = np.random.RandomState(seed)
    return lowerRequant(
        sX=0.031,
        sW=np.abs(rng.randn(CHANNELS)) * 1e-3 + 1e-4,
        sY=0.047,
        bias=rng.randn(CHANNELS) * 0.1,
    )


@pytest.mark.SingleLayerTests
def deepQuantTestSuryaNormQuantMatchesNumpy() -> None:
    """The module and the numpy reference compute the same codes."""
    torch.manual_seed(0)
    params = makeParams()
    acc = torch.randint(-(2**20), 2**20, (2, CHANNELS, 5, 5)).to(torch.float32)

    module = SuryaNormQuant(params.s16, params.b32, params.shift, cInit=params.cInit)
    reference = applyRequant(acc.numpy().astype(np.int64), params)
    assert np.array_equal(reference, module(acc).numpy().astype(np.int64))


@pytest.mark.SingleLayerTests
def deepQuantTestSuryaNormQuantExportMatchesExact() -> None:
    """The opset-13 form of the NormQuant equals the arithmetic shift form."""
    torch.manual_seed(1)
    params = makeParams(1)
    module = SuryaNormQuant(params.s16, params.b32, params.shift, cInit=params.cInit)
    acc = torch.randint(-(2**24), 2**24, (4, CHANNELS, 3, 3)).to(torch.float32)
    assertExportMatchesExact(module, acc)


@pytest.mark.SingleLayerTests
def deepQuantTestSuryaNormQuantRejectsOversizedFields() -> None:
    """A field that does not fit raises."""
    ones = np.ones(4, dtype=np.int64)
    with pytest.raises(ValueError, match="S16"):
        SuryaNormQuant(ones * 40000, ones * 0, ones * 0)
    with pytest.raises(ValueError, match="shift"):
        SuryaNormQuant(ones, ones * 0, ones * 40)


@pytest.mark.SingleLayerTests
@pytest.mark.parametrize("shift", [0, 7, 23])
def deepQuantTestSuryaCastExportMatchesExact(shift: int) -> None:
    """The logarithm form of the truncating cast equals the leading-zero form."""
    torch.manual_seed(2)
    values = torch.randint(-(2**31) + 1, 2**31 - 1, (4000,)).to(torch.float64)
    assertExportMatchesExact(SuryaCast(shift, rank=1, channelDim=0), values)


@pytest.mark.SingleLayerTests
def deepQuantTestSuryaCastTruncatesAndNeverRounds() -> None:
    """The cast only ever shrinks a magnitude."""
    torch.manual_seed(3)
    values = torch.randint(1, 2**31 - 1, (4000,)).to(torch.float64)
    cast = SuryaCast(0, rank=1, channelDim=0)
    assert bool((cast(values).to(torch.float64) <= values).all())


@pytest.mark.SingleLayerTests
def deepQuantTestSuryaDecastRoundsHalfToEven() -> None:
    values = torch.tensor([1.6, -1.6, 0.5, 1.5, -2.5, 300.0, -300.0])
    out = SuryaDecast(0, rank=1, channelDim=0)(values)
    assert torch.equal(out, torch.tensor([2.0, -2.0, 0.0, 2.0, -2.0, 127.0, -128.0]))


@pytest.mark.SingleLayerTests
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def deepQuantTestBf16RoundingIsTheSameOnTheGpu() -> None:
    # Every value is a tie between two BF16 neighbours.
    mantissa = torch.arange(257, 512, 2, dtype=torch.float64)[:, None]
    ties = mantissa * torch.exp2(torch.arange(-40, 40, dtype=torch.float64))
    assert torch.equal(roundToBf16(ties.cuda()).cpu(), roundToBf16(ties))


@pytest.mark.SingleLayerTests
def deepQuantTestSuryaResidualAddsTheRoundedSkip() -> None:
    """The accumulator start value, rounded from the BF16 tensor in memory."""
    torch.manual_seed(5)
    acc = torch.randint(-(2**20), 2**20, (100,)).float()
    skip = (torch.randn(100) * 2.0**4).to(torch.bfloat16).to(torch.float32)
    out = SuryaResidual()(acc, skip)
    assert torch.equal(out, acc.double() + torch.round(skip.double()))


@pytest.mark.SingleLayerTests
def deepQuantTestSetIdealSwitchesOneApproximation() -> None:
    """`setIdeal` runs the named operators in float and the others as the RTL."""
    torch.manual_seed(6)
    values = torch.randint(-(2**16), 2**16, (1, 4, 3, 3)).float()
    tap = SuryaTap(
        buildPwpa(nn.SiLU(), gain=1.3), fpShift=8, decShift=2, rank=4, channelDim=1
    )
    exact = tap(values)

    assert setIdeal(tap, ("pace",)) == 1
    bf16 = tap.cast(values)
    assert torch.equal(tap.pace(bf16), F.silu(bf16) * tap.pace.gain)
    assert float((tap(values) - exact).abs().max()) <= 1

    assert setIdeal(tap, ("cast",)) == 1
    assert torch.equal(tap.cast(values), values * 2.0**-8)

    assert setIdeal(tap) == 0
    assert torch.equal(tap(values), exact)
    with pytest.raises(ValueError, match="unknown approximation"):
        setIdeal(tap, ("table",))


@pytest.mark.SingleLayerTests
def deepQuantTestIdealNormQuantUsesTheFloatMultiplier() -> None:
    """The ideal NormQuant differs from the RTL one by the multiplier alone."""
    torch.manual_seed(9)
    params = makeParams()
    acc = torch.randint(-(2**20), 2**20, (2, CHANNELS, 5, 5)).to(torch.float32)
    fields = (params.s16, params.b32, params.shift)

    encoded = SuryaNormQuant(*fields, cInit=params.cInit)
    exact = encoded(acc)
    setIdeal(encoded, ("normquant",))
    assert torch.equal(encoded(acc), exact)

    ideal = SuryaNormQuant(*fields, cInit=params.cInit, multiplier=params.multiplier)
    setIdeal(ideal, ("normquant",))
    assert float((ideal(acc) - exact).abs().max()) <= 1


@pytest.mark.SingleLayerTests
def deepQuantTestSuryaTapRunsThePaceTable() -> None:
    """NORM_MODE 4 end to end, with the table the hardware holds.

    Every channel has its own decast shift.
    """
    torch.manual_seed(7)
    values = torch.randint(-(2**20), 2**20, (2, 4, 5, 5)).float()
    pace = buildPwpa(nn.SiLU(), gain=1.19)
    decShift = np.array([3, 4, 5, 6])
    tap = SuryaTap(pace, fpShift=16, decShift=decShift, rank=4, channelDim=1)
    out = tap(values)
    assert out.shape == values.shape
    assert bool(((out >= -128) & (out <= 127)).all())
    assert bool((out == out.round()).all())
    # The cast and the PACE table each have an export form.
    assertExportMatchesExact(tap.cast, values)
    assertExportMatchesExact(tap.pace, tap.cast(values))


@pytest.mark.SingleLayerTests
def deepQuantTestPwpaExportCountsBreakpoints() -> None:
    """The export counts breakpoints, because opset 13 has no SearchSorted.

    Both paths pick the same partition, also on a breakpoint and far outside the
    fit range.
    """
    pwpa = buildPwpa(nn.SiLU(), gain=1.0)
    table = buildTable("silu", -8.0, 8.0)
    edges = torch.as_tensor(table.breakpoints, dtype=torch.float32)
    torch.manual_seed(8)
    values = torch.cat(
        [
            edges,
            edges - 1e-3,
            edges + 1e-3,
            torch.randn(4000) * 12,
            torch.tensor([-310.0, -8.0, 0.0, 8.0, 310.0]),
        ]
    )
    assertExportMatchesExact(pwpa, values, name="SuryaPwpa")


@pytest.mark.SingleLayerTests
def deepQuantTestMoveSignIntoWeightsIsExact() -> None:
    """S16 is positive, so a negative gain moves its sign into the weight row."""
    weights = np.array([[1, -2], [3, 4]], dtype=np.int8)
    gains = np.array([-0.5, 2.0])
    moved, positive = moveSignIntoWeights(weights, gains)
    assert np.array_equal(moved, np.array([[-1, 2], [3, 4]], dtype=np.int8))
    assert np.array_equal(positive, np.array([0.5, 2.0]))
    # -128 has no positive counterpart in int8, so it must raise.
    with pytest.raises(ValueError, match="-128"):
        moveSignIntoWeights(np.array([[-128]], dtype=np.int8), np.array([-1.0]))


@pytest.mark.SingleLayerTests
def deepQuantTestLowerRequantHandlesOutOfRangeScales() -> None:
    """A multiplier with no encoding is clamped and counted, or refused by `strict`."""
    sW = np.full(4, 1e-3)
    # An output scale far too small: the multiplier needs more than 15 bits.
    high = lowerRequant(sX=1.0, sW=sW, sY=1e-9)
    assert high.clampedChannels == 4
    assert int(high.s16.max()) == NQ_SCALE_MAX
    # An output scale far too large: the multiplier is below 1 even at shift 31.
    low = lowerRequant(sX=1.0, sW=sW, sY=1e7)
    assert low.clampedChannels == 4
    assert int(low.s16.max()) == 1

    for sY in (1e-9, 1e7):
        with pytest.raises(ValueError, match="no encoding"):
            lowerRequant(sX=1.0, sW=sW, sY=sY, strict=True)

    # A layer whose channels all fit is not flagged.
    assert makeParams().clampedChannels == 0


# ---------------------------------------------------------------------------
# The PACE unit
# ---------------------------------------------------------------------------


@pytest.mark.SingleLayerTests
@pytest.mark.parametrize("function", ["silu", "gelu", "sigmoid", "tanh"])
def deepQuantTestPaceTableHoldsBf16(function: str) -> None:
    """A table holds 31 breakpoints and 32 coefficient pairs, all BF16 values."""
    table = buildTable(function, -8.0, 8.0)
    assert np.array_equal(table.coefficients, floatToBf16(table.coefficients))
    assert np.array_equal(table.breakpoints, floatToBf16(table.breakpoints))
    assert table.packed().size == 31 + 32 * 2


@pytest.mark.SingleLayerTests
def deepQuantTestPaceZeroesPartitionZero() -> None:
    """Partition 0 is zero for SiLU, which tends to zero below the fit range.

    A fitted line for partition 0 extrapolates to about +0.88 at -310.
    """
    fitted = SuryaPwpa(buildTable("silu", -8.0, 8.0, zeroBelow=False))
    zeroed = SuryaPwpa(buildTable("silu", -8.0, 8.0))
    tail = torch.tensor([-310.0, -100.0, -20.0])
    assert float(fitted(tail).abs().max()) > 0.5
    assert float(zeroed(tail).abs().max()) == 0.0


@pytest.mark.SingleLayerTests
def deepQuantTestPaceGainFoldsIntoCoefficients() -> None:
    """alpha*f is another degree-1 polynomial, so the gain folds into the table.

    Each scaled coefficient rounds to BF16, and so does the result. One BF16
    step is 2**-9 relative, and three steps cover both roundings and the sum.
    """
    for alpha in (0.7913, 1.0, 1.2649, 1.4142):
        plain = SuryaPwpa(buildTable("silu", -8.0, 8.0))
        folded = SuryaPwpa(scaleTable(buildTable("silu", -8.0, 8.0), alpha))
        x = torch.randn(50000, generator=torch.Generator().manual_seed(9)) * 4
        x = x.to(torch.bfloat16).to(torch.float32)
        want = plain(x) * alpha
        got = folded(x)
        assert float((got - want).abs().max()) < 3 * 2.0**-9 * float(want.abs().max())


# ---------------------------------------------------------------------------
# The accumulator start value
# ---------------------------------------------------------------------------


@pytest.mark.SingleLayerTests
def deepQuantTestAccumulatorBiasKeepsTheMultiplierSharp() -> None:
    """The start value takes the offset, so the shift is the one S16 wants."""
    rng = np.random.RandomState(11)
    sW = np.abs(rng.randn(CHANNELS)) * 1e-3 + 1e-4
    bias = rng.randn(CHANNELS) * 500.0  # large enough to squeeze B32
    withStart = lowerRequant(0.031, sW, 0.047, bias=bias, useAccumulatorBias=True)
    withoutStart = lowerRequant(0.031, sW, 0.047, bias=bias, useAccumulatorBias=False)
    assert withStart.relError < withoutStart.relError
    assert int(withStart.s16.min()) >= NQ_SCALE_MAX // 2
    assert withStart.cInit is not None


@pytest.mark.SingleLayerTests
def deepQuantTestAccumulatorBiasStaysWithinOneCode() -> None:
    """`(acc + C_init) * S16 + B32 >> shift` reproduces the float requantization."""
    rng = np.random.RandomState(12)
    sX, sY = 0.031, 0.047
    sW = np.abs(rng.randn(CHANNELS)) * 1e-3 + 1e-4
    bias = rng.randn(CHANNELS) * 500.0
    params = lowerRequant(sX, sW, sY, bias=bias)
    acc = rng.randint(-(2**18), 2**18, size=(64, CHANNELS)).astype(np.int64)
    got = applyRequant(acc, params, channelAxis=1)
    want = np.clip(np.floor((acc * sX * sW + bias) / sY), -128, 127).astype(np.int8)
    # One code of slack: S16 and shift approximate the float multiplier.
    assert int((np.abs(got.astype(int) - want.astype(int)) > 1).sum()) == 0


@pytest.mark.SingleLayerTests
def deepQuantTestNormQuantAppliesTheStartValue() -> None:
    ones = np.ones(4, dtype=np.int64)
    plain = SuryaNormQuant(ones * 1024, ones * 0, ones * 10, rank=1, channelDim=0)
    started = SuryaNormQuant(
        ones * 1024, ones * 0, ones * 10, rank=1, channelDim=0, cInit=ones * 7
    )
    acc = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert torch.equal(plain(acc), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(started(acc), torch.tensor([7.0, 8.0, 9.0, 10.0]))
    assertExportMatchesExact(started, acc)


@pytest.mark.SingleLayerTests
def deepQuantTestInt32NormQuantReturnsFloat64() -> None:
    """float32 cannot hold a 31-bit accumulator, so the INT32 path must not use it."""
    ones = np.ones(2, dtype=np.int64)
    node = SuryaNormQuant(
        ones * 30000, ones * 0, ones * 0, outputBits=32, rank=1, channelDim=0
    )
    out = node(torch.tensor([70000.0, -70000.0]))
    assert out.dtype == torch.float64
    assert out.tolist() == [2100000000.0, -2100000000.0]


@pytest.mark.SingleLayerTests
@pytest.mark.parametrize("columns", [16, 64, 256])
def deepQuantTestSuryaSoftmaxTracksTheFloatSoftmax(columns: int) -> None:
    """The BF16 steps cost under 0.025 of a probability, and exp is zero at xMin."""
    torch.manual_seed(10)
    x = torch.randn(2, 1, 4, columns) * 3.0
    out = SuryaSoftmax()(x)
    assert float((out - F.softmax(x, dim=-1)).abs().max()) < 0.025
    assert SuryaSoftmax()(torch.tensor([0.0, -11.0, -12.0])).tolist() == [1, 0, 0]


@pytest.mark.SingleLayerTests
def deepQuantTestSuryaSoftmaxIdealIsTheFloatSoftmax() -> None:
    torch.manual_seed(11)
    x = torch.randn(4, 32)
    softmax = SuryaSoftmax()
    assert setIdeal(softmax, ("softmax",)) == 1
    assert torch.equal(softmax(x), F.softmax(x, dim=-1))


@pytest.mark.SingleLayerTests
def deepQuantTestSuryaSoftmaxInt8StoreIsTheDecast() -> None:
    """NORM_MODE 4 is a NormQuant of S16 1, so the decast of each row decides."""
    torch.manual_seed(12)
    x = torch.randn(3, 40) * 2.0
    decShift = np.array([7, 8, 9])
    ones = np.ones(3, dtype=np.int64)
    unit = SuryaNormQuant(ones, ones * 0, ones * 0, rank=2, channelDim=0)
    softmax = SuryaSoftmax(decShift=decShift, normQuant=unit)
    decast = SuryaDecast(decShift, rank=2, channelDim=0)
    assert torch.equal(softmax(x), decast(SuryaSoftmax()(x)))
    setIdeal(softmax, ("softmax",))
    assert torch.equal(softmax(x), decast(F.softmax(x, dim=-1)))
