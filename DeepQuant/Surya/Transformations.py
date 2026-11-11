# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Lionnus Kesting <lkesting@iis.ee.ethz.ch>

"""
Transformation passes that put a Brevitas model on the Surya datapath.

`SuryaComputeTransformation` installs the integer steps of the hardware on a
compute layer:

    codes = quantInput / s_x                     INT8
    acc   = conv(codes, weightInt8)              INT32
    out   = NormQuant(acc)                       INT8
    value = out * s_y                            back to the float view

A layer that feeds a non-linearity takes NORM_MODE 4, the tail in `SuryaTapSpec`:

    out   = NormQuant(acc)                       INT32, no INT8 clamp
    out   = cast -> PACE -> decast               INT8
    value = out * (2**-dec_fp_shift / gain)      back to the float view

The caller supplies one `SuryaLayerSpec` for each layer, keyed by module path.
"""

import dataclasses
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer
from torch import Tensor

from ..CustomForwards.Common import UnpackQuantTensor
from ..CustomForwards.Linear import InnerForwardImplWrapperLinear
from ..CustomTracer import CustomBrevitasTracer
from ..Injects.Base import TransformationPass
from .Kernels import SuryaKernel, SuryaRequantPoint
from .Lowering import RequantParams
from .Operators import (
    SuryaCast,
    SuryaDecast,
    SuryaNormQuant,
    SuryaResidual,
    channelShape,
)
from .Pace import SuryaPwpa


@dataclasses.dataclass
class SuryaTapSpec:
    """The NORM_MODE 4 tail of one layer: cast, PACE, decast."""

    pace: nn.Module
    fpShift: int
    decShift: np.ndarray
    # 2**-decShift / gain for each channel: what one output code is worth.
    scaleOut: np.ndarray
    signed: bool = True


@dataclasses.dataclass
class SuryaLayerSpec:
    """Everything one compute layer needs to run on the Surya datapath."""

    weightInt8: np.ndarray
    params: RequantParams
    scaleIn: float
    scaleOut: float
    signed: bool = True
    rank: int = 4
    channelDim: int = 1
    # 32 on the tap, which keeps the accumulator in INT32.
    outputBits: int = 8
    # The INT32 -> BF16 cast after an INT32 NormQuant. None on the INT8 path.
    fpShift: Optional[int] = None
    # The non-linearity path: cast, function, decast to INT8.
    tap: Optional[SuryaTapSpec] = None
    # ACCUM_INIT_MODE 2: what one accumulator code is worth, for each channel.
    accumInit: Optional[np.ndarray] = None
    # The name and the geometry of the kernel node.
    kind: str = "conv"
    normMode: int = 0
    kernelShape: tuple = (1, 1)
    strides: tuple = (1, 1)
    pads: tuple = (0, 0)
    dilations: tuple = (1, 1)
    groups: int = 1


class _Scale(nn.Module):
    """Hold a scale: one number, or one for each channel."""

    def __init__(
        self,
        scale: Union[float, np.ndarray],
        rank: int = 4,
        channelDim: int = 1,
    ) -> None:
        super().__init__()
        values = torch.as_tensor(
            np.asarray(scale, dtype=np.float64), dtype=torch.float32
        )
        if values.numel() > 1:
            values = channelShape(values.flatten(), rank, channelDim)
        else:
            values = values.reshape(())
        self.register_buffer("scale", values)


class ScaleToCodes(_Scale):
    """Divide a tensor that already sits on a grid by its scale.

    The result is the INT8 code the CIM array reads. The round removes the
    float32 error of the division. The scale is one number for an activation
    and one for each channel for the residual skip.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.round(x / self.scale)


class CodesToScale(_Scale):
    """Put INT8 codes back on the real axis. The scale is per channel on the tap."""

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class AccumulatorGuard(nn.Module):
    """Count accumulator values that float32 cannot hold exactly.

    The convolution runs in float32, which is exact while the accumulator stays
    below 2**24. A non-zero count means the graph needs a wider accumulate dtype.
    """

    def __init__(self) -> None:
        super().__init__()
        self.inexact = 0
        self.seen = 0

    def forward(self, x: Tensor) -> Tensor:
        if not torch.jit.is_tracing():
            with torch.no_grad():
                self.inexact += int((x.abs() >= 2**24).sum())
                self.seen += int(x.numel())
        return x


class SuryaTap(nn.Module):
    """The NORM_MODE 4 path: cast, PACE, decast."""

    def __init__(
        self,
        pace: nn.Module,
        fpShift: Union[int, np.ndarray] = 0,
        decShift: Union[int, np.ndarray] = 0,
        signed: bool = True,
        rank: int = 4,
        channelDim: int = 1,
    ) -> None:
        super().__init__()
        self.cast = SuryaCast(fpShift, rank=rank, channelDim=channelDim)
        self.pace = pace
        self.decast = SuryaDecast(
            decShift, signed=signed, rank=rank, channelDim=channelDim
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decast(self.pace(self.cast(x)))


class SuryaMatMul(nn.Module):
    """`a @ b` on the codes of two activations, then the NormQuant.

    With `fpShift` the NormQuant keeps INT32 and the cast writes BF16, NORM_MODE
    2. Without it the NormQuant writes INT8 codes and the result is
    `codes * scaleOut`. Each of the `heads` on dimension -3 is one job: `params`
    holds one entry for the whole result, or one for each column of each head.
    """

    def __init__(
        self,
        scaleA: float,
        scaleB: float,
        params: RequantParams,
        scaleOut: float = 1.0,
        fpShift: Optional[int] = None,
        heads: int = 1,
    ) -> None:
        super().__init__()
        self.toCodesA = ScaleToCodes(scaleA)
        self.toCodesB = ScaleToCodes(scaleB)
        self.guard = AccumulatorGuard()
        self.normQuant = SuryaNormQuant(
            params.s16,
            params.b32,
            params.shift,
            outputBits=8 if fpShift is None else 32,
            rank=1,
            channelDim=0,
            cInit=params.cInit,
            multiplier=params.multiplier,
        )
        # The fields run over the columns, dimension -1, of each head.
        for name, field in self.normQuant.named_buffers():
            if heads > 1 and field.numel() > 1:
                setattr(self.normQuant, name, field.reshape(heads, 1, -1))
        if fpShift is None:
            self.store = CodesToScale(scaleOut)
        else:
            self.store = SuryaCast(fpShift, rank=1, channelDim=0)
            self.store.approximation = "store"

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        acc = self.guard(torch.matmul(self.toCodesA(a), self.toCodesB(b)))
        return self.store(self.normQuant(acc))


def roundingName(proxy: nn.Module) -> str:
    """The rounding mode of a quantizer, from its `float_to_int_impl` class."""
    impl = proxy
    for name in (
        "fused_activation_quant_proxy",
        "tensor_quant",
        "int_quant",
        "float_to_int_impl",
    ):
        impl = getattr(impl, name, None)
        if impl is None:
            return "unknown"
    name = type(impl).__name__.lower()
    if "floor" in name:
        return "floor"
    if "zero" in name or "trunc" in name:
        return "trunc"
    if "ceil" in name:
        return "ceil"
    return "round"


def suryaWBIOLForward(
    self: QuantWeightBiasInputOutputLayer, inp: Tensor, skip: Optional[Tensor] = None
) -> Tensor:
    """Integer forward pass for one compute layer on the Surya datapath.

    `skip` is the residual the array loads into its accumulator. The first
    parameter keeps a name from `brevitas.graph.base.INPUT_NAMES`, so an
    `EqualizedModule` binds it as the input and passes `skip` through.
    """
    if self.suryaRequantIn.settled:
        # The producer decided these codes, so `ScaleToCodes` reads them back.
        codes = self.suryaToCodes(inp)
    else:
        codes = self.suryaToCodes(self.suryaUnpack(self.input_quant(inp)))
    codes = self.suryaRequantIn(inp, codes)
    return self.suryaFromCodes(self.suryaKernel(codes, skip))


class SuryaComputeTransformation(TransformationPass):
    """Put every compute layer that has a spec on the integer datapath."""

    def __init__(self, specs: Dict[str, SuryaLayerSpec]) -> None:
        super().__init__(moduleCls=QuantWeightBiasInputOutputLayer, validationTol=1e-2)
        self.specs = specs
        self.installed: Dict[str, SuryaLayerSpec] = {}

    def transform(
        self, model: nn.Module, tracer: Optional[CustomBrevitasTracer] = None
    ) -> bool:
        # Keyed by module path: a spec depends on the place of its layer in the graph.
        done = False
        for path, module in model.named_modules():
            spec = self.specs.get(path)
            if (
                spec is None
                or not self.checkModuleType(module)
                or self.isTransformed(module)
            ):
                continue
            self.installForward(module, spec, tracer)
            self.markTransformed(module)
            self.installed[path] = spec
            done = True
        return done

    def injectForward(
        self, module: nn.Module, tracer: Optional[CustomBrevitasTracer] = None
    ) -> None:
        raise RuntimeError("the forward depends on the module path: call transform()")

    def installForward(
        self,
        module: nn.Module,
        spec: SuryaLayerSpec,
        tracer: Optional[CustomBrevitasTracer] = None,
    ) -> None:
        module.wrappedInnerForwardImpl = InnerForwardImplWrapperLinear(
            module.inner_forward_impl
        )
        device = module.weight.device
        module.suryaUnpack = UnpackQuantTensor()
        module.suryaToCodes = ScaleToCodes(spec.scaleIn).to(device)
        module.suryaRequantIn = SuryaRequantPoint(
            spec.scaleIn,
            signed=bool(getattr(module.input_quant, "is_signed", True)),
            rounding=roundingName(module.input_quant),
        ).to(device)
        module.suryaGuard = AccumulatorGuard()
        module.suryaResidual = (
            None if spec.accumInit is None else SuryaResidual().to(device)
        )
        module.suryaSkipToCodes = (
            None
            if spec.accumInit is None
            else ScaleToCodes(
                spec.accumInit, rank=spec.rank, channelDim=spec.channelDim
            ).to(device)
        )
        module.suryaTap = (
            None
            if spec.tap is None
            else SuryaTap(
                spec.tap.pace,
                fpShift=spec.tap.fpShift,
                decShift=spec.tap.decShift,
                signed=spec.tap.signed,
                rank=spec.rank,
                channelDim=spec.channelDim,
            ).to(device)
        )
        module.suryaCast = (
            None
            if spec.fpShift is None or spec.tap is not None
            else SuryaCast(spec.fpShift, rank=spec.rank, channelDim=spec.channelDim).to(
                device
            )
        )
        if module.suryaCast is not None:
            module.suryaCast.approximation = "store"
        module.suryaFromCodes = CodesToScale(
            spec.scaleOut if spec.tap is None else spec.tap.scaleOut,
            rank=spec.rank,
            channelDim=spec.channelDim,
        ).to(device)
        module.register_buffer(
            "suryaWeightInt8",
            torch.as_tensor(spec.weightInt8, dtype=torch.float32, device=device),
        )
        module.suryaNormQuant = SuryaNormQuant(
            spec.params.s16,
            spec.params.b32,
            spec.params.shift,
            signed=spec.signed,
            outputBits=spec.outputBits,
            rank=spec.rank,
            channelDim=spec.channelDim,
            cInit=spec.params.cInit,
            multiplier=spec.params.multiplier,
        ).to(device)
        module.suryaKernel = SuryaKernel(
            owner=module,
            kind=spec.kind,
            normMode=spec.normMode,
            signed=spec.signed,
            kernelShape=spec.kernelShape,
            strides=spec.strides,
            pads=spec.pads,
            dilations=spec.dilations,
            group=spec.groups,
        )
        module.forward = suryaWBIOLForward.__get__(module)

        if tracer:
            tracer.registerLeafModule(InnerForwardImplWrapperLinear)
            tracer.registerLeafModule(UnpackQuantTensor)
            tracer.registerLeafModule(ScaleToCodes)
            tracer.registerLeafModule(CodesToScale)
            tracer.registerLeafModule(AccumulatorGuard)
            tracer.registerLeafModule(SuryaRequantPoint)
            tracer.registerLeafModule(SuryaNormQuant)
            tracer.registerLeafModule(SuryaCast)
            tracer.registerLeafModule(SuryaPwpa)
            tracer.registerLeafModule(SuryaDecast)
            tracer.registerLeafModule(SuryaResidual)
            tracer.registerNonLeafModule(SuryaKernel)
            tracer.registerNonLeafModule(QuantWeightBiasInputOutputLayer)
            tracer.registerNonLeafModule(SuryaTap)


def accumulatorReport(model: nn.Module) -> Tuple[int, int]:
    """(inexact values, values seen) over every AccumulatorGuard in the model."""
    inexact = seen = 0
    for module in model.modules():
        if isinstance(module, AccumulatorGuard):
            inexact += module.inexact
            seen += module.seen
    return inexact, seen
