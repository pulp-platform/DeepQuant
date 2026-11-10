# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Lionnus Kesting <lkesting@iis.ee.ethz.ch>

"""
One ONNX node for one accelerator kernel, in the `surya` domain.

    surya::Gemm         M x N * N x P, the CIM array
    surya::Pointwise    a 1x1 convolution, which maps onto the same array
    surya::Depthwise    the 3x3 depthwise unit

The inputs are the buffers in the order the driver programs them, and the
attributes are the register fields:

    inputs      x, w, S16, B32, shift, C_init, fp_shift, dec_fp_shift, pace,
                accum_init
    attributes  NORM_MODE, signed_out, kernel_shape, strides, pads,
                dilations, group, ACCUM_INIT_MODE, PACE_FUNCTION, PACE_X_MIN,
                PACE_X_MAX, pace_gain, pace_partitions, pace_degree

`fp_shift`, `dec_fp_shift` and `pace` are absent on a NORM_MODE 0 node.
`accum_init` is the BF16 tensor the array loads into its accumulator, present
only when `ACCUM_INIT_MODE` is not 0.
"""

import itertools
import weakref
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from onnx import TensorProto
from torch import Tensor

SURYA_DOMAIN = "surya"
SURYA_OPSET_VERSION = 1

# A tensor this close to the grid, in codes, was written there by a NormQuant.
ON_GRID = 1e-3

# `symbolic` is static, so a module passes its id and is looked up here.
_REGISTRY: "weakref.WeakValueDictionary[int, SuryaKernel]" = (
    weakref.WeakValueDictionary()
)
_NEXT_ID = itertools.count()


class SuryaKernelFunction(torch.autograd.Function):
    """Name the datapath as one node in the `surya` domain.

    `forward` returns `out`, the tensor the datapath produced. `symbolic` does
    not read the stages that made it, so the exporter drops them as dead code.
    """

    @staticmethod
    def forward(ctx, kernelId, out, x, *operands):
        return out

    @staticmethod
    def symbolic(g, kernelId, out, x, *operands):
        kernel = _REGISTRY[kernelId]
        owner = kernel.owner[0]
        # A configured layer can still lack the accumulator init operand.
        loadsAccumInit = len(operands) > len(kernel.operandNames)
        attributes = {
            "NORM_MODE_i": kernel.normMode,
            "signed_out_i": int(kernel.signed),
            "kernel_shape_i": list(kernel.kernelShape),
            "strides_i": list(kernel.strides),
            "pads_i": list(kernel.pads),
            "dilations_i": list(kernel.dilations),
            "group_i": kernel.group,
            "ACCUM_INIT_MODE_i": 2 if loadsAccumInit else 0,
        }
        if owner.suryaTap is not None:
            pace = owner.suryaTap.pace
            attributes.update(
                PACE_FUNCTION_s=pace.functionName,
                PACE_X_MIN_f=float(pace.table.xMin),
                PACE_X_MAX_f=float(pace.table.xMax),
                pace_gain_f=pace.gain,
                # The flat `pace` operand holds (P - 1) + P*(D + 1) values.
                pace_partitions_i=pace.table.partitions,
                pace_degree_i=pace.table.degree,
            )
        # The float graph around the kernel holds integers: the Casts change no value.
        signedIn = owner.suryaRequantIn.signed
        codes = g.op(
            "Cast", x, to_i=TensorProto.INT8 if signedIn else TensorProto.UINT8
        )
        inputs = list(operands)
        if loadsAccumInit:
            # The array reads the accumulator init as BF16.
            inputs[-1] = g.op("Cast", inputs[-1], to_i=TensorProto.BFLOAT16)
        node = g.op(f"{SURYA_DOMAIN}::{kernel.opName}", codes, *inputs, **attributes)
        if kernel.normMode == 2:
            dtype = torch.bfloat16
        else:
            dtype = torch.int8 if kernel.signed else torch.uint8
        # The exporter cannot infer the shape of a custom node.
        node.setType(out.type().with_dtype(dtype))
        return g.op("Cast", node, to_i=TensorProto.FLOAT)


class SuryaRequantFunction(torch.autograd.Function):
    """Name one requantization boundary as a single node. `out` passes through."""

    @staticmethod
    def forward(ctx, unitId, out, x, scale, *fields):
        return out

    @staticmethod
    def symbolic(g, unitId, out, x, scale, *fields):
        unit = _REGISTRY[unitId]
        if unit.reinterprets:
            # The producer wrote the tensor on this grid: the accelerator runs no pass.
            if not unit.emitsCodes:
                return x
            return g.op("Round", g.op("Div", x, scale))
        codes = g.op(
            f"{SURYA_DOMAIN}::Requant",
            x,
            scale,
            *fields,
            signed_out_i=int(unit.signed),
            output_bits_i=8,
            rounding_s=unit.rounding,
            # `array` is a COMPUTE=0 NormQuant with S16, B32 and shift. `core`
            # is a pass a compute core runs, on an input that is on no code grid.
            unit_s=unit.unit,
        )
        dtype = torch.int8 if unit.signed else torch.uint8
        codes.setType(out.type().with_dtype(dtype))
        value = g.op("Cast", codes, to_i=TensorProto.FLOAT)
        # A layer reads the codes. Any other reader works on the real axis.
        return value if unit.emitsCodes else g.op("Mul", value, scale)


class SuryaRequantPoint(nn.Module):
    """One 8-bit code boundary: the scale and the rounding mode.

    `forward(x, out)` returns `out`, which the boundary computed from `x`.
    `reinterprets` is measured: it is True when no code moved in any batch seen,
    so the boundary only reads back the codes the producer wrote.
    """

    def __init__(
        self,
        scale: Union[float, np.ndarray, Tensor],
        signed: bool = True,
        rounding: str = "floor",
        emitsCodes: bool = True,
    ) -> None:
        super().__init__()
        self.unitId = next(_NEXT_ID)
        _REGISTRY[self.unitId] = self
        self.signed = bool(signed)
        self.rounding = str(rounding)
        self.emitsCodes = bool(emitsCodes)
        self.fused = False
        # An unmeasured boundary counts as one that quantizes.
        self.reinterprets = False
        self.seen = 0
        self.moved = 0
        self.values = 0
        # In codes. Near zero, the producer wrote the input on this grid.
        self.offGrid = 0.0
        # Set by `settleCodeBoundaries` when the input measures on this grid.
        self.settled = False
        self.register_buffer(
            "operand_scale", torch.as_tensor(np.asarray(scale, dtype=np.float32))
        )
        # `core` until `program` supplies the register fields of the array.
        self.unit = "core"
        self.fieldNames: tuple = ()

    def program(self, s16, b32, shift, gridIn) -> None:
        """Carry the fields a COMPUTE=0 pass programs, beside the scale.

        `S16`, `B32` and `shift` take the producer's grid to the grid of this
        boundary. `gridIn` is what one code of the producer is worth.
        """
        device = self.operand_scale.device

        def field(values, dtype):
            return _asInt(
                torch.as_tensor(np.asarray(values), dtype=torch.float64), dtype
            ).to(device)

        self.register_buffer("operand_S16", field(s16, torch.int16))
        self.register_buffer("operand_B32", field(b32, torch.int32))
        self.register_buffer("operand_shift", field(shift, torch.uint8))
        self.register_buffer(
            "operand_grid_in",
            torch.as_tensor(np.asarray(gridIn, dtype=np.float32)).to(device),
        )
        self.fieldNames = ("S16", "B32", "shift", "grid_in")
        self.unit = "array"

    def forward(self, x: Tensor, out: Tensor) -> Tensor:
        if not torch.jit.is_tracing():
            self._measure(x, out)
        if not self.fused:
            return out
        fields = [getattr(self, f"operand_{n}") for n in self.fieldNames]
        return SuryaRequantFunction.apply(
            self.unitId, out, x, self.operand_scale, *fields
        )

    @torch.no_grad()
    def _measure(self, x: Tensor, out: Tensor) -> None:
        """Record whether this boundary changed a code or only read the codes back."""
        exact = x / self.operand_scale
        plain = torch.round(exact)
        codes = out if self.emitsCodes else torch.round(out / self.operand_scale)
        free = bool(torch.equal(codes, plain))
        self.reinterprets = free if self.seen == 0 else (self.reinterprets and free)
        self.offGrid = max(self.offGrid, float((exact - plain).abs().max()))
        self.moved += int((codes != plain).sum())
        self.values += int(codes.numel())
        self.seen += 1


class SuryaRequantBoundary(nn.Module):
    """A quantizer that stands between two layers, as one node in the export.

    `inner` does the arithmetic and `point` carries the fields. The FX trace must
    stop at this module, or the marker leaves no node.
    """

    def __init__(self, inner: nn.Module, point: SuryaRequantPoint) -> None:
        super().__init__()
        self.inner = inner
        self.point = point

    def forward(self, x: Tensor) -> Tensor:
        return self.point(x, self.inner(x))


def _asInt(values: Tensor, dtype: torch.dtype) -> Tensor:
    """One register field, in the width the driver programs. Raises on overflow."""
    flat = values.detach().flatten().round()
    info = torch.iinfo(dtype)
    low, high = float(flat.min()), float(flat.max())
    if low < info.min or high > info.max:
        raise ValueError(
            f"a field spans {low:.0f}..{high:.0f}, which {dtype} cannot hold "
            f"({info.min}..{info.max})"
        )
    return flat.reshape(values.shape).to(dtype)


class SuryaKernel(nn.Module):
    """One accelerator kernel: the operands, the register fields, and the maths.

    `forward` runs the datapath stage by stage. Fused, it names the result as
    one node.
    """

    def __init__(
        self,
        owner: nn.Module,
        kind: str,
        normMode: int,
        signed: bool,
        kernelShape,
        strides,
        pads,
        dilations,
        group: int,
    ) -> None:
        super().__init__()
        self.owner = [owner]  # in a list, so it is not a child module
        self.kernelId = next(_NEXT_ID)
        _REGISTRY[self.kernelId] = self
        self.normMode = int(normMode)
        self.signed = bool(signed)
        self.kernelShape = tuple(int(v) for v in kernelShape)
        self.strides = tuple(int(v) for v in strides)
        self.pads = tuple(int(v) for v in pads)
        self.dilations = tuple(int(v) for v in dilations)
        self.group = int(group)
        self.fused = False
        # A convolution wider than 1x1 is a GEMM through im2col.
        if kind == "depthwise":
            self.opName = "Depthwise"
        elif kind == "linear" or self.kernelShape != (1, 1):
            self.opName = "Gemm"
        else:
            self.opName = "Pointwise"

        nq = owner.suryaNormQuant
        channels = nq.s16.numel()
        tap = owner.suryaTap
        cast = owner.suryaCast if tap is None else tap.cast

        # Every field is a buffer of this module: an autograd.Function applied
        # to a tensor the traced module does not own fails with `_Map_base::at`.
        names = ["w", "S16", "B32", "shift", "C_init"]
        self.register_buffer("operand_w", _asInt(owner.suryaWeightInt8, torch.int8))
        self.register_buffer("operand_S16", _asInt(nq.s16, torch.int16))
        self.register_buffer("operand_B32", _asInt(nq.b32, torch.int32))
        self.register_buffer("operand_shift", _asInt(nq.shift, torch.uint8))
        self.register_buffer(
            "operand_C_init",
            (
                torch.zeros(channels, dtype=torch.int32, device=nq.s16.device)
                if nq.cInit is None
                else _asInt(nq.cInit, torch.int32)
            ),
        )
        # The trailing operands stay contiguous: ONNX cannot skip an input.
        if cast is not None:
            shift = torch.round(-torch.log2(cast.scale.flatten().double()))
            self.register_buffer(
                "operand_fp_shift", _asInt(shift.expand(channels), torch.uint8)
            )
            names.append("fp_shift")
        if tap is not None:
            shift = torch.round(torch.log2(tap.decast.scale.flatten().double()))
            self.register_buffer(
                "operand_dec_fp_shift", _asInt(shift.expand(channels), torch.uint8)
            )
            names.append("dec_fp_shift")
            packed = torch.as_tensor(tap.pace.table.packed(), dtype=torch.float32)
            self.register_buffer("operand_pace", packed.to(nq.s16.device))
            names.append("pace")
        self.operandNames = tuple(names)

    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        owner = self.owner[0]
        # The skip arrives on the real axis and the accumulator loads integers.
        init = None if skip is None else owner.suryaSkipToCodes(skip)
        acc = owner.suryaGuard(
            owner.wrappedInnerForwardImpl(x, owner.suryaWeightInt8, None)
        )
        if init is not None:
            acc = owner.suryaResidual(acc, init)
        out = owner.suryaNormQuant(acc)
        if owner.suryaTap is not None:
            out = owner.suryaTap(out)
        elif owner.suryaCast is not None:
            out = owner.suryaCast(out)
        if not self.fused:
            return out
        operands = [getattr(self, f"operand_{n}") for n in self.operandNames]
        if init is not None:
            operands.append(init)
        return SuryaKernelFunction.apply(self.kernelId, out, x, *operands)


def fuseKernels(model: nn.Module, enabled: bool = True) -> int:
    """Switch every Surya node between one custom node and the stage by stage form."""
    count = 0
    for module in model.modules():
        if isinstance(module, (SuryaKernel, SuryaRequantPoint)):
            module.fused = enabled
            count += 1
    return count


def kernelSummary(model: nn.Module) -> dict:
    """How many nodes of each name and output mode the model holds."""
    counts: dict = {}
    for module in model.modules():
        if isinstance(module, SuryaKernel):
            key = f"{module.opName} NORM_MODE {module.normMode}"
            counts[key] = counts.get(key, 0) + 1
        elif isinstance(module, SuryaRequantPoint):
            if module.seen == 0:
                key = "Requant (never measured)"
            elif module.reinterprets:
                key = "code boundary, no pass"
            elif module.offGrid < ON_GRID:
                key = "on the grid already, but a code still moves"
            else:
                key = "Requant"
            counts[key] = counts.get(key, 0) + 1
    return counts


def settleCodeBoundaries(model: nn.Module) -> dict:
    """Stop a boundary from quantizing a code its producer already decided.

    The Brevitas proxy floors, which drops a code when the division lands one
    ULP below it. A boundary whose input measured on the grid skips the proxy
    and rounds. Call this after a forward pass.
    """
    report = {"settled": 0, "quantizing": 0, "unmeasured": 0}
    for module in model.modules():
        point = getattr(module, "suryaRequantIn", None)
        if not isinstance(point, SuryaRequantPoint):
            continue
        if not point.seen:
            report["unmeasured"] += 1
        elif point.offGrid < ON_GRID:
            point.settled = True
            # A settled boundary rounds, so its measurement starts again.
            point.rounding = "round"
            point.seen = 0
            point.reinterprets = False
            point.moved = point.values = 0
            report["settled"] += 1
        else:
            report["quantizing"] += 1
    return report


def boundaryReport(model: nn.Module) -> str:
    """What every requantization boundary measured, for the run's log."""
    points = [m for m in model.modules() if isinstance(m, SuryaRequantPoint)]
    if not points:
        return "no requantization boundary"
    rows = []
    for kind, keep in (
        ("free", lambda p: p.reinterprets),
        (
            "on the grid, a code still moves",
            lambda p: not p.reinterprets and p.offGrid < ON_GRID,
        ),
        (
            "really requantizes",
            lambda p: not p.reinterprets and p.offGrid >= ON_GRID,
        ),
    ):
        group = [p for p in points if p.seen and keep(p)]
        if not group:
            continue
        moved = sum(p.moved for p in group)
        values = sum(p.values for p in group) or 1
        rows.append(
            f"{len(group)} {kind}: {moved} of {values} codes move "
            f"({100 * moved / values:.3f}%), worst off grid "
            f"{max(p.offGrid for p in group):.2e} codes"
        )
    return "\n".join(f"    {r}" for r in rows)
