# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Lionnus Kesting <lkesting@iis.ee.ethz.ch>

"""
Operators of the Surya accelerator, for the DeepQuant export flow.

    Operators.py        NormQuant, Cast, Decast, Residual
    Pace.py             the piecewise polynomial and the table that drives it
    LayerNorm.py        the PACE LayerNorm
    Softmax.py          the PACE softmax
    Kernels.py          one `surya` ONNX node for each kernel
    Lowering.py         float scales -> S16, B32, shift, C_init
    Transformations.py  the passes that put a Brevitas model on the datapath
"""

from .Kernels import (
    SURYA_DOMAIN,
    SURYA_OPSET_VERSION,
    SuryaKernel,
    SuryaRequantBoundary,
    SuryaRequantPoint,
    boundaryReport,
    fuseKernels,
    kernelSummary,
    settleCodeBoundaries,
)
from .LayerNorm import SuryaLayerNorm
from .Lowering import (
    RequantParams,
    applyRequant,
    lowerRequant,
    moveSignIntoWeights,
)
from .Operators import (
    APPROXIMATIONS,
    NQ_SCALE_MAX,
    NQ_SHIFT_MAX,
    SuryaCast,
    SuryaDecast,
    SuryaNormQuant,
    SuryaOperator,
    SuryaResidual,
    assertExportMatchesExact,
    setExportMode,
    setIdeal,
)
from .Pace import (
    PaceTable,
    SuryaPwpa,
    buildPwpa,
    buildTable,
    scaleTable,
)
from .Softmax import SuryaSoftmax
from .Transformations import (
    SuryaComputeTransformation,
    SuryaLayerSpec,
    SuryaMatMul,
    SuryaTap,
    SuryaTapSpec,
    accumulatorReport,
)

__all__ = [
    "APPROXIMATIONS",
    "SURYA_DOMAIN",
    "SURYA_OPSET_VERSION",
    "SuryaKernel",
    "SuryaRequantBoundary",
    "SuryaRequantPoint",
    "fuseKernels",
    "boundaryReport",
    "settleCodeBoundaries",
    "kernelSummary",
    "NQ_SCALE_MAX",
    "NQ_SHIFT_MAX",
    "PaceTable",
    "RequantParams",
    "SuryaCast",
    "SuryaComputeTransformation",
    "SuryaDecast",
    "SuryaLayerSpec",
    "SuryaMatMul",
    "SuryaNormQuant",
    "SuryaOperator",
    "SuryaLayerNorm",
    "SuryaPwpa",
    "SuryaResidual",
    "SuryaSoftmax",
    "SuryaTap",
    "SuryaTapSpec",
    "accumulatorReport",
    "applyRequant",
    "assertExportMatchesExact",
    "buildPwpa",
    "buildTable",
    "lowerRequant",
    "scaleTable",
    "moveSignIntoWeights",
    "setExportMode",
    "setIdeal",
]
