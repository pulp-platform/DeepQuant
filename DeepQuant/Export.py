# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from DeepQuant.Pipeline.DequantUnify import mergeDequants
from DeepQuant.Pipeline.Injection import injectCustomForwards
from DeepQuant.Pipeline.OnnxExport import exportToOnnx
from DeepQuant.Pipeline.OriginalTracing import traceOriginalModel
from DeepQuant.Pipeline.QuantSplit import splitQuantNodes


def brevitasToTrueQuant(
    model: nn.Module,
    exampleInput: torch.Tensor,
    exportPath: Optional[Union[str, Path]] = Path.cwd() / "Tests" / "ONNX",
    debug: bool = False,
) -> nn.Module:
    """
    Export a Brevitas model to an FX GraphModule with unrolled quantization operations.

    This function applies a series of transformations to make the quantization steps
    explicit in the model's computation graph, enabling efficient integer-only execution.
    """

    # Pipeline Step 1: Trace the original model
    tracedModel, originalOutput = traceOriginalModel(model, exampleInput, debug)

    # Pipeline Step 2: Inject custom forward implementations
    transformedModel, transformedOutput = injectCustomForwards(
        tracedModel, exampleInput, originalOutput, debug
    )

    # Pipeline Step 3: Split quantization nodes
    splitModel, splitOutput = splitQuantNodes(
        transformedModel, exampleInput, transformedOutput, debug
    )

    # Pipeline Step 4: Unify dequant nodes
    unifiedModel, _ = mergeDequants(splitModel, exampleInput, splitOutput, debug)

    # Pipeline Step 5: Export to ONNX
    onnxFile, _ = exportToOnnx(unifiedModel, exampleInput, exportPath, debug)

    return unifiedModel
