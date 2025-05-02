# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from DeepQuant.Utils.ConsoleFormatter import ConsoleColor as cc


def exportToOnnx(
    model: nn.Module,
    exampleInput: torch.Tensor,
    exportPath: Union[str, Path],
    debug: bool = False,
) -> Tuple[Path, np.ndarray]:
    """Export model to ONNX format and save input/output data."""
    exportPath = Path(exportPath)
    exportPath.mkdir(parents=True, exist_ok=True)

    onnxFile = exportPath / "network.onnx"
    inputFile = exportPath / "inputs.npz"
    outputFile = exportPath / "outputs.npz"

    torch.onnx.export(
        model,
        args=exampleInput,
        f=onnxFile,
        opset_version=13,
        keep_initializers_as_inputs=False,  # FBRANCASI: Prevent warnings
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    onnxModel = onnx.load(onnxFile)
    inferredModel = onnx.shape_inference.infer_shapes(onnxModel)
    onnx.save(inferredModel, onnxFile)

    np.savez(inputFile, input=exampleInput.cpu().numpy())
    if debug:
        print()
        print(cc.success(f"Input data saved to {inputFile}"))

    ortSession = ort.InferenceSession(onnxFile)
    ortInputs = {"input": exampleInput.cpu().numpy()}
    ortOutput = ortSession.run(None, ortInputs)[0]

    np.savez(outputFile, output=ortOutput)
    if debug:
        print(cc.success(f"Output data saved to {outputFile}\n"))

    return onnxFile, ortOutput
