# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from typing import Tuple

import torch
import torch.nn as nn

from DeepQuant.QuantManipulation.QuantizationParameterExtractor import (
    extractBrevitasProxyParams,
    printQuantParams,
)
from DeepQuant.QuantManipulation.QuantNodesDivider import convertQuantOperations
from DeepQuant.Utils.ConsoleFormatter import ConsoleColor as cc
from DeepQuant.Utils.GraphPrinter import GraphModulePrinter


def splitQuantNodes(
    model: nn.Module,
    exampleInput: torch.Tensor,
    referenceOutput: torch.Tensor,
    debug: bool = False,
    checkEquivalence: bool = False,
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Split quantization nodes into separate Quant and Dequant nodes.

    This step transforms each quantization operation into explicit
    Quant and Dequant node pairs, providing clear separation between
    quantized and floating-point operations.
    """
    printer = GraphModulePrinter()

    proxyParams = extractBrevitasProxyParams(model)

    if debug:
        printQuantParams(proxyParams)

    splitModel = convertQuantOperations(model, proxyParams, debug)
    splitModel.recompile()

    with torch.no_grad():
        output = splitModel(exampleInput)

    if checkEquivalence:
        # FBRANCASI: Handle case where output/referenceOutput might be tuples
        refToCompare = referenceOutput[0] if isinstance(referenceOutput, tuple) else referenceOutput
        outToCompare = output[0] if isinstance(output, tuple) else output
        if torch.allclose(refToCompare, outToCompare, atol=1e-5):
            if debug:
                print(cc.success("Split of Quant Nodes: output is consistent"))
        else:
            raise RuntimeError(
                cc.error("Split of Quant Nodes changed the output significantly")
            )

    if debug:
        print(cc.header("3. Network after Split of Quant Nodes"))
        printer.printTabular(splitModel)
        print()

    return splitModel, output
