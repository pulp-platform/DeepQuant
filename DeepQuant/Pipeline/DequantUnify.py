# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from typing import Tuple

import torch
import torch.nn as nn

from DeepQuant.QuantManipulation.DequantModifier import unifyLinearDequants
from DeepQuant.Utils.ConsoleFormatter import ConsoleColor as cc
from DeepQuant.Utils.GraphPrinter import GraphModulePrinter
from DeepQuant.Utils.TensorRecorder import TensorRecorder


def mergeDequants(
    model: nn.Module,
    exampleInput: torch.Tensor,
    referenceOutput: torch.Tensor,
    debug: bool = False,
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Unify dequantization nodes to enable integer-only computation.

    This step modifies the dequantization nodes in the graph to allow
    operations to run in the integer domain, applying dequantization
    only after the computations are complete (Requantization).
    """
    printer = GraphModulePrinter()
    tensorRecorder = TensorRecorder(debug=debug)

    if debug:
        # FBRANCASI: Register hooks to record tensors from the split model (before dequant modification)
        tensorRecorder.registerForwardHooks(
            model,
            nodeTypes=[
                "wrappedInnerForwardImpl",
                "dequant",
                "unified_dequant",
                "linear",
                "conv",
                "quant",
                "act",
                "bias_quant",
                "act_quant",
                "relu",
            ],
        )

    # FBRANCASI: Run the model to record tensors before modification
    with torch.no_grad():
        _ = model(exampleInput)

    if debug:
        # FBRANCASI: Save tensors as reference for comparison
        tensorRecorder.setReferenceTensors()

        # FBRANCASI: Register mappings from wrappedInnerForwardImpl nodes to expected unified_dequant nodes
        for node in model.graph.nodes:
            if node.op == "call_module" and "wrappedInnerForwardImpl" in node.target:
                baseName = node.target.replace(".wrappedInnerForwardImpl", "")
                dequantName = f"{baseName}_unified_dequant"
                dequantName = dequantName.replace(".", "_")

                tensorRecorder.recordNodeMapping(node.target, dequantName)

    unifiedModel = unifyLinearDequants(model, debug=debug)
    unifiedModel.recompile()

    if debug:
        print(cc.header("4. Network after Modification of Dequant Nodes"))
        printer.printTabular(unifiedModel)
        print()

    with torch.no_grad():
        output = unifiedModel(exampleInput)

    # FBRANCASI: Check output equivalence with a warning instead of error
    if not torch.allclose(referenceOutput, output, atol=1e-5) and debug:
        print(
            cc.warning(
                "Modification of Dequant Nodes may have changed the output slightly"
            )
        )

    if debug:
        # FBRANCASI: Register hooks for the unified model and compare tensors
        tensorRecorder.registerForwardHooks(
            unifiedModel,
            nodeTypes=[
                "wrappedInnerForwardImpl",
                "dequant",
                "unified_dequant",
                "linear",
                "conv",
                "quant",
                "act",
                "bias_quant",
                "act_quant",
                "relu",
            ],
        )

        # FBRANCASI: Run the model to record tensors after modification
        with torch.no_grad():
            _ = unifiedModel(exampleInput)

        # FBRANCASI: Compare tensors before and after modification
        print(cc.info("Tensor Comparison Before/After Dequant Unification:"))
        results = tensorRecorder.compareTensors()
        tensorRecorder.printComparisonResults(results)

        tensorRecorder.removeHooks()

    return unifiedModel, output
