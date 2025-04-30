# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from typing import Any, Dict, List, Tuple

import torch.fx as fx
import torch.nn as nn

from DeepQuant.QuantManipulation.QuantDequantNodes import Dequant, Quant
from DeepQuant.Utils.ConsoleFormatter import ConsoleColor as cc


def insertQuantDequantPair(
    graph: fx.Graph,
    node: fx.Node,
    fxModel: fx.GraphModule,
    quantName: str,
    dequantName: str,
    originalModule: nn.Module,
    paramDict: Dict[str, Any],
) -> Tuple[fx.Node, fx.Node]:
    """Create separate Quant and Dequant nodes for a given FX node."""
    if "bias_quant" in node.target.lower():
        mainArg = node.args[0]
    elif "weight_quant" in node.target.lower():
        mainArg = node.args[0]
    else:
        mainArg = node.args[0]

    scaleVal = paramDict.get("scale", None)
    zpVal = paramDict.get("zero_point", None)
    bwVal = paramDict.get("bit_width", None)
    signedVal = paramDict.get("is_signed", True)

    fxModel.add_module(
        quantName, Quant(originalModule, scaleVal, zpVal, bwVal, signed=signedVal)
    )
    fxModel.add_module(
        dequantName,
        Dequant(originalModule, scaleVal, zpVal, bwVal, signed=signedVal),
    )

    with fxModel.graph.inserting_after(node):
        import IPython; IPython.embed()
        quantNode = fxModel.graph.call_module(quantName, args=(mainArg,))

    with graph.inserting_after(quantNode):
        dequantNode = graph.call_module(dequantName, args=(quantNode,))

    return quantNode, dequantNode


def convertQuantOperations(
    fxModel: fx.GraphModule, fullParamsDict: Dict[str, Dict[str, Any]], debug: bool
) -> fx.GraphModule:
    """Split quantization nodes into separate Quant and Dequant nodes."""
    graph = fxModel.graph
    nodesToRemove: List[fx.Node] = []

    if debug:
        print(cc.info("Starting Quantization Node Splitting..."))

    allNodes = list(graph.nodes)

    for node in allNodes:
        if (
            node.op == "call_module"
            and "quant" in node.target.lower()
            and "act_impl" not in node.target.lower()
        ):
            topLevel = node.target.split(".")[0]
            if topLevel in ["sigmoid"]:
                continue  # FBRANCASI: Skip sigmoid

            originalModule = fxModel.get_submodule(node.target)
            safeTarget = node.target.replace(".", "_").replace("_quant", "")
            quantName = f"{safeTarget}_quant_1"
            dequantName = f"{safeTarget}_dequant"
            paramInfo = fullParamsDict.get(node.target, {})

            quantNode, dequantNode = insertQuantDequantPair(
                graph,
                node,
                fxModel,
                quantName,
                dequantName,
                originalModule,
                paramInfo,
            )

            usersUpdated = False
            for userNode in list(node.users.keys()):
                if (
                    userNode.op == "call_function"
                    and hasattr(userNode.target, "__name__")
                    and userNode.target.__name__ == "cat"
                ):
                    # FBRANCASI: This is a concatenation operation - Special Handling
                    newCatArgs = list(userNode.args)
                    if len(newCatArgs) >= 1 and isinstance(newCatArgs[0], list):
                        tensorsList = newCatArgs[0]
                        updatedTensors = []
                        for tensor in tensorsList:
                            if tensor is node:
                                updatedTensors.append(dequantNode)
                            else:
                                updatedTensors.append(tensor)
                        newCatArgs[0] = updatedTensors
                        userNode.args = tuple(newCatArgs)
                        usersUpdated = True
                else:
                    # FBRANCASI: Standard node reference replacement
                    newArgs = []
                    for arg in userNode.args:
                        newArgs.append(dequantNode if arg is node else arg)
                    userNode.args = tuple(newArgs)
                    usersUpdated = True

            if usersUpdated:
                nodesToRemove.append(node)

    for eraseNode in nodesToRemove:
        graph.erase_node(eraseNode)

    graph.lint()

    if debug:
        print(cc.info("Quantization Node Splitting completed Successfully"))

    return fxModel
