# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
This module provides a function to unify the linear dequant nodes (input, weight, bias)
into a single final dequant node after the linear wrappedInnerForwardImpl.

Key steps:
  1) Rewire the linear's wrappedInnerForwardImpl to the quant nodes of its input, weight and
     bias, looking through shape-only ops between the input dequant and the linear.
  2) Insert one Dequant node after the linear. It reuses the bias dequant parameters, or uses
     input scale times weight scale when the bias is not quantized; an unquantized bias is
     then added after the Dequant.
  3) Remove the old dequant nodes once they have no more users.
  4) Recompile the FX GraphModule so that the generated forward code no longer references
     the removed nodes.

By the end, the linear operation is in the integer domain, and the final dequant occurs only once.
"""

import operator
import warnings
from typing import Any, List, Optional, Tuple

import torch
import torch.fx as fx

from DeepQuant.QuantManipulation.QuantDequantNodes import Dequant
from DeepQuant.QuantManipulation.QuantNodesDivider import quant_tensor_value


BLUE = "\033[94m"
ENDC = "\033[0m"
CHECK = " ✓"
ARROW = " ›"

SHAPE_METHODS = ("view", "reshape", "flatten")


def isDequant(fxModel: fx.GraphModule, node: Any) -> bool:
    return getattr(node, "op", None) == "call_module" and isinstance(
        fxModel.get_submodule(node.target), Dequant
    )


def isShapeOp(node: Any) -> bool:
    op = getattr(node, "op", None)
    if op == "call_method":
        return node.target in SHAPE_METHODS
    return op == "call_function" and node.target in (
        torch.flatten,
        torch.reshape,
        quant_tensor_value,
    )


def findDequant(
    fxModel: fx.GraphModule, node: Any
) -> Tuple[Optional[fx.Node], List[fx.Node]]:
    chain = []
    while not isDequant(fxModel, node):
        if not isShapeOp(node):
            return None, []
        chain.append(node)
        node = node.args[0]
    return node, chain[::-1]


def isZero(value: Any) -> bool:
    return value is not None and bool(torch.all(torch.as_tensor(value) == 0))


def isScalar(value: Any) -> bool:
    return value is not None and torch.as_tensor(value).numel() == 1


def uniqueName(fxModel: fx.GraphModule, name: str) -> str:
    candidate, index = name, 1
    while hasattr(fxModel, candidate):
        index += 1
        candidate = f"{name}_{index}"
    return candidate


def fetchAttr(module: Any, target: str) -> Any:
    for part in target.split("."):
        module = getattr(module, part)
    return module


def unifyLinearDequants(fxModel: fx.GraphModule, debug: bool = False) -> fx.GraphModule:
    """
    Unify the linear dequant nodes (input, weight, bias) into a single final dequant node.

    This transformation:
      * Redirects the linear's inputs to the quant nodes (removing input_dequant, weight_dequant).
      * Uses the quantized bias, or adds an unquantized bias after the new Dequant node.
      * Creates a new Dequant node after the linear operation, with the bias dequant parameters
        or with input scale times weight scale.
      * Erases the old dequant nodes that have no users left.
      * Recompiles the graph so the final forward does not reference removed nodes.

    Args:
        fxModel (fx.GraphModule): The input FX GraphModule to be modified.
        debug (bool): If True, prints debug information.

    Returns:
        fx.GraphModule: The modified FX GraphModule with a single dequant node after the linear.
    """
    graph = fxModel.graph
    allNodes = list(graph.nodes)

    if debug:
        print(f"{BLUE}{ARROW} Starting Modification of Dequant Nodes...{ENDC}")

    for node in allNodes:
        if node.op != "call_module" or "wrappedInnerForwardImpl" not in node.target:
            continue

        inputDequant, inputChain = findDequant(fxModel, node.args[0])
        weightDequant = node.args[1] if isDequant(fxModel, node.args[1]) else None
        bias = node.args[2] if len(node.args) > 2 else None
        biasDequant = bias if isDequant(fxModel, bias) else None
        floatBias = bias if bias is not None and biasDequant is None else None

        if inputDequant is None or weightDequant is None:
            warnings.warn(
                f"Dequant not moved past {node.target}: its input or weight is not dequantized."
            )
            continue

        inputMod = fxModel.get_submodule(inputDequant.target)
        weightMod = fxModel.get_submodule(weightDequant.target)
        if biasDequant is not None:
            reference = fxModel.get_submodule(biasDequant.target)
            scale, zeroPoint, bitWidth = (
                reference.scale,
                reference.zero_point,
                reference.bit_width,
            )
        elif (
            isZero(inputMod.zero_point)
            and isZero(weightMod.zero_point)
            and isScalar(inputMod.scale)
            and isScalar(weightMod.scale)
        ):
            reference = weightMod
            scale, zeroPoint, bitWidth = inputMod.scale * weightMod.scale, 0.0, 32.0
        else:
            warnings.warn(
                f"Dequant not moved past {node.target}: no quantized bias and no per-tensor scales."
            )
            continue

        quantInput = inputDequant.args[0]
        for user in list(inputDequant.users):
            if user.op == "call_method" and user.target in ("size", "dim"):
                user.replace_input_with(inputDequant, quantInput)
        for shapeNode in inputChain:
            with graph.inserting_before(node):
                quantInput = graph.create_node(
                    shapeNode.op,
                    shapeNode.target,
                    (quantInput,) + tuple(shapeNode.args[1:]),
                    dict(shapeNode.kwargs),
                )
        quantBias = biasDequant.args[0] if biasDequant is not None else None
        node.args = (quantInput, weightDequant.args[0], quantBias) + tuple(
            node.args[3:]
        )

        baseName = node.target.replace(".wrappedInnerForwardImpl", "").replace(".", "_")
        dequantName = uniqueName(fxModel, baseName + "_unified_dequant")
        fxModel.add_module(
            dequantName,
            Dequant(
                original_module=reference.original_module,
                scale=scale,
                zero_point=zeroPoint,
                bit_width=bitWidth,
            ),
        )
        with graph.inserting_after(node):
            newDequantNode = graph.call_module(dequantName, args=(node,))
        output = newDequantNode

        if floatBias is not None:
            layer = fxModel.get_submodule(node.target.rsplit(".", 1)[0])
            shape = (-1,) + (1,) * (layer.weight.dim() - 2)
            with graph.inserting_after(newDequantNode):
                if floatBias.op == "get_attr":
                    biasName = uniqueName(fxModel, baseName + "_float_bias")
                    fxModel.register_buffer(
                        biasName,
                        fetchAttr(fxModel, floatBias.target)
                        .detach()
                        .reshape(shape)
                        .clone(),
                    )
                    biasValue = graph.get_attr(biasName)
                else:
                    biasValue = graph.call_method("reshape", (floatBias, shape))
            with graph.inserting_after(biasValue):
                output = graph.call_function(operator.add, (newDequantNode, biasValue))

        for user in list(node.users):
            if user is not newDequantNode:
                user.replace_input_with(node, output)

        for oldNode in (*inputChain[::-1], inputDequant, weightDequant, biasDequant):
            if oldNode is not None and len(oldNode.users) == 0:
                graph.erase_node(oldNode)

        if debug:
            print(f"    {CHECK} Modification done for {node.target}")

    # Clean up any leftover references
    graph.lint()
    graph.eliminate_dead_code()

    # Remove submodules that are now unused
    fxModel.delete_all_unused_submodules()

    # Recompile so that the generated forward code no longer references removed nodes
    fxModel.recompile()

    if debug:
        print(
            f"{BLUE}{ARROW} Modification of Dequant Nodes completed successfully{ENDC}"
        )

    return fxModel
