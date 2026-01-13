# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import torch.fx as fx

from DeepQuant.QuantManipulation.QuantDequantNodes import Dequant
from DeepQuant.Utils.ConsoleFormatter import ConsoleColor as cc


def unifyLinearDequants(fxModel: fx.GraphModule, debug: bool = False) -> fx.GraphModule:
    """Unify the linear dequant nodes (input, weight, bias) into a single final dequant node."""
    graph = fxModel.graph
    allNodes = list(graph.nodes)

    if debug:
        print(cc.info("Starting Modification of Dequant Nodes..."))

    for node in allNodes:
        if node.op != "call_module" or "wrappedInnerForwardImpl" not in node.target:
            continue

        oldArgs = list(node.args)

        biasDequantNode = None
        inputDequantNode = None
        weightDequantNode = None

        newLinArgs = []

        for arg in oldArgs:
            # FCONTI: there is no Bias, propagate this to the newLinArgs
            if arg is None:
                newLinArgs.append(arg)
            elif arg.op == "call_module" and "dequant" in arg.target.lower():
                if "bias_dequant" in arg.target.lower():
                    biasDequantNode = arg
                elif "weight_dequant" in arg.target.lower():
                    weightDequantNode = arg
                else:
                    inputDequantNode = arg

                quantNode = arg.args[0]
                newLinArgs.append(quantNode)
            else:
                newLinArgs.append(arg)

        node.args = tuple(newLinArgs)

        if biasDequantNode is None:
            # FCONTI: this happens if a linear layer has no bias
            if debug:
                print(f"Skipping bias for {node.target}: no biasDequantNode found.")
            biasQuantNode = None
        else:
            biasQuantNode = biasDequantNode.args[0]
            if (
                biasQuantNode.op == "call_module"
                and "bias_quant" in biasQuantNode.target.lower()
            ):
                newBqArgs = list(biasQuantNode.args)
                for i, bqArg in enumerate(newBqArgs):
                    if bqArg.op == "call_module" and "dequant" in bqArg.target.lower():
                        newBqArgs[i] = bqArg.args[0]
                biasQuantNode.args = tuple(newBqArgs)
            else:
                if debug:
                    print(
                        "Warning: Did not find a typical 'bias_quant' node shape in the graph."
                    )

        # FCONTI: if there is a bias node, use it for scale/zeropoint/bitwidth.
        #         otherwise, rely on weight*input
        if biasDequantNode is not None:
            oldBiasDequantMod = fxModel.get_submodule(biasDequantNode.target)
            dequantScale = oldBiasDequantMod.scale
            dequantZeroPoint = oldBiasDequantMod.zeroPoint
            dequantBitWidth = oldBiasDequantMod.bitWidth
            oldDequantMod = oldBiasDequantMod
        else:
            oldInputDequantMod = fxModel.get_submodule(inputDequantNode.target)
            oldWeightDequantMod = fxModel.get_submodule(weightDequantNode.target)
            dequantScale = oldWeightDequantMod.scale * oldInputDequantMod.scale
            # FCONTI: technically it should be:
            #         dZP = oWDM.zP * oIDM.zP - oWDM.scale * oIDM.zP * sum(weights)
            #         how to appropriately compute sum(weights)?
            #         for now we restrict ourselves to oIDM.zP = 0, so dZP = 0
            if debug and oldInputDequantMod.zeroPoint != 0.0:
                print(
                    f"Warning: input Dequant node for {node.target} has non-zero zero-point (unsupported). Expect wrong results!"
                )
            dequantZeroPoint = 0.0
            dequantBitWidth = 32  # FCONTI: this is simply a reasonable assumption: is there a less arbitrary one?
            oldDequantMod = oldWeightDequantMod

        for dnode in (inputDequantNode, weightDequantNode):
            if dnode is not None:
                for usr in list(dnode.users.keys()):
                    dnode.users[usr] = None
                if hasattr(fxModel, dnode.target):
                    delattr(fxModel, dnode.target)
                graph.erase_node(dnode)

        newDequantModName = (
            node.target.replace(".wrappedInnerForwardImpl", "") + "_unified_dequant"
        )
        # JUNGVI: Torch modules name cannot contain "."
        newDequantModName = newDequantModName.replace(".", "_")

        unifiedDequantMod = Dequant(
            originalModule=oldDequantMod.originalModule,
            scale=dequantScale,
            zeroPoint=dequantZeroPoint,
            bitWidth=dequantBitWidth,
        )

        fxModel.add_module(newDequantModName, unifiedDequantMod)

        with graph.inserting_after(node):
            newDequantNode = graph.call_module(newDequantModName, args=(node,))

        oldUsers = list(node.users.keys())
        for usr in oldUsers:
            if usr is not newDequantNode:
                newArgs = list(usr.args)
                for i, a in enumerate(newArgs):
                    if a is node:
                        newArgs[i] = newDequantNode
                usr.args = tuple(newArgs)

        if biasDequantNode is not None:
            for usr in list(biasDequantNode.users.keys()):
                biasDequantNode.users[usr] = None
            if hasattr(fxModel, biasDequantNode.target):
                delattr(fxModel, biasDequantNode.target)
            graph.erase_node(biasDequantNode)

        if debug:
            print(cc.success(f"Modification done for {node.target}"))

    graph.lint()
    graph.eliminate_dead_code()

    fxModel.delete_all_unused_submodules()

    fxModel.recompile()

    if debug:
        print(cc.info("Modification of Dequant Nodes completed successfully"))

    return fxModel
