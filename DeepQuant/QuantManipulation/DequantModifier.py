# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
This module provides a function to unify the linear dequant nodes (input, weight, bias)
into a single final dequant node after the linear wrappedInnerForwardImpl.

Key steps:
  1) Rewire bias quant to reference the quant nodes of input/weight instead of their dequant.
  2) Rewire the linear's wrappedInnerForwardImpl so it references bias_quant instead of bias_dequant.
  3) Clone the bias dequant parameters (scale/zero_point/bit_width) to a new Dequant node
     placed after the linear, removing the old bias_dequant node from the graph.
  4) Remove the input_dequant and weight_dequant nodes as well, once they have no more users.
  5) Recompile the FX GraphModule so that the generated forward code no longer references
     the removed nodes.

By the end, the linear operation is in the integer domain, and the final dequant occurs only once.
"""

import torch.fx as fx

from DeepQuant.QuantManipulation.QuantDequantNodes import Dequant


BLUE = "\033[94m"
ENDC = "\033[0m"
CHECK = " ✓"
ARROW = " ›"


def unifyLinearDequants(
    fxModel: fx.GraphModule, debug: bool = False
) -> fx.GraphModule:
    """
    Unify the linear dequant nodes (input, weight, bias) into a single final dequant node.

    This transformation:
      * Redirects the linear's inputs to the quant nodes (removing input_dequant, weight_dequant).
      * Updates bias_quant to reference those same quant nodes, removing references to dequant.
      * Creates a new Dequant node after the linear operation, reusing the bias dequant parameters.
      * Erases the old dequant nodes from the graph and submodules.
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
        # Identify the "wrappedInnerForwardImpl" call for linear
        if node.op != "call_module" or "wrappedInnerForwardImpl" not in node.target:
            continue

        # Typically the node args are:
        #   (linear1_input_dequant, linear1_weight_dequant, linear1_bias_dequant)
        oldArgs = list(node.args)

        biasDequantNode = None
        inputDequantNode = None
        weightDequantNode = None

        newLinArgs = []

        # Collect and rewire the linear's arguments
        for arg in oldArgs:
            if arg.op == "call_module" and "dequant" in arg.target.lower():
                if "bias_dequant" in arg.target.lower():
                    biasDequantNode = arg
                elif "weight_dequant" in arg.target.lower():
                    weightDequantNode = arg
                else:
                    inputDequantNode = arg

                # Replace the dequant input with the corresponding quant node
                quantNode = arg.args[0]
                newLinArgs.append(quantNode)
            else:
                newLinArgs.append(arg)

        node.args = tuple(newLinArgs)

        if biasDequantNode is None:
            # This would be unusual if a linear is missing bias or missing a bias_dequant
            if debug:
                print(f"Skipping {node.target}: no biasDequantNode found.")
            continue

        # The bias_quant node that feeds biasDequantNode might reference input/weight dequant
        # We rewrite it so that it references the input/weight quant nodes
        biasQuantNode = biasDequantNode.args[0]
        if (
            biasQuantNode.op == "call_module"
            and "bias_quant" in biasQuantNode.target.lower()
        ):
            new_bq_args = list(biasQuantNode.args)
            # Typically new_bq_args = [bias, input_dequant, weight_dequant]
            for i, bq_arg in enumerate(new_bq_args):
                if bq_arg.op == "call_module" and "dequant" in bq_arg.target.lower():
                    new_bq_args[i] = bq_arg.args[0]  # The corresponding quant node
            biasQuantNode.args = tuple(new_bq_args)
        else:
            if debug:
                print(
                    "Warning: Did not find a typical 'bias_quant' node shape in the graph."
                )

        # Erase input_dequant/weight_dequant from the graph
        # They should now have zero real users
        for dnode in (inputDequantNode, weightDequantNode):
            if dnode is not None:
                # For safety, remove all references
                for usr in list(dnode.users.keys()):
                    dnode.users[usr] = None
                if hasattr(fxModel, dnode.target):
                    delattr(fxModel, dnode.target)
                graph.erase_node(dnode)

        # Now we create the final single Dequant node after the linear
        # by cloning the bias_dequant submodule's parameters
        oldBiasDequantMod = fxModel.get_submodule(biasDequantNode.target)

        # Construct a new Dequant module from the old bias_dequant
        newDequantModName = (
            node.target.replace(".wrappedInnerForwardImpl", "") + "_unified_dequant"
        )
        # JUNGVI: Torch modules name cannot contain "."
        newDequantModName = newDequantModName.replace(".", "_")

        unifiedDequantMod = Dequant(
            original_module=oldBiasDequantMod.original_module,
            scale=oldBiasDequantMod.scale,
            zero_point=oldBiasDequantMod.zero_point,
            bit_width=oldBiasDequantMod.bit_width,
        )

        fxModel.add_module(newDequantModName, unifiedDequantMod)

        # Insert the new dequant node after the linear's forward_impl
        with graph.inserting_after(node):
            newDequantNode = graph.call_module(newDequantModName, args=(node,))

        # Reroute all users of node to the new dequant node
        old_users = list(node.users.keys())
        for usr in old_users:
            if usr is not newDequantNode:
                newArgs = list(usr.args)
                for i, a in enumerate(newArgs):
                    if a is node:
                        newArgs[i] = newDequantNode
                usr.args = tuple(newArgs)

        # Remove the old bias_dequant node from the graph
        for usr in list(biasDequantNode.users.keys()):
            biasDequantNode.users[usr] = None
        if hasattr(fxModel, biasDequantNode.target):
            delattr(fxModel, biasDequantNode.target)
        graph.erase_node(biasDequantNode)

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
