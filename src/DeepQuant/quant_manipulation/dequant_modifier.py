# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
This module provides a function to unify the linear dequant nodes (input, weight, bias)
into a single final dequant node after the linear wrapped_inner_forward_impl.

Key steps:
  1) Rewire bias quant to reference the quant nodes of input/weight instead of their dequant.
  2) Rewire the linear's wrapped_inner_forward_impl so it references bias_quant instead of bias_dequant.
  3) Clone the bias dequant parameters (scale/zero_point/bit_width) to a new Dequant node
     placed after the linear, removing the old bias_dequant node from the graph.
  4) Remove the input_dequant and weight_dequant nodes as well, once they have no more users.
  5) Recompile the FX GraphModule so that the generated forward code no longer references
     the removed nodes.

By the end, the linear operation is in the integer domain, and the final dequant occurs only once.
"""

import torch.fx as fx

# We assume Dequant is defined in brevitexporter.quant_divider.quant_dequant_nodes
from DeepQuant.quant_manipulation.quant_dequant_nodes import Dequant

BLUE = "\033[94m"
ENDC = "\033[0m"
CHECK = " ✓"
ARROW = " ›"


def unify_linear_dequants(
    fx_model: fx.GraphModule, debug: bool = False
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
        fx_model (fx.GraphModule): The input FX GraphModule to be modified.
        debug (bool): If True, prints debug information.

    Returns:
        fx.GraphModule: The modified FX GraphModule with a single dequant node after the linear.
    """
    graph = fx_model.graph
    all_nodes = list(graph.nodes)

    if debug:
        print(f"{BLUE}{ARROW} Starting Modification of Dequant Nodes...{ENDC}")

    for node in all_nodes:
        # Identify the "wrapped_inner_forward_impl" call for linear
        if node.op != "call_module" or "wrapped_inner_forward_impl" not in node.target:
            continue

        # Typically the node args are:
        #   (linear1_input_dequant, linear1_weight_dequant, linear1_bias_dequant)
        old_args = list(node.args)

        bias_dequant_node = None
        input_dequant_node = None
        weight_dequant_node = None

        new_lin_args = []

        # Collect and rewire the linear's arguments
        for arg in old_args:
            if arg.op == "call_module" and "dequant" in arg.target.lower():
                if "bias_dequant" in arg.target.lower():
                    bias_dequant_node = arg
                elif "weight_dequant" in arg.target.lower():
                    weight_dequant_node = arg
                else:
                    input_dequant_node = arg

                # Replace the dequant input with the corresponding quant node
                quant_node = arg.args[0]
                new_lin_args.append(quant_node)
            else:
                new_lin_args.append(arg)

        node.args = tuple(new_lin_args)

        if bias_dequant_node is None:
            # This would be unusual if a linear is missing bias or missing a bias_dequant
            if debug:
                print(f"Skipping {node.target}: no bias_dequant_node found.")
            continue

        # The bias_quant node that feeds bias_dequant_node might reference input/weight dequant
        # We rewrite it so that it references the input/weight quant nodes
        bias_quant_node = bias_dequant_node.args[0]
        if (
            bias_quant_node.op == "call_module"
            and "bias_quant" in bias_quant_node.target.lower()
        ):
            new_bq_args = list(bias_quant_node.args)
            # Typically new_bq_args = [bias, input_dequant, weight_dequant]
            for i, bq_arg in enumerate(new_bq_args):
                if bq_arg.op == "call_module" and "dequant" in bq_arg.target.lower():
                    new_bq_args[i] = bq_arg.args[0]  # The corresponding quant node
            bias_quant_node.args = tuple(new_bq_args)
        else:
            if debug:
                print(
                    "Warning: Did not find a typical 'bias_quant' node shape in the graph."
                )

        # Erase input_dequant/weight_dequant from the graph
        # They should now have zero real users
        for dnode in (input_dequant_node, weight_dequant_node):
            if dnode is not None:
                # For safety, remove all references
                for usr in list(dnode.users.keys()):
                    dnode.users[usr] = None
                if hasattr(fx_model, dnode.target):
                    delattr(fx_model, dnode.target)
                graph.erase_node(dnode)

        # Now we create the final single Dequant node after the linear
        # by cloning the bias_dequant submodule's parameters
        old_bias_dequant_mod = fx_model.get_submodule(bias_dequant_node.target)

        # Construct a new Dequant module from the old bias_dequant
        new_dequant_mod_name = (
            node.target.replace(".wrapped_inner_forward_impl", "") + "_unified_dequant"
        )

        unified_dequant_mod = Dequant(
            original_module=old_bias_dequant_mod.original_module,
            scale=old_bias_dequant_mod.scale,
            zero_point=old_bias_dequant_mod.zero_point,
            bit_width=old_bias_dequant_mod.bit_width,
        )

        fx_model.add_module(new_dequant_mod_name, unified_dequant_mod)

        # Insert the new dequant node after the linear's forward_impl
        with graph.inserting_after(node):
            new_dequant_node = graph.call_module(new_dequant_mod_name, args=(node,))

        # Reroute all users of node to the new dequant node
        old_users = list(node.users.keys())
        for usr in old_users:
            if usr is not new_dequant_node:
                new_args = list(usr.args)
                for i, a in enumerate(new_args):
                    if a is node:
                        new_args[i] = new_dequant_node
                usr.args = tuple(new_args)

        # Remove the old bias_dequant node from the graph
        for usr in list(bias_dequant_node.users.keys()):
            bias_dequant_node.users[usr] = None
        if hasattr(fx_model, bias_dequant_node.target):
            delattr(fx_model, bias_dequant_node.target)
        graph.erase_node(bias_dequant_node)

        if debug:
            print(f"    {CHECK} Modification done for {node.target}")

    # Clean up any leftover references
    graph.lint()
    graph.eliminate_dead_code()

    # Remove submodules that are now unused
    fx_model.delete_all_unused_submodules()

    # Recompile so that the generated forward code no longer references removed nodes
    fx_model.recompile()

    if debug:
        print(
            f"{BLUE}{ARROW} Modification of Dequant Nodes completed successfully{ENDC}"
        )

    return fx_model
