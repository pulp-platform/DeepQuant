# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import torch.fx as fx
from typing import Dict, Any, List, Tuple
from .QuantDequantNodes import Quant, Dequant
import torch.nn as nn

BLUE = "\033[94m"
ENDC = "\033[0m"
ARROW = " ›"


def create_quant_dequant_nodes(
    graph: fx.Graph,
    node: fx.Node,
    fx_model: fx.GraphModule,
    quant_name: str,
    dequant_name: str,
    original_module: nn.Module,
    param_dict: Dict[str, Any],
) -> Tuple[fx.Node, fx.Node]:
    """Create separate Quant and Dequant nodes for a given FX node."""
    if "bias_quant" in node.target.lower():
        main_arg = node.args[0]
    elif "weight_quant" in node.target.lower():
        main_arg = node.args[0]
    else:
        main_arg = node.args[0]

    scale_val = param_dict.get("scale", None)
    zp_val = param_dict.get("zero_point", None)
    bw_val = param_dict.get("bit_width", None)
    signed_val = param_dict.get("is_signed", True)

    fx_model.add_module(
        quant_name, Quant(original_module, scale_val, zp_val, bw_val, signed=signed_val)
    )
    fx_model.add_module(
        dequant_name,
        Dequant(original_module, scale_val, zp_val, bw_val, signed=signed_val),
    )

    with graph.inserting_after(node):
        quant_node = graph.call_module(quant_name, args=(main_arg,))

    with graph.inserting_after(quant_node):
        dequant_node = graph.call_module(dequant_name, args=(quant_node,))

    return quant_node, dequant_node


def split_quant_nodes(
    fx_model: fx.GraphModule, full_params_dict: Dict[str, Dict[str, Any]], debug: bool
) -> fx.GraphModule:
    graph = fx_model.graph
    nodes_to_erase: List[fx.Node] = []

    if debug:
        print(f"{BLUE}{ARROW} Starting Quantization Node Splitting...{ENDC}")

    all_nodes = list(graph.nodes)

    for node in all_nodes:
        if (
            node.op == "call_module"
            and "quant" in node.target.lower()
            and "act_impl" not in node.target.lower()
        ):
            top_level = node.target.split(".")[0]
            if top_level in ["sigmoid"]:
                continue  # FBRANCASI: Skip sigmoid

            original_module = fx_model.get_submodule(node.target)
            safe_target = node.target.replace(".", "_").replace("_quant", "")
            quant_name = f"{safe_target}_quant_1"
            dequant_name = f"{safe_target}_dequant"
            param_info = full_params_dict.get(node.target, {})

            quant_node, dequant_node = create_quant_dequant_nodes(
                graph,
                node,
                fx_model,
                quant_name,
                dequant_name,
                original_module,
                param_info,
            )

            users_updated = False
            for user_node in list(node.users.keys()):
                if (
                    user_node.op == "call_function"
                    and hasattr(user_node.target, "__name__")
                    and user_node.target.__name__ == "cat"
                ):
                    # FBRANCASI: This is a concatenation operation - Special Handling
                    new_cat_args = list(user_node.args)
                    if len(new_cat_args) >= 1 and isinstance(new_cat_args[0], list):
                        tensors_list = new_cat_args[0]
                        updated_tensors = []
                        for tensor in tensors_list:
                            if tensor is node:
                                updated_tensors.append(dequant_node)
                            else:
                                updated_tensors.append(tensor)
                        new_cat_args[0] = updated_tensors
                        user_node.args = tuple(new_cat_args)
                        users_updated = True
                else:
                    # FBRANCASI: Standard node reference replacement
                    new_args = []
                    for arg in user_node.args:
                        new_args.append(dequant_node if arg is node else arg)
                    user_node.args = tuple(new_args)
                    users_updated = True

            if users_updated:
                nodes_to_erase.append(node)

    for erase_node in nodes_to_erase:
        graph.erase_node(erase_node)

    graph.lint()

    if debug:
        print(f"{BLUE}{ARROW} Quantization Node Splitting completed Successfully{ENDC}")

    return fx_model
