# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from typing import List, Literal
import torch.fx as fx

from colorama import Fore, Back, Style
from tabulate import tabulate


class GraphModulePrinter:
    @staticmethod
    def quant_info(
        node: fx.Node, prop: Literal["eps_in", "eps_out", "n_levels", "signed"]
    ) -> str:
        if "quant" not in node.meta:
            return "{}"

        qmeta = node.meta["quant"]

        if prop == "eps_in":
            return str(qmeta.get("eps_in", "{}"))
        elif prop == "eps_out":
            return str(qmeta.get("eps_out", "{}"))
        elif prop == "n_levels":
            n_in = qmeta.get("n_levels_in", "{}")
            n_out = qmeta.get("n_levels_out", "{}")
            return f"{n_in} -> {n_out}"
        elif prop == "signed":
            s_in = qmeta.get("signed_in", "{}")
            s_out = qmeta.get("signed_out", "{}")
            return f"{s_in} -> {s_out}"

        return "{}"

    @staticmethod
    def class_info(node: fx.Node, gm: fx.GraphModule, unicode: bool = False) -> str:
        if node.op == "call_module":
            submodule = gm.get_submodule(node.target)
            class_name = submodule.__class__.__name__
            if not unicode:
                return class_name
            if "PACT" in class_name:
                return Fore.GREEN + class_name + Style.RESET_ALL
            return class_name
        return ""

    @staticmethod
    def node_info(node: fx.Node, attr: str, unicode: bool = False) -> str:
        if not hasattr(node, attr):
            return ""
        value = getattr(node, attr)
        if attr == "op":
            if node.op == "call_function" and unicode:
                whitelist_functions = ["getitem"]
                if node.target.__name__ not in whitelist_functions:
                    return Back.YELLOW + str(value) + Style.RESET_ALL
        return str(value)

    @classmethod
    def get_node_spec(
        cls,
        node: fx.Node,
        gm: fx.GraphModule,
        show_opcode: bool = True,
        show_class: bool = True,
        show_name: bool = True,
        show_target: bool = True,
        show_args: bool = True,
        show_kwargs: bool = True,
        show_eps: bool = False,
        show_nlevels: bool = True,
        show_signed: bool = True,
        unicode: bool = False,
    ) -> List[str]:
        node_specs: List[str] = []

        if show_opcode:
            node_specs.append(cls.node_info(node, "op", unicode))
        if show_class:
            node_specs.append(cls.class_info(node, gm, unicode))
        if show_name:
            node_specs.append(cls.node_info(node, "name", unicode))
        if show_target:
            node_specs.append(cls.node_info(node, "target", unicode))
        if show_args:
            node_specs.append(cls.node_info(node, "args", unicode))
        if show_kwargs:
            node_specs.append(cls.node_info(node, "kwargs", unicode))

        if show_nlevels:
            node_specs.append(cls.quant_info(node, "n_levels"))
        if show_signed:
            node_specs.append(cls.quant_info(node, "signed"))
        if show_eps:
            node_specs.append(cls.quant_info(node, "eps_in"))
            node_specs.append(cls.quant_info(node, "eps_out"))

        return node_specs

    @classmethod
    def print_tabular(
        cls,
        gm: fx.GraphModule,
        show_opcode: bool = True,
        show_class: bool = True,
        show_name: bool = True,
        show_target: bool = True,
        show_args: bool = False,
        show_kwargs: bool = False,
        show_eps: bool = False,
        show_nlevels: bool = False,
        show_signed: bool = False,
        unicode: bool = False,
    ) -> None:

        node_list = list(gm.graph.nodes)
        node_specs = [
            cls.get_node_spec(
                node,
                gm,
                show_opcode=show_opcode,
                show_class=show_class,
                show_name=show_name,
                show_target=show_target,
                show_args=show_args,
                show_kwargs=show_kwargs,
                show_eps=show_eps,
                show_nlevels=show_nlevels,
                show_signed=show_signed,
                unicode=unicode,
            )
            for node in node_list
        ]

        headers = []
        if show_opcode:
            headers.append("opcode")
        if show_class:
            headers.append("class")
        if show_name:
            headers.append("name")
        if show_target:
            headers.append("target")
        if show_args:
            headers.append("args")
        if show_kwargs:
            headers.append("kwargs")
        if show_nlevels:
            headers.append("n_levels")
        if show_signed:
            headers.append("signed")
        if show_eps:
            headers.append("eps_in")
            headers.append("eps_out")

        print(tabulate(node_specs, headers=headers, tablefmt="mixed_grid"))
