# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
This module provides a specialized GraphModulePrinter class to display an FX GraphModule
in a tabular format, including optional metadata about quantization (like eps, n_levels, signed).

Usage:
    from DeepQuant.graph_printer import GraphModulePrinter

    printer = GraphModulePrinter()
    printer.print_tabular(
        fx_model,
        show_opcode=True,
        show_class=True,
        show_name=True,
        show_target=True,
        show_args=True,
        show_kwargs=True,
        show_eps=False,
        show_nlevels=True,
        show_signed=True,
        unicode=False
    )

Note:
- This example assumes that each node in the graph may have a `node.meta['quant']` dict
  with fields like eps_in, eps_out, n_levels_in, n_levels_out, signed_in, and signed_out.
- If these fields are not present, the code will gracefully skip them or display placeholders.
- If you do not have such metadata in node.meta, you can adapt the logic to suit your needs.
"""

import math
from typing import Any, List, Literal, Optional
import torch.fx as fx

try:
    # Optional: colorama for colored output (requires `pip install colorama`)
    from colorama import Fore, Back, Style

    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

try:
    # Optional: tabulate for printing tables (requires `pip install tabulate`)
    from tabulate import tabulate

    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False


class GraphModulePrinter:
    """
    Class for printing an FX GraphModule in a tabular format, optionally displaying
    quantization metadata stored in node.meta['quant'].

    The code is based on an example snippet from a supervisor. The logic is adjusted
    to fit our code style and to gracefully handle missing metadata.
    """

    @staticmethod
    def quant_info(
        node: fx.Node, prop: Literal["eps_in", "eps_out", "n_levels", "signed"]
    ) -> str:
        """
        Retrieve a string representation of the quantization property for a given node.

        Args:
            node: The FX node containing potential quantization metadata.
            prop: The quantization property to display. One of 'eps_in', 'eps_out',
                  'n_levels', or 'signed'.

        Returns:
            A string representation of the requested property if it exists, or '{}' otherwise.
        """
        if "quant" not in node.meta:
            return "{}"

        # At this point, we assume node.meta['quant'] is a dict-like object containing
        # fields such as eps_in, eps_out, n_levels_in, n_levels_out, signed_in, signed_out, etc.
        qmeta = node.meta["quant"]

        if prop == "eps_in":
            return str(qmeta.get("eps_in", "{}"))
        elif prop == "eps_out":
            return str(qmeta.get("eps_out", "{}"))
        elif prop == "n_levels":
            # This is just an example: we might have n_levels_in, n_levels_out, etc.
            n_in = qmeta.get("n_levels_in", "{}")
            n_out = qmeta.get("n_levels_out", "{}")
            return f"{n_in} -> {n_out}"
        elif prop == "signed":
            # Example: 'signed_in' and 'signed_out'
            s_in = qmeta.get("signed_in", "{}")
            s_out = qmeta.get("signed_out", "{}")
            return f"{s_in} -> {s_out}"

        return "{}"

    @staticmethod
    def class_info(node: fx.Node, gm: fx.GraphModule, unicode: bool = False) -> str:
        """
        Retrieve class name for call_module nodes. For example, if node.target is
        referencing a submodule of type nn.Conv2d, this returns 'Conv2d'.

        Args:
            node: The FX node to analyze.
            gm: The FX GraphModule containing the node.
            unicode: If True, optionally highlight certain classes.

        Returns:
            The class name as a string, or '' if not applicable.
        """
        if node.op == "call_module":
            submodule = gm.get_submodule(node.target)
            class_name = submodule.__class__.__name__
            if not COLORAMA_AVAILABLE or not unicode:
                return class_name
            # Optionally highlight if it's a special class, e.g. 'PACT' or so.
            if "PACT" in class_name:
                return Fore.GREEN + class_name + Style.RESET_ALL
            return class_name
        return ""

    @staticmethod
    def node_info(node: fx.Node, attr: str, unicode: bool = False) -> str:
        """
        Retrieve a specified attribute from the node (e.g. 'op', 'name', 'target', 'args').

        Args:
            node: The FX node.
            attr: The name of the attribute to retrieve (e.g. 'op', 'name', 'target', 'args').
            unicode: If True, highlight certain functions in color.

        Returns:
            A string representation of the requested attribute, or '' if not present.
        """
        if not hasattr(node, attr):
            return ""
        value = getattr(node, attr)
        if attr == "op":
            # Optionally highlight certain call_function ops
            if node.op == "call_function" and COLORAMA_AVAILABLE and unicode:
                # Example of a function whitelist
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
        """
        Collect string representations of the node's attributes/metadata for printing.

        Args:
            node: The FX node to process.
            gm: The FX GraphModule containing the node.
            show_opcode: Whether to display the node's op code.
            show_class: Whether to display the submodule class name (for call_module).
            show_name: Whether to display the node's name.
            show_target: Whether to display the node's target.
            show_args: Whether to display the node's args.
            show_kwargs: Whether to display the node's kwargs.
            show_eps: Whether to display the quantization eps_in/eps_out (if available).
            show_nlevels: Whether to display the n_levels_in -> n_levels_out.
            show_signed: Whether to display the signed_in -> signed_out.
            unicode: If True, apply color highlights for certain attributes.

        Returns:
            A list of strings representing each requested attribute in order.
        """
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
        """
        Print the graph in a tabular format with optional quantization metadata.

        Args:
            gm: The FX GraphModule to display.
            show_opcode: Whether to display the node's op code.
            show_class: Whether to display the submodule class name (for call_module).
            show_name: Whether to display the node's name.
            show_target: Whether to display the node's target.
            show_args: Whether to display the node's args.
            show_kwargs: Whether to display the node's kwargs.
            show_eps: Whether to display the quantization eps_in/eps_out (if available).
            show_nlevels: Whether to display the n_levels_in -> n_levels_out.
            show_signed: Whether to display the signed_in -> signed_out.
            unicode: If True, apply color highlights for certain attributes.

        Returns:
            None
        """
        if not TABULATE_AVAILABLE:
            print(
                "Warning: 'tabulate' is not installed. Install via 'pip install tabulate' to use print_tabular."
            )
            return

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

        from tabulate import tabulate  # safe import inside method

        print(tabulate(node_specs, headers=headers, tablefmt="mixed_grid"))
