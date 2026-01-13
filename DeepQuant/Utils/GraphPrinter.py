# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from typing import List, Literal

import torch.fx as fx
from colorama import Back, Fore, Style
from tabulate import tabulate


class GraphModulePrinter:
    """Formatter and printer for FX graph modules."""

    @staticmethod
    def quantInfo(
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
    def classInfo(node: fx.Node, gm: fx.GraphModule, unicode: bool = False) -> str:
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
    def nodeInfo(node: fx.Node, attr: str, unicode: bool = False) -> str:
        if not hasattr(node, attr):
            return ""
        value = getattr(node, attr)
        if attr == "op":
            if node.op == "call_function" and unicode:
                whitelist_functions = ["getitem"]
                if (
                    hasattr(node.target, "__name__")
                    and node.target.__name__ not in whitelist_functions
                ):
                    return Back.YELLOW + str(value) + Style.RESET_ALL
        return str(value)

    @classmethod
    def getNodeSpec(
        cls,
        node: fx.Node,
        gm: fx.GraphModule,
        showOpcode: bool = True,
        showClass: bool = True,
        showName: bool = True,
        showTarget: bool = True,
        showArgs: bool = True,
        showKwargs: bool = True,
        showEps: bool = False,
        showNlevels: bool = True,
        showSigned: bool = True,
        unicode: bool = False,
    ) -> List[str]:
        nodeSpecs: List[str] = []

        if showOpcode:
            nodeSpecs.append(cls.nodeInfo(node, "op", unicode))
        if showClass:
            nodeSpecs.append(cls.classInfo(node, gm, unicode))
        if showName:
            nodeSpecs.append(cls.nodeInfo(node, "name", unicode))
        if showTarget:
            nodeSpecs.append(cls.nodeInfo(node, "target", unicode))
        if showArgs:
            nodeSpecs.append(cls.nodeInfo(node, "args", unicode))
        if showKwargs:
            nodeSpecs.append(cls.nodeInfo(node, "kwargs", unicode))

        if showNlevels:
            nodeSpecs.append(cls.quantInfo(node, "n_levels"))
        if showSigned:
            nodeSpecs.append(cls.quantInfo(node, "signed"))
        if showEps:
            nodeSpecs.append(cls.quantInfo(node, "eps_in"))
            nodeSpecs.append(cls.quantInfo(node, "eps_out"))

        return nodeSpecs

    @classmethod
    def printTabular(
        cls,
        gm: fx.GraphModule,
        showOpcode: bool = True,
        showClass: bool = True,
        showName: bool = True,
        showTarget: bool = True,
        showArgs: bool = False,
        showKwargs: bool = False,
        showEps: bool = False,
        showNlevels: bool = False,
        showSigned: bool = False,
        unicode: bool = False,
    ) -> None:
        nodeList = list(gm.graph.nodes)
        nodeSpecs = [
            cls.getNodeSpec(
                node,
                gm,
                showOpcode=showOpcode,
                showClass=showClass,
                showName=showName,
                showTarget=showTarget,
                showArgs=showArgs,
                showKwargs=showKwargs,
                showEps=showEps,
                showNlevels=showNlevels,
                showSigned=showSigned,
                unicode=unicode,
            )
            for node in nodeList
        ]

        headers = []
        if showOpcode:
            headers.append("opcode")
        if showClass:
            headers.append("class")
        if showName:
            headers.append("name")
        if showTarget:
            headers.append("target")
        if showArgs:
            headers.append("args")
        if showKwargs:
            headers.append("kwargs")
        if showNlevels:
            headers.append("n_levels")
        if showSigned:
            headers.append("signed")
        if showEps:
            headers.append("eps_in")
            headers.append("eps_out")

        print(tabulate(nodeSpecs, headers=headers, tablefmt="mixed_grid"))
