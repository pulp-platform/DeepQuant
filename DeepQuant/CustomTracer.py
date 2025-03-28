# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Custom Brevitas tracer implementation for handling module transformation and tracing.
"""

import torch.nn as nn
from brevitas.fx.brevitas_tracer import (
    _symbolic_trace,
    _is_brevitas_leaf_module,
    Tracer,
)
from torch.fx.graph_module import GraphModule
from typing import List, Type, Optional


class CustomBrevitasTracer(Tracer):
    """
    A custom tracer that allows explicit control over leaf and non-leaf module designation.

    This tracer extends the Brevitas tracer to provide fine-grained control over which modules
    should be treated as leaf modules (traced as a single unit) vs non-leaf modules
    (traced into their constituent operations).
    """

    def __init__(
        self,
        leafClasses: Optional[List[Type[nn.Module]]] = None,
        nonLeafClasses: Optional[List[Type[nn.Module]]] = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the custom tracer with optional leaf and non-leaf module lists.

        Args:
            leaf_classes: List of module classes to be treated as leaf modules.
            non_leaf_classes: List of module classes to be treated as non-leaf modules.
            debug: Whether to print debug information during tracing.
        """
        super().__init__()
        self.leafClasses = leafClasses if leafClasses is not None else []
        self.nonLeafClasses = nonLeafClasses if nonLeafClasses is not None else []
        self.debug = debug

    def registerLeafModule(self, moduleCls: Type[nn.Module]) -> None:
        """
        Add a module class to the list of leaf modules.

        Args:
            module_cls: The module class to register as a leaf module.
        """
        if moduleCls not in self.leafClasses:
            self.leafClasses.append(moduleCls)

    def registerNonLeafModule(self, moduleCls: Type[nn.Module]) -> None:
        """
        Add a module class to the list of non-leaf modules.

        Args:
            module_cls: The module class to register as a non-leaf module.
        """
        if moduleCls not in self.nonLeafClasses:
            self.nonLeafClasses.append(moduleCls)

    def is_leaf_module(self, m: nn.Module, moduleQualifiedName: str) -> bool:
        """
        Determine whether a module should be treated as a leaf module.

        The decision follows this priority:
        1. If module is in leaf_classes, treat as leaf
        2. If module is in non_leaf_classes, treat as non-leaf
        3. Otherwise, fall back to default Brevitas behavior

        Args:
            m: The module to check.
            module_qualified_name: The fully qualified name of the module.

        Returns:
            bool: True if the module should be treated as a leaf module, False otherwise.
        """
        # First check explicitly registered classes
        if any(isinstance(m, lc) for lc in self.leafClasses):
            return True
        if any(isinstance(m, nlc) for nlc in self.nonLeafClasses):
            return False
        # Fall back to default Brevitas behavior
        return _is_brevitas_leaf_module(m, moduleQualifiedName)


def customBrevitasTrace(
    root: nn.Module, concreteArgs=None, tracer: Optional[CustomBrevitasTracer] = None
) -> GraphModule:
    """
    Create an FX GraphModule using the CustomBrevitasTracer.

    Args:
        root: The root module to trace.
        concrete_args: Concrete arguments to use for tracing.
        tracer: Optional pre-configured CustomBrevitasTracer instance.

    Returns:
        GraphModule: The traced module.
    """
    if tracer is None:
        tracer = CustomBrevitasTracer()
    return _symbolic_trace(tracer, root, concreteArgs)
