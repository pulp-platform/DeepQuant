# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from typing import List, Optional, Type

import torch.nn as nn
from brevitas.fx.brevitas_tracer import (
    Tracer,
    _is_brevitas_leaf_module,
    _symbolic_trace,
)
from torch.fx.graph_module import GraphModule


class QuantTracer(Tracer):
    """Enhanced tracer with fine-grained control over module tracing."""

    def __init__(
        self,
        leafClasses: Optional[List[Type[nn.Module]]] = None,
        nonLeafClasses: Optional[List[Type[nn.Module]]] = None,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.leafClasses = leafClasses if leafClasses is not None else []
        self.nonLeafClasses = nonLeafClasses if nonLeafClasses is not None else []
        self.debug = debug

    def registerLeafModule(self, moduleCls: Type[nn.Module]) -> None:
        """Register a module class as a leaf module."""
        if moduleCls not in self.leafClasses:
            self.leafClasses.append(moduleCls)

    def registerNonLeafModule(self, moduleCls: Type[nn.Module]) -> None:
        """Register a module class as a non-leaf module."""
        if moduleCls not in self.nonLeafClasses:
            self.nonLeafClasses.append(moduleCls)

    def is_leaf_module(self, m: nn.Module, moduleQualifiedName: str) -> bool:
        """Determine if a module should be treated as a leaf module."""
        if any(isinstance(m, lc) for lc in self.leafClasses):
            return True
        if any(isinstance(m, nlc) for nlc in self.nonLeafClasses):
            return False
        return _is_brevitas_leaf_module(m, moduleQualifiedName)


def customBrevitasTrace(
    root: nn.Module, concreteArgs=None, tracer: Optional[QuantTracer] = None
) -> GraphModule:
    """Create an FX GraphModule using the QuantTracer (a custom Brevitas tracer)."""
    if tracer is None:
        tracer = QuantTracer()
    return _symbolic_trace(tracer, root, concreteArgs)
