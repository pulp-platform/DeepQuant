# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from functools import partial
from typing import List, Optional, Type, Callable

import torch
import torch.nn as nn
from torch._dynamo import allow_in_graph
from torch.fx.graph_module import GraphModule

from brevitas.fx.brevitas_tracer import (
    Tracer,
    _is_brevitas_leaf_module,
    _symbolic_trace,
)


class QuantTracer():
    """Enhanced tracer with fine-grained control over module tracing."""

    def __init__(
        self,
        leafClasses: Optional[List[Type[nn.Module]]] = None,
        nonLeafClasses: Optional[List[Type[nn.Module]]] = None,
        debug: bool = False,
    ) -> None:
        self.leafClasses = leafClasses if leafClasses is not None else []
        self.nonLeafClasses = nonLeafClasses if nonLeafClasses is not None else []
        self.debug = debug

    def registerLeafModule(self, moduleCls: Type[nn.Module]) -> None:
        if moduleCls not in self.leafClasses:
            self.leafClasses.append(moduleCls)

    def registerNonLeafModule(self, moduleCls: Type[nn.Module]) -> None:
        if moduleCls not in self.nonLeafClasses:
            self.nonLeafClasses.append(moduleCls)

    def trace(self, model: nn.Module, exampleInput):

        brevitasClasses = [(m, id(m.__class__)) for _, m in model.named_modules() if m.__module__.startswith('brevitas.nn') or m.__module__.startswith('brevitas.core') or m.__module__.startswith('brevitas.proxy')]

        # leafClasses = (set(brevitasClasses) | set(self.leafClasses)) - set(self.nonLeafClasses)
        leafClasses = brevitasClasses
        graphs: List[torch.fx.GraphModule] = []

        def dynamo_graph_extract_compiler(gm: GraphModule, inputs: torch.Tensor) -> Callable:
            graphs.append(gm)
            return gm.forward
        
        torch._dynamo.reset()
        torch._dynamo.config.verbose = True

        # for op, _ in leafClasses:
        #     allow_in_graph(op.forward)

        # for _, m in model.named_modules():
        #     if m.__module__.startswith('brevitas.nn') or m.__module__.startswith('brevitas.core') or m.__module__.startswith('brevitas.proxy'):
        #        allow_in_graph(m) 

        allow_in_graph(model.inputQuant.__class__)
        allow_in_graph(model.inputQuant.forward)

        allow_in_graph(model.linear1.__class__)
        allow_in_graph(model.linear1.forward)

        model_fn = torch.compile(model, backend = dynamo_graph_extract_compiler, dynamic = False)
        from brevitas.export.inference import quant_inference_mode
        with torch.no_grad(), quant_inference_mode(model_fn):
            _ = model_fn(exampleInput)

        import IPython; IPython.embed()
        return graphs[0]

# def customBrevitasTrace(
#     root: nn.Module, concreteArgs=None, tracer: Optional[QuantTracer] = None
# ) -> GraphModule:
#     """Create an FX GraphModule using the QuantTracer (a custom Brevitas tracer)."""
#     if tracer is None:
#         tracer = QuantTracer()
#     return _symbolic_trace(tracer, root, concreteArgs)
