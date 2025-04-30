# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from typing import Tuple

import torch
import torch.nn as nn
from brevitas.export.inference import quant_inference_mode
from brevitas.fx import brevitas_symbolic_trace

from DeepQuant.Utils.ConsoleFormatter import ConsoleColor as cc
from DeepQuant.Utils.GraphPrinter import GraphModulePrinter

from torch._dynamo import allow_in_graph

def traceOriginalModel(
    model: nn.Module, exampleInput: torch.Tensor, debug: bool = False
) -> Tuple[nn.Module, torch.Tensor]:
    """Symbolically trace the original model using Brevitas."""
    printer = GraphModulePrinter()

    # tracedModel = brevitas_symbolic_trace(model)
    graphs = []

    def dynamo_graph_extract_compiler(gm, inputs: torch.Tensor):
        graphs.append(gm)
        return gm.forward
    

    torch._dynamo.reset()
    torch._dynamo.config.verbose = True

    allow_in_graph(model.inputQuant)
    allow_in_graph(model.inputQuant.__class__)
    allow_in_graph(model.inputQuant.forward)

    allow_in_graph(model.inputQuant)
    allow_in_graph(model.linear1.__class__)
    allow_in_graph(model.linear1.forward)
    # JUNGVI: For Philip, dynamo uses the id of the thing passed in allow_in_graph to filter them. But it does not seems to work at least for brevitas layers, IDK if they have smth special...

    import IPython; IPython.embed()

    model_fn = torch.compile(model, backend = dynamo_graph_extract_compiler, dynamic = False)
    
    with torch.no_grad():
        _ = model_fn(exampleInput)

    import IPython; IPython.embed()

    tracedModel = graphs[0]

    if debug:
        print(cc.header("1. Original Network"))
        printer.printTabular(tracedModel)
        print()

    with torch.no_grad(), quant_inference_mode(model):
        output = model(exampleInput)

        # FBRANCASI: Handle case where output is a tuple (e.g., MHA)
        if isinstance(output, tuple):
            output = output[0]

    return tracedModel, output
