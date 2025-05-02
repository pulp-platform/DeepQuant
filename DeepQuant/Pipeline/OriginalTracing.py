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


def traceOriginalModel(
    model: nn.Module, exampleInput: torch.Tensor, debug: bool = False
) -> Tuple[nn.Module, torch.Tensor]:
    """Symbolically trace the original model using Brevitas."""
    printer = GraphModulePrinter()

    tracedModel = brevitas_symbolic_trace(model)

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
