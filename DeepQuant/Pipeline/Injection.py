# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from typing import Tuple

import torch
import torch.nn as nn

from DeepQuant.Transforms.Executor import TransformationExecutor
from DeepQuant.Transforms.Transformations import (
    ActivationTransformation,
    LinearTransformation,
    MHATransformation,
)
from DeepQuant.Utils.ConsoleFormatter import ConsoleColor as cc
from DeepQuant.Utils.CustomTracer import QuantTracer, customBrevitasTrace
from DeepQuant.Utils.GraphPrinter import GraphModulePrinter


def injectCustomForwards(
    model: nn.Module,
    exampleInput: torch.Tensor,
    referenceOutput: torch.Tensor,
    debug: bool = False,
    checkEquivalence: bool = False,
) -> Tuple[nn.Module, torch.Tensor]:
    """Inject custom forward implementations into the model."""
    printer = GraphModulePrinter()

    tracer = QuantTracer(debug=debug)

    transformations = [
        MHATransformation(),
        LinearTransformation(),
        ActivationTransformation(),
    ]

    executor = TransformationExecutor(transformations, debug=debug, tracer=tracer)
    transformedModel = executor.execute(model, exampleInput)

    fxModel = customBrevitasTrace(
        root=transformedModel,
        tracer=tracer,
    )
    fxModel.recompile()

    with torch.no_grad():
        output = fxModel(exampleInput)

    if checkEquivalence:
        outputToCompare = output[0] if isinstance(output, tuple) else output
        if torch.allclose(referenceOutput, outputToCompare, atol=1e-5):
            if debug:
                print(cc.success("Injection of New Modules: output is consistent"))
        else:
            raise RuntimeError(
                cc.error("Injection of New Modules changed the output significantly")
            )

    if debug:
        print(cc.header("2. Network after Injection of New Modules"))
        printer.printTabular(fxModel)
        print()

    return fxModel, output
