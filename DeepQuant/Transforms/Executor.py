# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import torch
import torch.nn as nn
from typing import List, Optional
from .Base import TransformationPass
from ..Utils.CustomTracer import CustomBrevitasTracer
from ..Utils.ConsoleColor import ConsoleColor as cc


class TransformationExecutor:
    """Runs a list of passes and checks output drift after each step."""

    def __init__(
        self,
        transformations: List[TransformationPass],
        debug: bool = False,
        tracer: Optional[CustomBrevitasTracer] = None,
    ) -> None:
        self.transformations = transformations
        self.debug = debug
        self.tracer = tracer

    def execute(self, model: nn.Module, exampleInput: torch.Tensor) -> nn.Module:
        model.eval()
        with torch.no_grad():
            outputBefore = model(exampleInput)
            if isinstance(outputBefore, tuple):
                outputBefore = outputBefore[0]

            for transformation in self.transformations:
                if transformation.transform(model, tracer=self.tracer):
                    outputAfter = model(exampleInput)
                    if isinstance(outputAfter, tuple):
                        outputAfter = outputAfter[0]

                    if not transformation.validateTransformation(
                        outputBefore, outputAfter
                    ):
                        raise RuntimeError(
                            cc.wrap(
                                f" ✗ {transformation.__class__.__name__} failed - outputs mismatch",
                                cc.red,
                            )
                        )

                    if self.debug:
                        print(
                            cc.wrap(
                                f" ✓ {transformation.__class__.__name__} transformation successful\n",
                                cc.blue,
                            ),
                            f"      leafClasses: {self.tracer.leafClasses}\n"
                            f"      nonLeafClasses: {self.tracer.nonLeafClasses}\n",
                        )
                    outputBefore = outputAfter

        return model
