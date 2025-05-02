# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from typing import List, Optional

import torch
import torch.nn as nn

from DeepQuant.Transforms.Base import TransformationPass
from DeepQuant.Utils.ConsoleFormatter import ConsoleColor as cc
from DeepQuant.Utils.CustomTracer import QuantTracer


class TransformationExecutor:
    """Runs a sequence of transformation passes."""

    def __init__(
        self,
        transformations: List[TransformationPass],
        debug: bool = False,
        tracer: Optional[QuantTracer] = None,
    ) -> None:
        self.transformations = transformations
        self.debug = debug
        self.tracer = tracer

    def execute(self, model: nn.Module, exampleInput: torch.Tensor) -> nn.Module:
        """Execute all transformations on the model."""
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
                            cc.error(
                                f"{transformation.__class__.__name__} failed - outputs mismatch"
                            )
                        )

                    if self.debug:
                        print(
                            cc.success(
                                f"{transformation.__class__.__name__} transformation successful"
                            )
                        )
                        if self.tracer:
                            print(f"    leafClasses: {self.tracer.leafClasses}")
                            print(f"    nonLeafClasses: {self.tracer.nonLeafClasses}")

                    outputBefore = outputAfter

        return model
