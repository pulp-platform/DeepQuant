# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Executor module for handling transformation sequences in the Brevitas export process.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .Base import TransformationPass
from ..CustomTracer import CustomBrevitasTracer

# ANSI color codes
BLUE = "\033[94m"
RED = "\033[91m"
ENDC = "\033[0m"


class TransformationExecutor:
    """
    Manages and executes a sequence of model transformations.

    The executor applies each transformation in sequence, validating that model outputs
    remain consistent after each transformation step.
    """

    def __init__(
        self,
        transformations: List[TransformationPass],
        debug: bool = False,
        tracer: Optional[CustomBrevitasTracer] = None,
    ) -> None:
        """
        Initialize the transformation executor.

        Args:
            transformations: List of transformation passes to apply.
            debug: Whether to print debug information during execution.
            tracer: Optional CustomBrevitasTracer instance for module registration.
        """
        self.transformations = transformations
        self.debug = debug
        self.tracer = tracer

    def execute(self, model: nn.Module, exampleInput: torch.Tensor) -> nn.Module:
        """
        Execute all transformations on the model in sequence.

        For each transformation:
        1. Apply the transformation
        2. Validate that model outputs remain consistent
        3. Update the reference output for the next transformation

        Args:
            model: The PyTorch model to transform.
            example_input: A representative input tensor for validation.

        Returns:
            nn.Module: The transformed model.

        Raises:
            RuntimeError: If any transformation results in output mismatch.
        """
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
                            f"{RED} ✗ {transformation.__class__.__name__} failed - outputs mismatch{ENDC}"
                        )

                    if self.debug:
                        print(
                            f"{BLUE} ✓ {transformation.__class__.__name__} transformation successful\n{ENDC}"
                            f"      leafClasses: {self.tracer.leafClasses}\n"
                            f"      nonLeafClasses: {self.tracer.nonLeafClasses}\n"
                        )

                    outputBefore = outputAfter

        return model
