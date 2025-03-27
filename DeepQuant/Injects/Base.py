# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Base transformation infrastructure for the Brevitas export process.

This module provides the foundational TransformationPass class that handles:
- Module type matching
- Forward method injection
- Output validation
- Recursive submodule transformation
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Tuple
from ..CustomTracer import CustomBrevitasTracer


class TransformationPass(ABC):
    """
    Generic transformation pass for modifying Brevitas modules.

    A transformation pass targets specific module types and applies custom forward
    implementations while ensuring output consistency.
    """

    def __init__(
        self,
        moduleCls: Union[type, Tuple[type, ...]],
        validationTol: float = 1e-6,
    ) -> None:
        """
        Initialize a transformation pass.

        Args:
            module_cls: Module class(es) this transformation targets.
            injection_fn: Function that modifies the module's forward pass.
            validation_tol: Tolerance for numerical comparison in validation.
        """
        self.moduleCls = moduleCls
        self.validationTol = validationTol

    def checkModuleType(self, module: nn.Module) -> bool:
        """
        Check if a module is an instance of the target class(es).

        Args:
            module: Module to check.

        Returns:
            bool: True if module is an instance of self.module_cls.
        """
        return isinstance(module, self.moduleCls)

    @abstractmethod
    def injectForward(
        self, module: nn.Module, tracer: Optional[CustomBrevitasTracer] = None
    ) -> None:
        """
        Inject the custom forward implementation into a module.

        Args:
            module: Module whose forward method will be replaced.
            tracer: Optional tracer for registering module classes.
        """
        pass

    def validateTransformation(
        self, outputBefore: Any, outputAfter: Any, atol: Optional[float] = None
    ) -> bool:
        """
        Validate transformation by comparing outputs.

        Args:
            output_before: Model output before transformation.
            output_after: Model output after transformation.
            atol: Optional custom tolerance for comparison.

        Returns:
            bool: True if outputs match within tolerance.
        """
        if atol is None:
            atol = self.validationTol
        return torch.allclose(outputBefore, outputAfter, atol=atol)

    def transform(
        self, model: nn.Module, tracer: Optional[CustomBrevitasTracer] = None
    ) -> bool:
        """
        Apply the transformation to all matching submodules.

        Args:
            model: Model containing submodules to transform.
            tracer: Optional tracer for registering transformed modules.

        Returns:
            bool: True if any modules were transformed.
        """
        transformDone = False
        for _, submodule in model.named_modules():
            if self.checkModuleType(submodule):
                self.injectForward(submodule, tracer)
                transformDone = True
        return transformDone
