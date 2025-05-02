# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from DeepQuant.Utils.CustomTracer import QuantTracer


class TransformationPass(ABC):
    """Base class for module transformation passes."""

    def __init__(
        self,
        moduleCls: Union[type, Tuple[type, ...]],
        validationTol: float = 1e-6,
    ) -> None:
        self.moduleCls = moduleCls
        self.validationTol = validationTol

    def checkModuleType(self, module: nn.Module) -> bool:
        """Check if a module is an instance of the target class(es)."""
        return isinstance(module, self.moduleCls)

    @abstractmethod
    def injectForward(
        self, module: nn.Module, tracer: Optional[QuantTracer] = None
    ) -> None:
        """Inject the custom forward implementation into a module."""
        pass

    def validateTransformation(
        self, outputBefore: Any, outputAfter: Any, atol: Optional[float] = None
    ) -> bool:
        """Validate transformation by comparing outputs."""
        if atol is None:
            atol = self.validationTol
        return torch.allclose(outputBefore, outputAfter, atol=atol)

    def transform(self, model: nn.Module, tracer: Optional[QuantTracer] = None) -> bool:
        """Apply the transformation to all matching submodules."""
        transformDone = False
        for _, submodule in model.named_modules():
            if self.checkModuleType(submodule):
                self.injectForward(submodule, tracer)
                transformDone = True
        return transformDone
