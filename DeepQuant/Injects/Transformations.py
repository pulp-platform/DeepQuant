# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Transformation classes for different types of Brevitas modules.

This module provides specific transformation passes for each type of quantized module:
- Linear layers (QuantLinear, QuantConv2d)
- Activation functions (QuantReLU, QuantSigmoid)
- Multi-head attention (QuantMultiheadAttention)

Each transformation class implements the abstract injectForward method from TransformationPass
to define its specific module transformation logic.
"""

import torch.nn as nn
from typing import Optional
from brevitas.nn.quant_layer import (
    QuantWeightBiasInputOutputLayer,
    QuantNonLinearActLayer,
)
from brevitas.nn.quant_mha import QuantMultiheadAttention

from .Base import TransformationPass
from ..CustomForwards.Linear import InnerForwardImplWrapperLinear, quantWBIOLForward
from ..CustomForwards.MultiHeadAttention import unrolledQuantMhaForward
from ..CustomTracer import CustomBrevitasTracer
from ..CustomForwards.Activations import (
    InnerForwardImplWrapperActivation,
    quantActivationForward,
)


class LinearTransformation(TransformationPass):
    """
    Transformation pass for quantized linear layers (QuantLinear, QuantConv2d).

    Replaces the default forward with an unrolled implementation that exposes
    all quantization steps in the computation graph.
    """

    def __init__(self) -> None:
        """Initialize the linear transformation pass."""
        super().__init__(
            moduleCls=QuantWeightBiasInputOutputLayer,
            validationTol=1e-6,
        )

    def injectForward(
        self, module: nn.Module, tracer: Optional[CustomBrevitasTracer] = None
    ) -> None:
        """
        Inject custom forward implementation for linear layers.

        Args:
            module: The linear module to transform.
            tracer: Optional tracer for registering transformed modules.
        """
        module.wrappedInnerForwardImpl = InnerForwardImplWrapperLinear(
            module.inner_forward_impl
        )
        module.forward = quantWBIOLForward.__get__(module)

        if tracer:
            tracer.registerLeafModule(InnerForwardImplWrapperLinear)
            tracer.registerNonLeafModule(QuantWeightBiasInputOutputLayer)


class ActivationTransformation(TransformationPass):
    """
    Transformation pass for quantized activation functions.

    Replaces the default forward with an unrolled implementation that exposes
    the input quantization and activation quantization steps.
    """

    def __init__(self) -> None:
        """Initialize the activation transformation pass."""
        super().__init__(
            moduleCls=QuantNonLinearActLayer,
            validationTol=1e-6,
        )

    def injectForward(
        self, module: nn.Module, tracer: Optional[CustomBrevitasTracer] = None
    ) -> None:
        """
        Inject custom forward implementation for activation layers.

        This method instantiates the original activation function (if provided) and
        wraps it using InnerForwardImplWrapperActivation, then overrides the forward method.

        Args:
            module: The activation module to transform.
            tracer: Optional tracer for registering transformed modules.
        """
        # If the activation implementation was provided (e.g. nn.ReLU for QuantReLU),
        # instantiate it. Otherwise, default to an identity.
        if hasattr(module, "act_impl") and module.act_impl is not None:
            actInstance = module.act_impl()  # e.g. nn.ReLU()
        else:
            actInstance = nn.Identity()

        module.wrappedActImpl = InnerForwardImplWrapperActivation(actInstance)
        module.forward = quantActivationForward.__get__(module)

        if tracer:
            tracer.registerLeafModule(InnerForwardImplWrapperActivation)
            tracer.registerNonLeafModule(QuantNonLinearActLayer)


class MHATransformation(TransformationPass):
    """
    Transformation pass for quantized multi-head attention layers.

    Replaces the default forward with an unrolled implementation that exposes
    all attention operations and their associated quantization steps.
    """

    def __init__(self) -> None:
        """Initialize the MHA transformation pass."""
        super().__init__(
            moduleCls=QuantMultiheadAttention,
            validationTol=1e-5,
        )

    def injectForward(
        self, module: nn.Module, tracer: Optional[CustomBrevitasTracer] = None
    ) -> None:
        """
        Inject custom forward implementation for MHA layers.

        Args:
            module: The MHA module to transform.
            tracer: Optional tracer for registering transformed modules.
        """
        module.forward = unrolledQuantMhaForward.__get__(module)

        if tracer:
            tracer.registerNonLeafModule(QuantMultiheadAttention)
