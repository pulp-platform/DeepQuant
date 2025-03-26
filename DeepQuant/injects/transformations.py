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

Each transformation class implements the abstract inject_forward method from TransformationPass
to define its specific module transformation logic.
"""

import torch.nn as nn
from typing import Optional
from brevitas.nn.quant_layer import (
    QuantWeightBiasInputOutputLayer,
    QuantNonLinearActLayer,
)
from brevitas.nn.quant_mha import QuantMultiheadAttention
from brevitas.nn.quant_activation import QuantIdentity

from .base import TransformationPass
from ..custom_forwards.linear import InnerForwardImplWrapperLinear, quantWBIOL_forward
from ..custom_forwards.multiheadattention import unrolled_quant_mha_forward
from ..custom_tracer import CustomBrevitasTracer
from ..custom_forwards.activations import (
    InnerForwardImplWrapperActivation,
    quant_activation_forward,
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
            module_cls=QuantWeightBiasInputOutputLayer,
            validation_tol=1e-6,
        )

    def inject_forward(
        self, module: nn.Module, tracer: Optional[CustomBrevitasTracer] = None
    ) -> None:
        """
        Inject custom forward implementation for linear layers.

        Args:
            module: The linear module to transform.
            tracer: Optional tracer for registering transformed modules.
        """
        module.wrapped_inner_forward_impl = InnerForwardImplWrapperLinear(
            module.inner_forward_impl
        )
        module.forward = quantWBIOL_forward.__get__(module)

        if tracer:
            tracer.register_leaf_module(InnerForwardImplWrapperLinear)
            tracer.register_non_leaf_module(QuantWeightBiasInputOutputLayer)


class ActivationTransformation(TransformationPass):
    """
    Transformation pass for quantized activation functions.

    Replaces the default forward with an unrolled implementation that exposes
    the input quantization and activation quantization steps.
    """

    def __init__(self) -> None:
        """Initialize the activation transformation pass."""
        super().__init__(
            module_cls=QuantNonLinearActLayer,
            validation_tol=1e-6,
        )

    def inject_forward(
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
            act_instance = module.act_impl()  # e.g. nn.ReLU()
        else:
            act_instance = nn.Identity()

        module.wrapped_act_impl = InnerForwardImplWrapperActivation(act_instance)
        module.forward = quant_activation_forward.__get__(module)

        if tracer:
            tracer.register_leaf_module(InnerForwardImplWrapperActivation)
            tracer.register_non_leaf_module(QuantNonLinearActLayer)


class MHATransformation(TransformationPass):
    """
    Transformation pass for quantized multi-head attention layers.

    Replaces the default forward with an unrolled implementation that exposes
    all attention operations and their associated quantization steps.
    """

    def __init__(self) -> None:
        """Initialize the MHA transformation pass."""
        super().__init__(
            module_cls=QuantMultiheadAttention,
            validation_tol=1e-5,
        )

    def inject_forward(
        self, module: nn.Module, tracer: Optional[CustomBrevitasTracer] = None
    ) -> None:
        """
        Inject custom forward implementation for MHA layers.

        Args:
            module: The MHA module to transform.
            tracer: Optional tracer for registering transformed modules.
        """
        module.forward = unrolled_quant_mha_forward.__get__(module)

        if tracer:
            tracer.register_non_leaf_module(QuantMultiheadAttention)
