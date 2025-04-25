# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from typing import Optional

import torch.nn as nn
from brevitas.nn.quant_layer import (
    QuantNonLinearActLayer,
    QuantWeightBiasInputOutputLayer,
)
from brevitas.nn.quant_mha import QuantMultiheadAttention

from DeepQuant.CustomForwards.Activations import WrapperActivation, activationForward
from DeepQuant.CustomForwards.Linear import WrapperLinear, linearForward
from DeepQuant.CustomForwards.MultiHeadAttention import mhaForward
from DeepQuant.Transforms.Base import TransformationPass
from DeepQuant.Utils.CustomTracer import QuantTracer


class LinearTransformation(TransformationPass):
    """Transforms quantized linear layers."""

    def __init__(self) -> None:
        super().__init__(
            moduleCls=QuantWeightBiasInputOutputLayer,
            validationTol=1e-6,
        )

    def injectForward(
        self, module: nn.Module, tracer: Optional[QuantTracer] = None
    ) -> None:
        """Inject custom forward for linear layers."""
        module.wrappedInnerForwardImpl = WrapperLinear(module.inner_forward_impl)
        module.forward = linearForward.__get__(module)

        if tracer:
            tracer.registerLeafModule(WrapperLinear)
            tracer.registerNonLeafModule(QuantWeightBiasInputOutputLayer)


class ActivationTransformation(TransformationPass):
    """Transforms quantized activation layers."""

    def __init__(self) -> None:
        super().__init__(
            moduleCls=QuantNonLinearActLayer,
            validationTol=1e-6,
        )

    def injectForward(
        self, module: nn.Module, tracer: Optional[QuantTracer] = None
    ) -> None:
        """Inject custom forward for activation layers."""
        # FBRANCASI: If the activation implementation was provided (e.g. nn.ReLU
        # for QuantReLU), instantiate it. Otherwise, default to an identity.
        if hasattr(module, "act_impl") and module.act_impl is not None:
            actInstance = module.act_impl()
        else:
            actInstance = nn.Identity()

        module.wrappedActImpl = WrapperActivation(actInstance)
        module.forward = activationForward.__get__(module)

        if tracer:
            tracer.registerLeafModule(WrapperActivation)
            tracer.registerNonLeafModule(QuantNonLinearActLayer)


class MHATransformation(TransformationPass):
    """Transforms quantized multi-head attention layers."""

    def __init__(self) -> None:
        super().__init__(
            moduleCls=QuantMultiheadAttention,
            validationTol=1e-5,
        )

    def injectForward(
        self, module: nn.Module, tracer: Optional[QuantTracer] = None
    ) -> None:
        """Inject custom forward for multi-head attention layers."""
        module.forward = mhaForward.__get__(module)

        if tracer:
            tracer.registerNonLeafModule(QuantMultiheadAttention)
