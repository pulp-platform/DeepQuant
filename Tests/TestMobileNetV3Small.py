# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Victor Juing <jungvi@ethz.ch>

import pytest
import torch
import torch.nn as nn
import torchvision.models as models
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.per_input import AdaptiveAvgPoolToAvgPool
import brevitas.nn as qnn
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)
from brevitas.graph.quantize import quantize

from DeepQuant.ExportBrevitas import exportBrevitas


def prepareMBNetV3Model() -> nn.Module:
    """
    Prepare a quantized MobileNetV3Small model for testing.
    Steps:
      1) Load the torchvision MobileNetV3Small.
      2) Convert it to eval mode.
      3) Preprocess and adapt average pooling.
      4) Quantize it using Brevitas.

    Returns:
        A quantized MobileNetV3Small model ready for export tests.
    """
    baseModel = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    baseModel = baseModel.eval()

    computeLayerMap = {
        nn.Conv2d: (
            qnn.QuantConv2d,
            {
                "input_quant": Int8ActPerTensorFloat,
                "weight_quant": Int8WeightPerTensorFloat,
                "output_quant": Int8ActPerTensorFloat,
                "bias_quant": Int32Bias,
                "bias": True,
                "return_quant_tensor": True,
                "output_bit_width": 8,
                "weight_bit_width": 4,
            },
        ),
        nn.Linear: (
            qnn.QuantLinear,
            {
                "input_quant": Int8ActPerTensorFloat,
                "weight_quant": Int8WeightPerTensorFloat,
                "output_quant": Int8ActPerTensorFloat,
                "bias_quant": Int32Bias,
                "bias": True,
                "return_quant_tensor": True,
                "output_bit_width": 8,
                "weight_bit_width": 4,
            },
        ),
    }

    quantActMap = {
        nn.ReLU: (
            qnn.QuantReLU,
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 8,
            },
        ),
    }

    quantIdentityMap = {
        "signed": (
            qnn.QuantIdentity,
            {
                "act_quant": Int8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 8,
            },
        ),
        "unsigned": (
            qnn.QuantIdentity,
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 8,
            },
        ),
    }

    baseModel = preprocess_for_quantize(
        baseModel, equalize_iters=20, equalize_scale_computation="range"
    )
    baseModel = AdaptiveAvgPoolToAvgPool().apply(
        baseModel, torch.ones(1, 3, 224, 224)
    )

    quantizedModel = quantize(
        graph_model=baseModel,
        compute_layer_map=computeLayerMap,
        quant_act_map=quantActMap,
        quant_identity_map=quantIdentityMap,
    )

    return quantizedModel


@pytest.mark.ModelTests
def deepQuantTestMobileNetV3Small() -> None:

    torch.manual_seed(42)

    quantizedModel = prepareMBNetV3Model()
    sampleInput = torch.randn(1, 3, 224, 224)

    exportBrevitas(quantizedModel, sampleInput, debug=True)
