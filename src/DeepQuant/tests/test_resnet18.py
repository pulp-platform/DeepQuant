# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

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

from DeepQuant.export_brevitas import exportBrevitas


def prepare_resnet18_model() -> nn.Module:
    """
    Prepare a quantized ResNet18 model for testing.
    Steps:
      1) Load the torchvision ResNet18.
      2) Convert it to eval mode.
      3) Preprocess and adapt average pooling.
      4) Quantize it using Brevitas.

    Returns:
        A quantized ResNet18 model ready for export tests.
    """
    base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    base_model = base_model.eval()

    compute_layer_map = {
        # PROBLEM WITH TruncAvgPool2d USED WITH quant_inference_mode
        # nn.AvgPool2d: (qnn.TruncAvgPool2d, {"return_quant_tensor": True}),
        nn.Conv2d: (
            qnn.QuantConv2d,
            {
                "weight_quant": Int8WeightPerTensorFloat,
                "output_quant": Int8ActPerTensorFloat,
                "bias_quant": Int32Bias,
                "return_quant_tensor": True,
                "output_bit_width": 8,
                "weight_bit_width": 8,
            },
        ),
        nn.Linear: (
            qnn.QuantLinear,
            {
                "weight_quant": Int8WeightPerTensorFloat,
                "output_quant": Int8ActPerTensorFloat,
                "bias_quant": Int32Bias,
                "return_quant_tensor": True,
                "output_bit_width": 8,
                "weight_bit_width": 8,
            },
        ),
    }

    quant_act_map = {
        nn.ReLU: (
            qnn.QuantReLU,
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 7,
            },
        ),
    }

    quant_identity_map = {
        "signed": (
            qnn.QuantIdentity,
            {
                "act_quant": Int8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 7,
            },
        ),
        "unsigned": (
            qnn.QuantIdentity,
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 7,
            },
        ),
    }

    base_model = preprocess_for_quantize(
        base_model, equalize_iters=20, equalize_scale_computation="range"
    )
    base_model = AdaptiveAvgPoolToAvgPool().apply(
        base_model, torch.ones(1, 3, 224, 224)
    )

    quantized_resnet = quantize(
        base_model,
        compute_layer_map=compute_layer_map,
        quant_act_map=quant_act_map,
        quant_identity_map=quant_identity_map,
    )

    return quantized_resnet


def test_resnet18_quant_export() -> None:
    """
    Test function for exporting a quantized ResNet18 using BrevitasExporter.
    Validates the export process by running exportBrevitas and printing the FX graph.

    Returns:
        None
    """
    print("\n=== Testing BrevitasExporter with a quantized ResNet18 ===\n")

    torch.manual_seed(42)

    quantized_model = prepare_resnet18_model()
    sample_input = torch.randn(1, 3, 224, 224)

    fx_model = exportBrevitas(quantized_model, sample_input, debug=True)
