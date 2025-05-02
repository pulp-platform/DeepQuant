# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import brevitas.nn as qnn
import pytest
import torch
import torch.nn as nn
from brevitas.graph.quantize import preprocess_for_quantize, quantize
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)

from DeepQuant import brevitasToTrueQuant


def prepareYOLOv5Backbone() -> nn.Module:
    """Prepare a quantized partial YOLOv5 model for testing."""
    from ultralytics import YOLO

    model = YOLO("Models/yolov5nu.pt")
    pytorchModel = model.model

    # FBRANCASI: Just first few layers for simplicity
    backbone = pytorchModel.model[0:4]

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
        nn.SiLU: (
            qnn.QuantReLU,  # FBRANCASI: As a substitute for now
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 8,
            },
        ),
        nn.ReLU: (
            qnn.QuantReLU,
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 8,
            },
        ),
        nn.LeakyReLU: (
            qnn.QuantReLU,  # FBRANCASI: As a substitute for now
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

    backbone = preprocess_for_quantize(
        backbone, equalize_iters=10, equalize_scale_computation="range"
    )

    quantizedModel = quantize(
        graph_model=backbone,
        compute_layer_map=computeLayerMap,
        quant_act_map=quantActMap,
        quant_identity_map=quantIdentityMap,
    )

    return quantizedModel


@pytest.mark.ModelTests
def deepQuantTestYOLOv5():
    torch.manual_seed(42)
    quantizedModel = prepareYOLOv5Backbone()
    sampleInput = torch.randn(1, 3, 128, 128)
    quantizedModel.eval()
    brevitasToTrueQuant(quantizedModel, sampleInput, debug=True)
