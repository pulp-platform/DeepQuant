# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import pytest
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)
from brevitas.graph.quantize import quantize, preprocess_for_quantize

from DeepQuant import exportQuantModel


def prepareYOLOv5Backbone() -> nn.Module:
    from ultralytics import YOLO

    model = YOLO("Models/yolov5n.pt")
    pytorch_model = model.model

    backbone = pytorch_model.model[
        0:4
    ]  # FBRANCASI: Just first few layers for simplicity

    compute_layer_map = {
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

    quant_act_map = {
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

    quant_identity_map = {
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

    quantized_model = quantize(
        graph_model=backbone,
        compute_layer_map=compute_layer_map,
        quant_act_map=quant_act_map,
        quant_identity_map=quant_identity_map,
    )

    return quantized_model


@pytest.mark.ModelTests
def deepQuantTestYOLOv5():

    torch.manual_seed(42)

    quantizedModel = prepareYOLOv5Backbone()
    sample_input = torch.randn(1, 3, 128, 128)

    quantizedModel.eval()

    exportQuantModel(quantizedModel, sample_input, debug=True)
