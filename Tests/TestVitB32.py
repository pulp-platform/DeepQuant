# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import brevitas.nn as qnn
import pytest
import torch
import torch.nn as nn
import torchvision.models as models
from brevitas.graph.quantize import preprocess_for_quantize, quantize
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)

from DeepQuant import brevitasToTrueQuant


def prepare_vit_b_32(model: nn.Module) -> nn.Module:
    """
    Prepare a quantized ViT-B/32 model using Brevitas.
    """

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
                "weight_bit_width": 8,
            },
        ),
        nn.MultiheadAttention: (
            qnn.QuantMultiheadAttention,
            {
                "in_proj_input_quant": Int8ActPerTensorFloat,
                "in_proj_weight_quant": Int8WeightPerTensorFloat,
                "in_proj_bias_quant": Int32Bias,
                "attn_output_weights_quant": Uint8ActPerTensorFloat,
                "q_scaled_quant": Int8ActPerTensorFloat,
                "k_transposed_quant": Int8ActPerTensorFloat,
                "v_quant": Int8ActPerTensorFloat,
                "out_proj_input_quant": Int8ActPerTensorFloat,
                "out_proj_weight_quant": Int8WeightPerTensorFloat,
                "out_proj_bias_quant": Int32Bias,
                "out_proj_output_quant": Int8ActPerTensorFloat,
                "return_quant_tensor": True,
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
                "weight_bit_width": 8,
            },
        ),
    }

    quant_act_map = {
        nn.GELU: (
            qnn.QuantReLU,  # FBRANCASI: Approximating GELU with QuantReLU
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

    print("\nPreprocessing model for quantization...")
    model = preprocess_for_quantize(
        model,
        equalize_iters=10,
        equalize_scale_computation="range",
    )

    print("\nQuantizing model...")
    quantized_model = quantize(
        graph_model=model,
        compute_layer_map=compute_layer_map,
        quant_act_map=quant_act_map,
        quant_identity_map=quant_identity_map,
    )

    return quantized_model


@pytest.mark.ModelTests
def deepQuantTestViT():
    torch.manual_seed(42)
    sampleInput = torch.randn(1, 3, 224, 224)

    vit_model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
    vit_model.eval()

    print(f"\nTesting ViT-B/32 model with input shape: {sampleInput.shape}")

    quantized_vit = prepare_vit_b_32(vit_model)

    with torch.no_grad():
        output = quantized_vit(sampleInput)
        if isinstance(output, tuple):
            output = output[0]
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    brevitasToTrueQuant(quantized_vit, sampleInput, debug=True, checkEquivalence=False)
