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

from Tests.Models.CCT import cct_2_3x2_32


def prepareCCT(model) -> nn.Module:
    """
    Prepare a quantized CCT model for testing with export support.
    """
    import operator

    from brevitas.fx.brevitas_tracer import symbolic_trace
    from brevitas.graph.utils import replace_all_uses_except

    # First trace the model
    if not hasattr(model, "graph"):
        model = symbolic_trace(model)

    print("=== FIXING QUANTIZATION ISSUES ===")

    # Collect all modifications first, then apply them
    transpose_fixes = []
    qkv_fixes = []

    # Fix 1: Find transpose -> add patterns
    for node in model.graph.nodes:
        if node.op == "call_method" and node.target == "transpose":
            for user in node.users:
                if (
                    "add" in user.name
                    or user.target
                    in [
                        torch.add,
                        operator.add,
                        operator.iadd,
                        operator.__add__,
                        operator.__iadd__,
                    ]
                    or (user.op == "call_method" and user.target in ["add", "add_"])
                ):
                    transpose_fixes.append((node, user))
                    break

    # Fix 2: Find QKV -> reshape patterns
    for node in model.graph.nodes:
        if node.op == "call_module" and "qkv" in node.target:
            for user in node.users:
                if user.op == "call_method" and user.target == "reshape":
                    qkv_fixes.append((node, user))
                    break

    # Apply transpose fixes
    print(f"\nApplying {len(transpose_fixes)} transpose fixes...")
    for node, user in transpose_fixes:
        print(f"  Fixing: {node.name} -> {user.name}")

        # Create a QuantIdentity
        quant_identity = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

        # Add to model
        quant_name = f"{node.name}_quant_fix"
        model.add_module(quant_name, quant_identity)

        # Insert in the graph after transpose
        with model.graph.inserting_after(node):
            quant_node = model.graph.call_module(quant_name, args=(node,))

        # Replace uses
        replace_all_uses_except(node, quant_node, [quant_node])

    # Apply QKV fixes
    print(f"\nApplying {len(qkv_fixes)} QKV fixes...")
    for node, reshape_user in qkv_fixes:
        print(f"  Fixing: {node.name} -> {reshape_user.name}")

        # Create a QuantIdentity to handle the tensor properly
        quant_identity = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,  # Important: return regular tensor for reshape
        )

        # Add to model
        quant_name = f"{node.name}_reshape_fix"
        model.add_module(quant_name, quant_identity)

        # Insert in the graph between qkv and reshape
        with model.graph.inserting_after(node):
            quant_node = model.graph.call_module(quant_name, args=(node,))

        # Update reshape to use the quant_node output
        reshape_user.update_arg(0, quant_node)

    # Recompile graph only once after all modifications
    model.recompile()
    model.graph.lint()

    print("\n=== Graph modifications complete ===")

    # Define quantization mappings
    computeLayerMap = {
        nn.Conv2d: (
            qnn.QuantConv2d,
            {
                "input_quant": Int8ActPerTensorFloat,
                "weight_quant": Int8WeightPerTensorFloat,
                "output_quant": Int8ActPerTensorFloat,
                "bias_quant": Int32Bias,
                "bias": False,
                "return_quant_tensor": True,
                "output_bit_width": 8,
                "weight_bit_width": 4,
            },
        ),
        nn.MultiheadAttention: (
            qnn.QuantMultiheadAttention,
            {
                "in_proj_weight_quant": Int8WeightPerTensorFloat,
                "in_proj_bias_quant": Int32Bias,
                "attn_output_weights_quant": Uint8ActPerTensorFloat,
                "q_scaled_quant": Int8ActPerTensorFloat,
                "k_transposed_quant": Int8ActPerTensorFloat,
                "v_quant": Int8ActPerTensorFloat,
                "out_proj_input_quant": Int8ActPerTensorFloat,
                "out_proj_weight_quant": Int8WeightPerTensorFloat,
                "out_proj_bias_quant": Int32Bias,
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
        nn.GELU: (
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

    # Preprocess model
    model = preprocess_for_quantize(
        model,
        equalize_iters=10,
        equalize_scale_computation="range",
        trace_model=False,  # Already traced
    )

    # Quantize model
    quantizedModel = quantize(
        graph_model=model,
        compute_layer_map=computeLayerMap,
        quant_act_map=quantActMap,
        quant_identity_map=quantIdentityMap,
    )

    return quantizedModel


@pytest.mark.ModelTests
def deepQuantTestCCT():
    torch.manual_seed(42)
    sampleInput = torch.randn(1, 3, 32, 32)

    model = cct_2_3x2_32()  # 2 encoder layers, kernel dim 3, 2 convs, 32x32
    model.eval()

    print(model)

    quantizedModel = prepareCCT(model)

    # Test the quantized model
    print(f"\nTesting with input shape: {sampleInput.shape}")
    with torch.no_grad():
        output = quantizedModel(sampleInput)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
