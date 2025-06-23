# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from typing import Tuple

import brevitas.nn as qnn
import pytest
import torch
import torch.nn as nn
from brevitas.fx.brevitas_tracer import symbolic_trace
from brevitas.graph.quantize import preprocess_for_quantize, quantize
from brevitas.graph.utils import replace_all_uses_except
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)

from DeepQuant.Transforms.Executor import TransformationExecutor
from DeepQuant.Transforms.Transformations import LinearTransformation, MHATransformation
from DeepQuant.Utils.ConsoleFormatter import ConsoleColor as cc
from DeepQuant.Utils.CustomTracer import QuantTracer, customBrevitasTrace
from DeepQuant.Utils.GraphPrinter import GraphModulePrinter
from Tests.Models.CCT import cct_2_3x2_32


def injectCustomForwards(
    model: nn.Module,
    exampleInput: torch.Tensor,
    referenceOutput: torch.Tensor,
    debug: bool = False,
    checkEquivalence: bool = False,
) -> Tuple[nn.Module, torch.Tensor]:
    """Custom inject function for CCT that excludes ActivationTransformation."""
    printer = GraphModulePrinter()

    tracer = QuantTracer(debug=debug)

    transformations = [
        MHATransformation(),
        LinearTransformation(),
        # ActivationTransformation(),  # FBRANCASI: Commented out for CCT compatibility
    ]

    executor = TransformationExecutor(transformations, debug=debug, tracer=tracer)
    transformedModel = executor.execute(model, exampleInput)

    fxModel = customBrevitasTrace(
        root=transformedModel,
        tracer=tracer,
    )
    fxModel.recompile()

    with torch.no_grad():
        output = fxModel(exampleInput)

    if checkEquivalence:
        if torch.allclose(referenceOutput, output, atol=1e-5):
            if debug:
                print(cc.success("Injection of New Modules: output is consistent"))
        else:
            raise RuntimeError(
                cc.error("Injection of New Modules changed the output significantly")
            )

    if debug:
        print(cc.header("2. Network after Injection of New Modules"))
        printer.printTabular(fxModel)
        print()

    return fxModel, output


def prepareCCT(model) -> nn.Module:
    """
    Prepare a quantized CCT model for testing with export support.
    """

    if not hasattr(model, "graph"):
        model = symbolic_trace(model)

    print("=== FIXING QUANTIZATION ISSUES ===")

    transpose_fixes = []
    qkv_fixes = []

    # FBRANCASI: Fix 1, Find transpose -> add patterns
    for node in model.graph.nodes:
        if node.op == "call_method" and node.target == "transpose":
            for user in node.users:
                if (
                    "add" in user.name
                    or user.target in [torch.add]
                    or (user.op == "call_method" and user.target in ["add", "add_"])
                ):
                    transpose_fixes.append((node, user))
                    break

    # FBRANCASI: Fix 2, Find QKV -> reshape patterns
    for node in model.graph.nodes:
        if node.op == "call_module" and "qkv" in node.target:
            for user in node.users:
                if user.op == "call_method" and user.target == "reshape":
                    qkv_fixes.append((node, user))
                    break

    # FBRANCASI: Apply transpose fixes
    print(f"\nApplying {len(transpose_fixes)} transpose fixes...")
    for node, user in transpose_fixes:
        print(f"  Fixing: {node.name} -> {user.name}")

        quant_identity = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )

        quant_name = f"{node.name}_quant_fix"
        model.add_module(quant_name, quant_identity)

        with model.graph.inserting_after(node):
            quant_node = model.graph.call_module(quant_name, args=(node,))

        # Replace uses
        replace_all_uses_except(node, quant_node, [quant_node])

    # FBRANCASI: Apply QKV fixes
    print(f"\nApplying {len(qkv_fixes)} QKV fixes...")
    for node, reshape_user in qkv_fixes:
        print(f"  Fixing: {node.name} -> {reshape_user.name}")

        quant_identity = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,  # FBRANCASI: return regular tensor for reshape
        )

        quant_name = f"{node.name}_reshape_fix"
        model.add_module(quant_name, quant_identity)

        with model.graph.inserting_after(node):
            quant_node = model.graph.call_module(quant_name, args=(node,))

        reshape_user.update_arg(0, quant_node)

    model.recompile()
    model.graph.lint()

    print("\n=== GRAPH MODIFICATION COMPLETE ===")

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

    quantActMap = {}

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

    model = preprocess_for_quantize(
        model,
        equalize_iters=10,
        equalize_scale_computation="range",
        trace_model=False,  # FBRANCASI: Already traced
    )

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

    model = cct_2_3x2_32()  # FBRANCASI: 2 encoder layers, kernel dim 3, 2 convs, 32x32
    model.eval()

    print(model)

    quantizedModel = prepareCCT(model)

    print(f"\nTesting the Quantized Model with input shape: {sampleInput.shape}")
    with torch.no_grad():
        output = quantizedModel(sampleInput)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # FBRANCASI: Override the injectCustomForwards function in the module before DeepQuant.Export imports it
    import DeepQuant.Pipeline.Injection as injection_module

    # FBRANCASI: Store original function
    original_inject = injection_module.injectCustomForwards

    # FBRANCASI: Override with our custom function
    injection_module.injectCustomForwards = injectCustomForwards

    # FBRANCASI: Force reload of Export module to pick up the override
    import importlib

    import DeepQuant.Export

    importlib.reload(DeepQuant.Export)

    try:
        from DeepQuant.Export import brevitasToTrueQuant

        brevitasToTrueQuant(quantizedModel, sampleInput, debug=True)
    finally:
        # FBRANCASI: Restore original function and reload Export module again
        injection_module.injectCustomForwards = original_inject
        importlib.reload(DeepQuant.Export)
