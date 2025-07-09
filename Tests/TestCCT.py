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
from Tests.Models.CCT.CCT.cct import cct_2_3x2_32


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
    matmul_fixes = []

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

    # FBRANCASI: Fix 3, Find matmul operations that need dequantization
    for node in model.graph.nodes:
        if node.op == "call_function" and node.target == torch.matmul:
            matmul_fixes.append(node)
        elif node.op == "call_method" and node.target == "matmul":
            matmul_fixes.append(node)
        elif (
            node.op == "call_function"
            and hasattr(node.target, "__name__")
            and node.target.__name__ == "matmul"
        ):
            matmul_fixes.append(node)
        elif hasattr(node, "name") and "matmul" in node.name:
            matmul_fixes.append(node)
        elif (
            node.op == "call_function"
            and hasattr(node.target, "__module__")
            and node.target.__module__ == "operator"
            and hasattr(node.target, "__name__")
            and node.target.__name__ == "matmul"
        ):
            matmul_fixes.append(node)

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

    # FBRANCASI: Apply matmul fixes
    print(f"\nApplying {len(matmul_fixes)} matmul fixes...")
    for node in matmul_fixes:
        print(
            f"  Fixing matmul: {node.name}, args: {[arg.name if hasattr(arg, 'name') else str(arg) for arg in node.args]}"
        )

        # FBRANCASI: Add dequantization before both inputs of matmul
        for i, arg in enumerate(node.args):
            if isinstance(arg, torch.fx.Node):
                print(f"    Processing arg {i}: {arg.name}")
                dequant_identity = qnn.QuantIdentity(
                    act_quant=Int8ActPerTensorFloat,
                    return_quant_tensor=False,  # FBRANCASI: Return regular tensor for matmul
                )

                dequant_name = f"{arg.name}_matmul_dequant_{i}"
                model.add_module(dequant_name, dequant_identity)

                with model.graph.inserting_before(node):
                    dequant_node = model.graph.call_module(dequant_name, args=(arg,))

                # Update the matmul argument
                node.update_arg(i, dequant_node)
                print(f"    Updated arg {i} to: {dequant_node.name}")

    model.recompile()
    model.graph.lint()

    print("\n=== GRAPH MODIFICATION COMPLETE ===")

    # Debug: Print graph structure to understand the flow
    print("\n=== DEBUG: Graph structure after fixes ===")
    for node in model.graph.nodes:
        if (
            "matmul" in node.name
            or (node.op == "call_method" and node.target == "transpose")
            or "permute" in node.name
        ):
            print(
                f"Node: {node.name}, op: {node.op}, target: {node.target}, args: {[arg.name if hasattr(arg, 'name') else str(arg) for arg in node.args]}"
            )
            # Print users of permute and transpose nodes
            if "permute" in node.name or (
                node.op == "call_method" and node.target == "transpose"
            ):
                print(f"  Users: {[user.name for user in node.users]}")

    # FBRANCASI: First pass - identify which Linear layers feed into matmul through permute/transpose
    linear_to_matmul = set()
    for node in model.graph.nodes:
        if hasattr(node, "name") and "matmul" in node.name:
            # Trace back through the args to find Linear layers
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    # Check if this path leads back to a linear layer
                    current = arg
                    visited = set()
                    while current and current not in visited:
                        visited.add(current)
                        if current.op == "call_module" and any(
                            proj in current.target
                            for proj in ["q_proj", "k_proj", "v_proj"]
                        ):
                            linear_to_matmul.add(current.target)
                            break
                        # Trace back through the first argument
                        if current.args and isinstance(current.args[0], torch.fx.Node):
                            current = current.args[0]
                        else:
                            break

    print(f"\nLinear layers that feed into matmul: {linear_to_matmul}")

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
                "weight_bit_width": 8,
            },
        ),
        nn.Linear: (
            qnn.QuantLinear,
            {
                "input_quant": Int8ActPerTensorFloat,
                "weight_quant": Int8WeightPerTensorFloat,
                "output_quant": Int8ActPerTensorFloat,
                "bias_quant": Int32Bias,
                "return_quant_tensor": True,  # FBRANCASI: We'll handle this specially for q,k,v projections
                "output_bit_width": 8,
                "weight_bit_width": 8,
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

    # FBRANCASI: Apply post-quantization fixes for matmul operations
    print("\n=== POST-QUANTIZATION FIXES ===")

    nodes_needing_dequant = set()

    node_map = {node.name: node for node in quantizedModel.graph.nodes}

    import operator

    for node in quantizedModel.graph.nodes:
        # FBRANCASI: Look for @ operator (represented as call_function with operator.matmul)
        is_matmul = False
        if node.op == "call_function":
            if node.target == operator.matmul:
                is_matmul = True
            elif node.target == torch.matmul:
                is_matmul = True
            elif hasattr(node.target, "__name__") and node.target.__name__ == "matmul":
                is_matmul = True

        if is_matmul:
            print(f"\nFound matmul node: {node.name}")
            print(f"  Target: {node.target}")
            print(f"  Args: {node.args}")
            print(f"  Arg types: {[type(arg) for arg in node.args]}")

            # FBRANCASI: Mark both arguments as needing dequantization
            for i, arg in enumerate(node.args):
                print(f"    Checking arg {i}: type={type(arg)}")
                if hasattr(arg, "name") and hasattr(arg, "op"):
                    nodes_needing_dequant.add(arg)
                    print(f"    Added node to dequant: {arg.name}")
                else:
                    print(f"    Skipped arg {i}: {arg}")

    print(f"\nNodes needing dequantization: {[n.name for n in nodes_needing_dequant]}")

    # FBRANCASI: Insert dequantization for each node that feeds into matmul
    dequant_nodes = {}
    for node in nodes_needing_dequant:
        print(f"\nAdding dequantization after node: {node.name}")

        dequant_identity = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,  # FBRANCASI: Return regular tensor for matmul
        )

        dequant_name = f"{node.name}_dequant_for_matmul"
        quantizedModel.add_module(dequant_name, dequant_identity)

        with quantizedModel.graph.inserting_after(node):
            dequant_node = quantizedModel.graph.call_module(dequant_name, args=(node,))

        dequant_nodes[node] = dequant_node

        for user in list(node.users):
            is_matmul_user = False
            if user.op == "call_function":
                if user.target == operator.matmul or user.target == torch.matmul:
                    is_matmul_user = True
                elif (
                    hasattr(user.target, "__name__")
                    and user.target.__name__ == "matmul"
                ):
                    is_matmul_user = True
                elif (
                    hasattr(user.target, "__module__")
                    and user.target.__module__ == "operator"
                    and hasattr(user.target, "__name__")
                    and user.target.__name__ == "matmul"
                ):
                    is_matmul_user = True

            if is_matmul_user:
                print(f"  Updating matmul {user.name} to use dequantized input")
                new_args = []
                for i, arg in enumerate(user.args):
                    if arg == node:
                        new_args.append(dequant_node)
                        print(
                            f"    Updated arg {i} from {node.name} to {dequant_node.name}"
                        )
                    else:
                        new_args.append(arg)
                user.args = tuple(new_args)

    quantizedModel.recompile()
    quantizedModel.graph.lint()

    print("\n=== POST-QUANTIZATION FIXES COMPLETE ===")

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

        quantizedModel.eval()
        brevitasToTrueQuant(quantizedModel, sampleInput, debug=True)
    finally:
        # FBRANCASI: Restore original function and reload Export module again
        injection_module.injectCustomForwards = original_inject
        importlib.reload(DeepQuant.Export)
        importlib.reload(DeepQuant.Export)

    # FBRANCASI: Important note
    # Right now ONNX is not exporting the graph with GELUs folded and some nodes dont have shapes.
    #
    # If you need to use this ONNX in Deeploy (https://github.com/pulp-platform/Deeploy), please run
    # these commands on the generated network.onnx to fix these problems that can arise in Deeploy:
    #
    # > python -m onnxruntime.transformers.optimizer --input Tests/ONNX/network.onnx --output network.onnx
    #   --model_type vit --num_heads 6 --hidden_size 384 --use_multi_head_attention --disable_bias_gelu
    #   --disable_bias_skip_layer_norm --disable_skip_layer_norm --use_multi_head_attention --opt_level 0
    #
    # > python -m onnxruntime.tools.symbolic_shape_infer --input network.onnx --output network.onnx
    #
    # Also, if you have duplicated shared Floor constants in the graph (this will create problems in
    # Deeploy), you can fix this using the script FixCTT2Graph.py under the Utils folder of DeepQuant
