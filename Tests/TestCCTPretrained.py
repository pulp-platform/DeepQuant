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
import torchvision
import torchvision.transforms as transforms
from brevitas.fx.brevitas_tracer import symbolic_trace
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.quantize import preprocess_for_quantize, quantize
from brevitas.graph.utils import replace_all_uses_except
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

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


def evaluateModel(model, dataLoader, evalDevice, name="Model"):
    model.eval()
    correctTop1 = 0
    correctTop5 = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataLoader, desc=f"Evaluating {name}"):
            isTQ = "TQ" in name

            if isTQ:
                # FBRANCASI: Process different batches for the TQ model
                for i in range(inputs.size(0)):
                    singleInput = inputs[i : i + 1].to(evalDevice)
                    singleOutput = model(singleInput)

                    _, predicted = singleOutput.max(1)
                    if predicted.item() == targets[i].item():
                        correctTop1 += 1

                    _, top5Pred = singleOutput.topk(5, dim=1, largest=True, sorted=True)
                    if targets[i].item() in top5Pred[0].cpu().numpy():
                        correctTop5 += 1

                    total += 1
            else:
                inputs = inputs.to(evalDevice)
                targets = targets.to(evalDevice)
                output = model(inputs)

                _, predicted = output.max(1)
                correctTop1 += (predicted == targets).sum().item()

                _, top5Pred = output.topk(5, dim=1, largest=True, sorted=True)
                for i in range(targets.size(0)):
                    if targets[i] in top5Pred[i]:
                        correctTop5 += 1

                total += targets.size(0)

    top1Accuracy = 100.0 * correctTop1 / total
    top5Accuracy = 100.0 * correctTop5 / total

    print(
        f"{name} - Top-1 Accuracy: {top1Accuracy:.2f}% ({correctTop1}/{total}), "
        f"Top-5 Accuracy: {top5Accuracy:.2f}%"
    )

    return top1Accuracy, top5Accuracy


def calibrateModel(model, calibLoader):
    model.eval()
    with torch.no_grad(), calibration_mode(model):
        for inputs, _ in tqdm(calibLoader, desc="Calibrating model"):
            inputs = inputs.to("cpu")
            model(inputs)
    print("Calibration completed.")


def prepareFQCCT(model) -> nn.Module:
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

    # FBRANCASI: Fix 3, Find matmul operations that need dequantization (Run version)
    for node in model.graph.nodes:
        if node.op == "call_function" and node.target == torch.matmul:
            matmul_fixes.append(node)
        elif node.op == "call_method" and node.target == "__matmul__":
            matmul_fixes.append(node)
        elif hasattr(node, "target") and str(node.target) == "matmul":
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

    # FBRANCASI: Note matmul fixes found for later processing
    print(f"\nFound {len(matmul_fixes)} matmul operations for post-quantization fixing")

    model.recompile()
    model.graph.lint()

    # FBRANCASI: Print graph structure for debugging (Run version)
    print("\n=== GRAPH STRUCTURE AFTER INITIAL FIXES ===")
    for node in model.graph.nodes:
        if node.op != "placeholder" and node.op != "output":
            print(f"  {node.name}: {node.op} - {node.target}")

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
                "weight_bit_width": 8,
            },
        ),
        # FBRANCASI: Linear layers ENABLED in Run version
        nn.Linear: (
            qnn.QuantLinear,
            {
                "input_quant": Int8ActPerTensorFloat,
                "weight_quant": Int8WeightPerTensorFloat,
                "output_quant": Int8ActPerTensorFloat,
                "bias_quant": Int32Bias,
                "return_quant_tensor": True,
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

    # FBRANCASI: Post-quantization matmul dequantization (Run version specific)
    print("\n=== POST-QUANTIZATION MATMUL FIXES ===")

    # Find all matmul operations using @ operator
    matmul_nodes = []
    for node in quantizedModel.graph.nodes:
        if hasattr(node, "target") and (
            (hasattr(node.target, "__name__") and node.target.__name__ == "matmul")
            or str(node.target) == "<built-in function matmul>"
            or (node.op == "call_function" and node.target == torch.matmul)
        ):
            matmul_nodes.append(node)
            print(f"Found matmul node: {node.name}")

    print(f"\nTotal matmul nodes found: {len(matmul_nodes)}")

    # For each matmul, trace back to find linear layers and insert dequantization
    for matmul_node in matmul_nodes:
        print(f"\nProcessing matmul node: {matmul_node.name}")

        # Check both arguments of matmul
        for arg_idx, arg in enumerate(matmul_node.args):
            if hasattr(arg, "op"):
                print(f"  Checking arg {arg_idx}: {arg.name}")

                # Trace back to find if this comes from a linear layer
                def find_linear_source(node, visited=None):
                    if visited is None:
                        visited = set()
                    if node in visited:
                        return None
                    visited.add(node)

                    if node.op == "call_module" and isinstance(
                        quantizedModel.get_submodule(node.target), qnn.QuantLinear
                    ):
                        return node

                    # Check node inputs
                    for inp in node.all_input_nodes:
                        result = find_linear_source(inp, visited)
                        if result:
                            return result
                    return None

                linear_source = find_linear_source(arg)

                if linear_source:
                    print(f"    Found linear source: {linear_source.name}")

                    # Insert dequantization after the argument node
                    dequant_identity = qnn.QuantIdentity(
                        act_quant=Int8ActPerTensorFloat,
                        return_quant_tensor=False,  # Return regular tensor
                    )

                    dequant_name = f"{arg.name}_matmul_dequant"
                    quantizedModel.add_module(dequant_name, dequant_identity)

                    with quantizedModel.graph.inserting_after(arg):
                        dequant_node = quantizedModel.graph.call_module(
                            dequant_name, args=(arg,)
                        )

                    # Update matmul to use dequantized input
                    matmul_node.update_arg(arg_idx, dequant_node)
                    print(f"    Inserted dequantization: {dequant_name}")

    quantizedModel.recompile()
    quantizedModel.graph.lint()

    print("\n=== FINAL QUANTIZATION COMPLETE ===")

    return quantizedModel


@pytest.mark.ModelTests
def deepQuantTestCCT():
    torch.manual_seed(42)

    # FBRANCASI: Setup CIFAR-10 dataset
    transformsVal = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root="./Tests/Data/CIFAR", train=False, download=True, transform=transformsVal
    )

    DATASET_LIMIT = 256
    dataset = Subset(dataset, list(range(DATASET_LIMIT)))
    print(f"Validation dataset size set to {len(dataset)} images.")

    calibLoader = DataLoader(
        Subset(dataset, list(range(128))), batch_size=32, shuffle=False, pin_memory=True
    )
    valLoader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)

    # FBRANCASI: Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    print(f"Using device: {device}")

    # FBRANCASI: Load original floating point model
    originalModel = cct_2_3x2_32()
    checkpointPath = "./Tests/Data/checkpoint_epoch_200_cct2_cifar10.pth"
    checkpoint = torch.load(checkpointPath, map_location="cpu", weights_only=False)

    # FBRANCASI: Convert state dict from qkv to q_proj, k_proj, v_proj format
    original_state_dict = checkpoint["model_state_dict"]
    converted_state_dict = {}

    for key, value in original_state_dict.items():
        if "qkv.weight" in key:
            # Split QKV weight into separate Q, K, V weights
            dim = value.shape[0] // 3
            q_weight = value[:dim]
            k_weight = value[dim : 2 * dim]
            v_weight = value[2 * dim :]

            # Create new keys for separate projections
            base_key = key.replace("qkv.weight", "")
            converted_state_dict[base_key + "q_proj.weight"] = q_weight
            converted_state_dict[base_key + "k_proj.weight"] = k_weight
            converted_state_dict[base_key + "v_proj.weight"] = v_weight
        else:
            # Keep all other weights as is
            converted_state_dict[key] = value

    originalModel.load_state_dict(converted_state_dict)
    originalModel = originalModel.eval().to(device)
    print("Original CCT-2 loaded from checkpoint with converted attention weights.")

    print("Evaluating original model...")
    originalTop1, originalTop5 = evaluateModel(
        originalModel, valLoader, device, "Original CCT-2"
    )

    print("Preparing and quantizing CCT-2...")
    FQModel = prepareFQCCT(originalModel.to("cpu"))

    print("Calibrating FQ model...")
    calibrateModel(FQModel, calibLoader)

    print("Evaluating FQ model...")
    # FBRANCASI: Use CPU for brevitas models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FQTop1, FQTop5 = evaluateModel(FQModel, valLoader, device, "FQ CCT-2")

    sampleInput = torch.randn(1, 3, 32, 32).to("cpu")

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

        TQModel = brevitasToTrueQuant(FQModel, sampleInput, debug=True)
    finally:
        # FBRANCASI: Restore original function and reload Export module again
        injection_module.injectCustomForwards = original_inject
        importlib.reload(DeepQuant.Export)

    numParameters = sum(p.numel() for p in TQModel.parameters())
    print(f"Number of parameters: {numParameters:,}")

    print("Evaluating TQ model...")
    TQTop1, TQTop5 = evaluateModel(TQModel, valLoader, device, "TQ CCT-2")

    print("\nComparison Summary:")
    print(f"{'Model':<25} {'Top-1 Accuracy':<25} {'Top-5 Accuracy':<25}")
    print("-" * 75)
    print(f"{'Original CCT-2':<25} {originalTop1:<24.2f} {originalTop5:<24.2f}")
    print(f"{'FQ CCT-2':<25} {FQTop1:<24.2f} {FQTop5:<24.2f}")
    print(f"{'TQ CCT-2':<25} {TQTop1:<24.2f} {TQTop5:<24.2f}")
    print(
        f"{'FQ Drop':<25} {originalTop1 - FQTop1:<24.2f} {originalTop5 - FQTop5:<24.2f}"
    )
    print(
        f"{'TQ Drop':<25} {originalTop1 - TQTop1:<24.2f} {originalTop5 - TQTop5:<24.2f}"
    )

    if abs(FQTop1 - TQTop1) > 5.0:
        print(
            f"Warning: Large accuracy drop between FQ and TQ models. "
            f"Difference: {abs(FQTop1 - TQTop1):.2f}%"
        )

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
