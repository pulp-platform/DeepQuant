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


def evaluateModel(model, dataLoader, evalDevice, name="Model"):
    model.eval()
    correct = 0
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
                        correct += 1

                    total += 1
            else:
                inputs = inputs.to(evalDevice)
                targets = targets.to(evalDevice)
                output = model(inputs)

                _, predicted = output.max(1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

    accuracy = 100.0 * correct / total
    print(f"{name} - Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


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
    checkpoint = torch.load(checkpointPath, map_location="cpu")
    originalModel.load_state_dict(checkpoint["model_state_dict"])
    originalModel = originalModel.eval().to(device)
    print("Original CCT-2 loaded from checkpoint.")

    print("Evaluating original model...")
    originalAccuracy = evaluateModel(originalModel, valLoader, device, "Original CCT-2")

    print("Preparing and quantizing CCT-2...")
    FQModel = prepareFQCCT(originalModel.to("cpu"))

    print("Calibrating FQ model...")
    calibrateModel(FQModel, calibLoader)

    print("Evaluating FQ model...")
    # FBRANCASI: Use CPU for brevitas models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FQAccuracy = evaluateModel(FQModel, valLoader, device, "FQ CCT-2")

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
    TQAccuracy = evaluateModel(TQModel, valLoader, device, "TQ CCT-2")

    print("\nComparison Summary:")
    print(f"{'Model':<25} {'Accuracy':<25}")
    print("-" * 50)
    print(f"{'Original CCT-2':<25} {originalAccuracy:<24.2f}")
    print(f"{'FQ CCT-2':<25} {FQAccuracy:<24.2f}")
    print(f"{'TQ CCT-2':<25} {TQAccuracy:<24.2f}")
    print(f"{'FQ Drop':<25} {originalAccuracy - FQAccuracy:<24.2f}")
    print(f"{'TQ Drop':<25} {originalAccuracy - TQAccuracy:<24.2f}")

    if abs(FQAccuracy - TQAccuracy) > 5.0:
        print(
            f"Warning: Large accuracy drop between FQ and TQ models. "
            f"Difference: {abs(FQAccuracy - TQAccuracy):.2f}%"
        )
