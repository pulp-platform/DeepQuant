# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import torch
import torch.nn as nn
from pathlib import Path

from DeepQuant.Transforms.Transformations import (
    LinearTransformation,  # Transformation for quantized linear layers (QuantLinear, QuantConv2d)
    ActivationTransformation,  # Transformation for quantized activation functions (QuantReLU, etc.)
    MHATransformation,  # Transformation for quantized multi-head attention modules
)
from DeepQuant.Transforms.Executor import (
    TransformationExecutor,
)  # Orchestrates sequential transformations
from .Utils.CustomTracer import (
    CustomBrevitasTracer,
    customBrevitasTrace,
)  # Custom FX tracer for Brevitas modules
from DeepQuant.QuantManipulation.ParameterExtractor import (
    extract_brevitas_proxy_params,  # Extracts quantization parameters from Brevitas proxies
    print_quant_params,  # Displays quantization parameters in a readable format
)
from DeepQuant.QuantManipulation.QuantNodesDivider import (
    split_quant_nodes,
)  # Splits quantization nodes into Quant/Dequant pairs
from brevitas.export.inference import (
    quant_inference_mode,
)  # Inference mode for quantized models
from brevitas.export import (
    export_onnx_qcdq,
)  # Native Brevitas ONNX export functions
from DeepQuant.QuantManipulation.DequantModifier import (
    unifyLinearDequants,
)  # Unifies dequant nodes in linear layers
from brevitas.fx import brevitas_symbolic_trace  # Brevitas-specific symbolic tracing
from DeepQuant.Utils.GraphPrinter import (
    GraphModulePrinter,
)  # Custom Graph Printer
from DeepQuant.Utils.TensorRecorder import TensorRecorder
from DeepQuant.Utils.ConsoleColor import ConsoleColor as cc


def exportQuantModel(
    model: nn.Module, exampleInput: torch.Tensor, debug: bool = False
) -> nn.Module:
    """
    Export a Brevitas model to an FX GraphModule with unrolled quantization operations.

    This function applies a series of transformations to make the quantization steps
    explicit in the model's computation graph, then traces the transformed model using
    a custom FX tracer.

    Args:
        model: The Brevitas-based model to export.
        example_input: A representative input tensor for shape tracing.
        debug: If True, prints transformation progress information.

    Returns:
        nn.Module: An FX GraphModule with explicit quantization operations.
    """

    EXPORT_FOLDER = Path().cwd()
    if Path().cwd().name == "DeepQuant":
        EXPORT_FOLDER = EXPORT_FOLDER / "Tests/ONNX"
        EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)

    printer = GraphModulePrinter()
    tensor_recorder = TensorRecorder(debug=debug)

    ###############################################################################
    # 1. Original Network
    ###############################################################################

    model = brevitas_symbolic_trace(
        model
    )  # Symbolically trace the original model using Brevitas
    if debug:
        print("\n\n=== 1. Original Network ===\n")
        printer.print_tabular(model)
        print()

    with (
        torch.no_grad(),
        quant_inference_mode(model),
    ):  # Disable gradients and use quantized inference mode
        outputModel = model(
            exampleInput
        )  # Compute original model output on example input for validation

    # export_onnx_qcdq(  # Export original model to ONNX format with QCDQ (Quant-Cast-DeQuant) nodes
    #     model,  # Model to export
    #     args=exampleInput,  # Example input for tracing
    #     export_path=EXPORT_FOLDER / "1_model_qcdq_original.onnx",
    #     opset_version=13,
    # )

    # return model

    ###############################################################################
    # 2. Injection of New Modules
    ###############################################################################

    # Create transformation sequence in appropriate order
    transformations = [
        MHATransformation(),  # Multi-head attention transformation (applied first)
        LinearTransformation(),  # Quantized linear layers transformation
        ActivationTransformation(),  # Quantized activation functions transformation
    ]

    # Initialize custom tracer for Brevitas
    tracer = CustomBrevitasTracer(debug=debug)

    # Create and execute transformation sequence using the executor
    executor = TransformationExecutor(transformations, debug=debug, tracer=tracer)
    transformedModel = executor.execute(
        model, exampleInput
    )  # Apply all transformations to the model

    # Generate FX graph using the same tracer for consistency
    fxModel = customBrevitasTrace(
        root=transformedModel,  # Transformed model to trace
        # concreteArgs=(exampleInput,),
        tracer=tracer,  # Use same tracer to maintain consistency with transformations
    )
    fxModel.recompile()  # Recompile the FX module to update its forward method
    with torch.no_grad():
        outputFxModel = fxModel(exampleInput)  # Compute transformed model output

    if isinstance(outputModel, tuple):
        outputModel = outputModel[0]

    if torch.allclose(
        outputFxModel, outputModel, atol=1e-5
    ):  # Check numerical equivalence within tolerance
        if debug:
            print(cc.wrap(" ✓ Injection of New Modules: output is consistent", cc.blue))
    else:
        raise RuntimeError(  # Raise error if outputs differ significantly
            cc.wrap(
                " ✗ Injection of New Modules changed the output significantly", cc.red
            )
        )

    if debug:
        print(cc.wrap(" ✓ All transformations completed successfully!", cc.blue))

    if debug:
        print(
            cc.wrap(
                "\n=== 2. Network after the Injection of New Modules ===\n", cc.blue
            )
        )
        printer.print_tabular(fxModel)

    # export_onnx_qcdq(  # Export transformed model to ONNX
    #     fxModel,  # Transformed model
    #     args=exampleInput,
    #     export_path=EXPORT_FOLDER / "2_model_qcdq_transformed.onnx",
    #     opset_version=13,
    # )


    ###############################################################################
    # 3. Extraction of Parameters & Split of Quant Nodes
    ###############################################################################

    # Extract quantization parameters from the network's proxies
    proxyParams = extract_brevitas_proxy_params(
        fxModel
    )  # Get scale, zero_point, bit_width for each quant node

    if debug:
        print_quant_params(
            proxyParams
        )  # Display extracted parameters in a readable format

    # Split quantization nodes into separate Quant and Dequant nodes
    splitFxModel = split_quant_nodes(
        fxModel, proxyParams, debug
    )  # Transform quant nodes into quant-dequant pairs
    splitFxModel.recompile()  # Recompile to update forward method with new nodes

    if debug:
        # Register hooks to record tensors from the split model (before dequant modification)
        tensor_recorder.register_forward_hooks(
            splitFxModel,
            node_types=[
                "wrappedInnerForwardImpl",
                "dequant",
                "unified_dequant",
                "linear",
                "conv",
                "quant",
                "act",
                "bias_quant",
                "act_quant",
                "relu",
            ],
        )

    with torch.no_grad():
        outputFxModelSplitQuant = splitFxModel(
            exampleInput
        )  # Compute output after node splitting

    if debug:
        # Save the tensors as reference for later comparison
        tensor_recorder.set_reference_tensors()

        # Register mappings from wrappedInnerForwardImpl nodes to expected unified_dequant nodes
        for node in splitFxModel.graph.nodes:
            if node.op == "call_module" and "wrappedInnerForwardImpl" in node.target:
                # For each wrappedInnerForwardImpl node, derive the expected unified_dequant name
                base_name = node.target.replace(".wrappedInnerForwardImpl", "")
                unified_dequant_name = f"{base_name}_unified_dequant"
                unified_dequant_name = unified_dequant_name.replace(".", "_")

                # Register the mapping
                tensor_recorder.record_node_mapping(node.target, unified_dequant_name)
                if debug:
                    print(f"Registered mapping: {node.target} → {unified_dequant_name}")

    if torch.allclose(
        outputModel, outputFxModelSplitQuant, atol=1e-5
    ):  # Verify numerical consistency
        if debug:
            print(cc.wrap(" ✓ Split of Quant Nodes: output is consistent", cc.blue))
    else:
        raise RuntimeError(  # Raise error if inconsistent
            cc.wrap(" ✗ Split of Quant Nodes changed the output significantly", cc.red)
        )

    if debug:
        print("\n=== 3. Network after the Split of Quant Nodes ===\n")
        printer.print_tabular(splitFxModel)
        print()

    torch.onnx.export(
        splitFxModel,
        args=exampleInput,
        f=EXPORT_FOLDER / "3_model_splitted_quant.onnx",
        opset_version=13,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
    )

    ###############################################################################
    # 4. Modification of Dequant Nodes (shift them down)
    ###############################################################################

    # Perform the unification of linear dequant nodes (move dequantization after computation)
    fxModelUnified = unifyLinearDequants(splitFxModel, debug=debug)
    fxModelUnified.recompile()  # Recompile to update forward method with new node arrangement

    if debug:
        tensor_recorder.register_forward_hooks(
            fxModelUnified,
            node_types=[
                "wrappedInnerForwardImpl",
                "dequant",
                "unified_dequant",
                "linear",
                "conv",
                "quant",
                "act",
                "bias_quant",
                "act_quant",
                "relu",
            ],
        )

    # Compute output after dequant node unification
    with torch.no_grad():
        outputFxModelDequantModified = fxModelUnified(
            exampleInput
        )  # Output after dequant modification

    if debug:
        # Use the integrated comparison that automatically handles wrappedInnerForwardImpl -> unified_dequant
        print("\n=== Tensor Comparison Before/After Dequant Unification ===")
        results = tensor_recorder.compare_tensors()
        tensor_recorder.print_comparison_results(results)

        # Clean up hooks
        tensor_recorder.remove_hooks()

    if debug:
        print("\n=== 4. Network after the Modification of Dequant Nodes ===\n")
        printer.print_tabular(fxModelUnified)
        print()

    onnxFile: str = EXPORT_FOLDER / "4_model_dequant_moved.onnx"
    torch.onnx.export(
        fxModelUnified,
        args=exampleInput,
        # f=EXPORT_FOLDER / "4_model_dequant_moved.onnx",
        f=onnxFile,
        opset_version=13,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
    )

    # Verify numerical consistency after dequant modification
    if torch.allclose(
        outputModel, outputFxModelDequantModified, atol=1e-5
    ):  # Verify numerical consistency
        if debug:
            print(
                cc.wrap(
                    " ✓ Modification of Dequant Nodes: output is consistent", cc.blue
                )
            )
    # else:
    #     raise RuntimeError(  # Raise error if inconsistent
    #         cc.wrap(
    #             " ✗ Modification of Dequant Nodes changed the output significantly",
    #             cc.red,
    #         )
    #     )

    import numpy as np
    import onnxruntime as ort
    import onnx

    onnxModel = onnx.load(onnxFile)
    inferredModel = onnx.shape_inference.infer_shapes(onnxModel)

    onnx.save(inferredModel, onnxFile)

    inputFile: str = EXPORT_FOLDER / "inputs.npz"
    np.savez(inputFile, input=exampleInput.cpu())
    print(f"Input data saved to {inputFile} ✓")

    ortSession: ort.InferenceSession = ort.InferenceSession(onnxFile)
    ortInputs: dict = {"input": exampleInput.cpu().numpy()}
    ortOutput: np.ndarray = ortSession.run(None, ortInputs)[0]

    outputFile: str = EXPORT_FOLDER / "outputs.npz"
    np.savez(outputFile, output=ortOutput)
    print(f"Output data saved to {outputFile} ✓")

    return fxModelUnified
