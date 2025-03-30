# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import torch
import torch.nn as nn
from pathlib import Path

from DeepQuant.Injects.Transformations import (
    LinearTransformation,  # Transformation for quantized linear layers (QuantLinear, QuantConv2d)
    ActivationTransformation,  # Transformation for quantized activation functions (QuantReLU, etc.)
    MHATransformation,  # Transformation for quantized multi-head attention modules
)
from DeepQuant.Injects.Executor import (
    TransformationExecutor,
)  # Orchestrates sequential transformations
from .CustomTracer import (
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
from DeepQuant.Utils.FxInterpreter import NodeTracer


# ANSI color codes for improved debug output readability
BLUE = "\033[94m"
RED = "\033[31m"
ENDC = "\033[0m"


def exportBrevitas(
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
        concreteArgs=(exampleInput,),
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
            print(f"{BLUE} ✓ Injection of New Modules: output is consistent{ENDC}")
    else:
        raise RuntimeError(  # Raise error if outputs differ significantly
            f"{RED} ✗ Injection of New Modules changed the output significantly{ENDC}"
        )

    if debug:
        print(f"{BLUE} ✓ All transformations completed successfully!{ENDC}")
    if debug:
        print("\n=== 2. Network after the Injection of New Modules ===\n")
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

    with torch.no_grad():
        outputFxModelSplitQuant = splitFxModel(
            exampleInput
        )  # Compute output after node splitting

    # print("Output Original: ", output_model)
    # print("Output Split:    ", output_fx_model_split_quant)

    if torch.allclose(
        outputModel, outputFxModelSplitQuant, atol=1e-5
    ):  # Verify numerical consistency
        if debug:
            print(f"{BLUE} ✓ Split of Quant Nodes: output is consistent{ENDC}")
    else:
        raise RuntimeError(  # Raise error if inconsistent
            f"{RED} ✗ Split of Quant Nodes changed the output significantly{ENDC}"
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

    # return split_fx_model

    ###############################################################################
    # 4. Modification of Dequant Nodes (shift them down)
    ###############################################################################

    # Perform the unification of linear dequant nodes (move dequantization after computation)
    fxModelUnified = unifyLinearDequants(splitFxModel, debug=debug)
    fxModelUnified.recompile()  # Recompile to update forward method with new node arrangement

    # Compute output after dequant node unification
    with torch.no_grad():
        outputFxModelDequantModified = fxModelUnified(
            exampleInput
        )  # Output after dequant modification

    print("Output Original:         ", outputModel)
    print("Output Dequant Modified: ", outputFxModelDequantModified)

    if debug:
        print("\n=== 4. Network after the Modification of Dequant Nodes ===\n")
        printer.print_tabular(fxModelUnified)
        print()

    # # Verify numerical consistency after dequant modification
    # if torch.allclose(
    #     output_model, output_fx_model_dequant_modified, atol=1e-5
    # ):  # Verify numerical consistency
    #     if debug:
    #         print(f"{BLUE} ✓ Modification of Dequant Nodes: output is consistent{ENDC}")
    # else:
    #     raise RuntimeError(  # Raise error if inconsistent
    #         f"{RED} ✗ Modification of Dequant Nodes changed the output significantly{ENDC}"
    #     )

    # if debug:
    #     print("\n=== 4. Network after the Modification of Dequant Nodes ===\n")
    #     printer.print_tabular(fx_model_unified)
    #     print()

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
            print(f"{BLUE} ✓ Modification of Dequant Nodes: output is consistent{ENDC}")
    else:
        raise RuntimeError(  # Raise error if inconsistent
            f"{RED} ✗ Modification of Dequant Nodes changed the output significantly{ENDC}"
        )

    import numpy as np
    import onnxruntime as ort
    import onnx

    # Step 2: Load the model and run shape inference
    # (All tensors in ONNX graph should have explicit shape information)
    onnxModel = onnx.load(onnxFile)
    inferredModel = onnx.shape_inference.infer_shapes(onnxModel)

    # Step 3: Save the model with inferred shapes
    onnx.save(inferredModel, onnxFile)

    inputFile: str = EXPORT_FOLDER / "inputs.npz"
    np.savez(inputFile, input=exampleInput.cpu())
    print("Input npz: ", exampleInput)
    print(f"Input data saved to {inputFile} ✓")

    # onnxruntime to run the exported model
    ortSession: ort.InferenceSession = ort.InferenceSession(onnxFile)
    ortInputs: dict = {"input": exampleInput.cpu().numpy()}
    ortOutput: np.ndarray = ortSession.run(None, ortInputs)[0]

    outputFile: str = EXPORT_FOLDER / "outputs.npz"
    np.savez(outputFile, output=ortOutput)
    print("Output npz: ", ortOutput)
    print(f"Output data saved to {outputFile} ✓")

    return fxModelUnified  # Return the final optimized FX GraphModule
