# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import torch
import torch.nn as nn
from pathlib import Path

from .injects.transformations import (
    LinearTransformation,  # Transformation for quantized linear layers (QuantLinear, QuantConv2d)
    ActivationTransformation,  # Transformation for quantized activation functions (QuantReLU, etc.)
    MHATransformation,  # Transformation for quantized multi-head attention modules
)
from .injects.executor import (
    TransformationExecutor,
)  # Orchestrates sequential transformations
from .custom_tracer import (
    CustomBrevitasTracer,
    custom_brevitas_trace,
)  # Custom FX tracer for Brevitas modules
from .quant_manipulation.parameter_extractor import (
    extract_brevitas_proxy_params,  # Extracts quantization parameters from Brevitas proxies
    print_quant_params,  # Displays quantization parameters in a readable format
)
from .quant_manipulation.quant_nodes_divider import (
    split_quant_nodes,
)  # Splits quantization nodes into Quant/Dequant pairs
from brevitas.export.inference import (
    quant_inference_mode,
)  # Inference mode for quantized models
from brevitas.export import (
    export_onnx_qcdq,
)  # Native Brevitas ONNX export functions
from DeepQuant.quant_manipulation.dequant_modifier import (
    unify_linear_dequants,
)  # Unifies dequant nodes in linear layers
from brevitas.fx import brevitas_symbolic_trace  # Brevitas-specific symbolic tracing
from DeepQuant.utils.graph_printer import (
    GraphModulePrinter,
)  # Custom Graph Printer
from DeepQuant.utils.fx_interpreter import NodeTracer


# ANSI color codes for improved debug output readability
BLUE = "\033[94m"
RED = "\033[31m"
ENDC = "\033[0m"


def exportBrevitas(
    model: nn.Module, example_input: torch.Tensor, debug: bool = False
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

    # EXPORT_FOLDER = (
    #     Path().cwd()
    # )  # Initialize export folder to current working directory
    # print(EXPORT_FOLDER)  # Display export folder path for reference
    # if Path().cwd().name != "onnx":  # Check if already in 'onnx' directory
    #     EXPORT_FOLDER = (
    #         EXPORT_FOLDER / "onnx"
    #     )  # If not, create/use an 'onnx' subdirectory

    EXPORT_FOLDER = Path().cwd()
    print(EXPORT_FOLDER)
    if Path().cwd().name != "onnx":
        EXPORT_FOLDER = EXPORT_FOLDER / "onnx"
        EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)

    printer = GraphModulePrinter()  # Instantiation of the custom printer

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
        output_model = model(
            example_input
        )  # Compute original model output on example input for validation

    # torch.onnx.export(  
    #     model,  # Model to export
    #     args=example_input,  # Example input for tracing
    #     f=EXPORT_FOLDER / "1_model_qcdq_original.onnx",
    #     opset_version=13,
    # )
    export_onnx_qcdq(  # Export original model to ONNX format with QCDQ (Quant-Cast-DeQuant) nodes
        model,  # Model to export
        args=example_input,  # Example input for tracing
        export_path=EXPORT_FOLDER / "1_model_qcdq_original.onnx",
        opset_version=13,
    )

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
    transformed_model = executor.execute(
        model, example_input
    )  # Apply all transformations to the model

    # Generate FX graph using the same tracer for consistency
    fx_model = custom_brevitas_trace(
        transformed_model,  # Transformed model to trace
        concrete_args=(example_input,),
        tracer=tracer,  # Use same tracer to maintain consistency with transformations
    )
    fx_model.recompile()  # Recompile the FX module to update its forward method
    with torch.no_grad():
        output_fx_model = fx_model(example_input)  # Compute transformed model output

    if isinstance(output_model, tuple):
        output_model = output_model[0]

    if torch.allclose(
        output_fx_model, output_model, atol=1e-5
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
        printer.print_tabular(fx_model)

    export_onnx_qcdq(  # Export transformed model to ONNX
        fx_model,  # Transformed model
        args=example_input,
        export_path=EXPORT_FOLDER / "2_model_qcdq_transformed.onnx",
        opset_version=13,
    )

    ###############################################################################
    # 3. Extraction of Parameters & Split of Quant Nodes
    ###############################################################################

    # Extract quantization parameters from the network's proxies
    proxy_params = extract_brevitas_proxy_params(
        fx_model
    )  # Get scale, zero_point, bit_width for each quant node

    if debug:
        print_quant_params(
            proxy_params
        )  # Display extracted parameters in a readable format

    # Split quantization nodes into separate Quant and Dequant nodes
    split_fx_model = split_quant_nodes(
        fx_model, proxy_params, debug
    )  # Transform quant nodes into quant-dequant pairs
    split_fx_model.recompile()  # Recompile to update forward method with new nodes

    with torch.no_grad():
        output_fx_model_split_quant = split_fx_model(
            example_input
        )  # Compute output after node splitting

    # print("Output Original: ", output_model)
    # print("Output Split:    ", output_fx_model_split_quant)

    if torch.allclose(
        output_model, output_fx_model_split_quant, atol=1e-5
    ):  # Verify numerical consistency
        if debug:
            print(f"{BLUE} ✓ Split of Quant Nodes: output is consistent{ENDC}")
    else:
        raise RuntimeError(  # Raise error if inconsistent
            f"{RED} ✗ Split of Quant Nodes changed the output significantly{ENDC}"
        )

    if debug:
        print("\n=== 3. Network after the Split of Quant Nodes ===\n")
        printer.print_tabular(split_fx_model)
        print()

    torch.onnx.export(
        split_fx_model,
        args=example_input,
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
    fx_model_unified = unify_linear_dequants(split_fx_model, debug=debug)
    fx_model_unified.recompile()  # Recompile to update forward method with new node arrangement

    # Compute output after dequant node unification
    with torch.no_grad():
        output_fx_model_dequant_modified = fx_model_unified(
            example_input
        )  # Output after dequant modification

    print("Output Original:         ", output_model)
    print("Output Dequant Modified: ", output_fx_model_dequant_modified)

    if debug:
        print("\n=== 4. Network after the Modification of Dequant Nodes ===\n")
        printer.print_tabular(fx_model_unified)
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

    onnx_file: str = EXPORT_FOLDER / "4_model_dequant_moved.onnx"
    torch.onnx.export(
        fx_model_unified,
        args=example_input,
        # f=EXPORT_FOLDER / "4_model_dequant_moved.onnx",
        f=onnx_file,
        opset_version=13,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
    )

    # Verify numerical consistency after dequant modification
    if torch.allclose(
        output_model, output_fx_model_dequant_modified, atol=1e-5
    ):  # Verify numerical consistency
        if debug:
            print(f"{BLUE} ✓ Modification of Dequant Nodes: output is consistent{ENDC}")
    else:
        raise RuntimeError(  # Raise error if inconsistent
            f"{RED} ✗ Modification of Dequant Nodes changed the output significantly{ENDC}"
        )

    # try:
    #     tracer = NodeTracer(debug=True)
    #     tracer.trace(fx_model_unified, example_input)
    #     if debug:
    #         print(f"{BLUE} ✓ Tracing completed{ENDC}")
    # except Exception as e:
    #     print(f"{RED} ✗ Tracing failed: {str(e)}{ENDC}")
    #     print("This doesn't affect the validity of the exported model")

    import numpy as np
    import onnxruntime as ort
    import onnx

    # Step 2: Load the model and run shape inference
    # (All tensors in ONNX graph should have explicit shape information)
    onnx_model = onnx.load(onnx_file)
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)

    # Step 3: Save the model with inferred shapes
    onnx.save(inferred_model, onnx_file)

    input_file: str = EXPORT_FOLDER / "inputs.npz"
    np.savez(input_file, input=example_input.numpy())
    print("Input npz: ", example_input.numpy())
    print(f"Input data saved to {input_file} ✓")

    # onnxruntime to run the exported model
    ort_session: ort.InferenceSession = ort.InferenceSession(onnx_file)
    ort_inputs: dict = {"input": example_input.numpy()}
    ort_output: np.ndarray = ort_session.run(None, ort_inputs)[0]

    output_file: str = EXPORT_FOLDER / "outputs.npz"
    np.savez(output_file, output=ort_output)
    print("Output npz: ", ort_output)
    print(f"Output data saved to {output_file} ✓")

    return fx_model_unified  # Return the final optimized FX GraphModule
