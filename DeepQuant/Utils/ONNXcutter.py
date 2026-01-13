# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
ONNX Model Cutter Utility

This script allows to cut an ONNX model at any specified output tensor name
and optionally generate the output values at the cut point.

Basic Usage:
    python onnx_cutter.py <input_model> <cut_point> <output_model>

Examples:
    # Cut model at specific tensor
    python onnx_cutter.py model.onnx /MatMul_4_output_0 cut_model.onnx

    # List all available tensor names in the model
    python onnx_cutter.py model.onnx --list

    # Cut model and generate output.npz with random input
    python onnx_cutter.py model.onnx /Conv_output_0 cut_model.onnx --generate-output

    # Cut model and generate output.npz using specific input data
    python onnx_cutter.py model.onnx /Conv_output_0 cut_model.onnx --generate-output --input-npz inputs.npz

    # Cut model and test it after creation
    python onnx_cutter.py model.onnx /Conv_output_0 cut_model.onnx --test

Arguments:
    input_model:    Path to input ONNX model
    cut_point:      Tensor name where to cut (e.g., /MatMul_4_output_0)
    output_model:   Path to save the cut model

Options:
    --list, -l:           List all available tensor names in the model
    --test, -t:           Test the cut model after creation
    --generate-output, -g: Generate outputs.npz file containing the output at cut point
    --input-npz:          Path to input.npz file for generating output (optional)

Output Files:
    - <output_model>: The cut ONNX model
    - outputs.npz: (if --generate-output) Contains the output tensor values at cut point
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import utils


def create_deterministic_session():
    options = ort.SessionOptions()

    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    options.use_deterministic_compute = True
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False

    options.log_severity_level = 3
    options.enable_profiling = False

    return options


def list_available_outputs(model_path):
    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)

    print("\nAvailable intermediate tensor names:")
    print("=" * 50)

    all_outputs = []
    for node in model.graph.node:
        for output in node.output:
            all_outputs.append(output)

    # Sort and display
    all_outputs.sort()
    for i, output in enumerate(all_outputs, 1):
        print(f"{i:3d}. {output}")

    print(f"\nTotal intermediate tensors: {len(all_outputs)}")
    return all_outputs


def cut_onnx_model(
    input_path, cut_point, output_path, generate_output=False, input_data=None
):
    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)

    all_outputs = []
    for node in model.graph.node:
        for output in node.output:
            all_outputs.append(output)

    if cut_point not in all_outputs:
        print(f"Error: Cut point '{cut_point}' not found in model!")
        print(f"Available outputs: {len(all_outputs)}")

        similar = [out for out in all_outputs if cut_point.split("/")[-1] in out]
        if similar:
            print("Similar outputs found:")
            for sim in similar[:10]:
                print(f"  - {sim}")
        return False

    print(f"Cutting model at: {cut_point}")

    utils.extract_model(
        input_path,
        output_path,
        input_names=[inp.name for inp in model.graph.input],
        output_names=[cut_point],
    )

    print(f"Cut model saved to: {output_path}")

    try:
        cut_model = onnx.load(output_path)
        print(f"Verification: Cut model has {len(cut_model.graph.node)} nodes")
        print(f"Input: {[inp.name for inp in cut_model.graph.input]}")
        print(f"Output: {[out.name for out in cut_model.graph.output]}")

        try:
            inferred_model = onnx.shape_inference.infer_shapes(cut_model)
            onnx.save(inferred_model, output_path)
            print("Shape inference applied successfully")
        except Exception as e:
            print(f"Warning: Could not apply shape inference: {e}")

        if generate_output:
            output_dir = Path(output_path).parent
            outputFile = output_dir / "outputs.npz"

            options = create_deterministic_session()
            ortSession = ort.InferenceSession(
                output_path, sess_options=options, providers=["CPUExecutionProvider"]
            )
            input_info = ortSession.get_inputs()[0]

            if input_data is None:
                shape = input_info.shape
                shape = [
                    1 if isinstance(dim, str) or dim == "batch_size" else dim
                    for dim in shape
                ]
                input_data = np.random.randn(*shape).astype(np.float32)
                print(f"Generated random input with shape: {shape}")

            ortInputs = {input_info.name: input_data}
            ortOutput = ortSession.run(None, ortInputs)[0]

            np.savez(outputFile, output=ortOutput)
            print(f"Output data saved to {outputFile}")
            print(f"Output shape: {ortOutput.shape}")

    except Exception as e:
        print(f"Error verifying cut model: {e}")
        return False

    return True


def test_cut_model(model_path, input_shape=None):
    try:
        session = ort.InferenceSession(model_path)
        input_info = session.get_inputs()[0]

        print("\nTesting cut model:")
        print(f"Input name: {input_info.name}")
        print(f"Input shape: {input_info.shape}")
        print(f"Input type: {input_info.type}")

        if input_shape is None:
            shape = input_info.shape
            shape = [1 if isinstance(dim, str) else dim for dim in shape]
        else:
            shape = input_shape

        dummy_input = np.random.randn(*shape).astype(np.float32)

        outputs = session.run(None, {input_info.name: dummy_input})

        print("Test successful!")
        print(f"Output shape: {outputs[0].shape}")
        print(f"Output type: {outputs[0].dtype}")

    except Exception as e:
        print(f"Error testing cut model: {e}")


def main():
    parser = argparse.ArgumentParser(description="Cut ONNX model at specified tensor")
    parser.add_argument("input_model", help="Input ONNX model path")
    parser.add_argument(
        "cut_point", nargs="?", help="Tensor name to cut at (e.g., /MatMul_4_output_0)"
    )
    parser.add_argument("output_model", nargs="?", help="Output ONNX model path")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available tensor names"
    )
    parser.add_argument(
        "--test", "-t", action="store_true", help="Test the cut model after creation"
    )
    parser.add_argument(
        "--generate-output", "-g", action="store_true", help="Generate output.npz file"
    )
    parser.add_argument(
        "--input-npz", help="Path to input.npz file (for generating output)"
    )

    args = parser.parse_args()

    if not Path(args.input_model).exists():
        print(f"Error: Input model '{args.input_model}' not found!")
        sys.exit(1)

    if args.list:
        list_available_outputs(args.input_model)
        return

    if not args.cut_point or not args.output_model:
        print("Error: cut_point and output_model are required!")
        print("Use --list to see available tensor names")
        print(
            "Example: python onnx_cutter.py CCTTQ.onnx /MatMul_4_output_0 cut_model.onnx"
        )
        sys.exit(1)

    input_data = None
    if args.input_npz:
        if not Path(args.input_npz).exists():
            print(f"Error: Input npz file '{args.input_npz}' not found!")
            sys.exit(1)
        try:
            data = np.load(args.input_npz)
            input_data = (
                data["input"] if "input" in data else data[list(data.keys())[0]]
            )
            print(
                f"Loaded input data from {args.input_npz} with shape: {input_data.shape}"
            )
        except Exception as e:
            print(f"Error loading input npz file: {e}")
            sys.exit(1)

    success = cut_onnx_model(
        args.input_model,
        args.cut_point,
        args.output_model,
        generate_output=args.generate_output,
        input_data=input_data,
    )

    if success and args.test:
        test_cut_model(args.output_model)


if __name__ == "__main__":
    main()
