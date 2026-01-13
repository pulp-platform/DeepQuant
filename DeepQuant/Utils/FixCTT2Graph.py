# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Script to fix the CCTTrueQuantized ONNX model by duplicating shared constants.
This resolves the issue where a single Floor constant (onnx::Floor_772) is shared
across multiple bias quantization operations.
"""

import argparse
import os

import numpy as np
import onnx
from onnx import helper


def fix_shared_constants(model_path, output_path):
    """Fix shared constants in ONNX model by creating unique copies."""
    print(f"Loading ONNX model from: {model_path}")
    model = onnx.load(model_path)

    graph = model.graph

    shared_floor_tensor = None
    for initializer in graph.initializer:
        if initializer.name == "onnx::Floor_772":
            shared_floor_tensor = initializer
            break

    if shared_floor_tensor is None:
        print("No shared Floor constant found. Model may already be fixed.")
        return False

    print(f"Found shared Floor constant: {shared_floor_tensor.name}")
    print(f"Tensor shape: {shared_floor_tensor.dims}")

    floor_nodes = []
    for node in graph.node:
        if node.op_type == "Floor":
            for input_name in node.input:
                if input_name == shared_floor_tensor.name:
                    floor_nodes.append(node)
                    break

    print(f"Found {len(floor_nodes)} Floor nodes sharing the constant:")
    for node in floor_nodes:
        print(f"  - {node.name}")

    new_initializers = []
    for i, node in enumerate(floor_nodes):
        unique_name = f"Floor_772_unique_{i}_{node.name.replace('/', '_')}"

        new_tensor = helper.make_tensor(
            name=unique_name,
            data_type=shared_floor_tensor.data_type,
            dims=shared_floor_tensor.dims,
            vals=(
                shared_floor_tensor.float_data
                if shared_floor_tensor.float_data
                else np.frombuffer(
                    shared_floor_tensor.raw_data, dtype=np.float32
                ).tolist()
            ),
        )

        new_initializers.append(new_tensor)

        for j, input_name in enumerate(node.input):
            if input_name == shared_floor_tensor.name:
                node.input[j] = unique_name
                break

        print(f"  Created unique constant: {unique_name} for node: {node.name}")

    graph.initializer.remove(shared_floor_tensor)

    for new_tensor in new_initializers:
        graph.initializer.append(new_tensor)

    inputs_to_remove = []
    for input_tensor in graph.input:
        if input_tensor.name == shared_floor_tensor.name:
            inputs_to_remove.append(input_tensor)

    for input_tensor in inputs_to_remove:
        graph.input.remove(input_tensor)

    try:
        onnx.checker.check_model(model)
        print("Model validation passed!")
    except Exception as e:
        print(f"Model validation failed: {e}")
        return False

    print(f"Saving fixed model to: {output_path}")
    onnx.save(model, output_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix shared constants in CCTTrueQuantized ONNX model"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output fixed ONNX model",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return 1

    success = fix_shared_constants(args.input, args.output)

    if success:
        print("Successfully fixed the ONNX model!")
        print(f"Original model: {args.input}")
        print(f"Fixed model: {args.output}")

        # FBRANCASI: Replace the original model with the fixed one
        backup_path = args.input + ".backup"
        print(f"Creating backup: {backup_path}")
        os.rename(args.input, backup_path)
        os.rename(args.output, args.input)
        print("Replaced original model with fixed version")

        return 0
    else:
        print("Failed to fix the ONNX model")
        return 1


if __name__ == "__main__":
    exit(main())
