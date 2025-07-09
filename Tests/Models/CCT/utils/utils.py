import onnx
import os
import re
import subprocess
import yaml
import torch
from onnx import helper, numpy_helper, shape_inference
import numpy as np
import onnxruntime.tools
from onnxruntime.tools import symbolic_shape_infer
import copy
from .fixshape import print_onnx_shapes
from .trainoptimization import *
import random
from onnx import TensorProto

def make_c_name(name, count=0):
    if name.lower() in ["input", "output"]:
        return name  # Keep 'input' and 'output' as is

    name = re.sub(r'input|output', '', name, flags=re.IGNORECASE)  # Remove 'input' and 'output' from other names
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if name is None or name == "":
        name = f'node_{count}'
    if name[0].isdigit() or name[0] == '_':
        name = f'node_{count}' + name
    return name

def rename_onnx_nodes(model):
    i_node = 0
    for node in model.graph.node:
        i_node += 1
        node.name = make_c_name(node.name, i_node)
        for i, input_name in enumerate(node.input):
            node.input[i] = make_c_name(input_name)
        for i, output_name in enumerate(node.output):
            node.output[i] = make_c_name(output_name)

    for input in model.graph.input:
        input.name = make_c_name(input.name)
    for output in model.graph.output:
        output.name = make_c_name(output.name)

    for init in model.graph.initializer:
        init.name = make_c_name(init.name)

    return model

def rename_and_save_onnx(input_onnx, output_onnx):
    model = onnx.load(input_onnx)
    model = rename_onnx_nodes(model)
    onnx.save(model, output_onnx)
    print(f"✅ Renamed ONNX model saved to {output_onnx}")

def run_onnx_optimization_infer(onnx_file, embedding_dim, num_heads, input_shape):
    
    batch_size, channels, height, width = input_shape  # Extract input dimensions
    try:
        print("🔹 Fixing dynamic shape...")
        subprocess.run([
            "python", "-m", "onnxruntime.tools.make_dynamic_shape_fixed",
            "--input_name", "input",
            "--input_shape", f"{batch_size},{channels},{height},{width}",
            onnx_file, onnx_file
        ], check=True)

        print("🔹 Running symbolic shape inference...")
        subprocess.run([
            "python", "-m", "onnxruntime.tools.symbolic_shape_infer",
            "--input", onnx_file, "--output", onnx_file, "--verbose", "3"
        ], check=True)

        print("🔹 Optimizing ONNX model for ViT...")
        subprocess.run([
            "python", "-m", "onnxruntime.transformers.optimizer",
            "--input", onnx_file, "--output", onnx_file,
            "--model_type", "vit",
            "--num_heads", str(num_heads),  # Controlled via config
            "--hidden_size", str(embedding_dim),  # Ensures hidden size = embedding_dim
            "--use_multi_head_attention",
            "--disable_bias_skip_layer_norm",
            "--disable_skip_layer_norm",
            "--disable_bias_gelu"
        ], check=True)

        print("✅ ONNX model optimization complete!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during ONNX optimization: {e}")

def run_onnx_optimization(onnx_file, embedding_dim, num_heads, input_shape):
    """ Run ONNX Runtime tools to optimize the model """

    batch_size, channels, height, width = input_shape  # Extract input dimensions

    try:
        print("🔹 Fixing dynamic shape...")
        subprocess.run([
            "python", "-m", "onnxruntime.tools.make_dynamic_shape_fixed",
            "--input_name", "input",
            "--input_shape", f"{batch_size},{channels},{height},{width}",
            onnx_file, onnx_file
        ], check=True)

        print("🔹 Running symbolic shape inference...")
        subprocess.run([
            "python", "-m", "onnxruntime.tools.symbolic_shape_infer",
            "--input", onnx_file, "--output", onnx_file, "--verbose", "3"
        ], check=True)

        print("🔹 Optimizing ONNX model for ViT...")
        subprocess.run([
            "python", "-m", "onnxruntime.transformers.optimizer",
            "--input", onnx_file, "--output", onnx_file,
            "--model_type", "vit",
            "--num_heads", str(num_heads),  # Controlled via config
            "--hidden_size", str(embedding_dim),  # Ensures hidden size = embedding_dim
            "--use_multi_head_attention",
            "--disable_bias_skip_layer_norm",
            "--disable_skip_layer_norm",
            "--disable_bias_gelu",
            "--disable_layer_norm",  # compatible with opset 15
        ], check=True)

        print("✅ ONNX model optimization complete!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during ONNX optimization: {e}")

def load_config(config_filename="../config.yaml"):
    """Load and parse config.yaml, returning CCT-specific parameters in a single return statement."""
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_filename)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("cct", {})

    return (
        config["pretrained"],
        config["img_size"],
        config["num_classes"],
        config["embedding_dim"],
        config["num_heads"],
        config["num_layers"],
        config["batch_size"],
        config.get("opset_version", 12)  # Default value for opset_version
    )

def load_train_config(config_filename="../config.yaml"):
    """Load and parse config.yaml, returning CCT-specific parameters in a single return statement."""
    # Resolve config.yaml relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, config_filename)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f).get("training", {})
    
    return config.get("learning_rate", 0.01)
    

def run_train_onnx_optimization(onnx_train_file, onnx_output_file):
    # remove the second output of maxpool
    print(f"🔹 Running optimization for {onnx_train_file}...")

    run_optmization_remove_biasgelu(onnx_train_file, onnx_train_file)
    print(f"✅ Successfully removed BiasGeluFusion. Saved as {onnx_train_file}")

    fix_layernorm_output(onnx_train_file, onnx_train_file)
    print(
        f"✅ Successfully fixed LayerNormalization opset version. Saved as {onnx_train_file}"
    )
    optimize_softmax_axis(onnx_train_file, onnx_train_file)
    print(
        f"✅ Successfully optimized Softmax axis. Saved as {onnx_train_file}")

    optimize_reshape_fusion(onnx_train_file, onnx_train_file)
    print(
        f"✅ Successfully optimized Reshape nodes. Saved as {onnx_output_file}")

    modify_conflict_outputs(onnx_train_file, onnx_train_file)
    print(
        f"✅ Successfully removed all second outputs from Maxpool nodes. Saved as {onnx_output_file}"
    )

    convert_squeeze_unsqueeze_input_to_attr(onnx_train_file, onnx_train_file)
    print(
        f"✅ Successfully converted Squeeze inputs to attributes. Saved as {onnx_output_file}"
    )

    add_c_to_gemm(onnx_train_file, onnx_output_file)
    print(f"✅ Successfully added C to Gemm nodes. Saved as {onnx_output_file}")

    remove_identity_reducesum(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully removed Identity and ReduceSum nodes. Saved as {onnx_output_file}"
    )

    convert_reducesum_axes_to_attr(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully converted ReduceSum axes to attributes. Saved as {onnx_output_file}"
    )

    convert_fusedmatmul_to_gemm(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully converted FusedMatMul to Gemm nodes. Saved as {onnx_output_file}"
    )

    convert_sum_to_add(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully converted Sum to Add nodes. Saved as {onnx_output_file}"
    )

    rename_softmaxgrad_op(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully renamed SoftmaxGrad nodes. Saved as {onnx_output_file}"
    )

    remove_softmax_loss_outputs(onnx_output_file, onnx_output_file)
    print(f"✅ Successfully removed Softmax Loss outputs. Saved as {onnx_output_file}")

    remove_softmax_grad_loss_inputs(onnx_output_file, onnx_output_file)
    print(
        f"✅ Successfully removed Softmax Grad Loss inputs. Saved as {onnx_output_file}"
    )
    
    print_onnx_shapes(onnx_output_file)

def rename_nodes(model_path, output_path):
    """
    Rename nodes in an ONNX model by replacing all characters that are invalid
    for C variable names with underscores.
    
    Args:
        model_path: Path to the input ONNX model
        output_path: Path to save the renamed model
    """
    # Load the model
    model = onnx.load(model_path)

    # Create a map to store original to new name mappings
    name_map = {}

    # Helper function to replace invalid C variable name characters with underscores
    def clean_name(name):
        if name is None:
            return None
        # Replace any character that's not alphanumeric or underscore with underscore
        # Ensure name starts with a letter or underscore (C variable rule)
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if cleaned and cleaned[0].isdigit():
            cleaned = '_' + cleaned
        return cleaned

    # Process graph inputs
    for input in model.graph.input:
        if input.name:
            new_name = clean_name(input.name)
            name_map[input.name] = new_name
            input.name = new_name

    # Process graph outputs
    for output in model.graph.output:
        if output.name:
            new_name = clean_name(output.name)
            name_map[output.name] = new_name
            output.name = new_name

    # Process initializers
    for initializer in model.graph.initializer:
        if initializer.name:
            new_name = clean_name(initializer.name)
            name_map[initializer.name] = new_name
            initializer.name = new_name

    # Process nodes
    for node in model.graph.node:
        # Rename node name if it exists
        if node.name:
            node.name = clean_name(node.name)

        # Rename node inputs
        for i, input_name in enumerate(node.input):
            if input_name in name_map:
                node.input[i] = name_map[input_name]
            else:
                new_name = clean_name(input_name)
                if new_name != input_name:
                    name_map[input_name] = new_name
                    node.input[i] = new_name

        # Rename node outputs
        for i, output_name in enumerate(node.output):
            if output_name in name_map:
                node.output[i] = name_map[output_name]
            else:
                new_name = clean_name(output_name)
                if new_name != output_name:
                    name_map[output_name] = new_name
                    node.output[i] = new_name

        # Rename attribute names if they contain node names
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                # Handle subgraphs if present (recursive call would be needed)
                pass
            elif attribute.type == onnx.AttributeProto.STRINGS:
                # Handle string attributes that might contain node names
                for i, s in enumerate(attribute.strings):
                    s_str = s.decode('utf-8') if isinstance(s, bytes) else s
                    if s_str in name_map:
                        attribute.strings[i] = name_map[s_str].encode('utf-8')

    # Check for any value_info that might need renaming
    for value_info in model.graph.value_info:
        if value_info.name:
            new_name = clean_name(value_info.name)
            name_map[value_info.name] = new_name
            value_info.name = new_name

    # Save the updated model
    onnx.save(model, output_path)

    return name_map

def randomize_layernorm_params(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            with torch.no_grad():
                module.weight.data = module.weight.data + torch.randn_like(module.weight.data) * 1e-6
                module.bias.data = module.bias.data + torch.randn_like(module.bias.data) * 1e-6

    return model

def randomize_onnx_initializers(model, seed=None, exclude_patterns=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if exclude_patterns is None:
        exclude_patterns = ["const", "shape", "Constant"]

    graph = model.graph

    modified_count = 0
    zero_count = 0
    skipped_count = 0

    for initializer in graph.initializer:
        # Skip initializers with specific patterns in their names
        if any(pattern in initializer.name for pattern in exclude_patterns):
            skipped_count += 1
            continue

        # Convert initializer to numpy array
        np_array = numpy_helper.to_array(initializer)

        # Check if array contains only zeros
        if np.all(np_array == 0):
            zero_count += 1

            # Determine appropriate scale based on tensor dimension and type
            if np_array.dtype == np.float32 or np_array.dtype == np.float64:
                # Use Kaiming/He initialization for weights
                if len(np_array.shape) > 1:
                    fan_in = np_array.shape[0]
                    scale = np.sqrt(2.0 / fan_in)
                    np_array = np.random.normal(0, scale, np_array.shape).astype(np_array.dtype)
                else:
                    # For bias terms or 1D tensors
                    np_array = np.random.uniform(-0.1, 0.1, np_array.shape).astype(np_array.dtype)
            elif np_array.dtype == np.int64 or np_array.dtype == np.int32:
                # For integer tensors (e.g., indices)
                max_val = min(100, 2**(np_array.itemsize*8 - 1) - 1)  # Avoid overflow
                np_array = np.random.randint(-max_val, max_val, np_array.shape).astype(np_array.dtype)

            # Create new tensor from modified numpy array
            new_tensor = numpy_helper.from_array(np_array, initializer.name)

            # Replace original initializer with new tensor
            initializer.CopyFrom(new_tensor)
            modified_count += 1


    print(f"Randomization complete:")
    print(f"- Total initializers: {len(graph.initializer)}")
    print(f"- Zero initializers found and randomized: {zero_count}")
    print(f"- Skipped initializers (based on patterns): {skipped_count}")
    print(f"- Modified initializers: {modified_count}")

    return model

def type_inference(input_model_path, output_model_path):
    """
    Perform type inference on ONNX model, setting float32 type for variables without explicit types.
    
    Args:
        input_model_path: Input ONNX model path
        output_model_path: Output ONNX model path
    """
    # Load the ONNX model
    model = onnx.load(input_model_path)
    graph = model.graph
    
    # Process input tensors
    for input_tensor in graph.input:
        if not input_tensor.type.tensor_type.elem_type:
            print(f"Setting input variable {input_tensor.name} type to FLOAT")
            input_tensor.type.tensor_type.elem_type = TensorProto.FLOAT
    
    # Process output tensors
    for output_tensor in graph.output:
        if not output_tensor.type.tensor_type.elem_type:
            print(f"Setting output variable {output_tensor.name} type to FLOAT")
            output_tensor.type.tensor_type.elem_type = TensorProto.FLOAT
    
    # Process intermediate variables
    for value_info in graph.value_info:
        if not value_info.type.tensor_type.elem_type:
            print(f"Setting intermediate variable {value_info.name} type to FLOAT")
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT
    
    # Check for any tensors mentioned in nodes but missing type info
    processed_tensors = set([tensor.name for tensor in graph.input] + 
                           [tensor.name for tensor in graph.output] + 
                           [tensor.name for tensor in graph.value_info])
    
    missing_tensors = set()
    for node in graph.node:
        for input_name in node.input:
            if input_name not in processed_tensors and input_name:
                missing_tensors.add(input_name)
        for output_name in node.output:
            if output_name not in processed_tensors and output_name:
                missing_tensors.add(output_name)
    
    # Add missing tensors to value_info with FLOAT type (keeping existing shapes)
    for tensor_name in missing_tensors:
        # Create a basic ValueInfo for the tensor (shape will be inferred by ONNX)
        tensor_value_info = onnx.helper.make_tensor_value_info(
            name=tensor_name,
            elem_type=TensorProto.FLOAT,
            shape=None  # Let ONNX infer the shape
        )
        graph.value_info.append(tensor_value_info)
        print(f"Added missing tensor {tensor_name} with FLOAT type")
    
    # Save the modified model
    onnx.save(model, output_model_path)
    print(f"Successfully saved type-inferred model to {output_model_path}")