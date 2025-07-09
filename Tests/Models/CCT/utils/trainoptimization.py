import onnx
import os
import re
import subprocess
import yaml
from onnx import helper, numpy_helper, shape_inference
import numpy as np
import copy

def add_c_to_gemm(input_model_path, output_model_path):
    model = onnx.load(input_model_path)
    graph = model.graph
    
    for node in graph.node:
        if node.op_type == 'Gemm':

            if len(node.input) == 2:
                print(f"Find Gemm without C: {node.name}")
                
                input_a_name = node.input[0]
                input_b_name = node.input[1]
                
    
                b_shape = None
                for init in graph.initializer:
                    if init.name == input_b_name:
                        b_tensor = numpy_helper.to_array(init)
                        b_shape = b_tensor.shape
                        break
                

                if b_shape is None:
                    for vi in graph.value_info:
                        if vi.name == input_b_name:
                            b_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                            break
      
                if b_shape is None:
                    output_name = node.output[0]
                    for vi in graph.value_info + [graph.output[i] for i in range(len(graph.output))]:
                        if vi.name == output_name:
                            output_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                          
                            transB = 0
                            for attr in node.attribute:
                                if attr.name == 'transB' and attr.i == 1:
                                    transB = 1
                            
                         
                            c_length = output_shape[-1]
                            b_shape = [c_length, 0] if transB == 0 else [0, c_length]
                            break
                
                if b_shape is not None:
        
                    transB = 0
                    for attr in node.attribute:
                        if attr.name == 'transB' and attr.i == 1:
                            transB = 1
                    
                    c_shape = [b_shape[1]] if not transB else [b_shape[0]]
       
                    c_tensor = np.zeros(c_shape, dtype=np.float32)
                    c_name = f"{node.name}_c_bias"
                    
                    c_initializer = numpy_helper.from_array(c_tensor, name=c_name)
                    graph.initializer.append(c_initializer)
                    
                  
                    node.input.append(c_name)
                    print(f"Add C: {c_name}, Shape: {c_shape}")
                else:
                    print(f"Warning: Cannot find {node.name} shape, pass this node.")
    

    onnx.save(model, output_model_path)
    print(f"Saved to: {output_model_path}")

def replace_biasgelu_with_gelu_add(input_model_path, output_model_path):
 
    model = onnx.load(input_model_path)
    
    # Collect all value_info entries by name for easy lookup
    value_info_map = {}
    for vi in model.graph.value_info:
        value_info_map[vi.name] = vi
    
    # Add input and output value_info to the map
    for inp in model.graph.input:
        value_info_map[inp.name] = inp
    
    for out in model.graph.output:
        value_info_map[out.name] = out
    
    # Create new node list and value_info list
    new_nodes = []
    new_value_info = []
    biasgelu_count = 0
    
    # Counter for generating unique names
    unique_id = 0
    def get_unique_name(prefix):
        nonlocal unique_id
        name = f"{prefix}_{unique_id}"
        unique_id += 1
        return name
    
    # Process all nodes
    for node in model.graph.node:
        if node.op_type == 'BiasGelu':
            biasgelu_count += 1
            
            # Get BiasGelu inputs and outputs
            input_name = node.input[0]  # X
            bias_name = node.input[1]   # Bias
            output_name = node.output[0]  # Y
            
            # Generate unique name prefix
            prefix = node.name if node.name else f"gelu_add"
            
            # Step 1: First apply Add operation to add bias
            add_output = get_unique_name(f"{prefix}_add_out")
            add_node = helper.make_node(
                'Add',
                inputs=[input_name, bias_name],
                outputs=[add_output],
                name=f"{prefix}_add"
            )
            new_nodes.append(add_node)
            
            # Create value_info for add_output with proper type and shape
            # Use the same type and shape as the input tensor if available
            if input_name in value_info_map:
                input_value_info = value_info_map[input_name]
                add_output_value_info = helper.make_tensor_value_info(
                    add_output,
                    input_value_info.type.tensor_type.elem_type,
                    [d.dim_value if d.dim_value else d.dim_param for d in input_value_info.type.tensor_type.shape.dim]
                )
                new_value_info.append(add_output_value_info)
                value_info_map[add_output] = add_output_value_info
            
            # Step 2: Then apply Gelu activation function
            gelu_node = helper.make_node(
                'Gelu',
                inputs=[add_output],
                outputs=[output_name],
                name=f"{prefix}_gelu"
            )
            new_nodes.append(gelu_node)
            
            # If we have output value_info, make sure it's preserved
            # Otherwise, create it with the same shape and type as the input to Gelu
            if output_name not in value_info_map and add_output in value_info_map:
                output_value_info = helper.make_tensor_value_info(
                    output_name,
                    value_info_map[add_output].type.tensor_type.elem_type,
                    [d.dim_value if d.dim_value else d.dim_param for d in value_info_map[add_output].type.tensor_type.shape.dim]
                )
                new_value_info.append(output_value_info)
                value_info_map[output_name] = output_value_info
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)
    
    print(f"Replaced {biasgelu_count} BiasGelu nodes with Gelu+Add combinations")
    
    # Create new graph with all collected value_info
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        value_info=list(model.graph.value_info) + new_value_info
    )
    
    # Build new model, preserving original model metadata
    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string
    )
    
    # Copy opset imports
    del new_model.opset_import[:]
    new_model.opset_import.extend(model.opset_import)
    
    # Add Microsoft domain if not present (for Gelu)
    has_ms_domain = any(opset.domain == "com.microsoft" for opset in new_model.opset_import)
    if not has_ms_domain:
        ms_opset = helper.make_opsetid("com.microsoft", 1)
        new_model.opset_import.append(ms_opset)
    
    # Copy IR version
    new_model.ir_version = model.ir_version
    
    # Run shape inference to ensure all shapes are properly defined
    try:
        new_model = shape_inference.infer_shapes(new_model)
        print("Shape inference successful")
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}")
    
    # Skip validation and directly save if needed
    try:
        onnx.checker.check_model(new_model)
        print("Model validation successful!")
    except Exception as e:
        print(f"Warning: Model validation failed, but still saving: {e}")
    
    # Save the modified model
    onnx.save(new_model, output_model_path)
    print(f"Saved modified model to {output_model_path}")
    
    return new_model

def fix_layernorm_output(input_model_path: str, output_model_path: str) -> bool:
    """
    Fix output types and shapes for all LayerNorm operators in an ONNX model.
    
    Args:
        input_model_path (str): Path to the input model file
        output_model_path (str): Path to save the output model file
        
    Returns:
        bool: True if the operation succeeded, False otherwise
    """
    try:
        # Load the model
        model = onnx.load(input_model_path)
        graph = model.graph
        
        # Find all LayerNorm nodes
        layernorm_count = 0
        updated_count = 0
        tensor_info = {}
        
        # Collect tensor information
        # Process input tensors
        for input_tensor in graph.input:
            name = input_tensor.name
            shape = [dim.dim_value if dim.dim_value > 0 else None for dim in input_tensor.type.tensor_type.shape.dim]
            elem_type = input_tensor.type.tensor_type.elem_type
            tensor_info[name] = {"shape": shape, "elem_type": elem_type}
        
        # Process intermediate and output tensors
        for value_info in list(graph.value_info) + list(graph.output):
            name = value_info.name
            shape = [dim.dim_value if dim.dim_value > 0 else None for dim in value_info.type.tensor_type.shape.dim]
            elem_type = value_info.type.tensor_type.elem_type
            tensor_info[name] = {"shape": shape, "elem_type": elem_type}
        
        # Fix each LayerNorm node
        for node in graph.node:
            if node.op_type == 'LayerNormalization':
                layernorm_count += 1
                
                if not node.input:
                    continue
                
                # Get input information
                input_name = node.input[0]
                if input_name not in tensor_info:
                    continue
                
                input_info = tensor_info[input_name]
                input_shape = input_info["shape"]
                input_elem_type = input_info["elem_type"]
                
                # Get axis attribute
                axis = -1
                for attr in node.attribute:
                    if attr.name == "axis":
                        axis = attr.i
                        break
                
                # Process all outputs
                for i, output_name in enumerate(node.output):
                    # Determine correct output shape and type
                    output_shape = None
                    output_elem_type = input_elem_type
                    
                    if i == 0:  # Main output - same shape as input
                        output_shape = input_shape
                    else:  # mean and std outputs - shape depends on normalization axis
                        # Handle negative axis index
                        if axis < 0 and input_shape and None not in input_shape:
                            axis = len(input_shape) + axis
                        
                        # Create shape for mean and std (remove normalization axis)
                        if input_shape and None not in input_shape and 0 <= axis < len(input_shape):
                            output_shape = list(input_shape)
                            output_shape.pop(axis)  # Remove normalization axis
                    
                    # Find and remove existing value info
                    for value_info in list(graph.value_info):
                        if value_info.name == output_name:
                            graph.value_info.remove(value_info)
                            break
                    
                    # Create new value info
                    if output_shape and None not in output_shape:
                        new_value_info = onnx.helper.make_tensor_value_info(
                            output_name, 
                            output_elem_type, 
                            output_shape
                        )
                        graph.value_info.append(new_value_info)
                    
                    # Update graph output if needed
                    for j, output in enumerate(list(graph.output)):
                        if output.name == output_name:
                            if output_shape and None not in output_shape:
                                new_output = onnx.helper.make_tensor_value_info(
                                    output_name,
                                    output_elem_type,
                                    output_shape
                                )
                                graph.output.remove(output)
                                graph.output.insert(j, new_output)
                            break
                    
                    # Update tensor info dictionary
                    tensor_info[output_name] = {
                        "shape": output_shape,
                        "elem_type": output_elem_type
                    }
                    
                    print(f"  Output {i}: {output_name}, shape={output_shape}")  # Debug info
                
                updated_count += 1
        
        # Save the model
        onnx.save(model, output_model_path)
        print(f"Updated {updated_count}/{layernorm_count} LayerNorm nodes, model saved to {output_model_path}")
        return True
        
    except Exception as e:
        print(f"Error fixing LayerNorm outputs: {str(e)}")
        return False


def modify_conflict_outputs(input_model_path, output_model_path):
    model = onnx.load(input_model_path)
    graph = model.graph
    
    select_nodes = []
    for node in graph.node:
        if node.op_type == 'LayerNormalization' or node.op_type == 'MaxPool':
        # if node.op_type == 'MaxPool':
            select_nodes.append(node)
    
    print(f"Find {len(select_nodes)} Maxpool")
    
    outputs_to_remove = []
    
    new_nodes = []
    
    for node in graph.node:
        if (node.op_type == 'LayerNormalization' or node.op_type == 'MaxPool') and len(node.output) > 1:
        # if (node.op_type == 'MaxPool') and len(node.output) > 1:
            outputs_to_remove.extend(node.output[1:])
            
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            first_output = node.output[0]
            
            del new_node.output[:]
            new_node.output.append(first_output)
            
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)
    
    del graph.node[:]
    graph.node.extend(new_nodes)
    
    new_outputs = []
    for output in graph.output:
        if output.name not in outputs_to_remove:
            new_outputs.append(output)
    
    del graph.output[:]
    graph.output.extend(new_outputs)
    
    onnx.save(model, output_model_path)
    print(f"Saved to: {output_model_path}")
    
def convert_squeeze_unsqueeze_input_to_attr(input_model_path, output_model_path):
    """
    Convert Squeeze and Unsqueeze nodes with axes as input to axes as attribute.
    This is useful for compatibility with older ONNX versions where axes was only supported as an attribute.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the converted ONNX model
    """
    model = onnx.load(input_model_path)
    
    modified_nodes = []
    modified_count = 0
    
    initializers = {init.name: init for init in model.graph.initializer}
    
    for node in model.graph.node:
        # Check if the node is Squeeze or Unsqueeze with more than one input
        if (node.op_type in ['Squeeze', 'Unsqueeze']) and len(node.input) > 1:
            modified_count += 1
            
            data_input = node.input[0]
            
            axes_input_name = node.input[1]
            
            if axes_input_name in initializers:
                # Get the axes values from the initializer
                axes_initializer = initializers[axes_input_name]
                axes_np = numpy_helper.to_array(axes_initializer)
                axes_list = axes_np.tolist()
                
                # Make the axes a scalar if it's a single value
                if isinstance(axes_list, list) and len(axes_list) == 1:
                    axes_list = axes_list[0]
                
                # Create a new node with axes as attribute instead of input
                new_node = helper.make_node(
                    op_type=node.op_type,
                    inputs=[data_input],  
                    outputs=list(node.output),
                    name=node.name,
                    axes=axes_list 
                )
                
                # Copy other attributes if they exist
                for attr in node.attribute:
                    if attr.name != 'axes':
                        new_node.attribute.append(attr)
                
                modified_nodes.append(new_node)
            else:
                # If we can't find the axes initializer, keep the original node
                print(f"Warning: Cannot find '{node.name}' axes initializer. Keep the original node.")
                modified_nodes.append(node)
        else:
            # Keep all other nodes as they are
            modified_nodes.append(node)
    
    print(f"Modified {modified_count} Squeeze/Unsqueeze nodes")
    
    # Identify initializers that are no longer referenced
    # This happens when we convert the axes from input to attribute
    used_inputs = set()
    for node in modified_nodes:
        for input_name in node.input:
            used_inputs.add(input_name)
    
    unused_initializers = set()
    for init in model.graph.initializer:
        if init.name not in used_inputs:
            unused_initializers.add(init.name)
    
    # Create a new graph with the modified nodes and without unused initializers
    new_graph = helper.make_graph(
        nodes=modified_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=[init for init in model.graph.initializer if init.name not in unused_initializers]
    )
    
    # Copy over value_info from the original model
    for vi in model.graph.value_info:
        new_graph.value_info.append(vi)
    
    # Create a new model with the updated graph
    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string
    )
    
    # Copy over IR version and opset imports
    new_model.ir_version = model.ir_version
    new_model.opset_import.extend(model.opset_import)
    
    # Save the model
    onnx.save(new_model, output_model_path)
    print(f"Saved to {output_model_path}")
    
    return new_model

def run_optmization_remove_biasgelu(onnx_train_file, onnx_out_file):
    """
    Replace BiasGelu operations with Add+Gelu while maintaining shape consistency.
    
    Args:
        onnx_train_file: Path to input ONNX model file
        onnx_out_file: Path to output ONNX model file
    """
    # Load the model
    model = onnx.load(onnx_train_file)
    graph = model.graph
    
    # Create new nodes list to replace the old ones
    new_nodes = []
    replaced_count = 0
    
    # Process all nodes
    for node in graph.node:
        if node.op_type == "BiasGelu":
            print(f"🔄 Replacing BiasGeluFusion: {node.name}")
            replaced_count += 1
            
            # Get input and output tensors
            X, Bias = node.input
            output = node.output[0]
            
            # Create intermediate tensor name
            intermediate_output = f"{X}_add_bias"
            
            # Create Add node
            add_node = helper.make_node(
                "Add",
                inputs=[X, Bias],
                outputs=[intermediate_output],
                name=f"{node.name}_Add",
            )
            
            # Create Gelu node
            gelu_node = helper.make_node(
                "Gelu",
                inputs=[intermediate_output],
                outputs=node.output,
                name=f"{node.name}_Gelu",
            )
            
            # Add shape information for the intermediate tensor
            # Try to find X's shape info
            X_shape = None
            X_type = 1  # Default to FLOAT
            
            # Look for X in inputs, outputs, value_info, or initializers
            for info in graph.input:
                if info.name == X:
                    X_shape = [d.dim_value if d.HasField("dim_value") else -1 for d in info.type.tensor_type.shape.dim]
                    X_type = info.type.tensor_type.elem_type
                    break
                    
            if X_shape is None:
                for info in graph.value_info:
                    if info.name == X:
                        X_shape = [d.dim_value if d.HasField("dim_value") else -1 for d in info.type.tensor_type.shape.dim]
                        X_type = info.type.tensor_type.elem_type
                        break
            
            # Add value_info for intermediate tensor
            if X_shape:
                value_info = helper.make_tensor_value_info(
                    intermediate_output,
                    X_type,
                    X_shape
                )
                graph.value_info.append(value_info)
            
            # Add the new nodes to our list
            new_nodes.extend([add_node, gelu_node])
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)
    
    # Replace nodes in the graph
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    
    # Create a copy of the model for selective shape inference
    safe_model = copy.deepcopy(model)
    
    # Remove Microsoft custom operators that might cause shape inference to fail
    ms_nodes = []
    for node in safe_model.graph.node:
        if node.domain == "com.microsoft":
            ms_nodes.append(node)
    
    if ms_nodes:
        print(f"⚠️ Found {len(ms_nodes)} Microsoft custom operators that might affect shape inference")
        
        # Try to run shape inference on the modified model without MS operators
        try:
            # Create a temporary graph without Microsoft operators
            temp_model = copy.deepcopy(safe_model)
            temp_graph = temp_model.graph
            
            # Remove Microsoft custom operators
            temp_nodes = [node for node in temp_graph.node if node.domain != "com.microsoft"]
            temp_graph.ClearField("node")
            temp_graph.node.extend(temp_nodes)
            
            # Run shape inference on this simplified model
            inferred_model = shape_inference.infer_shapes(temp_model)
            
            # Collect inferred shapes for our new nodes
            inferred_value_infos = {}
            for value_info in inferred_model.graph.value_info:
                inferred_value_infos[value_info.name] = value_info
            
            # Update the original model with any newly inferred shapes
            for name, value_info in inferred_value_infos.items():
                # Skip if already exists
                if any(info.name == name for info in model.graph.value_info):
                    continue
                
                model.graph.value_info.append(value_info)
                
            print("✅ Partial shape inference completed for non-Microsoft operators")
        except Exception as e:
            print(f"⚠️ Partial shape inference failed: {e}")
    else:
        # No Microsoft operators, try regular shape inference
        try:
            model = shape_inference.infer_shapes(model)
            print("✅ Shape inference completed successfully")
        except Exception as e:
            print(f"⚠️ Shape inference failed: {e}")
    
    # Save the modified model
    onnx.save(model, onnx_out_file)
    
    if replaced_count > 0:
        print(f"✅ Successfully replaced {replaced_count} BiasGelu nodes with Add + GELU.")
    else:
        print("⚠️ No BiasGelu nodes were replaced.")
    
    return model

def optimize_reshape_fusion(input_model_path: str, output_model_path: str) -> None:
    """
    Optimize ONNX model by fusing consecutive Reshape operations.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path where the optimized ONNX model will be saved
    """
    print(f"Loading model: {input_model_path}")
    model = onnx.load(input_model_path)
    
    # Create mapping from node name to node
    node_map = {}
    for node in model.graph.node:
        node_map[node.name] = node
    
    # Create mapping from input name to producing node
    input_to_node = {}
    for node in model.graph.node:
        for output in node.output:
            input_to_node[output] = node
    
    # Create mapping from output name to consuming nodes
    output_to_nodes = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name not in output_to_nodes:
                output_to_nodes[input_name] = []
            output_to_nodes[input_name].append(node)
    
    # Find all Reshape nodes
    reshape_nodes = [node for node in model.graph.node if node.op_type == "Reshape"]
    
    # Track nodes to be removed by index rather than node objects
    # This avoids the "unhashable type: 'NodeProto'" error
    nodes_to_remove_indices = []
    
    # Track value info to keep
    value_info_to_keep = set(vi.name for vi in model.graph.value_info)
    
    # For each Reshape node, check if its input is also from a Reshape node
    for reshape_node in reshape_nodes:
        # Get the input of the current Reshape node
        input_name = reshape_node.input[0]
        
        # Check if the input comes from another Reshape operation
        if input_name in input_to_node and input_to_node[input_name].op_type == "Reshape":
            previous_reshape = input_to_node[input_name]
            
            # Check if the previous Reshape is only used by the current Reshape
            if input_name in output_to_nodes and len(output_to_nodes[input_name]) == 1:
                print(f"Found fusible Reshape pair: {previous_reshape.name} -> {reshape_node.name}")
                
                # Get the shape tensors for both Reshape nodes
                prev_shape_tensor_name = previous_reshape.input[1]
                current_shape_tensor_name = reshape_node.input[1]
                
                # Modify the current Reshape node to connect directly to the input of the previous Reshape
                reshape_node.input[0] = previous_reshape.input[0]
                
                # Mark the previous Reshape node for removal by its index
                for i, node in enumerate(model.graph.node):
                    if (node.name == previous_reshape.name and 
                        node.op_type == previous_reshape.op_type and
                        node.input == previous_reshape.input and 
                        node.output == previous_reshape.output):
                        nodes_to_remove_indices.append(i)
                        break
                
                # Intermediate value info doesn't need to be kept
                if input_name in value_info_to_keep:
                    value_info_to_keep.remove(input_name)
    
    # Handle custom nodes from Microsoft
    # Since Microsoft nodes might have a different structure or behavior
    # We need to be careful when dealing with them
    custom_nodes = [node for node in model.graph.node if node.domain.startswith('com.microsoft')]
    print(f"Found {len(custom_nodes)} Microsoft custom nodes. These will be preserved.")
    
    # Create a new graph excluding the nodes to be removed
    new_nodes = []
    for i, node in enumerate(model.graph.node):
        if i not in nodes_to_remove_indices:
            new_nodes.append(node)
    
    # Create a new value info list, keeping only the needed value info
    new_value_info = [vi for vi in model.graph.value_info if vi.name in value_info_to_keep]
    
    # Create a new graph
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        value_info=new_value_info
    )
    
    # Create a new model
    new_model = helper.make_model(
        new_graph, 
        producer_name="ONNX Reshape Fusion Optimizer",
        ir_version=model.ir_version,
        opset_imports=model.opset_import
    )
    
    # Preserve custom opsets from the original model
    new_model.opset_import.extend([opset for opset in model.opset_import if opset.domain.startswith('com.microsoft')])
    
    # Save the optimized model
    onnx.save(new_model, output_model_path)
    
    # Print statistics
    print(f"Original model node count: {len(model.graph.node)}")
    print(f"Optimized model node count: {len(new_model.graph.node)}")
    print(f"Removed Reshape nodes: {len(nodes_to_remove_indices)}")
    print(f"Optimized model saved to: {output_model_path}")


def remove_identity_reducesum(input_model_path, output_model_path):
    """
    Remove Identity and removable ReduceSum nodes from the model,
    ensuring correct output naming
    
    Args:
        input_model_path (str): Input ONNX model path
        output_model_path (str): Output ONNX model path
    
    Returns:
        onnx.ModelProto: Processed model
    """
    import onnx
    import numpy as np
    from onnx import shape_inference, helper, TensorProto
    
    # Load the model and infer shapes
    model = onnx.load(input_model_path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}. Continuing without shape information.")
    
    graph = model.graph
    
    # Build node mapping
    node_map = {node.name: node for node in graph.node}
    
    # Store tensor shapes from value_info, inputs, and outputs
    shape_info = {}
    for info in list(graph.value_info) + list(graph.input) + list(graph.output):
        if hasattr(info.type.tensor_type.shape, 'dim'):
            dims = []
            for dim in info.type.tensor_type.shape.dim:
                if dim.dim_value:
                    dims.append(dim.dim_value)
                else:
                    dims.append(-1)
            shape_info[info.name] = dims
    
    # Get initializer shapes
    for initializer in graph.initializer:
        shape_info[initializer.name] = list(initializer.dims)
    
    # Store nodes to remove and replacement mapping
    nodes_to_remove = []
    replacement_map = {}
    reshape_nodes_to_add = []
    
    # Process Identity nodes
    for node in graph.node:
        if node.op_type == "Identity":
            input_name = node.input[0]
            output_name = node.output[0]
            
            replacement_map[output_name] = input_name
            nodes_to_remove.append(node)
    
    # Process ReduceSum nodes
    for node in graph.node:
        if node.op_type == "ReduceSum":
            input_name = node.input[0]
            output_name = node.output[0]
            
            # Check for dimension 1 reduction with keepdims=0
            keepdims = 1  # Default value
            for attr in node.attribute:
                if attr.name == "keepdims":
                    keepdims = attr.i
                    break
            
            # Get reduction axes
            axes = []
            for attr in node.attribute:
                if attr.name == "axes":
                    axes = list(attr.ints)
                    break
            
            # If opset >= 13, axes might be an input
            if len(node.input) > 1 and not axes:
                axes_name = node.input[1]
                for initializer in graph.initializer:
                    if initializer.name == axes_name:
                        axes = onnx.numpy_helper.to_array(initializer).tolist()
                        if not isinstance(axes, list):
                            axes = [axes]
                        break
            
            # Get input shape
            if input_name in shape_info:
                input_shape = shape_info[input_name]
                
                # Check if all reduction axes have dimension 1
                all_dim_one = True
                for axis in axes:
                    # Handle negative axis
                    if axis < 0:
                        axis = len(input_shape) + axis
                    
                    if 0 <= axis < len(input_shape) and input_shape[axis] == 1:
                        continue
                    else:
                        all_dim_one = False
                        break
                
                if all_dim_one and axes:
                    if keepdims == 1:
                        # Simple replacement case
                        replacement_map[output_name] = input_name
                        nodes_to_remove.append(node)
                    elif keepdims == 0:
                        # Need to add a Reshape node
                        # Calculate output shape by removing dimensions with size 1
                        output_shape = []
                        for i, dim in enumerate(input_shape):
                            if i not in axes and (i + len(input_shape) not in axes):
                                output_shape.append(dim)
                        
                        # Create shape tensor for Reshape
                        shape_tensor_name = f"{node.name}_shape"
                        shape_tensor = helper.make_tensor(
                            name=shape_tensor_name,
                            data_type=TensorProto.INT64,
                            dims=[len(output_shape)],
                            vals=output_shape
                        )
                        
                        # Create Reshape node
                        reshape_node = helper.make_node(
                            "Reshape",
                            inputs=[input_name, shape_tensor_name],
                            outputs=[output_name],
                            name=f"{node.name}_reshape"
                        )
                        
                        # Store for later addition
                        reshape_nodes_to_add.append((reshape_node, shape_tensor))
                        nodes_to_remove.append(node)
    
    # Update inputs of other nodes
    for node in graph.node:
        if node not in nodes_to_remove:
            for i, input_name in enumerate(node.input):
                if input_name in replacement_map:
                    node.input[i] = replacement_map[input_name]
    
    # Update graph outputs
    for output in graph.output:
        if output.name in replacement_map:
            output.name = replacement_map[output.name]
    
    # Remove nodes and add new reshape nodes
    new_nodes = [node for node in graph.node if node not in nodes_to_remove]
    
    # Add shape tensors to initializers
    for _, shape_tensor in reshape_nodes_to_add:
        graph.initializer.append(shape_tensor)
    
    # Add reshape nodes
    for reshape_node, _ in reshape_nodes_to_add:
        new_nodes.append(reshape_node)
    
    # Clear and re-add nodes
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    
    # Save model
    onnx.save(model, output_model_path)
    
    print(f"Saved to {output_model_path}")
    print(f"Removed {len(nodes_to_remove)} nodes")
    print(f"Added {len(reshape_nodes_to_add)} Reshape nodes")
    
    return model


def convert_reducesum_axes_to_attr(input_file: str, output_file: str):
    model = onnx.load(input_file)
    graph = model.graph
    
    new_nodes = []
    
    initializers = {init.name: init for init in graph.initializer}
    
    for node in graph.node:
        if node.op_type == "ReduceSum":
            if len(node.input) >= 2:
                data_input = node.input[0]
                axes_input = node.input[1]
                
                if axes_input in initializers:
                    axes_tensor = initializers[axes_input]
                    axes_np = numpy_helper.to_array(axes_tensor)
                    axes_list = axes_np.tolist()
                    
                    new_node = helper.make_node(
                        op_type="ReduceSum",
                        inputs=[data_input],
                        outputs=node.output,
                        name=node.name,
                        axes=axes_list
                    )
                    
                    for attr in node.attribute:
                        if attr.name != "axes":
                            new_node.attribute.append(attr)
                    
                    new_nodes.append(new_node)
                else:
                    new_nodes.append(node)
            else:
                new_nodes.append(node)
        else:
            new_nodes.append(node)
    
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=graph.input,
        outputs=graph.output,
        initializer=graph.initializer,
        value_info=graph.value_info
    )
    
    new_model = helper.make_model(
        new_graph,
        producer_name="ReduceSumAxesConverter",
        ir_version=model.ir_version,
        opset_imports=model.opset_import
    )
    
    new_model.metadata_props.extend(model.metadata_props)
    
    for domain in model.domain:
        new_model.domain.append(domain)
    
    onnx.save(new_model, output_file)
    print(f"Model converted and saved to: {output_file}")


def convert_fusedmatmul_to_gemm(input_model_path, output_model_path):
    """
    Convert Microsoft's FusedMatMul nodes to standard Gemm nodes in an ONNX model.
    This function handles custom ops and adds a zero tensor for the C input of Gemm when needed.
    The three inputs to Gemm will be named A, B, and C in the function implementation.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the converted ONNX model
    """
    # Load the model
    model = onnx.load(input_model_path)
    
    # Track necessary changes
    new_nodes = []
    new_initializers = []
    
    # Process each node in the graph
    for node in model.graph.node:
        # Check if the node is a FusedMatMul from Microsoft domain
        if node.op_type == "FusedMatMul" and node.domain == "com.microsoft":
            # Extract attributes from FusedMatMul
            alpha = 1.0
            transA = 0
            transB = 0
            
            for attr in node.attribute:
                if attr.name == "alpha":
                    alpha = attr.f
                elif attr.name == "transA":
                    transA = attr.i
                elif attr.name == "transB":
                    transB = attr.i
            
            # Get inputs and output of FusedMatMul
            # In our implementation, we'll call these A and B
            A = node.input[0]
            B = node.input[1]
            output = node.output[0]
            
            # Create a name for the zero tensor (C input for Gemm)
            C = f"{output}_zero_bias"
            
            # To determine the shape of C, we need to find the output shape
            # For this, we need to analyze the graph and infer shapes
            
            a_shape = None
            b_shape = None
            
            # Try to find shapes from value_info or initializers
            for vi in model.graph.value_info:
                if vi.name == A:
                    a_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                elif vi.name == B:
                    b_shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
            
            # Check initializers if shapes not found in value_info
            if a_shape is None or b_shape is None:
                for init in model.graph.initializer:
                    if init.name == A:
                        a_shape = list(init.dims)
                    elif init.name == B:
                        b_shape = list(init.dims)
            
            # If we couldn't determine exact shapes, use a placeholder approach
            if a_shape and b_shape:
                # Calculate output shape based on MatMul rules and transA/transB
                if transA:
                    a_shape = a_shape[::-1]
                if transB:
                    b_shape = b_shape[::-1]
                
                # For matmul: [M,K] * [K,N] = [M,N]
                # The bias/C needs to be shape [N]
                c_shape = [b_shape[-1]]
            else:
                # If we can't determine shapes, we'll add a placeholder initializer
                c_shape = [1]  # Placeholder
            
            # Create a zero tensor for C input
            zero_tensor = numpy_helper.from_array(
                np.zeros(c_shape, dtype=np.float32),
                name=C
            )
            new_initializers.append(zero_tensor)
            
            # Create the Gemm node
            gemm_node = helper.make_node(
                "Gemm",
                inputs=[A, B, C],  # Using A, B, C naming convention
                outputs=[output],
                name=f"{node.name}_gemm",
                alpha=alpha,
                beta=1.0,  # Standard beta value
                transA=transA,
                transB=transB
            )
            
            new_nodes.append(gemm_node)
        else:
            # Keep other nodes as they are
            new_nodes.append(node)
    
    # Create a new graph with updated nodes and initializers
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=list(model.graph.initializer) + new_initializers,
        value_info=model.graph.value_info
    )
    
    # Create a new model with the updated graph
    # Preserve opset imports and other model metadata
    new_model = helper.make_model(
        new_graph,
        producer_name="FusedMatMul2Gemm",
        opset_imports=model.opset_import,
        ir_version=model.ir_version
    )
    
    # Copy domain information for custom ops
    for domain in model.domain:
        new_model.domain.append(domain)
    
    # Copy model metadata
    new_model.metadata_props.extend(model.metadata_props)

    # Save the new model
    onnx.save(new_model, output_model_path)
    print(f"Converted model saved to {output_model_path}")
    
    return new_model

def convert_sum_to_add(input_model_path, output_model_path):
    """
    Convert Sum operators to Add operators in an ONNX model.
    Sum operator can take multiple inputs, while Add takes exactly two inputs.
    This function breaks down Sum operators with >2 inputs into a series of Add operators.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the converted ONNX model
    """
    # Load the model
    model = onnx.load(input_model_path)
    
    # Track necessary changes
    new_nodes = []
    processed_nodes = set()
    
    # Process each node in the graph
    for i, node in enumerate(model.graph.node):
        # Skip already processed nodes
        if i in processed_nodes:
            continue
            
        # Check if the node is a Sum operator
        if node.op_type == "Sum":
            input_count = len(node.input)
            
            if input_count == 1:
                # Sum with one input is just an Identity
                identity_node = helper.make_node(
                    "Identity",
                    inputs=[node.input[0]],
                    outputs=node.output,
                    name=f"{node.name}_identity"
                )
                new_nodes.append(identity_node)
            
            elif input_count == 2:
                # Sum with two inputs can be directly converted to Add
                add_node = helper.make_node(
                    "Add",
                    inputs=[node.input[0], node.input[1]],
                    outputs=node.output,
                    name=f"{node.name}_add"
                )
                new_nodes.append(add_node)
                
            else:
                # Sum with more than two inputs needs to be broken down into a series of Add operations
                # We'll create intermediate outputs for all but the last Add
                intermediate_outputs = []
                
                for j in range(input_count - 1):
                    if j == 0:
                        # First Add takes the first two inputs of Sum
                        input1 = node.input[0]
                        input2 = node.input[1]
                    else:
                        # Subsequent Adds take the output of the previous Add and the next input
                        input1 = intermediate_outputs[-1]
                        input2 = node.input[j + 1]
                    
                    # For the last Add, use the original output, otherwise create an intermediate output
                    if j == input_count - 2:
                        output = node.output[0]
                    else:
                        output = f"{node.name}_intermediate_{j}"
                        intermediate_outputs.append(output)
                    
                    # Create the Add node
                    add_node = helper.make_node(
                        "Add",
                        inputs=[input1, input2],
                        outputs=[output],
                        name=f"{node.name}_add_{j}"
                    )
                    new_nodes.append(add_node)
            
            # Mark this node as processed
            processed_nodes.add(i)
        else:
            # Keep other nodes as they are
            new_nodes.append(node)
    
    # Create a new graph with updated nodes
    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        value_info=model.graph.value_info
    )
    
    # Create a new model with the updated graph
    # Preserve opset imports and other model metadata
    new_model = helper.make_model(
        new_graph,
        producer_name="SumToAddConverter",
        opset_imports=model.opset_import,
        ir_version=model.ir_version
    )
    
    # Copy domain information for custom ops
    for domain in model.domain:
        new_model.domain.append(domain)
    
    # Copy model metadata
    new_model.metadata_props.extend(model.metadata_props)
    
    # Save the new model
    onnx.save(new_model, output_model_path)
    print(f"Converted model saved to {output_model_path}")
    
    return new_model

def rename_softmaxgrad_op(input_model_path: str, output_model_path: str, 
                          old_op_name: str = "SoftmaxGrad_13", 
                          new_op_name: str = "SoftmaxGrad"):
    """
    Rename Microsoft's custom operator SoftmaxGrad_13 to SoftmaxGrad.
    
    Args:
        input_model_path: Path to the input ONNX model
        output_model_path: Path to save the converted ONNX model
        old_op_name: Original operator name (default: "SoftmaxGrad_13")
        new_op_name: New operator name (default: "SoftmaxGrad")
    """
    model = onnx.load(input_model_path)
    
    modified_nodes = []
    modified_count = 0
    
    # Process each node in the graph
    for node in model.graph.node:
        # Check if the node is the target Microsoft domain operator
        if node.op_type == old_op_name and node.domain == "com.microsoft":
            modified_count += 1
            
            # Create a new node with the updated op_type
            new_node = helper.make_node(
                op_type=new_op_name,
                inputs=list(node.input),
                outputs=list(node.output),
                name=node.name,
                domain=node.domain  # Keep the original domain
            )
            
            # Copy all attributes from the original node
            for attr in node.attribute:
                new_node.attribute.append(attr)
            
            modified_nodes.append(new_node)
        else:
            # Keep all other nodes as they are
            modified_nodes.append(node)
    
    print(f"Modified {modified_count} {old_op_name} nodes to {new_op_name}")
    
    # Create a new graph with the modified nodes
    new_graph = helper.make_graph(
        nodes=modified_nodes,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer
    )
    
    # Copy over value_info from the original model
    for vi in model.graph.value_info:
        new_graph.value_info.append(vi)
    
    # Create a new model with the updated graph
    new_model = helper.make_model(
        new_graph,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        domain=model.domain,
        model_version=model.model_version,
        doc_string=model.doc_string
    )
    
    # Copy over IR version and opset imports
    new_model.ir_version = model.ir_version
    new_model.opset_import.extend(model.opset_import)
    
    # Save the model
    onnx.save(new_model, output_model_path)
    print(f"Saved to {output_model_path}")
    
    return new_model

def remove_softmax_loss_outputs(input_model_path, output_model_path):
    """
    Remove loss outputs from SoftmaxCrossEntropyLoss nodes, keeping only the log probability output.
    
    Args:
        input_model_path (str): Path to the input ONNX model
        output_model_path (str): Path to save the modified ONNX model
    """
    import onnx
    
    # Load the model
    model = onnx.load(input_model_path)
    graph = model.graph
    
    # Find SoftmaxCrossEntropyLoss nodes
    target_nodes = []
    for node in graph.node:
        if node.op_type == 'SoftmaxCrossEntropyLoss':
            target_nodes.append(node)
    
    print(f"Found {len(target_nodes)} SoftmaxCrossEntropyLoss nodes")
    
    # Outputs to remove (first output - loss)
    outputs_to_remove = []
    
    # Create new nodes with modified outputs
    new_nodes = []
    for node in graph.node:
        if node.op_type == 'SoftmaxCrossEntropyLoss' and len(node.output) > 1:
            # Keep only the second output (log probabilities) and remove the first (loss)
            outputs_to_remove.append(node.output[0])
            
            # Create a new node with only the second output
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            log_prob_output = node.output[1]
            
            # Clear outputs and set only the log probability output
            del new_node.output[:]
            new_node.output.append(log_prob_output)
            
            new_nodes.append(new_node)
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)
    
    # Replace all nodes with the new set
    del graph.node[:]
    graph.node.extend(new_nodes)
    
    # Filter graph outputs to remove loss outputs
    new_outputs = []
    for output in graph.output:
        if output.name not in outputs_to_remove:
            new_outputs.append(output)
    
    # Replace graph outputs with filtered list
    del graph.output[:]
    graph.output.extend(new_outputs)
    
    # Save the modified model
    onnx.save(model, output_model_path)
    print(f"Saved model with loss outputs removed to: {output_model_path}")

def remove_softmax_grad_loss_inputs(input_model_path, output_model_path):
    
    # Load the model
    model = onnx.load(input_model_path)
    graph = model.graph
    
    # Find SoftmaxCrossEntropyLossGrad nodes
    target_nodes = []
    for node in graph.node:
        if node.op_type == 'SoftmaxCrossEntropyLossGrad':
            target_nodes.append(node)
    
    print(f"Found {len(target_nodes)} SoftmaxCrossEntropyLossGrad nodes")
    
    # Inputs to remove (first input)
    inputs_to_remove = []
    
    # Create new nodes with modified inputs
    new_nodes = []
    for node in graph.node:
        if node.op_type == 'SoftmaxCrossEntropyLossGrad' and len(node.input) > 2:
            # Remove the first input and keep the rest
            first_input = node.input[0]
            inputs_to_remove.append(first_input)
            
            # Create a new node without the first input
            new_node = onnx.NodeProto()
            new_node.CopyFrom(node)
            
            # Keep only the second and third inputs
            remaining_inputs = list(node.input[1:])
            del new_node.input[:]
            new_node.input.extend(remaining_inputs)
            
            new_nodes.append(new_node)
        else:
            # Keep other nodes unchanged
            new_nodes.append(node)
    
    # Replace all nodes with the new set
    del graph.node[:]
    graph.node.extend(new_nodes)
    
    # Save the modified model
    onnx.save(model, output_model_path)
    print(f"Saved model with SoftmaxCrossEntropyLossGrad first input removed to: {output_model_path}")

def optimize_softmax_axis(input_model_path, output_model_path):
   
    model = onnx.load(input_model_path)
    
    # Track if we made any changes
    optimized = False
    
    # Create a map of value_info by name for easy access
    value_info_map = {vi.name: vi for vi in model.graph.value_info}
    value_info_map.update({vi.name: vi for vi in model.graph.input})
    value_info_map.update({vi.name: vi for vi in model.graph.output})
    
    # Function to get shape from value_info
    def get_shape(tensor_name):
        if tensor_name in value_info_map:
            shape = []
            for dim in value_info_map[tensor_name].type.tensor_type.shape.dim:
                if dim.dim_param:
                    # Handle symbolic dimensions (set to -1 for dynamic dimension)
                    shape.append(-1)
                else:
                    shape.append(dim.dim_value)
            return shape
        return None
    
    # Track the names of nodes to be removed
    nodes_to_remove = []
    
    # Track new nodes and value_infos to be added
    new_nodes = []
    new_value_infos = []
    
    # For each node in the graph
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Softmax":
            # Get the input and output names
            input_name = node.input[0]
            output_name = node.output[0]
            
            # Get the axis attribute
            axis = None
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
                    break
            
            # If axis is not set, it defaults to 1 in ONNX
            if axis is None:
                axis = 1
            
            # Get the input shape
            input_shape = get_shape(input_name)
            if input_shape is None:
                print(f"Warning: Could not determine shape for {input_name}, skipping optimization")
                continue
            
            # Check if all dimensions after axis are 1
            all_ones_after_axis = all(dim == 1 for dim in input_shape[axis+1:]) if axis+1 < len(input_shape) else True
            
            # Only optimize if the axis is not the last dimension and all subsequent dimensions are 1
            if axis != len(input_shape) - 1 and all_ones_after_axis and axis >= 0:
                print(f"Optimizing Softmax node with input shape {input_shape} and axis={axis}")
                
                # Create unique names for intermediate tensors
                reshape_before_output = f"{input_name}_reshaped_before_softmax"
                softmax_output = f"{output_name}_after_softmax"
                
                # Calculate new shapes
                # Move the axis dimension to the end and flatten all the 1s
                new_shape_before = []
                for i in range(len(input_shape)):
                    if i < axis:
                        new_shape_before.append(input_shape[i])
                    elif i == axis:
                        continue
                    elif i > axis:
                        continue
                new_shape_before.append(input_shape[axis])
                
                # Create reshape node before softmax
                reshape_before_node = helper.make_node(
                    "Reshape",
                    inputs=[input_name, f"{input_name}_shape_before"],
                    outputs=[reshape_before_output],
                    name=f"Reshape_before_softmax_{output_name}"
                )
                
                # Create initializer for the shape tensor
                shape_tensor_before = numpy_helper.from_array(
                    np.array(new_shape_before, dtype=np.int64),
                    name=f"{input_name}_shape_before"
                )
                
                # Create new softmax node with axis set to -1 (last dimension)
                new_softmax_node = helper.make_node(
                    "Softmax",
                    inputs=[reshape_before_output],
                    outputs=[softmax_output],
                    name=f"Softmax_optimized_{output_name}",
                    axis=-1  # Use -1 to always target the last dimension
                )
                
                # Create reshape node after softmax to restore original shape
                reshape_after_node = helper.make_node(
                    "Reshape",
                    inputs=[softmax_output, f"{output_name}_shape_after"],
                    outputs=[output_name],
                    name=f"Reshape_after_softmax_{output_name}"
                )
                
                # Create initializer for the shape tensor
                shape_tensor_after = numpy_helper.from_array(
                    np.array(input_shape, dtype=np.int64),
                    name=f"{output_name}_shape_after"
                )
                
                # Create value info for reshape_before_output
                reshape_before_vi = helper.make_tensor_value_info(
                    reshape_before_output,
                    value_info_map[input_name].type.tensor_type.elem_type,
                    new_shape_before
                )
                
                # Create value info for softmax_output
                softmax_output_vi = helper.make_tensor_value_info(
                    softmax_output,
                    value_info_map[output_name].type.tensor_type.elem_type,
                    new_shape_before  # Shape doesn't change after softmax
                )
                
                # Add all new nodes and value infos
                new_nodes.extend([reshape_before_node, new_softmax_node, reshape_after_node])
                new_value_infos.extend([reshape_before_vi, softmax_output_vi])
                model.graph.initializer.extend([shape_tensor_before, shape_tensor_after])
                
                # Mark the original node for removal
                nodes_to_remove.append(node)
                optimized = True
    
    # Remove the original nodes that were optimized
    for node in nodes_to_remove:
        model.graph.node.remove(node)
    
    # Add the new nodes and value infos
    model.graph.node.extend(new_nodes)
    model.graph.value_info.extend(new_value_infos)
    
    # Save the optimized model
    print(f"Saving optimized model to {output_model_path}")
    onnx.save(model, output_model_path)
    
    print(f"Optimization complete. Modified {len(nodes_to_remove)} Softmax nodes.")
    return optimized

