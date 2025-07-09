import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
import re

def add_sgd_nodes(model_path, output_path, learning_rate=0.01):
    """
    Add SGD nodes to the ONNX model and remove original gradient outputs.
    
    Args:
        model_path: Path to the original ONNX model
        output_path: Path to save the modified ONNX model
        learning_rate: Learning rate for SGD (default: 0.01)
    """
    # Load the ONNX model
    model = onnx.load(model_path)
    graph = model.graph
    
    # Get reference to the output nodes
    node_133_output = None
    classifier_fc_weight_grad = None
    
    # Store all outputs for later filtering
    original_outputs = []
    grad_outputs_to_remove = []
    
    for output in graph.output:
        original_outputs.append(output)
        if re.match(r"node_\d+_classifier_fc_Gemm_Grad_dC_reduced", output.name):
            node_classifier_output = output
            grad_outputs_to_remove.append(output)
        elif output.name == "classifier_fc_weight_grad":
            classifier_fc_weight_grad = output
            grad_outputs_to_remove.append(output)
    
    if not node_classifier_output or not classifier_fc_weight_grad:
        raise ValueError("Could not find required output nodes")
    
    # Find the original classifier_fc_Gemm node to get initializers
    classifier_fc_weight = None
    classifier_fc_bias = None
    
    for initializer in graph.initializer:
        if initializer.name == "classifier_fc_weight":
            classifier_fc_weight = initializer
        elif initializer.name == "classifier_fc_bias":
            classifier_fc_bias = initializer
    
    if not classifier_fc_weight or not classifier_fc_bias:
        raise ValueError("Could not find required initializers")
    
    # Check output shapes and ensure they match
    from onnx import numpy_helper
    weight_shape = list(numpy_helper.to_array(classifier_fc_weight).shape)  # Convert to list
    bias_shape = list(numpy_helper.to_array(classifier_fc_bias).shape)  # Convert to list
    
    weight_grad_shape = get_output_shape(graph, "classifier_fc_weight_grad")
    bias_grad_shape = get_output_shape(graph, "classifier_fc_weight_grad")
    
    print(f"Weight shape: {weight_shape}, Weight grad shape: {weight_grad_shape}")
    print(f"Bias shape: {bias_shape}, Bias grad shape: {bias_grad_shape}")
    
    # Compare shapes after converting to the same type (list)
    if weight_shape != weight_grad_shape:
        print("Warning: Weight shape doesn't match gradient shape, but will continue")
        print(f"Weight type: {type(weight_shape)}, Gradient type: {type(weight_grad_shape)}")
    
    # Also convert bias shapes to ensure proper comparison
    if bias_shape != bias_grad_shape:
        print("Warning: Bias shape doesn't match gradient shape, but will continue")
        print(f"Bias type: {type(bias_shape)}, Gradient type: {type(bias_grad_shape)}")
    
    # Convert shapes back to tuples for creating tensor_value_info
    weight_shape_tuple = tuple(weight_shape)
    bias_shape_tuple = tuple(bias_shape)
    
    # Get the proper tensor types for output shape inference
    weight_type = get_value_info_type(classifier_fc_weight)
    bias_type = get_value_info_type(classifier_fc_bias)
    
    # Create SGD node for the weight gradient with learning_rate as attribute
    sgd_weight_node = helper.make_node(
        op_type="SGD",
        inputs=[
            "classifier_fc_weight",      # weights to update
            "classifier_fc_weight_grad", # gradient
        ],
        outputs=["classifier_fc_weight_updated"],
        name="classifier_fc_weight_sgd",
        domain="",
        lr=float(learning_rate)  # Ensure learning_rate is a float and set as attribute
    )
    
    # Create SGD node for the bias gradient with learning_rate as attribute
    sgd_bias_node = helper.make_node(
        op_type="SGD",
        inputs=[
            "classifier_fc_bias",             # bias to update
            "classifier_fc_weight_grad",  # gradient
        ],
        outputs=["classifier_fc_bias_updated"],
        name="classifier_fc_bias_sgd",
        domain="",
        lr=float(learning_rate)  # Ensure learning_rate is a float and set as attribute
    )
    
    # Add the new SGD nodes to the graph
    graph.node.extend([sgd_weight_node, sgd_bias_node])
    
    # Create value info for the outputs with proper shapes
    updated_weight_output = helper.make_tensor_value_info(
        name="classifier_fc_weight_updated",
        elem_type=TensorProto.FLOAT,
        shape=weight_shape_tuple
    )
    
    updated_bias_output = helper.make_tensor_value_info(
        name="classifier_fc_bias_updated",
        elem_type=TensorProto.FLOAT,
        shape=bias_shape_tuple
    )
    
    # Clear original outputs and add only what we want to keep
    graph.ClearField("output")
    
    # Add only the SGD-updated outputs
    graph.output.extend([updated_weight_output, updated_bias_output])
    
    # Keep any other original outputs that weren't gradients
    for output in original_outputs:
        if output not in grad_outputs_to_remove:
            graph.output.append(output)
    
    # Run shape inference to verify and update all shapes
    try:
        inferred_model = shape_inference.infer_shapes(model)
        print("✅ Shape inference successful")
        model = inferred_model
    except Exception as e:
        print(f"⚠️ Shape inference warning: {e}")
        print("Continuing with explicit shapes...")
    
    # Save the modified model
    onnx.save(model, output_path)
    print(f"✅ Modified model saved to {output_path}")
    print(f"Learning rate set to {learning_rate} as node attribute")
    print("Original gradient outputs have been removed")

def get_output_shape(graph, output_name):
    """
    Get the shape of an output tensor by name.
    
    Args:
        graph: ONNX graph
        output_name: Name of the output tensor
        
    Returns:
        List representing the tensor shape
    """
    # First check among graph outputs
    for output in graph.output:
        if output.name == output_name:
            return [dim.dim_value for dim in output.type.tensor_type.shape.dim]
    
    # If not found in outputs, look for value_info
    for value_info in graph.value_info:
        if value_info.name == output_name:
            return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
    
    raise ValueError(f"Cannot find shape for output {output_name}")

def get_value_info_type(tensor):
    """
    Get the data type of a tensor.
    
    Args:
        tensor: ONNX tensor
        
    Returns:
        ONNX data type
    """
    return tensor.data_type