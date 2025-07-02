import onnx
import numpy as np
import argparse
from onnx import helper, numpy_helper, shape_inference, TensorProto
import logging
from collections import defaultdict
import sys
import copy
import onnx
import numpy as np
from onnx import shape_inference
from typing import List, Dict, Any, Optional

def register_custom_shape_inference():
   
    
    from onnx.shape_inference import _bring_proto, _get_shape_calculator_dict
    
    def softmax_cross_entropy_grad_shape_inference(ctx):
    
        node = ctx.node
        
  
        log_prob_type_proto = ctx.get_input_type(2)
        if log_prob_type_proto is None:
            return
            
     
        ctx.set_output_type(0, log_prob_type_proto)
  
        print(f"SoftmaxCrossEntropyGrad shape inference: output shape set from log_prob input")
    
 
    shape_calculator_dict = _get_shape_calculator_dict()
    shape_calculator_dict["com.microsoft.SoftmaxCrossEntropyGrad"] = softmax_cross_entropy_grad_shape_inference


def infer_shapes_with_custom_ops(model_path: str, output_model_path: Optional[str] = None) -> onnx.ModelProto:

    model = onnx.load(model_path)
    
    op_types = set(node.op_type for node in model.graph.node)
    microsoft_ops = [op for op in op_types if op.startswith("com.microsoft")]
    print(f"Find Microsoft Custom Op: {microsoft_ops}")
    
    try:
    
        register_custom_shape_inference()
        
  
        inferred_model = shape_inference.infer_shapes(model)
        print("Shape Inference Succeeded")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Try to infer ...")

        inferred_model = model
        for i, node in enumerate(model.graph.node):
            try:
             
                subgraph_model = extract_subgraph(model, [node])
            
                inferred_subgraph = shape_inference.infer_shapes(subgraph_model)
         
                update_model_with_inferred_shapes(inferred_model, inferred_subgraph, i)
                print(f"Node {i}: {node.op_type} Succeed")
            except Exception as node_err:
                print(f"Node {i}: {node.op_type} Shape inference fail: {str(node_err)}")
                if node.op_type.startswith("com.microsoft"):
                    print(f"Try to use microsoft : {node.op_type}")
                    try:
                        apply_custom_inference(inferred_model.graph, node)
                        print(f"Node {i}: {node.op_type} infered by custom inference")
                    except Exception as custom_err:
                        print(f"Infered failed: {str(custom_err)}")
    
    if output_model_path:
        onnx.save(inferred_model, output_model_path)
        print(f"Onnx with shape saved: {output_model_path}")
    
    return inferred_model


def apply_custom_inference(graph: onnx.GraphProto, node: onnx.NodeProto) -> None:
  
    if node.op_type == "com.microsoft.SoftmaxCrossEntropyGrad":
       
      
        if len(node.input) >= 3 and len(node.output) >= 1:
            log_prob_shape = get_tensor_shape(graph, node.input[2])
            if log_prob_shape:
                set_tensor_shape(graph, node.output[0], log_prob_shape)
                print(f"SoftmaxCrossEntropyGrad: {log_prob_shape}")
  
    elif node.op_type.startswith("com.microsoft"):
    
        if "Grad" in node.op_type:
       
            if len(node.input) >= 2 and len(node.output) >= 1:
                input_shape = get_tensor_shape(graph, node.input[1])
                if input_shape:
                    set_tensor_shape(graph, node.output[0], input_shape)
                    print(f"Set {node.op_type} Output with same size of input {node.input[1]} : {input_shape}")


def extract_subgraph(model: onnx.ModelProto, nodes: List[onnx.NodeProto]) -> onnx.ModelProto:
   
    subgraph = onnx.ModelProto()
    subgraph.CopyFrom(model)
    
   
    del subgraph.graph.node[:]
    subgraph.graph.node.extend(nodes)
    
    return subgraph


def update_model_with_inferred_shapes(model: onnx.ModelProto, inferred_subgraph: onnx.ModelProto, node_index: int) -> None:
   
    if node_index < len(model.graph.node):
        node = model.graph.node[node_index]
        inferred_node = inferred_subgraph.graph.node[0]
        
   
        for i, output_name in enumerate(node.output):
            if i < len(inferred_node.output):
                update_value_info_shape(model.graph, output_name, 
                                       get_value_info_by_name(inferred_subgraph.graph, inferred_node.output[i]))


def get_value_info_by_name(graph: onnx.GraphProto, name: str) -> Optional[onnx.ValueInfoProto]:
 
    for info in graph.output:
        if info.name == name:
            return info
    

    for info in graph.value_info:
        if info.name == name:
            return info
    

    for info in graph.input:
        if info.name == name:
            return info
    
    return None


def update_value_info_shape(graph: onnx.GraphProto, name: str, value_info: Optional[onnx.ValueInfoProto]) -> None:

    if not value_info or not value_info.type.tensor_type.shape:
        return
    
  
    existing_info = get_value_info_by_name(graph, name)
    if existing_info:
        existing_info.type.tensor_type.shape.CopyFrom(value_info.type.tensor_type.shape)
    else:
      
        new_info = onnx.ValueInfoProto()
        new_info.name = name
        new_info.type.tensor_type.shape.CopyFrom(value_info.type.tensor_type.shape)
        graph.value_info.append(new_info)


def set_tensor_shape(graph: onnx.GraphProto, tensor_name: str, shape: List[int]) -> None:
 
    value_info = get_value_info_by_name(graph, tensor_name)
    if not value_info:
    
        value_info = onnx.ValueInfoProto()
        value_info.name = tensor_name
        graph.value_info.append(value_info)
    
  
    value_info.type.tensor_type.shape.Clear()
    
 
    for dim_value in shape:
        dim = value_info.type.tensor_type.shape.dim.add()
        if dim_value > 0:
            dim.dim_value = dim_value
        else:
          
            dim.dim_param = "?"

def get_tensor_shape(model, tensor_name):
    for initializer in model.graph.initializer:
        if initializer.name == tensor_name:
            return tuple(initializer.dims)
    
    for input_tensor in model.graph.input:
        if input_tensor.name == tensor_name:
            return tuple(dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim)
    
    for output_tensor in model.graph.output:
        if output_tensor.name == tensor_name:
            return tuple(dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim)
    
    return None

def print_onnx_shapes(model_path):
    model= onnx.load(model_path)
    graph = model.graph
    shape_info = {}
    for input_info in graph.input:
        shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        shape_info[input_info.name] = shape
    

    for output_info in graph.output:
        shape = []
        for dim in output_info.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        shape_info[output_info.name] = shape
    
    for value_info in graph.value_info:
        shape = []
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
        shape_info[value_info.name] = shape

    initializers = {init.name for init in graph.initializer}
    
    print("\nInput:")
    for input_info in graph.input:
        if input_info.name in shape_info:
            print(f"  {input_info.name}: {shape_info[input_info.name]}")
        else:
            print(f"  {input_info.name}: Unknown")
    
    
    print("\nOutput:")
    for output_info in graph.output:
        if output_info.name in shape_info:
            print(f"  {output_info.name}: {shape_info[output_info.name]}")
        else:
            print(f"  {output_info.name}: Unknown")
    

    print("\nNode info:")
    for i, node in enumerate(graph.node):
        print(f"\nNode {i+1}: {node.name} (type: {node.op_type})")
        
        print("  Input:")
        for j, input_name in enumerate(node.input):
            if input_name in initializers:
                print(f"    {j+1}. {input_name}: [Initializer]")
            elif input_name in shape_info:
                print(f"    {j+1}. {input_name}: {shape_info[input_name]}")
            else:
                print(f"    {j+1}. {input_name}: Unknown")
        
        print("  Output:")
        for j, output_name in enumerate(node.output):
            if output_name in shape_info:
                print(f"    {j+1}. {output_name}: {shape_info[output_name]}")
            else:
                print(f"    {j+1}. {output_name}: Unknown")
        
        if node.attribute:
            print("  Property:")
            for attr in node.attribute:
                print(f"    {attr.name}")