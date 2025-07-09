import numpy as np
import onnxruntime as ort
import onnx
import os
import torch
import torchvision
from torchvision import transforms
from utils.utils import *

def preprocess_mnist(batch_size, image_size):
    """
    Preprocess MNIST dataset with configurable image size.
    
    Args:
        batch_size: Number of images to process
        image_size: Size to resize images to (will be used as both height and width)
        
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    
    dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    indices = np.random.choice(len(dataset), batch_size, replace=False)
    images = torch.stack([dataset[i][0] for i in indices])
    labels = np.array([dataset[i][1] for i in indices], dtype=np.int64)
    
    return images.numpy(), labels

def run_original_onnx_model(input_data, labels, model_path):
    """
    Run inference on original ONNX model to get gradients.
    
    Args:
        input_data: Input data for the model
        labels: Labels for the model
        model_path: Path to the original ONNX model (without SGD)
        
    Returns:
        Dictionary of model outputs (gradients)
    """
    ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    output_names = [output.name for output in ort_session.get_outputs()]
    print(f"Model has {len(output_names)} outputs: {output_names}")
    
    outputs = ort_session.run(None, {"input": input_data, "labels": labels})
    
    output_dict = {}
    for i, name in enumerate(output_names):
        output_dict[name] = outputs[i]
    
    return output_dict

def get_initializer_from_onnx(model_path, initializer_name):
    """
    Extract initializer tensor from ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        initializer_name: Name of the initializer to extract
        
    Returns:
        Numpy array of the initializer tensor
    """
    model = onnx.load(model_path)
    for initializer in model.graph.initializer:
        if initializer.name == initializer_name:
            # Convert ONNX tensor to numpy array
            from onnx import numpy_helper
            return numpy_helper.to_array(initializer)
    
    raise ValueError(f"Initializer {initializer_name} not found in model")

def apply_sgd_update(weight, gradient, learning_rate=0.01):
    """
    Manually apply SGD update to weights.
    
    Args:
        weight: Current weight tensor
        gradient: Gradient tensor
        learning_rate: Learning rate for SGD
        
    Returns:
        Updated weight tensor
    """
    return weight - learning_rate * gradient

def create_test_input_output():
    """
    Create test input and output files with manual SGD implementation.
    """
    # Load config
    config = load_config()
    if isinstance(config, tuple):
        # Handle the case where load_config returns a tuple of values
        pretrained, img_size, num_classes, embedding_dim, num_heads, num_layers, batch_size, opset_version = config
    else:
        # Handle the case where load_config returns a dictionary
        img_size = config.get("img_size", 16)  # Default to 16 if not specified
        batch_size = config.get("batch_size", 8)  # Default to 8 if not specified
        embedding_dim = config.get("embedding_dim", 384)
        num_heads = config.get("num_heads", 6)
        num_layers = config.get("num_layers", 7)
    
    print(f"Using image size: {img_size}, batch size: {batch_size}")
    
    folder_name = f"CCT_train_{img_size}_{embedding_dim}_{num_heads}_{num_layers}"
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, "onnx", folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Path to original training network
    network_path = os.path.join(base_dir, "onnx", folder_name, "network_train.onnx")
    input_path = os.path.join(base_dir, "onnx", folder_name, "inputs.npz")
    output_path = os.path.join(base_dir, "onnx", folder_name, "outputs.npz")
    
    print(f"Original network path: {network_path}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"ONNX model file not found: {network_path}")
    
    # Create input data with the specified image size
    input_data, labels = preprocess_mnist(batch_size, img_size)
    np.savez(input_path, input=input_data, labels=labels)
    print(f"✅ Input saved to inputs.npz (image size: {img_size}x{img_size}, batch size: {batch_size})")
    
    # Run the original model to get gradients
    outputs_dict = run_original_onnx_model(input_data, labels, model_path=network_path)
    
    # Extract parameter gradients from outputs
    weight_grad_name = None
    bias_grad_name = None
    
    # Try to find the gradient outputs by name patterns
    for name in outputs_dict.keys():
        if "classifier_fc_weight_grad" in name:
            weight_grad_name = name
        elif "classifier_fc_Gemm_Grad_dC_reduced" in name or "classifier_fc_bias_grad" in name:
            bias_grad_name = name
    
    if not weight_grad_name:
        print("❌ Could not find weight gradient in outputs")
        print(f"Available outputs: {list(outputs_dict.keys())}")
        return
    
    if not bias_grad_name:
        print("❌ Could not find bias gradient in outputs")
        print(f"Available outputs: {list(outputs_dict.keys())}")
        return
    
    print(f"Found weight gradient: {weight_grad_name}")
    print(f"Found bias gradient: {bias_grad_name}")
    
    # Extract original weights from model
    try:
        classifier_fc_weight = get_initializer_from_onnx(network_path, "classifier_fc_weight")
        classifier_fc_bias = get_initializer_from_onnx(network_path, "classifier_fc_bias")
        print(f"✅ Successfully extracted original parameters from model")
        print(f"  Weight shape: {classifier_fc_weight.shape}")
        print(f"  Bias shape: {classifier_fc_bias.shape}")
    except Exception as e:
        print(f"❌ Error extracting parameters: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Apply SGD manually
    learning_rate = load_train_config()
    
    weight_grad = outputs_dict[weight_grad_name]
    bias_grad = outputs_dict[bias_grad_name]
    
    # Apply SGD update
    weight_updated = apply_sgd_update(classifier_fc_weight, weight_grad, learning_rate)
    bias_updated = apply_sgd_update(classifier_fc_bias, bias_grad, learning_rate)
    
    print(f"✅ Successfully applied SGD updates to parameters")
    
    # Create output dict with updated parameters
    sgd_outputs = {
        "classifier_fc_weight_updated": weight_updated,
        "classifier_fc_bias_updated": bias_updated
    }
    
    # Save updated parameters to output file
    np.savez(output_path, **sgd_outputs)
    print(f"✅ Updated parameters saved to {output_path}")
    
    # Print output shapes
    print("Final output shapes:")
    for name, arr in sgd_outputs.items():
        print(f"  {name}: {arr.shape}")

def main():
    """
    Main function to run the script with better error handling
    """
    try:
        create_test_input_output()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()