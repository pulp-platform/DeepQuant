import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
import torch
from CCT.cct import cct_test  
from utils.utils import *

def generate_cct_onnx_and_data(save_path=None):
    """ Generate ONNX model for CCT based on config, with optional save path """

    pretrained, img_size, num_classes, embedding_dim, num_heads, num_layers, batch_size, opset_version = load_config()
    print(f"✅ Loaded config: img_size={img_size}, embedding_dim={embedding_dim}, num_heads={num_heads}, num_layers={num_layers}, opset_version={opset_version}")

    input_shape = (1, 3, img_size, img_size)  

    folder_name = f"CCT_infer_{img_size}_{embedding_dim}_{num_heads}_{num_layers}"

    base_path = save_path if save_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx", folder_name)

    onnx_file = os.path.join(base_path, "network.onnx")
    input_file = os.path.join(base_path, "inputs.npz")
    output_file = os.path.join(base_path, "outputs.npz")

    os.makedirs(base_path, exist_ok=True)

    model = cct_test(
        pretrained=pretrained, 
        img_size=img_size, 
        num_classes=num_classes, 
        embedding_dim=embedding_dim, 
        num_heads=num_heads,  
        num_layers=num_layers,
        n_conv_layers=2  
    )
    model.eval()
    model = randomize_layernorm_params(model)

    input_data = np.random.randn(*input_shape).astype(np.float32)
    np.savez(input_file, input=input_data)

    input_tensor = torch.tensor(input_data)

    torch.onnx.export(
        model,
        input_tensor,
        onnx_file,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    onnx_model = onnx.load(onnx_file)
    onnx_model = randomize_onnx_initializers(onnx_model)
    print(f"✅ ONNX model saved to {onnx_file}")
    rename_and_save_onnx(onnx_file, onnx_file)

    run_onnx_optimization_infer(onnx_file, embedding_dim, num_heads, input_shape)
    rename_and_save_onnx(onnx_file, onnx_file)
    ort_session = ort.InferenceSession(onnx_file)
    output_data = ort_session.run(None, {"input": input_data})[0]

    np.savez(output_file, output=output_data)
    print(f"✅ Output data saved to {output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_cct_onnx_and_data(save_path)
