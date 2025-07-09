import onnx
from onnx import helper
import torch
import os
import sys
import io
from CCT.cct import cct_test  
from onnxruntime.training import artifacts
from utils.utils import *
from utils.fixshape import infer_shapes_with_custom_ops, print_onnx_shapes
from mnistCheckpoint import create_test_input_output
from utils.appendoptimizer import *

def generate_cct_training_onnx(save_path=None):
    """ Generate ONNX training model for CCT based on config, with optional save path """

    pretrained, img_size, num_classes, embedding_dim, num_heads, num_layers, batch_size, opset_version = load_config()

    input_shape = (batch_size, 3, img_size, img_size)  

    folder_name = f"CCT_train_{img_size}_{embedding_dim}_{num_heads}_{num_layers}"
    

    base_path = save_path if save_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx", folder_name)
    os.makedirs(base_path, exist_ok=True)  # Ensure directory exists

    onnx_infer_file = os.path.join(base_path, "network_infer.onnx")  
    onnx_train_file = os.path.join(base_path, "network_train.onnx")  
    onnx_output_file = os.path.join(base_path, "network.onnx")
    onnx_train_optim = os.path.join(base_path, "network_train_optim.onnx")

    # Create CCT model and randomize layer norm parameters
    model = cct_test(
        pretrained=pretrained, 
        img_size=img_size, 
        num_classes=num_classes, 
        embedding_dim=embedding_dim, 
        num_heads=num_heads,  
        num_layers=num_layers,
        n_conv_layers = 2
    )
    model.train()
    model = randomize_layernorm_params(model)

    # Generate random input data for export
    input_tensor = torch.randn(*input_shape, dtype=torch.float32)

    # Export model to ONNX in training mode
    f = io.BytesIO()
    torch.onnx.export(
        model,
        input_tensor,
        f,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        do_constant_folding=False,  # Ensure parameters are not folded into constants
        # training=torch.onnx.TrainingMode.TRAINING,
        # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        export_params=True,
        keep_initializers_as_inputs=False,
    )

    # Load ONNX model from buffer and save it as network_infer.onnx
    onnx_model = onnx.load_model_from_string(f.getvalue())
    # print("Randomizing initializers in inference model...")
    onnx_model = randomize_onnx_initializers(onnx_model)

    onnx.save(onnx_model, onnx_infer_file)
    print(f"✅ Inference ONNX model saved to {onnx_infer_file}")

    # Run optimization on the inference model
    rename_and_save_onnx(onnx_infer_file, onnx_infer_file)
    run_onnx_optimization(onnx_infer_file, embedding_dim, num_heads, input_shape)
    print_onnx_shapes(onnx_infer_file)
    onnx_model = onnx.load(onnx_infer_file)

    # Get all parameter names and require_grad names
    all_param_names = [init.name for init in onnx_model.graph.initializer]
    print(f" All Parameters: {all_param_names}")

    # requires_grad = [name for name in all_param_names if name in [
    # 'classifier_norm_bias', 'classifier_norm_weight', 'classifier_attention_pool_weight', 'classifier_attention_pool_bias', 'classifier_fc_weight', 'classifier_blocks_0_pre_norm_bias', 'classifier_fc_bias', 'node_0_classifier_attention_pool_Transpose__0'
    # ]]  
    # requires_grad = [ name for name in all_param_names if "const" not in name]
    
    # requires_grad = [name for name in all_param_names if name in [
    # 'classifier_fc_weight', 'classifier_fc_bias',  'node_0_classifier_attention_pool_Transpose__0', 'classifier_norm_weight', 'classifier_norm_bias', 'classifier_attention_pool_bias' ]]
    requires_grad = [name for name in all_param_names if name in [
    'classifier_fc_weight', 'classifier_fc_bias' ]]
    # requires_grad = [name for name in all_param_names if name in [
    # 'classifier_fc_weight', 'classifier_fc_bias']]

    # requires_grad = [name for name in all_param_names if name in [
    # 'node_0_classifier_blocks_0_linear1_Transpose__0', 'classifier_blocks_0_linear1_bias', 'node_0_classifier_blocks_0_linear2_Transpose__0'
    # ]]  
    # requires_grad = [name for name in all_param_names if name in [
    # 'node_0_classifier_blocks_0_self_attn_q_proj_Transpose__0', 'node_0_classifier_blocks_0_self_attn_k_proj_Transpose__0', 'node_0_classifier_blocks_0_self_attn_v_proj_Transpose__0', 
    # 'node_0_classifier_blocks_0_self_attn_proj_Transpose__0', 'classifier_blocks_0_self_attn_proj_bias', 'classifier_blocks_0_pre_norm_weight', 'classifier_blocks_0_pre_norm_bias', 'classifier_positional_emb'
    # ]]  

    frozen_params = [name for name in all_param_names if name not in requires_grad]
    
    print(f"🔹 Training Only: {requires_grad}")
    print(f"🔹 Frozen Parameters: {frozen_params}")


    # Generate artifacts for training
    artifacts.generate_artifacts(
        onnx_model,
        optimizer=artifacts.OptimType.SGD,
        loss=artifacts.LossType.CrossEntropyLoss,
        requires_grad=requires_grad,
        frozen_params=frozen_params,  
        artifact_directory=base_path,

    )

    training_model_path = os.path.join(base_path, "training_model.onnx")
    if os.path.exists(training_model_path):
        os.rename(training_model_path, onnx_train_file)
        print(f"✅ Final Training ONNX model saved as {onnx_train_file}")

    # load the training model
    onnx_model = onnx.load(onnx_train_file)
    graph = onnx_model.graph
    grad_tensor_names = [ name + '_grad' for name in requires_grad ]
    

    for grad_name in grad_tensor_names:
        if not any(output.name == grad_name for output in graph.output):

            grad_output = helper.make_tensor_value_info(grad_name, onnx.TensorProto.FLOAT, None)
            graph.output.append(grad_output)  
    onnx.save(onnx_model, onnx_train_optim)
    onnx.save(onnx_model, onnx_train_file)

    # train file for generating golden model debug
    # train_optim file for further optimization

    # Run optimization on the training model
    onnx_output_file = os.path.join(base_path, "network.onnx")
    run_train_onnx_optimization(onnx_train_optim, onnx_output_file)
    infer_shapes_with_custom_ops(onnx_output_file, onnx_output_file)
    rename_nodes(onnx_output_file, onnx_output_file)
    print_onnx_shapes(onnx_output_file)
   
    print(f"✅ Training ONNX model saved to {onnx_output_file}")
    create_test_input_output()
    print(f"✅ Created test input and output data")

    learning_rate = load_train_config()
    add_sgd_nodes(onnx_output_file, onnx_output_file, learning_rate=learning_rate)
    infer_shapes_with_custom_ops(onnx_output_file, onnx_output_file)
    type_inference(onnx_output_file, onnx_output_file)
    print(f"✅ Added SGD nodes to {onnx_output_file}")

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_cct_training_onnx(save_path)
