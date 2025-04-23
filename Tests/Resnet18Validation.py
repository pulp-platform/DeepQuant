# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import os
import sys
import json
import tarfile
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

import brevitas.nn as qnn
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)
from brevitas.graph.quantize import preprocess_for_quantize, quantize
from brevitas.graph.per_input import AdaptiveAvgPoolToAvgPool
from brevitas.graph.calibrate import calibration_mode
from DeepQuant.ExportBrevitas import exportBrevitas


def compare_model_outputs(model_fq, model_tq, sample_image, device):
    print("\n===== FQ vs TQ COMPARISON ANALYSIS =====")

    model_fq.eval()
    model_tq.eval()

    if sample_image.dim() == 3:
        sample_image = sample_image.unsqueeze(0)

    with torch.no_grad():
        input_fq = sample_image.to(device)
        output_fq_raw = model_fq(input_fq)

        if hasattr(output_fq_raw, "value"):
            output_fq = output_fq_raw.value.cpu()
        else:
            output_fq = output_fq_raw.cpu()

        input_tq = sample_image.to("cpu")
        output_tq = model_tq(input_tq).cpu()

    pred_fq = output_fq.argmax(dim=1).item()
    pred_tq = output_tq.argmax(dim=1).item()

    print(f"FQ model predicted class: {pred_fq}")
    print(f"TQ model predicted class: {pred_tq}")
    print(f"Identical classification: {pred_fq == pred_tq}")

    total_elements = output_fq.numel()
    exactly_equal = (output_fq == output_tq).sum().item()
    exactly_equal_percent = (exactly_equal / total_elements) * 100

    print(f"\nOutput values exact equality analysis:")
    print(f"Total number of elements: {total_elements}")
    print(
        f"Elements that are exactly equal: {exactly_equal} ({exactly_equal_percent:.2f}%)"
    )
    print(
        f"Elements that differ: {total_elements - exactly_equal} ({100 - exactly_equal_percent:.2f}%)"
    )

    if exactly_equal < total_elements:

        if total_elements - exactly_equal > 5:
            abs_diff = (output_fq - output_tq).abs()
            print("\nTop 5 largest differences:")
            flat_abs_diff = abs_diff.view(-1)
            top_values, top_indices = flat_abs_diff.topk(5)

            for i in range(5):
                idx = top_indices[i].item()
                fq_val = output_fq.view(-1)[idx].item()
                tq_val = output_tq.view(-1)[idx].item()
                diff_val = top_values[i].item()

                print(
                    f"Index {idx}: FQ={fq_val:.6f}, TQ={tq_val:.6f}, Δ={diff_val:.6f}"
                )

    fq_softmax = torch.nn.functional.softmax(output_fq, dim=1)
    tq_softmax = torch.nn.functional.softmax(output_tq, dim=1)

    fq_confidence = fq_softmax.max().item()
    tq_confidence = tq_softmax.max().item()

    print(f"\nFQ model confidence: {fq_confidence:.6f}")
    print(f"TQ model confidence: {tq_confidence:.6f}")
    print(f"Confidence difference: {abs(fq_confidence - tq_confidence):.6f}")


def main():
    # --------------------------------------------------
    # PART 1: IMAGENET VALIDATION
    # --------------------------------------------------

    HOME_DIR = str(Path.home())
    IMAGENET_DIR = os.path.join(HOME_DIR, "Documents/Imagenet")
    IMG_VAL_TAR = os.path.join(IMAGENET_DIR, "ILSVRC2012_img_val.tar")
    VAL_DIR = os.path.join(IMAGENET_DIR, "ILSVRC2012_img_val")
    JSON_PATH = os.path.join(IMAGENET_DIR, "imagenet_class_index.json")

    if not os.path.exists(VAL_DIR):
        sys.exit(f"Validation directory not found at {VAL_DIR}")
    if not os.path.exists(JSON_PATH):
        sys.exit(f"JSON file not found at {JSON_PATH}")

    if os.path.exists(IMG_VAL_TAR) and (
        not os.listdir(VAL_DIR)
        or all(not os.path.isdir(os.path.join(VAL_DIR, f)) for f in os.listdir(VAL_DIR))
    ):
        print(f"Extracting validation images to {VAL_DIR}...")
        with tarfile.open(IMG_VAL_TAR, "r:") as tar:
            for member in tqdm(tar.getmembers(), desc="Extracting images"):
                if member.isreg():
                    tar.extract(member, VAL_DIR)
        print(f"Extraction complete. Files available in {VAL_DIR}")

    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageFolder(root=VAL_DIR, transform=val_transforms)

    with open(JSON_PATH, "r") as f:
        class_index = json.load(f)
    synset_to_idx = {v[0]: int(k) for k, v in class_index.items()}

    new_class_to_idx = {}
    for folder_name in dataset.classes:
        if folder_name in synset_to_idx:
            new_class_to_idx[folder_name] = synset_to_idx[folder_name]
        else:
            print(
                f"Warning: Folder {folder_name} not found in JSON mapping. It will be skipped."
            )
    dataset.class_to_idx = new_class_to_idx

    # FBRANCASI: Optional, reduce number of example for faster validation
    # DATASET_LIMIT = 1000
    # dataset = Subset(dataset, list(range(DATASET_LIMIT)))
    # print(f"Validation dataset size set to {len(dataset)} images.")

    CALIB_BATCH_SIZE = 32
    CALIB_SIZE = 256

    calib_dataset = Subset(dataset, list(range(CALIB_SIZE)))
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=CALIB_BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )
    print(
        f"Calibration DataLoader created (batch size = {CALIB_BATCH_SIZE}, samples = {CALIB_SIZE})."
    )

    BATCH_SIZE = 32
    val_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )
    print(f"Validation DataLoader created (batch size = {BATCH_SIZE}).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else device
    )  # FBRANCASI: I'm on mac, so mps for me
    print(f"Using device: {device}")

    original_model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    original_model = original_model.eval().to(device)
    print("Original ResNet18 loaded.")

    def evaluate_model(model, data_loader, eval_device, name="Model"):
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc=f"Evaluating {name}"):
                is_TQ = "TQ" in name

                if is_TQ:
                    # Process different batches for the TQ model
                    for i in range(inputs.size(0)):
                        single_input = inputs[i : i + 1].to(eval_device)
                        single_output = model(single_input)

                        _, predicted = single_output.max(1)
                        if predicted.item() == targets[i].item():
                            correct_top1 += 1

                        _, top5_pred = single_output.topk(
                            5, dim=1, largest=True, sorted=True
                        )
                        if targets[i].item() in top5_pred[0].cpu().numpy():
                            correct_top5 += 1

                        total += 1
                else:
                    inputs = inputs.to(eval_device)
                    targets = targets.to(eval_device)
                    output = model(inputs)

                    _, predicted = output.max(1)
                    correct_top1 += (predicted == targets).sum().item()

                    _, top5_pred = output.topk(5, dim=1, largest=True, sorted=True)
                    for i in range(targets.size(0)):
                        if targets[i] in top5_pred[i]:
                            correct_top5 += 1

                    total += targets.size(0)

        top1_accuracy = 100.0 * correct_top1 / total
        top5_accuracy = 100.0 * correct_top5 / total
        print(
            f"{name} - Top-1 Accuracy: {top1_accuracy:.2f}% ({correct_top1}/{total}), "
            f"Top-5 Accuracy: {top5_accuracy:.2f}%"
        )
        return top1_accuracy, top5_accuracy

    print("Evaluating original model...")
    original_top1, original_top5 = evaluate_model(
        original_model, val_loader, device, "Original ResNet18"
    )

    def calibrate_model(model, calib_loader):
        model.eval()
        with torch.no_grad(), calibration_mode(model):
            for inputs, _ in tqdm(calib_loader, desc="Calibrating model"):
                inputs = inputs.to("cpu")
                model(inputs)
        print("Calibration completed.")

    def prepare_FQ_resnet18():
        base_model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
        base_model = base_model.eval().to("cpu")

        compute_layer_map = {
            nn.Conv2d: (
                qnn.QuantConv2d,
                {
                    "input_quant": Int8ActPerTensorFloat,
                    "weight_quant": Int8WeightPerTensorFloat,
                    "output_quant": Int8ActPerTensorFloat,
                    "bias_quant": Int32Bias,
                    "bias": True,
                    "return_quant_tensor": True,
                    "output_bit_width": 8,
                    "weight_bit_width": 8,
                },
            ),
            nn.Linear: (
                qnn.QuantLinear,
                {
                    "input_quant": Int8ActPerTensorFloat,
                    "weight_quant": Int8WeightPerTensorFloat,
                    "output_quant": Int8ActPerTensorFloat,
                    "bias_quant": Int32Bias,
                    "bias": True,
                    "return_quant_tensor": True,
                    "output_bit_width": 8,
                    "weight_bit_width": 8,
                },
            ),
        }

        quant_act_map = {
            nn.ReLU: (
                qnn.QuantReLU,
                {
                    "act_quant": Uint8ActPerTensorFloat,
                    "return_quant_tensor": True,
                    "bit_width": 8,
                },
            ),
        }

        quant_identity_map = {
            "signed": (
                qnn.QuantIdentity,
                {
                    "act_quant": Int8ActPerTensorFloat,
                    "return_quant_tensor": True,
                    "bit_width": 8,
                },
            ),
            "unsigned": (
                qnn.QuantIdentity,
                {
                    "act_quant": Uint8ActPerTensorFloat,
                    "return_quant_tensor": True,
                    "bit_width": 8,
                },
            ),
        }

        dummy_input = torch.ones(1, 3, 224, 224).to("cpu")

        print("Preprocessing model for quantization...")
        base_model = preprocess_for_quantize(
            base_model, equalize_iters=20, equalize_scale_computation="range"
        )

        print("Converting AdaptiveAvgPool to AvgPool...")
        base_model = AdaptiveAvgPoolToAvgPool().apply(base_model, dummy_input)

        print("Quantizing model...")
        FQ_model = quantize(
            graph_model=base_model,
            compute_layer_map=compute_layer_map,
            quant_act_map=quant_act_map,
            quant_identity_map=quant_identity_map,
        )

        return FQ_model

    print("Preparing and quantizing ResNet18...")
    FQ_model = prepare_FQ_resnet18()

    print("Calibrating FQ model...")
    calibrate_model(FQ_model, calib_loader)

    print("Evaluating FQ model...")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # FBRANCASI: I'm on mac, mps doesn't work with brevitas
    FQ_top1, FQ_top5 = evaluate_model(FQ_model, val_loader, device, "FQ ResNet18")

    print("Exporting FQ model with exportBrevitas...")
    sample_input_img = None
    sample_target = None
    for inputs, targets in val_loader:
        sample_input_img = inputs[17]
        sample_target = targets[17].item()
        break
    sample_input_img = sample_input_img.unsqueeze(0)

    # sample_input_img = torch.randn(1, 3, 224, 224).to("cpu")
    # FBRANCASI: If the model doesn't pass the validations in exportBrevitas, but
    # you want still to validate, remove the "raise RuntimeError" in exportBrevitas
    TQ_model = exportBrevitas(FQ_model, sample_input_img, debug=True)

    num_parameters = sum(p.numel() for p in TQ_model.parameters())
    print(f"Number of parameters: {num_parameters:,}")

    print("Evaluating TQ model...")
    TQ_top1, TQ_top5 = evaluate_model(TQ_model, val_loader, device, "TQ ResNet18")

    print("\nComparison Summary:")
    print(f"{'Model':<25} {'Top-1 Accuracy':<25} {'Top-5 Accuracy':<25}")
    print("-" * 75)
    print(f"{'Original ResNet18':<25} {original_top1:<24.2f} {original_top5:<24.2f}")
    print(f"{'FQ ResNet18':<25} {FQ_top1:<24.2f} {FQ_top5:<24.2f}")
    print(f"{'TQ ResNet18':<25} {TQ_top1:<24.2f} {TQ_top5:<24.2f}")
    print(
        f"{'FQ Drop':<25} {original_top1 - FQ_top1:<24.2f} {original_top5 - FQ_top5:<24.2f}"
    )
    print(
        f"{'TQ Drop':<25} {original_top1 - TQ_top1:<24.2f} {original_top5 - TQ_top5:<24.2f}"
    )

    # --------------------------------------------------
    # PART 2: FQ VS TQ COMPARISON
    # --------------------------------------------------

    sample_input_img = None
    sample_target = None
    for inputs, targets in val_loader:
        sample_input_img = inputs[17]
        sample_target = targets[17].item()
        break

    print(f"\nGround truth class of the sample image: {sample_target}")
    compare_model_outputs(FQ_model, TQ_model, sample_input_img, device)


if __name__ == "__main__":
    main()
