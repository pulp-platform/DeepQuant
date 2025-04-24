# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import tarfile
from pathlib import Path
import pytest
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
import urllib
from DeepQuant import exportQuantModel


def evaluate_model(model, data_loader, eval_device, name="Model"):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc=f"Evaluating {name}"):
            is_TQ = "TQ" in name

            if is_TQ:
                # FBRANCASI: Process different batches for the TQ model
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


@pytest.mark.ModelTests
def deepQuantTestResnet18() -> None:
    HOME = Path.home()
    BASE = HOME / "Documents" / "ImagenetV2"
    TAR_URL = (
        "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/"
        "imagenetv2-matched-frequency.tar.gz"
    )
    TAR_PATH = BASE / "imagenetv2-matched-frequency.tar.gz"
    EXTRACT_DIR = BASE / "imagenetv2-matched-frequency-format-val"

    if not TAR_PATH.exists():
        BASE.mkdir(parents=True, exist_ok=True)
        print(f"Scarico ImageNetV2 da {TAR_URL}...")
        urllib.request.urlretrieve(TAR_URL, TAR_PATH)

    if not EXTRACT_DIR.exists():
        print(f"Estrazione in corso in {EXTRACT_DIR}...")
        with tarfile.open(TAR_PATH, "r:*") as tar:
            for member in tqdm(tar.getmembers(), desc="Extracting files"):
                tar.extract(member, BASE)
        print("Estrazione completata.")

    transforms_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageFolder(root=str(EXTRACT_DIR), transform=transforms_val)

    dataset.classes = sorted(dataset.classes, key=lambda x: int(x))

    dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}

    new_samples = []
    for path, _ in dataset.samples:
        cls_name = Path(path).parent.name
        new_label = dataset.class_to_idx[cls_name]
        new_samples.append((path, new_label))
    dataset.samples = new_samples
    dataset.targets = [s[1] for s in new_samples]

    # FBRANCASI: Optional, reduce number of example for faster validation
    DATASET_LIMIT = 256
    dataset = Subset(dataset, list(range(DATASET_LIMIT)))
    print(f"Validation dataset size set to {len(dataset)} images.")

    calib_loader = DataLoader(
        Subset(dataset, list(range(256))), batch_size=32, shuffle=False, pin_memory=True
    )
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)

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

    print("Evaluating original model...")
    original_top1, original_top5 = evaluate_model(
        original_model, val_loader, device, "Original ResNet18"
    )

    print("Preparing and quantizing ResNet18...")
    FQ_model = prepare_FQ_resnet18()

    print("Calibrating FQ model...")
    calibrate_model(FQ_model, calib_loader)

    print("Evaluating FQ model...")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # FBRANCASI: I'm on mac, mps doesn't work with brevitas
    FQ_top1, FQ_top5 = evaluate_model(FQ_model, val_loader, device, "FQ ResNet18")

    sample_input_img = torch.randn(1, 3, 224, 224).to("cpu")
    # FBRANCASI: If the model doesn't pass the validations in exportQuantModel, but
    # you want still to validate, remove the "raise RuntimeError" in exportQuantModel
    TQ_model = exportQuantModel(FQ_model, sample_input_img, debug=True)

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

    if abs(FQ_top1 - TQ_top1) > 5.0 or abs(FQ_top5 - TQ_top5) > 5.0:
        raise RuntimeError(
            "✗ Modification of Dequant Nodes changed the output significantly. "
            f"Top-1 difference: {abs(FQ_top1 - TQ_top1):.2f}%, "
            f"Top-5 difference: {abs(FQ_top5 - TQ_top5):.2f}%"
        )
