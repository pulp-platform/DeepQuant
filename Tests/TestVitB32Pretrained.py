# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

import tarfile
import urllib.request
from pathlib import Path

import brevitas.nn as qnn
import pytest
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.per_input import AdaptiveAvgPoolToAvgPool
from brevitas.graph.quantize import preprocess_for_quantize, quantize
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from DeepQuant import brevitasToTrueQuant


def evaluateModel(model, dataLoader, evalDevice, name="Model"):
    model.eval()
    correctTop1 = 0
    correctTop5 = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataLoader, desc=f"Evaluating {name}"):
            isTQ = "TQ" in name

            if isTQ:
                # FBRANCASI: Process different batches for the TQ model
                for i in range(inputs.size(0)):
                    singleInput = inputs[i : i + 1].to(evalDevice)
                    singleOutput = model(singleInput)
                    if isinstance(singleOutput, tuple):
                        singleOutput = singleOutput[0]

                    _, predicted = singleOutput.max(1)
                    if predicted.item() == targets[i].item():
                        correctTop1 += 1

                    _, top5Pred = singleOutput.topk(5, dim=1, largest=True, sorted=True)
                    if targets[i].item() in top5Pred[0].cpu().numpy():
                        correctTop5 += 1

                    total += 1
            else:
                inputs = inputs.to(evalDevice)
                targets = targets.to(evalDevice)
                output = model(inputs)
                if isinstance(output, tuple):
                    output = output[0]

                _, predicted = output.max(1)
                correctTop1 += (predicted == targets).sum().item()

                _, top5Pred = output.topk(5, dim=1, largest=True, sorted=True)
                for i in range(targets.size(0)):
                    if targets[i] in top5Pred[i]:
                        correctTop5 += 1

                total += targets.size(0)

    top1Accuracy = 100.0 * correctTop1 / total
    top5Accuracy = 100.0 * correctTop5 / total

    print(
        f"{name} - Top-1 Accuracy: {top1Accuracy:.2f}% ({correctTop1}/{total}), "
        f"Top-5 Accuracy: {top5Accuracy:.2f}%"
    )

    return top1Accuracy, top5Accuracy


def calibrateModel(model, calibLoader):
    model.eval()
    with torch.no_grad(), calibration_mode(model):
        for inputs, _ in tqdm(calibLoader, desc="Calibrating model"):
            inputs = inputs.to("cpu")
            output = model(inputs)
            if isinstance(output, tuple):
                output = output[0]
    print("Calibration completed.")


def prepareFQVitB32():
    """Prepare a fake-quantized (FQ) ViT-B/32 model."""
    baseModel = torchvision.models.vit_b_32(
        weights=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1
    )
    baseModel = baseModel.eval().to("cpu")

    computeLayerMap = {
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
        nn.MultiheadAttention: (
            qnn.QuantMultiheadAttention,
            {
                "in_proj_input_quant": Int8ActPerTensorFloat,
                "in_proj_weight_quant": Int8WeightPerTensorFloat,
                "in_proj_bias_quant": Int32Bias,
                "attn_output_weights_quant": Uint8ActPerTensorFloat,
                "q_scaled_quant": Int8ActPerTensorFloat,
                "k_transposed_quant": Int8ActPerTensorFloat,
                "v_quant": Int8ActPerTensorFloat,
                "out_proj_input_quant": Int8ActPerTensorFloat,
                "out_proj_weight_quant": Int8WeightPerTensorFloat,
                "out_proj_bias_quant": Int32Bias,
                "out_proj_output_quant": Int8ActPerTensorFloat,
                "return_quant_tensor": True,
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

    quantActMap = {}

    quantIdentityMap = {
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

    dummyInput = torch.ones(1, 3, 224, 224).to("cpu")

    print("Preprocessing model for quantization...")
    baseModel = preprocess_for_quantize(
        baseModel, equalize_iters=20, equalize_scale_computation="range"
    )

    print("Converting AdaptiveAvgPool to AvgPool...")
    baseModel = AdaptiveAvgPoolToAvgPool().apply(baseModel, dummyInput)

    print("Quantizing model...")
    FQModel = quantize(
        graph_model=baseModel,
        compute_layer_map=computeLayerMap,
        quant_act_map=quantActMap,
        quant_identity_map=quantIdentityMap,
    )

    return FQModel


@pytest.mark.ModelTests
def deepQuantTestVitB32Pretrained() -> None:
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
        print(f"Downloading ImageNetV2 from {TAR_URL}...")
        urllib.request.urlretrieve(TAR_URL, TAR_PATH)

    if not EXTRACT_DIR.exists():
        print(f"Extracting to {EXTRACT_DIR}...")
        with tarfile.open(TAR_PATH, "r:*") as tar:
            for member in tqdm(tar.getmembers(), desc="Extracting files"):
                tar.extract(member, BASE)
        print("Extraction completed.")

    transformsVal = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageFolder(root=str(EXTRACT_DIR), transform=transformsVal)
    dataset.classes = sorted(dataset.classes, key=lambda x: int(x))
    dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}

    newSamples = []
    for path, _ in dataset.samples:
        clsName = Path(path).parent.name
        newLabel = dataset.class_to_idx[clsName]
        newSamples.append((path, newLabel))
    dataset.samples = newSamples
    dataset.targets = [s[1] for s in newSamples]

    # FBRANCASI: Optional, reduce number of example for faster validation
    DATASET_LIMIT = 256
    dataset = Subset(dataset, list(range(DATASET_LIMIT)))
    print(f"Validation dataset size set to {len(dataset)} images.")

    calibLoader = DataLoader(
        Subset(dataset, list(range(256))), batch_size=32, shuffle=False, pin_memory=True
    )
    valLoader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)

    # FBRANCASI: I'm on mac, so mps for me
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    print(f"Using device: {device}")

    originalModel = torchvision.models.vit_b_32(
        weights=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1
    )
    originalModel = originalModel.eval().to(device)
    print("Original ViT-B/32 loaded.")

    print("Evaluating original model...")
    originalTop1, originalTop5 = evaluateModel(
        originalModel, valLoader, device, "Original ViT-B/32"
    )

    print("Preparing and quantizing ViT-B/32...")
    FQModel = prepareFQVitB32()

    print("Calibrating FQ model...")
    calibrateModel(FQModel, calibLoader)

    print("Evaluating FQ model...")
    # FBRANCASI: I'm on mac, mps doesn't work with brevitas
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FQTop1, FQTop5 = evaluateModel(FQModel, valLoader, device, "FQ ViT-B/32")

    sampleInputImg = torch.randn(1, 3, 224, 224).to("cpu")
    TQModel = brevitasToTrueQuant(FQModel, sampleInputImg, debug=True)

    numParameters = sum(p.numel() for p in TQModel.parameters())
    print(f"Number of parameters: {numParameters:,}")

    print("Evaluating TQ model...")
    TQTop1, TQTop5 = evaluateModel(TQModel, valLoader, device, "TQ ViT-B/32")

    print("\nComparison Summary:")
    print(f"{'Model':<25} {'Top-1 Accuracy':<25} {'Top-5 Accuracy':<25}")
    print("-" * 75)
    print(f"{'Original ViT-B/32':<25} {originalTop1:<24.2f} {originalTop5:<24.2f}")
    print(f"{'FQ ViT-B/32':<25} {FQTop1:<24.2f} {FQTop5:<24.2f}")
    print(f"{'TQ ViT-B/32':<25} {TQTop1:<24.2f} {TQTop5:<24.2f}")
    print(
        f"{'FQ Drop':<25} {originalTop1 - FQTop1:<24.2f} {originalTop5 - FQTop5:<24.2f}"
    )
    print(
        f"{'TQ Drop':<25} {originalTop1 - TQTop1:<24.2f} {originalTop5 - TQTop5:<24.2f}"
    )

    if abs(FQTop1 - TQTop1) > 5.0 or abs(FQTop5 - TQTop5) > 5.0:
        print(
            f"Warning: Large accuracy drop between FQ and TQ models. "
            f"Top-1 difference: {abs(FQTop1 - TQTop1):.2f}%, "
            f"Top-5 difference: {abs(FQTop5 - TQTop5):.2f}%"
        )
