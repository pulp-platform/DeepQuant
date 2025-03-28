# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>


import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_cuda.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_cudnn.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_mps.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_mkldnn.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*experimental feature.*"
)
warnings.filterwarnings("ignore", category=UserWarning, message=".*deprecated.*")

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import brevitas.nn as qnn
from brevitas.graph.quantize import preprocess_for_quantize, quantize
from brevitas.graph.calibrate import calibration_mode
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)

from DeepQuant.ExportBrevitas import exportBrevitas


class SimpleFCNN(nn.Module):
    """Simple fully connected model for MNIST classification."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def trainModel(
    model: nn.Module,
    trainLoader: DataLoader,
    testLoader: DataLoader,
    savePath: Path,
    epochs: int = 10,
    learningRate: float = 0.001,
) -> nn.Module:
    """Train the model if no saved weights exist."""

    if savePath.exists():
        print(f"Loading existing model from {savePath}")
        model.load_state_dict(torch.load(savePath))
        return model

    print("No saved model found. Starting training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        model.train()
        runningLoss = 0.0
        for images, labels in trainLoader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            runningLoss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {runningLoss/len(trainLoader):.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testLoader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {100 * correct / total:.2f}%")

    # Save model
    torch.save(model.state_dict(), savePath)
    print(f"Model saved to {savePath}")

    return model


def calibrate_model(
    model: nn.Module, calib_loader: DataLoader, device: torch.device
) -> None:
    """Calibrate the quantized model."""
    model.eval()
    model.to(device)
    with (
        torch.no_grad(),
        calibration_mode(model),
        tqdm(calib_loader, desc="Calibrating") as pbar,
    ):
        for images, _ in pbar:
            images = images.to(device)
            images = images.to(torch.float)
            model(images)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPORT_FOLDER = Path().cwd() / "Tests"
MODEL_PATH = EXPORT_FOLDER / "Models"
DATA_PATH = EXPORT_FOLDER / "Data"

def deepQuantTestSimpleFCNN() -> None:
    
    EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Data loading
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=DATA_PATH, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=DATA_PATH, train=False, download=True, transform=transform
    )

    trainLoader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testLoader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, pin_memory=True
    )

    # Train or load model
    m = SimpleFCNN()
    model = trainModel(m, trainLoader, testLoader, MODEL_PATH / "mnist_model.pth")

    # Prepare for quantization
    model = preprocess_for_quantize(model)

    # Quantization configurations
    computeLayerMap = {
        nn.Linear: (
            qnn.QuantLinear,
            {
                "input_quant": Int8ActPerTensorFloat,
                "weight_quant": Int8WeightPerTensorFloat,
                "output_quant": Int8ActPerTensorFloat,
                "bias_quant": Int32Bias,
                "return_quant_tensor": True,
                "output_bit_width": 8,
                "weight_bit_width": 8,
            },
        ),
    }

    quantActMap = {
        nn.ReLU: (
            qnn.QuantReLU,
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 7,
            },
        ),
    }

    quantIdentityMap = {
        "signed": (
            qnn.QuantIdentity,
            {
                "act_quant": Int8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 7,
            },
        ),
        "unsigned": (
            qnn.QuantIdentity,
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 7,
            },
        ),
    }

    # Quantize and calibrate
    modelQuant = quantize(
        model,
        compute_layer_map=computeLayerMap,
        quant_act_map=quantActMap,
        quant_identity_map=quantIdentityMap,
    )

    calibrate_model(modelQuant, testLoader, DEVICE)

    # Export and transform
    sampleInput, _ = next(iter(testLoader))
    sampleInput = sampleInput[0:1]
    print(f"Sample input shape: {sampleInput.shape}")

    exportBrevitas(modelQuant, sampleInput.to(DEVICE), debug=True)
