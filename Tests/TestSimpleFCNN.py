# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from pathlib import Path

import brevitas.nn as qnn
import torch
import torch.nn as nn
import torch.optim as optim
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.quantize import preprocess_for_quantize, quantize
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from DeepQuant import brevitasToTrueQuant


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

    torch.save(model.state_dict(), savePath)
    print(f"Model saved to {savePath}")

    return model


def calibrateModel(
    model: nn.Module, calibLoader: DataLoader, device: torch.device
) -> None:
    model.eval()
    model.to(device)
    with (
        torch.no_grad(),
        calibration_mode(model),
        tqdm(calibLoader, desc="Calibrating") as pbar,
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

    trainDataset = datasets.MNIST(
        root=DATA_PATH, train=True, download=True, transform=transform
    )
    testDataset = datasets.MNIST(
        root=DATA_PATH, train=False, download=True, transform=transform
    )

    trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=64, shuffle=False, pin_memory=True)

    model = SimpleFCNN()
    model = trainModel(model, trainLoader, testLoader, MODEL_PATH / "mnist_model.pth")

    model = preprocess_for_quantize(model)

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

    modelQuant = quantize(
        model,
        compute_layer_map=computeLayerMap,
        quant_act_map=quantActMap,
        quant_identity_map=quantIdentityMap,
    )

    calibrateModel(modelQuant, testLoader, DEVICE)

    sampleInput, _ = next(iter(testLoader))
    sampleInput = sampleInput[0:1]

    brevitasToTrueQuant(modelQuant, sampleInput.to(DEVICE), debug=True)
