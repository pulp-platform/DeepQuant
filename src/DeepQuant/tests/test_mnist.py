# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Complete script for MNIST model training, quantization, and transformation.
"""

import warnings

# Suppress all warnings at the very beginning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_cuda.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_cudnn.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_mps.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*has_mkldnn.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*experimental feature.*"
)
warnings.filterwarnings("ignore", category=UserWarning, message=".*deprecated.*")

import copy
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

from DeepQuant.export_brevitas import exportBrevitas


class SimpleFCModel(nn.Module):
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    save_path: Path,
    epochs: int = 10,
    learning_rate: float = 0.001,
) -> nn.Module:
    """Train the model if no saved weights exist."""

    if save_path.exists():
        print(f"Loading existing model from {save_path}")
        model.load_state_dict(torch.load(save_path))
        return model

    print("No saved model found. Starting training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {100 * correct / total:.2f}%")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

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


def test_mnist_quant_export() -> None:
    """Main execution function."""
    # Setup paths and device
    EXPORT_FOLDER = Path().cwd()
    if Path().cwd().name != "src/brevitexporter/tests/":
        EXPORT_FOLDER = EXPORT_FOLDER / "src/brevitexporter/tests/"
    EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = EXPORT_FOLDER / "mnist_fc.pth"

    # Data loading
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=EXPORT_FOLDER / "data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=EXPORT_FOLDER / "data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, pin_memory=True
    )

    # Train or load model
    model = SimpleFCModel()
    model = train_model(model, train_loader, test_loader, MODEL_PATH)

    # Prepare for quantization
    model = preprocess_for_quantize(model)

    # Quantization configurations
    compute_layer_map = {
        nn.Linear: (
            qnn.QuantLinear,
            {
                "weight_quant": Int8WeightPerTensorFloat,
                "output_quant": Int8ActPerTensorFloat,
                "bias_quant": Int32Bias,
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
                "bit_width": 7,
            },
        ),
    }

    quant_identity_map = {
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
    model_quant = quantize(
        copy.deepcopy(model),
        compute_layer_map=compute_layer_map,
        quant_act_map=quant_act_map,
        quant_identity_map=quant_identity_map,
    )

    calibrate_model(model_quant, test_loader, DEVICE)

    # Export and transform
    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input[0:1]
    print(f"Sample input shape: {sample_input.shape}")

    fx_model = exportBrevitas(model_quant, sample_input, debug=True)
