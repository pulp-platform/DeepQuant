# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0


import pytest

import torch
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int8WeightPerChannelFloat,
    Int32Bias,
)

from DeepQuant.ExportBrevitas import exportBrevitas
from DeepQuant.QuantManipulation.QuantDequantNodes import Dequant, Quant


def assertIntegerLayers(fxModel: nn.Module, count: int) -> None:
    layers = [
        n
        for n in fxModel.graph.nodes
        if n.op == "call_module" and "wrappedInnerForwardImpl" in n.target
    ]
    assert len(layers) == count
    for node in layers:
        for arg in node.args[:2]:
            assert isinstance(
                fxModel.get_submodule(arg.target), Quant
            ), f"{node.target} reads {arg.name}"
        users = list(node.users)
        assert len(users) == 1 and isinstance(
            fxModel.get_submodule(users[0].target), Dequant
        ), node.target


def export(model: nn.Module, sampleInput: torch.Tensor) -> nn.Module:
    torch.manual_seed(42)
    model.eval()
    return exportBrevitas(model, sampleInput, debug=True)


class UpstreamInputQuantNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputQuant = qnn.QuantIdentity(return_quant_tensor=True)
        self.linear = qnn.QuantLinear(
            16,
            32,
            bias=True,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.inputQuant(x))


class FloatBiasNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputQuant = qnn.QuantIdentity(return_quant_tensor=True)
        self.linear1 = qnn.QuantLinear(16, 32, bias=True, return_quant_tensor=True)
        self.relu = qnn.QuantReLU(return_quant_tensor=True)
        self.linear2 = qnn.QuantLinear(32, 10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(self.inputQuant(x))))


class NoBiasNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputQuant = qnn.QuantIdentity(return_quant_tensor=True)
        self.linear = qnn.QuantLinear(
            16,
            32,
            bias=False,
            output_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.inputQuant(x))


class ValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantA = qnn.QuantIdentity(return_quant_tensor=True)
        self.quantB = qnn.QuantIdentity(return_quant_tensor=True)
        self.linear = qnn.QuantLinear(
            16, 16, bias=True, bias_quant=Int32Bias, return_quant_tensor=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.quantA(x)
        b = self.quantB(self.linear(a).value + a.value)
        return b.value


class SimilarNamesNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputQuant = qnn.QuantIdentity(return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(
            16,
            32,
            bias=True,
            bias_quant=Int32Bias,
            input_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.fc1_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(
            32,
            8,
            bias=True,
            bias_quant=Int32Bias,
            input_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1_quant(self.fc1(self.inputQuant(x))))


class SignedReLUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputQuant = qnn.QuantIdentity(return_quant_tensor=True)
        self.linear1 = qnn.QuantLinear(
            16,
            32,
            bias=True,
            bias_quant=Int32Bias,
            input_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.relu = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )
        self.linear2 = qnn.QuantLinear(
            32,
            8,
            bias=True,
            bias_quant=Int32Bias,
            input_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(self.inputQuant(x))))


class InputReshapeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputQuant = qnn.QuantIdentity(return_quant_tensor=True)
        self.linear = qnn.QuantLinear(
            64,
            10,
            bias=True,
            bias_quant=Int32Bias,
            input_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.linear(self.inputQuant(x))


class UpstreamQuantCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputQuant = qnn.QuantIdentity(return_quant_tensor=True)
        self.conv = qnn.QuantConv2d(
            1, 8, 3, padding=1, bias=True, return_quant_tensor=True
        )
        self.relu = qnn.QuantReLU(return_quant_tensor=True)
        self.pool = nn.MaxPool2d(2)
        self.pool_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.pre_linear_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.classifier = qnn.QuantLinear(8 * 4 * 4, 10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool_quant(self.pool(self.relu(self.conv(self.inputQuant(x)))))
        x = self.pre_linear_quant(x.view(x.size(0), -1))
        return self.classifier(x)


class PerChannelCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputQuant = qnn.QuantIdentity(return_quant_tensor=True)
        self.conv = qnn.QuantConv2d(
            1,
            8,
            3,
            padding=1,
            bias=True,
            weight_quant=Int8WeightPerChannelFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=True,
        )
        self.relu = qnn.QuantReLU(return_quant_tensor=True)
        self.classifier = qnn.QuantLinear(
            8 * 8 * 8,
            10,
            bias=True,
            weight_quant=Int8WeightPerChannelFloat,
            input_quant=Int8ActPerTensorFloat,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(self.inputQuant(x)))
        return self.classifier(x.view(x.size(0), -1))


@pytest.mark.SingleLayerTests
def deepQuantTestUpstreamInputQuant() -> None:
    assertIntegerLayers(export(UpstreamInputQuantNet(), torch.randn(1, 4, 16)), 1)


@pytest.mark.SingleLayerTests
def deepQuantTestFloatBias() -> None:
    assertIntegerLayers(export(FloatBiasNet(), torch.randn(1, 4, 16)), 2)


@pytest.mark.SingleLayerTests
def deepQuantTestNoBias() -> None:
    assertIntegerLayers(export(NoBiasNet(), torch.randn(1, 4, 16)), 1)


@pytest.mark.SingleLayerTests
def deepQuantTestQuantTensorValue() -> None:
    assertIntegerLayers(export(ValueNet(), torch.randn(1, 4, 16)), 1)


@pytest.mark.SingleLayerTests
def deepQuantTestSimilarModuleNames() -> None:
    assertIntegerLayers(export(SimilarNamesNet(), torch.randn(1, 4, 16)), 2)


@pytest.mark.SingleLayerTests
def deepQuantTestSignedQuantReLU() -> None:
    assertIntegerLayers(export(SignedReLUNet(), torch.randn(1, 4, 16)), 2)


@pytest.mark.SingleLayerTests
def deepQuantTestInputReshape() -> None:
    assertIntegerLayers(export(InputReshapeNet(), torch.randn(1, 1, 8, 8)), 1)


@pytest.mark.ModelTests
def deepQuantTestUpstreamQuantCNN() -> None:
    assertIntegerLayers(export(UpstreamQuantCNN(), torch.randn(1, 1, 8, 8)), 2)


@pytest.mark.SingleLayerTests
def deepQuantTestPerChannelWeights() -> None:
    assertIntegerLayers(export(PerChannelCNN(), torch.randn(1, 1, 8, 8)), 2)
