# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from typing import Any, Dict
import torch
import torch.nn as nn
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.proxy.parameter_quant import (
    WeightQuantProxyFromInjector,
    BiasQuantProxyFromInjector,
)
from colorama import Fore, Style


def safeGetScale(quantObj: Any) -> Any:
    """Safely extract scale parameter from quantization object."""
    if quantObj is None:
        return None
    maybeScale = quantObj.scale() if callable(quantObj.scale) else quantObj.scale
    if maybeScale is None:
        return None
    if isinstance(maybeScale, torch.Tensor):
        return maybeScale.item()
    elif isinstance(maybeScale, float):
        return maybeScale
    try:
        return float(maybeScale)
    except Exception:
        return None


def safeGetZeroPoint(quantObj: Any) -> Any:
    """Safely extract zero point parameter from quantization object."""
    if quantObj is None:
        return None
    maybeZp = (
        quantObj.zero_point()
        if callable(quantObj.zero_point)
        else quantObj.zero_point
    )
    if maybeZp is None:
        return None
    if isinstance(maybeZp, torch.Tensor):
        return maybeZp.item()
    elif isinstance(maybeZp, float):
        return maybeZp
    try:
        return float(maybeZp)
    except Exception:
        return None


def safeGetIsSigned(quantObj: Any) -> bool:
    """Safely determine if quantization is signed."""
    if hasattr(quantObj, "is_signed"):
        return getattr(quantObj, "is_signed")
    if hasattr(quantObj, "min_val"):
        try:
            return quantObj.min_val < 0
        except Exception:
            pass
    zp = safeGetZeroPoint(quantObj)
    if zp is not None:
        # If zero_point is near zero, assume unsigned quantization.
        return not (abs(zp) < 1e-5)
    return True


def extractBrevitasProxyParams(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Extract quantization parameters from Brevitas proxy modules."""
    paramsDict: Dict[str, Dict[str, Any]] = {}

    def recurseModules(parentMod: nn.Module, prefix: str = "") -> None:
        for childName, childMod in parentMod.named_children():
            fullName = f"{prefix}.{childName}" if prefix else childName
            if isinstance(
                childMod,
                (
                    ActQuantProxyFromInjector,
                    WeightQuantProxyFromInjector,
                    BiasQuantProxyFromInjector,
                ),
            ):
                scl = safeGetScale(childMod)
                zp = safeGetZeroPoint(childMod)
                bw = childMod.bit_width()
                isSigned = safeGetIsSigned(childMod)
                paramsDict[fullName] = {
                    "scale": scl,
                    "zero_point": zp,
                    "bit_width": bw,
                    "is_signed": isSigned,
                }
            recurseModules(childMod, prefix=fullName)

    recurseModules(model)
    return paramsDict


def printQuantParams(paramsDict: Dict[str, Dict[str, Any]]) -> None:
    """Print extracted quantization parameters in a readable format."""
    print(f"\n{Fore.BLUE}Extracted Parameters from the Network:{Style.RESET_ALL}")
    for layerName, quantValues in paramsDict.items():
        print(f"  {Fore.BLUE}{layerName}:{Style.RESET_ALL}")
        for paramKey, paramVal in quantValues.items():
            print(f"    {paramKey}: {paramVal}")
        print()