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


def safe_get_scale(quant_obj: Any) -> Any:
    if quant_obj is None:
        return None
    maybe_scale = quant_obj.scale() if callable(quant_obj.scale) else quant_obj.scale
    if maybe_scale is None:
        return None
    if isinstance(maybe_scale, torch.Tensor):
        return maybe_scale.item()
    elif isinstance(maybe_scale, float):
        return maybe_scale
    try:
        return float(maybe_scale)
    except Exception:
        return None


def safe_get_zero_point(quant_obj: Any) -> Any:
    if quant_obj is None:
        return None
    maybe_zp = (
        quant_obj.zero_point()
        if callable(quant_obj.zero_point)
        else quant_obj.zero_point
    )
    if maybe_zp is None:
        return None
    if isinstance(maybe_zp, torch.Tensor):
        return maybe_zp.item()
    elif isinstance(maybe_zp, float):
        return maybe_zp
    try:
        return float(maybe_zp)
    except Exception:
        return None


def safe_get_is_signed(quant_obj: Any) -> bool:
    if hasattr(quant_obj, "is_signed"):
        return getattr(quant_obj, "is_signed")
    if hasattr(quant_obj, "min_val"):
        try:
            return quant_obj.min_val < 0
        except Exception:
            pass
    zp = safe_get_zero_point(quant_obj)
    if zp is not None:
        # If zero_point is near zero, assume unsigned quantization.
        return not (abs(zp) < 1e-5)
    return True


def extract_brevitas_proxy_params(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Recursively scan the model to extract the scale, zero_point, bit_width, and deduced signedness.
    """
    params_dict: Dict[str, Dict[str, Any]] = {}

    def recurse_modules(parent_mod: nn.Module, prefix: str = "") -> None:
        for child_name, child_mod in parent_mod.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(
                child_mod,
                (
                    ActQuantProxyFromInjector,
                    WeightQuantProxyFromInjector,
                    BiasQuantProxyFromInjector,
                ),
            ):
                scl = safe_get_scale(child_mod)
                zp = safe_get_zero_point(child_mod)
                bw = child_mod.bit_width()
                is_signed = safe_get_is_signed(child_mod)
                params_dict[full_name] = {
                    "scale": scl,
                    "zero_point": zp,
                    "bit_width": bw,
                    "is_signed": is_signed,
                }
            recurse_modules(child_mod, prefix=full_name)

    recurse_modules(model)
    return params_dict


def print_quant_params(params_dict: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n{Fore.BLUE}Extracted Parameters from the Network:{Style.RESET_ALL}")
    for layer_name, quant_values in params_dict.items():
        print(f"  {Fore.BLUE}{layer_name}:{Style.RESET_ALL}")
        for param_key, param_val in quant_values.items():
            print(f"    {param_key}: {param_val}")
        print()
