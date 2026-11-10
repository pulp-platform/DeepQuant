# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Lionnus Kesting <lkesting@iis.ee.ethz.ch>

"""
The output stage of an unrolled forward, as a module.

`QuantLayerMixin.pack_output` tests the runtime type of its argument, and FX
records the branch a Proxy takes. A module is one FX node and unpacks at run
time.
"""

import torch.nn as nn
from brevitas.quant_tensor.base_quant_tensor import _unpack_quant_tensor
from torch import Tensor


class UnpackQuantTensor(nn.Module):
    """Return the value a QuantTensor carries, or the tensor itself."""

    def forward(self, x) -> Tensor:
        return _unpack_quant_tensor(x)


def outputPackFor(module: nn.Module) -> nn.Module:
    """The output stage of an unrolled layer, chosen from return_quant_tensor."""
    if getattr(module, "return_quant_tensor", False):
        return nn.Identity()
    return UnpackQuantTensor()
