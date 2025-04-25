# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

# FBRANCASI: Workaround for PyTorch/FX API change: ensure private alias exists
import torch.fx.node as _fx_node

if not hasattr(_fx_node.Node, "_Node__update_args_kwargs"):
    _fx_node.Node._Node__update_args_kwargs = _fx_node.Node._update_args_kwargs

from DeepQuant.Export import brevitasToTrueQuant

__all__ = ["brevitasToTrueQuant"]
