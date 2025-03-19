# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
Pytest configuration file that suppresses specific warnings, including those
related to torch.tensor constant registration in FX tracing.
"""

import warnings
import pytest

# Attempt to import TracerWarning from torch.fx.proxy;
# if unavailable, skip filtering by category.
try:
    from torch.fx.proxy import TracerWarning

    warnings.filterwarnings("ignore", category=TracerWarning)
except ImportError:
    pass

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="Named tensors.*")
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*__torch_function__.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Was not able to add assertion.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="'has_cuda' is deprecated.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="'has_cudnn' is deprecated.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="'has_mps' is deprecated.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="'has_mkldnn' is deprecated.*"
)
