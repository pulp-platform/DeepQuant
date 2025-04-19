# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>


class ConsoleColor:
    blue = "\033[94m"
    green = "\033[92m"
    red = "\033[91m"
    yellow = "\033[93m"
    reset = "\033[0m"

    @staticmethod
    def wrap(text: str, color: str) -> str:
        return f"{color}{text}{ConsoleColor.reset}"
