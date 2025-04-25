# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>
class ConsoleColor:
    """Console color utilities for formatted terminal output."""

    # Color codes
    blue = "\033[94m"
    green = "\033[92m"
    red = "\033[91m"
    yellow = "\033[93m"
    cyan = "\033[96m"
    magenta = "\033[95m"
    bold = "\033[1m"
    reset = "\033[0m"

    # Symbols
    checkmark = " ✓"
    cross = " ✗"
    arrow = " ›"

    @staticmethod
    def wrap(text: str, color: str) -> str:
        """Wrap text with color codes."""
        return f"{color}{text}{ConsoleColor.reset}"

    @staticmethod
    def success(text: str) -> str:
        """Format a success message."""
        return ConsoleColor.wrap(f"{ConsoleColor.checkmark} {text}", ConsoleColor.green)

    @staticmethod
    def error(text: str) -> str:
        """Format an error message."""
        return ConsoleColor.wrap(f"{ConsoleColor.cross} {text}", ConsoleColor.red)

    @staticmethod
    def info(text: str) -> str:
        """Format an informational message."""
        return ConsoleColor.wrap(f"{ConsoleColor.arrow} {text}", ConsoleColor.blue)

    @staticmethod
    def warning(text: str) -> str:
        """Format a warning message."""
        return ConsoleColor.wrap(text, ConsoleColor.yellow)

    @staticmethod
    def header(text: str) -> str:
        """Format a step header with separator lines."""
        separator = "=" * 50
        header_text = f"{separator}\n{text}\n{separator}"
        return f"\n{ConsoleColor.wrap(header_text, ConsoleColor.magenta)}"
