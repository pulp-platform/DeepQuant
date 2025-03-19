# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

"""
FX Graph tracer that traces each node by wrapping submodules with proxy objects.
"""

import torch
import torch.nn as nn
import torch.fx as fx
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, Callable
import functools
import inspect


class NodeTracer:
    """
    Traces execution through an FX graph by wrapping each module with a
    proxy that logs input and output values.
    """

    def __init__(self, debug: bool = True) -> None:
        """
        Initialize the tracer.

        Args:
            debug: Whether to print debug information.
        """
        self.debug = debug
        self.BLUE = "\033[94m"
        self.GREEN = "\033[92m"
        self.YELLOW = "\033[93m"
        self.RED = "\033[91m"
        self.RESET = "\033[0m"
        self.traced_modules: Dict[str, nn.Module] = {}
        self.call_count: Dict[str, int] = {}

    def trace(
        self, model: fx.GraphModule, example_input: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Trace the execution of the model by wrapping modules with proxies.

        Args:
            model: The FX GraphModule to trace.
            example_input: The input tensor.

        Returns:
            The model output, if successful.
        """
        if self.debug:
            print(
                f"\n{self.BLUE}===== Starting FX Graph Execution Tracing ====={self.RESET}\n"
            )
            print(
                f"{self.BLUE}Input shape: {tuple(example_input.shape)}, dtype: {example_input.dtype}{self.RESET}\n"
            )

        # Wrap all submodules with our proxy
        self._wrap_modules(model)

        # Create a copy of the original model to restore wrapped modules after tracing
        original_modules = {
            name: module
            for name, module in model.named_modules()
            if not isinstance(module, fx.GraphModule)
        }

        try:
            # Execute the model with the example input
            with torch.no_grad():
                output = model(example_input)

            if self.debug:
                print(f"\n{self.GREEN}Execution completed successfully!{self.RESET}")
                if isinstance(output, torch.Tensor):
                    print(
                        f"{self.GREEN}Output shape: {tuple(output.shape)}, dtype: {output.dtype}{self.RESET}"
                    )
                else:
                    print(f"{self.GREEN}Output type: {type(output)}{self.RESET}")

            return output

        except Exception as e:
            if self.debug:
                print(f"\n{self.RED}Error during execution: {str(e)}{self.RESET}")
            return None

        finally:
            # Restore original modules
            self._restore_modules(model, original_modules)

    def _wrap_modules(self, model: fx.GraphModule) -> None:
        """
        Wrap all relevant modules with tracing proxies.

        Args:
            model: The model containing modules to wrap.
        """
        # Find relevant modules that match nodes in the graph
        for name, module in list(model.named_modules()):
            if not isinstance(module, fx.GraphModule):
                if hasattr(module, "forward"):
                    original_forward = module.forward
                    self.traced_modules[name] = original_forward

                    # Create wrapped forward method with tracing
                    @functools.wraps(original_forward)
                    def traced_forward(self, *args, **kwargs):
                        module_name = self._tracing_name

                        # Increment call count
                        self._tracer.call_count.setdefault(module_name, 0)
                        self._tracer.call_count[module_name] += 1
                        call_idx = self._tracer.call_count[module_name]

                        # Print module info before call
                        if self._tracer.debug:
                            module_type = type(self).__name__
                            print(
                                f"\n{self._tracer.YELLOW}[{module_name} ({module_type}) - Call #{call_idx}]{self._tracer.RESET}"
                            )

                            # Print input tensor info
                            for i, arg in enumerate(args):
                                if isinstance(arg, torch.Tensor):
                                    print(
                                        f"  Input {i}: Tensor{tuple(arg.shape)} ({arg.dtype})"
                                    )
                                    # Sample values for extra context
                                    if arg.numel() > 0:
                                        flat = arg.reshape(-1)
                                        sample = flat[:3].tolist()
                                        sample_str = ", ".join(
                                            (
                                                f"{x:.6f}"
                                                if isinstance(x, float)
                                                else str(x)
                                            )
                                            for x in sample
                                        )
                                        print(
                                            f"    Values: [{sample_str}{'...' if flat.numel() > 3 else ''}]"
                                        )
                                elif (
                                    isinstance(arg, (list, tuple))
                                    and len(arg) > 0
                                    and isinstance(arg[0], torch.Tensor)
                                ):
                                    print(
                                        f"  Input {i}: {type(arg).__name__} of {len(arg)} Tensors"
                                    )
                                else:
                                    print(f"  Input {i}: {type(arg).__name__}")

                        # Call original forward method
                        result = self._original_forward(*args, **kwargs)

                        # Print output info
                        if self._tracer.debug:
                            if isinstance(result, torch.Tensor):
                                print(
                                    f"  {self._tracer.GREEN}Output: Tensor{tuple(result.shape)} ({result.dtype}){self._tracer.RESET}"
                                )
                                # Sample output values
                                if result.numel() > 0:
                                    flat = result.reshape(-1)
                                    sample = flat[:3].tolist()
                                    sample_str = ", ".join(
                                        f"{x:.6f}" if isinstance(x, float) else str(x)
                                        for x in sample
                                    )
                                    print(
                                        f"    Values: [{sample_str}{'...' if flat.numel() > 3 else ''}]"
                                    )
                            elif isinstance(result, (list, tuple)) and len(result) > 0:
                                print(
                                    f"  {self._tracer.GREEN}Output: {type(result).__name__} of length {len(result)}{self._tracer.RESET}"
                                )
                            else:
                                print(
                                    f"  {self._tracer.GREEN}Output: {type(result).__name__}{self._tracer.RESET}"
                                )

                        return result

                    # Attach tracer reference and original forward to the wrapped method
                    traced_forward.__self__ = module
                    traced_forward.__self__._tracer = self
                    traced_forward.__self__._original_forward = original_forward
                    traced_forward.__self__._tracing_name = name

                    # Replace forward with wrapped version
                    module.forward = traced_forward.__get__(module)

    def _restore_modules(
        self, model: fx.GraphModule, original_modules: Dict[str, nn.Module]
    ) -> None:
        """
        Restore original forward methods for all wrapped modules.

        Args:
            model: The model containing wrapped modules.
            original_modules: Dictionary of original modules.
        """
        for name, original_forward in self.traced_modules.items():
            parts = name.split(".")
            current = model

            # Navigate to the module
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    break

            # Restore original forward if found
            if hasattr(current, "forward") and hasattr(current, "_original_forward"):
                current.forward = original_forward
