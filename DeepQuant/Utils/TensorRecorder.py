# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

from collections import OrderedDict
from typing import Dict, List, Optional, Set

import torch
import torch.fx as fx

from DeepQuant.Utils.ConsoleColor import ConsoleColor as cc


class TensorRecorder:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._current: Dict[str, torch.Tensor] = {}
        self._reference: Optional[Dict[str, torch.Tensor]] = None
        self._execution_order: List[str] = []
        self._name_map: Dict[str, str] = {}
        self._ignore: Set[str] = set()

    def clear(self) -> None:
        self.remove_hooks()
        self._current.clear()
        self._reference = None
        self._execution_order.clear()
        self._name_map.clear()
        self._ignore.clear()

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def register_forward_hooks(
        self, model: fx.GraphModule, node_types: Optional[List[str]] = None
    ) -> None:
        self.remove_hooks()
        wanted = [w.lower() for w in node_types]

        def make_hook(name: str):
            def hook(_, __, output):
                if isinstance(output, torch.Tensor):
                    self._current[name] = output.detach().clone()
                    if name not in self._execution_order:
                        self._execution_order.append(name)
                    # FBRANCASI: uncomment if you want to print logs
                    # if self.debug:
                    #     print(cc.wrap(f"{name}: {tuple(output.shape)}", cc.blue))

            return hook

        for name, module in model.named_modules():
            if name and any(w in name.lower() for w in wanted):
                self._hooks.append(module.register_forward_hook(make_hook(name)))
                # FBRANCASI: uncomment if you want to print logs
                # if self.debug:
                #     print(cc.wrap(f"hook {name}", cc.blue))

    def record_node_mapping(self, reference_name: str, current_name: str) -> None:
        self._name_map[reference_name] = current_name

    def set_reference_tensors(self) -> None:
        self._reference = {k: v.clone() for k, v in self._current.items()}
        self._reference_order = list(self._execution_order)

    def compare_tensors(self) -> Dict[str, Dict]:
        if self._reference is None:
            raise RuntimeError("set_reference_tensors has not been called")

        results: Dict[str, Dict] = OrderedDict()
        for ref_name, ref_tensor in self._reference.items():
            if ref_name in self._ignore:
                continue
            cur_name = self._name_map.get(ref_name, ref_name)
            if cur_name not in self._current:
                results[ref_name] = {"match": False, "error": f"missing '{cur_name}'"}
                continue
            cur_tensor = self._current[cur_name]
            equal = torch.equal(ref_tensor, cur_tensor)
            diff_mask = ref_tensor != cur_tensor
            results[ref_name] = {
                "match": equal,
                "mapped": cur_name != ref_name,
                "current_name": cur_name,
                "shape": tuple(ref_tensor.shape),
                "diff_count": diff_mask.sum().item() if not equal else 0,
                "diff_mask": diff_mask,
                "ref_tensor": ref_tensor,
                "cur_tensor": cur_tensor,
            }
        return results

    # FBRANCASI: helper to summarise most common absolute differences
    def _top_differences(
        self, ref: torch.Tensor, cur: torch.Tensor, diff_mask: torch.Tensor
    ) -> List[str]:
        mask_flat = diff_mask.view(-1).bool()
        if mask_flat.sum() == 0:
            return []
        abs_diff = (ref - cur).abs().view(-1)[mask_flat]
        unique, counts = torch.unique(abs_diff, return_counts=True)
        order = counts.argsort(descending=True)
        lines: List[str] = []
        for idx in order[:5]:
            delta = unique[idx].item()
            count = counts[idx].item()
            sample_index = (abs_diff == delta).nonzero(as_tuple=False)[0].item()
            global_index = mask_flat.nonzero(as_tuple=False)[sample_index].item()
            before_value = ref.view(-1)[global_index].item()
            after_value = cur.view(-1)[global_index].item()
            lines.append(
                f"    · Δ={delta:.32f}  ({count} values) e.g. idx {global_index}: {before_value:.32f} → {after_value:.32f}"
            )
        return lines

    def print_comparison_results(self, results: Dict[str, Dict]) -> None:
        if not results:
            print("No comparison data available.")
            return

        matches = sum(1 for r in results.values() if r["match"])
        total = len(results)
        print(cc.wrap("===== Tensor comparison =====", cc.blue))
        print(
            f"Compared {total}: "
            f"{cc.wrap(str(matches) + ' equal', cc.green)}, "
            f"{cc.wrap(str(total - matches) + ' different', cc.red)}\n"
        )

        ordered_names = getattr(self, "_reference_order", list(results.keys()))
        for name in ordered_names:
            if name not in results:
                continue
            res = results[name]
            status_color = cc.green if res["match"] else cc.red
            status_tag = cc.wrap("[OK]" if res["match"] else "[DIFF]", status_color)
            mapped_note = f" → {res['current_name']}" if res["mapped"] else ""
            print(f"  {status_tag} {name}{mapped_note} | shape {res['shape']}")
            if res["match"]:
                continue
            if "error" in res:
                print(cc.wrap(f"    {res['error']}", cc.yellow))
                continue
            diff_count = res["diff_count"]
            total_values = torch.tensor(res["shape"]).prod().item()
            percentage = diff_count / total_values * 100
            abs_diff = (res["ref_tensor"] - res["cur_tensor"]).abs()
            non_zero = abs_diff[abs_diff > 0]
            min_diff = non_zero.min().item() if non_zero.numel() else 0.0
            print(f"    Max diff: {abs_diff.max().item():.8f}")
            print(f"    Min diff: {min_diff:.8f}")
            print(f"    Mean diff: {abs_diff.mean().item():.8f}")
            print(
                f"    Total differing values: {diff_count} out of {total_values} ({percentage:.4f}%)"
            )
            top_lines = self._top_differences(
                res["ref_tensor"], res["cur_tensor"], res["diff_mask"]
            )
            if top_lines:
                print("    Most common differences (up to 5):")
                for line in top_lines:
                    print(line)
