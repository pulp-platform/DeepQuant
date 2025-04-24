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
    """Records and compares tensor values during model execution."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._current: Dict[str, torch.Tensor] = {}
        self._reference: Optional[Dict[str, torch.Tensor]] = None
        self._executionOrder: List[str] = []
        self._nameMap: Dict[str, str] = {}
        self._ignore: Set[str] = set()

    def clear(self) -> None:
        """Clear all recorded data and hooks."""
        self.removeHooks()
        self._current.clear()
        self._reference = None
        self._executionOrder.clear()
        self._nameMap.clear()
        self._ignore.clear()

    def removeHooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def registerForwardHooks(
        self, model: fx.GraphModule, nodeTypes: Optional[List[str]] = None
    ) -> None:
        """Register forward hooks for specified node types."""
        self.removeHooks()
        wanted = [w.lower() for w in nodeTypes] if nodeTypes else []

        def makeHook(name: str):
            def hook(_, __, output):
                if isinstance(output, torch.Tensor):
                    self._current[name] = output.detach().clone()
                    if name not in self._executionOrder:
                        self._executionOrder.append(name)
            return hook

        for name, module in model.named_modules():
            if name and any(w in name.lower() for w in wanted):
                self._hooks.append(module.register_forward_hook(makeHook(name)))

    def recordNodeMapping(self, referenceName: str, currentName: str) -> None:
        """Record a mapping between reference and current node names."""
        self._nameMap[referenceName] = currentName
        if self.debug:
            print(f"Registered mapping: {referenceName} → {currentName}")

    def setReferenceTensors(self) -> None:
        """Save current tensors as reference tensors."""
        self._reference = {k: v.clone() for k, v in self._current.items()}
        self._referenceOrder = list(self._executionOrder)

    def compareTensors(self) -> Dict[str, Dict]:
        """Compare current tensors to reference tensors."""
        if self._reference is None:
            raise RuntimeError("setReferenceTensors has not been called")

        results: Dict[str, Dict] = OrderedDict()
        for refName, refTensor in self._reference.items():
            if refName in self._ignore:
                continue
                
            curName = self._nameMap.get(refName, refName)
            if curName not in self._current:
                results[refName] = {"match": False, "error": f"missing '{curName}'"}
                continue
                
            curTensor = self._current[curName]
            equal = torch.equal(refTensor, curTensor)
            diffMask = refTensor != curTensor
            
            results[refName] = {
                "match": equal,
                "mapped": curName != refName,
                "current_name": curName,
                "shape": tuple(refTensor.shape),
                "diff_count": diffMask.sum().item() if not equal else 0,
                "diff_mask": diffMask,
                "ref_tensor": refTensor,
                "cur_tensor": curTensor,
            }
        return results

    def _topDifferences(
        self, ref: torch.Tensor, cur: torch.Tensor, diffMask: torch.Tensor
    ) -> List[str]:
        """Summarize the most common absolute differences between tensors."""
        maskFlat = diffMask.view(-1).bool()
        if maskFlat.sum() == 0:
            return []
            
        absDiff = (ref - cur).abs().view(-1)[maskFlat]
        unique, counts = torch.unique(absDiff, return_counts=True)
        order = counts.argsort(descending=True)
        
        lines: List[str] = []
        for idx in order[:5]:
            delta = unique[idx].item()
            count = counts[idx].item()
            sampleIndex = (absDiff == delta).nonzero(as_tuple=False)[0].item()
            globalIndex = maskFlat.nonzero(as_tuple=False)[sampleIndex].item()
            beforeValue = ref.view(-1)[globalIndex].item()
            afterValue = cur.view(-1)[globalIndex].item()
            
            lines.append(
                f"    · Δ={delta:.6f}  ({count} values) e.g. idx {globalIndex}: "
                f"{beforeValue:.6f} → {afterValue:.6f}"
            )
        return lines

    def printComparisonResults(self, results: Dict[str, Dict]) -> None:
        """Print tensor comparison results in a readable format."""
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

        orderedNames = getattr(self, "_referenceOrder", list(results.keys()))
        for name in orderedNames:
            if name not in results:
                continue
                
            res = results[name]
            statusColor = cc.green if res["match"] else cc.red
            statusTag = cc.wrap("[OK]" if res["match"] else "[DIFF]", statusColor)
            mappedNote = f" → {res['current_name']}" if res["mapped"] else ""
            
            print(f"  {statusTag} {name}{mappedNote} | shape {res['shape']}")
            if res["match"]:
                continue
                
            if "error" in res:
                print(cc.wrap(f"    {res['error']}", cc.yellow))
                continue
                
            diffCount = res["diff_count"]
            totalValues = torch.tensor(res["shape"]).prod().item()
            percentage = diffCount / totalValues * 100
            absDiff = (res["ref_tensor"] - res["cur_tensor"]).abs()
            nonZero = absDiff[absDiff > 0]
            minDiff = nonZero.min().item() if nonZero.numel() else 0.0
            
            print(f"    Max diff: {absDiff.max().item():.8f}")
            print(f"    Min diff: {minDiff:.8f}")
            print(f"    Mean diff: {absDiff.mean().item():.8f}")
            print(
                f"    Total differing values: {diffCount} of {totalValues} ({percentage:.4f}%)"
            )
            
            topLines = self._topDifferences(
                res["ref_tensor"], res["cur_tensor"], res["diff_mask"]
            )
            if topLines:
                print("    Most common differences (up to 5):")
                for line in topLines:
                    print(line)