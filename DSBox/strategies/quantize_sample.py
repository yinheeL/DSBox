# file: selectiontools/quantize_sample.py
"""
子模块采样方法集合
"""
from __future__ import annotations

import random
import numpy as np
import torch
from typing import Dict, Any

methods: Dict[str, Any] = {}



def _to_torch(arr, device, dtype=torch.float16):

    return torch.as_tensor(arr, dtype=dtype, device=device)



class Submodular:


    def __init__(self, dataset, args, fraction: float, seed: int, **kwargs):

        self.dataset = dataset
        self.N = len(dataset)
        self.fraction = fraction
        self.k = max(1, int(self.N * fraction))
        self.seed = seed


    def select(self, device: str | None = None, dtype=torch.float16):

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")


        embs = _to_torch([self.dataset[i][0] for i in range(self.N)],
                         device, dtype)                # (N, D)
        embs = torch.nn.functional.normalize(embs, dim=1)


        S = embs @ embs.T


        torch.manual_seed(self.seed)
        first = torch.randint(0, self.N, (1,), device=device).item()

        selected_mask = torch.zeros(self.N, dtype=torch.bool, device=device)
        selected_mask[first] = True
        coverage = S[:, first].clone()                # (N,)
        selected = [first]

        for _ in range(1, self.k):
            # Δgain_j = Σ_i max(coverage_i, S_ij) − Σ_i coverage_i
            gains = torch.clamp(S - coverage[:, None], min=0).sum(dim=0)
            mask_val = torch.finfo(gains.dtype).min  # -65504 for fp16
            gains[selected_mask] = mask_val
            j = torch.argmax(gains).item()

            selected.append(j)
            selected_mask[j] = True
            coverage = torch.maximum(coverage, S[:, j])

        return {"indices": torch.tensor(selected).cpu().numpy()}


# 注册
methods["Submodular"] = Submodular
