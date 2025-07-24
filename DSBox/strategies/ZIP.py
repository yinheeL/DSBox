# selectiontools/strategies/ZIP.py

import zlib
import math
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .base_strategy import BaseStrategy, register_strategy

@register_strategy('ZIP')
class ZipSampling(BaseStrategy):
    """
    ZIP three-stage greedy compression rate sampling, with progress bar display:
    1) Calculate the compression rate of a single sample in the entire pool π_D (progress display)
    2) Loop until the budget is full:
    - Stage1 & 2: Combine the selected + candidate scores in the remaining pool and take the minimum k2 (progress display)
    - Stage3: Greedily select k3 from the k2 candidates (progress display)
    3) Return to the global index list
    """
    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config)
        self.k1     = 10000
        self.k2     = 800
        self.k3     = 300
        self.n_jobs = 4

    def _get_compression_ratio(self, text: str) -> float:
        raw  = text.encode('utf-8')
        comp = zlib.compress(raw, level=9)
        return len(raw) / len(comp)

    def select(self, budget: int) -> list[int]:

        all_codes = [ self.dataset.get_item(i) for i in range(len(self.dataset)) ]
        idx_pool  = list(range(len(all_codes)))


        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            global_rates = list(tqdm(
                executor.map(self._get_compression_ratio, all_codes),
                total=len(all_codes),
                desc="ZIP: global compression"
            ))
        global_state = torch.tensor(global_rates)

        selected = []
        cur_pool = idx_pool[:]


        while len(selected) < budget:
            remaining = budget - len(selected)


            def score12(idx):
                combined = "".join([all_codes[i] for i in selected] + [all_codes[idx]])
                return self._get_compression_ratio(combined)

            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                scores12 = list(tqdm(
                    executor.map(score12, cur_pool),
                    total=len(cur_pool),
                    desc=f"ZIP: Stage12 scoring ({len(selected)}/{budget})"
                ))


            if self.k1 != 'all' and isinstance(self.k1, int):
                pool_sorted = sorted(cur_pool, key=lambda i: global_state[i])
                subset = pool_sorted[:self.k1]
            else:
                subset = cur_pool


            idx_scores = list(zip(subset, [scores12[cur_pool.index(i)] for i in subset]))
            idx_scores.sort(key=lambda x: x[1])
            cand2 = [i for i, _ in idx_scores[:self.k2]]


            for _ in tqdm(range(min(self.k3, remaining)),
                          desc=f"ZIP: Stage3 greedy select ({len(selected)}/{budget})"):

                best = min(
                    cand2,
                    key=lambda idx: self._get_compression_ratio(
                        "".join([all_codes[i] for i in selected] + [all_codes[idx]])
                    )
                )
                selected.append(best)
                cand2.remove(best)
                cur_pool.remove(best)


        return selected[:budget]
