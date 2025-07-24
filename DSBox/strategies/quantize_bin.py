# file: selectiontools/quantize_bin.py

import numpy as np
import random
from collections import defaultdict

methods = {}

class Uniform:


    def __init__(self, dataset, args, fraction, seed, balance=True):

        self.dataset = dataset
        self.M = len(dataset)
        self.fraction = fraction
        self.m = max(1, int(self.M * fraction))
        self.seed = seed
        self.balance = balance

    def select(self):
        random.seed(self.seed)
        indices = list(range(self.M))
        if not self.balance:
            chosen = random.sample(indices, self.m)
        else:

            buckets = defaultdict(list)
            for i in indices:
                label = self.dataset[i][1]
                buckets[label].append(i)

            C = len(buckets)
            base = self.m // C
            extra = self.m - base * C
            chosen = []
            for lbl, inds in buckets.items():
                cnt = base + (1 if extra > 0 else 0)
                extra -= 1
                cnt = min(cnt, len(inds))
                chosen.extend(random.sample(inds, cnt))
        return {"indices": np.array(chosen, dtype=int)}

methods["Uniform"] = Uniform
