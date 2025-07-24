# file: selectiontools/strategies/dataset_quantization.py
from __future__ import annotations

import os
import numpy as np
import torch
from tqdm import tqdm
from types import SimpleNamespace
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader

from .base_strategy   import BaseStrategy, register_strategy
from .quantize_sample import methods as qs_methods
from .quantize_bin    import methods as qb_methods


@register_strategy("DatasetQuantization")
class DatasetQuantization(BaseStrategy):
    """
    Dataset Quantization Strategy:

    Steps:
    1. Batch encode all unlabeled samples → embs (U, D)
    2. Repeat num_bins times:
    a. Submodularize the remaining pool to select bin_fraction×pool_size
    b. Delete the selected bin from the pool
    3. Uniform sampling in the bin, per_bin_budget = ceil(budget/num_bins)
    4. Merge all bin results, and randomly truncate if it exceeds the budget
    """

    def __init__(self, model, dataset, config: dict | None = None):
        super().__init__(model, dataset, config)


        self.num_bins     = 3
        self.bin_fraction = 0.10
        self.seed         = 42
        self.balance      = True
        self.batch_size   = 16

        if config:
            self.num_bins     = config.get("num_bins", self.num_bins)
            self.bin_fraction = config.get("bin_fraction", self.bin_fraction)
            self.seed         = config.get("seed", self.seed)
            self.balance      = config.get("balance", self.balance)
            self.batch_size   = config.get("batch_size", self.batch_size)

    # ------------------------------------------------------------------

    def select(self, budget: int):
        unlabeled = self.dataset.unlabeled_indices
        U = len(unlabeled)
        if budget > U:
            raise ValueError(f"budget={budget} is greater than the number of unlabeled samples U={U}")

        # -------- Batch encoding of unlabeled data --------
        self.model.encoder.eval()
        all_embs, all_idxs = [], []

        loader = DataLoader(
            UnlabeledDataset(self.dataset, unlabeled, self.model.tokenizer),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )


        amp_ctx = torch.cuda.amp.autocast if torch.cuda.is_available() else nullcontext
        with torch.no_grad(), amp_ctx():
            for input_ids, attention_mask, batch_pos in tqdm(loader, desc="Embedding"):
                input_ids = input_ids.to(self.model.device, non_blocking=True)
                attention_mask = attention_mask.to(self.model.device, non_blocking=True)

                out = self.model.encoder(input_ids=input_ids,
                                          attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :].cpu().numpy()

                all_embs.append(cls)
                all_idxs.extend(batch_pos.numpy().tolist())

        all_embs = np.vstack(all_embs)            # (U, D)

        # -------- Multiple rounds of Submodular generation of disjoint bins--------
        rng = np.random.RandomState(self.seed)
        pool_loc = np.arange(U)
        bins_loc = []

        for b in range(self.num_bins):
            if len(pool_loc) == 0:
                break

            ds_sub = EmbeddingDataset(all_embs[pool_loc])
            args = SimpleNamespace(fraction=self.bin_fraction,
                                   seed=self.seed + b,
                                   balance=self.balance)

            method = qs_methods["Submodular"](ds_sub, args,
                                               self.bin_fraction,
                                               self.seed + b)
            ret = method.select()                 # {"indices": np.array}
            bin_loc = pool_loc[ret["indices"]]
            bins_loc.append(bin_loc)


            pool_loc = pool_loc[~np.isin(pool_loc, bin_loc)]

        # -------- Uniform sampling in each bin --------
        per_bin_budget = int(np.ceil(budget / max(1, len(bins_loc))))
        selected_loc = []

        for i, bin_loc in enumerate(bins_loc):
            ds_bin = EmbeddingDataset(all_embs[bin_loc])
            args = SimpleNamespace(fraction=per_bin_budget / len(bin_loc),
                                   seed=self.seed + i,
                                   balance=self.balance)

            method2 = qb_methods["Uniform"](ds_bin, args,
                                             args.fraction, self.seed + i)
            ret2 = method2.select()

            chos = bin_loc[ret2["indices"]]
            if len(chos) > per_bin_budget:
                rng.shuffle(chos)
                chos = chos[:per_bin_budget]

            selected_loc.extend(chos.tolist())

        selected_loc = np.asarray(selected_loc, dtype=int)
        if len(selected_loc) > budget:
            rng.shuffle(selected_loc)
            selected_loc = selected_loc[:budget]


        return [all_idxs[i] for i in selected_loc]



class UnlabeledDataset(Dataset):

    def __init__(self, devign_ds, idxs, tokenizer, max_len: int = 256):
        self.ds = devign_ds
        self.idxs = idxs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        gidx = self.idxs[i]
        code = self.ds.functions[gidx]
        toks = self.tokenizer(code,
                              truncation=True,
                              padding="max_length",
                              max_length=self.max_len,
                              return_tensors="pt")
        return (toks.input_ids.squeeze(0),
                toks.attention_mask.squeeze(0),
                torch.tensor(i, dtype=torch.long))


class EmbeddingDataset(Dataset):

    def __init__(self, embeddings: np.ndarray):
        self.embs = embeddings

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, i):

        return self.embs[i], 0
