# file: selectiontools/strategies/BatchBALD.py

import os
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from batchbald_redux.batchbald import get_batchbald_batch
from .base_strategy import BaseStrategy, register_strategy

class UnlabeledCodeDataset(Dataset):
    def __init__(self, codes):
        self.codes = codes
    def __len__(self):
        return len(self.codes)
    def __getitem__(self, idx):
        return self.codes[idx], idx

@register_strategy('BatchBALD')
class BatchBALDSampling(BaseStrategy):
    def __init__(self, model, dataset, config=None):

        config = {} if config is None else config
        super().__init__(model, dataset, config)

        #  self.config
        self.batch_size  = self.config.get('batch_size', 32)
        self.K           = self.config.get('K', 5)
        self.num_samples = self.config.get('num_samples', 2000)
        self.pbar        = self.config.get('pbar', 'tqdm')

    def select(self, budget: int) -> list:
        clf = self.model
        ds  = self.dataset

        unlabeled_abs = np.array(ds.unlabeled_indices, dtype=int)
        U = len(unlabeled_abs)
        if budget > U:
            raise ValueError(f"budget={budget} is greater than the number of unlabeled samples U={U}")

        # 1) DataLoader
        codes = [ds.functions[i] for i in unlabeled_abs]
        u_dataset = UnlabeledCodeDataset(codes)
        def collate_fn(batch):
            codes, rel_idxs = zip(*batch)
            toks = clf.tokenizer(
                list(codes),
                truncation=True, max_length=256, padding='max_length',
                return_tensors="pt"
            )
            return toks.input_ids.to(clf.device), \
                   toks.attention_mask.to(clf.device), \
                   torch.tensor(rel_idxs, device='cpu')
        unlabeled_loader = DataLoader(
            u_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # 2) First forward pass, determine the number of C categories
        ids0, mask0, rel0 = next(iter(unlabeled_loader))
        out0 = clf.encoder(input_ids=ids0, attention_mask=mask0)
        logits0 = clf.classifier(out0.last_hidden_state[:,0,:])
        C = logits0.size(1)

        # 3) Initialize probs_all
        probs_all = np.zeros((U, self.K, C), dtype=np.float32)


        num_batches = len(unlabeled_loader)
        total_iters = self.K * num_batches
        use_pbar    = (self.pbar == 'tqdm')
        if use_pbar:
            pbar = tqdm(total=total_iters, desc="BatchBALD")

        #  K times MC-Dropout
        for k in range(self.K):
            # keep dropout active
            clf.encoder.train(); clf.classifier.train()
            for ids, masks, rel_idxs in unlabeled_loader:
                with torch.no_grad():
                    out = clf.encoder(input_ids=ids, attention_mask=masks)
                    logits = clf.classifier(out.last_hidden_state[:,0,:])
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                for i, rel in enumerate(rel_idxs.tolist()):
                    probs_all[rel, k, :] = probs[i]
                if use_pbar:
                    pbar.update(1)

        if use_pbar:
            pbar.close()

        #  log_probs_all Tensor

        log_probs_all = torch.from_numpy(np.log(probs_all + 1e-12))


        devnull = open(os.devnull, 'w')
        with redirect_stdout(devnull), redirect_stderr(devnull):
            acquisition = get_batchbald_batch(
                log_probs_all,
                budget,
                self.num_samples
            )
        devnull.close()


        rel_indices = acquisition.indices  # numpy array of size `budget`
        selected_abs = unlabeled_abs[rel_indices]
        return selected_abs.tolist()

    def __str__(self):
        return (f"BatchBALDSampling(batch_size={self.batch_size}, "
                f"K={self.K}, num_samples={self.num_samples})")
