# file: selectiontools/strategies/BALDDropout.py

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from .base_strategy import BaseStrategy, register_strategy

@register_strategy('BALDDropout')
class BALDDropout(BaseStrategy):

    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config or {})
        self.n_drop     = self.config.get('n_drop', 8)
        self.batch_size = self.config.get('batch_size', 32)
        self.max_length = self.config.get('max_length', 256)

    def select(self, budget: int):

        unlabeled = self.dataset.unlabeled_indices
        U = len(unlabeled)
        if budget > U:
            raise ValueError(f"budget={budget} > pool size={U}")


        tokenizer = self.model.tokenizer
        all_ids = []
        all_mask = []
        for gid in unlabeled:
            code = self.dataset.get_raw(gid)[0]
            toks = tokenizer(
                code,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            all_ids.append(toks.input_ids)        # [1, L] on CPU
            all_mask.append(toks.attention_mask)  # [1, L] on CPU

        # stack into CPU tensors [U, L]
        input_ids      = torch.cat(all_ids, dim=0)      # CPU LongTensor
        attention_mask = torch.cat(all_mask, dim=0)     # CPU LongTensor
        local_idxs     = torch.arange(U)                # CPU LongTensor

        #  DataLoader over cached CPU tensors, allow pin_memory
        ds = TensorDataset(input_ids, attention_mask, local_idxs)
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True  # now only CPU tensors enter, safe to pin
        )

        device = self.model.device
        encoder = self.model.encoder
        classifier = self.model.classifier

        # 4)  C by one batch
        encoder.train(); classifier.train()
        ids0, masks0, idxs0 = next(iter(loader))
        ids0   = ids0.to(device, non_blocking=True)
        masks0 = masks0.to(device, non_blocking=True)
        with torch.no_grad():
            out0    = encoder(input_ids=ids0, attention_mask=masks0)
            logits0 = classifier(out0.last_hidden_state[:,0,:])
        C = logits0.size(1)

        # 5)  shape (U, n_drop, C)
        probs = np.zeros((U, self.n_drop, C), dtype=np.float32)

        # 6)：n_drop * num_batches
        total_steps = self.n_drop * len(loader)
        pbar = tqdm(total=total_steps, desc='BALDDropout', leave=True)

        # 7) MC-Dropout
        for t in range(self.n_drop):
            encoder.train(); classifier.train()
            for ids_b, mask_b, locs_b in loader:
                # move batch to GPU
                ids_b   = ids_b.to(device, non_blocking=True)
                mask_b  = mask_b.to(device, non_blocking=True)
                with torch.no_grad():
                    out    = encoder(input_ids=ids_b, attention_mask=mask_b)
                    logits = classifier(out.last_hidden_state[:,0,:])
                    p      = F.softmax(logits, dim=1).cpu().numpy()  # [B, C]

                locs = locs_b.numpy()    # local CPU ints
                probs[locs, t, :] = p
                pbar.update(1)
        pbar.close()

        # 8)  BALD ：E[H] - H[E]
        pb = probs.mean(axis=1)                           # (U, C)
        ent_mean   = -np.sum(pb * np.log(pb + 1e-12), axis=1)
        ent_each   = -np.sum(probs * np.log(probs + 1e-12), axis=2)
        mean_ent   = ent_each.mean(axis=1)
        scores     = mean_ent - ent_mean                  # (U,)

        # 9) Top-k
        if budget == U:
            top_locs = np.arange(U)
        else:
            top_locs = np.argpartition(-scores, budget-1)[:budget]
            top_locs = top_locs[np.argsort(-scores[top_locs])]


        selected = [unlabeled[i] for i in top_locs.tolist()]
        return selected

    def __str__(self):
        return f'BALDDropout(n_drop={self.n_drop}, batch_size={self.batch_size})'
