import random
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.cluster import AgglomerativeClustering

from .base_strategy import BaseStrategy, register_strategy

# ----------------------------------------
class _LabeledPoolDataset(Dataset):

    def __init__(self, devign_ds):
        self.devign_ds = devign_ds
    def __len__(self):
        return len(self.devign_ds.labeled_indices)
    def __getitem__(self, idx):
        abs_idx = self.devign_ds.labeled_indices[idx]
        code = self.devign_ds.functions[abs_idx]
        label = self.devign_ds.labeled_labels[idx]
        return code, label


# -------------------- batch collate --------------------

def _collate_fn(batch, tokenizer, device, max_length=256):
    codes, labels = zip(*batch)
    tokens = tokenizer(
        list(codes),
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors="pt"
    )
    return (
        tokens["input_ids"].to(device),
        tokens["attention_mask"].to(device),
        torch.tensor(labels, dtype=torch.long).to(device)
    )


# ----------------------------------------

def _train_on_labeled(model, dataset, epochs, bs, lr, device):
    model.encoder.train(); model.classifier.train()
    loader = DataLoader(
        _LabeledPoolDataset(dataset),
        batch_size=bs,
        shuffle=True,
        collate_fn=lambda b: _collate_fn(b, model.tokenizer, device)
    )
    opt = AdamW(list(model.classifier.parameters()), lr=lr)
    total = len(loader)*epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1*total), total)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for ids, mask, y in loader:
            opt.zero_grad()
            feats = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:,0,:]
            logits = model.classifier(feats)
            crit(logits, y).backward(); opt.step(); sch.step()
    model.encoder.eval(); model.classifier.eval()


# ---------------------------------------

@register_strategy('ClusterMargin')
class ClusterMarginSampling(BaseStrategy):

    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config)

        self.p = 10; self.r = 2; self.k_m = 50; self.k_t = 5; self.eps = 0.5
        self.train_epochs = 1; self.train_bs = 8; self.train_lr = 2e-5
        self.linkage = 'average'
        self.device = torch.device(config.get('device')) if config and 'device' in config else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.device = self.device

    # ----------------- Margin Calculator -----------------
    def _calc_margins(self, abs_indices):
        dset = self.dataset
        probs = np.array(self.model.infer_batch([dset.functions[i] for i in abs_indices],
                                                 batch_size=self.train_bs))
        if probs.shape[1] >= 2:
            top2 = -np.sort(-probs, axis=1)[:,:2]
            margins = top2[:,0] - top2[:,1]
        else:
            margins = probs.flatten()
        return margins

    # --------------------------------------------------
    def select(self, budget=None):
        if budget is None: raise ValueError('ClusterMargin 需要 budget')
        unl_init = list(self.dataset.unlabeled_indices)  # 初始未标记索引快照
        total_unl = len(unl_init)

        B = int(math.ceil(budget*total_unl)) if isinstance(budget, float) and budget<=1 else int(budget)
        B = max(1, min(B, total_unl))  # clamp

        if B <= self.p:
            return random.sample(unl_init, B)

        X = self.dataset
        S = []
        # ------ 1) random seed p ------
        seed_abs = random.sample(unl_init, self.p)
        seed_labels = [X.full_labels[i] for i in seed_abs]
        X.update_with_selected([X.unlabeled_indices.index(i) for i in seed_abs], seed_labels)
        S.extend(seed_abs)
        _train_on_labeled(self.model, X, self.train_epochs, self.train_bs, self.train_lr, self.device)
        # ------ 2) embedding + HAC ------
        feats = [self.model._encode(X.functions[i]).cpu().numpy() for i in range(len(X.functions))]
        feats = np.stack(feats, axis=0)
        hac = AgglomerativeClustering(n_clusters=None, distance_threshold=self.eps, linkage=self.linkage)
        hac.fit(feats); clusters = hac.labels_
        # ------ 3) r  rounds------
        for _ in range(self.r):
            if len(S) >= B: break
            unl = list(X.unlabeled_indices)
            if not unl: break
            margins = self._calc_margins(unl)
            km = min(self.k_m, len(unl))
            cand_abs = [unl[i] for i in np.argsort(margins)[:km]]

            cluster_map = {}
            for a in cand_abs:
                cluster_map.setdefault(int(clusters[a]), []).append(a)
            buckets = sorted(cluster_map.values(), key=len)
            pick = []
            j = 0
            while len(pick)<self.k_t and buckets:
                if buckets[j]:
                    e = buckets[j].pop(random.randrange(len(buckets[j])))
                    pick.append(e)
                j = (j+1) % len(buckets)
            if not pick: break
            X.update_with_selected([X.unlabeled_indices.index(i) for i in pick],
                                   [X.full_labels[i] for i in pick])
            S.extend(pick)
            _train_on_labeled(self.model, X, self.train_epochs, self.train_bs, self.train_lr, self.device)

        if len(S) < B and X.unlabeled_indices:
            remaining = list(X.unlabeled_indices)
            margins = self._calc_margins(remaining)
            order = [remaining[i] for i in np.argsort(margins)]
            need = B - len(S)
            S.extend(order[:need])

        return S[:B]
