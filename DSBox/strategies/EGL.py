# file: selectiontools/strategies/EGL.py

import numpy as np
import torch
from typing import List, Dict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .base_strategy import BaseStrategy, register_strategy


class TransformersDataset(Dataset):
    def __init__(
        self,
        instances: List[Dict],
        tokenizer,
        tokenization_kwargs: Dict,
        text_column_name: str = "text",
    ):
        self.instances = instances
        self.tokenizer = tokenizer
        self.tokenization_kwargs = tokenization_kwargs
        self.text_column_name = text_column_name

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        enc = self.tokenizer(
            inst[self.text_column_name],
            **self.tokenization_kwargs,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"],      # [1, L]
            "attention_mask": enc["attention_mask"]   # [1, L]
        }


def initialize_gradient_lengths_array(dim):
    return np.zeros(dim, dtype=np.float64)

def initialize_gradients(batch_len, num_classes, device):
    return torch.zeros([num_classes, batch_len], device=device)

def compute_gradient_length(model, sm, gradients, j, k):
    params = torch.cat([
        p.grad.flatten()
        for p in model.model.parameters() if p.requires_grad
    ])
    gradients[j, k] = params.norm(p=2).item() * sm[k, j].item()

def aggregate_gradient_lengths_batch(batch_len, gradient_lengths, gradients, offset):
    gradient_lengths[offset:offset + batch_len] = gradients.sum(dim=0).cpu().numpy()

def compute_gradient_lengths_batch(model, criterion, x, gradients, all_classes):
    outputs = model.model(**x)
    logits  = outputs.logits
    sm      = torch.nn.functional.softmax(logits, dim=1)
    batch_len = logits.size(0)

    for j in range(all_classes.size(0)):
        loss = criterion(logits, all_classes[j])
        for k in range(batch_len):
            model.model.zero_grad()
            loss[k].backward(retain_graph=True)
            compute_gradient_length(model, sm, gradients, j, k)

def compute_gradient_lengths(model, criterion, gradient_lengths, offset, x, device):
    batch_len = x["input_ids"].size(0)
    num_labels = model.model.config.num_labels
    all_classes = torch.stack([
        torch.full((batch_len,), i, dtype=torch.long, device=device)
        for i in range(num_labels)
    ], dim=0)
    gradients = initialize_gradients(batch_len, num_labels, device)
    model.model.zero_grad()
    compute_gradient_lengths_batch(model, criterion, x, gradients, all_classes)
    aggregate_gradient_lengths_batch(batch_len, gradient_lengths, gradients, offset)

def compute_egl_scores(
    model,
    dataloader: DataLoader,
    device: torch.device,
):

    print("[EGL] start computing EGL scores")
    criterion = torch.nn.CrossEntropyLoss(reduction="none").to(device)
    num_obs = len(dataloader.dataset)
    gradient_lengths = initialize_gradient_lengths_array(num_obs)
    offset = 0


    for batch in tqdm(dataloader, desc="EGL batches"):
        batch = {k: v.to(device) for k, v in batch.items()}
        compute_gradient_lengths(model, criterion, gradient_lengths, offset, batch, device)
        offset += batch["input_ids"].size(0)

    print("[EGL] done computing EGL scores")
    return gradient_lengths


@register_strategy("EGL")
class EGLStrategy(BaseStrategy):
    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config or {})
        self.batch_size        = self.config.get("batch_size", 16)
        self.prefilter_ratio   = self.config.get("prefilter_ratio", 5)  # 默认 5 倍
        self.tokenization_kwargs = {
            "truncation": True,
            "padding":    "max_length",
            "max_length": self.config.get("max_length", 256)
        }

    def select(self, budget: int) -> List[int]:

        unlabeled = self.dataset.unlabeled_indices
        U = len(unlabeled)
        if budget > U:
            raise ValueError(f"EGL: budget={budget} > pool size={U}")


        from .EntropySampling import EntropySampling
        light = EntropySampling(self.model, self.dataset, config={ "batch_size": self.batch_size })
        pre_k = min(U, self.prefilter_ratio * budget)
        pre_idx = light.select(pre_k)  # 局部或全局索引都行，我们用全局索引
        # 构建一个临时小 dataset
        orig_unlabeled = self.dataset.unlabeled_indices
        self.dataset.unlabeled_indices = pre_idx


        instances = []
        for gid in pre_idx:
            code, _ = self.dataset.get_raw(gid)
            instances.append({"text": code})

        tsds = TransformersDataset(
            instances           = instances,
            tokenizer           = self.model.tokenizer,
            tokenization_kwargs = self.tokenization_kwargs,
            text_column_name    = "text",
        )
        loader = DataLoader(
            tsds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda batch: {
                "input_ids":      torch.cat([b["input_ids"] for b in batch], dim=0),
                "attention_mask": torch.cat([b["attention_mask"] for b in batch], dim=0)
            },
            pin_memory=False
        )


        device = self.model.device
        scores = compute_egl_scores(self.model, loader, device)  # 形状 (pre_k,)


        if budget == pre_k:
            top_local = np.arange(pre_k)
        else:
            top_local = np.argpartition(-scores, budget - 1)[:budget]
            top_local = top_local[np.argsort(-scores[top_local])]

        selected = [ pre_idx[i] for i in top_local.tolist() ]

        self.dataset.unlabeled_indices = orig_unlabeled
        return selected
