# file: selectiontools/strategies/LESS.py
import random
from pathlib import Path

import numpy as np
from torch.nn import CrossEntropyLoss

from .base_strategy import BaseStrategy, register_strategy
from .extract_grads import extract_gradients as extract_train_grads


@register_strategy("LESS")
class LESSInfluence(BaseStrategy):


    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config or {})

        self.data_dir = Path(self.config.get("data_dir", "data/devign")).resolve()
        self.train_name = self.config.get("train_name", "function")
        self.grad_type = self.config.get("grad_type", "adam")
        self.proj_dim = int(self.config.get("proj_dim", 8192))
        self.proj_block_size = int(self.config.get("proj_block_size", 1000))

        self.warmup_epochs = self.config.get("warmup_epochs", 4)
        self.warmup_batch_size = self.config.get("warmup_batch_size", 16)
        self.warmup_lr = self.config.get("warmup_lr", 2e-5)
        self.warmup_ratio = self.config.get("warmup_ratio", 0.03)
        self.ckpts = self.config.get("ckpts")

        self._seeded = False
        self._warmup_done = False
        self._grads_extracted = False

    def _grad_dir(self, ckpt: int) -> Path:
        return (
            self.data_dir
            / "grads"
            / "train"
            / f"{self.train_name}-ckpt{ckpt}-{self.grad_type}"
        )

    def select(self, budget):
        unl = list(self.dataset.unlabeled_indices)
        B = int(budget * len(unl)) if isinstance(budget, float) and budget <= 1 else int(budget)
        if B > len(unl):
            raise ValueError(f"budget({B}) > unlabeled({len(unl)})")

        # seed-label
        if not self._seeded:
            seed_sz = max(1, int(len(unl) * self.warmup_ratio))
            seed_idx = random.sample(unl, seed_sz)
            seed_lbl = [self.dataset.full_labels[i] for i in seed_idx]
            self.dataset.update_with_selected(seed_idx, seed_lbl)
            print(f"[LESS] seeded {seed_sz} samples")
            self._seeded = True

        # LoRA warm-up
        if not self._warmup_done:
            warm_dir = Path(
                self.model.lora_warmup(
                    self.dataset,
                    epochs=self.warmup_epochs,
                    batch_size=self.warmup_batch_size,
                    lr=self.warmup_lr,
                )
            ).resolve()
            if not self.ckpts:
                found = [
                    int(p.name.split("-", 1)[1])
                    for p in warm_dir.iterdir()
                    if p.is_dir() and p.name.startswith("checkpoint-")
                ]
                self.ckpts = sorted(found) or [0]
            print(f"[LESS] using checkpoints {self.ckpts}")
            self._warmup_done = True


        if not self._grads_extracted:
            loss_fn = CrossEntropyLoss()
            for ck in self.ckpts:
                grad_dir = self._grad_dir(ck)
                print(f"[LESS] start extract_grads for ckpt {ck}")
                extract_train_grads(
                    model=self.model,
                    dataset=self.dataset,
                    loss_fn=loss_fn,
                    grad_dir=grad_dir,
                    batch_size=self.warmup_batch_size,
                    device=getattr(self.model, "device", None),
                    proj_dim=self.proj_dim,
                    proj_block_size=self.proj_block_size,
                )
                print(f"[LESS] done extract_grads for ckpt {ck}")
            self._grads_extracted = True


        N = len(self.dataset)
        grads_list = []
        for ck in self.ckpts:
            proj_file = self._grad_dir(ck) / "grads_proj.npy"
            print(f"[LESS] loading projected grads for ckpt {ck}")
            arr = np.memmap(proj_file, dtype="float32", mode="r", shape=(N, self.proj_dim))
            grads_list.append(arr)

        agg = np.sum(np.stack(grads_list, axis=0), axis=0)

        # influence & select
        val_idx = sorted(self.dataset.labeled_indices)
        val_grads = agg[val_idx]
        infl = (agg @ val_grads.T).sum(axis=1)
        unl_scores = infl[unl]
        topk = np.argpartition(-unl_scores, B)[:B]
        return [unl[i] for i in topk]
