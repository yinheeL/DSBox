import json
import os
import random
from abc import ABC, abstractmethod
from typing import List
import torch
from transformers import RobertaTokenizer

# ──────────────────────────────────────────
# Abstract base class
# ──────────────────────────────────────────
class BaseDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_item(self, idx):
        pass

    def update_with_selected(self, selected_indices, labels):
        raise NotImplementedError

# ──────────────────────────────────────────
# DevignDataset
# ──────────────────────────────────────────
class DevignDataset(BaseDataset):


    _tokenizer = None

    @staticmethod
    def get_tokenizer():
        if DevignDataset._tokenizer is None:
            DevignDataset._tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        return DevignDataset._tokenizer

    def __init__(self, data_dir: str, shuffle: bool = True):
        super().__init__()
        json_path = os.path.join(data_dir, "function.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"function.json not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        self.functions = [e["func"] for e in data_list]
        self.full_labels = [int(e["target"]) for e in data_list]

        self.unlabeled_indices = list(range(len(self.functions)))
        if shuffle:
            random.shuffle(self.unlabeled_indices)
        self.labeled_indices: List[int] = []
        self.labeled_labels: List[int] = []

    # ========== Necessary interfaces ==========
    def __len__(self):
        return len(self.unlabeled_indices)

    def get_labeled(self):
        """
        Return a list of (code, label) tuples for all currently labeled samples.
        """
        return [
            (self.functions[i], self.full_labels[i])
            for i in getattr(self, "labeled_indices", [])
        ]

    def __getitem__(self, idx: int):
        """ PyTorch DataLoader / Trainer call, returns a dict field."""
        global_idx = self.unlabeled_indices[idx]
        code_str = self.functions[global_idx]
        label = self.full_labels[global_idx]

        tokenizer = self.get_tokenizer()
        enc = tokenizer(
            code_str,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

    def get_item(self, idx: int):
        """Called by selection/infer, returns the source string."""
        return self.functions[self.unlabeled_indices[idx]]

    # ========== active learning ==========
    def update_with_selected(self, selected: List[int], labels: List[int]):
        if len(selected) != len(labels):
            raise ValueError("The lengths of selected and labels are inconsistent")
        for gidx, lab in zip(selected, labels):
            if gidx in self.unlabeled_indices:
                self.unlabeled_indices.remove(gidx)
                self.labeled_indices.append(gidx)
                self.labeled_labels.append(lab)

    def get_raw(self, idx: int):
        return self.functions[idx], self.full_labels[idx]
