# active_selection_tool/strategies/coreset.py

import numpy as np
import torch
from tqdm import tqdm
from .base_strategy import BaseStrategy, register_strategy

@register_strategy('CoreSet')
class CoreSetSampling(BaseStrategy):
    """
    CoreSet sampling strategy:
    Using the CLS feature vector extracted by the model, the unlabeled set samples are covered by the selected centers as much as possible.
    Each time, a sample with the farthest distance from the current center set is selected as the next center, using a greedy algorithm.
    """
    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config)

        self.batch_size = self.config.get('batch_size', 16)

    def select(self, budget):
        num_unlabeled = len(self.dataset)
        if budget > num_unlabeled:
            raise ValueError(f"The budget {budget} is greater than the number of unlabeled samples {num_unlabeled}")


        device = getattr(self.model, 'device', torch.device('cuda'))

        # ============================
        # Extract CLS features of “labeled” samples
        # ============================
        labeled_idxs = self.dataset.labeled_indices
        if len(labeled_idxs) > 0:
            labeled_feats = []
            # Encoding labeled samples
            for abs_idx in tqdm(labeled_idxs, desc="Encoding labeled samples"):
                code_str = self.dataset.functions[abs_idx]
                feat = self.model._encode(code_str)        # Tensor(shape=(hidden_size,))
                labeled_feats.append(feat.cpu().numpy())
            labeled_feats = np.stack(labeled_feats, axis=0)  # (n_labeled, d)
        else:

            d = self.model.encoder.config.hidden_size
            labeled_feats = np.zeros((0, d), dtype=np.float32)

        # ============================
        # Extract CLS features of “unlabeled” samples
        # ============================
        unlabeled_feats = []
        # Encoding unlabeled samples
        for i in tqdm(range(num_unlabeled), desc="Encoding unlabeled samples"):
            code_str = self.dataset.get_item(i)
            feat = self.model._encode(code_str)  # Tensor(shape=(hidden_size,))
            unlabeled_feats.append(feat.cpu().numpy())
        unlabeled_feats = np.stack(unlabeled_feats, axis=0)  # (num_unlabeled, d)

        # ============================
        # Calculate the distance from each unlabeled sample to the nearest labeled center
        # ============================
        if labeled_feats.shape[0] > 0:
            dist_to_centers = np.full((num_unlabeled,), np.inf, dtype=np.float32)
            batch_size = self.batch_size
            # Init dist computation
            for start in tqdm(range(0, num_unlabeled, batch_size), desc="Init dist computation"):
                end = min(start + batch_size, num_unlabeled)
                batch = unlabeled_feats[start:end]  # (B, d)
                # ||u - c||^2 = ||u||^2 + ||c||^2 - 2 u·c
                u2 = np.sum(batch**2, axis=1, keepdims=True)            # (B, 1)
                c2 = np.sum(labeled_feats**2, axis=1, keepdims=True).T   # (1, n_labeled)
                cross = np.dot(batch, labeled_feats.T)                    # (B, n_labeled)
                d2 = u2 + c2 - 2 * cross
                d2 = np.maximum(d2, 0.0)
                d = np.sqrt(d2)                                           # (B, n_labeled)
                min_d = np.min(d, axis=1)                                 # (B,)
                dist_to_centers[start:end] = min_d
        else:

            dist_to_centers = np.full((num_unlabeled,), np.inf, dtype=np.float32)

        selected_indices = []
        # ============================================
        # Each time, select the unlabeled sample that is farthest from the nearest center
        # ============================================
        for _ in range(budget):
            idx_max = int(np.argmax(dist_to_centers))
            selected_indices.append(idx_max)
            new_center = unlabeled_feats[idx_max:idx_max + 1, :]  # (1, d)

            diff = unlabeled_feats - new_center                      # (num_unlabeled, d)
            d2 = np.sum(diff**2, axis=1)                             # (num_unlabeled,)
            d = np.sqrt(np.maximum(d2, 0.0))                         # (num_unlabeled,)

            dist_to_centers = np.minimum(dist_to_centers, d)


            dist_to_centers[idx_max] = -np.inf

        return selected_indices
