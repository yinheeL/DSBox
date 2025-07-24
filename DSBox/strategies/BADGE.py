

import numpy as np
from tqdm import tqdm
import torch
from sklearn.preprocessing import normalize as sk_normalize


try:
    # sklearn >= 1.0
    from sklearn.cluster import kmeans_plusplus
except ImportError:
    # sklearn < 1.0
    from sklearn.cluster._kmeans import k_means_plusplus as kmeans_plusplus

from .base_strategy import BaseStrategy, register_strategy

@register_strategy('BADGE')
class BADGESampling(BaseStrategy):
    """
    BADGE (Batch Active learning by Diverse Gradient Embeddings) strategy:
    1. For each unlabeled sample, get its penultimate embedding h_i (through model._encode).
    2. Get the predicted probability p_i of the sample through model.infer_batch(...) (length = number of categories C).
    3. Construct "gradient embedding" g_i, size = (C * d), where d = h_i dimension; specifically:
    g_i = [ p_i[0] * h_i , p_i[1] * h_i , ..., p_i[C-1] * h_i ] Flatten after cascading by class.
    4. On the gradient embedding {g_i} of all unlabeled samples, use k-means++ to select the index corresponding to the budget center in the U point.
    5. Return the global index list corresponding to the relative index of these budget points in the "unlabeled subset".

    Parameter config Can contain:
    - 'batch_size' (int): batch size when infer_batch and encode are processed in batches, default is 64.
    - 'normalize' (bool): whether to perform L2 normalization on penultimate embedding h, default is True.
    - 'pbar' (str or None): whether to display the tqdm progress bar, 'tqdm' to display, None to not display, default is 'tqdm'.
    """

    def __init__(self, model, dataset, config=None):

        super().__init__(model, dataset, config)

        if config is None:
            config = {}
        self.batch_size = config.get('batch_size', 64)
        self.normalize = config.get('normalize', True)
        self.pbar = config.get('pbar', 'tqdm')

    def select(self, budget: int) -> list:
        """
        Implements select(self, budget) → list required by BaseStrategy:
        Returns a list of "global indices" of length budget.
        """
        clf = self.model
        ds = self.dataset

        # 1. Get the unlabeled list and its length
        unlabeled_abs = np.array(ds.unlabeled_indices, dtype=int)  # 全局索引
        U = unlabeled_abs.shape[0]
        if budget > U:
            raise ValueError(f"预算 budget={budget} 大于未标注样本数 U={U}")

        # 2. Extract penultimate embedding h_i (only for unlabeled samples)
        embeddings_h = []
        if self.pbar == 'tqdm':
            iterator = tqdm(unlabeled_abs, desc="Encoding embeddings for BADGE")
        else:
            iterator = unlabeled_abs

        for abs_idx in iterator:
            code_str = ds.functions[abs_idx]
            feat = clf._encode(code_str)
            embeddings_h.append(feat.cpu().numpy())
        embeddings_h = np.stack(embeddings_h, axis=0)  # (U, d)

        # Optionally L2 normalize the penultimate embedding
        if self.normalize:
            embeddings_h = sk_normalize(embeddings_h, axis=1)

        # 3. Extract the predicted probability p_i (only for unlabeled samples)
        unlabeled_codes = [ds.functions[i] for i in unlabeled_abs]
        proba_list = clf.infer_batch(unlabeled_codes, batch_size=self.batch_size)
        embeddings_proba = np.array(proba_list)  # (U, C)

        # 4. Construct gradient embedding g_i: shape = (U, C * d)
        U, d = embeddings_h.shape
        C = embeddings_proba.shape[1]
        # h_expanded: (U, 1, d) → (U, C, d)
        h_exp = embeddings_h[:, None, :]               # (U, 1, d)
        p_exp = embeddings_proba[:, :, None]           # (U, C, 1)
        grad_embeddings = (h_exp * p_exp).reshape(U, C * d)  # (U, C*d)

        # 5. Initialize and select budget centers using k-means++ on {g_i}
        # kmeans_plusplus returns (centers_array, indices_array)
        if self.pbar == 'tqdm':

            _ = tqdm(total=0, desc="Selecting with k-means++", leave=False)
            centers, indices = kmeans_plusplus(
                grad_embeddings,
                n_clusters=budget,
                random_state=None
            )
        else:
            centers, indices = kmeans_plusplus(
                grad_embeddings,
                n_clusters=budget,
                random_state=None
            )


        selected_abs = unlabeled_abs[indices]
        return selected_abs.tolist()

    def __str__(self):
        return (f"BADGESampling(batch_size={self.batch_size}, normalize={self.normalize})")
