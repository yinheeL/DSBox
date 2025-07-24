# active_selection_tool/strategies/contrastive.py

import numpy as np
from tqdm import tqdm
import torch
from scipy.special import rel_entr
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.neighbors import NearestNeighbors

from .base_strategy import BaseStrategy, register_strategy

@register_strategy('contrastive')
class ContrastiveActiveLearning(BaseStrategy):
    """
    Contrastive Active Learning (MVB+21) strategy:
    Select samples with the largest average KL divergence between their k nearest neighbors' predicted probability distributions.

    Parameters
    -----
    model: DevignCodeBERTModel instance or other classifier that implements _encode and infer_batch methods.
    dataset: DevignDataset instance, used to access function lists, unlabeled_indices, labeled_indices, full_labels, etc.
    config: dict, which can contain:
    - 'k' (int): The number of neighbors used when calculating KL, the default is 10 (excluding itself).
    - 'normalize' (bool): Whether to perform L2 normalization on embedding, the default is True.
    - 'batch_size' (int): If the number of unlabeled samples is large, the batch size of kNN+KL is calculated in batches, the default is 64.
    - 'pbar' (str or None): Whether to display tqdm Progress bar, 'tqdm' will display it, None will not display it, default is 'tqdm'.
    """
    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config)

        if config is None:
            config = {}
        self.k = config.get('k', 10)
        self.normalize = config.get('normalize', True)
        self.batch_size = config.get('batch_size', 64)
        self.pbar = config.get('pbar', 'tqdm')

    def select(self, budget: int) -> list:

        clf = self.model
        ds = self.dataset

        # embedding
        N = len(ds.functions)
        embeddings = []
        if self.pbar == 'tqdm':
            iterable = tqdm(range(N), desc="Encoding all samples")
        else:
            iterable = range(N)

        for i in iterable:
            code_str = ds.functions[i]
            feat = clf._encode(code_str)
            embeddings.append(feat.cpu().numpy())
        embeddings = np.stack(embeddings, axis=0)  # (N, d)


        all_codes = [ds.functions[i] for i in range(N)]
        proba_list = clf.infer_batch(all_codes, batch_size=self.batch_size)
        embeddings_proba = np.array(proba_list)  # (N, C)


        selected_abs_nd = self.sample(
            clf=clf,
            dataset=ds,
            indices_unlabeled=np.array(ds.unlabeled_indices),
            indices_labeled=np.array(ds.labeled_indices),
            y=None,
            n=budget,
            embeddings=embeddings,
            embeddings_proba=embeddings_proba
        )

        return selected_abs_nd.tolist()

    def sample(self,
               clf,
               dataset,
               indices_unlabeled,
               indices_labeled,
               y,
               n,
               embeddings,
               embeddings_proba=None
               ) -> np.ndarray:


        if embeddings_proba is None:
            raise ValueError(
                "ContrastiveActiveLearning requires embeddings_proba, please make sure the model's infer_batch returns probabilities"
            )

        unlabeled_abs = indices_unlabeled
        U = unlabeled_abs.shape[0]


        if self.normalize:
            embeddings = sk_normalize(embeddings, axis=1)


        nn_model = NearestNeighbors(n_neighbors=self.k + 1,
                                    algorithm='auto',
                                    metric='euclidean',
                                    n_jobs=-1)
        nn_model.fit(embeddings)  # (N, d)


        emb_unl = embeddings[unlabeled_abs]          # (U, d)
        proba_unl = embeddings_proba[unlabeled_abs]  # (U, C)

        scores = np.zeros(U, dtype=float)


        if U <= self.batch_size:
            batches = [np.arange(U)]
        else:
            splits = max(1, U // self.batch_size + 1)
            if self.pbar == 'tqdm':
                batches = list(tqdm(
                    np.array_split(np.arange(U), splits),
                    desc="Splitting unlabeled"
                ))
            else:
                batches = np.array_split(np.arange(U), splits)

        for batch_idx in batches:
            if len(batch_idx) == 0:
                continue


            Xq = emb_unl[batch_idx]  # (B, d)
            distances, neighbors = nn_model.kneighbors(Xq, return_distance=True)

            for j, rel_idx in enumerate(batch_idx):
                p_v = proba_unl[rel_idx]  # (C,)
                nbrs = neighbors[j, 1:self.k + 1]
                kl_vals = []
                for nbr_abs in nbrs:
                    p_i = embeddings_proba[nbr_abs]  # (C,)
                    kl = np.sum(rel_entr(p_i, p_v))
                    kl_vals.append(kl)
                scores[rel_idx] = np.mean(kl_vals)


        topn_rel = np.argpartition(-scores, n)[:n]
        topn_abs = unlabeled_abs[topn_rel]
        return topn_abs

    def __str__(self):
        return (f"ContrastiveActiveLearning(k={self.k}, normalize={self.normalize}, "
                f"batch_size={self.batch_size})")
