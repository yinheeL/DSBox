# active_selection_tool/strategies/kmeans.py

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from .base_strategy import BaseStrategy, register_strategy

@register_strategy('KMeans')
class KMeansSampling(BaseStrategy):
    """
    KMeans sampling strategy:
    Extract CLS features for all unlabeled samples, use sklearn.cluster.KMeans clustering,
    set the number of clusters to budget (the number of samples to be selected). Then select the sample closest to the cluster center in each cluster (the one with the smallest Euclid distance in the feature space), and return the index list corresponding to these samples.

    config can be passed:
    - 'batch_size': used for batching when extracting features (optional, default 16).
    - 'kmeans_kwargs': additional parameters passed to sklearn.cluster.KMeans (optional).
    For example: {'n_init': 10, 'random_state': 42}, etc.
    """
    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config)
        # batch_size 用于提取特征时的分批（避免一次性对大量字符串逐条 encode）
        self.batch_size = self.config.get('batch_size', 16)
        # 可以把需要传给 sklearn KMeans 的参数放在 kmeans_kwargs 中
        self.kmeans_kwargs = self.config.get('kmeans_kwargs', {})

    def select(self, budget):
        num_unlabeled = len(self.dataset)
        if budget > num_unlabeled:
            raise ValueError(f"The budget {budget} is greater than the number of unlabeled samples {num_unlabeled}")


        unlabeled_feats = []

        for i in tqdm(range(num_unlabeled), desc="Encoding unlabeled samples"):
            code_str = self.dataset.get_item(i)

            feat = self.model._encode(code_str)    # Tensor(shape=(hidden_size,))
            unlabeled_feats.append(feat.cpu().numpy())

        unlabeled_feats = np.stack(unlabeled_feats, axis=0)  # dtype=float32


        kmeans = KMeans(n_clusters=budget, **self.kmeans_kwargs)

        kmeans.fit(unlabeled_feats)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_


        distances_per_cluster = {
            k: (np.inf, None) for k in range(budget)
        }


        for idx in range(num_unlabeled):
            lbl = labels[idx]

            diff = unlabeled_feats[idx] - centers[lbl]
            dist = np.linalg.norm(diff)
            if dist < distances_per_cluster[lbl][0]:
                distances_per_cluster[lbl] = (dist, idx)

        selected_indices = []
        for k in range(budget):
            _, idx_min = distances_per_cluster[k]

            selected_indices.append(int(idx_min))

        return selected_indices
