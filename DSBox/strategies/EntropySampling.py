import numpy as np
from .base_strategy import BaseStrategy, register_strategy

@register_strategy('EntropySampling')
class EntropySampling(BaseStrategy):
    """
    Entropy Sampling Strategy:
    For each unlabeled sample, call model.infer_one or model.infer_batch to get the probability list probs,
    Calculate the entropy value entropy = -sum(p_i * log(p_i)), and select the budget index with the largest entropy.
    """
    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config)
        self.batch_size = self.config.get('batch_size', None)

    def select(self, budget):
        num_unlabeled = len(self.dataset)
        if budget > num_unlabeled:
            raise ValueError(f"The budget {budget} is greater than the number of unlabeled samples {num_unlabeled}")


        try:
            samples = [self.dataset.get_item(i) for i in range(num_unlabeled)]
            probs_matrix = self.model.infer_batch(samples)  # list of lists
            probs_matrix = np.array(probs_matrix)  # shape = (num_unlabeled, num_classes)

            eps = 1e-12
            entropies = -np.sum(probs_matrix * np.log(probs_matrix + eps), axis=1)
            scores = entropies
        except Exception:
            # fallback to infer_one
            scores = np.zeros(num_unlabeled)
            for i in range(num_unlabeled):
                sample = self.dataset.get_item(i)
                probs = np.array(self.model.infer_one(sample))
                eps = 1e-12
                scores[i] = -np.sum(probs * np.log(probs + eps))


        if budget == num_unlabeled:
            selected_indices = list(range(num_unlabeled))
        else:
            topk_idxs = np.argpartition(-scores, budget - 1)[:budget]
            topk_scores = scores[topk_idxs]
            sorted_topk = topk_idxs[np.argsort(-topk_scores)]
            selected_indices = sorted_topk.tolist()

        return selected_indices
