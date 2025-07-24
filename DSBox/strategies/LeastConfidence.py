
import numpy as np
from .base_strategy import BaseStrategy, register_strategy

@register_strategy('LeastConfidence')
class UncertaintyLeastConfidence(BaseStrategy):
    """
    Least Confidence Strategy:
    For each unlabeled sample, call model.infer_one or model.infer_batch to get the probability list probs,
    calculate score = 1 - max(probs), and select the budget index with the largest score.
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
            probs_matrix = self.model.infer_batch(samples)
            probs_matrix = np.array(probs_matrix)
            max_probs = np.max(probs_matrix, axis=1)
            scores = 1.0 - max_probs
        except Exception:
            scores = np.zeros(num_unlabeled)
            for i in range(num_unlabeled):
                sample = self.dataset.get_item(i)
                probs = self.model.infer_one(sample)
                scores[i] = 1.0 - max(probs)


        if budget == num_unlabeled:
            selected_indices = list(range(num_unlabeled))
        else:
            topk_idxs = np.argpartition(-scores, budget - 1)[:budget]
            topk_scores = scores[topk_idxs]
            sorted_topk = topk_idxs[np.argsort(-topk_scores)]
            selected_indices = sorted_topk.tolist()

        return selected_indices
