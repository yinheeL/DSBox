import numpy as np
from .base_strategy import BaseStrategy, register_strategy

@register_strategy('Margin')
class MarginSampling(BaseStrategy):
    """
    Margin Sampling:
    For each unlabeled sample, call model.infer_one or model.infer_batch to get the probability list probs,
    Find the highest and second highest probabilities p_max1, p_max2, and calculate margin = p_max1 - p_max2.
    Select the budget index with the smallest margin (highest uncertainty).
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
            probs_matrix = np.array(probs_matrix)  # shape = (num_unlabeled, num_classes)

            sorted_probs = -np.sort(-probs_matrix, axis=1)

            margins = sorted_probs[:, 0] - sorted_probs[:, 1]
            #  smallest margins
            scores = -margins
        except Exception:
            # fallback to infer_one
            scores = np.zeros(num_unlabeled)
            for i in range(num_unlabeled):
                sample = self.dataset.get_item(i)
                probs = np.array(self.model.infer_one(sample))

                if len(probs) < 2:

                    margin = probs[0]
                else:
                    sorted_p = -np.sort(-probs)
                    margin = sorted_p[0] - sorted_p[1]
                scores[i] = -margin


        if budget == num_unlabeled:
            selected_indices = list(range(num_unlabeled))
        else:
            topk_idxs = np.argpartition(-scores, budget - 1)[:budget]
            topk_scores = scores[topk_idxs]
            sorted_topk = topk_idxs[np.argsort(-topk_scores)]
            selected_indices = sorted_topk.tolist()

        return selected_indices
