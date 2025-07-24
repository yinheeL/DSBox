# active_selection_tool/strategies/learning_loss.py

import numpy as np
from .base_strategy import BaseStrategy, register_strategy

@register_strategy('LearningLoss')
class LearningLossSampling(BaseStrategy):
    """
    LearningLoss sampling strategy:
    Call model.predict_loss_batch() for each unlabeled sample to get a list of prediction loss scores,
    and then select the budget sample index with the largest score and return it.
    """
    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config)
        self.batch_size = self.config.get('batch_size', 16)

    def select(self, budget):
        num_unlabeled = len(self.dataset)
        if budget > num_unlabeled:
            raise ValueError(f"The budget {budget} is greater than the number of unlabeled samples {num_unlabeled}")


        samples = [self.dataset.get_item(i) for i in range(num_unlabeled)]


        scores = self.model.predict_loss_batch(samples, batch_size=self.batch_size)
        scores = np.array(scores)


        if budget == num_unlabeled:
            selected_indices = list(range(num_unlabeled))
        else:
            topk_idxs = np.argpartition(-scores, budget - 1)[:budget]
            topk_scores = scores[topk_idxs]
            sorted_topk = topk_idxs[np.argsort(-topk_scores)]
            selected_indices = sorted_topk.tolist()

        return selected_indices
