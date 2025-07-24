# active_selection_tool/strategies/random.py

import random
from .base_strategy import BaseStrategy, register_strategy

@register_strategy('random')
class RandomStrategy(BaseStrategy):

    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config)

    def select(self, budget):

        num_unlabeled = len(self.dataset)
        if budget > num_unlabeled:
            raise ValueError(f"The budget {budget} is greater than the number of unlabeled samples {num_unlabeled}")

        selected_indices = random.sample(range(num_unlabeled), budget)
        return selected_indices
