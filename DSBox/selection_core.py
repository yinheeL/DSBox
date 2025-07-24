"""
selection_core.py

Core interface: select the corresponding strategy according to the metric name and execute select, returning the selected sample index list.
"""

from strategies.base_strategy import get_strategy_class

def selection(model, dataset, metric, budget, config=None):
    """
Core method: select samples from unlabeled datasets based on metric and budget.

Parameters:
model : the model object passed in by the user, which needs to provide infer_one or infer_batch methods.
dataset : the dataset object passed in by the user, which needs to provide methods such as __len__() and get_item(idx).
metric : a string specifying the name of the selection strategy to be used (must be consistent with the key in the registry).
budget : an integer, the number of samples to be selected in this round.
config : optional, a dictionary or configuration object, used to pass to the hyperparameters of the strategy class.

Returns:
selected_indices (list[int]): a list of indices of the selected samples in the dataset, with length = budget.

Exceptions:
If the metric is not registered or the returned quantity does not match, a ValueError or RuntimeError will be thrown.
    """
    # 1. Get the strategy class based on the metric name
    StrategyClass = get_strategy_class(metric)

    # 2. init
    strategy = StrategyClass(model=model, dataset=dataset, config=config)

    # 3. Call the select method to get the selected sample index
    selected_indices = strategy.select(budget)

    # 4. Verify return value
    if not isinstance(selected_indices, list):
        raise RuntimeError(f"{metric}.select should return a list, but I get {type(selected_indices)}")
    if len(selected_indices) != budget:
        raise RuntimeError(f"{metric}.select returned num {len(selected_indices)} != budget {budget}")

    return selected_indices
