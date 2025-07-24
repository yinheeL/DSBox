from abc import ABC, abstractmethod

# Global policy registry: used to store the mapping from metric names to policy classes
_STRATEGY_REGISTRY = {}

def register_strategy(name):
    """
    Decorator: Registers the policy class to the global policy registry.
    Parameters:
    name (str): The metric name to look up in the selection function.
    Returns:
    The decorated class itself.
    """
    def decorator(cls):
        if name in _STRATEGY_REGISTRY:
            raise ValueError(f"The policy name '{name}' has already been registered and cannot be registered again.")
        _STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def get_strategy_class(name):
    """
    Get the policy class with the specified name from the registry.
    Parameters:
    name (str): Policy name, which is the name used when registering.
    Returns:
    The corresponding policy class.
    Exception:
    If name is not in the registry, ValueError is thrown.
    """
    if name not in _STRATEGY_REGISTRY:
        raise ValueError(f"未知的策略名 '{name}'，可用策略：{list(_STRATEGY_REGISTRY.keys())}")
    return _STRATEGY_REGISTRY[name]


class BaseStrategy(ABC):
    """
    Abstract base class: All specific data selection strategies must inherit from this class.
    """
    def __init__(self, model, dataset, config=None):
        """
        Initialization method.
        Parameters:
        model: The model object passed in by the upper layer. The strategy can call model.infer_one or model.infer_batch.
        dataset: The dataset object passed in by the upper layer. The strategy can call dataset.__len__(), dataset.get_item(idx), etc.
        config: An optional configuration dictionary containing hyperparameters required by the strategy.
        """
        self.model = model
        self.dataset = dataset
        self.config = config or {}

    @abstractmethod
    def select(self, budget):
        """
        Core method: Given the budget for this round, select budget data from the unlabeled dataset and return their index list.
        Parameters:
        budget (int): The number of samples to be selected in this round.
        Returns:
        selected_indices (list of int): The index of the selected samples in the dataset, with a length equal to budget.
        """
        pass
