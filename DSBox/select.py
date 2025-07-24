# file: selectiontools/select.py
import os
from dataset import DevignDataset
from selection_core import selection
from tqdm import tqdm


def test_selection():

    from model_interface import DevignCodeBERTModel


    data_dir = os.path.join(os.path.dirname(__file__), "../data/devign")
    ds = DevignDataset(data_dir=data_dir, shuffle=False)


    mdl = DevignCodeBERTModel(device="cuda", checkpoint_path=None)


    metric = "LeastConfidence"
    budget = 100


    selected_indices = selection(model=mdl, dataset=ds, metric=metric, budget=budget)
    print(f"{budget} indexes selected by {metric}: {selected_indices}")


if __name__ == "__main__":
    test_selection()
