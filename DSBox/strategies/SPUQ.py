# selectiontools/strategies/SPUQ.py
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .base_strategy import BaseStrategy, register_strategy
from selectiontools.strategies.spuq.llms import LLM
from selectiontools.strategies.spuq.spuq import SPUQ as _SPUQ

@register_strategy('SPUQ')
class SPUQSelection(BaseStrategy):
    """
SPUQ data selection strategy, using TemperaturePerturbation lightweight temperature perturbation:
1) For each source code, randomly perturb the temperature N times only within the range [T_min, T_max];
2) Generate N outputs using LLM.generate;
3) Use InterSampleAggregation (such as RougeL) to aggregate output similarity and calculate confidence;
4) Select budget samples with the lowest confidence (least uncertainty).
    """

    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config)
        self.llm = LLM(model='your gpt model')

        self.spuq = _SPUQ(
            llm=self.llm,
            perturbation='temperature',
            aggregation='rougeL',
            n_perturb=3,
            T_min=0.0,
            T_max=1.0,
            n_jobs=1
        )


    def select(self, budget: int) -> list[int]:
        N = len(self.dataset)

        def score_one(i):
            code = self.dataset.get_item(i)
            msgs = [{'role':'user','content':code}]
            return self.spuq.run(msgs, temperature=0.7)['confidence']

        with ThreadPoolExecutor(max_workers=8) as pool:
            scores = list(tqdm(
                pool.map(score_one, range(N)),
                total=N,
                desc="SPUQ over samples",
                unit="sample"
            ))

        ranked = sorted(range(N), key=lambda i: scores[i])
        return ranked[:budget]


