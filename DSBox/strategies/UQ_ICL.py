# selectiontools/strategies/UQ_ICL.py
from nltk.draw import cfg

from .base_strategy import BaseStrategy, register_strategy
from .UQICLutils import (
    uncertainty_calculation,
    token_uncertainty_calculation_new
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch

@register_strategy('UQ_ICL')
class UQ_ICLSelection(BaseStrategy):
    """
    Uncertainty decomposition (UQ_ICL) strategy based on In-Context Learning, all parameters are set here, config=None.
    Steps:
    1. Few-shot construction: set examples, prompt templates, sampling strategies, etc.;
    2. Generate samples for each code multiple times and collect preds & entropies;
    3. Decompose into AU/EU using information theory methods, and sort uncertainty by EU;
    4. Return the most uncertain top-k index list.
    """
    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config)
        # —— STEP1: 固定 Few-Shot 策略参数 ——
        self.llm_model_name: str = cfg.get("llm_model_name", "Qwen/Qwen1.5-0.5B")
        cache_path = "save model cache path"
        self.load_in_8bit = False
        self.num_demos = 6
        self.num_demos_per_class = 1
        self.sampling_strategy = 'class'
        self.iter_demos = 3
        self.prompt_template = (
            "Please determine whether the following code snippet contains a security vulnerability."
            "[0: no vulnerability, 1: vulnerability]\n\n{code}\nJust return the numeric label."
        )

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            cache_dir=cache_path,
            torch_dtype=torch.float16,
            device_map='auto',
            load_in_8bit=self.load_in_8bit
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_name,
            cache_dir=cache_path
        )

    def select(self, budget: int) -> list[int]:
        N = len(self.dataset)
        scores = []
        for idx in tqdm(range(N), desc='UQ_ICL scoring', unit='sample'):
            code = self.dataset.get_item(idx)
            prompt = self.prompt_template.format(code=code)

            preds, entropies = uncertainty_calculation(
                model=self.llm_model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                demos=self._load_demos(),
                decoding_strategy='beam_search',
                num_demos=self.num_demos,
                num_demos_per_class=self.num_demos_per_class,
                sampling_strategy=self.sampling_strategy,
                iter_demos=self.iter_demos
            )

            AU, EU = token_uncertainty_calculation_new(preds, entropies)
            scores.append(EU)

        ranked = sorted(range(N), key=lambda i: scores[i], reverse=True)
        return ranked[:budget]

    def _load_demos(self):

        demos = [
            ("def add(a, b): return a + b", 0),
            ("eval(input())", 1),
            ("open(file, 'r').read()", 0),
            ("import subprocess; subprocess.call(cmd)", 1),
            ("len([1,2,3])", 0),
            ("exec(code)", 1)
        ]
        return demos
