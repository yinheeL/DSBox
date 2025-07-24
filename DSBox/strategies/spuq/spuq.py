from . import perturbation
from .perturbation import Paraphrasing, RandSysMsg, DummyToken, TemperaturePerturbation
from .aggregation import IntraSampleAggregation, InterSampleAggregation
from .llms import LLM
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class SPUQ:
    def __init__(self,
                 llm,
                 perturbation: str,
                 aggregation: str,
                 n_perturb: int,
                 T_min: float = 0.0,
                 T_max: float = 1.0,
                 n_jobs: int = 1):
        self.llm          = llm
        self.n_perturb    = n_perturb
        self.n_jobs       = n_jobs

        # —— 扰动器初始化 ——
        if perturbation == 'temperature':
            self.perturbation = TemperaturePerturbation(
                n=n_perturb,
                T_min=T_min,
                T_max=T_max
            )
        elif perturbation == 'paraphrasing':
            self.perturbation = Paraphrasing(n=n_perturb)
        elif perturbation == 'rand_sysmsg':
            self.perturbation = RandSysMsg(n=n_perturb)
        elif perturbation == 'dummy':
            self.perturbation = DummyToken(n=n_perturb)
        else:
            raise ValueError(f"Unknown perturbation: {perturbation}")




        if aggregation in ['rouge1', 'rouge2', 'rougeL', 'sbert', 'bertscore']:

                self.aggregation = InterSampleAggregation(aggregation)
        elif aggregation in ['verbalized_word', 'verbalized_num']:

                self.aggregation = IntraSampleAggregation(self.llm, aggregation)
        else:
                raise ValueError(f"Invalid aggregation method: {aggregation}")

    def run(self, messages, temperature: float):
        # 1) Construct n_perturb times (msgs, temp_i) list
        inp_list = self.perturbation.perturb(messages, temperature)
        # 2) Call LLM.generate in parallel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            outs = list(pool.map(
                lambda args: self.llm.generate(args[0], temperature=args[1]),
                inp_list
            ))
        inp_out = list(zip([msgs for msgs, _ in inp_list], outs))
        # 3) Aggregation
        conf = self.aggregation.aggregate(inp_out)
        return {'confidence': conf}
    