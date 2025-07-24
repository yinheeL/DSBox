# file: selectiontools/strategies/STAFF.py
import math
import random
import torch
import numpy as np
from .base_strategy import BaseStrategy, register_strategy
from selectiontools.model_interface import DevignCodeBERTModel  # 绝对导入

@register_strategy('STAFF')
class StaffSampling(BaseStrategy):
    """
    STAFF two-stage effort + layered redistribution implementation:
    1) small_model (load LoRA adapter) calls get_effort_scores → speculative scores
    2) large_model (original model) calls get_effort_scores → verification scores
    3) Eliminate the lowest prune_ratio fraction, divide into k layers,
       allocate budget according to the average feedback weight of each layer,
       and select the final B samples.
    """

    def __init__(self, model: DevignCodeBERTModel, dataset, config=None):
        super().__init__(model, dataset, config or {})

        self.batch_size = int(self.config.get("batch_size", 16))
        self.k = int(self.config.get("k", 50))
        self.prune_ratio = float(self.config.get("prune_ratio", 0.1))
        self.small_adapter_dir = self.config.get(
            "small_adapter_dir", "path/to/your/loRa Adapter"
        )

        # 初始化两个模型
        self.small_model = model
        self.small_model.load_lora_adapter(self.small_adapter_dir)
        self.small_model.dataset = dataset

        self.large_model = DevignCodeBERTModel(device=model.device.type)
        self.large_model.model = model.model
        self.large_model.dataset = dataset

    def select(self, budget) -> list[int]:
        N = len(self.dataset)
        # 0) 预算解析：支持 fraction 或 count
        if isinstance(budget, float) and 0 < budget <= 1:
            B = max(1, math.ceil(budget * N))
        else:
            B = int(budget)
        if B > N:
            raise ValueError(f"budget({B}) > dataset size({N})")

        rate = B / N

        # 1) speculative Effort
        spec_scores = self.small_model.get_effort_scores(batch_size=self.batch_size)
        # 2) verification Effort
        ver_scores = self.large_model.get_effort_scores(batch_size=self.batch_size)
        # 保持 Python list
        ver_scores = ver_scores.tolist() if isinstance(ver_scores, np.ndarray) else list(ver_scores)

        overall = torch.tensor(spec_scores, dtype=torch.float32)
        scores, idxs = torch.sort(overall, descending=True)

        # prune 最低 prune_ratio 的样本
        prune_n = math.floor(self.prune_ratio * N)
        scores = scores[prune_n:]
        idxs = idxs[prune_n:]

        # stratify into k layers
        s_max, s_min = float(scores[0]), float(scores[-1])
        interval = (s_max - s_min) / self.k
        thresholds = [s_min + interval * (i + 1) for i in range(self.k)]
        strata = [[] for _ in range(self.k)]
        for pos, s in enumerate(scores):
            for layer, thr in enumerate(thresholds):
                if s <= thr:
                    strata[layer].append(int(idxs[pos].item()))
                    break

        # compute feedback per layer
        feedback = []
        for layer_idxs in strata:
            if not layer_idxs:
                feedback.append(0.0)
            else:
                feedback.append(sum(ver_scores[i] for i in layer_idxs) / len(layer_idxs))
        total_fb = sum(feedback) or 1e-12

        # allocate budget per layer
        coreset = []
        m_remain = B
        for layer_idxs, w in zip(strata, feedback):
            if not layer_idxs or m_remain <= 0:
                continue
            b_i = min(len(layer_idxs), math.floor(m_remain * w / total_fb))
            if b_i <= 0:
                continue
            ranked = sorted(layer_idxs, key=lambda i: overall[i], reverse=True)
            chosen = ranked[:b_i]
            coreset.extend(chosen)
            m_remain -= len(chosen)

        # fill remaining if any
        if m_remain > 0:
            pool = [i for i in idxs.tolist() if i not in coreset]
            extra = min(m_remain, len(pool))
            coreset.extend(random.sample(pool, extra))
            m_remain -= extra

        # ensure exactly B samples
        if len(coreset) < B:
            pool_all = [i for i in range(N) if i not in coreset]
            extra_needed = B - len(coreset)
            coreset.extend(random.sample(pool_all, extra_needed))
        if len(coreset) > B:
            coreset = coreset[:B]

        print(f"[STAFF] budget={budget}, B={B}, selected={len(coreset)}")
        return coreset
