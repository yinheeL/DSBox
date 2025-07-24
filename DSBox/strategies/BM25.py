from .base_strategy import BaseStrategy, register_strategy
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from collections import defaultdict
import numpy as np

@register_strategy('BM25')
class BM25Selection(BaseStrategy):
    """
    BM25 data selection strategy (representative selection), one-time optimization scoring to avoid O(N²) calculation:
    1) Build BM25 index and extract IDF, k1, b parameters;
    2) Build postings table postings(term -> [(doc_id, freq, denom)]);
    3) Pre-calculate the global contribution of each term term_sum = ∑ (f_j*(k1+1)/denom_j);
    4) When selecting, traverse the document terms and accumulate the representative scores according to the postings table and term_sum.
    """
    def __init__(self, model, dataset, config=None):
        super().__init__(model, dataset, config)

        N = len(self.dataset)
        self.tokenized_corpus = [
            self.dataset.get_item(i).strip().split()
            for i in range(N)
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.idf = self.bm25.idf
        self.k1 = self.bm25.k1
        self.b = self.bm25.b

        self.doc_lens = np.array([len(doc) for doc in self.tokenized_corpus], dtype=float)
        self.avgdl = float(np.mean(self.doc_lens))

        self.postings = defaultdict(list)
        for doc_id, tokens in enumerate(self.tokenized_corpus):
            freqs = defaultdict(int)
            for t in tokens:
                freqs[t] += 1
            len_norm = 1 - self.b + self.b * (self.doc_lens[doc_id] / self.avgdl)
            for t, f in freqs.items():
                denom = f + self.k1 * len_norm
                self.postings[t].append((doc_id, f, denom))

        factor = self.k1 + 1.0
        self.term_sum = {
            t: sum((f_j * factor) / denom_j for (_, f_j, denom_j) in plist)
            for t, plist in self.postings.items()
        }

    def select(self, budget: int) -> list[int]:
        N = len(self.dataset)
        scores = []
        factor = self.k1 + 1.0

        for idx in tqdm(range(N), desc='BM25 scoring', unit='sample'):
            tokens = self.tokenized_corpus[idx]
            freqs = defaultdict(int)
            for t in tokens:
                freqs[t] += 1
            len_norm = 1 - self.b + self.b * (self.doc_lens[idx] / self.avgdl)
            s = 0.0
            for t, f_i in freqs.items():
                denom_i = f_i + self.k1 * len_norm
                idf_t = self.idf.get(t, 0.0)
                term_factor = (f_i * factor) / denom_i
                s += idf_t * term_factor * self.term_sum.get(t, 0.0)
            scores.append(s)

        ranked = sorted(range(N), key=lambda i: scores[i], reverse=True)

        return ranked[:budget]
