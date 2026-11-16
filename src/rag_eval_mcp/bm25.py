"""Okapi BM25, written out instead of using rank_bm25.

It is ~40 lines, has no dependency beyond numpy, and an inverted index makes
scoring cost proportional to the postings of the query terms rather than the
whole corpus (rank_bm25 loops over every document for every term).

score(d, q) = sum over terms t in q of
    idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * len(d) / avg_len))
idf(t) = ln((N - df + 0.5) / (df + 0.5) + 1)   # Lucene's variant: never negative
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Drops punctuation, so "Agreement," == "agreement"."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        doc_len = np.array([len(toks) for toks in corpus_tokens], dtype=np.float64)
        avg_len = doc_len.mean() if self.n_docs and doc_len.mean() > 0 else 1.0
        # Per-document length normalisation, precomputed once.
        self._norm = k1 * (1.0 - b + b * doc_len / avg_len)

        postings: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for doc_idx, toks in enumerate(corpus_tokens):
            for term, tf in Counter(toks).items():
                postings[term][0].append(doc_idx)
                postings[term][1].append(tf)
        self._postings = {
            term: (np.array(idx, dtype=np.int64), np.array(tf, dtype=np.float64))
            for term, (idx, tf) in postings.items()
        }
        self.idf = {
            term: math.log((self.n_docs - len(idx) + 0.5) / (len(idx) + 0.5) + 1.0)
            for term, (idx, _) in self._postings.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """BM25 score for every document. Repeated query terms count once."""
        out = np.zeros(self.n_docs, dtype=np.float64)
        for term in set(query_tokens):
            if term not in self._postings:
                continue
            idx, tf = self._postings[term]
            out[idx] += self.idf[term] * tf * (self.k1 + 1.0) / (tf + self._norm[idx])
        return out
