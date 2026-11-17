"""Build the pieces every entry point needs (demo, eval, and later the MCP server),
so they all use the same chunking settings and models."""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk, chunk_text
from .cuad import Contract, load_cuad
from .embedding import Embedder
from .index import Index
from .rerank import Reranker
from .retrieval import Retriever

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


@dataclass
class Corpus:
    contracts: list[Contract]
    chunks: list[Chunk]

    def chunks_by_doc(self) -> dict[str, list[Chunk]]:
        out: dict[str, list[Chunk]] = {}
        for c in self.chunks:
            out.setdefault(c.doc_id, []).append(c)
        return out


def load_corpus(cuad_path: str, max_contracts: int | None = None,
                chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Corpus:
    contracts, _, _ = load_cuad(cuad_path, max_contracts)
    chunks = [ch for c in contracts for ch in chunk_text(c.doc_id, c.text, chunk_size, overlap)]
    return Corpus(contracts, chunks)


def make_models(fake: bool = False) -> tuple[Embedder, Reranker | None]:
    """fake=True: hashing embedder and no reranker. For tests/CI, not for real numbers."""
    if fake:
        from .embedding import HashingEmbedder
        return HashingEmbedder(), None
    from .embedding import SentenceTransformerEmbedder
    from .rerank import CrossEncoderReranker
    return SentenceTransformerEmbedder(), CrossEncoderReranker()


def make_retriever(corpus: Corpus, embedder: Embedder, reranker: Reranker | None,
                   cache_dir: str = "cache", candidates: int = 30) -> Retriever:
    index = Index.build_cached(corpus.chunks, embedder, cache_dir)
    return Retriever(index, reranker, candidates=candidates)
