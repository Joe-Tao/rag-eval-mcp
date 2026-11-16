"""Split documents into overlapping character windows that keep their source offsets.

Why offsets: the eval labels are character spans in the original contract
(CUAD answer_start + len(text)). If every chunk remembers [start, end), we can
decide "is this chunk relevant?" by span overlap, for ANY chunk size. That lets
us change chunking later without rebuilding the eval set.
"""

from __future__ import annotations

from dataclasses import dataclass

_WHITESPACE = (" ", "\n", "\t")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc_id>::<n>"
    doc_id: str
    text: str
    start: int  # char offset in the source document, inclusive
    end: int  # exclusive; text == source[start:end]


def _last_whitespace(text: str, lo: int, hi: int) -> int:
    """Index of the last whitespace char in text[lo:hi], or -1."""
    return max(text.rfind(ch, lo, hi) for ch in _WHITESPACE)


def _next_whitespace(text: str, lo: int, hi: int) -> int:
    """Index of the first whitespace char in text[lo:hi], or -1."""
    hits = [i for i in (text.find(ch, lo, hi) for ch in _WHITESPACE) if i != -1]
    return min(hits) if hits else -1


def chunk_text(doc_id: str, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    """Fixed-size windows of ~chunk_size chars, overlapping by ~overlap chars.

    Window edges are moved to whitespace so words are not cut in half. The end
    only moves back as far as the middle of the window, so a long run of text
    with no spaces still gets cut (at exactly chunk_size).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    chunks: list[Chunk] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            ws = _last_whitespace(text, start + chunk_size // 2, end)
            if ws != -1:
                end = ws

        piece = text[start:end]
        if piece.strip():
            chunks.append(Chunk(f"{doc_id}::{len(chunks)}", doc_id, piece, start, end))
        if end >= n:
            break

        # Step back by `overlap`, then forward to the next word boundary.
        next_start = end - overlap
        if next_start > 0 and text[next_start - 1] not in _WHITESPACE:
            ws = _next_whitespace(text, next_start, end)
            if ws != -1:
                next_start = ws + 1
        start = max(next_start, start + 1)  # always make progress
    return chunks


def overlaps(chunk: Chunk, span_start: int, span_end: int) -> bool:
    """True if the chunk shares at least one character with [span_start, span_end)."""
    return chunk.start < span_end and span_start < chunk.end
