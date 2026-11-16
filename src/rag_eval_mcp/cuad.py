"""Load CUAD and turn it into a retrieval eval set.

CUAD v1 is SQuAD-format: one "paragraph" per contract holding the full text,
and 41 questions per contract, one per clause category. Each answer is a
character span (answer_start, text). Many questions have no answer (the
contract simply has no such clause); those can't be used to measure retrieval
and are skipped.

Eval item = (contract, query, gold answer spans). A retrieved chunk counts as
relevant if it overlaps any gold span, which is decided at eval time, so the
eval set does not depend on chunk size.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunking import Chunk, overlaps

# 'Highlight the parts (if any) of this contract related to "Governing Law" that
#  should be reviewed by a lawyer. Details: Which state/country's law governs ...'
_QUESTION_RE = re.compile(r'related to "(?P<category>[^"]+)".*?Details:\s*(?P<details>.+)$', re.S)


@dataclass(frozen=True)
class Contract:
    doc_id: str
    title: str
    text: str


@dataclass
class EvalItem:
    qid: str
    doc_id: str
    category: str
    query: str
    spans: list[tuple[int, int]] = field(default_factory=list)  # gold [start, end) char spans

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "EvalItem":
        d = json.loads(line)
        d["spans"] = [tuple(s) for s in d["spans"]]
        return cls(**d)


def parse_question(question: str) -> tuple[str, str]:
    """-> (category, details). Raises ValueError if the CUAD template doesn't match."""
    m = _QUESTION_RE.search(question)
    if not m:
        raise ValueError(f"unexpected CUAD question format: {question[:80]!r}")
    return m.group("category").strip(), " ".join(m.group("details").split())


def make_query(category: str, details: str) -> str:
    """Drop the "Highlight the parts ... reviewed by a lawyer" boilerplate.

    It is identical for all 41 categories, so it carries no signal, and its
    words ("contract", "parties", "reviewed") would add noise to BM25.
    """
    return f"{category}. {details}"


def load_cuad(path: str | Path, max_contracts: int | None = None) -> tuple[list[Contract], list[EvalItem], dict]:
    """-> (contracts, answerable eval items, stats about what was skipped)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
    if max_contracts is not None:
        raw = raw[:max_contracts]

    contracts: list[Contract] = []
    items: list[EvalItem] = []
    stats = {"questions": 0, "unanswerable": 0, "bad_spans": 0}
    for n, entry in enumerate(raw):
        if len(entry["paragraphs"]) != 1:
            raise ValueError(f"expected one paragraph per contract, got {len(entry['paragraphs'])}")
        para = entry["paragraphs"][0]
        text = para["context"]
        doc_id = f"c{n:03d}"
        contracts.append(Contract(doc_id, entry["title"], text))

        for qa in para["qas"]:
            stats["questions"] += 1
            spans: set[tuple[int, int]] = set()
            for ans in qa["answers"]:
                start, end = ans["answer_start"], ans["answer_start"] + len(ans["text"])
                if text[start:end] == ans["text"]:
                    spans.add((start, end))
                else:
                    stats["bad_spans"] += 1  # offset doesn't match the text; don't trust it
            if not spans:
                stats["unanswerable"] += 1
                continue
            category, details = parse_question(qa["question"])
            items.append(EvalItem(qa["id"], doc_id, category, make_query(category, details), sorted(spans)))
    return contracts, items, stats


def relevant_chunk_ids(chunks: list[Chunk], spans: list[tuple[int, int]]) -> set[str]:
    """Chunks (of the item's own contract) that overlap any gold span."""
    return {c.chunk_id for c in chunks if any(overlaps(c, s, e) for s, e in spans)}


def write_jsonl(items: list[EvalItem], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.to_json() + "\n")


def read_jsonl(path: str | Path) -> list[EvalItem]:
    with Path(path).open(encoding="utf-8") as f:
        return [EvalItem.from_json(line) for line in f if line.strip()]
