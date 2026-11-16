import json

import pytest

from rag_eval_mcp.chunking import chunk_text
from rag_eval_mcp.cuad import load_cuad, parse_question, read_jsonl, relevant_chunk_ids, write_jsonl

Q = ('Highlight the parts (if any) of this contract related to "{cat}" that should be reviewed '
     'by a lawyer. Details: {details}')
CONTEXT = "SUPPLY AGREEMENT. " + "filler text " * 100 + "This Agreement is governed by Swedish law. " + "more " * 50


def fake_cuad(tmp_path):
    gov_start = CONTEXT.index("This Agreement is governed")
    data = {"version": "aok_v1.0", "data": [{
        "title": "ACME_Supply Agreement",
        "paragraphs": [{"context": CONTEXT, "qas": [
            {"id": "q1", "question": Q.format(cat="Governing Law", details="Which law governs?"),
             "answers": [{"text": "This Agreement is governed by Swedish law.", "answer_start": gov_start},
                         # duplicate span, should be merged
                         {"text": "This Agreement is governed by Swedish law.", "answer_start": gov_start}],
             "is_impossible": False},
            {"id": "q2", "question": Q.format(cat="Non-Compete", details="Is there a non-compete?"),
             "answers": [], "is_impossible": True},
            {"id": "q3", "question": Q.format(cat="Document Name", details="The name of the contract"),
             "answers": [{"text": "SUPPLY AGREEMENT", "answer_start": 5}],  # wrong offset
             "is_impossible": False},
        ]}],
    }]}
    path = tmp_path / "cuad.json"
    path.write_text(json.dumps(data))
    return path, gov_start


def test_parse_question_strips_boilerplate():
    cat, details = parse_question(Q.format(cat="Renewal Term", details="What is the\n renewal term?"))
    assert (cat, details) == ("Renewal Term", "What is the renewal term?")
    with pytest.raises(ValueError):
        parse_question("What is the governing law?")


def test_load_skips_unanswerable_and_bad_spans(tmp_path):
    path, gov_start = fake_cuad(tmp_path)
    contracts, items, stats = load_cuad(path)
    assert [c.doc_id for c in contracts] == ["c000"]
    assert [it.qid for it in items] == ["q1"]
    assert items[0].query == "Governing Law. Which law governs?"
    assert items[0].spans == [(gov_start, gov_start + len("This Agreement is governed by Swedish law."))]
    assert stats == {"questions": 3, "unanswerable": 2, "bad_spans": 1}


def test_relevant_chunks_follow_chunk_size(tmp_path):
    path, _ = fake_cuad(tmp_path)
    contracts, items, _ = load_cuad(path)
    text = contracts[0].text
    for size, overlap in [(200, 50), (500, 100)]:
        chunks = chunk_text("c000", text, size, overlap)
        rel = relevant_chunk_ids(chunks, items[0].spans)
        assert rel, f"no relevant chunk at size {size}"
        # Relevant chunks, joined, must contain the whole answer.
        joined = "".join(text[c.start:c.end] for c in chunks if c.chunk_id in rel)
        assert "Swedish" in joined
    # One big chunk -> it is the only relevant one.
    assert relevant_chunk_ids(chunk_text("c000", text, 5000, 0), items[0].spans) == {"c000::0"}


def test_jsonl_roundtrip(tmp_path):
    path, _ = fake_cuad(tmp_path)
    _, items, _ = load_cuad(path)
    out = tmp_path / "eval" / "x.jsonl"
    write_jsonl(items, out)
    assert read_jsonl(out) == items
