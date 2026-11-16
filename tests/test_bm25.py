import math

import numpy as np

from rag_eval_mcp.bm25 import BM25, tokenize


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("This Agreement, governed by New-York law (2019).") == [
        "this", "agreement", "governed", "by", "new", "york", "law", "2019"]


def test_matches_formula_by_hand():
    docs = [["governing", "law", "new", "york"], ["termination", "notice"], ["law", "law", "firm"]]
    bm = BM25(docs, k1=1.5, b=0.75)
    avg = (4 + 2 + 3) / 3
    idf_law = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)
    expected_doc0 = idf_law * 1 * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 4 / avg))
    expected_doc2 = idf_law * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 3 / avg))
    np.testing.assert_allclose(bm.scores(["law"]), [expected_doc0, 0.0, expected_doc2])


def test_rare_terms_weigh_more_and_unknown_terms_are_ignored():
    docs = [["law", "york"], ["law"], ["law"], ["other"]]
    bm = BM25(docs)
    assert bm.idf["york"] > bm.idf["law"] > 0
    assert bm.scores(["zzz"]).sum() == 0
    assert np.argmax(bm.scores(["law", "york"])) == 0


def test_empty_corpus():
    assert BM25([]).scores(["x"]).shape == (0,)
