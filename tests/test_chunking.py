import pytest

from rag_eval_mcp.chunking import Chunk, chunk_text, overlaps

TEXT = " ".join(f"word{i}" for i in range(500))  # ~3.9k chars, spaces everywhere


def test_offsets_point_back_into_source():
    for c in chunk_text("d", TEXT, chunk_size=300, overlap=50):
        assert TEXT[c.start:c.end] == c.text


def test_chunks_cover_the_whole_text():
    chunks = chunk_text("d", TEXT, chunk_size=300, overlap=50)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(TEXT)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start <= a.end  # no gaps
        assert b.start > a.start  # always moves forward


def test_respects_size_and_does_not_cut_words():
    for c in chunk_text("d", TEXT, chunk_size=300, overlap=50):
        assert len(c.text) <= 300
        assert not c.text.startswith(("ord", "rd")) and c.text.split()[0].startswith("word")


def test_overlap_is_roughly_as_requested():
    chunks = chunk_text("d", TEXT, chunk_size=300, overlap=50)
    for a, b in zip(chunks, chunks[1:-1]):
        assert 30 <= a.end - b.start <= 60


def test_text_without_spaces_still_terminates():
    chunks = chunk_text("d", "x" * 2500, chunk_size=1000, overlap=100)
    assert [len(c.text) for c in chunks] == [1000, 1000, 700]


def test_short_and_empty_text():
    assert chunk_text("d", "", 100, 10) == []
    assert chunk_text("d", "   \n ", 100, 10) == []
    [c] = chunk_text("d", "hello world", 100, 10)
    assert (c.chunk_id, c.start, c.end) == ("d::0", 0, 11)


def test_bad_params():
    with pytest.raises(ValueError):
        chunk_text("d", TEXT, chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text("d", TEXT, chunk_size=0, overlap=0)


def test_overlaps_is_half_open():
    c = Chunk("d::0", "d", "x" * 10, 10, 20)
    assert overlaps(c, 15, 16)
    assert overlaps(c, 0, 11)
    assert not overlaps(c, 0, 10)  # ends exactly where the chunk starts
    assert not overlaps(c, 20, 30)  # starts exactly where the chunk ends
