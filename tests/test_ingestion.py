"""Chunking: token limits, preserved overlap, sentence and table boundaries, and
deterministic IDs; plus corpus loading and the index manifest."""

from __future__ import annotations

import json

import pytest
from conftest import CORPUS, client, make_cfg, run

from src.ingestion import build_index as build
from src.ingestion import chunk_document, corpus_fingerprint, load_corpus
from src.retriever import load_index
from src.schemas import Document
from src.utils import TokenCounter

COUNTER = TokenCounter("approx")
PROSE = Document(
    doc_id="prose",
    text=" ".join(f"Sentence {i} describes item {i} in moderate detail for testing." for i in range(60)),
)


def chunks_of(doc: Document, size: int = 60, overlap: int = 15):
    return chunk_document(doc, COUNTER, size, overlap)


@pytest.mark.parametrize(("size", "overlap"), [(40, 0), (60, 15), (120, 30), (200, 40)])
def test_chunks_respect_the_token_limit(size, overlap):
    docs = [
        PROSE,
        Document(doc_id="long", text="word " * 700),  # one sentence far longer than a chunk
        Document(doc_id="url", text="see " + "x" * 2000 + " end."),  # a single enormous token
        Document(doc_id="table", text="| a | b |\n" + "\n".join(f"| r{i} | {i} |" for i in range(200))),
    ]
    for doc in docs:
        chunks = chunk_document(doc, COUNTER, size, overlap)
        assert chunks, doc.doc_id
        assert all(0 < c.n_tokens <= size for c in chunks), doc.doc_id
        assert all(c.n_tokens == COUNTER.count(c.text) for c in chunks)


def test_overlap_is_preserved_between_consecutive_chunks():
    chunks = chunks_of(PROSE)
    assert len(chunks) > 3
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.start_char < prev.end_char  # the next chunk starts inside the previous one
        shared = PROSE.text[nxt.start_char : prev.end_char]
        assert prev.text.endswith(shared) and nxt.text.startswith(shared)
        assert COUNTER.count(shared) <= 15


def test_zero_overlap_means_disjoint_chunks():
    chunks = chunks_of(PROSE, overlap=0)
    assert all(nxt.start_char >= prev.end_char for prev, nxt in zip(chunks, chunks[1:], strict=False))


def test_chunks_cover_the_whole_document():
    chunks = chunks_of(PROSE)
    assert chunks[0].start_char == 0 and chunks[-1].end_char == len(PROSE.text)
    assert all(nxt.start_char <= prev.end_char for prev, nxt in zip(chunks, chunks[1:], strict=False))


def test_chunks_start_and_end_on_sentence_boundaries():
    for c in chunks_of(PROSE, size=60, overlap=0):
        assert c.text.startswith("Sentence ") and c.text.endswith(".")


def test_long_sentence_overlap_falls_back_to_trailing_words():
    doc = Document(doc_id="d", text=" ".join(f"w{i}" for i in range(300)) + ".")
    chunks = chunks_of(doc, size=50, overlap=10)
    assert all(nxt.start_char < prev.end_char for prev, nxt in zip(chunks, chunks[1:], strict=False))


def test_ids_are_deterministic_across_runs():
    first, second = chunks_of(PROSE), chunks_of(PROSE)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.content_hash for c in first] == [c.content_hash for c in second]
    assert [c.chunk_id for c in first][:3] == ["prose::0000", "prose::0001", "prose::0002"]
    assert all(c.doc_id == "prose" for c in first)


def test_chunk_ids_change_only_with_the_text_or_settings():
    edited = PROSE.model_copy(update={"text": PROSE.text.replace("Sentence 59", "Sentence fifty-nine")})
    before, after = chunks_of(PROSE), chunks_of(edited)
    assert [c.chunk_id for c in before] == [c.chunk_id for c in after]
    assert before[0].content_hash == after[0].content_hash and before[-1].content_hash != after[-1].content_hash


def test_small_tables_stay_whole():
    table = "| year | revenue |\n| 2019 | 4.1 |\n| 2020 | 5.0 |"
    doc = Document(
        doc_id="t", text=f"Revenue by year follows. {'Filler sentence here. ' * 3}\n\n{table}\n\nThat is all."
    )
    chunks = chunk_document(doc, COUNTER, 40, 5)
    assert sum(table in c.text for c in chunks) == 1
    assert not any("| 2019" in c.text and table not in c.text for c in chunks)


def test_large_tables_repeat_their_header():
    rows = "\n".join(f"| company {i} | {i * 10} |" for i in range(80))
    doc = Document(doc_id="t", text=f"| name | value |\n{rows}")
    chunks = chunk_document(doc, COUNTER, 60, 10)
    assert len(chunks) > 1
    assert all(c.text.startswith("| name | value |") for c in chunks)
    assert all("company" in c.text for c in chunks)


def test_abbreviations_and_detached_periods_do_not_split_sentences():
    doc = Document(doc_id="d", text="Cadence Design Systems inc . reported 37.9% e.g. in 2011. Next sentence.")
    first = "Cadence Design Systems inc . reported 37.9% e.g. in 2011."
    chunks = chunk_document(doc, COUNTER, COUNTER.count(first) + 1, 0)  # room for one sentence only
    assert [c.text for c in chunks] == [first, "Next sentence."]


def test_empty_documents_produce_no_chunks():
    assert chunk_document(Document(doc_id="e", text=""), COUNTER, 50, 5) == []
    assert chunk_document(Document(doc_id="e", text="  \n\n "), COUNTER, 50, 5) == []


def test_title_and_metadata_are_copied_to_chunks():
    doc = Document(doc_id="d", text="One sentence.", title="Report", metadata={"domain": "finance"})
    (only,) = chunk_document(doc, COUNTER, 50, 5)
    assert only.metadata == {"domain": "finance", "title": "Report"}


def test_load_corpus_rejects_duplicates_and_bad_records(tmp_path):
    good = tmp_path / "corpus.jsonl"
    good.write_text("\n".join(json.dumps(d.model_dump()) for d in CORPUS) + "\n", encoding="utf-8")
    assert [d.doc_id for d in load_corpus(str(good))] == [d.doc_id for d in CORPUS]
    duplicate = tmp_path / "dup.jsonl"
    duplicate.write_text(good.read_text() + json.dumps(CORPUS[0].model_dump()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate doc_id"):
        load_corpus(str(duplicate))
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"doc_id": "x", "txt": "typo"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_corpus(str(bad))


def test_build_index_writes_a_consistent_manifest(tmp_path):
    cfg = make_cfg(tmp_path)
    manifest = run(build(cfg, client(cfg), CORPUS))
    chunks, embeddings, loaded = load_index(cfg.ingestion.index_dir)
    assert loaded == manifest
    assert manifest.n_chunks == len(chunks) == embeddings.shape[0] and manifest.n_documents == len(CORPUS)
    assert (manifest.embedding_provider, manifest.embedding_model) == ("offline", cfg.models.embedding)
    assert manifest.token_encoding == "approx" and manifest.corpus_hash == corpus_fingerprint(CORPUS)
