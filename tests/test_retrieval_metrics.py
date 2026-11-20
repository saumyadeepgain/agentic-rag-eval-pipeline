"""Retrieval metrics and RRF fusion against hand-computed values, plus retriever
behaviour on a real (offline-embedded) index: fusion on shared chunk IDs, BM25
tokenisation and IDF, filters, empty indexes and embedding-model mismatches."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import client, make_cfg, run

from src.evaluator import average_precision, mrr, ndcg_at_k, recall_at_k, retrieval_metrics, unique
from src.ingestion import build_index
from src.retriever import BM25Index, IndexMismatchError, IndexNotFoundError, Retriever, rank_indices, rrf_fuse
from src.schemas import Document

L2, L3, L5 = math.log2(2), math.log2(3), math.log2(5)
RETRIEVED = ["d3", "d1", "d7", "d2"]
GOLD = {"d1", "d2"}


# ──────────────────────────────────────────────────────────────── Recall@K, MRR, NDCG
def test_recall_at_k_hand_computed():
    assert recall_at_k(RETRIEVED, GOLD, 1) == 0.0
    assert recall_at_k(RETRIEVED, GOLD, 2) == 0.5
    assert recall_at_k(RETRIEVED, GOLD, 3) == 0.5
    assert recall_at_k(RETRIEVED, GOLD, 4) == 1.0


def test_mrr_hand_computed():
    assert mrr(RETRIEVED, GOLD) == 0.5
    assert mrr(["d1"], GOLD) == 1.0
    assert mrr(["x", "y", "z", "d2"], GOLD) == 0.25


def test_ndcg_hand_computed():
    ideal = 1 / L2 + 1 / L3  # two gold documents at ranks 1 and 2
    assert ndcg_at_k(RETRIEVED, GOLD, 2) == pytest.approx((1 / L3) / ideal)
    assert ndcg_at_k(RETRIEVED, GOLD, 4) == pytest.approx((1 / L3 + 1 / L5) / ideal)
    assert ndcg_at_k(["d1", "d2"], GOLD, 2) == pytest.approx(1.0)
    # k=1: the ideal ranking has only one slot, so a relevant first hit is perfect
    assert ndcg_at_k(["d2", "x"], GOLD, 1) == pytest.approx(1.0)


def test_empty_retrieved_list_scores_zero():
    assert recall_at_k([], GOLD, 5) == 0.0
    assert mrr([], GOLD) == 0.0
    assert ndcg_at_k([], GOLD, 5) == 0.0


def test_no_gold_documents_means_not_applicable():
    assert recall_at_k(RETRIEVED, set(), 5) is None
    assert mrr(RETRIEVED, []) is None
    assert ndcg_at_k(RETRIEVED, [], 5) is None


def test_duplicate_ids_are_counted_once():
    assert unique(["d3", "d3", "d1", "d3"]) == ["d3", "d1"]
    # without de-duplication "d3" would push d1 to rank 3 and give recall@2 = 0
    assert recall_at_k(["d3", "d3", "d1"], GOLD, 2) == 0.5
    assert mrr(["d3", "d3", "d1"], GOLD) == 0.5
    assert ndcg_at_k(["d1", "d1", "d1"], GOLD, 3) == pytest.approx(1 / (1 / L2 + 1 / L3))


def test_k_larger_than_the_list_and_the_gold_set():
    assert recall_at_k(["d1"], GOLD, 10) == 0.5
    assert ndcg_at_k(["d1", "d2"], GOLD, 10) == pytest.approx(1.0)  # ideal uses min(|gold|, k)


def test_retrieval_metrics_keys_and_bounds():
    out = retrieval_metrics(RETRIEVED, GOLD, [1, 3, 5])
    assert set(out) == {"recall@1", "ndcg@1", "recall@3", "ndcg@3", "recall@5", "ndcg@5", "mrr"}
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_average_precision():
    assert average_precision([True, False, True]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([False, True]) == pytest.approx(0.5)
    assert average_precision([False, False]) == 0.0
    assert average_precision([]) == 0.0


# ──────────────────────────────────────────────────────────────── RRF fusion
def test_rrf_hand_computed():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert fused == pytest.approx({"a": 1 / 61, "b": 1 / 62 + 1 / 61, "c": 1 / 63 + 1 / 62, "d": 1 / 63})
    assert [doc for doc, _ in rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60)] == ["b", "c", "a", "d"]


def test_rrf_weights():
    fused = rrf_fuse([["a", "b"], ["b", "c"]], k=60, weights=[2.0, 1.0])
    assert dict(fused) == pytest.approx({"a": 2 / 61, "b": 2 / 62 + 1 / 61, "c": 1 / 62})
    assert [doc for doc, _ in fused] == ["b", "a", "c"]
    with pytest.raises(ValueError):
        rrf_fuse([["a"], ["b"]], weights=[1.0])


def test_rrf_duplicates_inside_one_list_count_once():
    assert dict(rrf_fuse([["a", "a", "b"]], k=60)) == pytest.approx({"a": 1 / 61, "b": 1 / 62})


def test_rrf_edge_cases_and_ties():
    assert rrf_fuse([]) == [] and rrf_fuse([[], []]) == []
    assert [doc for doc, _ in rrf_fuse([["b"], ["a"]], k=60)] == ["a", "b"]  # tie → best rank → ID


def test_rank_indices_masks_and_ties():
    scores = np.array([0.5, 0.9, 0.9, 0.0, 0.3])
    assert rank_indices(scores, 3) == [1, 2, 0]
    assert rank_indices(scores, 10, mask=np.array([True, False, True, True, True])) == [2, 0, 4, 3]
    assert rank_indices(scores, 10, positive_only=True) == [1, 2, 0, 4]
    assert rank_indices(scores, 3, mask=np.zeros(5, dtype=bool)) == []


# ──────────────────────────────────────────────────────────────── BM25
def test_bm25_ignores_punctuation_and_case():
    bm25 = BM25Index(["Jane Smith is the CEO.", "The CFO is John."])
    scores = bm25.scores("Who is the CEO?")
    assert scores[0] > 0 and scores[1] == 0  # "ceo?" matches "CEO."; stop words carry no weight


def test_bm25_idf_is_positive_on_tiny_corpora():
    # Okapi IDF is log((N - df + .5) / (df + .5)) = 0 for a term in 1 of 2 documents
    bm25 = BM25Index(["warranty covers defects", "shipping takes five days"])
    assert bm25.scores("warranty")[0] > 0


# ──────────────────────────────────────────────────────────────── retriever on a real index
def test_hybrid_fuses_dense_and_bm25_on_the_same_chunk_id(retriever):
    outcome = run(retriever.retrieve("ACME revenue in 2020", top_k=4))
    ids = [r.chunk_id for r in outcome.results]
    assert len(ids) == len(set(ids))  # no chunk appears twice
    top = outcome.results[0]
    assert top.chunk_id == "acme-1::0000"
    assert top.dense_score is not None and top.bm25_score is not None and top.bm25_score > 0
    # found by both retrievers, so its RRF score sums two terms (> any single 1/(k+1))
    assert top.rrf_score is not None and top.rrf_score > 1 / 61
    assert top.rerank_score is not None and [r.rank for r in outcome.results] == [1, 2, 3, 4]


def test_metadata_filters_are_applied_before_ranking(retriever):
    outcome = run(retriever.retrieve("ACME revenue", top_k=4, filters={"domain": "medical"}))
    assert outcome.results and all(r.chunk.metadata["domain"] == "medical" for r in outcome.results)
    assert run(retriever.retrieve("ACME revenue", filters={"domain": "legal"})).results == []


def test_dense_scores_are_cosine_similarities(retriever):
    assert np.allclose(np.linalg.norm(retriever.embeddings, axis=1), 1.0, atol=1e-5)
    outcome = run(retriever.retrieve("ACME revenue in 2020", top_k=4))
    assert all(-1.0 - 1e-6 <= r.dense_score <= 1.0 + 1e-6 for r in outcome.results)


def test_blank_query_returns_nothing(retriever):
    assert run(retriever.retrieve("   ")).results == []


def test_missing_index_is_a_clear_error(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(IndexNotFoundError, match="make ingest"):
        Retriever.load(cfg, client(cfg))


def test_index_built_with_another_embedding_model_is_refused(built_index, tmp_path):
    other = make_cfg(
        tmp_path, overrides={"ingestion.index_dir": built_index.ingestion.index_dir, "models.embedding": "other-model"}
    )
    with pytest.raises(IndexMismatchError, match="other-model"):
        Retriever.load(other, client(other))


def test_empty_documents_cannot_build_an_index(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(ValueError, match="no chunks"):
        run(build_index(cfg, client(cfg), [Document(doc_id="empty", text="   ")]))
