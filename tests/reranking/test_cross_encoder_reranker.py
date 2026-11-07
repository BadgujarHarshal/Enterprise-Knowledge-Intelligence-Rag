# tests/reranking/test_cross_encoder_reranker.py

"""
Unit tests for the EKI-RAG CrossEncoderReranker.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from app.reranking.cross_encoder_reranker import (
    CrossEncoderReranker,
    RerankingResult,
)


@dataclass
class FakeDocument:
    text: str
    document_id: str


class FakeCrossEncoderModel:
    """
    Lightweight fake CrossEncoder used for unit testing.

    Scores are deterministic and supplied directly.
    """

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.calls = []

    def predict(
        self,
        pairs,
        batch_size=32,
        show_progress_bar=False,
    ):
        self.calls.append(
            {
                "pairs": pairs,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
            }
        )

        if len(pairs) != len(self.scores):
            raise AssertionError(
                "Fake model received an unexpected number of pairs."
            )

        return self.scores


def build_reranker(scores):
    """
    Build a reranker instance without loading a real model.
    """

    reranker = object.__new__(CrossEncoderReranker)

    reranker.model_name = "fake-model"
    reranker.batch_size = 32
    reranker.model = FakeCrossEncoderModel(scores)

    return reranker


def test_score_pairs_returns_one_score_per_document():
    reranker = build_reranker([0.8, 0.2, 0.6])

    scores = reranker.score_pairs(
        "What is the capital of France?",
        [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "France is a country in Europe.",
        ],
    )

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (3,)
    assert scores.tolist() == pytest.approx([0.8, 0.2, 0.6])


def test_score_pairs_preserves_query_document_pairs():
    reranker = build_reranker([0.9, 0.1])

    reranker.score_pairs(
        "test query",
        [
            "document one",
            "document two",
        ],
    )

    calls = reranker.model.calls

    assert len(calls) == 1

    assert calls[0]["pairs"] == [
        ("test query", "document one"),
        ("test query", "document two"),
    ]


def test_score_pairs_empty_documents_returns_empty_array():
    reranker = build_reranker([])

    scores = reranker.score_pairs(
        "test query",
        [],
    )

    assert isinstance(scores, np.ndarray)
    assert scores.size == 0


def test_score_pairs_rejects_empty_query():
    reranker = build_reranker([0.5])

    with pytest.raises(ValueError):
        reranker.score_pairs(
            "",
            ["document"],
        )


def test_score_pairs_rejects_non_string_document():
    reranker = build_reranker([0.5])

    with pytest.raises(TypeError):
        reranker.score_pairs(
            "query",
            [123],
        )


def test_rerank_orders_candidates_by_score():
    reranker = build_reranker([0.2, 0.9, 0.5])

    candidates = [
        FakeDocument("document one", "doc-1"),
        FakeDocument("document two", "doc-2"),
        FakeDocument("document three", "doc-3"),
    ]

    results = reranker.rerank(
        "query",
        candidates,
    )

    assert [result.document.document_id for result in results] == [
        "doc-2",
        "doc-3",
        "doc-1",
    ]

    assert [result.rank for result in results] == [
        1,
        2,
        3,
    ]

    assert [result.score for result in results] == pytest.approx(
        [0.9, 0.5, 0.2]
    )


def test_rerank_supports_top_k():
    reranker = build_reranker([0.2, 0.9, 0.5])

    candidates = [
        FakeDocument("document one", "doc-1"),
        FakeDocument("document two", "doc-2"),
        FakeDocument("document three", "doc-3"),
    ]

    results = reranker.rerank(
        "query",
        candidates,
        top_k=2,
    )

    assert len(results) == 2

    assert [result.document.document_id for result in results] == [
        "doc-2",
        "doc-3",
    ]

    assert [result.rank for result in results] == [
        1,
        2,
    ]


def test_rerank_supports_dictionary_candidates():
    reranker = build_reranker([0.4, 0.8])

    candidates = [
        {
            "text": "document one",
            "id": "doc-1",
        },
        {
            "text": "document two",
            "id": "doc-2",
        },
    ]

    results = reranker.rerank(
        "query",
        candidates,
    )

    assert results[0].document["id"] == "doc-2"
    assert results[1].document["id"] == "doc-1"


def test_rerank_supports_custom_text_getter():
    reranker = build_reranker([0.3, 0.7])

    candidates = [
        {
            "content": "first document",
            "id": "doc-1",
        },
        {
            "content": "second document",
            "id": "doc-2",
        },
    ]

    results = reranker.rerank(
        "query",
        candidates,
        document_text_getter=lambda item: item["content"],
    )

    assert results[0].document["id"] == "doc-2"
    assert results[1].document["id"] == "doc-1"


def test_rerank_rejects_empty_query():
    reranker = build_reranker([0.5])

    candidates = [
        FakeDocument("document", "doc-1"),
    ]

    with pytest.raises(ValueError):
        reranker.rerank(
            "",
            candidates,
        )


def test_rerank_empty_candidates_returns_empty_list():
    reranker = build_reranker([])

    results = reranker.rerank(
        "query",
        [],
    )

    assert results == []


def test_rerank_rejects_invalid_top_k():
    reranker = build_reranker([0.5])

    candidates = [
        FakeDocument("document", "doc-1"),
    ]

    with pytest.raises(ValueError):
        reranker.rerank(
            "query",
            candidates,
            top_k=0,
        )


def test_default_text_getter_rejects_missing_text():
    reranker = build_reranker([0.5])

    candidates = [
        {
            "id": "doc-1",
        }
    ]

    with pytest.raises(ValueError):
        reranker.rerank(
            "query",
            candidates,
        )


def test_reranking_result_contains_expected_fields():
    result = RerankingResult(
        rank=1,
        score=0.95,
        document="document",
    )

    assert result.rank == 1
    assert result.score == pytest.approx(0.95)
    assert result.document == "document"