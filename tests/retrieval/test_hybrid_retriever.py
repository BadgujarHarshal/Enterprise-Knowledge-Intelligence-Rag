# tests/retrieval/test_hybrid_retriever.py

import pytest

from app.retrieval.hybrid_retriever import HybridRetriever


class FakeDenseRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=10):
        return [
            {
                "rank": index + 1,
                "score": 1.0 / (index + 1),
                "document": document,
            }
            for index, document in enumerate(
                self.documents[:top_k]
            )
        ]


class FakeBM25Retriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=10):
        ordered = list(reversed(self.documents))

        return [
            {
                "rank": index + 1,
                "score": float(top_k - index),
                "document": document,
            }
            for index, document in enumerate(
                ordered[:top_k]
            )
        ]


@pytest.fixture
def documents():
    return [
        {
            "chunk_id": "chunk_001",
            "document_id": "doc_001",
            "title": "Python",
            "content": "Python machine learning",
        },
        {
            "chunk_id": "chunk_002",
            "document_id": "doc_002",
            "title": "Paris",
            "content": "Paris France",
        },
        {
            "chunk_id": "chunk_003",
            "document_id": "doc_003",
            "title": "Machine Learning",
            "content": "Machine learning models",
        },
    ]


def test_hybrid_retriever(documents):
    dense = FakeDenseRetriever(documents)
    bm25 = FakeBM25Retriever(documents)

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    results = retriever.retrieve(
        "machine learning",
        top_k=3,
    )

    assert len(results) == 3

    assert results[0].rank == 1

    assert all(
        result.document["chunk_id"]
        for result in results
    )


def test_hybrid_empty_query_raises(documents):
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(documents),
        bm25_retriever=FakeBM25Retriever(documents),
    )

    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_hybrid_invalid_top_k_raises(documents):
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(documents),
        bm25_retriever=FakeBM25Retriever(documents),
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "machine learning",
            top_k=0,
        )


def test_hybrid_invalid_candidate_k_raises(documents):
    retriever = HybridRetriever(
        dense_retriever=FakeDenseRetriever(documents),
        bm25_retriever=FakeBM25Retriever(documents),
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "machine learning",
            candidate_k=0,
        )


def test_hybrid_zero_weights_raise(documents):
    with pytest.raises(ValueError):
        HybridRetriever(
            dense_retriever=FakeDenseRetriever(documents),
            bm25_retriever=FakeBM25Retriever(documents),
            dense_weight=0,
            bm25_weight=0,
        )


def test_hybrid_invalid_rrf_k_raises(documents):
    with pytest.raises(ValueError):
        HybridRetriever(
            dense_retriever=FakeDenseRetriever(documents),
            bm25_retriever=FakeBM25Retriever(documents),
            rrf_k=0,
        )