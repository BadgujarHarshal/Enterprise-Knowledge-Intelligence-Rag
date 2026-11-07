# tests/retrieval/test_rank_fusion.py

import pytest

from app.retrieval.rank_fusion import reciprocal_rank_fusion


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


def test_rrf_combines_results(documents):
    dense_results = [
        {
            "rank": 1,
            "score": 0.9,
            "document": documents[0],
        },
        {
            "rank": 2,
            "score": 0.8,
            "document": documents[1],
        },
    ]

    bm25_results = [
        {
            "rank": 1,
            "score": 5.0,
            "document": documents[1],
        },
        {
            "rank": 2,
            "score": 4.0,
            "document": documents[2],
        },
    ]

    results = reciprocal_rank_fusion(
        [dense_results, bm25_results],
        k=60,
        top_k=3,
    )

    assert len(results) == 3

    returned_ids = [
        result.document["chunk_id"]
        for result in results
    ]

    assert set(returned_ids) == {
        "chunk_001",
        "chunk_002",
        "chunk_003",
    }


def test_rrf_duplicate_documents_are_merged(documents):
    dense_results = [
        {
            "rank": 1,
            "score": 0.9,
            "document": documents[0],
        }
    ]

    bm25_results = [
        {
            "rank": 1,
            "score": 5.0,
            "document": documents[0],
        }
    ]

    results = reciprocal_rank_fusion(
        [dense_results, bm25_results],
        k=60,
        top_k=5,
    )

    assert len(results) == 1

    assert results[0].document["chunk_id"] == "chunk_001"


def test_rrf_invalid_k_raises(documents):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [[{"document": documents[0]}]],
            k=0,
        )


def test_rrf_weight_length_validation(documents):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [
                [{"document": documents[0]}],
                [{"document": documents[1]}],
            ],
            weights=[1.0],
        )