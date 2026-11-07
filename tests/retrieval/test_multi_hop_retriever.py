# tests/retrieval/test_multi_hop_retriever.py

from dataclasses import dataclass

import pytest

from app.retrieval.multi_hop_retriever import (
    MultiHopRetriever,
)


@dataclass
class FakeResult:
    rank: int
    score: float
    document: dict


class FakeRetriever:
    """
    Deterministic retriever used for Phase 6 unit tests.
    """

    def __init__(
        self,
        responses,
    ):
        self.responses = responses
        self.calls = []

    def retrieve(
        self,
        query,
        top_k=10,
    ):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
            }
        )

        return self.responses.get(
            query,
            [],
        )[:top_k]


@pytest.fixture
def documents():
    return {
        "a": {
            "chunk_id": "chunk_a",
            "document_id": "doc_a",
            "title": "Entity A",
            "content": "Entity A information.",
        },
        "b": {
            "chunk_id": "chunk_b",
            "document_id": "doc_b",
            "title": "Entity B",
            "content": "Entity B information.",
        },
        "c": {
            "chunk_id": "chunk_c",
            "document_id": "doc_c",
            "title": "Entity C",
            "content": "Entity C information.",
        },
        "d": {
            "chunk_id": "chunk_d",
            "document_id": "doc_d",
            "title": "Entity D",
            "content": "Entity D information.",
        },
    }


def test_two_hop_retrieval(
    documents,
):
    original_query = (
        "How are Entity A and Entity B connected?"
    )

    expanded_query = (
        f"{original_query} Entity A"
    )

    retriever = FakeRetriever(
        {
            original_query: [
                FakeResult(
                    rank=1,
                    score=1.0,
                    document=documents["a"],
                ),
                FakeResult(
                    rank=2,
                    score=0.8,
                    document=documents["c"],
                ),
            ],
            expanded_query: [
                FakeResult(
                    rank=1,
                    score=0.9,
                    document=documents["b"],
                ),
            ],
        }
    )

    multi_hop = MultiHopRetriever(
        retriever,
        max_hops=2,
        candidates_per_hop=10,
        expansion_seeds=1,
    )

    results = multi_hop.retrieve(
        original_query,
        top_k=10,
    )

    assert len(results) == 3

    chunk_ids = [
        result.document["chunk_id"]
        for result in results
    ]

    assert "chunk_a" in chunk_ids
    assert "chunk_b" in chunk_ids
    assert "chunk_c" in chunk_ids

    assert any(
        result.hop == 2
        for result in results
    )


def test_second_hop_query_uses_top_evidence_title(
    documents,
):
    original_query = (
        "How are Entity A and Entity B connected?"
    )

    expanded_query = (
        f"{original_query} Entity A"
    )

    retriever = FakeRetriever(
        {
            original_query: [
                FakeResult(
                    rank=1,
                    score=1.0,
                    document=documents["a"],
                )
            ],
            expanded_query: [
                FakeResult(
                    rank=1,
                    score=0.9,
                    document=documents["b"],
                )
            ],
        }
    )

    multi_hop = MultiHopRetriever(
        retriever,
        max_hops=2,
        expansion_seeds=1,
    )

    multi_hop.retrieve(
        original_query
    )

    assert len(
        retriever.calls
    ) == 2

    assert (
        retriever.calls[0]["query"]
        == original_query
    )

    assert (
        retriever.calls[1]["query"]
        == expanded_query
    )


def test_max_hops_one_does_not_expand(
    documents,
):
    query = "Entity A"

    retriever = FakeRetriever(
        {
            query: [
                FakeResult(
                    rank=1,
                    score=1.0,
                    document=documents["a"],
                )
            ]
        }
    )

    multi_hop = MultiHopRetriever(
        retriever,
        max_hops=1,
    )

    results = multi_hop.retrieve(
        query
    )

    assert len(results) == 1

    assert len(
        retriever.calls
    ) == 1


def test_duplicate_documents_are_merged(
    documents,
):
    query = "Entity A"

    expanded_query = (
        f"{query} Entity A"
    )

    retriever = FakeRetriever(
        {
            query: [
                FakeResult(
                    rank=1,
                    score=1.0,
                    document=documents["a"],
                )
            ],
            expanded_query: [
                FakeResult(
                    rank=1,
                    score=0.5,
                    document=documents["a"],
                )
            ],
        }
    )

    multi_hop = MultiHopRetriever(
        retriever,
        max_hops=2,
        expansion_seeds=1,
    )

    results = multi_hop.retrieve(
        query,
        top_k=10,
    )

    assert len(results) == 1

    assert results[0].document[
        "chunk_id"
    ] == "chunk_a"

    assert results[0].hop_history == (
        1,
        2,
    )

    assert results[0].score == pytest.approx(
        1.4
    )


def test_hop_decay_is_applied(
    documents,
):
    query = "Entity A"

    expanded_query = (
        f"{query} Entity A"
    )

    retriever = FakeRetriever(
        {
            query: [
                FakeResult(
                    rank=1,
                    score=1.0,
                    document=documents["a"],
                )
            ],
            expanded_query: [
                FakeResult(
                    rank=1,
                    score=1.0,
                    document=documents["b"],
                )
            ],
        }
    )

    multi_hop = MultiHopRetriever(
        retriever,
        max_hops=2,
        expansion_seeds=1,
        hop_decay=0.5,
    )

    results = multi_hop.retrieve(
        query,
        top_k=10,
    )

    scores = {
        result.document["chunk_id"]: result.score
        for result in results
    }

    assert scores["chunk_a"] == pytest.approx(
        1.0
    )

    assert scores["chunk_b"] == pytest.approx(
        0.5
    )


def test_expansion_seed_limit(
    documents,
):
    query = "Main question"

    expanded_a = (
        f"{query} Entity A"
    )

    expanded_c = (
        f"{query} Entity C"
    )

    retriever = FakeRetriever(
        {
            query: [
                FakeResult(
                    rank=1,
                    score=1.0,
                    document=documents["a"],
                ),
                FakeResult(
                    rank=2,
                    score=0.9,
                    document=documents["c"],
                ),
            ],
            expanded_a: [
                FakeResult(
                    rank=1,
                    score=0.8,
                    document=documents["b"],
                )
            ],
            expanded_c: [
                FakeResult(
                    rank=1,
                    score=0.8,
                    document=documents["d"],
                )
            ],
        }
    )

    multi_hop = MultiHopRetriever(
        retriever,
        max_hops=2,
        expansion_seeds=1,
    )

    multi_hop.retrieve(
        query
    )

    called_queries = [
        call["query"]
        for call in retriever.calls
    ]

    assert expanded_a in called_queries

    assert expanded_c not in called_queries


def test_top_k_limits_final_results(
    documents,
):
    query = "Entity question"

    retriever = FakeRetriever(
        {
            query: [
                FakeResult(
                    rank=1,
                    score=1.0,
                    document=documents["a"],
                ),
                FakeResult(
                    rank=2,
                    score=0.9,
                    document=documents["b"],
                ),
                FakeResult(
                    rank=3,
                    score=0.8,
                    document=documents["c"],
                ),
            ]
        }
    )

    multi_hop = MultiHopRetriever(
        retriever,
        max_hops=1,
    )

    results = multi_hop.retrieve(
        query,
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].rank == 1
    assert results[1].rank == 2


def test_results_are_ranked_by_aggregated_score(
    documents,
):
    query = "Entity question"

    retriever = FakeRetriever(
        {
            query: [
                FakeResult(
                    rank=1,
                    score=0.2,
                    document=documents["a"],
                ),
                FakeResult(
                    rank=2,
                    score=0.9,
                    document=documents["b"],
                ),
            ]
        }
    )

    multi_hop = MultiHopRetriever(
        retriever,
        max_hops=1,
    )

    results = multi_hop.retrieve(
        query
    )

    assert (
        results[0].document["chunk_id"]
        == "chunk_b"
    )

    assert (
        results[1].document["chunk_id"]
        == "chunk_a"
    )


def test_empty_retrieval_returns_empty(
    documents,
):
    query = "unknown"

    retriever = FakeRetriever(
        {
            query: []
        }
    )

    multi_hop = MultiHopRetriever(
        retriever,
        max_hops=2,
    )

    results = multi_hop.retrieve(
        query
    )

    assert results == []


def test_empty_query_raises():
    retriever = FakeRetriever({})

    multi_hop = MultiHopRetriever(
        retriever
    )

    with pytest.raises(ValueError):
        multi_hop.retrieve("")


def test_non_string_query_raises():
    retriever = FakeRetriever({})

    multi_hop = MultiHopRetriever(
        retriever
    )

    with pytest.raises(TypeError):
        multi_hop.retrieve(None)


def test_invalid_top_k_raises():
    retriever = FakeRetriever({})

    multi_hop = MultiHopRetriever(
        retriever
    )

    with pytest.raises(ValueError):
        multi_hop.retrieve(
            "test",
            top_k=0,
        )


def test_invalid_max_hops_raises():
    retriever = FakeRetriever({})

    with pytest.raises(ValueError):
        MultiHopRetriever(
            retriever,
            max_hops=0,
        )


def test_invalid_candidates_per_hop_raises():
    retriever = FakeRetriever({})

    with pytest.raises(ValueError):
        MultiHopRetriever(
            retriever,
            candidates_per_hop=0,
        )


def test_invalid_expansion_seeds_raises():
    retriever = FakeRetriever({})

    with pytest.raises(ValueError):
        MultiHopRetriever(
            retriever,
            expansion_seeds=0,
        )


def test_invalid_hop_decay_raises():
    retriever = FakeRetriever({})

    with pytest.raises(ValueError):
        MultiHopRetriever(
            retriever,
            hop_decay=0.0,
        )


def test_missing_chunk_id_raises(
    documents,
):
    query = "test"

    invalid_document = {
        "document_id": "doc_invalid",
        "title": "Invalid",
        "content": "Invalid content.",
    }

    retriever = FakeRetriever(
        {
            query: [
                FakeResult(
                    rank=1,
                    score=1.0,
                    document=invalid_document,
                )
            ]
        }
    )

    multi_hop = MultiHopRetriever(
        retriever
    )

    with pytest.raises(ValueError):
        multi_hop.retrieve(query)


def test_object_and_dictionary_results_are_supported(
    documents,
):
    query = "test"

    retriever = FakeRetriever(
        {
            query: [
                {
                    "rank": 1,
                    "score": 1.0,
                    "document": documents["a"],
                }
            ]
        }
    )

    multi_hop = MultiHopRetriever(
        retriever,
        max_hops=1,
    )

    results = multi_hop.retrieve(
        query
    )

    assert len(results) == 1

    assert (
        results[0].document["chunk_id"]
        == "chunk_a"
    )