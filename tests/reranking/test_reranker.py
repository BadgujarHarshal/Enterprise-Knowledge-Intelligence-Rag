# tests/reranking/test_reranker.py

import numpy as np
import pytest

from app.reranking.cross_encoder import (
    CrossEncoderModel,
)
from app.reranking.reranker import (
    CrossEncoderReranker,
    RerankedResult,
)


class FakeCrossEncoderModel:
    """
    Deterministic fake model for unit testing.

    It assigns scores based on the document content.
    """

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def predict(
        self,
        pairs,
        *,
        batch_size=32,
        show_progress_bar=False,
    ):
        self.calls.append(
            {
                "pairs": list(pairs),
                "batch_size": batch_size,
                "show_progress_bar": (
                    show_progress_bar
                ),
            }
        )

        if len(pairs) != len(self.scores):
            raise AssertionError(
                "Unexpected number of pairs."
            )

        return np.asarray(
            self.scores,
            dtype=np.float32,
        )


class FakeCrossEncoderWrapper(
    CrossEncoderModel
):
    """
    CrossEncoderModel-compatible fake wrapper.

    We intentionally bypass actual model loading.
    """

    def __init__(self, scores):
        super().__init__()
        self.fake_scores = list(scores)
        self.calls = []

    def predict(
        self,
        pairs,
        *,
        batch_size=32,
        show_progress_bar=False,
    ):
        self.calls.append(
            {
                "pairs": list(pairs),
                "batch_size": batch_size,
                "show_progress_bar": (
                    show_progress_bar
                ),
            }
        )

        if len(pairs) != len(
            self.fake_scores
        ):
            raise AssertionError(
                "Unexpected number of pairs."
            )

        return np.asarray(
            self.fake_scores,
            dtype=np.float32,
        )


@pytest.fixture
def documents():
    return [
        {
            "chunk_id": "chunk_001",
            "document_id": "doc_001",
            "title": "Python",
            "content": "Python programming language.",
        },
        {
            "chunk_id": "chunk_002",
            "document_id": "doc_002",
            "title": "Paris",
            "content": "Paris is the capital of France.",
        },
        {
            "chunk_id": "chunk_003",
            "document_id": "doc_003",
            "title": "Machine Learning",
            "content": "Machine learning uses data.",
        },
    ]


@pytest.fixture
def candidates(documents):
    return [
        {
            "rank": 1,
            "score": 0.90,
            "document": documents[0],
        },
        {
            "rank": 2,
            "score": 0.80,
            "document": documents[1],
        },
        {
            "rank": 3,
            "score": 0.70,
            "document": documents[2],
        },
    ]


def test_reranker_sorts_by_cross_encoder_score(
    candidates,
):
    model = FakeCrossEncoderWrapper(
        scores=[
            0.10,
            0.90,
            0.50,
        ]
    )

    reranker = CrossEncoderReranker(
        model=model,
        batch_size=16,
    )

    results = reranker.rerank(
        "Where is Paris?",
        candidates,
    )

    assert len(results) == 3

    assert results[0].rank == 1
    assert results[0].document[
        "chunk_id"
    ] == "chunk_002"

    assert results[0].score == pytest.approx(
        0.90
    )

    assert results[1].document[
        "chunk_id"
    ] == "chunk_003"

    assert results[2].document[
        "chunk_id"
    ] == "chunk_001"


def test_reranker_preserves_original_retrieval_rank(
    candidates,
):
    model = FakeCrossEncoderWrapper(
        scores=[
            0.20,
            0.80,
            0.40,
        ]
    )

    reranker = CrossEncoderReranker(
        model=model,
    )

    results = reranker.rerank(
        "test query",
        candidates,
    )

    assert results[0].document[
        "chunk_id"
    ] == "chunk_002"

    assert results[0].retrieval_rank == 2

    assert results[1].document[
        "chunk_id"
    ] == "chunk_003"

    assert results[1].retrieval_rank == 3

    assert results[2].document[
        "chunk_id"
    ] == "chunk_001"

    assert results[2].retrieval_rank == 1


def test_reranker_uses_query_and_content(
    candidates,
):
    model = FakeCrossEncoderWrapper(
        scores=[
            0.1,
            0.2,
            0.3,
        ]
    )

    reranker = CrossEncoderReranker(
        model=model,
    )

    query = "Where is Paris?"

    reranker.rerank(
        query,
        candidates,
    )

    assert len(model.calls) == 1

    pairs = model.calls[0]["pairs"]

    assert pairs[0] == (
        query,
        "Python programming language.",
    )

    assert pairs[1] == (
        query,
        "Paris is the capital of France.",
    )

    assert pairs[2] == (
        query,
        "Machine learning uses data.",
    )


def test_reranker_top_k(
    candidates,
):
    model = FakeCrossEncoderWrapper(
        scores=[
            0.30,
            0.90,
            0.70,
        ]
    )

    reranker = CrossEncoderReranker(
        model=model,
    )

    results = reranker.rerank(
        "test query",
        candidates,
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].document[
        "chunk_id"
    ] == "chunk_002"

    assert results[1].document[
        "chunk_id"
    ] == "chunk_003"

    assert results[0].rank == 1
    assert results[1].rank == 2


def test_reranker_tie_preserves_retrieval_order(
    candidates,
):
    model = FakeCrossEncoderWrapper(
        scores=[
            0.50,
            0.50,
            0.20,
        ]
    )

    reranker = CrossEncoderReranker(
        model=model,
    )

    results = reranker.rerank(
        "test query",
        candidates,
    )

    assert results[0].document[
        "chunk_id"
    ] == "chunk_001"

    assert results[1].document[
        "chunk_id"
    ] == "chunk_002"

    assert results[2].document[
        "chunk_id"
    ] == "chunk_003"


def test_reranker_empty_candidates_returns_empty():
    model = FakeCrossEncoderWrapper(
        scores=[]
    )

    reranker = CrossEncoderReranker(
        model=model,
    )

    results = reranker.rerank(
        "test query",
        [],
    )

    assert results == []

    assert model.calls == []


def test_reranker_empty_query_raises(
    candidates,
):
    model = FakeCrossEncoderWrapper(
        scores=[
            0.1,
            0.2,
            0.3,
        ]
    )

    reranker = CrossEncoderReranker(
        model=model,
    )

    with pytest.raises(ValueError):
        reranker.rerank(
            "",
            candidates,
        )


def test_reranker_non_string_query_raises(
    candidates,
):
    model = FakeCrossEncoderWrapper(
        scores=[
            0.1,
            0.2,
            0.3,
        ]
    )

    reranker = CrossEncoderReranker(
        model=model,
    )

    with pytest.raises(TypeError):
        reranker.rerank(
            None,
            candidates,
        )


def test_reranker_invalid_top_k_raises(
    candidates,
):
    model = FakeCrossEncoderWrapper(
        scores=[
            0.1,
            0.2,
            0.3,
        ]
    )

    reranker = CrossEncoderReranker(
        model=model,
    )

    with pytest.raises(ValueError):
        reranker.rerank(
            "test query",
            candidates,
            top_k=0,
        )


def test_reranker_invalid_candidate_content_raises():
    candidates = [
        {
            "rank": 1,
            "score": 0.5,
            "document": {
                "chunk_id": "chunk_001",
                "title": "Invalid",
            },
        }
    ]

    model = FakeCrossEncoderWrapper(
        scores=[0.5]
    )

    reranker = CrossEncoderReranker(
        model=model,
    )

    with pytest.raises(ValueError):
        reranker.rerank(
            "test query",
            candidates,
        )


def test_reranker_supports_object_results(
    documents,
):
    class RetrievalObject:
        def __init__(
            self,
            rank,
            score,
            document,
        ):
            self.rank = rank
            self.score = score
            self.document = document

    candidates = [
        RetrievalObject(
            rank=1,
            score=0.9,
            document=documents[0],
        ),
        RetrievalObject(
            rank=2,
            score=0.8,
            document=documents[1],
        ),
    ]

    model = FakeCrossEncoderWrapper(
        scores=[
            0.2,
            0.8,
        ]
    )

    reranker = CrossEncoderReranker(
        model=model,
    )

    results = reranker.rerank(
        "test query",
        candidates,
    )

    assert len(results) == 2

    assert results[0].document[
        "chunk_id"
    ] == "chunk_002"

    assert results[0].retrieval_rank == 2


def test_cross_encoder_model_does_not_load_on_init():
    model = CrossEncoderModel()

    assert model.is_loaded is False


def test_cross_encoder_model_validates_max_length():
    with pytest.raises(ValueError):
        CrossEncoderModel(
            max_length=0
        )


def test_cross_encoder_model_validates_model_name():
    with pytest.raises(ValueError):
        CrossEncoderModel(
            model_name="   "
        )


def test_cross_encoder_model_empty_predict_returns_empty():
    model = CrossEncoderModel()

    scores = model.predict([])

    assert isinstance(
        scores,
        np.ndarray,
    )

    assert scores.dtype == np.float32

    assert len(scores) == 0