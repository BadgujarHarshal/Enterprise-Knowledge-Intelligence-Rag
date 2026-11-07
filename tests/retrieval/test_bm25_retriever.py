# tests/retrieval/test_bm25_retriever.py

import pytest

from app.retrieval.bm25_retriever import BM25Retriever


@pytest.fixture
def sample_documents():
    return [
        {
            "chunk_id": "chunk_001",
            "document_id": "doc_001",
            "title": "Python",
            "content": (
                "Python is a programming language "
                "used for machine learning."
            ),
            "source": "test",
            "chunk_index": 0,
            "metadata": {},
        },
        {
            "chunk_id": "chunk_002",
            "document_id": "doc_002",
            "title": "Paris",
            "content": (
                "Paris is the capital city of France."
            ),
            "source": "test",
            "chunk_index": 0,
            "metadata": {},
        },
        {
            "chunk_id": "chunk_003",
            "document_id": "doc_003",
            "title": "Machine Learning",
            "content": (
                "Machine learning uses data to "
                "train predictive models."
            ),
            "source": "test",
            "chunk_index": 0,
            "metadata": {},
        },
    ]


def test_bm25_build_and_retrieve(
    sample_documents,
):
    retriever = BM25Retriever.build(
        sample_documents
    )

    results = retriever.retrieve(
        "machine learning",
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].rank == 1

    assert results[0].document[
        "chunk_id"
    ] in {
        "chunk_001",
        "chunk_003",
    }


def test_bm25_result_order(
    sample_documents,
):
    retriever = BM25Retriever.build(
        sample_documents
    )

    results = retriever.retrieve(
        "capital city France",
        top_k=3,
    )

    assert len(results) == 3

    assert results[0].document[
        "chunk_id"
    ] == "chunk_002"

    assert results[0].score >= results[1].score


def test_empty_query_raises(
    sample_documents,
):
    retriever = BM25Retriever.build(
        sample_documents
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        retriever.retrieve(
            "",
            top_k=5,
        )


def test_invalid_top_k_raises(
    sample_documents,
):
    retriever = BM25Retriever.build(
        sample_documents
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        retriever.retrieve(
            "machine learning",
            top_k=0,
        )


def test_empty_documents_raise():
    with pytest.raises(
        ValueError,
        match="Documents cannot be empty",
    ):
        BM25Retriever.build([])


def test_invalid_document_raises():
    documents = [
        {
            "chunk_id": "chunk_001",
            "title": "Invalid",
            "content": None,
        }
    ]

    with pytest.raises(
        ValueError,
        match="does not contain valid text",
    ):
        BM25Retriever.build(
            documents
        )


def test_bm25_save_and_load(
    sample_documents,
    tmp_path,
):
    retriever = BM25Retriever.build(
        sample_documents
    )

    index_path = (
        tmp_path /
        "bm25_index"
    )

    retriever.save(
        index_path
    )

    loaded_retriever = (
        BM25Retriever.load(
            index_path,
            sample_documents,
        )
    )

    results = loaded_retriever.retrieve(
        "capital city France",
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].document[
        "chunk_id"
    ] == "chunk_002"


def test_bm25_load_missing_directory(
    sample_documents,
    tmp_path,
):
    missing_path = (
        tmp_path /
        "does_not_exist"
    )

    with pytest.raises(
        FileNotFoundError,
        match="BM25 index directory not found",
    ):
        BM25Retriever.load(
            missing_path,
            sample_documents,
        )