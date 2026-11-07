# tests\ingestion\test_chunker.py

from types import SimpleNamespace

from app.ingestion.chunker import SentenceAwareChunker


def test_short_document_creates_chunk():
    document = SimpleNamespace(
        document_id="doc_001",
        title="Test Document",
        content=(
            "This is the first sentence.\n"
            "This is the second sentence."
        ),
        source="test",
        metadata={
            "dataset": "hotpotqa"
        },
    )

    chunker = SentenceAwareChunker(
        target_characters=100,
        max_characters=200,
        overlap_sentences=1,
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) >= 1
    assert chunks[0].document_id == "doc_001"
    assert chunks[0].title == "Test Document"
    assert chunks[0].content
    assert chunks[0].chunk_id.startswith("doc_001_chunk_")


def test_long_document_creates_multiple_chunks():
    document = SimpleNamespace(
        document_id="doc_002",
        title="Long Document",
        content=(
            "Sentence one.\n"
            "Sentence two.\n"
            "Sentence three.\n"
            "Sentence four.\n"
            "Sentence five.\n"
            "Sentence six."
        ),
        source="test",
        metadata={
            "dataset": "hotpotqa"
        },
    )

    chunker = SentenceAwareChunker(
        target_characters=30,
        max_characters=45,
        overlap_sentences=1,
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.content
        assert chunk.document_id == "doc_002"
        assert "chunk_index" in chunk.metadata
        assert "character_count" in chunk.metadata