# tests\ingestion\test_chunk_corpus_builder.py

import json

from app.ingestion.chunk_corpus_builder import (
    ChunkCorpusBuilder,
)


def test_chunk_corpus_builder(tmp_path):
    input_path = (
        tmp_path / "input.jsonl"
    )

    output_path = (
        tmp_path / "output.jsonl"
    )

    documents = [
        {
            "document_id": "doc_001",
            "title": "Test Document",
            "source": "test",
            "content": (
                "Sentence one.\n"
                "Sentence two.\n"
                "Sentence three."
            ),
            "metadata": {
                "dataset": "test",
            },
        },
        {
            "document_id": "doc_002",
            "title": "Second Document",
            "source": "test",
            "content": (
                "Another sentence.\n"
                "Another sentence again."
            ),
            "metadata": {
                "dataset": "test",
            },
        },
    ]

    with input_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for document in documents:
            file.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
                + "\n"
            )

    builder = ChunkCorpusBuilder(
        input_path=input_path,
        output_path=output_path,
        target_characters=50,
        max_characters=80,
        overlap_sentences=1,
    )

    summary = builder.build()

    assert summary["documents_processed"] == 2
    assert summary["chunks_created"] > 0
    assert output_path.exists()

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    assert len(records) > 0

    first = records[0]

    assert "chunk_id" in first
    assert "document_id" in first
    assert "title" in first
    assert "content" in first
    assert "metadata" in first