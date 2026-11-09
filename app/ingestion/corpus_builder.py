# app\ingestion\corpus_builder.py

from dataclasses import dataclass


@dataclass
class Document:
    document_id: str
    title: str
    content: str
    source: str
    metadata: dict


def convert_record_to_document(record: dict) -> Document:
    """
    Convert our normalized corpus record into
    the internal Document representation.
    """

    return Document(
        document_id=record["document_id"],
        title=record["title"],
        content=record["content"],
        source=record["source"],
        metadata=record.get("metadata", {})
    )