# app\ingestion\chunker.py

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    content: str
    source: str
    chunk_index: int
    metadata: Dict


class SentenceAwareChunker:
    """
    Split documents into sentence-aware chunks.

    The chunker preserves sentence boundaries and creates
    overlapping chunks when a document is longer than the
    configured target size.
    """

    def __init__(
        self,
        target_characters: int = 800,
        max_characters: int = 1200,
        overlap_sentences: int = 1,
    ):
        if target_characters <= 0:
            raise ValueError("target_characters must be positive")

        if max_characters < target_characters:
            raise ValueError(
                "max_characters must be >= target_characters"
            )

        if overlap_sentences < 0:
            raise ValueError(
                "overlap_sentences cannot be negative"
            )

        self.target_characters = target_characters
        self.max_characters = max_characters
        self.overlap_sentences = overlap_sentences

    @staticmethod
    def _split_sentences(content: str) -> List[str]:
        sentences = []

        for line in content.splitlines():
            sentence = line.strip()

            if sentence:
                sentences.append(sentence)

        return sentences

    def chunk_document(self, document) -> List[Chunk]:
        sentences = self._split_sentences(document.content)

        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if (
                current_sentences
                and current_length + sentence_length > self.max_characters
            ):
                chunks.append(
                    self._create_chunk(
                        document=document,
                        sentences=current_sentences,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1

                overlap = current_sentences[
                    -self.overlap_sentences:
                ]

                current_sentences = list(overlap)
                current_length = sum(
                    len(item) for item in current_sentences
                )

            current_sentences.append(sentence)
            current_length += sentence_length

            if current_length >= self.target_characters:
                chunks.append(
                    self._create_chunk(
                        document=document,
                        sentences=current_sentences,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1

                overlap = current_sentences[
                    -self.overlap_sentences:
                ]

                current_sentences = list(overlap)
                current_length = sum(
                    len(item) for item in current_sentences
                )

        if current_sentences:
            chunks.append(
                self._create_chunk(
                    document=document,
                    sentences=current_sentences,
                    chunk_index=chunk_index,
                )
            )

        return chunks

    @staticmethod
    def _create_chunk(
        document,
        sentences: List[str],
        chunk_index: int,
    ) -> Chunk:

        content = " ".join(sentences)

        chunk_id = (
            f"{document.document_id}"
            f"_chunk_{chunk_index:04d}"
        )

        metadata = dict(document.metadata)

        metadata.update(
            {
                "chunk_index": chunk_index,
                "character_count": len(content),
                "sentence_count": len(sentences),
            }
        )

        return Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            title=document.title,
            content=content,
            source=document.source,
            chunk_index=chunk_index,
            metadata=metadata,
        )