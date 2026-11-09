# app/ingestion/chunk_corpus_builder.py

from pathlib import Path
import json

from app.ingestion.chunker import SentenceAwareChunker
from app.ingestion.corpus_builder import convert_record_to_document


class ChunkCorpusBuilder:
    """
    Stream a document JSONL corpus and write chunked
    documents to another JSONL file.
    """

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_characters: int = 800,
        max_characters: int = 1200,
        overlap_sentences: int = 1,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        self.chunker = SentenceAwareChunker(
            target_characters=target_characters,
            max_characters=max_characters,
            overlap_sentences=overlap_sentences,
        )

    def build(self) -> dict:
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Input corpus not found: {self.input_path}"
            )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        documents_processed = 0
        chunks_created = 0

        with (
            self.input_path.open(
                "r",
                encoding="utf-8",
            ) as input_file,
            self.output_path.open(
                "w",
                encoding="utf-8",
            ) as output_file,
        ):
            for line_number, line in enumerate(
                input_file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line "
                        f"{line_number}: "
                        f"{self.input_path}"
                    ) from exc

                try:
                    document = convert_record_to_document(
                        record
                    )
                except (KeyError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid document record on line "
                        f"{line_number}: "
                        f"{self.input_path}"
                    ) from exc

                chunks = self.chunker.chunk_document(
                    document
                )

                for chunk in chunks:
                    record = {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "content": chunk.content,
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata,
                    }

                    output_file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    chunks_created += 1

                documents_processed += 1

                if documents_processed % 10000 == 0:
                    print(
                        f"Processed documents: "
                        f"{documents_processed:,} | "
                        f"Chunks created: "
                        f"{chunks_created:,}"
                    )

        return {
            "documents_processed": documents_processed,
            "chunks_created": chunks_created,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "target_characters": (
                self.chunker.target_characters
            ),
            "max_characters": (
                self.chunker.max_characters
            ),
            "overlap_sentences": (
                self.chunker.overlap_sentences
            ),
        }