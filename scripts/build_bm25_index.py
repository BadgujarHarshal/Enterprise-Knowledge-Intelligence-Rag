# scripts/build_bm25_index.py

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterator

import bm25s


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CHUNKS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hotpotqa"
    / "hotpotqa_chunks.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hotpotqa"
    / "bm25_index"
)

DEFAULT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hotpotqa"
    / "bm25_metadata.json"
)


def iter_chunks(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(document, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object."
                )

            yield document


def load_documents(path: Path) -> list[Dict]:
    documents = list(iter_chunks(path))

    if not documents:
        raise ValueError(
            f"No documents found in {path}"
        )

    for index, document in enumerate(documents):
        if not isinstance(document.get("content"), str):
            raise ValueError(
                f"Document {index} has invalid 'content'."
            )

        if not document.get("chunk_id"):
            raise ValueError(
                f"Document {index} has no chunk_id."
            )

    return documents


def build_index(
    chunks_path: Path,
    output_dir: Path,
    metadata_path: Path,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> None:

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk corpus not found: {chunks_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("EKI-RAG BM25 INDEX BUILD")
    print("=" * 70)

    print(f"Chunk corpus : {chunks_path}")
    print(f"Output       : {output_dir}")
    print(f"k1           : {k1}")
    print(f"b            : {b}")
    print()

    start_time = time.perf_counter()

    print("Loading chunk corpus...")

    documents = load_documents(chunks_path)

    print(
        f"Loaded {len(documents):,} chunks."
    )

    contents = [
        document["content"]
        for document in documents
    ]

    print("Tokenizing corpus...")

    tokenized_corpus = bm25s.tokenize(
        contents,
        stopwords="english",
        show_progress=True,
    )

    print("Building BM25 index...")

    retriever = bm25s.BM25(
        k1=k1,
        b=b,
    )

    retriever.index(
        tokenized_corpus,
        show_progress=True,
    )

    print("Saving BM25 index...")

    retriever.save(
        str(output_dir),
        corpus=documents,
    )

    elapsed = time.perf_counter() - start_time

    metadata = {
        "project": "EKI-RAG",
        "dataset": "HotpotQA",
        "config": "distractor",
        "retriever": "BM25",
        "library": "bm25s",
        "chunk_count": len(documents),
        "k1": k1,
        "b": b,
        "source_chunks": str(chunks_path),
        "index_directory": str(output_dir),
        "elapsed_seconds": elapsed,
    }

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("BM25 BUILD COMPLETE")
    print("=" * 70)
    print(
        f"Documents : {len(documents):,}"
    )
    print(
        f"Time      : {elapsed:.2f} seconds"
    )
    print(
        f"Index     : {output_dir}"
    )
    print(
        f"Metadata  : {metadata_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BM25 index for EKI-RAG."
    )

    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
    )

    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
    )

    args = parser.parse_args()

    build_index(
        chunks_path=args.chunks,
        output_dir=args.output,
        metadata_path=args.metadata,
        k1=args.k1,
        b=args.b,
    )


if __name__ == "__main__":
    main()