# scripts/build_chunk_corpus.py

from pathlib import Path
import json
import sys


# Resolve the project root:
# Enterprise Knowledge Intelligence Rag/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make the project root available for imports such as:
# from app.ingestion.chunk_corpus_builder import ChunkCorpusBuilder
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.ingestion.chunk_corpus_builder import (
    ChunkCorpusBuilder,
)


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hotpotqa"
    / "hotpotqa_train_corpus.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hotpotqa"
    / "hotpotqa_chunks.jsonl"
)


def main():
    print("Project root:")
    print(PROJECT_ROOT)

    print("\nInput corpus:")
    print(INPUT_PATH)

    print("\nOutput corpus:")
    print(OUTPUT_PATH)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input corpus not found: {INPUT_PATH}"
        )

    builder = ChunkCorpusBuilder(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        target_characters=800,
        max_characters=1200,
        overlap_sentences=1,
    )

    summary = builder.build()

    print("\nChunking completed.")
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()