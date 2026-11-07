# tests\ingestion\test_dataset_loader.py

from pathlib import Path

from app.ingestion.dataset_loader import JSONLLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hotpotqa"
    / "hotpotqa_train_corpus.jsonl"
)


def test_corpus_exists():
    assert CORPUS_PATH.exists()


def test_corpus_loader():
    loader = JSONLLoader(CORPUS_PATH)

    records = loader.load()

    assert len(records) > 0

    first_record = records[0]

    assert "document_id" in first_record
    assert "title" in first_record
    assert "content" in first_record
    assert "source" in first_record