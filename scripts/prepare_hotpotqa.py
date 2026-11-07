# scripts\prepare_hotpotqa.py

from pathlib import Path
import json
from datasets import load_dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hotpotqa"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "hotpotqa"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation" / "hotpotqa"


DEVELOPMENT_SIZE = 1000


def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(records, output_path):
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_unique_corpus(dataset_split):
    """
    Extract unique Wikipedia documents from HotpotQA contexts.

    Each context contains:
        - title
        - sentences

    Multiple questions may reference the same document,
    so documents are deduplicated by title.
    """

    documents = {}

    for example in tqdm(
        dataset_split,
        desc="Building unique document corpus"
    ):
        context = example["context"]

        titles = context["title"]
        sentences = context["sentences"]

        for title, document_sentences in zip(titles, sentences):

            if title not in documents:
                documents[title] = {
                    "document_id": f"hotpotqa_{len(documents):08d}",
                    "title": title,
                    "source": "hotpotqa",
                    "content": "\n".join(
                        sentence.strip()
                        for sentence in document_sentences
                        if sentence.strip()
                    ),
                    "metadata": {
                        "dataset": "hotpotqa",
                        "source_type": "wikipedia"
                    }
                }

    return list(documents.values())


def build_evaluation_records(dataset_split):
    """
    Convert HotpotQA examples into our internal evaluation format.
    """

    records = []

    for example in tqdm(
        dataset_split,
        desc="Building evaluation records"
    ):
        context = example["context"]
        supporting_facts = example["supporting_facts"]

        record = {
            "id": example["id"],
            "question": example["question"],
            "answer": example["answer"],
            "type": example["type"],
            "level": example["level"],
            "supporting_facts": {
                "titles": supporting_facts["title"],
                "sentence_ids": supporting_facts["sent_id"]
            },
            "context": {
                "titles": context["title"],
                "sentences": context["sentences"]
            },
            "metadata": {
                "dataset": "hotpotqa"
            }
        }

        records.append(record)

    return records


def main():

    ensure_directories()

    print("=" * 70)
    print("Enterprise Knowledge Intelligence RAG")
    print("Phase 1 - HotpotQA Dataset Preparation")
    print("=" * 70)

    print("\nLoading HotpotQA...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor"
    )

    print("\nDataset loaded:")
    print(dataset)

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    print("\nTrain examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))

    # ---------------------------------------------------------
    # 1. Build unique corpus from training data
    # ---------------------------------------------------------

    print("\nBuilding unique training corpus...")

    train_corpus = build_unique_corpus(train_dataset)

    print(
        f"Unique training documents: {len(train_corpus):,}"
    )

    train_corpus_path = (
        PROCESSED_DIR / "hotpotqa_train_corpus.jsonl"
    )

    save_jsonl(
        train_corpus,
        train_corpus_path
    )

    print(
        f"Saved training corpus to:\n{train_corpus_path}"
    )

    # ---------------------------------------------------------
    # 2. Build development questions
    # ---------------------------------------------------------

    print("\nBuilding development dataset...")

    development_dataset = train_dataset.select(
        range(min(DEVELOPMENT_SIZE, len(train_dataset)))
    )

    development_records = build_evaluation_records(
        development_dataset
    )

    development_path = (
        EVALUATION_DIR / "hotpotqa_development.jsonl"
    )

    save_jsonl(
        development_records,
        development_path
    )

    print(
        f"Development examples: "
        f"{len(development_records):,}"
    )

    print(
        f"Saved development dataset to:\n"
        f"{development_path}"
    )

    # ---------------------------------------------------------
    # 3. Build validation dataset
    # ---------------------------------------------------------

    print("\nBuilding validation dataset...")

    validation_records = build_evaluation_records(
        validation_dataset
    )

    validation_path = (
        EVALUATION_DIR / "hotpotqa_validation.jsonl"
    )

    save_jsonl(
        validation_records,
        validation_path
    )

    print(
        f"Validation examples: "
        f"{len(validation_records):,}"
    )

    print(
        f"Saved validation dataset to:\n"
        f"{validation_path}"
    )

    # ---------------------------------------------------------
    # 4. Save metadata
    # ---------------------------------------------------------

    metadata = {
        "dataset": "HotpotQA",
        "configuration": "distractor",
        "source": "Hugging Face",
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "development_examples": len(development_records),
        "unique_training_documents": len(train_corpus),
        "processing_version": "1.0"
    }

    metadata_path = (
        PROCESSED_DIR / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"\nSaved metadata to:\n{metadata_path}"
    )

    # ---------------------------------------------------------
    # 5. Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1 DATA PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"Training examples       : {len(train_dataset):,}"
    )

    print(
        f"Validation examples     : {len(validation_dataset):,}"
    )

    print(
        f"Development examples    : "
        f"{len(development_records):,}"
    )

    print(
        f"Unique corpus documents : "
        f"{len(train_corpus):,}"
    )

    print("\nGenerated files:")

    print(
        f"1. {train_corpus_path}"
    )

    print(
        f"2. {development_path}"
    )

    print(
        f"3. {validation_path}"
    )

    print(
        f"4. {metadata_path}"
    )


if __name__ == "__main__":
    main()