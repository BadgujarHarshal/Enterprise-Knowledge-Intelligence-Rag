# scripts/evaluate_hybrid.py

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import bm25s

from app.models.embedding_model import EmbeddingModel
from app.retrieval.faiss_index import FAISSIndex
from app.retrieval.retriever import DenseRetriever
from app.retrieval.rank_fusion import reciprocal_rank_fusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_EVALUATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "hotpotqa"
    / "hotpotqa_validation_development.jsonl"
)

DEFAULT_CHUNKS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hotpotqa"
    / "hotpotqa_chunks.jsonl"
)

DEFAULT_FAISS = (
    PROJECT_ROOT
    / "models"
    / "embeddings"
    / "hotpotqa_faiss.index"
)

DEFAULT_EMBEDDINGS_METADATA = (
    PROJECT_ROOT
    / "models"
    / "embeddings"
    / "hotpotqa_chunk_metadata.json"
)

DEFAULT_BM25 = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hotpotqa"
    / "bm25_index"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "hotpotqa_hybrid_baseline.json"
)

DEFAULT_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    "Evaluation record must be a dictionary."
                )

            records.append(record)

    return records


def extract_supporting_titles(
    record: Dict[str, Any],
) -> Set[str]:

    supporting_facts = record.get(
        "supporting_facts"
    )

    if not isinstance(
        supporting_facts,
        dict,
    ):
        raise ValueError(
            "Invalid supporting_facts."
        )

    titles = supporting_facts.get(
        "titles"
    )

    if not isinstance(
        titles,
        list,
    ):
        raise ValueError(
            "supporting_facts titles must be a list."
        )

    return {
        str(title)
        for title in titles
    }


def evaluate(
    dense_retriever,
    bm25_retriever,
    records,
    *,
    dense_weight: float,
    bm25_weight: float,
    rrf_k: int,
    candidate_k: int,
    top_ks: List[int],
):

    max_k = max(top_ks)

    hit_counts = {
        k: 0
        for k in top_ks
    }

    coverage_sums = {
        k: 0.0
        for k in top_ks
    }

    mrr_sums = {
        k: 0.0
        for k in top_ks
    }

    start = time.perf_counter()

    for index, record in enumerate(
        records,
        start=1,
    ):

        question = record["question"]

        gold_titles = extract_supporting_titles(
            record
        )

        dense_results = dense_retriever.retrieve(
            question,
            top_k=candidate_k,
        )

        bm25_results = bm25_retriever.retrieve(
            question,
            top_k=candidate_k,
        )

        fused = reciprocal_rank_fusion(
            [
                dense_results,
                bm25_results,
            ],
            k=rrf_k,
            weights=[
                dense_weight,
                bm25_weight,
            ],
            top_k=max_k,
        )

        ranked_titles = [
            str(result.document["title"])
            for result in fused
            if result.document.get("title")
        ]

        for k in top_ks:

            retrieved = set(
                ranked_titles[:k]
            )

            overlap = (
                retrieved
                & gold_titles
            )

            if overlap:
                hit_counts[k] += 1

            coverage_sums[k] += (
                len(overlap)
                / len(gold_titles)
            )

            for rank, title in enumerate(
                ranked_titles[:k],
                start=1,
            ):

                if title in gold_titles:
                    mrr_sums[k] += (
                        1.0 / rank
                    )
                    break

        if index % 50 == 0:
            print(
                f"Evaluated "
                f"{index}/{len(records)}"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    metrics = {}

    for k in top_ks:

        metrics[f"hit@{k}"] = (
            hit_counts[k]
            / len(records)
        )

        metrics[f"coverage@{k}"] = (
            coverage_sums[k]
            / len(records)
        )

        metrics[f"mrr@{k}"] = (
            mrr_sums[k]
            / len(records)
        )

    return {
        "question_count": len(records),
        "candidate_k": candidate_k,
        "rrf_k": rrf_k,
        "dense_weight": dense_weight,
        "bm25_weight": bm25_weight,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Evaluate hybrid dense + BM25 retrieval."
    )

    parser.add_argument(
        "--evaluation-file",
        type=Path,
        default=DEFAULT_EVALUATION_FILE,
    )

    parser.add_argument(
        "--bm25",
        type=Path,
        default=DEFAULT_BM25,
    )

    parser.add_argument(
        "--faiss",
        type=Path,
        default=DEFAULT_FAISS,
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_EMBEDDINGS_METADATA,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--dense-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    records = load_jsonl(
        args.evaluation_file
    )

    if len(records) != 560:
        raise ValueError(
            "Expected exactly 560 frozen "
            f"evaluation records, found {len(records)}."
        )

    print("=" * 70)
    print("EKI-RAG HYBRID RETRIEVAL")
    print("=" * 70)

    print("Loading BM25...")

    bm25 = bm25s.BM25.load(
        str(args.bm25),
        load_corpus=True,
    )

    print("Loading embedding model...")

    embedding_model = EmbeddingModel(
        model_name=args.model
    )

    print("Loading FAISS index...")

    faiss_index = FAISSIndex.load(
        str(args.faiss)
    )

    with args.metadata.open(
        "r",
        encoding="utf-8",
    ) as file:
        chunk_metadata = json.load(file)

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        index=faiss_index,
        metadata=chunk_metadata,
    )

    result = evaluate(
        dense_retriever,
        bm25,
        records,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
        rrf_k=args.rrf_k,
        candidate_k=args.candidate_k,
        top_ks=[5, 10, 20],
    )

    output = {
        "project": "EKI-RAG",
        "dataset": "HotpotQA",
        "retriever": "Hybrid Dense + BM25",
        "fusion": "Reciprocal Rank Fusion",
        "evaluation_protocol": (
            "Frozen 560-question fully-evaluable "
            "HotpotQA validation subset"
        ),
        **result,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("HYBRID RESULTS")
    print("=" * 70)

    for metric, value in result[
        "metrics"
    ].items():

        print(
            f"{metric:15s}: {value:.4f}"
        )

    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()