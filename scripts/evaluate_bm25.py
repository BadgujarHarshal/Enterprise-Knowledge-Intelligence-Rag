# scripts/evaluate_bm25.py

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import bm25s


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

DEFAULT_INDEX = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hotpotqa"
    / "bm25_index"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "hotpotqa_bm25_baseline.json"
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {path}"
        )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid record at line {line_number}."
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"No evaluation records found in {path}"
        )

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
            "supporting_facts must be a dictionary."
        )

    titles = supporting_facts.get("titles")

    if not isinstance(titles, list):
        raise ValueError(
            "supporting_facts['titles'] must be a list."
        )

    return {
        str(title)
        for title in titles
        if title is not None
    }


def evaluate(
    retriever: bm25s.BM25,
    evaluation_records: List[Dict[str, Any]],
    *,
    top_ks: Iterable[int],
) -> Dict[str, Any]:

    top_ks = sorted(set(top_ks))
    max_k = max(top_ks)

    hit_counts = {k: 0 for k in top_ks}
    coverage_sums = {k: 0.0 for k in top_ks}
    mrr_sums = {k: 0.0 for k in top_ks}

    question_count = len(evaluation_records)

    total_supporting_titles = 0

    start_time = time.perf_counter()

    for question_index, record in enumerate(
        evaluation_records,
        start=1,
    ):

        question = record.get("question")

        if not isinstance(question, str):
            raise ValueError(
                f"Invalid question at index {question_index - 1}."
            )

        gold_titles = extract_supporting_titles(
            record
        )

        if not gold_titles:
            continue

        total_supporting_titles += len(
            gold_titles
        )

        query_tokens = bm25s.tokenize(
            [question],
            stopwords="english",
            show_progress=False,
        )

        results, _scores = retriever.retrieve(
            query_tokens,
            k=max_k,
            show_progress=False,
        )

        ranked_documents = results[0]

        ranked_titles = []

        for document in ranked_documents:
            if isinstance(document, dict):
                title = document.get("title")
            else:
                title = None

            if title is not None:
                ranked_titles.append(str(title))

        for k in top_ks:

            retrieved_titles = set(
                ranked_titles[:k]
            )

            overlap = (
                retrieved_titles
                & gold_titles
            )

            if overlap:
                hit_counts[k] += 1

            coverage_sums[k] += (
                len(overlap)
                / len(gold_titles)
            )

            first_support_rank = None

            for rank, title in enumerate(
                ranked_titles[:k],
                start=1,
            ):
                if title in gold_titles:
                    first_support_rank = rank
                    break

            if first_support_rank is not None:
                mrr_sums[k] += (
                    1.0 / first_support_rank
                )

        if question_index % 100 == 0:
            print(
                f"Evaluated "
                f"{question_index:,}/"
                f"{question_count:,}"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    metrics = {}

    for k in top_ks:
        metrics[f"hit@{k}"] = (
            hit_counts[k]
            / question_count
        )

        metrics[f"coverage@{k}"] = (
            coverage_sums[k]
            / question_count
        )

        metrics[f"mrr@{k}"] = (
            mrr_sums[k]
            / question_count
        )

    return {
        "question_count": question_count,
        "total_supporting_titles": total_supporting_titles,
        "top_k": top_ks,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Evaluate BM25 on EKI-RAG HotpotQA validation."
    )

    parser.add_argument(
        "--evaluation-file",
        type=Path,
        default=DEFAULT_EVALUATION_FILE,
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    if not args.index.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {args.index}"
        )

    print("=" * 70)
    print("EKI-RAG BM25 EVALUATION")
    print("=" * 70)

    records = load_jsonl(
        args.evaluation_file
    )

    if len(records) != 560:
        raise ValueError(
            "Frozen Phase 3.2 evaluation set must contain "
            f"exactly 560 records, found {len(records)}."
        )

    print(
        f"Evaluation questions: {len(records)}"
    )

    print("Loading BM25 index...")

    retriever = bm25s.BM25.load(
        str(args.index),
        load_corpus=True,
    )

    print("BM25 index loaded.")

    result = evaluate(
        retriever,
        records,
        top_ks=[5, 10, 20],
    )

    output = {
        "project": "EKI-RAG",
        "dataset": "HotpotQA",
        "retriever": "BM25",
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
    print("BM25 RESULTS")
    print("=" * 70)

    for metric, value in result["metrics"].items():
        print(
            f"{metric:15s}: {value:.4f}"
        )

    print(
        f"Time            : "
        f"{result['elapsed_seconds']:.2f}s"
    )

    print(
        f"Saved           : {args.output}"
    )


if __name__ == "__main__":
    main()