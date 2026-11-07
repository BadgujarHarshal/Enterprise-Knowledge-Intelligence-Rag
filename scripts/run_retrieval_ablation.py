# scripts/run_retrieval_ablation.py

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DENSE = (
    PROJECT_ROOT
    / "experiments"
    / "hotpotqa_dense_retrieval_baseline_fully_evaluable.json"
)

DEFAULT_BM25 = (
    PROJECT_ROOT
    / "experiments"
    / "hotpotqa_bm25_baseline.json"
)

DEFAULT_HYBRID = (
    PROJECT_ROOT
    / "experiments"
    / "hotpotqa_hybrid_baseline.json"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "hotpotqa_retrieval_ablation.json"
)


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing experiment result: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Build EKI-RAG retrieval ablation report."
    )

    parser.add_argument(
        "--dense",
        type=Path,
        default=DEFAULT_DENSE,
    )

    parser.add_argument(
        "--bm25",
        type=Path,
        default=DEFAULT_BM25,
    )

    parser.add_argument(
        "--hybrid",
        type=Path,
        default=DEFAULT_HYBRID,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    dense = load(args.dense)
    bm25 = load(args.bm25)
    hybrid = load(args.hybrid)

    report = {
        "project": "EKI-RAG",
        "dataset": "HotpotQA",
        "evaluation_protocol": (
            "Frozen 560-question fully-evaluable "
            "validation subset"
        ),
        "systems": {
            "dense": dense.get("metrics", {}),
            "bm25": bm25.get("metrics", {}),
            "hybrid": hybrid.get("metrics", {}),
        },
        "configuration": {
            "dense": {
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "bm25": {
                "library": "bm25s"
            },
            "hybrid": {
                "fusion": "RRF",
                "rrf_k": hybrid.get("rrf_k"),
                "dense_weight": hybrid.get(
                    "dense_weight"
                ),
                "bm25_weight": hybrid.get(
                    "bm25_weight"
                ),
            },
        },
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
            report,
            file,
            indent=2,
        )

    print("=" * 70)
    print("EKI-RAG RETRIEVAL ABLATION")
    print("=" * 70)

    metrics = [
        "hit@5",
        "hit@10",
        "hit@20",
        "coverage@5",
        "coverage@10",
        "coverage@20",
        "mrr@5",
        "mrr@10",
        "mrr@20",
    ]

    print(
        f"{'Metric':<16}"
        f"{'Dense':>12}"
        f"{'BM25':>12}"
        f"{'Hybrid':>12}"
    )

    print("-" * 52)

    for metric in metrics:

        dense_value = report[
            "systems"
        ]["dense"].get(metric)

        bm25_value = report[
            "systems"
        ]["bm25"].get(metric)

        hybrid_value = report[
            "systems"
        ]["hybrid"].get(metric)

        print(
            f"{metric:<16}"
            f"{dense_value if dense_value is not None else 0:>12.4f}"
            f"{bm25_value if bm25_value is not None else 0:>12.4f}"
            f"{hybrid_value if hybrid_value is not None else 0:>12.4f}"
        )

    print()
    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()