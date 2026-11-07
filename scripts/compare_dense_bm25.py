# scripts/compare_dense_bm25.py

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

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "hotpotqa_dense_vs_bm25.json"
)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Result file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Compare dense and BM25 retrieval."
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    dense = load_json(args.dense)
    bm25 = load_json(args.bm25)

    dense_metrics = dense.get(
        "metrics",
        {}
    )

    bm25_metrics = bm25.get(
        "metrics",
        {}
    )

    comparison = {
        "evaluation_protocol": (
            "Same frozen 560-question "
            "fully-evaluable validation subset"
        ),
        "dense": dense_metrics,
        "bm25": bm25_metrics,
        "delta_bm25_minus_dense": {},
    }

    metric_names = sorted(
        set(dense_metrics)
        | set(bm25_metrics)
    )

    for metric in metric_names:

        dense_value = dense_metrics.get(
            metric
        )

        bm25_value = bm25_metrics.get(
            metric
        )

        if (
            dense_value is not None
            and bm25_value is not None
        ):
            comparison[
                "delta_bm25_minus_dense"
            ][metric] = (
                bm25_value - dense_value
            )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            comparison,
            file,
            indent=2,
        )

    print("=" * 70)
    print("DENSE vs BM25")
    print("=" * 70)

    for metric in metric_names:

        dense_value = dense_metrics.get(
            metric
        )

        bm25_value = bm25_metrics.get(
            metric
        )

        if (
            dense_value is not None
            and bm25_value is not None
        ):
            delta = (
                bm25_value
                - dense_value
            )

            print(
                f"{metric:15s} "
                f"Dense={dense_value:.4f} "
                f"BM25={bm25_value:.4f} "
                f"Delta={delta:+.4f}"
            )

    print()
    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()