# scripts/phase8_evaluate_generation.py

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.rag_evaluator import (
    RAGEvaluator,
)


ROOT = Path(
    "/content/drive/MyDrive/EKI-RAG"
)

EXPERIMENTS_DIR = (
    ROOT / "experiments"
)


INPUT_PATH = (
    EXPERIMENTS_DIR /
    "hotpotqa_phase7_generation_results.json"
)

OUTPUT_PATH = (
    EXPERIMENTS_DIR /
    "hotpotqa_phase8_evaluation.json"
)


def main():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Phase 7 results not found: "
            f"{INPUT_PATH}"
        )


    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        results = json.load(
            file
        )


    evaluator = RAGEvaluator(
        results
    )


    summary = evaluator.evaluate()


    output = {

        "dataset":
            "HotpotQA",

        "evaluation_subset":
            "fully_evaluable_560",

        "question_count":
            summary.question_count,

        "metrics":
            {

                "exact_match":
                    summary.exact_match,

                "f1":
                    summary.f1,

                "citation_validity":
                    summary.citation_validity,

                "citation_title_precision":
                    summary.citation_title_precision,

                "citation_title_recall":
                    summary.citation_title_recall,

                "citation_coverage":
                    summary.citation_coverage,

                "evidence_overlap":
                    summary.evidence_overlap,

            },

    }


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )


    print("=" * 80)

    print(
        "EKI-RAG — PHASE 8 EVALUATION"
    )

    print("=" * 80)

    print(
        "Questions:",
        summary.question_count
    )

    print()

    print(
        "Exact Match:",
        f"{summary.exact_match:.4f}"
    )

    print(
        "F1:",
        f"{summary.f1:.4f}"
    )

    print(
        "Citation Validity:",
        f"{summary.citation_validity:.4f}"
    )

    print(
        "Citation Title Precision:",
        f"{summary.citation_title_precision:.4f}"
    )

    print(
        "Citation Title Recall:",
        f"{summary.citation_title_recall:.4f}"
    )

    print(
        "Citation Coverage:",
        f"{summary.citation_coverage:.4f}"
    )

    print(
        "Evidence Overlap:",
        f"{summary.evidence_overlap:.4f}"
    )

    print()

    print(
        "Output:",
        OUTPUT_PATH
    )


if __name__ == "__main__":

    main()