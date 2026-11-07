# scripts/phase9_error_analysis.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.error_analyzer import (
    ErrorAnalysisConfig,
    RAGErrorAnalyzer,
)


def load_config(
    path: Path,
) -> ErrorAnalysisConfig:
    if not path.exists():
        return ErrorAnalysisConfig()

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load the Phase 9 configuration."
        ) from exc

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(
            file
        ) or {}

    return ErrorAnalysisConfig(
        answer_f1_threshold=float(
            data.get(
                "answer_f1_threshold",
                0.5,
            )
        ),
        evidence_overlap_threshold=float(
            data.get(
                "evidence_overlap_threshold",
                0.25,
            )
        ),
        retrieval_complete_threshold=float(
            data.get(
                "retrieval_complete_threshold",
                1.0,
            )
        ),
        abstention_patterns=tuple(
            data.get(
                "abstention_patterns",
                [],
            )
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EKI-RAG Phase 9 error analysis."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to Phase 7 generation results JSON.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for the Phase 9 error-analysis JSON.",
    )

    parser.add_argument(
        "--config",
        default="configs/error_analysis.yaml",
        help="Phase 9 configuration path.",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    config_path = Path(
        args.config
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    config = load_config(
        config_path
    )

    analyzer = RAGErrorAnalyzer(
        config
    )

    records = analyzer.load_results(
        str(input_path)
    )

    summary, diagnoses = analyzer.analyze(
        records
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    analyzer.save_report(
        str(output_path),
        summary,
        diagnoses,
    )

    print()
    print("EKI-RAG Phase 9")
    print("RAG Error Analysis")
    print()
    print(
        f"Questions: {summary.question_count}"
    )
    print(
        f"Correct answers: {summary.correct_answers}"
    )
    print(
        f"Incorrect answers: {summary.incorrect_answers}"
    )
    print(
        f"Retrieval complete: {summary.retrieval_complete_questions}"
    )
    print(
        f"Retrieval incomplete: {summary.retrieval_incomplete_questions}"
    )
    print(
        f"Successful RAG: {summary.successful_rag}"
    )
    print(
        f"Retrieval failures: {summary.retrieval_failures}"
    )
    print(
        f"Generation failures: {summary.generation_failures}"
    )
    print(
        f"Lucky answers: {summary.lucky_answers}"
    )
    print(
        f"Appropriate abstentions: {summary.appropriate_abstentions}"
    )
    print(
        f"Over-conservative abstentions: {summary.over_conservative_abstentions}"
    )
    print(
        f"Unsupported-generation candidates: {summary.unsupported_generation_candidates}"
    )
    print(
        f"Citation failures: {summary.citation_failures}"
    )
    print()
    print(
        f"Mean Answer F1: {summary.mean_answer_f1:.4f}"
    )
    print(
        f"Mean Exact Match: {summary.mean_exact_match:.4f}"
    )
    print(
        f"Mean Context Title Recall: {summary.mean_context_title_recall:.4f}"
    )
    print(
        f"Mean Evidence Overlap: {summary.mean_evidence_overlap:.4f}"
    )
    print(
        f"Mean Citation Coverage: {summary.mean_citation_coverage:.4f}"
    )
    print(
        f"Mean Citation Precision: {summary.mean_citation_title_precision:.4f}"
    )
    print(
        f"Mean Citation Recall: {summary.mean_citation_title_recall:.4f}"
    )
    print(
        f"Mean Citation Validity: {summary.mean_citation_validity:.4f}"
    )
    print()
    print("Primary categories:")

    for category, count in sorted(
        summary.category_counts.items()
    ):
        print(
            f"  {category}: {count}"
        )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()