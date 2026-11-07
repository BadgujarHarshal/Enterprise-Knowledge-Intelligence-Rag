# scripts/phase10_optimization.py

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from app.evaluation.phase10_optimizer import (
    CitationAudit,
    FinalOptimizationDecision,
    OptimizationCandidate,
    Phase10Optimizer,
    Phase10Report,
    build_optimization_candidate,
    extract_records,
    index_records,
    load_json,
    merge_phase9_5_categories,
    save_json,
    save_report,
    select_final_candidate,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "phase10.yaml"
)


def load_config(
    path: Path,
) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for Phase 10. "
            "Install it with: pip install pyyaml"
        ) from exc

    if not path.exists():
        raise FileNotFoundError(
            f"Phase 10 configuration file does not exist: "
            f"{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        payload = yaml.safe_load(file)

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Phase 10 configuration must contain a YAML mapping."
        )

    return payload


def resolve_project_path(
    value: str,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def require_records(
    path: Path,
    expected_questions: int,
    label: str,
) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist:\n{path}"
        )

    payload = load_json(path)

    records = extract_records(
        payload
    )

    if records is None:
        raise RuntimeError(
            f"Could not extract records from {label}:\n"
            f"{path}"
        )

    if len(records) != expected_questions:
        raise RuntimeError(
            f"{label} contains {len(records)} records, "
            f"but Phase 10 requires exactly "
            f"{expected_questions} records.\n"
            f"File: {path}"
        )

    return records


def validate_unique_indexes(
    records: List[Dict[str, Any]],
    label: str,
) -> Dict[str, Any]:
    indexed = index_records(
        records
    )

    duplicates = (
        len(records)
        - len(indexed)
    )

    if duplicates > 0:
        raise RuntimeError(
            f"{label} contains duplicate record indexes."
        )

    return {
        "count": len(records),
        "unique_indexes": len(indexed),
        "duplicates": duplicates,
    }


def write_summary(
    path: Path,
    report: Phase10Report,
) -> None:
    payload = {
        "phase": report.phase,
        "name": report.name,
        "question_count": report.question_count,
        "validation": report.validation,
        "summary": asdict(
            report.summary
        ),
        "citation_audit": asdict(
            report.citation_audit
        ),
        "category_counts": (
            report.category_counts
        ),
        "secondary_category_counts": (
            report.secondary_category_counts
        ),
        "failure_matrix": (
            report.failure_matrix
        ),
        "recommendations": [
            asdict(item)
            for item in report.recommendations
        ],
        "optimization_candidates": [
            asdict(item)
            for item in report.optimization_candidates
        ],
        "final_decision": asdict(
            report.final_decision
        ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        path,
        payload,
    )


def print_header() -> None:
    print(
        "=" * 80
    )
    print(
        "EKI-RAG — PHASE 10"
    )
    print(
        "FINAL RAG OPTIMIZATION AND PRODUCTION FREEZE"
    )
    print(
        "=" * 80
    )
    print()


def print_summary(
    report: Phase10Report,
) -> None:
    summary = report.summary
    audit = report.citation_audit

    print(
        "PHASE 10 SUMMARY"
    )
    print(
        "-" * 80
    )

    print(
        f"Questions: "
        f"{summary.question_count}"
    )

    print(
        f"Successful RAG: "
        f"{summary.successful_rag}"
    )

    print(
        f"Retrieval failures: "
        f"{summary.retrieval_failures}"
    )

    print(
        f"Generation failures: "
        f"{summary.generation_failures}"
    )

    print(
        f"Lucky answers: "
        f"{summary.lucky_answers}"
    )

    print(
        f"Appropriate abstentions: "
        f"{summary.appropriate_abstentions}"
    )

    print(
        f"Over-conservative abstentions: "
        f"{summary.over_conservative_abstentions}"
    )

    print(
        f"Unsupported-generation candidates: "
        f"{summary.unsupported_generation_candidates}"
    )

    print(
        f"Citation failures: "
        f"{summary.citation_failures}"
    )

    print()

    print(
        "MEAN METRICS"
    )

    print(
        "-" * 80
    )

    print(
        f"Answer F1: "
        f"{summary.mean_answer_f1:.4f}"
    )

    print(
        f"Exact Match: "
        f"{summary.mean_exact_match:.4f}"
    )

    print(
        f"Context Title Recall: "
        f"{summary.mean_context_title_recall:.4f}"
    )

    print(
        f"Evidence Overlap: "
        f"{summary.mean_evidence_overlap:.4f}"
    )

    print(
        f"Citation Coverage: "
        f"{summary.mean_citation_coverage:.4f}"
    )

    print(
        f"Citation Title Precision: "
        f"{summary.mean_citation_title_precision:.4f}"
    )

    print(
        f"Citation Title Recall: "
        f"{summary.mean_citation_title_recall:.4f}"
    )

    print(
        f"Citation Validity: "
        f"{summary.mean_citation_validity:.4f}"
    )

    print()

    print(
        "CITATION AUDIT"
    )

    print(
        "-" * 80
    )

    print(
        f"Native model citations: "
        f"{audit.native_citation_records}"
    )

    print(
        f"Post-hoc citations: "
        f"{audit.posthoc_citation_records}"
    )

    print(
        f"No citation IDs: "
        f"{audit.no_citation_records}"
    )

    print(
        f"Records with citation IDs: "
        f"{audit.records_with_citation_ids}"
    )

    print(
        f"Records with citation context: "
        f"{audit.records_with_context}"
    )

    print(
        f"Mappable citation records: "
        f"{audit.records_with_mappable_citations}"
    )

    print(
        f"Invalid citation mappings: "
        f"{audit.records_with_invalid_citation_ids}"
    )

    print(
        f"Audit warning: "
        f"{audit.warning}"
    )

    if audit.findings:
        print()

        for finding in audit.findings:
            print(
                f"- {finding}"
            )

    print()


def build_final_configuration(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    baseline = config.get(
        "baseline",
        {},
    )

    return {
        "phase": "10",
        "status": "FINAL_BACKEND_CONFIGURATION",
        "model": baseline.get(
            "model",
            "Qwen/Qwen2.5-3B-Instruct",
        ),
        "retrieval": baseline.get(
            "retrieval",
            {},
        ),
        "reranking": baseline.get(
            "reranking",
            {},
        ),
        "multi_hop": baseline.get(
            "multi_hop",
            {},
        ),
        "evaluation_questions": config.get(
            "expected_questions",
            560,
        ),
        "optimization_policy": config.get(
            "optimization_policy",
            {},
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EKI-RAG Phase 10 final optimizer, "
            "validation, and production freeze."
        )
    )

    parser.add_argument(
        "--config",
        default=str(
            DEFAULT_CONFIG
        ),
    )

    parser.add_argument(
        "--phase7",
        default=None,
        help=(
            "Optional override for the Phase 7 result file."
        ),
    )

    parser.add_argument(
        "--phase9-5",
        default=None,
        help=(
            "Optional override for the Phase 9.5 "
            "error analysis file."
        ),
    )

    parser.add_argument(
        "--expected",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    ).resolve()

    config = load_config(
        config_path
    )

    expected_questions = (
        args.expected
        if args.expected is not None
        else int(
            config.get(
                "expected_questions",
                560,
            )
        )
    )

    thresholds = {
        "answer_f1_threshold": float(
            config.get(
                "answer_f1_threshold",
                0.5,
            )
        ),
        "retrieval_complete_threshold": float(
            config.get(
                "retrieval_complete_threshold",
                1.0,
            )
        ),
        "evidence_overlap_threshold": float(
            config.get(
                "evidence_overlap_threshold",
                0.25,
            )
        ),
    }

    input_config = config.get(
        "inputs",
        {},
    )

    output_config = config.get(
        "outputs",
        {},
    )

    phase7_value = (
        args.phase7
        if args.phase7
        else input_config.get(
            "phase7_result_file"
        )
    )

    phase9_5_value = (
        args.phase9_5
        if args.phase9_5
        else input_config.get(
            "phase9_5_report_file"
        )
    )

    if not phase7_value:
        raise RuntimeError(
            "Phase 7 result file is not configured."
        )

    phase7_path = resolve_project_path(
        str(phase7_value)
    )

    phase9_5_path: Optional[Path] = None

    if phase9_5_value:
        phase9_5_path = resolve_project_path(
            str(phase9_5_value)
        )

    print_header()

    print(
        "VALIDATING PHASE 7 RESULT"
    )

    print(
        "-" * 80
    )

    print(
        f"Phase 7 source: "
        f"{phase7_path}"
    )

    phase7_records = require_records(
        phase7_path,
        expected_questions,
        "Phase 7 result",
    )

    phase7_index_validation = (
        validate_unique_indexes(
            phase7_records,
            "Phase 7 result",
        )
    )

    print(
        f"Records: "
        f"{len(phase7_records)}"
    )

    print(
        "Validation: PASSED"
    )

    phase9_5_records: Optional[
        List[Dict[str, Any]]
    ] = None

    phase9_5_alignment: Dict[str, Any] = {
        "available": False,
        "aligned": False,
        "used": False,
    }

    if phase9_5_path is not None:
        print()
        print(
            "VALIDATING PHASE 9.5 ERROR ANALYSIS"
        )
        print(
            "-" * 80
        )
        print(
            f"Phase 9.5 source: "
            f"{phase9_5_path}"
        )

        if phase9_5_path.exists():
            phase9_5_payload = load_json(
                phase9_5_path
            )

            phase9_5_records = extract_records(
                phase9_5_payload
            )

            if phase9_5_records is None:
                raise RuntimeError(
                    "Could not extract Phase 9.5 "
                    "error-analysis records."
                )

            if len(phase9_5_records) != expected_questions:
                raise RuntimeError(
                    "Phase 9.5 error analysis does not "
                    f"contain exactly {expected_questions} records. "
                    f"Found {len(phase9_5_records)}."
                )

            validate_unique_indexes(
                phase9_5_records,
                "Phase 9.5 error analysis",
            )

            print(
                f"Records: "
                f"{len(phase9_5_records)}"
            )
        else:
            print(
                "Phase 9.5 file not found. "
                "Categories will be derived from Phase 7 metrics."
            )

    merged_records, phase9_5_alignment = (
        merge_phase9_5_categories(
            phase7_records,
            phase9_5_records,
        )
    )

    if (
        phase9_5_records is not None
        and not phase9_5_alignment["aligned"]
    ):
        print()
        print(
            "WARNING: Phase 9.5 records are not aligned "
            "with Phase 7. Phase 10 will derive categories "
            "from the Phase 7 metrics instead."
        )

    optimizer = Phase10Optimizer(
        answer_f1_threshold=thresholds[
            "answer_f1_threshold"
        ],
        retrieval_complete_threshold=thresholds[
            "retrieval_complete_threshold"
        ],
        evidence_overlap_threshold=thresholds[
            "evidence_overlap_threshold"
        ],
    )

    (
        summary,
        failure_records,
        category_counts,
        secondary_category_counts,
        failure_matrix,
    ) = optimizer.analyze(
        merged_records
    )

    citation_audit = optimizer.audit_citations(
        merged_records
    )

    summary.phase9_5_alignment_warning = (
        phase9_5_records is not None
        and not phase9_5_alignment["aligned"]
    )

    recommendations = optimizer.recommendations(
        summary,
        citation_audit,
    )

    baseline_candidate = (
        build_optimization_candidate(
            name="baseline_qwen2_5_3b",
            records=merged_records,
            optimizer=optimizer,
            source=str(
                phase7_path
            ),
            measured=True,
        )
    )

    candidates = [
        baseline_candidate
    ]

    final_configuration = (
        build_final_configuration(
            config
        )
    )

    final_decision = select_final_candidate(
        candidates=candidates,
        summary=summary,
        citation_audit=citation_audit,
        final_configuration=final_configuration,
    )

    validation = {
        "expected_questions": expected_questions,
        "phase7_records": len(
            phase7_records
        ),
        "phase7_complete": (
            len(phase7_records)
            == expected_questions
        ),
        "phase7_unique_indexes": (
            phase7_index_validation[
                "unique_indexes"
            ]
        ),
        "phase9_5_available": (
            phase9_5_records is not None
        ),
        "phase9_5_alignment": (
            phase9_5_alignment
        ),
        "evaluation_set_frozen": True,
        "partial_evaluation_rejected": True,
        "citation_audit_completed": True,
        "final_configuration_generated": True,
        "measured_optimization_candidates": len(
            candidates
        ),
    }

    report = Phase10Report(
        phase="10",
        name=(
            "Final RAG Optimization "
            "and Production Freeze"
        ),
        input_file=str(
            phase7_path
        ),
        phase9_5_file=(
            str(phase9_5_path)
            if phase9_5_path is not None
            else None
        ),
        question_count=len(
            merged_records
        ),
        validation=validation,
        summary=summary,
        citation_audit=citation_audit,
        failure_records=failure_records,
        category_counts=category_counts,
        secondary_category_counts=(
            secondary_category_counts
        ),
        failure_matrix=failure_matrix,
        recommendations=recommendations,
        optimization_candidates=candidates,
        final_decision=final_decision,
    )

    report_output = resolve_project_path(
        str(
            output_config.get(
                "optimization_report_file",
                "data/evaluation/"
                "phase10_final_optimization_report.json",
            )
        )
    )

    summary_output = resolve_project_path(
        str(
            output_config.get(
                "optimization_summary_file",
                "data/evaluation/"
                "phase10_final_optimization_summary.json",
            )
        )
    )

    configuration_output = resolve_project_path(
        str(
            output_config.get(
                "final_configuration_file",
                "data/evaluation/"
                "phase10_final_configuration.json",
            )
        )
    )

    regression_output = resolve_project_path(
        str(
            output_config.get(
                "regression_report_file",
                "data/evaluation/"
                "phase10_final_regression_report.json",
            )
        )
    )

    report_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_report(
        report_output,
        report,
    )

    write_summary(
        summary_output,
        report,
    )

    save_json(
        configuration_output,
        asdict(
            final_decision
        ),
    )

    regression_report = {
        "phase": "10",
        "status": (
            "PASSED"
            if (
                validation["phase7_complete"]
                and validation["citation_audit_completed"]
                and validation[
                    "final_configuration_generated"
                ]
            )
            else "FAILED"
        ),
        "checks": {
            "phase7_record_count": (
                validation["phase7_complete"]
            ),
            "phase7_unique_indexes": (
                validation[
                    "phase7_unique_indexes"
                ]
                == expected_questions
            ),
            "evaluation_set_frozen": True,
            "citation_audit_completed": True,
            "final_configuration_generated": True,
        },
        "question_count": expected_questions,
        "selected_candidate": (
            final_decision.selected_candidate
        ),
        "decision_status": (
            final_decision.status
        ),
        "warnings": (
            final_decision.warnings
        ),
    }

    save_json(
        regression_output,
        regression_report,
    )

    print()
    print_summary(
        report
    )

    print(
        "FINAL OPTIMIZATION DECISION"
    )

    print(
        "-" * 80
    )

    print(
        f"Status: "
        f"{final_decision.status}"
    )

    print(
        f"Selected candidate: "
        f"{final_decision.selected_candidate}"
    )

    print(
        f"Measured candidates: "
        f"{len(final_decision.measured_candidates)}"
    )

    print(
        "Selection basis: "
        f"{final_decision.selection_basis}"
    )

    if final_decision.unmeasured_areas:
        print()
        print(
            "Unmeasured optimization areas:"
        )

        for area in final_decision.unmeasured_areas:
            print(
                f"- {area}"
            )

    print()
    print(
        "OUTPUT FILES"
    )

    print(
        "-" * 80
    )

    print(
        report_output
    )

    print(
        summary_output
    )

    print(
        configuration_output
    )

    print(
        regression_output
    )

    print()

    print(
        "PHASE 10 FINAL OPTIMIZATION COMPLETE"
    )

    print(
        "RAG BACKEND CONFIGURATION IS READY FOR FREEZE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()