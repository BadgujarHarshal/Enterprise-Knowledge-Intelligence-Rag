# tests/evaluation/test_phase10_optimizer.py

from __future__ import annotations

from app.evaluation.phase10_optimizer import (
    Phase10Optimizer,
    build_optimization_candidate,
    extract_records,
    merge_phase9_5_categories,
    select_final_candidate,
    validate_record_alignment,
)


def make_record(
    index=1,
    category=None,
    f1=1.0,
    exact_match=1.0,
    recall=1.0,
    overlap=1.0,
    citation_coverage=1.0,
    citation_precision=1.0,
    citation_recall=1.0,
    citation_validity=1.0,
    citation_source="model",
    citations=None,
):
    if citations is None:
        citations = [1]

    record = {
        "index": index,
        "question": "test question",
        "gold_answer": "correct answer",
        "answer": "correct answer",
        "citations": citations,
        "citation_source": citation_source,
        "citation_repair_used": False,
        "supporting_titles": [
            "Title A",
        ],
        "citation_context": [
            {
                "citation_id": 1,
                "title": "Title A",
                "chunk_id": "chunk-1",
                "hop": 1,
                "score": 0.9,
            }
        ],
        "metrics": {
            "answer_f1": f1,
            "exact_match": exact_match,
            "answer_contains_gold": exact_match,
            "context_title_recall": recall,
            "answer_evidence_overlap": overlap,
            "citation_coverage": citation_coverage,
            "citation_title_precision": citation_precision,
            "citation_title_recall": citation_recall,
            "citation_validity": citation_validity,
        },
    }

    if category is not None:
        record["category"] = category

    return record


def test_extract_records_from_list():
    records = [
        make_record()
    ]

    assert extract_records(
        records
    ) == records


def test_extract_records_from_results():
    records = [
        make_record()
    ]

    payload = {
        "results": records
    }

    assert extract_records(
        payload
    ) == records


def test_extract_records_from_failure_records():
    records = [
        make_record()
    ]

    payload = {
        "failure_records": records
    }

    assert extract_records(
        payload
    ) == records


def test_successful_rag():
    optimizer = Phase10Optimizer()

    summary, failures, *_ = optimizer.analyze(
        [
            make_record()
        ]
    )

    assert summary.question_count == 1
    assert summary.successful_rag == 1
    assert summary.retrieval_failures == 0
    assert summary.generation_failures == 0
    assert failures[0].primary_category == "SUCCESS"


def test_retrieval_failure_is_derived():
    optimizer = Phase10Optimizer()

    record = make_record(
        f1=0.0,
        exact_match=0.0,
        recall=0.5,
        overlap=0.0,
    )

    summary, failures, *_ = optimizer.analyze(
        [
            record
        ]
    )

    assert summary.retrieval_failures == 1
    assert failures[0].primary_category == (
        "RETRIEVAL_FAILURE"
    )
    assert failures[0].retrieval_gap == 0.5


def test_generation_failure_is_derived():
    optimizer = Phase10Optimizer()

    record = make_record(
        f1=0.0,
        exact_match=0.0,
        recall=1.0,
        overlap=1.0,
    )

    summary, failures, *_ = optimizer.analyze(
        [
            record
        ]
    )

    assert summary.generation_failures == 1
    assert failures[0].primary_category == (
        "GENERATION_FAILURE"
    )
    assert failures[0].retrieval_gap == 0.0
    assert failures[0].generation_gap == 1.0


def test_phase9_5_category_is_preferred():
    optimizer = Phase10Optimizer()

    record = make_record(
        f1=0.0,
        exact_match=0.0,
        recall=0.5,
        overlap=0.0,
    )

    phase9_5_record = dict(record)

    phase9_5_record["category"] = (
        "LUCKY_ANSWER"
    )

    phase9_5_record[
        "secondary_categories"
    ] = []

    merged, alignment = (
        merge_phase9_5_categories(
            [record],
            [phase9_5_record],
        )
    )

    assert alignment["aligned"] is True
    assert alignment["used"] is True

    summary, failures, *_ = optimizer.analyze(
        merged
    )

    assert summary.lucky_answers == 1
    assert failures[0].primary_category == (
        "LUCKY_ANSWER"
    )


def test_citation_metric_warning():
    optimizer = Phase10Optimizer()

    record = make_record(
        citation_coverage=0.0,
        citation_precision=0.5,
        citation_recall=0.5,
        citation_validity=0.0,
        citation_source="model",
    )

    summary, *_ = optimizer.analyze(
        [
            record
        ]
    )

    assert summary.native_citation_records == 1
    assert summary.citation_metric_warning is True


def test_citation_audit_detects_invalid_mapping():
    optimizer = Phase10Optimizer()

    record = make_record()

    record["citations"] = [999]

    audit = optimizer.audit_citations(
        [
            record
        ]
    )

    assert audit.records_with_citation_ids == 1
    assert audit.records_with_context == 1
    assert audit.records_with_mappable_citations == 0
    assert audit.records_with_invalid_citation_ids == 1
    assert audit.warning is True


def test_citation_audit_accepts_valid_mapping():
    optimizer = Phase10Optimizer()

    record = make_record()

    audit = optimizer.audit_citations(
        [
            record
        ]
    )

    assert audit.records_with_citation_ids == 1
    assert audit.records_with_context == 1
    assert audit.records_with_mappable_citations == 1
    assert audit.records_with_invalid_citation_ids == 0


def test_no_citation_record():
    optimizer = Phase10Optimizer()

    record = make_record(
        f1=0.0,
        exact_match=0.0,
        citations=[],
        citation_source="none",
    )

    summary, failures, *_ = optimizer.analyze(
        [
            record
        ]
    )

    assert summary.no_citation_records == 1
    assert failures[0].citation_count == 0


def test_unsupported_generation_is_derived():
    optimizer = Phase10Optimizer()

    record = make_record(
        f1=0.0,
        exact_match=0.0,
        recall=1.0,
        overlap=0.0,
    )

    summary, failures, *_ = optimizer.analyze(
        [
            record
        ]
    )

    assert (
        "UNSUPPORTED_GENERATION"
        in failures[0].secondary_categories
    )

    assert (
        summary.unsupported_generation_candidates
        == 1
    )


def test_retrieval_recommendation():
    optimizer = Phase10Optimizer()

    records = []

    for index in range(10):
        records.append(
            make_record(
                index=index,
                f1=0.0,
                exact_match=0.0,
                recall=0.5,
                overlap=0.0,
            )
        )

    summary, *_ = optimizer.analyze(
        records
    )

    recommendations = optimizer.recommendations(
        summary
    )

    areas = {
        item.area
        for item in recommendations
    }

    assert "retrieval" in areas


def test_generation_recommendation():
    optimizer = Phase10Optimizer()

    records = []

    for index in range(10):
        records.append(
            make_record(
                index=index,
                f1=0.0,
                exact_match=0.0,
                recall=1.0,
                overlap=1.0,
            )
        )

    summary, *_ = optimizer.analyze(
        records
    )

    recommendations = optimizer.recommendations(
        summary
    )

    areas = {
        item.area
        for item in recommendations
    }

    assert "generation" in areas


def test_citation_integrity_recommendation():
    optimizer = Phase10Optimizer()

    record = make_record(
        citation_coverage=0.0,
        citation_precision=0.0,
        citation_recall=0.0,
        citation_validity=0.0,
    )

    summary, *_ = optimizer.analyze(
        [
            record
        ]
    )

    audit = optimizer.audit_citations(
        [
            record
        ]
    )

    recommendations = optimizer.recommendations(
        summary,
        audit,
    )

    areas = {
        item.area
        for item in recommendations
    }

    assert "citation_integrity" in areas


def test_validate_record_alignment_success():
    primary = [
        make_record(
            index=1
        ),
        make_record(
            index=2
        ),
    ]

    secondary = [
        make_record(
            index=1
        ),
        make_record(
            index=2
        ),
    ]

    result = validate_record_alignment(
        primary,
        secondary,
    )

    assert result["aligned"] is True
    assert result["missing_from_secondary"] == []
    assert result["extra_in_secondary"] == []
    assert result["question_mismatches"] == []


def test_validate_record_alignment_detects_missing():
    primary = [
        make_record(
            index=1
        ),
        make_record(
            index=2
        ),
    ]

    secondary = [
        make_record(
            index=1
        ),
    ]

    result = validate_record_alignment(
        primary,
        secondary,
    )

    assert result["aligned"] is False
    assert result["missing_from_secondary"] == [2]


def test_validate_record_alignment_detects_question_mismatch():
    primary = [
        make_record(
            index=1
        )
    ]

    secondary = [
        make_record(
            index=1
        )
    ]

    secondary[0]["question"] = (
        "different question"
    )

    result = validate_record_alignment(
        primary,
        secondary,
    )

    assert result["aligned"] is False
    assert result["question_mismatches"] == [1]


def test_build_optimization_candidate():
    optimizer = Phase10Optimizer()

    records = [
        make_record()
    ]

    candidate = build_optimization_candidate(
        name="baseline",
        records=records,
        optimizer=optimizer,
        source="phase7",
        measured=True,
    )

    assert candidate.name == "baseline"
    assert candidate.measured is True
    assert candidate.question_count == 1
    assert candidate.mean_answer_f1 == 1.0
    assert candidate.status == "MEASURED"


def test_select_final_candidate():
    optimizer = Phase10Optimizer()

    records = [
        make_record()
    ]

    summary, *_ = optimizer.analyze(
        records
    )

    audit = optimizer.audit_citations(
        records
    )

    candidate = build_optimization_candidate(
        name="baseline_qwen2_5_3b",
        records=records,
        optimizer=optimizer,
        source="phase7",
        measured=True,
    )

    decision = select_final_candidate(
        candidates=[
            candidate
        ],
        summary=summary,
        citation_audit=audit,
        final_configuration={
            "model": "Qwen/Qwen2.5-3B-Instruct"
        },
    )

    assert decision.status == (
        "MEASURED_CANDIDATE_SELECTED"
    )

    assert decision.selected_candidate == (
        "baseline_qwen2_5_3b"
    )

    assert decision.measured_candidates == [
        "baseline_qwen2_5_3b"
    ]


def test_empty_analysis():
    optimizer = Phase10Optimizer()

    summary, failures, categories, secondary, matrix = (
        optimizer.analyze([])
    )

    assert summary.question_count == 0
    assert failures == []
    assert categories == {}
    assert secondary == {}
    assert matrix == {}


def test_empty_citation_audit():
    optimizer = Phase10Optimizer()

    audit = optimizer.audit_citations([])

    assert audit.native_citation_records == 0
    assert audit.posthoc_citation_records == 0
    assert audit.no_citation_records == 0
    assert audit.warning is False