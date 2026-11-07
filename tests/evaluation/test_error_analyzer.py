# tests/evaluation/test_error_analyzer.py

from app.evaluation.error_analyzer import (
    ErrorAnalysisConfig,
    RAGErrorAnalyzer,
)


def build_record(
    *,
    index=1,
    answer="Japan",
    gold_answer="Japan",
    context_recall=1.0,
    f1=1.0,
    exact_match=1.0,
    contains_gold=1.0,
    evidence_overlap=1.0,
    citation_coverage=1.0,
    citation_precision=1.0,
    citation_recall=1.0,
    citation_validity=1.0,
    citations=None,
    supporting_titles=None,
    titles=None,
):
    if citations is None:
        citations = [1]

    if supporting_titles is None:
        supporting_titles = ["Japan"]

    if titles is None:
        titles = ["Japan"]

    return {
        "index": index,
        "question": "Where does the entity come from?",
        "gold_answer": gold_answer,
        "answer": answer,
        "supporting_titles": supporting_titles,
        "citations": citations,
        "citation_context": [
            {
                "citation_id": i + 1,
                "title": title,
                "content": f"Evidence about {title}.",
            }
            for i, title in enumerate(titles)
        ],
        "metrics": {
            "exact_match": exact_match,
            "answer_f1": f1,
            "answer_contains_gold": contains_gold,
            "context_title_recall": context_recall,
            "answer_evidence_overlap": evidence_overlap,
            "citation_coverage": citation_coverage,
            "citation_title_precision": citation_precision,
            "citation_title_recall": citation_recall,
            "citation_validity": citation_validity,
        },
        "citation_repair_used": False,
        "citation_source": "model",
    }


def test_success():
    analyzer = RAGErrorAnalyzer()

    summary, diagnoses = analyzer.analyze(
        [
            build_record()
        ]
    )

    assert summary.question_count == 1
    assert summary.successful_rag == 1
    assert diagnoses[0].category == "SUCCESS"
    assert diagnoses[0].answer_correct is True
    assert diagnoses[0].retrieval_complete is True


def test_retrieval_failure():
    analyzer = RAGErrorAnalyzer()

    summary, diagnoses = analyzer.analyze(
        [
            build_record(
                answer="France",
                gold_answer="Japan",
                context_recall=0.0,
                f1=0.0,
                exact_match=0.0,
                contains_gold=0.0,
                evidence_overlap=0.0,
                citation_coverage=0.0,
                citation_precision=0.0,
                citation_recall=0.0,
            )
        ]
    )

    assert summary.retrieval_failures == 1
    assert diagnoses[0].category == "RETRIEVAL_FAILURE"


def test_generation_failure():
    analyzer = RAGErrorAnalyzer()

    summary, diagnoses = analyzer.analyze(
        [
            build_record(
                answer="France",
                gold_answer="Japan",
                context_recall=1.0,
                f1=0.0,
                exact_match=0.0,
                contains_gold=0.0,
                evidence_overlap=0.0,
            )
        ]
    )

    assert summary.generation_failures == 1
    assert diagnoses[0].category == "GENERATION_FAILURE"
    assert (
        "UNSUPPORTED_GENERATION"
        in diagnoses[0].secondary_categories
    )


def test_lucky_answer():
    analyzer = RAGErrorAnalyzer()

    summary, diagnoses = analyzer.analyze(
        [
            build_record(
                context_recall=0.0,
                f1=1.0,
                exact_match=1.0,
                contains_gold=1.0,
            )
        ]
    )

    assert summary.lucky_answers == 1
    assert diagnoses[0].category == "LUCKY_ANSWER"


def test_appropriate_abstention():
    analyzer = RAGErrorAnalyzer()

    summary, diagnoses = analyzer.analyze(
        [
            build_record(
                answer="INSUFFICIENT",
                context_recall=0.0,
                f1=0.0,
                exact_match=0.0,
                contains_gold=0.0,
                evidence_overlap=0.0,
                citation_coverage=0.0,
                citation_precision=0.0,
                citation_recall=0.0,
            )
        ]
    )

    assert summary.appropriate_abstentions == 1
    assert diagnoses[0].category == "APPROPRIATE_ABSTENTION"


def test_over_conservative_abstention():
    analyzer = RAGErrorAnalyzer()

    summary, diagnoses = analyzer.analyze(
        [
            build_record(
                answer="INSUFFICIENT",
                context_recall=1.0,
                f1=0.0,
                exact_match=0.0,
                contains_gold=0.0,
                evidence_overlap=0.0,
                citation_coverage=0.0,
                citation_precision=0.0,
                citation_recall=0.0,
            )
        ]
    )

    assert summary.over_conservative_abstentions == 1
    assert (
        diagnoses[0].category
        == "OVER_CONSERVATIVE_ABSTENTION"
    )


def test_citation_failure_secondary_category():
    analyzer = RAGErrorAnalyzer()

    summary, diagnoses = analyzer.analyze(
        [
            build_record(
                citation_coverage=0.0,
                citation_precision=0.0,
                citation_recall=0.0,
                citation_validity=0.0,
                citations=[],
            )
        ]
    )

    assert summary.citation_failures == 1
    assert (
        "CITATION_FAILURE"
        in diagnoses[0].secondary_categories
    )


def test_empty_input():
    analyzer = RAGErrorAnalyzer()

    summary, diagnoses = analyzer.analyze(
        []
    )

    assert summary.question_count == 0
    assert summary.successful_rag == 0
    assert diagnoses == []


def test_flat_and_nested_titles():
    analyzer = RAGErrorAnalyzer()

    record = build_record()

    record["citation_context"] = [
        {
            "citation_id": 1,
            "document": {
                "title": "Japan",
                "content": "Evidence about Japan.",
            },
        }
    ]

    summary, diagnoses = analyzer.analyze(
        [record]
    )

    assert summary.question_count == 1
    assert diagnoses[0].retrieved_titles == [
        "Japan"
    ]