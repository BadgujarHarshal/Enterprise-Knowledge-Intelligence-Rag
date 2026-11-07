# tests/evaluation/test_rag_evaluator.py

# ================================================================================
# tests/evaluation/test_rag_evaluator.py
# ================================================================================
#
# Unit tests for the EKI-RAG evaluation layer.
#
# These tests intentionally use small deterministic fixtures.
# No model, GPU, network, or external dataset is required.
# ================================================================================


from app.evaluation.rag_evaluator import (
    EvaluationSummary,
    RAGEvaluator,
)


# ================================================================================
# TEST FIXTURE
# ================================================================================


def build_sample_results():
    """
    Build a deterministic single-question evaluation result.

    The question asks where Buck-Tick hails from.

    Citation [1] points to a context whose title is "Japan",
    and "Japan" is the gold supporting title.

    Therefore:
        citation coverage           = 1.0
        citation validity           = 1.0
        citation title precision    = 1.0
        citation title recall       = 1.0
    """

    return [
        {
            "question":
                "Where does Buck-Tick hail from?",

            "gold_answer":
                "Japan",

            "answer":
                "Buck-Tick hails from Japan [1].",

            "citations":
                [1],

            "context_count":
                2,

            "citation_context":
                [
                    {
                        "citation_id":
                            1,

                        "title":
                            "Japan",

                        "chunk_id":
                            "chunk-1",

                        "content":
                            "Buck-Tick is a Japanese rock band.",
                    },

                    {
                        "citation_id":
                            2,

                        "title":
                            "Canada",

                        "chunk_id":
                            "chunk-2",

                        "content":
                            "Hayden is Canadian.",
                    },
                ],

            "supporting_titles":
                ["Japan"],
        }
    ]


# ================================================================================
# BASIC EVALUATOR TEST
# ================================================================================


def test_rag_evaluator():

    results = build_sample_results()

    evaluator = RAGEvaluator(
        results
    )

    summary = evaluator.evaluate()

    assert isinstance(
        summary,
        EvaluationSummary,
    )

    assert (
        summary.question_count
        == 1
    )

    # "Japan" appears inside the generated sentence,
    # therefore exact match against the complete gold answer
    # is intentionally zero.
    assert (
        summary.exact_match
        == 0.0
    )

    # Token-level F1 must still detect overlap with "Japan".
    assert (
        summary.f1
        > 0.0
    )

    # One answer contains at least one citation.
    assert (
        summary.citation_coverage
        == 1.0
    )

    # Citation [1] is a valid citation because two contexts exist.
    assert (
        summary.citation_validity
        == 1.0
    )

    # Citation [1] points to the "Japan" context,
    # and "Japan" is a supporting title.
    assert (
        summary.citation_title_precision
        == 1.0
    )

    # The only supporting title, "Japan", was cited.
    assert (
        summary.citation_title_recall
        == 1.0
    )

    # "Japan" is present in the cited evidence.
    assert (
        summary.evidence_overlap
        > 0.0
    )


# ================================================================================
# EMPTY RESULTS
# ================================================================================


def test_rag_evaluator_empty_results():

    evaluator = RAGEvaluator(
        []
    )

    summary = evaluator.evaluate()

    assert isinstance(
        summary,
        EvaluationSummary,
    )

    assert (
        summary.question_count
        == 0
    )

    assert (
        summary.exact_match
        == 0.0
    )

    assert (
        summary.f1
        == 0.0
    )

    assert (
        summary.citation_validity
        == 0.0
    )

    assert (
        summary.citation_title_precision
        == 0.0
    )

    assert (
        summary.citation_title_recall
        == 0.0
    )

    assert (
        summary.citation_coverage
        == 0.0
    )

    assert (
        summary.evidence_overlap
        == 0.0
    )


# ================================================================================
# NO CITATIONS
# ================================================================================


def test_rag_evaluator_without_citations():

    results = [
        {
            "question":
                "Where does Buck-Tick hail from?",

            "gold_answer":
                "Japan",

            "answer":
                "Buck-Tick hails from Japan.",

            "citations":
                [],

            "context_count":
                2,

            "citation_context":
                [
                    {
                        "citation_id":
                            1,

                        "title":
                            "Japan",

                        "chunk_id":
                            "chunk-1",

                        "content":
                            "Buck-Tick is a Japanese rock band.",
                    },

                    {
                        "citation_id":
                            2,

                        "title":
                            "Canada",

                        "chunk_id":
                            "chunk-2",

                        "content":
                            "Hayden is Canadian.",
                    },
                ],

            "supporting_titles":
                ["Japan"],
        }
    ]

    evaluator = RAGEvaluator(
        results
    )

    summary = evaluator.evaluate()

    assert (
        summary.question_count
        == 1
    )

    assert (
        summary.citation_validity
        == 0.0
    )

    assert (
        summary.citation_title_precision
        == 0.0
    )

    assert (
        summary.citation_title_recall
        == 0.0
    )

    assert (
        summary.citation_coverage
        == 0.0
    )


# ================================================================================
# INVALID CITATION
# ================================================================================


def test_rag_evaluator_invalid_citation():

    results = [
        {
            "question":
                "Where does Buck-Tick hail from?",

            "gold_answer":
                "Japan",

            "answer":
                "Buck-Tick hails from Japan [99].",

            "citations":
                [99],

            "context_count":
                2,

            "citation_context":
                [
                    {
                        "citation_id":
                            1,

                        "title":
                            "Japan",

                        "chunk_id":
                            "chunk-1",

                        "content":
                            "Buck-Tick is a Japanese rock band.",
                    },

                    {
                        "citation_id":
                            2,

                        "title":
                            "Canada",

                        "chunk_id":
                            "chunk-2",

                        "content":
                            "Hayden is Canadian.",
                    },
                ],

            "supporting_titles":
                ["Japan"],
        }
    ]

    evaluator = RAGEvaluator(
        results
    )

    summary = evaluator.evaluate()

    assert (
        summary.question_count
        == 1
    )

    assert (
        summary.citation_coverage
        == 1.0
    )

    assert (
        summary.citation_validity
        == 0.0
    )

    assert (
        summary.citation_title_precision
        == 0.0
    )

    assert (
        summary.citation_title_recall
        == 0.0
    )


# ================================================================================
# WRONG CITATION
# ================================================================================


def test_rag_evaluator_wrong_supporting_title():

    results = [
        {
            "question":
                "Where does Buck-Tick hail from?",

            "gold_answer":
                "Japan",

            "answer":
                "Buck-Tick hails from Japan [2].",

            "citations":
                [2],

            "context_count":
                2,

            "citation_context":
                [
                    {
                        "citation_id":
                            1,

                        "title":
                            "Japan",

                        "chunk_id":
                            "chunk-1",

                        "content":
                            "Buck-Tick is a Japanese rock band.",
                    },

                    {
                        "citation_id":
                            2,

                        "title":
                            "Canada",

                        "chunk_id":
                            "chunk-2",

                        "content":
                            "Hayden is Canadian.",
                    },
                ],

            "supporting_titles":
                ["Japan"],
        }
    ]

    evaluator = RAGEvaluator(
        results
    )

    summary = evaluator.evaluate()

    assert (
        summary.question_count
        == 1
    )

    assert (
        summary.citation_coverage
        == 1.0
    )

    # [2] is syntactically valid because context 2 exists.
    assert (
        summary.citation_validity
        == 1.0
    )

    # But context 2 is not one of the supporting titles.
    assert (
        summary.citation_title_precision
        == 0.0
    )

    assert (
        summary.citation_title_recall
        == 0.0
    )


# ================================================================================
# MULTIPLE SUPPORTING TITLES
# ================================================================================


def test_rag_evaluator_multiple_supporting_titles():

    results = [
        {
            "question":
                "Which two locations are involved?",

            "gold_answer":
                "Japan and Canada",

            "answer":
                "The evidence connects Japan and Canada [1] [2].",

            "citations":
                [1, 2],

            "context_count":
                3,

            "citation_context":
                [
                    {
                        "citation_id":
                            1,

                        "title":
                            "Japan",

                        "chunk_id":
                            "chunk-1",

                        "content":
                            "The band is from Japan.",
                    },

                    {
                        "citation_id":
                            2,

                        "title":
                            "Canada",

                        "chunk_id":
                            "chunk-2",

                        "content":
                            "The singer is from Canada.",
                    },

                    {
                        "citation_id":
                            3,

                        "title":
                            "France",

                        "chunk_id":
                            "chunk-3",

                        "content":
                            "Another entity is from France.",
                    },
                ],

            "supporting_titles":
                [
                    "Japan",
                    "Canada",
                ],
        }
    ]

    evaluator = RAGEvaluator(
        results
    )

    summary = evaluator.evaluate()

    assert (
        summary.question_count
        == 1
    )

    assert (
        summary.citation_validity
        == 1.0
    )

    assert (
        summary.citation_title_precision
        == 1.0
    )

    assert (
        summary.citation_title_recall
        == 1.0
    )

    assert (
        summary.citation_coverage
        == 1.0
    )


# ================================================================================
# TITLE NORMALIZATION
# ================================================================================


def test_rag_evaluator_title_normalization():

    results = [
        {
            "question":
                "Where does Buck-Tick hail from?",

            "gold_answer":
                "Japan",

            "answer":
                "Buck-Tick hails from Japan [1].",

            "citations":
                [1],

            "context_count":
                1,

            "citation_context":
                [
                    {
                        "citation_id":
                            1,

                        "title":
                            "  JAPAN  ",

                        "chunk_id":
                            "chunk-1",

                        "content":
                            "Buck-Tick is a Japanese rock band.",
                    }
                ],

            "supporting_titles":
                [
                    "Japan"
                ],
        }
    ]

    evaluator = RAGEvaluator(
        results
    )

    summary = evaluator.evaluate()

    assert (
        summary.citation_title_precision
        == 1.0
    )

    assert (
        summary.citation_title_recall
        == 1.0
    )


# ================================================================================
# EVIDENCE OVERLAP
# ================================================================================


def test_rag_evaluator_evidence_overlap():

    results = [
        {
            "question":
                "Where does Buck-Tick hail from?",

            "gold_answer":
                "Japan",

            "answer":
                "Buck-Tick hails from Japan [1].",

            "citations":
                [1],

            "context_count":
                1,

            "citation_context":
                [
                    {
                        "citation_id":
                            1,

                        "title":
                            "Japan",

                        "chunk_id":
                            "chunk-1",

                        "content":
                            "Buck-Tick is a Japanese rock band from Japan.",
                    }
                ],

            "supporting_titles":
                ["Japan"],
        }
    ]

    evaluator = RAGEvaluator(
        results
    )

    summary = evaluator.evaluate()

    assert (
        summary.evidence_overlap
        > 0.0
    )


# ================================================================================
# FINAL SUMMARY TYPE
# ================================================================================


def test_rag_evaluator_returns_summary_object():

    evaluator = RAGEvaluator(
        build_sample_results()
    )

    summary = evaluator.evaluate()

    assert isinstance(
        summary,
        EvaluationSummary,
    )