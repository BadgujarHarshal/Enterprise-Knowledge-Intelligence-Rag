# tests/evaluation/test_citation_metrics.py

from app.evaluation.citation_metrics import (
    extract_citations,
    citation_validity,
    citation_title_precision,
    citation_title_recall,
    citation_coverage,
    evidence_overlap,
)


def make_context(title, content):

    return {
        "document": {
            "title": title,
            "content": content,
        }
    }


def test_extract_citations():

    result = extract_citations(
        "Japan is correct [1].",
        2,
    )

    assert result == [1]


def test_invalid_citation_is_removed():

    result = extract_citations(
        "Japan [9].",
        2,
    )

    assert result == []


def test_citation_validity():

    assert (
        citation_validity(
            [1],
            2,
        )
        == 1.0
    )


def test_citation_title_precision():

    contexts = [
        make_context(
            "Japan",
            "Buck-Tick is from Japan.",
        ),
        make_context(
            "Canada",
            "Hayden is from Canada.",
        ),
    ]

    score = citation_title_precision(
        [1],
        contexts,
        ["Japan"],
    )

    assert score == 1.0


def test_citation_title_precision_wrong_title():

    contexts = [
        make_context(
            "Canada",
            "Hayden is from Canada.",
        ),
    ]

    score = citation_title_precision(
        [1],
        contexts,
        ["Japan"],
    )

    assert score == 0.0


def test_citation_title_recall():

    contexts = [
        make_context(
            "Japan",
            "Buck-Tick is from Japan.",
        ),
        make_context(
            "Canada",
            "Hayden is from Canada.",
        ),
    ]

    score = citation_title_recall(
        [1],
        contexts,
        ["Japan"],
    )

    assert score == 1.0


def test_citation_coverage():

    assert (
        citation_coverage(
            [1]
        )
        == 1.0
    )

    assert (
        citation_coverage(
            []
        )
        == 0.0
    )


def test_evidence_overlap():

    contexts = [
        make_context(
            "Japan",
            "Buck-Tick is a Japanese rock band.",
        )
    ]

    score = evidence_overlap(
        "Buck-Tick is Japanese.",
        contexts,
    )

    assert score > 0.0