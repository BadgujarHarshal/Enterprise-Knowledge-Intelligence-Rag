# tests/evaluation/test_reranker_evaluation.py

"""
Tests for reranker evaluation metrics.
"""

import pytest

from app.evaluation.reranker_evaluation import (
    coverage_at_k,
    evaluate_ranked_results,
    extract_supporting_titles,
    hit_at_k,
    reciprocal_rank_at_k,
)


def make_record(titles):
    return {
        "supporting_facts": {
            "title": titles,
            "sent_id": [[] for _ in titles],
        }
    }


def make_candidate(title):
    return {
        "title": title,
        "text": f"Text for {title}",
    }


def test_extract_supporting_titles():
    record = make_record(
        [
            "Document A",
            "Document B",
        ]
    )

    titles = extract_supporting_titles(record)

    assert titles == {
        "Document A",
        "Document B",
    }


def test_hit_at_k_when_supporting_title_is_present():
    candidates = [
        make_candidate("Wrong"),
        make_candidate("Document B"),
        make_candidate("Another"),
    ]

    gold = {
        "Document B",
    }

    assert hit_at_k(candidates, gold, 2) == 1.0


def test_hit_at_k_when_supporting_title_is_absent():
    candidates = [
        make_candidate("Wrong A"),
        make_candidate("Wrong B"),
    ]

    gold = {
        "Document B",
    }

    assert hit_at_k(candidates, gold, 2) == 0.0


def test_coverage_at_k():
    candidates = [
        make_candidate("Document A"),
        make_candidate("Wrong"),
        make_candidate("Document B"),
        make_candidate("Document C"),
    ]

    gold = {
        "Document A",
        "Document B",
        "Document C",
    }

    assert coverage_at_k(candidates, gold, 3) == pytest.approx(
        2.0 / 3.0
    )


def test_reciprocal_rank_at_k():
    candidates = [
        make_candidate("Wrong"),
        make_candidate("Document B"),
        make_candidate("Document A"),
    ]

    gold = {
        "Document A",
        "Document B",
    }

    assert reciprocal_rank_at_k(
        candidates,
        gold,
        3,
    ) == pytest.approx(0.5)


def test_reciprocal_rank_zero_when_missing():
    candidates = [
        make_candidate("Wrong A"),
        make_candidate("Wrong B"),
    ]

    gold = {
        "Document A",
    }

    assert reciprocal_rank_at_k(
        candidates,
        gold,
        2,
    ) == 0.0


def test_evaluate_ranked_results():
    records = [
        make_record(["Document A"]),
        make_record(["Document B"]),
    ]

    ranked_results = [
        [
            make_candidate("Document A"),
            make_candidate("Wrong"),
        ],
        [
            make_candidate("Wrong"),
            make_candidate("Document B"),
        ],
    ]

    metrics = evaluate_ranked_results(
        ranked_results,
        records,
        ks=(1, 2),
    )

    assert metrics["hit@1"] == pytest.approx(0.5)
    assert metrics["hit@2"] == pytest.approx(1.0)

    assert metrics["coverage@1"] == pytest.approx(0.5)
    assert metrics["coverage@2"] == pytest.approx(1.0)

    assert metrics["mrr@1"] == pytest.approx(0.5)
    assert metrics["mrr@2"] == pytest.approx(0.75)