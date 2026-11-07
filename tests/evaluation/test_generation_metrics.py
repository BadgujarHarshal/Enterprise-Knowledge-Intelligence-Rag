# tests/evaluation/test_generation_metrics.py

from app.evaluation.generation_metrics import (
    exact_match,
    token_f1,
    answer_metrics,
)


def test_exact_match():

    assert (
        exact_match(
            "Japan",
            "Japan",
        )
        == 1.0
    )


def test_exact_match_is_case_insensitive():

    assert (
        exact_match(
            "JAPAN",
            "Japan",
        )
        == 1.0
    )


def test_exact_match_false():

    assert (
        exact_match(
            "Canada",
            "Japan",
        )
        == 0.0
    )


def test_token_f1_exact():

    assert (
        token_f1(
            "Buck Tick",
            "Buck Tick",
        )
        == 1.0
    )


def test_token_f1_partial():

    score = token_f1(
        "Buck Tick Japan",
        "Japan",
    )

    assert 0.0 < score < 1.0


def test_token_f1_no_overlap():

    assert (
        token_f1(
            "Canada",
            "Japan",
        )
        == 0.0
    )


def test_answer_metrics():

    result = answer_metrics(
        "Japan",
        "Japan",
    )

    assert (
        result[
            "exact_match"
        ]
        == 1.0
    )

    assert (
        result[
            "f1"
        ]
        == 1.0
    )