# app/evaluation/reranker_evaluation.py

"""
Evaluation utilities for EKI-RAG reranking.

This module evaluates ranked document results against HotpotQA-style
supporting titles.

The evaluator operates on already-produced ranked results. It does not
perform retrieval or model inference itself.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence


def extract_supporting_titles(record: dict) -> set[str]:
    """
    Extract supporting titles from a HotpotQA record.

    The current project dataset uses:

        supporting_facts["title"]

    Returns:
        Set of unique supporting document titles.
    """

    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary.")

    supporting_facts = record.get("supporting_facts")

    if not isinstance(supporting_facts, dict):
        raise ValueError(
            "record['supporting_facts'] must be a dictionary."
        )

    titles = supporting_facts.get("title")

    if not isinstance(titles, list):
        raise ValueError(
            "record['supporting_facts']['title'] must be a list."
        )

    cleaned_titles = {
        title.strip()
        for title in titles
        if isinstance(title, str) and title.strip()
    }

    if not cleaned_titles:
        raise ValueError(
            "No supporting titles were found."
        )

    return cleaned_titles


def extract_title_from_candidate(
    candidate: object,
) -> str:
    """
    Extract a document title from a candidate.

    Supports:

        candidate["title"]

    or:

        candidate.title
    """

    if isinstance(candidate, dict):
        if "title" not in candidate:
            raise ValueError(
                "Candidate dictionary must contain a 'title' field."
            )

        title = candidate["title"]

    else:
        if not hasattr(candidate, "title"):
            raise ValueError(
                "Candidate object must expose a 'title' attribute."
            )

        title = candidate.title

    if not isinstance(title, str) or not title.strip():
        raise ValueError(
            "Candidate title must be a non-empty string."
        )

    return title.strip()


def hit_at_k(
    ranked_candidates: Sequence[object],
    gold_titles: set[str],
    k: int,
) -> float:
    """
    Return 1.0 when at least one supporting title appears in top-k,
    otherwise 0.0.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not gold_titles:
        raise ValueError("gold_titles cannot be empty.")

    top_candidates = ranked_candidates[:k]

    retrieved_titles = {
        extract_title_from_candidate(candidate)
        for candidate in top_candidates
    }

    return 1.0 if retrieved_titles.intersection(gold_titles) else 0.0


def coverage_at_k(
    ranked_candidates: Sequence[object],
    gold_titles: set[str],
    k: int,
) -> float:
    """
    Fraction of supporting titles retrieved within top-k.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not gold_titles:
        raise ValueError("gold_titles cannot be empty.")

    top_candidates = ranked_candidates[:k]

    retrieved_titles = {
        extract_title_from_candidate(candidate)
        for candidate in top_candidates
    }

    found = retrieved_titles.intersection(gold_titles)

    return len(found) / len(gold_titles)


def reciprocal_rank_at_k(
    ranked_candidates: Sequence[object],
    gold_titles: set[str],
    k: int,
) -> float:
    """
    Reciprocal rank of the first supporting title within top-k.

    Returns 0.0 when no supporting title occurs within top-k.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not gold_titles:
        raise ValueError("gold_titles cannot be empty.")

    for rank, candidate in enumerate(
        ranked_candidates[:k],
        start=1,
    ):
        title = extract_title_from_candidate(candidate)

        if title in gold_titles:
            return 1.0 / rank

    return 0.0


def evaluate_ranked_results(
    ranked_results: Sequence[Sequence[object]],
    evaluation_records: Sequence[dict],
    *,
    ks: Iterable[int] = (5, 10, 20),
) -> dict[str, float]:
    """
    Evaluate a complete ranked-result collection.

    ranked_results[i] corresponds to evaluation_records[i].
    """

    if len(ranked_results) != len(evaluation_records):
        raise ValueError(
            "ranked_results and evaluation_records must have the same length."
        )

    ks = tuple(sorted(set(int(k) for k in ks)))

    if not ks:
        raise ValueError("At least one k value is required.")

    if any(k <= 0 for k in ks):
        raise ValueError("All k values must be greater than zero.")

    if not evaluation_records:
        raise ValueError("evaluation_records cannot be empty.")

    totals = {}

    for k in ks:
        totals[f"hit@{k}"] = 0.0
        totals[f"coverage@{k}"] = 0.0
        totals[f"mrr@{k}"] = 0.0

    for ranked_candidates, record in zip(
        ranked_results,
        evaluation_records,
    ):
        gold_titles = extract_supporting_titles(record)

        for k in ks:
            totals[f"hit@{k}"] += hit_at_k(
                ranked_candidates,
                gold_titles,
                k,
            )

            totals[f"coverage@{k}"] += coverage_at_k(
                ranked_candidates,
                gold_titles,
                k,
            )

            totals[f"mrr@{k}"] += reciprocal_rank_at_k(
                ranked_candidates,
                gold_titles,
                k,
            )

    count = float(len(evaluation_records))

    return {
        metric: value / count
        for metric, value in totals.items()
    }