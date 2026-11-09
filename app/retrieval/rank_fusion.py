# app/retrieval/rank_fusion.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class FusionResult:
    rank: int
    score: float
    document: Dict[str, Any]


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[Any]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_k: int = 10,
) -> List[FusionResult]:
    """
    Combine multiple ranked retrieval lists using Reciprocal Rank Fusion.

    RRF score:

        score(d) = sum_i weight_i / (k + rank_i)

    where rank starts at 1.

    Parameters
    ----------
    result_lists:
        Ranked result lists from different retrievers.

    k:
        RRF constant. 60 is the conventional default.

    weights:
        Optional weight for each retrieval system.
        If omitted, every retriever receives weight 1.0.

    top_k:
        Number of fused results to return.
    """

    if result_lists is None:
        raise ValueError("result_lists cannot be None.")

    if not result_lists:
        raise ValueError("result_lists cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if weights is None:
        weights = [1.0] * len(result_lists)
    else:
        weights = list(weights)

        if len(weights) != len(result_lists):
            raise ValueError(
                "weights length must match result_lists length."
            )

        if any(weight < 0 for weight in weights):
            raise ValueError("weights cannot contain negative values.")

    fused_scores: Dict[str, float] = {}
    documents: Dict[str, Dict[str, Any]] = {}

    for system_index, results in enumerate(result_lists):
        weight = float(weights[system_index])

        if weight == 0:
            continue

        for rank, result in enumerate(results, start=1):
            document = _extract_document(result)

            document_id = _document_identity(document)

            score = weight / (k + rank)

            fused_scores[document_id] = (
                fused_scores.get(document_id, 0.0) + score
            )

            if document_id not in documents:
                documents[document_id] = document

    ranked = sorted(
        fused_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    output: List[FusionResult] = []

    for rank, (document_id, score) in enumerate(
        ranked[:top_k],
        start=1,
    ):
        output.append(
            FusionResult(
                rank=rank,
                score=float(score),
                document=documents[document_id],
            )
        )

    return output


def _extract_document(result: Any) -> Dict[str, Any]:
    """
    Extract the underlying document from a retrieval result.

    Supports:
    - result.document
    - dictionary result containing "document"
    - raw dictionary document
    """

    if hasattr(result, "document"):
        document = result.document
    elif isinstance(result, Mapping) and "document" in result:
        document = result["document"]
    else:
        document = result

    if not isinstance(document, dict):
        raise TypeError(
            "Each retrieval result must contain a dictionary document."
        )

    return document


def _document_identity(document: Dict[str, Any]) -> str:
    """
    Return the stable identity used during fusion.

    chunk_id is preferred because retrieval operates at chunk level.
    """

    chunk_id = document.get("chunk_id")

    if chunk_id is not None:
        return str(chunk_id)

    document_id = document.get("document_id")

    if document_id is not None:
        return str(document_id)

    raise ValueError(
        "Document must contain either 'chunk_id' or 'document_id'."
    )