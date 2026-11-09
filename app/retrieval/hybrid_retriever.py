# app/retrieval/hybrid_retriever.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from app.retrieval.rank_fusion import reciprocal_rank_fusion


@dataclass(frozen=True)
class HybridRetrievalResult:
    rank: int
    score: float
    document: Dict[str, Any]


class HybridRetriever:
    """
    Hybrid dense + BM25 retriever using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        dense_retriever: Any,
        bm25_retriever: Any,
        *,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:

        if dense_retriever is None:
            raise ValueError(
                "dense_retriever cannot be None."
            )

        if bm25_retriever is None:
            raise ValueError(
                "bm25_retriever cannot be None."
            )

        if not isinstance(rrf_k, int):
            raise TypeError(
                "rrf_k must be an integer."
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        if dense_weight < 0:
            raise ValueError(
                "dense_weight cannot be negative."
            )

        if bm25_weight < 0:
            raise ValueError(
                "bm25_weight cannot be negative."
            )

        if dense_weight == 0 and bm25_weight == 0:
            raise ValueError(
                "At least one retriever weight must be greater than zero."
            )

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.dense_weight = float(dense_weight)
        self.bm25_weight = float(bm25_weight)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        candidate_k: int | None = None,
    ) -> List[HybridRetrievalResult]:

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if not isinstance(top_k, int):
            raise TypeError(
                "top_k must be an integer."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if candidate_k is None:
            candidate_k = max(top_k, 20)

        if not isinstance(candidate_k, int):
            raise TypeError(
                "candidate_k must be an integer."
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than zero."
            )

        dense_results = self.dense_retriever.retrieve(
            query,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_retriever.retrieve(
            query,
            top_k=candidate_k,
        )

        fused_results = reciprocal_rank_fusion(
            [
                dense_results,
                bm25_results,
            ],
            k=self.rrf_k,
            weights=[
                self.dense_weight,
                self.bm25_weight,
            ],
            top_k=top_k,
        )

        return [
            HybridRetrievalResult(
                rank=result.rank,
                score=result.score,
                document=result.document,
            )
            for result in fused_results
        ]