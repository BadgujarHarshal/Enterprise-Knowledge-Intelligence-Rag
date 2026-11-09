# app/reranking/cross_encoder_reranker.py

"""
Cross-Encoder Reranker for EKI-RAG.

The reranker receives an already-retrieved candidate set and scores
(query, document) pairs using a pretrained Sentence Transformers
CrossEncoder model.

Architecture:

    Query
      |
      v
    Initial Retriever
      |
      v
    Candidate Documents
      |
      v
    CrossEncoderReranker
      |
      v
    Re-ranked Documents

The reranker does NOT perform corpus-wide retrieval.
It only reranks candidates supplied by the retrieval layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


@dataclass(frozen=True)
class RerankingResult:
    """
    Represents one reranked document.

    Attributes:
        rank:
            Final reranked position, starting from 1.

        score:
            Cross-encoder relevance score.

        document:
            Original document/chunk object.
    """

    rank: int
    score: float
    document: object


class CrossEncoderReranker:
    """
    Cross-Encoder based reranker.

    The class intentionally keeps model loading separate from scoring
    so that the same model instance can be reused for many queries.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        max_length: int | None = None,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize the reranker.

        Args:
            model_name:
                Hugging Face / Sentence Transformers CrossEncoder model.

            device:
                Optional device such as "cuda", "cpu", or "mps".
                If None, Sentence Transformers selects the available device.

            max_length:
                Optional maximum input sequence length.

            batch_size:
                Number of query-document pairs processed per batch.
        """

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker."
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size

        model_kwargs = {}

        if device is not None:
            self.model = CrossEncoder(
                model_name,
                device=device,
                max_length=max_length,
            )
        else:
            self.model = CrossEncoder(
                model_name,
                max_length=max_length,
            )

    def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
    ) -> np.ndarray:
        """
        Score a query against multiple documents.

        Args:
            query:
                User query.

            documents:
                Candidate document texts.

        Returns:
            NumPy array containing one score per document.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if documents is None:
            raise ValueError("documents cannot be None.")

        if len(documents) == 0:
            return np.asarray([], dtype=np.float32)

        for index, document in enumerate(documents):
            if not isinstance(document, str):
                raise TypeError(
                    f"documents[{index}] must be a string, "
                    f"got {type(document).__name__}."
                )

        pairs = [(query, document) for document in documents]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        if len(scores) != len(documents):
            raise RuntimeError(
                "Cross-encoder returned an unexpected number of scores. "
                f"Expected {len(documents)}, got {len(scores)}."
            )

        return scores

    def rerank(
        self,
        query: str,
        candidates: Sequence[object],
        *,
        document_text_getter=None,
        top_k: int | None = None,
    ) -> list[RerankingResult]:
        """
        Rerank retrieved candidate documents.

        Args:
            query:
                User query.

            candidates:
                Candidate document objects returned by a retriever.

            document_text_getter:
                Function that converts a candidate object into the text
                passed to the CrossEncoder.

                If omitted, the method attempts to use:

                    candidate.text

                or, for dictionaries:

                    candidate["text"]

            top_k:
                Optional number of reranked results to return.

        Returns:
            List of RerankingResult ordered by descending score.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if candidates is None:
            raise ValueError("candidates cannot be None.")

        candidates = list(candidates)

        if not candidates:
            return []

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero when provided.")

        if document_text_getter is None:
            document_text_getter = self._default_document_text_getter

        documents = [
            document_text_getter(candidate)
            for candidate in candidates
        ]

        scores = self.score_pairs(
            query=query,
            documents=documents,
        )

        scored_candidates = list(
            zip(
                candidates,
                scores.tolist(),
            )
        )

        # Stable deterministic ordering:
        #
        # 1. Higher CrossEncoder score first.
        # 2. Original candidate position as tie-breaker.
        #
        # Python's stable sort preserves the original order for equal
        # scores, which gives deterministic behavior.
        scored_candidates = sorted(
            enumerate(scored_candidates),
            key=lambda item: (-float(item[1][1]), item[0]),
        )

        if top_k is not None:
            scored_candidates = scored_candidates[:top_k]

        results: list[RerankingResult] = []

        for rank, (_, (candidate, score)) in enumerate(
            scored_candidates,
            start=1,
        ):
            results.append(
                RerankingResult(
                    rank=rank,
                    score=float(score),
                    document=candidate,
                )
            )

        return results

    @staticmethod
    def _default_document_text_getter(candidate: object) -> str:
        """
        Extract document text from a candidate object.

        Supports:

            candidate.text

        and:

            candidate["text"]
        """

        if isinstance(candidate, dict):
            if "text" not in candidate:
                raise ValueError(
                    "Candidate dictionary must contain a 'text' field."
                )

            text = candidate["text"]

        else:
            if not hasattr(candidate, "text"):
                raise ValueError(
                    "Candidate object must expose a 'text' attribute."
                )

            text = candidate.text

        if not isinstance(text, str):
            raise TypeError(
                "Candidate document text must be a string."
            )

        if not text.strip():
            raise ValueError(
                "Candidate document text cannot be empty."
            )

        return text