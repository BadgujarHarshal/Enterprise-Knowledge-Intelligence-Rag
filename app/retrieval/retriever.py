# app\retrieval\retriever.py

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from app.models.embedding_model import EmbeddingModel
from app.retrieval.faiss_index import FAISSIndex


@dataclass
class RetrievalResult:
    index: int
    score: float
    document: dict


class DenseRetriever:
    """
    Dense vector retriever using Sentence Transformers
    and FAISS.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        index: FAISSIndex,
        documents: Sequence[dict],
    ):
        if index.size != len(documents):
            raise ValueError(
                "Number of indexed vectors must match "
                "number of documents."
            )

        self.embedding_model = embedding_model
        self.index = index
        self.documents = list(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most similar documents for a query.
        """

        query_embedding = (
            self.embedding_model.encode_query(
                query
            )
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k=top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index < 0:
                continue

            results.append(
                RetrievalResult(
                    index=int(index),
                    score=float(score),
                    document=self.documents[index],
                )
            )

        return results