# app\models\embedding_model.py

from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper around a Sentence Transformers embedding model.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    @property
    def dimension(self) -> int:
        """
        Return the embedding dimension.
        """
        return self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a sequence of texts.
        """

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Generate a normalized embedding for one query.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query must not be empty."
            )

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )