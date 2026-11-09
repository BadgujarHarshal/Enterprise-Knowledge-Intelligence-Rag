# app\retrieval\faiss_index.py

from pathlib import Path
from typing import Sequence

import faiss
import numpy as np


class FAISSIndex:
    """
    FAISS vector index for dense retrieval.

    Uses cosine similarity through normalized embeddings
    and inner-product search.
    """

    def __init__(
        self,
        dimension: int,
    ):
        if dimension <= 0:
            raise ValueError(
                "Embedding dimension must be positive."
            )

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

    @property
    def size(self) -> int:
        """
        Number of vectors currently stored.
        """
        return self.index.ntotal

    def add(
        self,
        embeddings: np.ndarray,
    ) -> None:
        """
        Add embeddings to the FAISS index.
        """

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"received {embeddings.shape[1]}."
            )

        if embeddings.shape[0] == 0:
            return

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ):
        """
        Search the index using one query embedding.

        Returns:
            scores, indices
        """

        if self.size == 0:
            raise ValueError(
                "Cannot search an empty FAISS index."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(
                1,
                -1,
            )

        if query_embedding.ndim != 2:
            raise ValueError(
                "Query embedding must be 1D or 2D."
            )

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query dimension "
                f"{self.dimension}, "
                f"received {query_embedding.shape[1]}."
            )

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.size),
        )

        return scores, indices

    def save(
        self,
        file_path: str | Path,
    ) -> None:
        """
        Save the FAISS index to disk.
        """

        file_path = Path(file_path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(file_path),
        )

    @classmethod
    def load(
        cls,
        file_path: str | Path,
    ):
        """
        Load a previously saved FAISS index.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {file_path}"
            )

        index = faiss.read_index(
            str(file_path)
        )

        instance = cls(
            dimension=index.d,
        )

        instance.index = index

        return instance