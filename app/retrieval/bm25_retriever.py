# app/retrieval/bm25_retriever.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import bm25s


@dataclass
class BM25RetrievalResult:
    rank: int
    score: float
    document: Dict[str, Any]


class BM25Retriever:
    """
    Sparse BM25 retriever over chunk-level documents.

    The retriever keeps the original chunk metadata alongside
    the BM25 index so retrieved results can be traced back to
    their source chunk and document.
    """

    def __init__(
        self,
        retriever: Any,
        documents: Sequence[Dict[str, Any]],
    ) -> None:

        if retriever is None:
            raise ValueError(
                "BM25 retriever cannot be None."
            )

        if documents is None:
            raise ValueError(
                "Documents cannot be None."
            )

        self.retriever = retriever
        self.documents = list(documents)

        if not self.documents:
            raise ValueError(
                "Documents cannot be empty."
            )

    @staticmethod
    def tokenize(
        texts: Sequence[str],
    ) -> List[List[str]]:
        """
        Tokenize text using bm25s tokenization.

        Returns one token list per input text.
        """

        if texts is None:
            raise ValueError(
                "Texts cannot be None."
            )

        normalized_texts = []

        for text in texts:

            if not isinstance(text, str):
                raise TypeError(
                    "Every text must be a string."
                )

            normalized_texts.append(
                text.strip()
            )

        return bm25s.tokenize(
            normalized_texts,
            stopwords="english",
            show_progress=False,
        )

    @classmethod
    def build(
        cls,
        documents: Sequence[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Retriever":
        """
        Build an in-memory BM25 retriever from chunk documents.

        Each document must contain a 'content' field.
        """

        if documents is None:
            raise ValueError(
                "Documents cannot be None."
            )

        documents = list(documents)

        if not documents:
            raise ValueError(
                "Documents cannot be empty."
            )

        if k1 <= 0:
            raise ValueError(
                "k1 must be greater than zero."
            )

        if not 0 <= b <= 1:
            raise ValueError(
                "b must be between 0 and 1."
            )

        corpus = []

        for index, document in enumerate(documents):

            if not isinstance(document, dict):
                raise TypeError(
                    f"Document at index {index} "
                    "must be a dictionary."
                )

            content = document.get(
                "content"
            )

            if not isinstance(content, str):
                raise ValueError(
                    f"Document at index {index} "
                    "does not contain valid text "
                    "in the 'content' field."
                )

            corpus.append(
                content.strip()
            )

        corpus_tokens = cls.tokenize(
            corpus
        )

        bm25 = bm25s.BM25(
            k1=k1,
            b=b,
        )

        bm25.index(
            corpus_tokens,
            show_progress=True,
        )

        return cls(
            retriever=bm25,
            documents=documents,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[BM25RetrievalResult]:
        """
        Retrieve the top-k chunk documents for a query.
        """

        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if not isinstance(top_k, int):
            raise TypeError(
                "top_k must be an integer."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        actual_k = min(
            top_k,
            len(self.documents),
        )

        query_tokens = self.tokenize(
            [query]
        )

        results, scores = self.retriever.retrieve(
            query_tokens,
            k=actual_k,
            show_progress=False,
        )

        results = results[0]
        scores = scores[0]

        retrieval_results = []

        for rank, (
            document_index,
            score,
        ) in enumerate(
            zip(results, scores),
            start=1,
        ):

            document_index = int(
                document_index
            )

            if document_index < 0:
                continue

            if document_index >= len(
                self.documents
            ):
                raise RuntimeError(
                    "BM25 returned an invalid "
                    "document index."
                )

            retrieval_results.append(
                BM25RetrievalResult(
                    rank=rank,
                    score=float(score),
                    document=self.documents[
                        document_index
                    ],
                )
            )

        return retrieval_results

    def save(
        self,
        directory: str | Path,
    ) -> None:
        """
        Save the BM25 index to a directory.
        """

        directory = Path(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.retriever.save(
            str(directory)
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        documents: Sequence[Dict[str, Any]],
    ) -> "BM25Retriever":
        """
        Load a previously saved BM25 index.
        """

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"BM25 index directory not found:\n"
                f"{directory}"
            )

        documents = list(documents)

        if not documents:
            raise ValueError(
                "Documents cannot be empty."
            )

        bm25 = bm25s.BM25.load(
            str(directory)
        )

        return cls(
            retriever=bm25,
            documents=documents,
        )