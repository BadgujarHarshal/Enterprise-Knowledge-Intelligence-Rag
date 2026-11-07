## tests\retrieval\test_faiss_index.py

import numpy as np
import pytest

from app.retrieval.faiss_index import FAISSIndex


def test_faiss_index_add_and_search():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = FAISSIndex(
        dimension=3
    )

    index.add(embeddings)

    assert index.size == 3

    query = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    scores, indices = index.search(
        query,
        top_k=2,
    )

    assert scores.shape == (1, 2)
    assert indices.shape == (1, 2)

    assert indices[0][0] == 0
    assert scores[0][0] == pytest.approx(1.0)


def test_faiss_dimension_validation():
    index = FAISSIndex(
        dimension=3
    )

    invalid_embeddings = np.array(
        [
            [1.0, 0.0]
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        index.add(invalid_embeddings)