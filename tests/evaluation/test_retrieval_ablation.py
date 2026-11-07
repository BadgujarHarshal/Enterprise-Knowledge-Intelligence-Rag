# tests/evaluation/test_retrieval_ablation.py

import pytest

from app.evaluation.retrieval_ablation import (
    AblationConfig,
    reciprocal_rank_fusion,
    validate_ablation_configs,
)


def test_rrf_returns_expected_order():
    result = reciprocal_rank_fusion(
        [
            [1, 2, 3],
            [2, 3, 4],
        ],
        rrf_k=60,
    )

    assert result[0] == 2
    assert result[1] == 3
    assert 1 in result
    assert 4 in result


def test_weighted_rrf_can_overcome_overlap_advantage():
    """
    A sufficiently strong weight on the first ranking should allow
    its unique top-ranked document to outrank a document that appears
    in both rankings.

    With rrf_k=60 and weights [63, 1]:

        doc 1 = 63 / 61

        doc 2 = 63 / 62 + 1 / 61

    Therefore doc 1 should rank first.
    """

    result = reciprocal_rank_fusion(
        [
            [1, 2, 3],
            [2, 3, 4],
        ],
        weights=[63.0, 1.0],
        rrf_k=60,
    )

    assert result[0] == 1


def test_weighted_rrf_preserves_shared_documents():
    """
    A document appearing in both rankings should still receive
    contributions from both rankings.
    """

    unweighted_result = reciprocal_rank_fusion(
        [
            [1, 2, 3],
            [2, 3, 4],
        ],
        weights=[1.0, 1.0],
        rrf_k=60,
    )

    weighted_result = reciprocal_rank_fusion(
        [
            [1, 2, 3],
            [2, 3, 4],
        ],
        weights=[5.0, 1.0],
        rrf_k=60,
    )

    assert 2 in unweighted_result
    assert 2 in weighted_result

    assert 3 in unweighted_result
    assert 3 in weighted_result


def test_rrf_duplicate_document_in_one_list_is_counted_once():
    result = reciprocal_rank_fusion(
        [
            [1, 1, 2],
            [2, 3],
        ],
        rrf_k=60,
    )

    assert result.count(1) == 1
    assert result.count(2) == 1
    assert result.count(3) == 1


def test_rrf_invalid_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [[1, 2, 3]],
            rrf_k=0,
        )


def test_rrf_invalid_weight_length():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [[1, 2, 3], [4, 5, 6]],
            weights=[1.0],
        )


def test_rrf_negative_weight():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [[1, 2, 3]],
            weights=[-1.0],
        )


def test_ablation_config_valid():
    config = AblationConfig(
        name="hybrid_equal",
        dense_weight=1.0,
        bm25_weight=1.0,
        rrf_k=60,
        candidate_k=100,
    )

    assert config.name == "hybrid_equal"
    assert config.dense_weight == 1.0
    assert config.bm25_weight == 1.0
    assert config.rrf_k == 60
    assert config.candidate_k == 100


def test_ablation_config_rejects_zero_weights():
    with pytest.raises(ValueError):
        AblationConfig(
            name="invalid",
            dense_weight=0.0,
            bm25_weight=0.0,
            rrf_k=60,
            candidate_k=100,
        )


def test_ablation_config_rejects_invalid_candidate_k():
    with pytest.raises(ValueError):
        AblationConfig(
            name="invalid",
            dense_weight=1.0,
            bm25_weight=1.0,
            rrf_k=60,
            candidate_k=0,
        )


def test_ablation_config_rejects_invalid_rrf_k():
    with pytest.raises(ValueError):
        AblationConfig(
            name="invalid",
            dense_weight=1.0,
            bm25_weight=1.0,
            rrf_k=0,
            candidate_k=100,
        )


def test_validate_configs_requires_unique_names():
    configs = [
        AblationConfig(
            name="same",
            dense_weight=1.0,
            bm25_weight=1.0,
            rrf_k=60,
            candidate_k=100,
        ),
        AblationConfig(
            name="same",
            dense_weight=2.0,
            bm25_weight=1.0,
            rrf_k=60,
            candidate_k=100,
        ),
    ]

    with pytest.raises(ValueError):
        validate_ablation_configs(configs)


def test_validate_configs_rejects_empty():
    with pytest.raises(ValueError):
        validate_ablation_configs([])