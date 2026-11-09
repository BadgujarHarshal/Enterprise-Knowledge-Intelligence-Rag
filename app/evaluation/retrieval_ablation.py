# app/evaluation/retrieval_ablation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class AblationConfig:
    """
    Configuration for one retrieval ablation experiment.
    """

    name: str
    dense_weight: float
    bm25_weight: float
    rrf_k: int
    candidate_k: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")

        if self.dense_weight < 0:
            raise ValueError("dense_weight must be >= 0")

        if self.bm25_weight < 0:
            raise ValueError("bm25_weight must be >= 0")

        if self.dense_weight == 0 and self.bm25_weight == 0:
            raise ValueError(
                "At least one of dense_weight or bm25_weight must be > 0"
            )

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be > 0")

        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    *,
    weights: Sequence[float] | None = None,
    rrf_k: int = 60,
) -> List[int]:
    """
    Fuse ranked document IDs using weighted Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists:
        Ordered document-ID lists. Earlier positions have higher rank.

    weights:
        Optional weight for each ranked list.

    rrf_k:
        RRF smoothing constant.

    Returns
    -------
    List[int]
        Document IDs ordered by fused score.
    """

    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")

    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError(
            "weights length must match ranked_lists length"
        )

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be >= 0")

    scores: Dict[int, float] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        if weight == 0:
            continue

        seen_in_list = set()

        for rank, document_id in enumerate(ranked_list, start=1):
            document_id = int(document_id)

            # Protect against duplicate document IDs inside one ranking.
            if document_id in seen_in_list:
                continue

            seen_in_list.add(document_id)

            contribution = weight / (rrf_k + rank)

            scores[document_id] = (
                scores.get(document_id, 0.0) + contribution
            )

    ranked_documents = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return [document_id for document_id, _ in ranked_documents]


def validate_ablation_configs(
    configs: Iterable[AblationConfig],
) -> List[AblationConfig]:
    """
    Validate and normalize an ablation configuration collection.
    """

    configs = list(configs)

    if not configs:
        raise ValueError("At least one ablation configuration is required")

    names = [config.name for config in configs]

    if len(names) != len(set(names)):
        raise ValueError("Ablation configuration names must be unique")

    return configs