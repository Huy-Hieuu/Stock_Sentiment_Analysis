from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .cohesion import ClusterCohesion
from .config import ClusteringConfig


@dataclass
class SelectedTopic:
    topic_id: int
    representative_doc_index: int
    cluster_size: int
    time_start: pd.Timestamp
    time_end: pd.Timestamp
    avg_pairwise_similarity: float
    is_high_cohesion: bool


def select_topics(cohesions: List[ClusterCohesion], cfg: ClusteringConfig) -> List[SelectedTopic]:
    high = [c for c in cohesions if c.is_high_cohesion]
    low = [c for c in cohesions if not c.is_high_cohesion]

    # Sort by similarity desc, then size desc
    high.sort(key=lambda c: (c.avg_pairwise_similarity, c.cluster_size), reverse=True)
    low.sort(key=lambda c: (c.avg_pairwise_similarity, c.cluster_size), reverse=True)

    selected: List[SelectedTopic] = []

    # Always include high-cohesion topics first
    for c in high:
        selected.append(
            SelectedTopic(
                topic_id=c.topic_id,
                representative_doc_index=c.representative_doc_index,
                cluster_size=c.cluster_size,
                time_start=c.time_start,
                time_end=c.time_end,
                avg_pairwise_similarity=c.avg_pairwise_similarity,
                is_high_cohesion=True,
            )
        )

    # If fewer than required, fill with up to max_low_cohesion_topics_to_fill
    if len(selected) < cfg.min_high_cohesion_topics:
        needed = cfg.min_high_cohesion_topics - len(selected)
        fill_count = min(needed, cfg.max_low_cohesion_topics_to_fill)
        for c in low[:fill_count]:
            # Enforce low-cohesion cluster size cap
            size_capped = min(c.cluster_size, cfg.low_cohesion_max_size)
            selected.append(
                SelectedTopic(
                    topic_id=c.topic_id,
                    representative_doc_index=c.representative_doc_index,
                    cluster_size=size_capped,
                    time_start=c.time_start,
                    time_end=c.time_end,
                    avg_pairwise_similarity=c.avg_pairwise_similarity,
                    is_high_cohesion=False,
                )
            )

    return selected

