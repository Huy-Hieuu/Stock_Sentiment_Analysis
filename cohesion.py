from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .clustering import average_pairwise_similarity
from .config import ClusteringConfig


@dataclass
class ClusterCohesion:
    topic_id: int
    avg_pairwise_similarity: float
    representative_doc_index: int
    cluster_size: int
    time_start: pd.Timestamp
    time_end: pd.Timestamp
    is_high_cohesion: bool


def assess_cohesion(
    topic_to_docs: Dict[int, List[int]],
    embeddings: np.ndarray,
    timestamps: List[pd.Timestamp],
    representatives: Dict[int, int],
    cfg: ClusteringConfig,
) -> List[ClusterCohesion]:
    results: List[ClusterCohesion] = []
    for topic_id, doc_indices in topic_to_docs.items():
        avg_sim = average_pairwise_similarity(embeddings, doc_indices)
        rep = representatives[topic_id]
        ts = [timestamps[i] for i in doc_indices]
        time_start = min(ts)
        time_end = max(ts)
        is_high = avg_sim > cfg.high_cohesion_similarity_threshold
        results.append(
            ClusterCohesion(
                topic_id=topic_id,
                avg_pairwise_similarity=float(avg_sim),
                representative_doc_index=rep,
                cluster_size=len(doc_indices),
                time_start=time_start,
                time_end=time_end,
                is_high_cohesion=is_high,
            )
        )
    return results

