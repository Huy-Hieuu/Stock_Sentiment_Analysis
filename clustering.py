from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ClusteringConfig


@dataclass
class ClusteredTopics:
    topics_df: pd.DataFrame  # columns: [topic_id, document_index, text]
    topic_to_docs: Dict[int, List[int]]
    topic_representatives: Dict[int, int]  # topic_id -> document_index of representative


def compute_pairwise_similarity(embeddings: np.ndarray) -> np.ndarray:
    # embeddings assumed to be L2-normalized if required
    return np.clip(embeddings @ embeddings.T, -1.0, 1.0)


def average_pairwise_similarity(embeddings: np.ndarray, doc_indices: List[int]) -> float:
    if len(doc_indices) <= 1:
        return 1.0
    sub = embeddings[doc_indices]
    sim = compute_pairwise_similarity(sub)
    n = sim.shape[0]
    # exclude diagonal
    return float((sim.sum() - np.trace(sim)) / (n * (n - 1)))


def select_centroid_representative(embeddings: np.ndarray, doc_indices: List[int]) -> int:
    sub = embeddings[doc_indices]
    centroid = sub.mean(axis=0, keepdims=True)
    # cosine sim since embeddings are normalized
    sims = (sub @ centroid.T).ravel()
    best_idx = int(np.argmax(sims))
    return doc_indices[best_idx]


def run_bertopic(texts: List[str], embeddings: np.ndarray, cfg: ClusteringConfig) -> ClusteredTopics:
    from bertopic import BERTopic

    model = BERTopic(language=cfg.language, top_n_words=cfg.top_n_words, min_topic_size=cfg.min_topic_size)
    topics, _ = model.fit_transform(texts, embeddings)

    topic_to_docs: Dict[int, List[int]] = {}
    for doc_idx, topic_id in enumerate(topics):
        topic_to_docs.setdefault(topic_id, []).append(doc_idx)

    # Remove outlier topic -1 if exists
    if -1 in topic_to_docs:
        del topic_to_docs[-1]

    representatives: Dict[int, int] = {}
    rows: List[Tuple[int, int, str]] = []
    for topic_id, doc_indices in topic_to_docs.items():
        rep = select_centroid_representative(embeddings, doc_indices)
        representatives[topic_id] = rep
        for di in doc_indices:
            rows.append((topic_id, di, texts[di]))

    topics_df = pd.DataFrame(rows, columns=["topic_id", "document_index", "text"])  # type: ignore
    return ClusteredTopics(topics_df=topics_df, topic_to_docs=topic_to_docs, topic_representatives=representatives)

