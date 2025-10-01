from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .embeddings import SentenceTransformerEmbedder
from .clustering import run_bertopic
from .cohesion import assess_cohesion
from .topic_selection import select_topics


def collect_news_via_finnhub(cfg: PipelineConfig, symbol: str) -> pd.DataFrame:
    # Placeholder: user should implement Finnhub fetch using their token.
    # Expected columns: [time: Timestamp, title: str, summary: str]
    # Combine title + summary into text externally or later.
    raise NotImplementedError("Implement Finnhub news retrieval with titles and summaries.")


def prepare_texts(news_df: pd.DataFrame) -> Tuple[List[str], List[pd.Timestamp]]:
    texts: List[str] = []
    timestamps: List[pd.Timestamp] = []
    for _, row in news_df.iterrows():
        title = str(row.get("title", "")).strip()
        summary = str(row.get("summary", "")).strip()
        text = (title + ". " + summary).strip()
        if not text:
            continue
        texts.append(text)
        timestamps.append(pd.to_datetime(row["time"]))
    return texts, timestamps


def run_preprocessing_for_symbol(cfg: PipelineConfig, symbol: str) -> Dict[str, object]:
    news_df = collect_news_via_finnhub(cfg, symbol)
    texts, timestamps = prepare_texts(news_df)
    if not texts:
        return {"symbol": symbol, "selected_topics": [], "cohesions": []}

    embedder = SentenceTransformerEmbedder(
        model_name=cfg.embeddings.model_name,
        device=cfg.embeddings.device,
        batch_size=cfg.embeddings.batch_size,
        normalize=cfg.embeddings.normalize_embeddings,
    )
    emb = embedder.encode(texts)

    clustered = run_bertopic(texts, emb.vectors, cfg.clustering)
    cohesions = assess_cohesion(
        topic_to_docs=clustered.topic_to_docs,
        embeddings=emb.vectors,
        timestamps=timestamps,
        representatives=clustered.topic_representatives,
        cfg=cfg.clustering,
    )
    selected = select_topics(cohesions, cfg.clustering)

    return {
        "symbol": symbol,
        "selected_topics": [asdict(s) for s in selected],
        "cohesions": [asdict(c) for c in cohesions],
    }


def run_pipeline(cfg: PipelineConfig) -> List[Dict[str, object]]:
    outputs: List[Dict[str, object]] = []
    for symbol in cfg.symbols:
        outputs.append(run_preprocessing_for_symbol(cfg, symbol))
    return outputs

