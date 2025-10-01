from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class FinnhubConfig:
    api_token: str
    base_url: str = "https://finnhub.io/api/v1"
    max_requests_per_minute: int = 50


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None  # e.g., "cpu" or "cuda"
    batch_size: int = 64
    normalize_embeddings: bool = True


@dataclass(frozen=True)
class ClusteringConfig:
    n_grams: int = 1
    top_n_words: int = 10
    min_topic_size: int = 2
    low_cohesion_max_size: int = 2
    high_cohesion_similarity_threshold: float = 0.6
    max_low_cohesion_topics_to_fill: int = 4
    min_high_cohesion_topics: int = 6
    language: str = "english"


@dataclass(frozen=True)
class PriceProcessingConfig:
    # Frequency for returns calculation is daily for HG
    trading_days_per_week: int = 5


@dataclass(frozen=True)
class PipelineConfig:
    symbols: List[str]
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    finnhub: FinnhubConfig
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    price: PriceProcessingConfig = field(default_factory=PriceProcessingConfig)

