from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class EmbeddingResult:
    vectors: np.ndarray  # shape: (N, D)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 64, normalize: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.normalize = normalize

    def encode(self, texts: Iterable[str]) -> EmbeddingResult:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return EmbeddingResult(vectors=vectors)

