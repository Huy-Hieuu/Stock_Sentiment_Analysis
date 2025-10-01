from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

import requests


class RateLimiter:
    def __init__(self, calls_per_minute: int) -> None:
        self.interval_seconds = 60.0 / max(1, calls_per_minute)
        self._last_call_ts: float = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self._last_call_ts
        if elapsed < self.interval_seconds:
            time.sleep(self.interval_seconds - elapsed)
        self._last_call_ts = time.time()


def http_get_json(url: str, params: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def chunked(iterable: Iterable[Any], chunk_size: int) -> Iterable[List[Any]]:
    chunk: List[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

