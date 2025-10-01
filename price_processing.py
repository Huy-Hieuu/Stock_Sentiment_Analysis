from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .config import PriceProcessingConfig


@dataclass
class PriceData:
    symbol: str
    daily: pd.DataFrame  # index: date, columns: [close, return]
    weekly: pd.DataFrame  # index: week_start, columns: [close, return]


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if "close" not in prices.columns:
        raise ValueError("prices must contain 'close' column")
    prices = prices.sort_index()
    prices["return"] = prices["close"].pct_change().fillna(0.0)
    return prices


def resample_to_weekly(prices: pd.DataFrame) -> pd.DataFrame:
    weekly_close = prices["close"].resample("W-FRI").last()
    weekly_return = weekly_close.pct_change().fillna(0.0)
    weekly = pd.DataFrame({"close": weekly_close, "return": weekly_return})
    return weekly


def build_price_data(symbol_to_prices: Dict[str, pd.DataFrame], cfg: PriceProcessingConfig) -> List[PriceData]:
    result: List[PriceData] = []
    for symbol, df in symbol_to_prices.items():
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("price data index must be DatetimeIndex")
        df = compute_daily_returns(df)
        weekly = resample_to_weekly(df)
        result.append(PriceData(symbol=symbol, daily=df, weekly=weekly))
    return result

