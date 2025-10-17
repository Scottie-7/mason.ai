"""In-memory persistence for the dashboard."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Deque, Dict

import pandas as pd


class DatabaseManager:
    """Store a rolling window of market data snapshots."""

    def __init__(self, max_points: int = 10_000) -> None:
        self._max_points = max_points
        self._storage: Dict[str, Deque[Dict]] = defaultdict(lambda: deque(maxlen=max_points))

    def store_stock_data(self, data: Dict) -> None:
        symbol = (data or {}).get("symbol")
        if not symbol:
            return
        payload = dict(data)
        payload.setdefault("timestamp", datetime.utcnow())
        self._storage[symbol.upper()].append(payload)

    def get_stock_history(self, symbol: str, lookback_minutes: int = 240) -> pd.DataFrame:
        sym = (symbol or "").upper().strip()
        if not sym:
            return pd.DataFrame()

        rows = list(self._storage.get(sym, []))
        if not rows:
            return pd.DataFrame()

        cutoff = datetime.utcnow() - timedelta(minutes=max(lookback_minutes, 1))
        recent = [row for row in rows if row.get("timestamp", cutoff) >= cutoff]
        if not recent:
            return pd.DataFrame()

        frame = pd.DataFrame(recent)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        return frame.sort_values("timestamp")


__all__ = ["DatabaseManager"]
