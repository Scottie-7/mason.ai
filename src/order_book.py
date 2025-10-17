"""Order book analytics utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class OrderLevel:
    price: float
    size: float


class OrderBookAnalyzer:
    """Create synthetic level-2 order book snapshots and metrics."""

    def get_order_book_data(self, symbol: str) -> Dict[str, List[Dict[str, float]]]:
        """Return a lightweight level-2 order book structure.

        The production system probably pulls live depth-of-book data.  Here we simply
        generate a symmetrical book around the last synthetic mid-price so the charts
        keep functioning.
        """

        base_price = 100.0 + hash(symbol.upper()) % 50
        ticks = np.linspace(-0.5, 0.5, 10)
        bids = [
            {"price": round(base_price + t, 2), "size": float(np.random.randint(50, 500))}
            for t in ticks[ticks < 0]
        ][::-1]
        asks = [
            {"price": round(base_price + t, 2), "size": float(np.random.randint(50, 500))}
            for t in ticks[ticks > 0]
        ]
        return {"bids": bids, "asks": asks}

    # ------------------------------------------------------------------
    @staticmethod
    def calculate_bid_pressure(order_book: Dict[str, Sequence[Dict[str, float]]]) -> float:
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        bid_volume = sum(level.get("size", 0.0) for level in bids)
        ask_volume = sum(level.get("size", 0.0) for level in asks)
        total = bid_volume + ask_volume
        return (bid_volume / total * 100.0) if total else 0.0

    @staticmethod
    def calculate_ask_pressure(order_book: Dict[str, Sequence[Dict[str, float]]]) -> float:
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        bid_volume = sum(level.get("size", 0.0) for level in bids)
        ask_volume = sum(level.get("size", 0.0) for level in asks)
        total = bid_volume + ask_volume
        return (ask_volume / total * 100.0) if total else 0.0

    @staticmethod
    def calculate_spread(order_book: Dict[str, Sequence[Dict[str, float]]]) -> float:
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        if not bids or not asks:
            return 0.0
        best_bid = max(level.get("price", 0.0) for level in bids)
        best_ask = min(level.get("price", 0.0) for level in asks)
        return max(0.0, best_ask - best_bid)

    @staticmethod
    def calculate_order_imbalance(order_book: Dict[str, Sequence[Dict[str, float]]]) -> float:
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        bid_volume = sum(level.get("size", 0.0) for level in bids)
        ask_volume = sum(level.get("size", 0.0) for level in asks)
        total = bid_volume + ask_volume
        if not total:
            return 0.0
        return (bid_volume - ask_volume) / total


__all__ = ["OrderBookAnalyzer", "OrderLevel"]
