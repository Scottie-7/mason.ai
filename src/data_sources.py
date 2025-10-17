"""Data loading helpers for the Streamlit dashboard.

The original prototype referenced an internal service layer.  This module provides a
self-contained implementation that fetches data from ``yfinance`` when available and
falls back to lightweight synthetic data so the UI remains interactive without
external credentials.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except Exception:  # pragma: no cover - keep the app running without yfinance
    yf = None


class StockDataManager:
    """Provide snapshot and historical market data.

    The real application likely wires into a proprietary market data feed.  For the
    open-source demo we try to use :mod:`yfinance` when it is installed.  When that
    fails (e.g. during offline development) we synthesise plausible looking data so
    the Streamlit widgets keep functioning.
    """

    def __init__(self) -> None:
        self._snapshot_cache: Dict[str, Dict[str, float]] = {}

    # ---------------------------------------------------------------------
    def get_real_time_data(self, symbol: str) -> Dict[str, float]:
        """Return the latest snapshot for ``symbol``.

        Parameters
        ----------
        symbol:
            Ticker symbol (case-insensitive).
        """

        sym = (symbol or "").upper().strip()
        if not sym:
            return {}

        data: Dict[str, float] = {}

        if yf is not None:
            try:
                ticker = yf.Ticker(sym)
                fast = getattr(ticker, "fast_info", None) or {}
                info = getattr(ticker, "info", {}) or {}

                price = float(
                    fast.get("last_price")
                    or fast.get("lastPrice")
                    or info.get("regularMarketPrice")
                    or info.get("currentPrice")
                    or 0.0
                )
                previous_close = float(
                    info.get("previousClose")
                    or fast.get("previous_close")
                    or price
                    or 0.0
                )
                high = float(
                    info.get("dayHigh")
                    or fast.get("day_high")
                    or price
                    or 0.0
                )
                low = float(
                    info.get("dayLow")
                    or fast.get("day_low")
                    or price
                    or 0.0
                )
                volume = float(
                    fast.get("last_volume")
                    or info.get("volume")
                    or 0.0
                )

                if price:
                    change = price - previous_close
                    change_pct = (change / previous_close * 100) if previous_close else 0.0
                    data = {
                        "symbol": sym,
                        "price": price,
                        "previous_close": previous_close,
                        "price_change": change,
                        "change_percent": change_pct,
                        "volume": volume,
                        "high": high,
                        "low": low,
                        "timestamp": datetime.now(timezone.utc).replace(tzinfo=None),
                    }
            except Exception:
                # Ignore network or parsing issues and fall back to synthetic data
                data = {}

        if not data:
            # Synthetic fallback – keep values consistent between refreshes so that
            # alerts and charts behave in a predictable way.
            cached = self._snapshot_cache.get(sym)
            base_price = float(cached.get("price", 100.0)) if cached else 100.0
            price = max(0.5, base_price * float(np.random.normal(1.0, 0.002)))
            previous_close = cached.get("previous_close", price)
            change = price - previous_close
            change_pct = (change / previous_close * 100) if previous_close else 0.0
            volume = max(1_000.0, (cached or {}).get("volume", 1_000_000.0) * float(np.random.normal(1.0, 0.05)))

            data = {
                "symbol": sym,
                "price": price,
                "previous_close": previous_close,
                "price_change": change,
                "change_percent": change_pct,
                "volume": volume,
                "high": max(price, previous_close),
                "low": min(price, previous_close),
                "timestamp": datetime.now(),
            }

        self._snapshot_cache[sym] = data
        return data

    # ------------------------------------------------------------------
    def get_historical_data(
        self, symbol: str, period: str = "1mo", interval: Optional[str] = None
    ) -> pd.DataFrame:
        """Return a historical OHLCV dataframe for ``symbol``.

        ``interval`` is optional; when omitted we let :mod:`yfinance` choose a
        sensible default.  The method always returns a dataframe with the canonical
        columns ``[Open, High, Low, Close, Volume]`` when data is available.
        """

        sym = (symbol or "").upper().strip()
        if not sym:
            return pd.DataFrame()

        if yf is not None:
            try:
                kwargs = {"progress": False, "auto_adjust": False}
                if interval:
                    kwargs["interval"] = interval
                data = yf.download(sym, period=period, **kwargs)
                if isinstance(data, pd.DataFrame) and not data.empty:
                    data = data.reset_index()
                    # Normalise column names and ensure datetime index
                    rename_map = {
                        "Date": "Date",
                        "Datetime": "Date",
                        "Open": "Open",
                        "High": "High",
                        "Low": "Low",
                        "Close": "Close",
                        "Adj Close": "Adj Close",
                        "Volume": "Volume",
                    }
                    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})
                    data = data.set_index("Date").sort_index()
                    return data[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]]
            except Exception:
                pass

        # Synthetic fallback – generate a simple geometric random walk.
        horizon = self._period_to_days(period)
        if horizon <= 0:
            horizon = 30
        freq = "1H" if horizon <= 7 else "1D"
        idx = pd.date_range(end=datetime.now(), periods=horizon * (24 if freq == "1H" else 1), freq=freq)
        if idx.empty:
            return pd.DataFrame()

        base = self._snapshot_cache.get(sym, {"price": 100.0})
        price = float(base.get("price", 100.0))
        returns = np.random.normal(loc=0.0005, scale=0.02, size=len(idx))
        prices = price * np.exp(np.cumsum(returns))
        high = prices * (1 + np.random.uniform(0, 0.01, size=len(idx)))
        low = prices * (1 - np.random.uniform(0, 0.01, size=len(idx)))
        open_price = np.concatenate(([price], prices[:-1]))
        volume = np.random.uniform(1e5, 5e5, size=len(idx))

        data = pd.DataFrame(
            {
                "Open": open_price,
                "High": np.maximum.reduce([open_price, high, prices]),
                "Low": np.minimum.reduce([open_price, low, prices]),
                "Close": prices,
                "Volume": volume,
            },
            index=idx,
        )
        return data

    # ------------------------------------------------------------------
    @staticmethod
    def _period_to_days(period: str) -> int:
        mapping = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 182,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
        }
        if period.isdigit():
            return int(period)
        return mapping.get(period.lower(), 30)


__all__ = ["StockDataManager"]
