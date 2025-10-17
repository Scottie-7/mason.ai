"""Plotting helpers used throughout the Streamlit dashboard."""

from __future__ import annotations

from typing import Dict

import pandas as pd

try:  # pragma: no cover - optional dependency
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    go = None
    make_subplots = None


class ChartGenerator:
    """Wrap Plotly figure creation with safe fallbacks."""

    def _require_plotly(self) -> None:
        if go is None or make_subplots is None:
            raise RuntimeError("Plotly is required for chart rendering.")

    def create_order_book_chart(self, order_book: Dict) -> go.Figure:
        self._require_plotly()
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        fig = make_subplots(rows=1, cols=1)
        if bids:
            fig.add_trace(
                go.Bar(
                    x=[level["size"] for level in bids],
                    y=[level["price"] for level in bids],
                    name="Bids",
                    orientation="h",
                    marker_color="#2ca02c",
                )
            )
        if asks:
            fig.add_trace(
                go.Bar(
                    x=[level["size"] for level in asks],
                    y=[level["price"] for level in asks],
                    name="Asks",
                    orientation="h",
                    marker_color="#d62728",
                )
            )
        fig.update_layout(
            title="Depth of Book",
            barmode="overlay",
            yaxis_title="Price",
            xaxis_title="Size",
            legend=dict(orientation="h"),
        )
        return fig

    def create_candlestick_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        self._require_plotly()
        frame = self._prepare_ohlc(df)
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=frame.index,
                    open=frame["Open"],
                    high=frame["High"],
                    low=frame["Low"],
                    close=frame["Close"],
                    name=symbol,
                )
            ]
        )
        fig.update_layout(title=f"{symbol} Price", xaxis_title="Date", yaxis_title="Price")
        return fig

    def create_line_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        self._require_plotly()
        frame = self._prepare_ohlc(df)
        fig = go.Figure(
            data=[go.Scatter(x=frame.index, y=frame["Close"], mode="lines", name="Close")]
        )
        fig.update_layout(title=f"{symbol} Close Price", xaxis_title="Date", yaxis_title="Price")
        return fig

    def create_volume_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        self._require_plotly()
        frame = self._prepare_ohlc(df)
        fig = go.Figure(
            data=[go.Bar(x=frame.index, y=frame["Volume"], name="Volume", marker_color="#1f77b4")]
        )
        fig.update_layout(title=f"{symbol} Volume", xaxis_title="Date", yaxis_title="Shares")
        return fig

    def create_technical_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        self._require_plotly()
        frame = self._prepare_ohlc(df)
        close = frame["Close"].astype(float)
        sma = close.rolling(window=20, min_periods=1).mean()
        ema = close.ewm(span=20, adjust=False).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frame.index, y=close, name="Close", mode="lines"))
        fig.add_trace(go.Scatter(x=frame.index, y=sma, name="SMA20", mode="lines"))
        fig.add_trace(go.Scatter(x=frame.index, y=ema, name="EMA20", mode="lines"))
        fig.update_layout(title=f"{symbol} Technicals", xaxis_title="Date", yaxis_title="Price")
        return fig

    @staticmethod
    def _prepare_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError("No data provided for plotting")
        frame = df.copy()
        ren = {c: c.title() for c in frame.columns}
        frame = frame.rename(columns=ren)
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing OHLC columns: {missing}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index)
        return frame


__all__ = ["ChartGenerator"]
