"""News aggregation helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


class NewsMonitor:
    """Return basic news items for a ticker."""

    def get_stock_news(self, symbol: str, sources: Iterable[str] | None = None) -> List[Dict]:
        sym = (symbol or "").upper().strip()
        if not sym:
            return []

        items: List[Dict] = []
        if yf is not None:
            try:
                ticker = yf.Ticker(sym)
                for story in ticker.news or []:
                    items.append(
                        {
                            "title": story.get("title"),
                            "summary": story.get("summary"),
                            "url": story.get("link"),
                            "source": story.get("publisher"),
                            "timestamp": datetime.fromtimestamp(
                                story.get("providerPublishTime", 0), tz=timezone.utc
                            ).astimezone(timezone.utc),
                        }
                    )
            except Exception:
                items = []

        if not items:
            # Offline fallback – fabricate a neutral update so the UI has content.
            items.append(
                {
                    "title": f"{sym} market update",
                    "summary": "Synthetic headline for demo purposes.",
                    "url": None,
                    "source": "Demo Feed",
                    "timestamp": datetime.now(timezone.utc),
                }
            )
        return items


__all__ = ["NewsMonitor"]
