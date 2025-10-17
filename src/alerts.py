"""Alerting helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional


class AlertManager:
    """Very small rule-based alerting engine."""

    def check_alerts(
        self,
        snapshot: Dict[str, float],
        pct_threshold: float,
        volume_multiplier: float,
    ) -> Optional[Dict[str, float]]:
        if not snapshot:
            return None

        change_pct = float(snapshot.get("change_percent") or 0.0)
        volume = float(snapshot.get("volume") or 0.0)
        avg_volume = float(snapshot.get("avg_volume") or snapshot.get("average_volume") or 0.0)
        symbol = snapshot.get("symbol")

        triggered = False
        reasons = []
        if pct_threshold and abs(change_pct) >= float(pct_threshold):
            triggered = True
            reasons.append(f"Price moved {change_pct:+.2f}%")

        if volume_multiplier and avg_volume > 0:
            ratio = volume / avg_volume
            if ratio >= volume_multiplier:
                triggered = True
                reasons.append(f"Volume spike {ratio:.1f}× avg")

        if not triggered:
            return None

        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow(),
            "change_percent": change_pct,
            "volume": volume,
            "reasons": reasons,
        }


__all__ = ["AlertManager"]
