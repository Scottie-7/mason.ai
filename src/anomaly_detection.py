"""Simple statistical anomaly detection placeholder."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class AnomalyDetector:
    """Detect basic anomalies on a price series."""

    def detect(self, series: pd.Series, zscore_threshold: float = 3.0) -> Dict[str, float]:
        if series is None or series.empty:
            return {"has_anomaly": False}

        values = series.astype(float).to_numpy()
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std == 0:
            return {"has_anomaly": False}

        z_scores = np.abs((values - mean) / std)
        has_anomaly = bool(np.any(z_scores > zscore_threshold))
        return {"has_anomaly": has_anomaly, "max_zscore": float(np.max(z_scores))}


__all__ = ["AnomalyDetector"]
