"""Congestion prediction.

Pipeline: per-frame PCU samples -> fixed time buckets -> sliding-window features
-> GradientBoostingRegressor trained on historical + current data of the same road.
When there is not enough data the model falls back to a linear-trend extrapolation,
and with <3 points to persistence. The method used is always reported to the UI.
"""
import numpy as np
import config
from modules import density


def bucketize(samples, bucket=None):
    """samples: list of dicts with 't' (s) and 'pcu'. Returns evenly spaced list of mean PCU."""
    bucket = bucket or config.BUCKET_SECONDS
    if not samples:
        return []
    groups = {}
    for s in samples:
        groups.setdefault(int(s["t"] // bucket), []).append(s["pcu"])
    lo, hi = min(groups), max(groups)
    series, last = [], None
    for k in range(lo, hi + 1):
        last = float(np.mean(groups[k])) if k in groups else (last if last is not None else 0.0)
        series.append(last)
    return series


def _features(w):
    w = np.asarray(w, dtype=float)
    slope = np.polyfit(np.arange(len(w)), w, 1)[0] if len(w) > 1 else 0.0
    return list(w) + [w.mean(), w.std(), slope, w[-1] - w[0]]


def _windows(series, window, horizon):
    X, y = [], []
    for i in range(window, len(series) - horizon + 1):
        X.append(_features(series[i - window:i]))
        y.append(series[i - 1 + horizon])
    return X, y


class CongestionPredictor:
    def __init__(self, horizon=None, window=None):
        self.horizon = horizon or config.HORIZON_STEPS
        self.window = window or config.WINDOW_STEPS

    def predict(self, samples, history=None, use_ml=True):
        """samples: current road samples. history: list of earlier samples-lists (same road)."""
        series = bucketize(samples)
        out = {"horizon_seconds": round(self.horizon * config.BUCKET_SECONDS), "points": len(series)}
        if not series:
            return {**out, "current_pcu": 0, "predicted_pcu": 0, "level": "Low", "trend": "Stable",
                    "method": "no data", "confidence": "None"}
        current = float(np.mean(series[-2:]))
        method, confidence, pred = "persistence", "Low", series[-1]

        ml_done = False
        if use_ml:
            all_series = [bucketize(h) for h in (history or [])] + [series]
            X, y = [], []
            for s in all_series:
                a, b = _windows(s, self.window, self.horizon)
                X += a
                y += b
            if len(X) >= config.MIN_ML_WINDOWS and len(series) >= self.window:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.1, random_state=0)
                model.fit(np.array(X), np.array(y))
                pred = float(model.predict(np.array([_features(series[-self.window:])]))[0])
                method, confidence, ml_done = f"Gradient Boosting ML ({len(X)} training windows)", "High" if len(X) > 100 else "Medium", True
        if not ml_done and len(series) >= 3:
            tail = series[-8:]
            slope, intercept = np.polyfit(np.arange(len(tail)), tail, 1)
            pred = float(slope * (len(tail) - 1 + self.horizon) + intercept)
            method, confidence = "Linear trend extrapolation", "Medium" if len(series) >= 8 else "Low"

        pred = max(0.0, pred)
        delta = pred - current
        trend = "Rising" if delta > 1.5 else "Falling" if delta < -1.5 else "Stable"
        return {**out, "current_pcu": round(current, 2), "predicted_pcu": round(pred, 2),
                "level": density.classify(pred), "percent": density.percent(pred),
                "trend": trend, "method": method, "confidence": confidence}
