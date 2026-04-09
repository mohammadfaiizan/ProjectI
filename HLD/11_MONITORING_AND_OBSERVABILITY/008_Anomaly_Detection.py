"""
ANOMALY DETECTION IN OBSERVABILITY
=====================================

Problem Statement:
Static thresholds for alerts (e.g., "error rate > 5%") miss anomalies
that are context-dependent (night vs day, Monday vs Saturday) and fail
for metrics with natural seasonality or growth trends.

Anomaly Detection Approaches:

  1. Static Threshold:
     Simple, predictable. Fails for seasonal metrics.

  2. Moving Average / Rolling Z-Score:
     z = (x - mean) / stddev over rolling window.
     Detects sudden spikes relative to recent history.

  3. Seasonal Decomposition (STL):
     Separates Trend + Seasonality + Residual.
     Alert on large residuals. Good for daily/weekly patterns.

  4. Exponentially Weighted Moving Average (EWMA):
     More weight to recent samples. Tracks slow-moving baselines.
     α controls reactivity: high α = fast adaptation, low α = stable.

  5. DBSCAN / Isolation Forest:
     Unsupervised ML. Good for multivariate anomalies.

  6. Facebook Prophet / SARIMA:
     Time series forecasting models. Alert on large forecast error.

  7. Percentile Bands:
     For each hour-of-week, compute p1/p99 from historical data.
     Alert when current value falls outside the band.

Practical Stack:
  Prometheus → recording rules (p99 rate) → Grafana ML plugin or
  Victoria Metrics anomaly detection → Alertmanager.
  Alternatively: metrics → Kafka → Python anomaly worker → DB → Grafana.

Key Metrics for Anomaly Detection:
  - Request rate (traffic): daily/weekly seasonality.
  - Error rate: should stay near-zero; spikes = incident.
  - Latency p99: baseline varies by endpoint.
  - Queue depth: grows slowly → alert early.
  - Memory/CPU: trending up = memory leak.
"""

from __future__ import annotations

import math
import time
import random
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# TIME SERIES POINT
# ─────────────────────────────────────────────

@dataclass
class Point:
    timestamp: float
    value:     float


# ─────────────────────────────────────────────
# ROLLING Z-SCORE DETECTOR
# ─────────────────────────────────────────────

class RollingZScore:
    """
    Computes z-score using a sliding window.
    z = (x - mean(window)) / std(window)
    Alert when |z| > threshold.
    """

    def __init__(self, window: int = 60, threshold: float = 3.0):
        self._window    = window
        self._threshold = threshold
        self._buf:      Deque[float] = deque(maxlen=window)

    def update(self, value: float) -> Tuple[float, bool]:
        """Returns (z_score, is_anomaly)."""
        self._buf.append(value)
        if len(self._buf) < 10:       # need enough data
            return 0.0, False

        mu    = statistics.mean(self._buf)
        sigma = statistics.stdev(self._buf) if len(self._buf) > 1 else 0.0
        if sigma < 1e-9:
            return 0.0, False

        z         = (value - mu) / sigma
        is_anomaly = abs(z) > self._threshold
        return z, is_anomaly


# ─────────────────────────────────────────────
# EWMA (Exponentially Weighted Moving Average)
# ─────────────────────────────────────────────

class EWMADetector:
    """
    Tracks baseline using EWMA; alerts when deviation > k * ewm_std.
    α = smoothing factor (0 < α < 1).
    High α: reacts fast (forgets history quickly).
    Low α:  stable baseline (slow to adapt).
    """

    def __init__(self, alpha: float = 0.1, k: float = 3.0):
        self._alpha  = alpha
        self._k      = k
        self._ewma:  Optional[float] = None
        self._ewm_var: float = 0.0

    def update(self, value: float) -> Tuple[float, float, bool]:
        """Returns (ewma, deviation, is_anomaly)."""
        if self._ewma is None:
            self._ewma    = value
            self._ewm_var = 0.0
            return value, 0.0, False

        prev_ewma      = self._ewma
        self._ewma     = self._alpha * value + (1 - self._alpha) * self._ewma
        diff           = value - prev_ewma
        self._ewm_var  = (self._alpha * diff**2 +
                          (1 - self._alpha) * self._ewm_var)

        ewm_std    = math.sqrt(self._ewm_var) if self._ewm_var > 0 else 0.0
        deviation  = abs(value - self._ewma)
        is_anomaly = ewm_std > 0 and deviation > self._k * ewm_std
        return self._ewma, deviation, is_anomaly


# ─────────────────────────────────────────────
# SEASONAL PERCENTILE BANDS
# ─────────────────────────────────────────────

class SeasonalBands:
    """
    Maintains per-bucket (e.g., hour-of-week) percentile bands
    from historical data. Alerts when value falls outside [p_low, p_high].

    Practical use: for each of 168 hours in a week, compute
    p1 and p99 from the last N weeks.
    """

    def __init__(self, n_buckets: int = 24,   # 24 hours-of-day
                 p_low: float = 5.0,
                 p_high: float = 95.0,
                 min_samples: int = 10):
        self._n_buckets    = n_buckets
        self._p_low        = p_low
        self._p_high       = p_high
        self._min_samples  = min_samples
        self._history: Dict[int, List[float]] = {i: [] for i in range(n_buckets)}

    def _bucket(self, ts: float) -> int:
        """Map timestamp to bucket index (hour of day)."""
        hour = int((ts % 86400) / 3600)
        return hour % self._n_buckets

    def _percentile(self, data: List[float], p: float) -> float:
        n    = len(data)
        if n == 0: return 0.0
        idx  = (p / 100) * (n - 1)
        lo   = int(idx)
        hi   = min(lo + 1, n - 1)
        frac = idx - lo
        s    = sorted(data)
        return s[lo] + frac * (s[hi] - s[lo])

    def record(self, ts: float, value: float):
        self._history[self._bucket(ts)].append(value)

    def check(self, ts: float, value: float
              ) -> Tuple[Optional[float], Optional[float], bool]:
        """Returns (lower_band, upper_band, is_anomaly)."""
        bucket = self._bucket(ts)
        hist   = self._history[bucket]
        if len(hist) < self._min_samples:
            return None, None, False
        lo = self._percentile(hist, self._p_low)
        hi = self._percentile(hist, self._p_high)
        return lo, hi, (value < lo or value > hi)


# ─────────────────────────────────────────────
# STL DECOMPOSITION (simplified)
# ─────────────────────────────────────────────

class SimplifiedSTL:
    """
    Simplified Seasonal-Trend decomposition.
    Trend:      centred moving average over full period.
    Seasonal:   average residual per sub-period bucket.
    Remainder:  original - trend - seasonal.
    Alert when |remainder| > k * std(remainder).
    """

    def __init__(self, period: int = 24, k: float = 3.0):
        self._period  = period
        self._k       = k

    def decompose(self, series: List[float]
                  ) -> Tuple[List[float], List[float], List[float]]:
        n    = len(series)
        half = self._period // 2

        # Trend: centred moving average
        trend = [0.0] * n
        for i in range(n):
            lo_  = max(0, i - half)
            hi_  = min(n, i + half + 1)
            trend[i] = sum(series[lo_:hi_]) / (hi_ - lo_)

        # Detrend
        detrended = [series[i] - trend[i] for i in range(n)]

        # Seasonal: average by bucket
        buckets = [[] for _ in range(self._period)]
        for i, v in enumerate(detrended):
            buckets[i % self._period].append(v)
        seasonal_avg = [sum(b)/len(b) if b else 0.0 for b in buckets]

        seasonal  = [seasonal_avg[i % self._period] for i in range(n)]
        remainder = [series[i] - trend[i] - seasonal[i] for i in range(n)]

        return trend, seasonal, remainder

    def anomaly_indices(self, series: List[float]) -> List[int]:
        if len(series) < self._period * 2:
            return []
        _, _, remainder = self.decompose(series)
        if not remainder:
            return []
        mu    = sum(remainder) / len(remainder)
        sigma = statistics.stdev(remainder) if len(remainder) > 1 else 0.0
        if sigma < 1e-9:
            return []
        return [i for i, r in enumerate(remainder) if abs(r - mu) > self._k * sigma]


# ─────────────────────────────────────────────
# ISOLATION FOREST (simplified 1D)
# ─────────────────────────────────────────────

class SimpleIsolationForest:
    """
    1D Isolation Forest approximation.
    Anomaly score based on average path length to isolate a point.
    Short path → easy to isolate → anomaly.
    """

    def __init__(self, n_trees: int = 100, subsample: int = 256,
                 contamination: float = 0.05):
        self._n_trees      = n_trees
        self._subsample    = subsample
        self._contamination = contamination
        self._threshold:   Optional[float] = None
        self._scores:      List[float] = []

    def _c(self, n: int) -> float:
        """Average path length for BST of size n."""
        if n <= 1:
            return 0.0
        return 2 * (math.log(n - 1) + 0.5772) - 2 * (n - 1) / n

    def _isolation_depth(self, value: float, sample: List[float],
                         depth: int = 0, limit: int = 8) -> int:
        """Recursively split; return depth when isolated."""
        if len(sample) <= 1 or depth >= limit:
            return depth + self._c(len(sample))

        lo, hi = min(sample), max(sample)
        if lo == hi:
            return depth + self._c(len(sample))

        split  = lo + random.random() * (hi - lo)
        left   = [v for v in sample if v < split]
        right  = [v for v in sample if v >= split]
        if value < split:
            return self._isolation_depth(value, left, depth + 1, limit)
        else:
            return self._isolation_depth(value, right, depth + 1, limit)

    def fit(self, data: List[float]):
        import random as _r
        _r.seed(42)
        self._scores = []
        for x in data:
            depths = []
            for _ in range(self._n_trees):
                sample = _r.sample(data, min(self._subsample, len(data)))
                depths.append(self._isolation_depth(x, sample))
            avg_depth     = sum(depths) / len(depths)
            score         = 2 ** (-avg_depth / self._c(self._subsample))
            self._scores.append(score)

        sorted_scores    = sorted(self._scores)
        idx              = int((1 - self._contamination) * len(sorted_scores))
        self._threshold  = sorted_scores[min(idx, len(sorted_scores) - 1)]

    def predict(self, data: List[float]) -> List[bool]:
        """Returns True for each anomaly."""
        results = []
        for x in data:
            depths = [self._isolation_depth(x, data[:self._subsample])
                      for _ in range(20)]
            avg   = sum(depths) / len(depths)
            score = 2 ** (-avg / max(self._c(min(self._subsample, len(data))), 1))
            results.append(score > (self._threshold or 0.6))
        return results


# ─────────────────────────────────────────────
# METRIC ANOMALY REPORT
# ─────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    detector:  str
    timestamp: float
    value:     float
    score:     float       # anomaly score / z-score
    message:   str


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_anomaly_detection():
    print("=" * 65)
    print("ANOMALY DETECTION IN OBSERVABILITY")
    print("=" * 65)

    random.seed(99)

    # ── Generate synthetic time series ────────
    print("\n[1] GENERATING SYNTHETIC TIME SERIES")
    print("─" * 55)

    # 48 hours of hourly data with daily seasonality + 2 injected anomalies
    n = 48
    ts_base = time.time() - n * 3600

    series: List[Point] = []
    for i in range(n):
        hour = i % 24
        # Daily pattern: low at night, peak at business hours
        seasonal = 100 + 50 * math.sin(2 * math.pi * (hour - 6) / 24)
        noise    = random.gauss(0, 5)
        value    = seasonal + noise

        # Inject anomalies at hour 14 and hour 36
        if i == 14: value = 280.0   # spike
        if i == 36: value = 10.0    # drop

        series.append(Point(ts_base + i * 3600, value))

    print(f"  Generated {n} hourly data points")
    print(f"  Normal range: ~50 to ~150")
    print(f"  Injected anomaly at h=14 (spike=280) and h=36 (drop=10)")

    # ── Rolling Z-Score ───────────────────────
    print("\n[2] ROLLING Z-SCORE DETECTOR")
    print("─" * 55)

    detector_z  = RollingZScore(window=24, threshold=2.5)
    anomalies_z = []
    for p in series:
        z, is_anom = detector_z.update(p.value)
        if is_anom:
            anomalies_z.append(AnomalyEvent(
                "rolling_z", p.timestamp, p.value, z,
                f"z={z:.2f} (|z|>{detector_z._threshold})"))

    print(f"  Anomalies detected: {len(anomalies_z)}")
    for ev in anomalies_z:
        h = int((ev.timestamp - ts_base) / 3600)
        print(f"    h={h:2d}  value={ev.value:.1f}  {ev.message}")

    # ── EWMA ──────────────────────────────────
    print("\n[3] EWMA DETECTOR (α=0.2, k=3)")
    print("─" * 55)

    detector_e  = EWMADetector(alpha=0.2, k=3.0)
    anomalies_e = []
    for p in series:
        ewma, dev, is_anom = detector_e.update(p.value)
        if is_anom:
            anomalies_e.append(AnomalyEvent(
                "ewma", p.timestamp, p.value, dev,
                f"ewma={ewma:.1f}  deviation={dev:.1f}"))

    print(f"  Anomalies detected: {len(anomalies_e)}")
    for ev in anomalies_e:
        h = int((ev.timestamp - ts_base) / 3600)
        print(f"    h={h:2d}  value={ev.value:.1f}  {ev.message}")

    # ── Seasonal Percentile Bands ─────────────
    print("\n[4] SEASONAL PERCENTILE BANDS")
    print("─" * 55)

    bands = SeasonalBands(n_buckets=24, p_low=5, p_high=95, min_samples=5)

    # Train on first 24 hours
    for p in series[:24]:
        bands.record(p.timestamp, p.value)

    # Check second 24 hours
    anomalies_b = []
    for p in series[24:]:
        lo, hi, is_anom = bands.check(p.timestamp, p.value)
        if is_anom and lo is not None:
            h = int((p.timestamp - ts_base) / 3600)
            anomalies_b.append((h, p.value, lo, hi))

    print(f"  Anomalies detected: {len(anomalies_b)}")
    for h, v, lo, hi in anomalies_b:
        print(f"    h={h:2d}  value={v:.1f}  band=[{lo:.1f}, {hi:.1f}]")

    # ── STL Decomposition ─────────────────────
    print("\n[5] STL DECOMPOSITION (period=24)")
    print("─" * 55)

    stl    = SimplifiedSTL(period=24, k=2.5)
    values = [p.value for p in series]
    anom_indices = stl.anomaly_indices(values)

    print(f"  Anomaly indices: {anom_indices}")
    for i in anom_indices:
        print(f"    h={i:2d}  value={values[i]:.1f}")

    # Also show decomposition for a few hours
    trend, seasonal, remainder = stl.decompose(values)
    print("\n  Sample decomposition (h=14 spike):")
    for h in [12, 13, 14, 15, 16]:
        print(f"    h={h:2d}  raw={values[h]:.1f}  "
              f"trend={trend[h]:.1f}  seasonal={seasonal[h]:.1f}  "
              f"residual={remainder[h]:.1f}")

    # ── Isolation Forest ──────────────────────
    print("\n[6] ISOLATION FOREST (1D)")
    print("─" * 55)

    iforest  = SimpleIsolationForest(n_trees=50, subsample=48, contamination=0.05)
    iforest.fit(values)
    predictions = iforest.predict(values)
    anom_if = [(i, values[i]) for i, flag in enumerate(predictions) if flag]

    print(f"  Threshold score: {iforest._threshold:.3f}")
    print(f"  Anomalies detected: {len(anom_if)}")
    for h, v in anom_if[:5]:
        print(f"    h={h:2d}  value={v:.1f}")

    # ── Detector Comparison ───────────────────
    print("\n[7] DETECTOR COMPARISON")
    print("─" * 55)

    print(f"  {'Detector':<25} {'Anomalies Found':<18} {'Notes'}")
    print("  " + "─" * 70)
    for name, count, note in [
        ("Rolling Z-Score",        len(anomalies_z), "Fast; misses gradual drift"),
        ("EWMA",                   len(anomalies_e), "Adapts slowly; less flapping"),
        ("Seasonal Bands",         len(anomalies_b), "Best for seasonal metrics"),
        ("STL Decomposition",      len(anom_indices),"Handles trend + seasonality"),
        ("Isolation Forest (1D)",  len(anom_if),     "Good for multivariate in prod"),
    ]:
        print(f"  {name:<25} {count:<18} {note}")

    # ── Alert Routing ─────────────────────────
    print("\n[8] ANOMALY → ALERT ROUTING")
    print("─" * 55)

    guidance = [
        ("Low  z-score (1-2σ)", "DEBUG log; monitor dashboard"),
        ("Med  z-score (2-3σ)", "INFO alert to Slack; no page"),
        ("High z-score (>3σ)",  "WARNING alert; may page if sustained"),
        ("Burn rate > 14×",     "CRITICAL page; likely incident"),
        ("Repeated anomalies",  "Open SLO review; adjust threshold"),
        ("False positives",     "Widen band; raise threshold; add seasonality"),
    ]
    for score_desc, action in guidance:
        print(f"  {score_desc:<25} → {action}")


if __name__ == "__main__":
    demonstrate_anomaly_detection()
