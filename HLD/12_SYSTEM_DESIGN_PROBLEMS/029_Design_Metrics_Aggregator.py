"""
METRICS AGGREGATION SYSTEM
============================

FUNCTIONAL REQUIREMENTS:
- Ingest time-series metrics from thousands of services
- Aggregate: SUM, AVG, MIN, MAX, P50/P95/P99 percentiles, COUNT
- Retention: raw data 24h → 1-min rollups 30 days → 1-hour rollups 1 year
- Query: range queries, instant queries, label filtering
- Alerting: threshold-based on aggregated metrics

NON-FUNCTIONAL REQUIREMENTS:
- Ingest: 1 M data points/second (each point: metric_name + labels + value + ts)
- Storage: raw = 1M × 86400 = 86.4 B points/day (too large → rollups essential)
- Query latency: < 500 ms for last 24h of one metric (1-min rollup)
- Horizontal scalability: add ingestor/storage nodes independently

ARCHITECTURE:
  Service ──UDP/TCP──▶ Ingestor (StatsD/OTel Collector)
                              │
                     Kafka (partitioned by metric_name hash)
                              │
                     ┌────────┴──────────┐
                     ▼                   ▼
              Raw Storage          Rollup Worker
              (Cassandra/TS)   (time-window aggregation)
                     │                   │
              Query Engine ◀─── Rollup Store (Cassandra)
                     │
              Alert Evaluator

KEY DESIGN DECISIONS:
1. STORAGE ENGINE — columnar time-series:
   Facebook Gorilla (in-memory, delta-of-delta compression): 14 bytes → 1.37 bytes/point
   Prometheus: TSDB with chunks; 1-2 bytes/point with XOR float compression.
   Cassandra: partition=metric_name+label_hash, clustering=timestamp.

2. ROLLUP STRATEGY:
   Raw → 1-min (24h retention): aggregate every 60 points
   1-min → 5-min (7d retention): aggregate 5 one-minute points
   5-min → 1-hour (30d retention): aggregate 12 five-minute points
   1-hour → 1-day (1y retention): aggregate 24 one-hour points

3. AGGREGATION FUNCTIONS:
   Simple: SUM, MIN, MAX, COUNT → combine rollup windows exactly
   AVG: store (sum, count) → compute avg = sum/count
   Percentiles: can't combine without raw data → approximate with t-Digest or HDR Histogram
   P99 of P99 values ≠ true P99; must store t-Digest sketch per rollup window.

4. QUERY EXECUTION:
   Tier routing: for last-1h query → use 1-min rollup tier
   Multi-tier: stitch raw (last 5 min) + rollup for older windows
   PromQL-style: rate(), sum(), avg_over_time(), histogram_quantile()

5. DELTA-OF-DELTA TIMESTAMP COMPRESSION:
   Timestamps mostly at regular intervals → delta small → store delta-of-delta
   e.g. interval=10s: deltas = [10,10,10,...] → delta-of-delta = [0,0,0,...]
   Encode 0 as single bit; non-zero with variable-length encoding.

6. XOR FLOAT COMPRESSION (Gorilla):
   Consecutive float values often similar → XOR result mostly zeros
   If XOR = 0: store single 0 bit
   Else: store leading/trailing zero counts + meaningful bits

7. CARDINALITY PROBLEM:
   High-cardinality labels (user_id, IP) = billions of series → avoid
   Max series per metric: cap at 10K unique label combinations
"""

from __future__ import annotations
import time
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
from collections import defaultdict
import threading
import heapq


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelSet:
    """Immutable, hashable set of key-value labels for a metric series."""
    labels: Tuple[Tuple[str, str], ...]

    @staticmethod
    def from_dict(d: Dict[str, str]) -> "LabelSet":
        return LabelSet(tuple(sorted(d.items())))

    def to_dict(self) -> Dict[str, str]:
        return dict(self.labels)

    def matches(self, selector: Dict[str, str]) -> bool:
        d = self.to_dict()
        return all(d.get(k) == v for k, v in selector.items())


@dataclass
class DataPoint:
    metric_name: str
    labels: LabelSet
    value: float
    timestamp: float   # Unix epoch seconds


class AggType(Enum):
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"


# ---------------------------------------------------------------------------
# T-Digest for approximate percentiles
# ---------------------------------------------------------------------------

class TDigest:
    """
    Simplified t-Digest for computing approximate percentiles on streams.
    Real implementation: clusters of (mean, weight) pairs.
    """

    def __init__(self, compression: float = 100.0):
        self._compression = compression
        self._centroids: List[List[float]] = []  # [mean, weight]
        self._count = 0

    def add(self, value: float, weight: float = 1.0) -> None:
        self._centroids.append([value, weight])
        self._count += weight
        if len(self._centroids) > self._compression * 10:
            self._compress()

    def _compress(self) -> None:
        """Merge nearby centroids."""
        self._centroids.sort(key=lambda c: c[0])
        merged = []
        for mean, weight in self._centroids:
            if merged and abs(merged[-1][0] - mean) < 1.0:
                total_w = merged[-1][1] + weight
                merged[-1][0] = (merged[-1][0] * merged[-1][1] + mean * weight) / total_w
                merged[-1][1] = total_w
            else:
                merged.append([mean, weight])
        self._centroids = merged

    def percentile(self, q: float) -> Optional[float]:
        """Compute q-th percentile (q in [0, 1])."""
        if not self._centroids:
            return None
        self._compress()
        target = q * self._count
        cumulative = 0.0
        for mean, weight in sorted(self._centroids, key=lambda c: c[0]):
            cumulative += weight
            if cumulative >= target:
                return mean
        return self._centroids[-1][0]

    def merge(self, other: "TDigest") -> "TDigest":
        result = TDigest(self._compression)
        result._centroids = self._centroids + other._centroids
        result._count = self._count + other._count
        result._compress()
        return result

    def count(self) -> int:
        return int(self._count)


# ---------------------------------------------------------------------------
# Raw Time Series Store (in-memory, 24h window)
# ---------------------------------------------------------------------------

class RawStore:
    """
    Stores raw data points with automatic expiry.
    In production: Gorilla in-memory TSDB or InfluxDB.
    """

    def __init__(self, retention_seconds: float = 86400):
        self._data: Dict[str, Dict[LabelSet, List[DataPoint]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._retention = retention_seconds
        self._lock = threading.Lock()

    def write(self, point: DataPoint) -> None:
        with self._lock:
            self._data[point.metric_name][point.labels].append(point)

    def query(self, metric_name: str, label_selector: Dict[str, str],
               start: float, end: float) -> List[DataPoint]:
        with self._lock:
            series = self._data.get(metric_name, {})
            result = []
            for labels, points in series.items():
                if labels.matches(label_selector):
                    result.extend(p for p in points if start <= p.timestamp <= end)
            return sorted(result, key=lambda p: p.timestamp)

    def evict_old(self) -> int:
        """Remove data points older than retention window."""
        cutoff = time.time() - self._retention
        evicted = 0
        with self._lock:
            for metric_data in self._data.values():
                for labels in list(metric_data.keys()):
                    old_len = len(metric_data[labels])
                    metric_data[labels] = [p for p in metric_data[labels]
                                            if p.timestamp > cutoff]
                    evicted += old_len - len(metric_data[labels])
        return evicted


# ---------------------------------------------------------------------------
# Rollup Worker
# ---------------------------------------------------------------------------

@dataclass
class RollupPoint:
    metric_name: str
    labels: LabelSet
    bucket_start: float    # Start of the time bucket
    bucket_end: float
    agg_sum: float
    agg_count: int
    agg_min: float
    agg_max: float
    digest: Optional[TDigest] = None  # For percentile computation

    @property
    def agg_avg(self) -> float:
        return self.agg_sum / self.agg_count if self.agg_count else 0.0

    def aggregate(self, agg_type: AggType) -> float:
        if agg_type == AggType.SUM:
            return self.agg_sum
        elif agg_type == AggType.AVG:
            return self.agg_avg
        elif agg_type == AggType.MIN:
            return self.agg_min
        elif agg_type == AggType.MAX:
            return self.agg_max
        elif agg_type == AggType.COUNT:
            return float(self.agg_count)
        elif agg_type == AggType.P50 and self.digest:
            return self.digest.percentile(0.50) or 0
        elif agg_type == AggType.P95 and self.digest:
            return self.digest.percentile(0.95) or 0
        elif agg_type == AggType.P99 and self.digest:
            return self.digest.percentile(0.99) or 0
        return self.agg_avg


class RollupStore:
    """Stores pre-aggregated rollups at multiple resolutions."""

    def __init__(self):
        # (metric, labels, resolution) → list of RollupPoints
        self._rollups: Dict[Tuple[str, LabelSet, int], List[RollupPoint]] = defaultdict(list)

    def save(self, point: RollupPoint, resolution_seconds: int) -> None:
        key = (point.metric_name, point.labels, resolution_seconds)
        # Remove existing rollup for this bucket
        self._rollups[key] = [r for r in self._rollups[key]
                               if r.bucket_start != point.bucket_start]
        self._rollups[key].append(point)

    def query(self, metric_name: str, labels: LabelSet, resolution_seconds: int,
               start: float, end: float) -> List[RollupPoint]:
        key = (metric_name, labels, resolution_seconds)
        points = self._rollups.get(key, [])
        return sorted(
            [p for p in points if p.bucket_start >= start and p.bucket_end <= end],
            key=lambda p: p.bucket_start
        )

    def all_series(self, metric_name: str, resolution: int) -> List[LabelSet]:
        return [labels for m, labels, res in self._rollups.keys()
                if m == metric_name and res == resolution]


class RollupWorker:
    """Aggregates raw data into rollup buckets."""

    RESOLUTIONS = [60, 300, 3600, 86400]  # 1min, 5min, 1hour, 1day

    def __init__(self, raw_store: RawStore, rollup_store: RollupStore):
        self._raw = raw_store
        self._rollup = rollup_store

    def rollup_metric(self, metric_name: str, labels: LabelSet,
                       resolution: int, start: float, end: float) -> List[RollupPoint]:
        """Aggregate raw data for a time range into rollup buckets."""
        raw_points = self._raw.query(metric_name, labels.to_dict(), start, end)
        if not raw_points:
            return []

        # Group by time bucket
        buckets: Dict[float, List[float]] = defaultdict(list)
        for p in raw_points:
            bucket = math.floor(p.timestamp / resolution) * resolution
            buckets[bucket].append(p.value)

        rollup_points = []
        for bucket_start, values in sorted(buckets.items()):
            digest = TDigest()
            for v in values:
                digest.add(v)

            rp = RollupPoint(
                metric_name=metric_name,
                labels=labels,
                bucket_start=float(bucket_start),
                bucket_end=float(bucket_start + resolution),
                agg_sum=sum(values),
                agg_count=len(values),
                agg_min=min(values),
                agg_max=max(values),
                digest=digest,
            )
            rollup_points.append(rp)
            self._rollup.save(rp, resolution)

        return rollup_points

    def run_all(self, metric_name: str, lookback_seconds: float = 3600):
        """Roll up all series for a metric over the lookback window."""
        now = time.time()
        start = now - lookback_seconds
        # Get all label sets for this metric
        for metric_data in self._raw._data.get(metric_name, {}).items():
            labels, _ = metric_data
            for res in self.RESOLUTIONS:
                self.rollup_metric(metric_name, labels, res, start, now)


# ---------------------------------------------------------------------------
# Delta-of-Delta Timestamp Compression (Gorilla-style)
# ---------------------------------------------------------------------------

class TimestampCompressor:
    """
    Delta-of-delta encoding for regular time series.
    For timestamps at 10s intervals: [1000, 1010, 1020, 1030]
    Deltas: [10, 10, 10] → delta-of-delta: [0, 0]
    """

    @staticmethod
    def encode(timestamps: List[float]) -> List[int]:
        if len(timestamps) < 2:
            return []
        deltas = [int(timestamps[i] - timestamps[i-1]) for i in range(1, len(timestamps))]
        if len(deltas) < 2:
            return deltas
        dod = [deltas[0]] + [deltas[i] - deltas[i-1] for i in range(1, len(deltas))]
        return dod

    @staticmethod
    def decode(first_ts: float, dod: List[int]) -> List[float]:
        if not dod:
            return [first_ts]
        deltas = [dod[0]]
        for i in range(1, len(dod)):
            deltas.append(deltas[-1] + dod[i])
        timestamps = [first_ts]
        for d in deltas:
            timestamps.append(timestamps[-1] + d)
        return timestamps

    @staticmethod
    def compression_ratio(timestamps: List[float]) -> float:
        if len(timestamps) < 2:
            return 1.0
        dod = TimestampCompressor.encode(timestamps)
        # Original: 8 bytes per float timestamp
        original_bytes = len(timestamps) * 8
        # Compressed: assume 1 byte for zeros, 4 bytes for non-zeros
        compressed_bytes = sum(1 if d == 0 else 4 for d in dod) + 8  # +8 for first ts
        return original_bytes / max(compressed_bytes, 1)


# ---------------------------------------------------------------------------
# XOR Float Compression
# ---------------------------------------------------------------------------

class XORCompressor:
    """
    Gorilla XOR delta compression for float values.
    Consecutive similar values XOR to mostly-zero bit patterns.
    """

    @staticmethod
    def xor_encode(values: List[float]) -> Tuple[float, List[int]]:
        """Returns (first_value, list_of_xor_deltas_as_ints)."""
        if not values:
            return 0.0, []
        import struct
        first = values[0]
        prev_bits = struct.unpack("Q", struct.pack("d", first))[0]
        xor_list = [0]
        for v in values[1:]:
            curr_bits = struct.unpack("Q", struct.pack("d", v))[0]
            xor = prev_bits ^ curr_bits
            xor_list.append(xor)
            prev_bits = curr_bits
        return first, xor_list

    @staticmethod
    def compression_stats(values: List[float]) -> Dict[str, Any]:
        _, xors = XORCompressor.xor_encode(values)
        zero_count = sum(1 for x in xors if x == 0)
        original_bytes = len(values) * 8
        # Conservative: 1 bit for zero, 8 bytes for non-zero
        compressed_bits = sum(1 if x == 0 else 65 for x in xors)  # 65-bit for non-zero
        compressed_bytes = compressed_bits // 8 + 1
        return {
            "values": len(values),
            "zero_xors": zero_count,
            "zero_pct": zero_count / max(len(xors), 1),
            "original_bytes": original_bytes,
            "compressed_bytes_est": compressed_bytes,
            "ratio": original_bytes / max(compressed_bytes, 1),
        }


# ---------------------------------------------------------------------------
# Query Engine
# ---------------------------------------------------------------------------

class QueryEngine:
    def __init__(self, raw_store: RawStore, rollup_store: RollupStore):
        self._raw = raw_store
        self._rollup = rollup_store

    def query_range(self, metric_name: str, label_selector: Dict[str, str],
                     start: float, end: float,
                     agg_type: AggType = AggType.AVG,
                     step: int = 60) -> List[Tuple[float, float]]:
        """
        Return list of (timestamp, value) for a metric over time range.
        Automatically selects appropriate rollup resolution.
        """
        duration = end - start
        resolution = self._pick_resolution(duration)

        # Find matching label sets in rollup store
        all_series = self._rollup.all_series(metric_name, resolution)
        matching = [ls for ls in all_series if ls.matches(label_selector)]

        result = []
        for labels in matching:
            points = self._rollup.query(metric_name, labels, resolution, start, end)
            for rp in points:
                result.append((rp.bucket_start, rp.aggregate(agg_type)))

        # Sort by time
        result.sort(key=lambda x: x[0])
        return result

    def instant_query(self, metric_name: str, label_selector: Dict[str, str],
                       agg_type: AggType = AggType.AVG) -> Optional[float]:
        """Latest value for a metric."""
        end = time.time()
        start = end - 300  # last 5 min
        points = self._raw.query(metric_name, label_selector, start, end)
        if not points:
            return None
        values = [p.value for p in points]
        if agg_type == AggType.AVG:
            return sum(values) / len(values)
        elif agg_type == AggType.SUM:
            return sum(values)
        elif agg_type == AggType.MAX:
            return max(values)
        elif agg_type == AggType.MIN:
            return min(values)
        elif agg_type == AggType.COUNT:
            return float(len(values))
        return values[-1]

    @staticmethod
    def _pick_resolution(duration_seconds: float) -> int:
        """Pick coarsest resolution that gives at least 100 data points."""
        if duration_seconds <= 6000:     # ≤ 100 min → 1-min rollup
            return 60
        elif duration_seconds <= 25200:  # ≤ 7h → 5-min rollup
            return 300
        elif duration_seconds <= 360000: # ≤ 100h → 1-hour rollup
            return 3600
        else:
            return 86400


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demonstrate_1_ingest_and_query():
    print("\n=== 1. Ingest & Range Query ===")
    raw = RawStore(retention_seconds=3600)
    rollup_store = RollupStore()
    worker = RollupWorker(raw, rollup_store)
    engine = QueryEngine(raw, rollup_store)

    now = time.time()
    service_labels = LabelSet.from_dict({"service": "api", "env": "prod"})

    # Simulate 30 minutes of request rate metrics (1 point every 10 seconds)
    values = []
    for i in range(180):  # 180 points × 10s = 30 min
        ts = now - 1800 + i * 10
        rps = 100 + 50 * math.sin(i / 30 * math.pi) + random.gauss(0, 5)
        values.append(rps)
        raw.write(DataPoint("http_requests_total", service_labels, rps, ts))

    print(f"Ingested 180 data points over 30 minutes")

    # Rollup into 1-minute buckets
    rollups = worker.rollup_metric(
        "http_requests_total", service_labels, 60,
        now - 1800, now
    )
    print(f"Created {len(rollups)} 1-minute rollup buckets")

    # Query
    query_results = engine.query_range(
        "http_requests_total", {"service": "api"}, now - 1800, now,
        agg_type=AggType.AVG, step=60
    )
    print(f"Range query returned {len(query_results)} points")
    if query_results:
        vals = [v for _, v in query_results]
        print(f"  AVG: {sum(vals)/len(vals):.1f}, MIN: {min(vals):.1f}, MAX: {max(vals):.1f}")


def demonstrate_2_rollup_aggregations():
    print("\n=== 2. Multi-Level Rollup Aggregations ===")
    raw = RawStore(retention_seconds=86400)
    rollup_store = RollupStore()
    worker = RollupWorker(raw, rollup_store)

    now = time.time()
    labels = LabelSet.from_dict({"host": "server-01"})

    # Generate 6 hours of CPU data (every 10s)
    for i in range(2160):  # 2160 × 10s = 6h
        ts = now - 21600 + i * 10
        cpu = 40 + 20 * math.sin(i / 360 * 2 * math.pi) + random.uniform(0, 10)
        raw.write(DataPoint("cpu_usage_pct", labels, cpu, ts))

    # Create rollups at multiple resolutions
    for res in [60, 300, 3600]:
        rollups = worker.rollup_metric("cpu_usage_pct", labels, res, now - 21600, now)
        print(f"  Resolution {res}s: {len(rollups)} buckets")

    # Query at different resolutions for various time windows
    engine = QueryEngine(raw, rollup_store)
    for window_name, duration in [("Last 30min", 1800), ("Last 3h", 10800)]:
        results = engine.query_range("cpu_usage_pct", {"host": "server-01"},
                                      now - duration, now, AggType.AVG)
        vals = [v for _, v in results]
        if vals:
            print(f"\n{window_name} (avg): {sum(vals)/len(vals):.1f}% "
                  f"(max={max(vals):.1f}%, min={min(vals):.1f}%)")


def demonstrate_3_percentiles_tdigest():
    print("\n=== 3. Percentile Computation with T-Digest ===")
    # Simulate API latency distribution (bimodal: fast and slow paths)
    digest = TDigest(compression=50)
    latencies = (
        [random.gauss(20, 5) for _ in range(900)] +   # 90% fast: ~20ms
        [random.gauss(200, 30) for _ in range(100)]   # 10% slow: ~200ms
    )
    random.shuffle(latencies)

    for lat in latencies:
        digest.add(lat)

    print(f"Latency distribution ({len(latencies)} samples):")
    for q_pct, q in [(50, 0.50), (90, 0.90), (95, 0.95), (99, 0.99)]:
        est = digest.percentile(q)
        print(f"  P{q_pct}: {est:.1f}ms")

    # Merge two digests
    d1 = TDigest()
    d2 = TDigest()
    for v in latencies[:500]:
        d1.add(v)
    for v in latencies[500:]:
        d2.add(v)
    merged = d1.merge(d2)
    print(f"\nAfter merging two T-Digests (P99): {merged.percentile(0.99):.1f}ms")


def demonstrate_4_timestamp_compression():
    print("\n=== 4. Delta-of-Delta Timestamp Compression ===")
    # Regular 10-second intervals
    base = time.time()
    regular_ts = [base + i * 10 for i in range(1000)]
    dod_regular = TimestampCompressor.encode(regular_ts)
    zero_pct = sum(1 for d in dod_regular if d == 0) / max(len(dod_regular), 1)
    ratio_regular = TimestampCompressor.compression_ratio(regular_ts)

    # Irregular timestamps (network jitter ±1s)
    irregular_ts = [base + i * 10 + random.uniform(-1, 1) for i in range(1000)]
    ratio_irregular = TimestampCompressor.compression_ratio(irregular_ts)

    print(f"Regular 10s intervals (1000 points):")
    print(f"  Delta-of-delta zeros: {zero_pct:.1%}")
    print(f"  Compression ratio: {ratio_regular:.1f}x")
    print(f"\nIrregular intervals (±1s jitter):")
    print(f"  Compression ratio: {ratio_irregular:.1f}x")

    # Verify round-trip
    dod = TimestampCompressor.encode(regular_ts[:10])
    decoded = TimestampCompressor.decode(regular_ts[0], dod)
    print(f"\nRound-trip verification (first 5 timestamps):")
    for orig, dec in zip(regular_ts[:5], decoded[:5]):
        print(f"  {orig:.0f} → {dec:.0f} (diff={abs(orig-dec):.0f})")


def demonstrate_5_xor_compression():
    print("\n=== 5. XOR Float Compression (Gorilla-style) ===")
    test_cases = [
        ("Slowly changing (CPU %)", [50.0 + i * 0.01 + random.gauss(0, 0.1)
                                      for i in range(100)]),
        ("Random (noise)", [random.uniform(0, 100) for _ in range(100)]),
        ("Constant (idle)", [50.0] * 100),
        ("Step function", [10.0] * 50 + [90.0] * 50),
    ]

    for name, values in test_cases:
        stats = XORCompressor.compression_stats(values)
        print(f"\n{name}:")
        print(f"  Zero XOR deltas: {stats['zero_pct']:.1%}")
        print(f"  Original: {stats['original_bytes']} bytes → "
              f"Compressed: ~{stats['compressed_bytes_est']} bytes "
              f"({stats['ratio']:.1f}x)")


if __name__ == "__main__":
    demonstrate_1_ingest_and_query()
    demonstrate_2_rollup_aggregations()
    demonstrate_3_percentiles_tdigest()
    demonstrate_4_timestamp_compression()
    demonstrate_5_xor_compression()
