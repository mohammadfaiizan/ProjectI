"""
Problem 29: Design a Metrics Aggregation System
=================================================
Working simulation of a time-series metrics pipeline with:
- TimeSeriesStore: in-memory per-metric+label time-ordered store
- AggregationWindow: tumbling windows (sum/avg/min/max/count)
- RollupEngine: downsample 1m → 5m → 1h
- AlertEngine: threshold-based alert rule evaluation
- MetricsScraper: pull-model scrape simulation
- PromQLParser: simple subset (rate, sum, avg, gauge queries)
- GorillaDeltaEncoder: delta-of-delta + XOR compression simulation
"""

import re
import time
import math
import struct
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum


# ─── Metric Types ─────────────────────────────────────────────────────────────

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Sample:
    timestamp: float   # Unix epoch seconds
    value: float

    def __lt__(self, other: 'Sample') -> bool:
        return self.timestamp < other.timestamp


@dataclass
class MetricSeries:
    """A single time-series: metric_name + label_set → ordered list of samples."""
    name: str
    labels: dict[str, str]
    metric_type: MetricType
    samples: list[Sample] = field(default_factory=list)

    @property
    def series_key(self) -> str:
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))
        return f"{self.name}{{{label_str}}}"

    def add_sample(self, timestamp: float, value: float) -> None:
        self.samples.append(Sample(timestamp, value))

    def samples_in_range(self, start: float, end: float) -> list[Sample]:
        return [s for s in self.samples if start <= s.timestamp <= end]

    def latest_value(self) -> Optional[float]:
        return self.samples[-1].value if self.samples else None


@dataclass
class AlertRule:
    name: str
    expr: str            # Simplified: "metric_name > threshold" or "rate(m) > threshold"
    threshold: float
    comparison: str      # ">", "<", ">=", "<="
    for_duration: float  # seconds — must be firing for this long before alerting
    severity: str = "warning"
    pending_since: Optional[float] = None
    is_firing: bool = False


@dataclass
class AlertEvent:
    rule_name: str
    state: str           # FIRING, RESOLVED, PENDING
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    labels: dict = field(default_factory=dict)


# ─── Gorilla Delta Encoder ────────────────────────────────────────────────────

class GorillaDeltaEncoder:
    """
    Simplified simulation of Facebook's Gorilla time-series compression.
    Delta-of-delta for timestamps; XOR encoding for float values.
    Reports compression ratio vs raw storage (8B timestamp + 8B value = 16B/sample).
    """

    def __init__(self):
        self._raw_bits = 0
        self._compressed_bits = 0

    def encode_timestamps(self, timestamps: list[float]) -> tuple[int, float]:
        """
        Encode timestamps using delta-of-delta.
        Returns (compressed_bits, compression_ratio).
        """
        if len(timestamps) < 2:
            return len(timestamps) * 64, 1.0

        raw_bits = len(timestamps) * 64  # 8 bytes per timestamp
        compressed = 64  # First timestamp stored raw

        deltas = [int(timestamps[i+1] - timestamps[i]) for i in range(len(timestamps)-1)]
        dods = [deltas[0]] + [deltas[i] - deltas[i-1] for i in range(1, len(deltas))]

        for dod in dods:
            if dod == 0:
                compressed += 1            # '0' bit
            elif -63 <= dod <= 64:
                compressed += 2 + 7       # '10' + 7-bit value
            elif -255 <= dod <= 256:
                compressed += 2 + 9       # '110' + 9-bit value
            elif -2047 <= dod <= 2048:
                compressed += 2 + 12      # '1110' + 12-bit value
            else:
                compressed += 4 + 32      # '1111' + 32-bit value

        return compressed, raw_bits / max(1, compressed)

    def encode_values(self, values: list[float]) -> tuple[int, float]:
        """
        Encode float values using XOR encoding.
        Returns (compressed_bits, compression_ratio).
        """
        if not values:
            return 0, 1.0

        raw_bits = len(values) * 64  # 8 bytes per float64
        compressed = 64  # First value stored raw

        prev_bits = struct.unpack('Q', struct.pack('d', values[0]))[0]
        prev_lz = 0
        prev_meaningful_len = 64

        for v in values[1:]:
            curr_bits = struct.unpack('Q', struct.pack('d', v))[0]
            xor = prev_bits ^ curr_bits

            if xor == 0:
                compressed += 1  # '0' bit — same value
            else:
                leading_zeros = (64 - xor.bit_length()) if xor else 64
                trailing_zeros = (xor & -xor).bit_length() - 1 if xor else 0
                meaningful_len = 64 - leading_zeros - trailing_zeros

                if leading_zeros >= prev_lz and meaningful_len <= prev_meaningful_len:
                    # Reuse previous block
                    compressed += 2 + meaningful_len  # '10' + meaningful bits
                else:
                    # New block
                    compressed += 1 + 5 + 6 + meaningful_len  # '11' + lz + len + bits
                    prev_lz = leading_zeros
                    prev_meaningful_len = meaningful_len

            prev_bits = curr_bits

        return compressed, raw_bits / max(1, compressed)

    def analyze(self, samples: list[Sample]) -> dict:
        """Analyze compression for a list of samples."""
        if not samples:
            return {}
        timestamps = [s.timestamp for s in samples]
        values = [s.value for s in samples]
        ts_bits, ts_ratio = self.encode_timestamps(timestamps)
        val_bits, val_ratio = self.encode_values(values)
        raw_bytes = len(samples) * 16  # 8B ts + 8B value
        compressed_bytes = (ts_bits + val_bits) // 8
        return {
            "sample_count": len(samples),
            "raw_bytes": raw_bytes,
            "compressed_bytes": compressed_bytes,
            "overall_ratio": round(raw_bytes / max(1, compressed_bytes), 2),
            "bytes_per_sample": round(compressed_bytes / len(samples), 2),
            "timestamp_ratio": round(ts_ratio, 2),
            "value_ratio": round(val_ratio, 2)
        }


# ─── Time-Series Store ────────────────────────────────────────────────────────

class TimeSeriesStore:
    """
    In-memory time-ordered store for metric samples.
    Keyed by series_key (metric_name + labels).
    """

    def __init__(self, max_samples_per_series: int = 10_000):
        self._series: dict[str, MetricSeries] = {}
        self.max_samples = max_samples_per_series

    def write(self, name: str, labels: dict[str, str], metric_type: MetricType,
              timestamp: float, value: float) -> str:
        """Write a sample. Returns the series key."""
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        key = f"{name}{{{label_str}}}"

        if key not in self._series:
            self._series[key] = MetricSeries(name, labels, metric_type)

        series = self._series[key]
        series.add_sample(timestamp, value)

        # Evict oldest samples to stay within limit
        if len(series.samples) > self.max_samples:
            series.samples = series.samples[-self.max_samples:]

        return key

    def query(self, name: str, label_filters: dict[str, str],
              start: float, end: float) -> list[MetricSeries]:
        """Return all series matching name and label filters within time range."""
        results = []
        for key, series in self._series.items():
            if series.name != name:
                continue
            if not all(series.labels.get(k) == v for k, v in label_filters.items()):
                continue
            filtered = MetricSeries(series.name, series.labels, series.metric_type)
            filtered.samples = series.samples_in_range(start, end)
            if filtered.samples:
                results.append(filtered)
        return results

    def get_latest(self, name: str, label_filters: dict[str, str] = None) -> list[tuple[str, float]]:
        """Return latest value for each matching series."""
        results = []
        for key, series in self._series.items():
            if series.name != name:
                continue
            if label_filters and not all(series.labels.get(k) == v for k, v in label_filters.items()):
                continue
            if series.samples:
                results.append((key, series.samples[-1].value))
        return results

    def series_count(self) -> int:
        return len(self._series)


# ─── Aggregation Window ───────────────────────────────────────────────────────

class AggregationWindow:
    """
    Tumbling window aggregation: sum/avg/min/max/count over time buckets.
    """

    def __init__(self, window_seconds: float):
        self.window_seconds = window_seconds

    def aggregate(self, samples: list[Sample], agg_func: str = "avg") -> list[Sample]:
        """Aggregate samples into tumbling windows."""
        if not samples:
            return []

        start_time = samples[0].timestamp
        end_time = samples[-1].timestamp
        buckets: dict[int, list[float]] = defaultdict(list)

        for sample in samples:
            bucket_idx = int((sample.timestamp - start_time) / self.window_seconds)
            buckets[bucket_idx].append(sample.value)

        result = []
        for idx in sorted(buckets.keys()):
            values = buckets[idx]
            bucket_ts = start_time + idx * self.window_seconds

            if agg_func == "sum":
                agg_val = sum(values)
            elif agg_func == "avg":
                agg_val = sum(values) / len(values)
            elif agg_func == "min":
                agg_val = min(values)
            elif agg_func == "max":
                agg_val = max(values)
            elif agg_func == "count":
                agg_val = float(len(values))
            elif agg_func == "rate":
                # Rate = (last - first) / window_seconds
                agg_val = (values[-1] - values[0]) / self.window_seconds if len(values) > 1 else 0.0
            else:
                agg_val = sum(values) / len(values)

            result.append(Sample(bucket_ts, agg_val))
        return result


# ─── Rollup Engine ────────────────────────────────────────────────────────────

class RollupEngine:
    """
    Downsample time-series data:
    Raw (15s) → 1m → 5m → 1h rollups.
    """

    RESOLUTIONS = [
        ("1m",  60.0,    "15s"),
        ("5m",  300.0,   "1m"),
        ("1h",  3600.0,  "5m"),
        ("1d",  86400.0, "1h"),
    ]

    def __init__(self, store: TimeSeriesStore):
        self.store = store
        self._rollup_store: dict[str, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
        self._agg = AggregationWindow(60.0)

    def rollup(self, series_key: str, series: MetricSeries, target_resolution: str) -> list[Sample]:
        """Downsample a series to the target resolution."""
        for res_name, window_sec, _ in self.RESOLUTIONS:
            if res_name == target_resolution:
                window = AggregationWindow(window_sec)
                agg_func = "rate" if series.metric_type == MetricType.COUNTER else "avg"
                rollups = window.aggregate(series.samples, agg_func)
                self._rollup_store[target_resolution][series_key] = rollups
                return rollups
        return []

    def get_rollups(self, series_key: str, resolution: str) -> list[Sample]:
        return self._rollup_store[resolution].get(series_key, [])

    def run_all_rollups(self, series: dict[str, MetricSeries]) -> dict:
        """Run all rollup levels for all series."""
        stats = defaultdict(int)
        for key, s in series.items():
            if not s.samples:
                continue
            for res_name, _, _ in self.RESOLUTIONS[:3]:
                rollups = self.rollup(key, s, res_name)
                stats[res_name] += len(rollups)
        return dict(stats)


# ─── Alert Engine ─────────────────────────────────────────────────────────────

class AlertEngine:
    """Evaluate alert rules against current metrics. Supports threshold and rate alerts."""

    def __init__(self, store: TimeSeriesStore):
        self.store = store
        self._rules: list[AlertRule] = []
        self._events: list[AlertEvent] = []
        self._notifications: list[str] = []

    def add_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)

    def evaluate_all(self, current_time: Optional[float] = None) -> list[AlertEvent]:
        """Evaluate all alert rules. Returns list of new AlertEvents."""
        now = current_time or time.time()
        new_events = []

        for rule in self._rules:
            value = self._evaluate_expr(rule, now)
            if value is None:
                continue

            fires = self._compare(value, rule.threshold, rule.comparison)

            if fires:
                if rule.pending_since is None:
                    rule.pending_since = now
                    new_events.append(AlertEvent(rule.name, "PENDING", value, rule.threshold, now))

                elif now - rule.pending_since >= rule.for_duration:
                    if not rule.is_firing:
                        rule.is_firing = True
                        event = AlertEvent(rule.name, "FIRING", value, rule.threshold, now)
                        new_events.append(event)
                        self._events.append(event)
                        msg = (f"ALERT FIRING [{rule.severity.upper()}]: {rule.name} "
                               f"value={value:.4f} {rule.comparison} {rule.threshold}")
                        self._notifications.append(msg)
                        print(f"  [ALERT] {msg}")
            else:
                if rule.is_firing:
                    rule.is_firing = False
                    event = AlertEvent(rule.name, "RESOLVED", value, rule.threshold, now)
                    new_events.append(event)
                    self._events.append(event)
                    msg = f"ALERT RESOLVED: {rule.name}"
                    self._notifications.append(msg)
                    print(f"  [RESOLVED] {msg}")
                rule.pending_since = None

        return new_events

    def _evaluate_expr(self, rule: AlertRule, now: float) -> Optional[float]:
        """Simple expression evaluator: supports plain metric lookup."""
        metric_name = rule.expr.strip()
        latest = self.store.get_latest(metric_name)
        if not latest:
            return None
        values = [v for _, v in latest]
        return sum(values) / len(values)  # avg across all matching series

    def _compare(self, value: float, threshold: float, op: str) -> bool:
        ops = {">": value > threshold, "<": value < threshold,
               ">=": value >= threshold, "<=": value <= threshold, "==": value == threshold}
        return ops.get(op, False)

    def get_firing(self) -> list[AlertRule]:
        return [r for r in self._rules if r.is_firing]

    def get_event_history(self) -> list[AlertEvent]:
        return list(self._events)


# ─── PromQL Parser (Simple Subset) ───────────────────────────────────────────

class PromQLParser:
    """
    Parse and evaluate a simple subset of PromQL:
    - Instant vector: metric_name{label="value"}
    - Range vector: metric_name[5m]
    - Functions: rate(), sum(), avg(), max(), min()
    """

    def __init__(self, store: TimeSeriesStore):
        self.store = store

    def query_instant(self, expr: str, timestamp: Optional[float] = None) -> list[dict]:
        """Evaluate an instant vector query at a given timestamp."""
        now = timestamp or time.time()
        results = []

        # Parse: optional_func(metric_name{filters}[range])
        # Simple metric name
        if re.match(r'^[a-z_][a-z0-9_]*$', expr):
            for key, series in self.store._series.items():
                if series.name == expr and series.samples:
                    results.append({
                        "metric": {series.name: series.labels},
                        "value": [now, series.samples[-1].value],
                        "series_key": key
                    })
            return results

        # rate(metric[duration])
        rate_match = re.match(r'rate\(([a-z_][a-z0-9_]*)\[(\d+)(m|s|h)\]\)', expr)
        if rate_match:
            metric = rate_match.group(1)
            duration_num = int(rate_match.group(2))
            unit = rate_match.group(3)
            duration_sec = duration_num * {"s": 1, "m": 60, "h": 3600}[unit]
            start = now - duration_sec

            for key, series in self.store._series.items():
                if series.name != metric:
                    continue
                if series.metric_type != MetricType.COUNTER:
                    continue
                window_samples = series.samples_in_range(start, now)
                if len(window_samples) >= 2:
                    rate = (window_samples[-1].value - window_samples[0].value) / duration_sec
                    results.append({
                        "metric": {series.name: series.labels},
                        "value": [now, max(0.0, rate)],
                        "series_key": key
                    })
            return results

        # sum(metric)
        sum_match = re.match(r'sum\(([a-z_][a-z0-9_]*)\)', expr)
        if sum_match:
            metric = sum_match.group(1)
            total = sum(v for k, v in self.store.get_latest(metric))
            return [{"metric": {"__name__": f"sum({metric})"},
                     "value": [now, total], "series_key": f"sum:{metric}"}]

        return []

    def query_range(self, expr: str, start: float, end: float, step: float) -> list[dict]:
        """Evaluate a range query. Returns matrix results."""
        results_by_key: dict[str, list] = defaultdict(list)
        t = start
        while t <= end:
            instant = self.query_instant(expr, timestamp=t)
            for r in instant:
                results_by_key[r["series_key"]].append(r["value"])
            t += step
        return [{"series_key": k, "values": v} for k, v in results_by_key.items()]


# ─── Metrics Scraper ─────────────────────────────────────────────────────────

class MetricsScraper:
    """
    Pull-model metrics collection.
    Registers targets with scrape functions; runs scrapes on interval.
    """

    def __init__(self, store: TimeSeriesStore, scrape_interval: float = 15.0):
        self.store = store
        self.scrape_interval = scrape_interval
        self._targets: list[tuple[str, Callable]] = []  # (target_name, scrape_fn)
        self._scrape_count = 0
        self._last_scrape: dict[str, float] = {}

    def register_target(self, target_name: str, scrape_fn: Callable) -> None:
        """Register a target with a scrape function that returns list of (name, labels, type, value)."""
        self._targets.append((target_name, scrape_fn))
        print(f"  Registered scrape target: {target_name}")

    def scrape_all(self, timestamp: Optional[float] = None) -> int:
        """Scrape all registered targets. Returns total samples written."""
        now = timestamp or time.time()
        total = 0
        for target_name, scrape_fn in self._targets:
            try:
                metrics = scrape_fn(now)
                for name, labels, m_type, value in metrics:
                    self.store.write(name, labels, m_type, now, value)
                    total += 1
                self._last_scrape[target_name] = now
            except Exception as e:
                print(f"  [SCRAPE FAIL] {target_name}: {e}")
        self._scrape_count += 1
        return total

    def get_stats(self) -> dict:
        return {"registered_targets": len(self._targets),
                "total_scrapes": self._scrape_count,
                "last_scrape_times": dict(self._last_scrape)}


# ─── Metrics Aggregator (Top-Level) ──────────────────────────────────────────

class MetricsAggregator:
    """
    Top-level metrics aggregation system.
    Combines scraper, store, rollup, alerting, and query into one interface.
    """

    def __init__(self):
        self.store = TimeSeriesStore()
        self.scraper = MetricsScraper(self.store)
        self.rollup_engine = RollupEngine(self.store)
        self.alert_engine = AlertEngine(self.store)
        self.encoder = GorillaDeltaEncoder()
        self._promql = PromQLParser(self.store)

    def record_metric(self, name: str, labels: dict, metric_type: MetricType,
                      value: float, timestamp: Optional[float] = None) -> None:
        self.store.write(name, labels, metric_type, timestamp or time.time(), value)

    def query_range(self, expr: str, start: float, end: float, step: float = 60.0) -> list[dict]:
        return self._promql.query_range(expr, start, end, step)

    def query_instant(self, expr: str) -> list[dict]:
        return self._promql.query_instant(expr)

    def aggregate_window(self, name: str, labels: dict, window: float,
                         func: str = "avg", lookback: float = 3600.0) -> Optional[float]:
        now = time.time()
        series_list = self.store.query(name, labels, now - lookback, now)
        if not series_list:
            return None
        agg = AggregationWindow(window)
        all_samples = []
        for s in series_list:
            all_samples.extend(s.samples)
        all_samples.sort()
        results = agg.aggregate(all_samples, func)
        return results[-1].value if results else None


# ─── Demo / Simulation ────────────────────────────────────────────────────────

def simulate_api_server(base_rps: float = 100.0):
    """Generate a scrape function for a simulated API server."""
    _request_counter = [0.0]
    _error_counter = [0.0]

    def scrape(now: float) -> list:
        # Simulate traffic with random variation
        import random
        rps = base_rps + random.gauss(0, base_rps * 0.1)
        err_rate = 0.02 + random.gauss(0, 0.005)
        _request_counter[0] += rps * 15  # 15s scrape interval
        _error_counter[0] += rps * 15 * err_rate

        return [
            ("http_requests_total", {"method": "GET", "status": "200"}, MetricType.COUNTER, _request_counter[0]),
            ("http_requests_total", {"method": "GET", "status": "500"}, MetricType.COUNTER, _error_counter[0]),
            ("http_request_duration_seconds", {"quantile": "0.5"}, MetricType.GAUGE, 0.05 + random.gauss(0, 0.01)),
            ("http_request_duration_seconds", {"quantile": "0.99"}, MetricType.GAUGE, 0.2 + random.gauss(0, 0.05)),
            ("memory_usage_bytes", {}, MetricType.GAUGE, 512_000_000 + random.gauss(0, 50_000_000)),
            ("active_connections", {}, MetricType.GAUGE, max(0, int(rps * 0.1))),
        ]
    return scrape


def run_simulation():
    print("=" * 65)
    print("METRICS AGGREGATION SYSTEM SIMULATION")
    print("=" * 65)

    aggregator = MetricsAggregator()

    # ── Register scrape targets ────────────────────────────────
    print("\n--- Registering Scrape Targets ---")
    aggregator.scraper.register_target("api-server-1", simulate_api_server(100.0))
    aggregator.scraper.register_target("api-server-2", simulate_api_server(80.0))
    aggregator.scraper.register_target("api-server-3", simulate_api_server(120.0))

    # ── Configure alert rules ──────────────────────────────────
    print("\n--- Configuring Alert Rules ---")
    rules = [
        AlertRule("HighMemoryUsage",  "memory_usage_bytes", 600_000_000, ">",  60.0, "warning"),
        AlertRule("LowConnections",   "active_connections",  2.0,         "<",  30.0, "info"),
        AlertRule("SlowP99Response",  "http_request_duration_seconds", 0.3, ">", 120.0, "critical"),
    ]
    for rule in rules:
        aggregator.alert_engine.add_rule(rule)
        print(f"  Added rule: {rule.name} ({rule.expr} {rule.comparison} {rule.threshold})")

    # ── Simulate scraping over time ───────────────────────────
    print("\n--- Simulating 10 Scrape Cycles (15s intervals) ---")
    base_time = time.time() - 150.0
    for cycle in range(10):
        ts = base_time + cycle * 15.0
        count = aggregator.scraper.scrape_all(timestamp=ts)
        print(f"  Cycle {cycle+1:02d} @ T+{cycle*15:3d}s: {count} samples written "
              f"| Total series: {aggregator.store.series_count()}")

    # ── Rollup engine ──────────────────────────────────────────
    print("\n--- Running Rollup Engine ---")
    rollup_stats = aggregator.rollup_engine.run_all_rollups(aggregator.store._series)
    for resolution, count in rollup_stats.items():
        print(f"  {resolution} rollups: {count} aggregate points generated")

    # ── PromQL instant queries ────────────────────────────────
    print("\n--- PromQL Instant Queries ---")
    queries = [
        "memory_usage_bytes",
        "active_connections",
        f"rate(http_requests_total[5m])",
        "sum(active_connections)",
    ]
    for q in queries:
        results = aggregator.query_instant(q)
        print(f"\n  Query: {q}")
        for r in results[:3]:
            val = r['value'][1]
            print(f"    {r['series_key'][:70]} = {val:.2f}")
        if len(results) > 3:
            print(f"    ... and {len(results)-3} more series")

    # ── Range query ───────────────────────────────────────────
    print("\n--- PromQL Range Query: memory_usage_bytes ---")
    end_t = time.time()
    start_t = end_t - 150.0
    range_results = aggregator.query_range("memory_usage_bytes", start_t, end_t, step=30.0)
    for r in range_results[:2]:
        vals = r['values']
        print(f"  {r['series_key'][:60]}")
        print(f"    {len(vals)} data points | latest: {vals[-1][1]:.0f} bytes")

    # ── Alert evaluation ──────────────────────────────────────
    print("\n--- Alert Rule Evaluation ---")
    # Inject high memory reading to trigger alert
    aggregator.record_metric("memory_usage_bytes", {}, MetricType.GAUGE, 700_000_000)
    # First evaluation → PENDING
    events = aggregator.alert_engine.evaluate_all(current_time=base_time + 150.0)
    print(f"  After 1st eval: {len(events)} events")
    for e in events:
        print(f"    {e.rule_name}: {e.state} (value={e.value:.0f})")

    # Simulate passing `for` duration → FIRING
    aggregator.alert_engine._rules[0].pending_since = base_time  # Force past for_duration
    events = aggregator.alert_engine.evaluate_all(current_time=base_time + 300.0)
    print(f"\n  After for_duration: {len(events)} events")

    # Resolve by recording normal value
    aggregator.record_metric("memory_usage_bytes", {}, MetricType.GAUGE, 400_000_000)
    events = aggregator.alert_engine.evaluate_all(current_time=base_time + 310.0)
    for e in events:
        print(f"    {e.rule_name}: {e.state} (value={e.value:.0f})")

    # ── Gorilla Compression Analysis ──────────────────────────
    print("\n--- Gorilla Encoding Analysis ---")
    encoder = GorillaDeltaEncoder()

    # Regular scrape timestamps (15s apart — best case for delta-of-delta)
    regular_ts = [base_time + i * 15.0 for i in range(100)]
    regular_vals = [100.0 + (i % 10) * 0.1 for i in range(100)]  # Slowly changing gauge
    counter_vals = [float(i * 1500) for i in range(100)]  # Monotonically increasing counter

    regular_samples = [Sample(t, v) for t, v in zip(regular_ts, regular_vals)]
    counter_samples = [Sample(t, v) for t, v in zip(regular_ts, counter_vals)]

    gauge_stats = encoder.analyze(regular_samples)
    counter_stats = encoder.analyze(counter_samples)

    print(f"\n  Gauge metric (slowly changing):")
    print(f"    Raw bytes        : {gauge_stats['raw_bytes']}")
    print(f"    Compressed bytes : {gauge_stats['compressed_bytes']}")
    print(f"    Overall ratio    : {gauge_stats['overall_ratio']}x compression")
    print(f"    Bytes/sample     : {gauge_stats['bytes_per_sample']}")

    print(f"\n  Counter metric (monotonically increasing):")
    print(f"    Raw bytes        : {counter_stats['raw_bytes']}")
    print(f"    Compressed bytes : {counter_stats['compressed_bytes']}")
    print(f"    Overall ratio    : {counter_stats['overall_ratio']}x compression")
    print(f"    Bytes/sample     : {counter_stats['bytes_per_sample']}")

    # ── Store stats ───────────────────────────────────────────
    print(f"\n--- Store Statistics ---")
    print(f"  Total unique series  : {aggregator.store.series_count()}")
    print(f"  Scraper stats        : {aggregator.scraper.get_stats()}")
    print(f"  Firing alerts        : {[r.name for r in aggregator.alert_engine.get_firing()]}")
    print(f"  Total alert events   : {len(aggregator.alert_engine.get_event_history())}")

    print("\n" + "=" * 65)
    print("Simulation complete.")


if __name__ == "__main__":
    run_simulation()
