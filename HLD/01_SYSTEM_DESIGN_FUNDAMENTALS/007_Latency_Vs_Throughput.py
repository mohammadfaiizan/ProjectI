"""
LATENCY VS THROUGHPUT
======================

Problem Statement:
Two of the most important performance dimensions in system design are
latency (how fast a single request completes) and throughput (how many
requests the system can handle per second). They often trade off, and
understanding their relationship is essential for designing performant systems.

Key Concepts:
- Latency    : Time to complete one request (ms). Measured at percentiles: p50, p95, p99, p999
- Throughput : Requests processed per unit time (RPS / QPS)
- Little's Law: L = λ × W  (avg concurrency = arrival rate × avg response time)
- Amdahl's Law: Max speedup bounded by serial fraction of the workload
- Tail Latency : p99/p999 latency — affects user experience at scale

Little's Law Example:
  - System handles 1000 RPS (λ = 1000/s)
  - Average response time W = 100ms = 0.1s
  - Average in-flight requests L = 1000 × 0.1 = 100 concurrent requests
"""

from dataclasses import dataclass
from typing import List, Dict
import random
import math


@dataclass
class LatencyPercentile:
    """Latency distribution at different percentiles (milliseconds)."""
    p50 : float
    p90 : float
    p95 : float
    p99 : float
    p999: float

    def report(self, label: str = ""):
        tag = f" [{label}]" if label else ""
        print(f"  Latency{tag}:")
        print(f"    p50  (median)   : {self.p50:.1f} ms")
        print(f"    p90             : {self.p90:.1f} ms")
        print(f"    p95             : {self.p95:.1f} ms")
        print(f"    p99             : {self.p99:.1f} ms")
        print(f"    p999 (tail)     : {self.p999:.1f} ms")


@dataclass
class ThroughputMetric:
    label           : str
    requests_per_sec: float
    bytes_per_sec   : float = 0.0

    def report(self):
        bps = f"  ({self.bytes_per_sec/1e6:.1f} MB/s)" if self.bytes_per_sec else ""
        print(f"  Throughput [{self.label}]: {self.requests_per_sec:,.0f} RPS{bps}")


class LittlesLaw:
    """
    L = λ × W
    L = average number of requests in the system (concurrency)
    λ = arrival rate (requests per second)
    W = average time a request spends in the system (seconds)
    """

    @staticmethod
    def concurrency(arrival_rate_rps: float, avg_latency_ms: float) -> float:
        return arrival_rate_rps * (avg_latency_ms / 1000.0)

    @staticmethod
    def max_throughput(concurrency: float, avg_latency_ms: float) -> float:
        return concurrency / (avg_latency_ms / 1000.0)

    @staticmethod
    def required_latency_ms(target_rps: float, max_concurrency: float) -> float:
        """What avg latency must be to hit target RPS given concurrency limit."""
        return (max_concurrency / target_rps) * 1000.0

    @classmethod
    def analyse(cls, label: str, rps: float, latency_ms: float, concurrency_limit: int):
        L = cls.concurrency(rps, latency_ms)
        print(f"\n  Little's Law [{label}]:")
        print(f"    Arrival rate (λ)   : {rps:,.0f} RPS")
        print(f"    Avg latency (W)    : {latency_ms:.0f} ms")
        print(f"    Avg concurrency (L): {L:.1f} in-flight requests")
        print(f"    Concurrency limit  : {concurrency_limit}")
        if L > concurrency_limit:
            print(f"    ⚠  Queue builds up! L ({L:.0f}) > limit ({concurrency_limit})")
        else:
            print(f"    ✅ System stable. {concurrency_limit - L:.0f} concurrency headroom.")


class PerformanceAnalyzer:
    """Analyses a system's performance profile from a list of simulated response times."""

    def __init__(self, label: str):
        self.label    = label
        self.samples  : List[float] = []

    def add_sample(self, latency_ms: float):
        self.samples.append(latency_ms)

    def simulate(self, count: int, base_ms: float, noise_factor: float = 0.5,
                 outlier_pct: float = 0.01, outlier_ms: float = 2000):
        """Generate synthetic latency samples with tail behaviour."""
        for _ in range(count):
            if random.random() < outlier_pct:
                self.samples.append(outlier_ms + random.uniform(0, 1000))
            else:
                val = base_ms * (1 + random.uniform(-noise_factor, noise_factor))
                self.samples.append(max(1, val))

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        sorted_s = sorted(self.samples)
        idx = int(math.ceil((p / 100.0) * len(sorted_s))) - 1
        return sorted_s[max(0, idx)]

    def throughput(self, window_sec: float = 1.0) -> float:
        return len(self.samples) / window_sec

    def latency_profile(self) -> LatencyPercentile:
        return LatencyPercentile(
            p50 =self.percentile(50),
            p90 =self.percentile(90),
            p95 =self.percentile(95),
            p99 =self.percentile(99),
            p999=self.percentile(99.9),
        )

    def report(self):
        profile = self.latency_profile()
        profile.report(self.label)
        avg = sum(self.samples) / len(self.samples)
        print(f"    avg             : {avg:.1f} ms")
        print(f"    samples         : {len(self.samples):,}")


class BottleneckDetector:
    """Identifies whether a system is CPU, memory, I/O or network bound."""

    @staticmethod
    def detect(cpu_pct: float, mem_pct: float, io_wait_pct: float,
               network_util_pct: float) -> str:
        if cpu_pct > 80:
            return f"CPU-bound ({cpu_pct:.0f}% CPU) — optimise algorithms, add CPUs, scale out"
        if mem_pct > 85:
            return f"Memory-bound ({mem_pct:.0f}% RAM) — add caching tier, reduce object size, scale up RAM"
        if io_wait_pct > 30:
            return f"I/O-bound ({io_wait_pct:.0f}% I/O wait) — use async I/O, add SSD, read replicas, caching"
        if network_util_pct > 80:
            return f"Network-bound ({network_util_pct:.0f}% NIC) — add CDN, compression, reduce payload sizes"
        return "✅ System not obviously bottlenecked"


def demonstrate_latency_vs_throughput():
    print("=" * 65)
    print("LATENCY VS THROUGHPUT")
    print("=" * 65)

    # ── Percentile profiles ───────────────────
    print("\n[1] LATENCY PERCENTILE DISTRIBUTIONS")
    print("─" * 50)

    # Fast cached API
    cached = PerformanceAnalyzer("Cached API (Redis)")
    cached.simulate(10_000, base_ms=5, noise_factor=0.4, outlier_pct=0.005, outlier_ms=200)
    cached.report()

    print()

    # DB-backed API
    db_api = PerformanceAnalyzer("DB-backed API")
    db_api.simulate(10_000, base_ms=50, noise_factor=0.6, outlier_pct=0.02, outlier_ms=3000)
    db_api.report()

    print()

    # Video transcoding
    trans = PerformanceAnalyzer("Transcoding (async job)")
    trans.simulate(1_000, base_ms=2000, noise_factor=0.3, outlier_pct=0.05, outlier_ms=30_000)
    trans.report()

    # ── Little's Law ──────────────────────────
    print("\n\n[2] LITTLE'S LAW ANALYSIS")
    print("─" * 50)
    LittlesLaw.analyse("Twitter timeline read",   rps=12_000, latency_ms=50,  concurrency_limit=1_000)
    LittlesLaw.analyse("Payment API",             rps=500,    latency_ms=200, concurrency_limit=200)
    LittlesLaw.analyse("DB query under high load", rps=5_000, latency_ms=80,  concurrency_limit=200)

    # ── Throughput vs Latency trade-off ───────
    print("\n\n[3] THROUGHPUT vs LATENCY TRADE-OFF (batching)")
    print("─" * 50)
    print("  Scenario: Write events to DB individually vs batched")
    print()
    configs = [
        ("No batching (1 event)",    1,    2.0),
        ("Small batch (10 events)",  10,   4.0),
        ("Medium batch (100 events)",100,  12.0),
        ("Large batch (1000 events)",1000, 80.0),
    ]
    for label, batch_size, latency_ms in configs:
        throughput = (batch_size / latency_ms) * 1000
        print(f"  {label:<30} latency={latency_ms:5.0f}ms  throughput={throughput:8,.0f} events/sec")

    # ── Bottleneck Detection ──────────────────
    print("\n\n[4] BOTTLENECK DETECTION")
    print("─" * 50)
    scenarios = [
        ("High-traffic web server", 92.0, 40.0, 10.0, 30.0),
        ("Analytics DB node",       25.0, 90.0, 55.0, 20.0),
        ("File upload service",     35.0, 45.0, 72.0, 35.0),
        ("CDN edge node",           20.0, 30.0, 15.0, 88.0),
        ("Healthy API server",      45.0, 50.0, 15.0, 40.0),
    ]
    for name, cpu, mem, io, net in scenarios:
        diagnosis = BottleneckDetector.detect(cpu, mem, io, net)
        print(f"\n  System: {name}")
        print(f"    CPU={cpu:.0f}%  MEM={mem:.0f}%  IO={io:.0f}%  NET={net:.0f}%")
        print(f"    → {diagnosis}")

    # ── Key Rules of Thumb ────────────────────
    print("\n\n[5] RULES OF THUMB")
    print("─" * 50)
    rules = [
        "Halving avg latency DOUBLES throughput (Little's Law)",
        "p99 latency ≈ 7–10× p50 in typical web services",
        "p999 (tail) is often 10–100× p50 — design for it",
        "A single slow dependency poisons the entire request (fan-out)",
        "Adding more servers improves throughput, not latency",
        "Caching improves latency AND throughput simultaneously",
        "Async processing trades latency for throughput",
    ]
    for i, rule in enumerate(rules, 1):
        print(f"  {i}. {rule}")


if __name__ == "__main__":
    demonstrate_latency_vs_throughput()
