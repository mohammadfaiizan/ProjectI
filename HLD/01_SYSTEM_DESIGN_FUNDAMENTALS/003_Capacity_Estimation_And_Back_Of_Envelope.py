"""
CAPACITY ESTIMATION AND BACK-OF-ENVELOPE CALCULATIONS
======================================================

Problem Statement:
Before picking technologies, engineers must estimate the scale of the system:
QPS, storage, bandwidth, and memory. Quick mental math (back-of-envelope)
keeps the design grounded in reality and avoids over/under-engineering.

Key Concepts:
- QPS / RPS   : Queries / Requests per second
- Storage     : Bytes needed for data at rest
- Bandwidth   : Bytes per second on the wire
- Memory      : RAM needed for caching hot data

Useful Constants (Powers of 2 / Time):
  1 KB = 10^3 B     1 MB = 10^6 B    1 GB = 10^9 B    1 TB = 10^12 B
  1 day = 86,400 s  1 year ≈ 31.5 M s

Latency Numbers Every Engineer Must Know:
  L1 cache hit            :    0.5 ns
  Branch mispredict        :    5   ns
  L2 cache hit            :    7   ns
  Mutex lock/unlock        :   25   ns
  Main memory reference   :  100   ns
  Compress 1KB (Snappy)   :    3   μs
  Send 1KB over network   :   10   μs
  SSD random read         :  150   μs
  Read 1MB sequentially   :  250   μs (memory)
  Round trip within DC    :  500   μs
  SSD sequential read 1MB :    1   ms
  HDD seek                :   10   ms
  Network round trip (CA↔NL): 150 ms
  Read 1GB sequentially   :    5   s  (HDD)
"""

from dataclasses import dataclass
from typing import Dict


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

class StorageUnit:
    BYTE     = 1
    KB       = 1_000
    MB       = 1_000_000
    GB       = 1_000_000_000
    TB       = 1_000_000_000_000
    PB       = 1_000_000_000_000_000

    @staticmethod
    def human(n_bytes: float) -> str:
        for unit, size in [("PB", 1e15), ("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)]:
            if n_bytes >= size:
                return f"{n_bytes/size:.2f} {unit}"
        return f"{n_bytes:.0f} B"


class LatencyTable:
    """Known latency numbers (nanoseconds unless noted)."""
    L1_CACHE_NS        = 0.5
    L2_CACHE_NS        = 7
    RAM_NS             = 100
    COMPRESS_1KB_US    = 3          # microseconds
    NETWORK_SEND_1KB_US= 10
    SSD_RANDOM_READ_US = 150
    SSD_SEQ_1MB_MS     = 1          # milliseconds
    HDD_SEEK_MS        = 10
    NETWORK_SAME_DC_US = 500
    NETWORK_CROSS_REGION_MS = 150

    @classmethod
    def print_table(cls):
        print("\nLATENCY NUMBERS EVERY ENGINEER MUST KNOW:")
        rows = [
            ("L1 cache hit",              f"{cls.L1_CACHE_NS} ns"),
            ("L2 cache hit",              f"{cls.L2_CACHE_NS} ns"),
            ("RAM reference",             f"{cls.RAM_NS} ns"),
            ("Compress 1KB (Snappy)",     f"{cls.COMPRESS_1KB_US} μs"),
            ("Send 1KB over network",     f"{cls.NETWORK_SEND_1KB_US} μs"),
            ("SSD random read",           f"{cls.SSD_RANDOM_READ_US} μs"),
            ("Intra-DC network roundtrip",f"{cls.NETWORK_SAME_DC_US} μs"),
            ("SSD sequential 1MB",        f"{cls.SSD_SEQ_1MB_MS} ms"),
            ("HDD seek",                  f"{cls.HDD_SEEK_MS} ms"),
            ("Cross-region roundtrip",    f"{cls.NETWORK_CROSS_REGION_MS} ms"),
        ]
        for label, value in rows:
            print(f"  {label:<30} {value:>10}")


# ─────────────────────────────────────────────
# ESTIMATION CLASSES
# ─────────────────────────────────────────────

@dataclass
class QPSEstimate:
    label          : str
    daily_events   : float
    seconds_per_day: float = 86_400

    @property
    def average_qps(self) -> float:
        return self.daily_events / self.seconds_per_day

    @property
    def peak_qps(self) -> float:
        # Rule of thumb: peak ≈ 2–3× average
        return self.average_qps * 2.5

    def report(self):
        print(f"  {self.label}:")
        print(f"    Daily events : {self.daily_events:,.0f}")
        print(f"    Avg QPS      : {self.average_qps:,.1f}")
        print(f"    Peak QPS     : {self.peak_qps:,.1f}")


class StorageCalculator:
    """Estimates storage requirements."""

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.components: Dict[str, float] = {}  # name → bytes/day

    def add(self, component: str, bytes_per_event: float, events_per_day: float):
        daily_bytes = bytes_per_event * events_per_day
        self.components[component] = daily_bytes

    def total_daily(self) -> float:
        return sum(self.components.values())

    def total_yearly(self) -> float:
        return self.total_daily() * 365

    def total_for_years(self, years: int) -> float:
        return self.total_yearly() * years

    def report(self, retention_years: int = 5):
        print(f"\n  Storage Estimates for [{self.system_name}]:")
        print(f"  {'Component':<30} {'Daily':>14} {'Yearly':>14}")
        print(f"  {'─'*57}")
        for name, daily_bytes in self.components.items():
            print(f"  {name:<30} {StorageUnit.human(daily_bytes):>14} {StorageUnit.human(daily_bytes*365):>14}")
        print(f"  {'─'*57}")
        print(f"  {'TOTAL Daily':<30} {StorageUnit.human(self.total_daily()):>14}")
        print(f"  {'TOTAL Yearly':<30} {StorageUnit.human(self.total_yearly()):>14}")
        print(f"  {'TOTAL ' + str(retention_years) + ' Years':<30} {StorageUnit.human(self.total_for_years(retention_years)):>14}")


class BandwidthCalculator:
    """Estimates network bandwidth."""

    def __init__(self):
        self.inbound : float = 0.0   # bytes/sec
        self.outbound: float = 0.0

    def add_inbound(self, label: str, bytes_per_request: float, qps: float):
        bw = bytes_per_request * qps
        self.inbound += bw
        print(f"  Inbound  [{label}]: {StorageUnit.human(bytes_per_request)}/req × {qps:.0f} QPS = {StorageUnit.human(bw)}/s")

    def add_outbound(self, label: str, bytes_per_request: float, qps: float):
        bw = bytes_per_request * qps
        self.outbound += bw
        print(f"  Outbound [{label}]: {StorageUnit.human(bytes_per_request)}/req × {qps:.0f} QPS = {StorageUnit.human(bw)}/s")

    def summary(self):
        print(f"\n  Total Inbound  : {StorageUnit.human(self.inbound)}/s")
        print(f"  Total Outbound : {StorageUnit.human(self.outbound)}/s")


class CacheEstimator:
    """Estimates RAM needed to cache hot data (80/20 rule)."""

    def __init__(self, label: str, total_items: int, avg_item_bytes: int, hot_pct: float = 0.20):
        self.label          = label
        self.total_items    = total_items
        self.avg_item_bytes = avg_item_bytes
        self.hot_pct        = hot_pct

    @property
    def hot_items(self) -> int:
        return int(self.total_items * self.hot_pct)

    @property
    def ram_needed(self) -> float:
        return self.hot_items * self.avg_item_bytes

    def report(self):
        print(f"  Cache [{self.label}]:")
        print(f"    Total items     : {self.total_items:,}")
        print(f"    Hot items (20%) : {self.hot_items:,}")
        print(f"    Avg item size   : {self.avg_item_bytes} B")
        print(f"    RAM needed      : {StorageUnit.human(self.ram_needed)}")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_capacity_estimation():
    print("=" * 60)
    print("CAPACITY ESTIMATION: Twitter-like System")
    print("Assumptions: 300M DAU, 500M tweets/day, 100:1 read/write")
    print("=" * 60)

    # ── Latency Reference ─────────────────────
    LatencyTable.print_table()

    # ── QPS Estimates ─────────────────────────
    print("\nQPS ESTIMATES:")
    QPSEstimate("Tweet writes",    500_000_000).report()
    QPSEstimate("Timeline reads",  500_000_000 * 100).report()   # 100x reads
    QPSEstimate("Search queries",  300_000_000 * 0.1).report()   # 10% DAU search/day
    QPSEstimate("Notification delivery", 500_000_000 * 0.3).report()

    # ── Storage ───────────────────────────────
    print()
    storage = StorageCalculator("Twitter")
    # Tweet metadata: ~200 bytes (id, user_id, text, timestamp, counters)
    storage.add("Tweet metadata",       200,         500_000_000)
    # Media (images): 30% of tweets have image, avg 300KB compressed
    storage.add("Tweet images",         300_000,     500_000_000 * 0.3)
    # Video (5% of tweets, avg 5MB)
    storage.add("Tweet videos",         5_000_000,   500_000_000 * 0.05)
    # User profiles (updated rarely): 1KB each, 10M new users/day
    storage.add("User profiles",        1_000,       10_000_000)
    storage.report(retention_years=5)

    # ── Bandwidth ─────────────────────────────
    print("\nBANDWIDTH ESTIMATES:")
    bw = BandwidthCalculator()
    tweet_write_qps  = 500_000_000 / 86_400
    timeline_read_qps = tweet_write_qps * 100
    bw.add_inbound("Tweet write",   200,      tweet_write_qps)
    bw.add_inbound("Image upload",  300_000,  tweet_write_qps * 0.3)
    bw.add_outbound("Timeline feed",500,      timeline_read_qps)   # 10 tweets × 50B each
    bw.add_outbound("Image delivery",300_000, timeline_read_qps * 0.3)
    bw.summary()

    # ── Cache ─────────────────────────────────
    print("\nCACHE ESTIMATES (80/20 rule: 20% items → 80% traffic):")
    CacheEstimator("Tweet cache",    500_000_000, 200,    hot_pct=0.20).report()
    CacheEstimator("Timeline cache", 300_000_000, 5_000,  hot_pct=0.10).report()
    CacheEstimator("User profile",   500_000_000, 1_000,  hot_pct=0.05).report()

    # ── Server Estimate ───────────────────────
    print("\nSERVER ESTIMATES:")
    reads_qps  = timeline_read_qps
    server_cap = 10_000   # each app server handles 10K QPS
    read_servers = int(reads_qps / server_cap) + 1
    write_servers = max(1, int(tweet_write_qps / server_cap) + 1)
    print(f"  Read  servers  : ~{read_servers} (at {server_cap:,} QPS/server)")
    print(f"  Write servers  : ~{write_servers} (at {server_cap:,} QPS/server)")
    print(f"  Rule of thumb  : add 50% headroom → Read: {int(read_servers*1.5)}, Write: {int(write_servers*1.5)}")

    print("\n✅ Capacity estimation complete.")


if __name__ == "__main__":
    demonstrate_capacity_estimation()
