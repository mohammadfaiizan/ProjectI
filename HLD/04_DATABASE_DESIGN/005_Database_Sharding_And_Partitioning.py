"""
DATABASE SHARDING AND PARTITIONING
=====================================

Problem Statement:
When a single database node can't handle the data volume or write throughput,
you need to split data across multiple nodes. Sharding (horizontal partitioning)
distributes rows across nodes; vertical partitioning splits columns.

Sharding Strategies:
  Range Sharding  : shard by value range (user_id 0-1M → shard1, etc.)
                    → Hot spots (recent data), simple range queries
  Hash Sharding   : hash(key) % N → even distribution
                    → No hot spots, but range queries hit all shards
  Directory-Based : lookup table maps key → shard
                    → Most flexible, requires lookup overhead
  Geo-Based       : route by geography (EU users → EU shard)
                    → Low latency, data sovereignty compliance

Partitioning (within one node):
  Horizontal (row-based): same as sharding but within one DB
  Vertical (column-based): split columns across tables
  List Partition: specific values per partition
  Range Partition: date range per partition (common for time-series)
  Hash Partition: even distribution within one DB

Challenges:
  Cross-shard joins: expensive or impossible — must denormalize
  Cross-shard transactions: 2PC required (complex)
  Rebalancing: adding a shard requires moving data (consistent hashing helps)
  Hot spots: certain shards get more traffic (range-based)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import time
import random


class ShardStrategy(Enum):
    RANGE       = "range"
    HASH        = "hash"
    DIRECTORY   = "directory"
    CONSISTENT_HASH = "consistent_hash"


@dataclass
class Shard:
    shard_id   : str
    host       : str
    port       : int
    _data      : Dict[str, Any] = field(default_factory=dict)
    writes     : int = 0
    reads      : int = 0

    def write(self, key: str, value: Any):
        self._data[key] = value
        self.writes += 1

    def read(self, key: str) -> Optional[Any]:
        self.reads += 1
        return self._data.get(key)

    def count(self) -> int:
        return len(self._data)

    def keys(self) -> List[str]:
        return list(self._data.keys())


# ─────────────────────────────────────────────
# HASH SHARD ROUTER
# ─────────────────────────────────────────────

class HashShardRouter:
    """
    Routes keys by hash(key) % num_shards.
    Even distribution but breaks range queries.
    """

    def __init__(self, shards: List[Shard]):
        self.shards    = shards
        self.num_shards = len(shards)
        self.total_writes = 0
        self.total_reads  = 0

    def _shard_for(self, key: str) -> Shard:
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return self.shards[h % self.num_shards]

    def write(self, key: str, value: Any):
        shard = self._shard_for(key)
        shard.write(key, value)
        self.total_writes += 1

    def read(self, key: str) -> Optional[Any]:
        shard = self._shard_for(key)
        self.total_reads += 1
        return shard.read(key)

    def scatter_gather(self, predicate) -> List[Any]:
        """Range/filter query: must hit all shards (expensive)."""
        results = []
        for shard in self.shards:
            for k, v in shard._data.items():
                if predicate(v):
                    results.append(v)
        return results

    def distribution_report(self):
        print(f"\n  Hash Shard Distribution ({self.num_shards} shards):")
        total = sum(s.count() for s in self.shards)
        for s in self.shards:
            pct = s.count() / max(1, total) * 100
            bar = "█" * int(pct / 2)
            print(f"    {s.shard_id}: {s.count():>6} rows ({pct:.1f}%)  {bar}")


# ─────────────────────────────────────────────
# RANGE SHARD ROUTER
# ─────────────────────────────────────────────

@dataclass
class RangeShardConfig:
    shard    : Shard
    low      : int
    high     : int

    def contains(self, value: int) -> bool:
        return self.low <= value < self.high


class RangeShardRouter:
    """
    Routes by numeric range.
    Range queries efficient within one shard; hot spots possible.
    """

    def __init__(self):
        self._ranges : List[RangeShardConfig] = []

    def add_range(self, shard: Shard, low: int, high: int):
        self._ranges.append(RangeShardConfig(shard, low, high))

    def _shard_for(self, key_val: int) -> Optional[Shard]:
        for r in self._ranges:
            if r.contains(key_val):
                return r.shard
        return None

    def write(self, user_id: int, value: Any) -> bool:
        shard = self._shard_for(user_id)
        if not shard:
            print(f"  No shard for user_id={user_id}")
            return False
        shard.write(str(user_id), value)
        return True

    def read(self, user_id: int) -> Optional[Any]:
        shard = self._shard_for(user_id)
        return shard.read(str(user_id)) if shard else None

    def range_query(self, lo: int, hi: int) -> List[Any]:
        """Range query may hit one or several shards."""
        shards_needed = {r.shard for r in self._ranges
                          if not (r.high <= lo or r.low >= hi)}
        results = []
        for shard in shards_needed:
            for k, v in shard._data.items():
                if lo <= int(k) < hi:
                    results.append(v)
        return results

    def distribution_report(self):
        print(f"\n  Range Shard Distribution:")
        for r in self._ranges:
            print(f"    {r.shard.shard_id}: [{r.low:,}–{r.high:,})  "
                  f"{r.shard.count()} rows  writes={r.shard.writes}")


# ─────────────────────────────────────────────
# CONSISTENT HASHING
# ─────────────────────────────────────────────

class ConsistentHashRing:
    """
    Maps keys to nodes on a virtual ring.
    Adding/removing a node only moves ~1/N keys (vs N-1 nodes for mod hashing).
    Each node has multiple virtual nodes (vnodes) for even distribution.
    """

    def __init__(self, vnodes_per_shard: int = 150):
        self.vnodes_per_shard = vnodes_per_shard
        self._ring    : Dict[int, str] = {}   # position → shard_id
        self._shards  : Dict[str, Shard] = {}
        self._sorted_positions: List[int] = []

    def add_shard(self, shard: Shard):
        self._shards[shard.shard_id] = shard
        for i in range(self.vnodes_per_shard):
            key = f"{shard.shard_id}:vnode:{i}"
            pos = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)
            self._ring[pos] = shard.shard_id
        self._sorted_positions = sorted(self._ring.keys())

    def remove_shard(self, shard_id: str):
        self._shards.pop(shard_id, None)
        self._ring = {pos: sid for pos, sid in self._ring.items() if sid != shard_id}
        self._sorted_positions = sorted(self._ring.keys())

    def _shard_for(self, key: str) -> Optional[Shard]:
        if not self._sorted_positions:
            return None
        h = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)
        # Find first position >= h (clockwise)
        idx = 0
        for i, pos in enumerate(self._sorted_positions):
            if pos >= h:
                idx = i
                break
        else:
            idx = 0   # wrap around
        shard_id = self._ring[self._sorted_positions[idx]]
        return self._shards.get(shard_id)

    def write(self, key: str, value: Any):
        shard = self._shard_for(key)
        if shard:
            shard.write(key, value)

    def read(self, key: str) -> Optional[Any]:
        shard = self._shard_for(key)
        return shard.read(key) if shard else None

    def keys_affected_by_add(self, new_shard_id: str) -> int:
        """Estimate how many keys move when adding a node."""
        if not self._shards:
            return 0
        total = sum(s.count() for s in self._shards.values())
        return total // (len(self._shards) + 1)   # approximately 1/N keys

    def distribution_report(self):
        print(f"\n  Consistent Hash Distribution:")
        for sid, shard in self._shards.items():
            print(f"    {sid}: {shard.count()} rows  vnodes={self.vnodes_per_shard}")


# ─────────────────────────────────────────────
# TABLE PARTITIONING (within single DB)
# ─────────────────────────────────────────────

class RangePartitionedTable:
    """
    PostgreSQL-style declarative partitioning.
    Partitions data by date range (monthly, yearly).
    """

    def __init__(self, table_name: str):
        self.table_name  = table_name
        self._partitions : Dict[str, List[Dict]] = {}

    def add_partition(self, name: str, year: int, month: int):
        self._partitions[name] = []

    def _partition_for(self, timestamp: float) -> str:
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        return f"p_{dt.year}_{dt.month:02d}"

    def insert(self, row: Dict):
        ts   = row.get("created_at", time.time())
        part = self._partition_for(ts)
        self._partitions.setdefault(part, []).append(row)

    def query_by_range(self, start_ts: float, end_ts: float) -> List[Dict]:
        """Partition pruning: only scan relevant partitions."""
        import datetime
        start = datetime.datetime.fromtimestamp(start_ts)
        end   = datetime.datetime.fromtimestamp(end_ts)
        result = []
        for part_name, rows in self._partitions.items():
            # Parse partition key: p_2024_01
            try:
                parts = part_name.split("_")
                year, month = int(parts[1]), int(parts[2])
                part_ts = datetime.datetime(year, month, 1).timestamp()
                if start_ts <= part_ts + 32 * 86400 and part_ts <= end_ts:
                    result.extend(r for r in rows
                                   if start_ts <= r.get("created_at", 0) <= end_ts)
            except (IndexError, ValueError):
                pass
        return result

    def show(self):
        print(f"\n  Table: {self.table_name} (range partitioned by created_at)")
        total = sum(len(rows) for rows in self._partitions.values())
        for pname, rows in sorted(self._partitions.items()):
            pct = len(rows) / max(1, total) * 100
            print(f"    {pname}: {len(rows):>6} rows ({pct:.0f}%)")
        print(f"    Total: {total} rows")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_sharding():
    print("=" * 65)
    print("DATABASE SHARDING AND PARTITIONING")
    print("=" * 65)
    random.seed(42)

    # ── Hash Sharding ─────────────────────────
    print("\n[1] HASH SHARDING (4 shards)")
    print("─" * 55)
    hash_shards = [Shard(f"shard-{i}", f"10.0.{i}.1", 5432) for i in range(4)]
    hash_router = HashShardRouter(hash_shards)

    N = 10_000
    for i in range(N):
        hash_router.write(f"user:{i}", {"user_id": i, "name": f"User{i}", "active": i % 3 != 0})

    hash_router.distribution_report()
    print(f"  Hash sharding distributes ~evenly: no hot spots")

    # Range query requires scatter-gather
    active = hash_router.scatter_gather(lambda v: v.get("active", False))
    print(f"  Filter active=True: scanned all shards, found {len(active)} rows")

    # ── Range Sharding ────────────────────────
    print("\n\n[2] RANGE SHARDING (4 shards by user_id)")
    print("─" * 55)
    range_shards = [Shard(f"range-{i}", f"10.1.{i}.1", 5432) for i in range(4)]
    range_router = RangeShardRouter()
    range_router.add_range(range_shards[0], 0,       250_000)
    range_router.add_range(range_shards[1], 250_000, 500_000)
    range_router.add_range(range_shards[2], 500_000, 750_000)
    range_router.add_range(range_shards[3], 750_000, 1_000_000)

    for i in range(N):
        uid = random.randint(0, 999_999)
        range_router.write(uid, {"user_id": uid, "name": f"User{uid}"})

    range_router.distribution_report()

    # Range query is efficient
    result = range_router.range_query(lo=100_000, hi=120_000)
    print(f"  Range query [100K–120K]: {len(result)} rows from 1 shard only")

    # ── Consistent Hashing ────────────────────
    print("\n\n[3] CONSISTENT HASHING (adding shard without full reshuffle)")
    print("─" * 55)
    ch = ConsistentHashRing(vnodes_per_shard=50)
    for i in range(3):
        ch.add_shard(Shard(f"ch-{i}", f"10.2.{i}.1", 5432))

    for i in range(1000):
        ch.write(f"key:{i}", {"id": i, "data": f"value_{i}"})

    ch.distribution_report()
    keys_moving = ch.keys_affected_by_add("ch-3-new")
    print(f"\n  Adding new shard: ~{keys_moving} keys need to move (out of 1000)")
    print(f"  Mod hashing: would move ~750 keys (3/4 of total)")

    # ── Table Partitioning ────────────────────
    print("\n\n[4] RANGE PARTITIONING (PostgreSQL-style by month)")
    print("─" * 55)
    events = RangePartitionedTable("events")
    import datetime
    base = datetime.datetime(2024, 1, 1).timestamp()
    for i in range(5000):
        ts = base + random.randint(0, 365 * 86400)
        events.insert({"event_id": i, "type": "click", "created_at": ts})

    events.show()
    start_ts = datetime.datetime(2024, 3, 1).timestamp()
    end_ts   = datetime.datetime(2024, 3, 31).timestamp()
    march = events.query_by_range(start_ts, end_ts)
    print(f"  Query March 2024: scans only March partition → {len(march)} rows")

    # ── Sharding Challenges ───────────────────
    print("\n\n[5] SHARDING CHALLENGES AND SOLUTIONS")
    print("─" * 55)
    challenges = [
        ("Cross-shard joins",     "Denormalize data; use single-shard data models"),
        ("Cross-shard txns",      "2PC (slow) or Saga pattern (async compensation)"),
        ("Rebalancing shards",    "Consistent hashing — move only 1/N keys"),
        ("Hot spots",             "Use hash sharding; add shard-level caching"),
        ("Auto-increment IDs",    "Use UUID/Snowflake IDs — no global sequence"),
        ("Schema migrations",     "Run migration on each shard; blue-green per shard"),
        ("Uneven data growth",    "Virtual shards — one physical node hosts multiple"),
    ]
    for challenge, solution in challenges:
        print(f"  • {challenge:<28} → {solution}")


if __name__ == "__main__":
    demonstrate_sharding()
