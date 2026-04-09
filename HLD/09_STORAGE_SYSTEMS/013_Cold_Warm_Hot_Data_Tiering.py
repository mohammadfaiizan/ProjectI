"""
COLD / WARM / HOT DATA TIERING
================================

Problem Statement:
Not all data is accessed equally. Storing all data on fast, expensive
NVMe SSDs wastes money. Data tiering matches storage cost to access frequency.

The 80/20 Rule in Storage:
  80% of requests access 20% of data (hot).
  The remaining 80% of data (cold) is rarely accessed.
  Storing cold data on cheap media and hot data on fast media optimizes cost.

Tiers:
  Hot tier:   NVMe SSD / RAM cache. Sub-millisecond. Most expensive.
              Recent data, frequently accessed objects.
  Warm tier:  SATA SSD / HDD. Milliseconds. Moderate cost.
              Less-frequently accessed data (last 30-90 days).
  Cold tier:  Object storage (S3 Standard-IA, GCS Nearline). Cents/GB.
              Data accessed monthly or less (>90 days).
  Frozen tier:S3 Glacier / Deep Archive. Very cheap. Hours to retrieve.
              Compliance archives, legal holds, disaster recovery.

Tiering Policies:
  Time-based: age > N days → move to next tier.
  Access-based: not accessed in N days → demote. Accessed → promote.
  Size-based: small objects stay hot; large objects demote sooner.

Access Patterns to Measure:
  Last access time (LRU). Access frequency (LFU). Write pattern.
  Access recency + frequency → ARC (Adaptive Replacement Cache) policy.

Automated Tiering:
  AWS S3 Intelligent-Tiering: monitors access and moves automatically.
  NetApp ONTAP FabricPool: auto-tiers between SSD and object storage.
  Ceph RADOS: tiering between SSD and spinning disk pools.

Rehydration:
  Fetching data from cold/frozen tier back to hot.
  Glacier: Expedited (1-5 min), Standard (3-5 hours), Bulk (5-12 hours).
  S3 Intelligent-Tiering: automatic restore on access (Instant Retrieval).
  Planning: rehydrate before scheduled access (ETL jobs, reporting windows).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time
import random


# ─────────────────────────────────────────────
# STORAGE TIER DEFINITIONS
# ─────────────────────────────────────────────

class Tier(Enum):
    HOT    = "HOT"
    WARM   = "WARM"
    COLD   = "COLD"
    FROZEN = "FROZEN"


@dataclass
class TierConfig:
    tier            : Tier
    cost_per_gb_mo  : float   # $/GB/month
    latency_ms      : float   # read latency in milliseconds
    throughput_mbps : float   # MB/s
    demote_after_days: Optional[int]  # days of no access before demotion
    restore_time_ms : float   # time to make available if not cached


TIER_CONFIGS = {
    Tier.HOT   : TierConfig(Tier.HOT,    0.10,   0.1,   3000,  7,   0),
    Tier.WARM  : TierConfig(Tier.WARM,   0.023,  5.0,    500,  30,  100),
    Tier.COLD  : TierConfig(Tier.COLD,   0.0125, 50.0,   200,  90,  5_000),
    Tier.FROZEN: TierConfig(Tier.FROZEN, 0.004,  None,   50,   None,3_600_000),
}


# ─────────────────────────────────────────────
# TIERED OBJECT
# ─────────────────────────────────────────────

@dataclass
class TieredObject:
    object_id   : str
    size_bytes  : int
    data        : bytes
    tier        : Tier
    created_at  : float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int    = 0
    tier_moves  : List[Tuple[Tier, float]] = field(default_factory=list)  # [(tier, ts)]

    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def idle_days(self) -> float:
        return (time.time() - self.last_accessed) / 86400

    def move_to(self, new_tier: Tier):
        self.tier_moves.append((self.tier, time.time()))
        self.tier = new_tier

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)


# ─────────────────────────────────────────────
# TIERED STORAGE ENGINE
# ─────────────────────────────────────────────

class TieredStorageEngine:
    """
    Multi-tier storage with automatic promotion and demotion.
    Uses access recency + frequency for tiering decisions.
    """

    def __init__(self, tier_configs: Dict[Tier, TierConfig] = None):
        self._configs  = tier_configs or TIER_CONFIGS
        self._objects  : Dict[str, TieredObject] = {}
        self.promotions = 0
        self.demotions  = 0
        self.reads      = 0
        self.rehydrations = 0

    def store(self, object_id: str, data: bytes, tier: Tier = Tier.HOT) -> TieredObject:
        obj = TieredObject(object_id=object_id, size_bytes=len(data),
                            data=data, tier=tier)
        self._objects[object_id] = obj
        return obj

    def read(self, object_id: str) -> Tuple[Optional[bytes], float]:
        """Returns (data, latency_ms). Promotes on access."""
        obj = self._objects.get(object_id)
        if obj is None:
            return None, 0.0

        config = self._configs[obj.tier]
        latency = config.restore_time_ms + config.latency_ms if config.latency_ms else config.restore_time_ms

        obj.last_accessed = time.time()
        obj.access_count  += 1
        self.reads += 1

        # Auto-promote if accessed from cold/frozen
        if obj.tier in (Tier.COLD, Tier.FROZEN):
            self._promote(obj)
            self.rehydrations += 1

        return obj.data, latency

    def _promote(self, obj: TieredObject):
        """Move object up one tier."""
        tiers_ordered = [Tier.FROZEN, Tier.COLD, Tier.WARM, Tier.HOT]
        idx = tiers_ordered.index(obj.tier)
        if idx < len(tiers_ordered) - 1:
            obj.move_to(tiers_ordered[idx + 1])
            self.promotions += 1

    def _demote(self, obj: TieredObject):
        """Move object down one tier."""
        tiers_ordered = [Tier.HOT, Tier.WARM, Tier.COLD, Tier.FROZEN]
        idx = tiers_ordered.index(obj.tier)
        if idx < len(tiers_ordered) - 1:
            obj.move_to(tiers_ordered[idx + 1])
            self.demotions += 1

    def run_tiering_policy(self, simulate_days_elapsed: float = 0) -> Dict:
        """
        Apply tiering policy: demote objects that haven't been accessed
        for longer than the tier's demote_after_days threshold.
        """
        demoted = 0
        for obj in self._objects.values():
            idle   = obj.idle_days() + simulate_days_elapsed
            config = self._configs[obj.tier]
            if config.demote_after_days and idle > config.demote_after_days:
                self._demote(obj)
                demoted += 1
        return {"demoted": demoted}

    def monthly_cost(self) -> Dict[Tier, float]:
        costs: Dict[Tier, float] = {}
        for obj in self._objects.values():
            cfg  = self._configs[obj.tier]
            cost = obj.size_gb * cfg.cost_per_gb_mo
            costs[obj.tier] = costs.get(obj.tier, 0.0) + cost
        return costs

    def tier_distribution(self) -> Dict[Tier, Dict]:
        dist: Dict[Tier, Dict] = {t: {"count": 0, "bytes": 0} for t in Tier}
        for obj in self._objects.values():
            dist[obj.tier]["count"]  += 1
            dist[obj.tier]["bytes"]  += obj.size_bytes
        return dist

    def hot_objects(self, top_n: int = 5) -> List[TieredObject]:
        """Most frequently accessed objects."""
        return sorted(self._objects.values(),
                      key=lambda o: o.access_count, reverse=True)[:top_n]

    def stats(self) -> Dict:
        return {
            "total_objects" : len(self._objects),
            "promotions"    : self.promotions,
            "demotions"     : self.demotions,
            "rehydrations"  : self.rehydrations,
            "reads"         : self.reads,
        }


# ─────────────────────────────────────────────
# COST OPTIMIZER
# ─────────────────────────────────────────────

def optimize_tiering(objects: List[Dict],
                     budget_per_month: float) -> List[Dict]:
    """
    Simple greedy: put most-accessed objects in cheapest tier that fits budget.
    """
    tiers_asc_cost = sorted(TIER_CONFIGS.values(), key=lambda c: -c.cost_per_gb_mo)
    assignments = []
    spent = 0.0
    for obj in sorted(objects, key=lambda o: -o["accesses_per_day"]):
        size_gb = obj["size_bytes"] / (1024 ** 3)
        for tier_cfg in tiers_asc_cost:
            monthly_cost = size_gb * tier_cfg.cost_per_gb_mo
            if spent + monthly_cost <= budget_per_month:
                assignments.append({**obj, "tier": tier_cfg.tier.value,
                                    "cost": monthly_cost,
                                    "latency_ms": tier_cfg.latency_ms})
                spent += monthly_cost
                break
        else:
            # Must assign to cheapest tier regardless
            tier_cfg = min(TIER_CONFIGS.values(), key=lambda c: c.cost_per_gb_mo)
            assignments.append({**obj, "tier": tier_cfg.tier.value,
                                "cost": size_gb * tier_cfg.cost_per_gb_mo})
    return assignments, spent


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_tiering():
    print("=" * 65)
    print("COLD / WARM / HOT DATA TIERING")
    print("=" * 65)

    engine = TieredStorageEngine()
    random.seed(42)

    # ── Store Objects in Different Tiers ──────────
    print("\n[1] INITIAL DATA PLACEMENT")
    print("─" * 55)

    objects = []
    for i in range(20):
        size = random.choice([1, 10, 100, 1000]) * 1024 * 1024  # 1MB, 10MB, 100MB, 1GB
        tier = random.choice([Tier.HOT, Tier.HOT, Tier.WARM, Tier.COLD])
        obj  = engine.store(f"obj-{i:02d}", b"data" * (size // 4), tier)
        objects.append(obj)

    dist = engine.tier_distribution()
    for tier, info in dist.items():
        size_gb = info["bytes"] / (1024**3)
        if info["count"] > 0:
            print(f"  {tier.value:<8}: {info['count']:2} objects  {size_gb:.2f} GB")

    # ── Access Pattern Simulation ─────────────────
    print("\n\n[2] ACCESS SIMULATION — HOT OBJECTS GET PROMOTED")
    print("─" * 55)

    # Simulate realistic access: some objects accessed frequently
    hot_ids   = ["obj-00", "obj-03", "obj-07"]
    warm_ids  = ["obj-01", "obj-05", "obj-10"]

    for _ in range(20):
        for oid in hot_ids:
            engine.read(oid)
        for oid in warm_ids:
            engine.read(oid)
    for _ in range(3):
        engine.read("obj-15")  # cold object accessed → rehydrated

    print(f"  Reads done: {engine.reads}")
    print(f"  Rehydrations (cold→warm): {engine.rehydrations}")
    print(f"  Promotions: {engine.promotions}")

    print(f"\n  Top 5 hottest objects:")
    for obj in engine.hot_objects(5):
        _, lat = engine.read(obj.object_id)
        print(f"    {obj.object_id}: {obj.access_count} accesses, "
              f"tier={obj.tier.value}, size={obj.size_bytes//1024//1024}MB")

    # ── Tiering Policy Execution ──────────────────
    print("\n\n[3] AUTOMATED TIERING POLICY")
    print("─" * 55)

    tier_before = {t: v["count"] for t, v in engine.tier_distribution().items()}
    result = engine.run_tiering_policy(simulate_days_elapsed=31)
    tier_after  = {t: v["count"] for t, v in engine.tier_distribution().items()}

    print(f"  Simulate 31 days elapsed:")
    print(f"  Objects demoted: {result['demoted']}")
    print(f"  {'Tier':<10} {'Before':>8} {'After':>8} {'Change':>8}")
    print(f"  {'─'*36}")
    for tier in Tier:
        b = tier_before[tier]
        a = tier_after.get(tier, 0)
        print(f"  {tier.value:<10} {b:>8} {a:>8} {a - b:>+8}")

    # ── Cost Analysis ─────────────────────────────
    print("\n\n[4] MONTHLY COST BY TIER")
    print("─" * 55)

    costs = engine.monthly_cost()
    total_cost = sum(costs.values())
    for tier in Tier:
        cost = costs.get(tier, 0)
        if cost > 0:
            cfg = TIER_CONFIGS[tier]
            print(f"  {tier.value:<8}: ${cost:>8.4f}/mo  "
                  f"(${cfg.cost_per_gb_mo}/GB)  {cost/total_cost:.0%} of bill")
    print(f"  {'Total':<8}: ${total_cost:>8.4f}/mo")

    # ── Tier Comparison ───────────────────────────
    print("\n\n[5] TIER CHARACTERISTICS")
    print("─" * 55)

    print(f"  {'Tier':<10} {'$/GB/mo':>9} {'Latency':>12} {'Throughput':>12} {'Demote after'}")
    print(f"  {'─'*58}")
    for tier, cfg in TIER_CONFIGS.items():
        latency  = f"{cfg.latency_ms:.1f}ms" if cfg.latency_ms else "hours"
        restore  = f"{cfg.restore_time_ms/1000:.0f}s" if cfg.restore_time_ms >= 1000 else "instant"
        demote   = f"{cfg.demote_after_days}d" if cfg.demote_after_days else "never"
        print(f"  {tier.value:<10} ${cfg.cost_per_gb_mo:>7.4f}  "
              f"{latency:>10}  {cfg.throughput_mbps:>8.0f}MB/s  {demote}")

    # ── S3 Intelligent-Tiering ────────────────────
    print("\n\n[6] S3 INTELLIGENT-TIERING SIMULATION")
    print("─" * 55)

    transitions = [
        ("Frequent Access",   0,    "Accesses daily/weekly"),
        ("Infrequent Access", 30,   "Not accessed for 30 days → auto-moved"),
        ("Archive Instant",   90,   "Not accessed for 90 days → archive"),
        ("Deep Archive",      180,  "Not accessed for 180 days → deep archive"),
    ]
    print(f"  {'Access Class':<24} {'Days Idle':>10} {'Action'}")
    print(f"  {'─'*58}")
    for cls, days, action in transitions:
        print(f"  {cls:<24} {days:>10}  {action}")

    print("\n  Cost savings vs all-Standard:")
    print("  After 30d: 46% cheaper (IA tier)")
    print("  After 90d: 83% cheaper (Archive Instant)")


if __name__ == "__main__":
    demonstrate_tiering()
