"""
TRADE-OFF ANALYSIS IN HIGH LEVEL DESIGN
=========================================

Problem Statement:
Every design decision in HLD involves trade-offs. There are no universally
correct answers — only choices that are better or worse for a specific context.
The ability to identify, articulate, and defend trade-offs is what separates
senior engineers from juniors in system design interviews.

Common Trade-off Axes:
  Consistency ↔ Availability      (CAP theorem)
  Latency     ↔ Throughput        (batch vs. individual)
  Cost        ↔ Performance       (reserved vs. spot instances)
  Simplicity  ↔ Scalability       (monolith vs. microservices)
  Read speed  ↔ Write speed       (index trade-off)
  Durability  ↔ Speed             (sync vs. async writes)
  Flexibility ↔ Efficiency        (NoSQL vs. SQL)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


class TradeOffDimension(Enum):
    CONSISTENCY   = "consistency"
    AVAILABILITY  = "availability"
    LATENCY       = "latency"
    THROUGHPUT    = "throughput"
    COST          = "cost"
    SIMPLICITY    = "simplicity"
    SCALABILITY   = "scalability"
    DURABILITY    = "durability"
    FLEXIBILITY   = "flexibility"
    MAINTAINABILITY = "maintainability"


@dataclass
class DesignOption:
    name        : str
    description : str
    scores      : Dict[TradeOffDimension, int] = field(default_factory=dict)
    # scores: 1 (poor) → 5 (excellent) on each dimension

    def score(self, dimension: TradeOffDimension, value: int):
        assert 1 <= value <= 5
        self.scores[dimension] = value
        return self

    def weighted_score(self, weights: Dict[TradeOffDimension, float]) -> float:
        total = 0.0
        for dim, weight in weights.items():
            total += self.scores.get(dim, 3) * weight
        return total


class TradeOffMatrix:
    """Compares multiple design options across multiple dimensions."""

    def __init__(self, decision: str, dimensions: List[TradeOffDimension]):
        self.decision   = decision
        self.dimensions = dimensions
        self.options    : List[DesignOption] = []

    def add_option(self, option: DesignOption):
        self.options.append(option)

    def print_matrix(self, weights: Dict[TradeOffDimension, float] = None):
        if weights is None:
            weights = {d: 1.0/len(self.dimensions) for d in self.dimensions}

        print(f"\n  Trade-off Matrix: {self.decision}")
        dim_labels = [d.value[:9] for d in self.dimensions]
        header = f"  {'Option':<22}" + "".join(f"{l:>11}" for l in dim_labels) + "  SCORE"
        print("  " + "─" * (len(header) - 2))
        print(header)
        print("  " + "─" * (len(header) - 2))

        for opt in self.options:
            row = f"  {opt.name:<22}"
            for dim in self.dimensions:
                val = opt.scores.get(dim, 3)
                row += f"  {'★'*val + '☆'*(5-val):>9}"
            ws = opt.weighted_score(weights)
            row += f"  {ws:.1f}"
            print(row)
        print("  " + "─" * (len(header) - 2))

    def recommend(self, weights: Dict[TradeOffDimension, float] = None) -> DesignOption:
        if weights is None:
            weights = {d: 1.0/len(self.dimensions) for d in self.dimensions}
        return max(self.options, key=lambda o: o.weighted_score(weights))


@dataclass
class TradeOff:
    """Represents a single trade-off decision with context."""
    decision_area : str
    chosen        : str
    sacrificed    : str
    context       : str
    example       : str


class CommonTradeOffs:
    """Catalogue of standard HLD trade-offs with real-world examples."""

    CATALOGUE: List[TradeOff] = [
        TradeOff("Consistency vs Availability",
                 "Availability (AP)", "Strong Consistency",
                 "Shopping cart — losing a cart item is better than cart page being down",
                 "Amazon DynamoDB for shopping cart"),
        TradeOff("Read speed vs Write speed",
                 "Faster reads", "Slower writes",
                 "Build indexes to speed reads; indexes slow down writes",
                 "MySQL: covering index on user_id, created_at for timeline queries"),
        TradeOff("Latency vs Throughput",
                 "Higher Throughput", "Higher Latency",
                 "Batch DB writes: 100ms latency per batch but 1000× more writes/sec",
                 "Kafka batching: linger.ms=5 → 5ms extra latency, 10× throughput"),
        TradeOff("Simplicity vs Scalability",
                 "Scalability", "Simplicity",
                 "Microservices allow independent scaling but add ops complexity",
                 "Netflix: 700+ microservices (vs. original DVD-era monolith)"),
        TradeOff("Cost vs Performance",
                 "Performance", "Cost",
                 "SSD (NVMe) vs HDD: 100× faster random reads at 5× cost",
                 "Hot data on SSD; cold archive data on S3 Glacier"),
        TradeOff("Flexibility vs Efficiency",
                 "Flexibility", "Storage efficiency",
                 "NoSQL document stores allow schema evolution; waste space on sparse fields",
                 "MongoDB user profiles vs PostgreSQL (fixed schema)"),
        TradeOff("Durability vs Write speed",
                 "Durability", "Write throughput",
                 "Synchronous replication ensures no data loss but adds write latency",
                 "MySQL: sync binlog replication (safe) vs async (fast but potential loss)"),
    ]

    @classmethod
    def print_catalogue(cls):
        print("\n  COMMON HLD TRADE-OFF CATALOGUE:")
        for t in cls.CATALOGUE:
            print(f"\n  📊 {t.decision_area}")
            print(f"     Chose  : {t.chosen}")
            print(f"     Gave up: {t.sacrificed}")
            print(f"     Context: {t.context}")
            print(f"     Example: {t.example}")


def demonstrate_trade_off_analysis():
    print("=" * 65)
    print("TRADE-OFF ANALYSIS IN HLD")
    print("=" * 65)

    # ── Decision 1: SQL vs NoSQL for user profiles ────
    print("\n[DECISION 1] Database for User Profiles (Instagram scale)")
    print("─" * 55)
    dims1 = [TradeOffDimension.SCALABILITY, TradeOffDimension.CONSISTENCY,
             TradeOffDimension.FLEXIBILITY, TradeOffDimension.LATENCY,
             TradeOffDimension.SIMPLICITY]
    matrix1 = TradeOffMatrix("User Profile Storage", dims1)

    opt_sql = (DesignOption("PostgreSQL", "Relational DB with strict schema")
               .score(TradeOffDimension.SCALABILITY, 3)
               .score(TradeOffDimension.CONSISTENCY, 5)
               .score(TradeOffDimension.FLEXIBILITY, 2)
               .score(TradeOffDimension.LATENCY, 4)
               .score(TradeOffDimension.SIMPLICITY, 4))

    opt_nosql = (DesignOption("DynamoDB", "Key-value store, partition by user_id")
                 .score(TradeOffDimension.SCALABILITY, 5)
                 .score(TradeOffDimension.CONSISTENCY, 3)
                 .score(TradeOffDimension.FLEXIBILITY, 4)
                 .score(TradeOffDimension.LATENCY, 5)
                 .score(TradeOffDimension.SIMPLICITY, 3))

    opt_doc = (DesignOption("MongoDB", "Document store, flexible schema")
               .score(TradeOffDimension.SCALABILITY, 4)
               .score(TradeOffDimension.CONSISTENCY, 3)
               .score(TradeOffDimension.FLEXIBILITY, 5)
               .score(TradeOffDimension.LATENCY, 4)
               .score(TradeOffDimension.SIMPLICITY, 3))

    matrix1.add_option(opt_sql)
    matrix1.add_option(opt_nosql)
    matrix1.add_option(opt_doc)

    # Scale-focused weights (Instagram has 2B users)
    weights1 = {TradeOffDimension.SCALABILITY: 0.40,
                TradeOffDimension.LATENCY: 0.25,
                TradeOffDimension.CONSISTENCY: 0.15,
                TradeOffDimension.FLEXIBILITY: 0.10,
                TradeOffDimension.SIMPLICITY: 0.10}
    matrix1.print_matrix(weights1)
    rec = matrix1.recommend(weights1)
    print(f"\n  Recommendation: {rec.name}  ← best for 2B user scale with low-latency profile reads")

    # ── Decision 2: Cache strategy ────────────
    print("\n\n[DECISION 2] Cache Strategy for News Feed")
    print("─" * 55)
    dims2 = [TradeOffDimension.CONSISTENCY, TradeOffDimension.LATENCY,
             TradeOffDimension.THROUGHPUT, TradeOffDimension.SIMPLICITY]
    matrix2 = TradeOffMatrix("Feed Caching Strategy", dims2)

    opts = [
        (DesignOption("Cache-aside (lazy)", "Load on miss, TTL expiry")
         .score(TradeOffDimension.CONSISTENCY, 3).score(TradeOffDimension.LATENCY, 4)
         .score(TradeOffDimension.THROUGHPUT, 4).score(TradeOffDimension.SIMPLICITY, 5)),
        (DesignOption("Write-through", "Write to cache AND DB on every write")
         .score(TradeOffDimension.CONSISTENCY, 5).score(TradeOffDimension.LATENCY, 3)
         .score(TradeOffDimension.THROUGHPUT, 3).score(TradeOffDimension.SIMPLICITY, 3)),
        (DesignOption("Write-behind async", "Write cache first, flush to DB async")
         .score(TradeOffDimension.CONSISTENCY, 2).score(TradeOffDimension.LATENCY, 5)
         .score(TradeOffDimension.THROUGHPUT, 5).score(TradeOffDimension.SIMPLICITY, 2)),
    ]
    for opt in opts:
        matrix2.add_option(opt)

    weights2 = {TradeOffDimension.LATENCY: 0.40,
                TradeOffDimension.CONSISTENCY: 0.30,
                TradeOffDimension.THROUGHPUT: 0.20,
                TradeOffDimension.SIMPLICITY: 0.10}
    matrix2.print_matrix(weights2)
    rec2 = matrix2.recommend(weights2)
    print(f"\n  Recommendation: {rec2.name}  ← balance of freshness and simplicity for feeds")

    # ── Decision 3: Sync vs Async payment ────
    print("\n\n[DECISION 3] Payment Processing: Sync vs Async")
    print("─" * 55)
    dims3 = [TradeOffDimension.CONSISTENCY, TradeOffDimension.LATENCY,
             TradeOffDimension.DURABILITY, TradeOffDimension.THROUGHPUT]
    matrix3 = TradeOffMatrix("Payment Processing", dims3)

    p1 = (DesignOption("Synchronous (direct)", "Call payment processor inline")
          .score(TradeOffDimension.CONSISTENCY, 5).score(TradeOffDimension.LATENCY, 2)
          .score(TradeOffDimension.DURABILITY, 4).score(TradeOffDimension.THROUGHPUT, 2))
    p2 = (DesignOption("Async + idempotency", "Queue + worker + idempotency key")
          .score(TradeOffDimension.CONSISTENCY, 4).score(TradeOffDimension.LATENCY, 5)
          .score(TradeOffDimension.DURABILITY, 5).score(TradeOffDimension.THROUGHPUT, 5))
    matrix3.add_option(p1)
    matrix3.add_option(p2)

    weights3 = {TradeOffDimension.CONSISTENCY: 0.40,
                TradeOffDimension.DURABILITY: 0.30,
                TradeOffDimension.THROUGHPUT: 0.20,
                TradeOffDimension.LATENCY: 0.10}
    matrix3.print_matrix(weights3)
    rec3 = matrix3.recommend(weights3)
    print(f"\n  Recommendation: {rec3.name}  ← durability > speed for money movement")

    # ── Common Trade-off Catalogue ────────────
    print("\n\n[COMMON TRADE-OFF CATALOGUE]")
    CommonTradeOffs.print_catalogue()

    # ── Key Interview Principle ───────────────
    print("\n\n[KEY INTERVIEW PRINCIPLE]")
    print("─" * 55)
    print("  ✅ There is no right answer — only justified answers.")
    print("  ✅ State your assumption, make a choice, defend it.")
    print("  ✅ Acknowledge what you gave up — shows senior thinking.")
    print("  ✅ Revisit trade-offs if requirements change mid-interview.")


if __name__ == "__main__":
    demonstrate_trade_off_analysis()
