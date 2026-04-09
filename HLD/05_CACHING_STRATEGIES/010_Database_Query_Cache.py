"""
DATABASE QUERY CACHE
======================

Problem Statement:
Expensive DB queries (complex JOINs, aggregations, full-table scans)
take 100ms-10s. The same query with the same parameters may be issued
hundreds of times per second. Caching query results in Redis gives
sub-millisecond responses for repeated queries.

Approaches:

  1. Application-Level Query Cache:
     App code: check cache → miss → run DB query → set cache → return.
     Fine-grained control: choose what to cache, with what TTL.
     Most common production approach.

  2. ORM Cache (SQLAlchemy dogpile.cache, Django's cache_page):
     Cache integrated into ORM. Transparent to business logic.
     Works per query or per view.

  3. MySQL Query Cache (deprecated in MySQL 8.0):
     Server-side cache of SELECT results.
     Invalidated on ANY write to a referenced table (too aggressive).
     Removed in MySQL 8.0 due to mutex contention issues.

  4. Materialized Views (PostgreSQL / Snowflake):
     Pre-computed query stored as a table.
     REFRESH MATERIALIZED VIEW updates on demand or schedule.
     Not a cache per se — persistent, persistent, scheduled refresh.

  5. Read-Through ORM Pattern:
     cache.get_or_set(key, lambda: db.query(...), timeout=300)
     Simple: default to DB query if cache miss, set cache on load.

Cache Key Design for Queries:
  Key must uniquely identify the query and its parameters.
  Simple: hash(sql + sorted(params))
  Include: query name, tenant_id, pagination params
  Avoid: including timestamps in key (creates unbounded key space)

Invalidation for DB Query Cache:
  TTL: simplest — accept staleness window
  Table-level invalidation: any write to table X → delete all keys for table X
  Row-level invalidation: only invalidate specific entity (entity_id)
  CQRS: write side publishes events → cache layer invalidates specific keys

Query Cache Anti-patterns:
  Caching write-heavy queries → constant invalidation = low hit ratio
  No TTL → stale data grows indefinitely
  Caching per-user queries without tenant scoping → cache key collisions
  Too fine-grained keys → unbounded key explosion
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
import hashlib
import time
import random
from collections import defaultdict


# ─────────────────────────────────────────────
# QUERY CACHE KEY BUILDER
# ─────────────────────────────────────────────

class QueryCacheKey:
    """Builds deterministic cache keys for SQL queries and their parameters."""

    @staticmethod
    def build(query_name: str, params: Dict[str, Any],
              tenant_id: str = None, version: int = 1) -> str:
        """
        Build a stable cache key from query name + params.
        - Sort params for determinism
        - Include tenant_id for multi-tenancy
        - Include version for cache busting on deploy
        """
        param_str  = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
        tenant_pfx = f"t{tenant_id}:" if tenant_id else ""
        raw        = f"v{version}:{tenant_pfx}{query_name}:{param_str}"
        digest     = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"qc:{query_name}:{digest}"

    @staticmethod
    def table_tag(table_name: str) -> str:
        """Tag key for table-level invalidation."""
        return f"table:{table_name}"


# ─────────────────────────────────────────────
# SIMULATED DB
# ─────────────────────────────────────────────

class Database:
    def __init__(self, latency_ms: float = 50.0):
        self.latency_ms = latency_ms
        self.queries    = 0
        self.total_ms   = 0.0
        # Fake data
        self._orders = [
            {"order_id": i, "user_id": i % 10, "status": random.choice(["pending","shipped","done"]),
             "amount": round(random.uniform(10, 500), 2)}
            for i in range(1, 10001)
        ]
        self._users = [
            {"user_id": i, "name": f"User{i}", "tier": random.choice(["free","pro","enterprise"])}
            for i in range(1, 101)
        ]

    def _simulate_latency(self) -> float:
        latency = random.uniform(self.latency_ms * 0.5, self.latency_ms * 1.5)
        time.sleep(latency / 1000)
        self.queries    += 1
        self.total_ms   += latency
        return latency

    def get_user_orders(self, user_id: int, status: str = None) -> List[Dict]:
        self._simulate_latency()
        orders = [o for o in self._orders if o["user_id"] == user_id]
        if status:
            orders = [o for o in orders if o["status"] == status]
        return orders

    def get_order_stats(self, status: str = None) -> Dict:
        self._simulate_latency()
        orders = self._orders if not status else [o for o in self._orders if o["status"] == status]
        if not orders:
            return {"count": 0, "total": 0.0, "avg": 0.0}
        amounts = [o["amount"] for o in orders]
        return {"count": len(orders), "total": round(sum(amounts), 2),
                "avg": round(sum(amounts) / len(amounts), 2)}

    def get_top_users_by_spend(self, limit: int = 10) -> List[Dict]:
        self._simulate_latency()
        spend: Dict[int, float] = defaultdict(float)
        for o in self._orders:
            spend[o["user_id"]] += o["amount"]
        top = sorted(spend.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [{"user_id": uid, "total_spend": round(amt, 2)} for uid, amt in top]

    @property
    def avg_query_ms(self) -> float:
        return self.total_ms / self.queries if self.queries else 0.0


# ─────────────────────────────────────────────
# QUERY CACHE LAYER
# ─────────────────────────────────────────────

class QueryCache:
    """Application-level query result cache with table-tag invalidation."""

    def __init__(self, default_ttl_s: float = 300.0):
        self._store     : Dict[str, Any]          = {}
        self._expires   : Dict[str, float]        = {}
        self._table_tags: Dict[str, List[str]]    = defaultdict(list)  # table → [keys]
        self.default_ttl= default_ttl_s
        self.hits       = 0
        self.misses     = 0
        self.sets       = 0
        self.invalidations = 0

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None or time.time() > self._expires.get(key, 0):
            if key in self._store:
                del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def set(self, key: str, value: Any, ttl_s: float = None,
            table_tags: List[str] = None):
        ttl = ttl_s or self.default_ttl
        self._store[key]   = value
        self._expires[key] = time.time() + ttl
        for tag in (table_tags or []):
            self._table_tags[tag].append(key)
        self.sets += 1

    def invalidate_table(self, table_name: str) -> int:
        """Invalidate all cached queries that touched this table."""
        tag = QueryCacheKey.table_tag(table_name)
        keys = self._table_tags.pop(tag, [])
        count = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                self._expires.pop(key, None)
                count += 1
        self.invalidations += count
        return count

    def delete(self, key: str):
        self._store.pop(key, None)
        self._expires.pop(key, None)

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def size(self) -> int:
        return len(self._store)


# ─────────────────────────────────────────────
# CACHED REPOSITORY (get_or_set pattern)
# ─────────────────────────────────────────────

class OrderRepository:
    """Repository with transparent query caching (get-or-set pattern)."""

    def __init__(self, db: Database, cache: QueryCache):
        self.db    = db
        self.cache = cache

    def get_user_orders(self, user_id: int, status: str = None,
                        ttl_s: float = 60.0) -> List[Dict]:
        key = QueryCacheKey.build("user_orders", {"user_id": user_id, "status": str(status)})
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self.db.get_user_orders(user_id, status)
        self.cache.set(key, result, ttl_s=ttl_s,
                       table_tags=[QueryCacheKey.table_tag("orders")])
        return result

    def get_order_stats(self, status: str = None, ttl_s: float = 120.0) -> Dict:
        key = QueryCacheKey.build("order_stats", {"status": str(status)})
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self.db.get_order_stats(status)
        self.cache.set(key, result, ttl_s=ttl_s,
                       table_tags=[QueryCacheKey.table_tag("orders")])
        return result

    def get_top_users(self, limit: int = 10, ttl_s: float = 300.0) -> List[Dict]:
        key = QueryCacheKey.build("top_users_by_spend", {"limit": limit})
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self.db.get_top_users_by_spend(limit)
        self.cache.set(key, result, ttl_s=ttl_s,
                       table_tags=[QueryCacheKey.table_tag("orders"),
                                   QueryCacheKey.table_tag("users")])
        return result

    def on_order_write(self):
        """Called after any order INSERT/UPDATE/DELETE — invalidates order caches."""
        count = self.cache.invalidate_table(QueryCacheKey.table_tag("orders"))
        return count


# ─────────────────────────────────────────────
# MATERIALIZED VIEW SIMULATOR
# ─────────────────────────────────────────────

class MaterializedView:
    """
    Simulates a PostgreSQL MATERIALIZED VIEW.
    Pre-computes expensive aggregation; refreshed on schedule or demand.
    """

    def __init__(self, db: Database, refresh_interval_s: float = 300.0):
        self.db             = db
        self.refresh_interval = refresh_interval_s
        self._data          : Optional[Dict] = None
        self._last_refresh  : float = 0.0
        self.refresh_count  = 0
        self.reads          = 0

    def refresh(self):
        """REFRESH MATERIALIZED VIEW — runs the underlying query."""
        self._data = {
            "stats"       : self.db.get_order_stats(),
            "top_users"   : self.db.get_top_users_by_spend(10),
            "refreshed_at": time.time(),
        }
        self._last_refresh = time.time()
        self.refresh_count += 1

    def query(self) -> Optional[Dict]:
        """Read from materialized view (always fast — pre-computed)."""
        self.reads += 1
        if self._data is None:
            self.refresh()
        age = time.time() - self._last_refresh
        if age > self.refresh_interval:
            self.refresh()   # stale — auto-refresh
        return self._data

    @property
    def age_s(self) -> float:
        return time.time() - self._last_refresh if self._last_refresh else float("inf")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_db_query_cache():
    print("=" * 65)
    print("DATABASE QUERY CACHE")
    print("=" * 65)

    random.seed(42)

    db    = Database(latency_ms=40.0)
    cache = QueryCache(default_ttl_s=120.0)
    repo  = OrderRepository(db, cache)

    # ── Cold Cache Baseline ────────────────────
    print("\n[1] COLD CACHE — FIRST REQUESTS (ALL DB HITS)")
    print("─" * 55)

    start = time.perf_counter()
    stats = repo.get_order_stats()
    top   = repo.get_top_users(limit=5)
    for uid in range(1, 6):
        repo.get_user_orders(uid)
    cold_ms = (time.perf_counter() - start) * 1000

    print(f"  Order stats: count={stats['count']} total=${stats['total']:,.2f}")
    print(f"  Top user by spend: user_id={top[0]['user_id']} ${top[0]['total_spend']:,.2f}")
    print(f"  Cold cache requests (7 queries): {cold_ms:.0f}ms  DB queries={db.queries}")
    print(f"  Cache size: {cache.size()}")

    # ── Warm Cache ────────────────────────────
    print("\n\n[2] WARM CACHE — REPEATED REQUESTS (ALL CACHE HITS)")
    print("─" * 55)
    db_queries_before = db.queries

    start = time.perf_counter()
    for _ in range(100):
        repo.get_order_stats()
        repo.get_top_users(limit=5)
        repo.get_user_orders(random.randint(1, 5))
    warm_ms = (time.perf_counter() - start) * 1000

    db_queries_after = db.queries
    new_db_queries   = db_queries_after - db_queries_before
    print(f"  100 repetitions (300 logical queries): {warm_ms:.1f}ms")
    print(f"  New DB queries: {new_db_queries}  Cache hits: {cache.hits}")
    print(f"  Hit ratio: {cache.hit_ratio:.1%}")
    print(f"  Speedup: ~{cold_ms / max(warm_ms / 100, 0.001):.0f}x per-request")

    # ── Cache Key Examples ─────────────────────
    print("\n\n[3] CACHE KEY DESIGN")
    print("─" * 55)
    key_examples = [
        ("user_orders", {"user_id": 42, "status": "pending"},   "tenant123"),
        ("user_orders", {"user_id": 42, "status": "shipped"},   "tenant123"),
        ("order_stats", {"status": "None"},                     None),
        ("top_users_by_spend", {"limit": 10},                   None),
    ]
    for qname, params, tid in key_examples:
        key = QueryCacheKey.build(qname, params, tenant_id=tid)
        print(f"  {qname}({params})  tenant={tid}")
        print(f"    → {key}")

    # ── Table-Level Invalidation ───────────────
    print("\n\n[4] TABLE-LEVEL INVALIDATION ON WRITE")
    print("─" * 55)
    print(f"  Cache size before write: {cache.size()}")
    print(f"  Simulating ORDER INSERT (new order placed)...")
    invalidated = repo.on_order_write()
    print(f"  Invalidated {invalidated} cached queries (all 'orders' table queries)")
    print(f"  Cache size after write: {cache.size()}")

    # Next read must go to DB
    db_before = db.queries
    repo.get_order_stats()
    print(f"  Next get_order_stats: DB queries +{db.queries - db_before} (cache miss → DB)")

    # ── Materialized View ─────────────────────
    print("\n\n[5] MATERIALIZED VIEW (pre-computed aggregations)")
    print("─" * 55)
    mat_view = MaterializedView(db, refresh_interval_s=300.0)

    db_before = db.queries
    start     = time.perf_counter()
    data      = mat_view.query()
    first_ms  = (time.perf_counter() - start) * 1000
    print(f"  First query (refresh): {first_ms:.0f}ms  DB queries: {db.queries - db_before}")
    print(f"  Stats: count={data['stats']['count']}  top_user={data['top_users'][0]['user_id']}")

    db_before = db.queries
    start     = time.perf_counter()
    for _ in range(1000):
        mat_view.query()   # all reads — no DB queries
    fast_ms = (time.perf_counter() - start) * 1000

    print(f"\n  1000 reads after refresh: {fast_ms:.1f}ms  DB queries: {db.queries - db_before}")
    print(f"  avg: {fast_ms/1000:.3f}ms per read  (pure in-memory — no DB)")
    print(f"  Refresh count: {mat_view.refresh_count}  reads: {mat_view.reads}")

    # ── Anti-patterns ─────────────────────────
    print("\n\n[6] QUERY CACHE ANTI-PATTERNS")
    print("─" * 55)
    antipatterns = [
        ("Caching write-heavy queries",
         "orders table written 1000x/s → constant invalidation → 0% hit ratio"),
        ("No TTL set",
         "Cache fills with stale data → never fresh → correctness issues"),
        ("Per-user keys without tenant scoping",
         "key='stats' for all tenants → cross-tenant data leak"),
        ("Caching large result sets",
         "Full table scans cached → evicts small hot keys → degraded performance"),
        ("Including timestamp in key",
         "key='orders:2024-01-15-14:30:00' → unique per second → 0% hit ratio"),
        ("Caching user-specific queries globally",
         "user:1 and user:2 have different results — wrong cache collision"),
    ]
    for pattern, consequence in antipatterns:
        print(f"\n  ❌ {pattern}")
        print(f"     {consequence}")

    # ── Summary ───────────────────────────────
    print("\n\n[7] QUERY CACHE DECISION GUIDE")
    print("─" * 55)
    guide = [
        ("Aggregations (COUNT, SUM, AVG)", "YES", "Expensive, rarely change — high TTL (5-30 min)"),
        ("Leaderboards / rankings",        "YES", "Expensive sort — cache 1-5 min with jitter"),
        ("User-specific lists (orders)",   "YES", "Short TTL (30-60s), invalidate on write"),
        ("Real-time counters",             "NO",  "Use Redis INCR directly, not DB query cache"),
        ("User auth/session",              "NO",  "Use dedicated session store (Redis)"),
        ("High-write OLTP queries",        "NO",  "Constant invalidation = 0% hit ratio"),
        ("Report queries (dashboard)",     "YES", "Materialized view or cache 5-60 min"),
    ]
    print(f"  {'Query Type':<34} {'Cache?':<8} {'Strategy'}")
    print(f"  {'─'*80}")
    for qtype, cache_yn, strategy in guide:
        print(f"  {qtype:<34} {cache_yn:<8} {strategy}")


if __name__ == "__main__":
    demonstrate_db_query_cache()
