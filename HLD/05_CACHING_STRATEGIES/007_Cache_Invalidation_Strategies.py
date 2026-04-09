"""
CACHE INVALIDATION STRATEGIES
================================

"There are only two hard things in Computer Science: cache invalidation
and naming things." — Phil Karlton

Problem Statement:
When the database changes, cached values become stale. How do you know
when to remove or update cache entries? Too aggressive: poor hit ratio.
Too lenient: users see stale data. The strategy depends on acceptable
staleness and data change frequency.

Strategies:

  1. TTL-based (Time-To-Live):
     Every entry has an expiry. Stale data served up to TTL window.
     Simple but imprecise — entry may be valid or stale at expiry.
     Use: product prices, public content, aggregated stats.

  2. Event-driven (Active Invalidation):
     On every DB write, publish event → cache layer deletes key.
     Near-realtime consistency. Requires pub/sub infrastructure.
     Use: user profiles, inventory counts, any write-then-read data.

  3. Write-through invalidation:
     On write: update cache with new value (not just delete).
     Avoids next-request miss. Risk: race condition.
     Use: high-read, low-write keys (homepage, popular products).

  4. Cache versioning:
     Cache key includes a version: "user:123:v7"
     Increment version on write → old key becomes orphaned (auto-expires).
     Avoids delete races. Old clients still see old version.
     Use: microservice fan-out where multiple services cache same entity.

  5. Tag-based invalidation:
     Associate cache entries with tags (e.g., tag="product_catalog").
     On catalog change: invalidate all entries with that tag.
     Use: page fragments that depend on multiple DB entities.

  6. Manual (admin-triggered):
     Cache cleared by ops team on deploy or data fix.
     Last resort, not for normal operations.

Stale-While-Revalidate:
  Serve stale entry immediately; async refresh in background.
  Hides revalidation latency from user. May serve one stale response.
  Used in CDN (Cache-Control: stale-while-revalidate=60).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
import time
import threading
import random
from collections import defaultdict


# ─────────────────────────────────────────────
# TTL-BASED INVALIDATION
# ─────────────────────────────────────────────

class TTLCache:
    """
    Every entry has a TTL. Expired entries are lazy-evicted on access.
    Background sweeper cleans up expired entries periodically.
    """

    def __init__(self, default_ttl_s: float = 60.0):
        self._store      : Dict[str, Dict] = {}
        self.default_ttl = default_ttl_s
        self.hits        = 0
        self.misses      = 0
        self.expirations = 0

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if not entry:
            self.misses += 1
            return None
        if time.time() > entry["expires_at"]:
            del self._store[key]
            self.expirations += 1
            self.misses += 1
            return None
        self.hits += 1
        return entry["value"]

    def set(self, key: str, value: Any, ttl_s: float = None):
        ttl = ttl_s or self.default_ttl
        self._store[key] = {"value": value, "expires_at": time.time() + ttl}

    def delete(self, key: str):
        self._store.pop(key, None)

    def sweep_expired(self) -> int:
        now     = time.time()
        expired = [k for k, v in self._store.items() if now > v["expires_at"]]
        for k in expired:
            del self._store[k]
            self.expirations += 1
        return len(expired)

    def time_until_stale(self, key: str) -> Optional[float]:
        entry = self._store.get(key)
        if not entry:
            return None
        return max(0.0, entry["expires_at"] - time.time())

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ─────────────────────────────────────────────
# EVENT-DRIVEN INVALIDATION
# ─────────────────────────────────────────────

@dataclass
class InvalidationEvent:
    entity_type : str   # "user", "product", "order"
    entity_id   : str
    operation   : str   # "update", "delete"
    timestamp   : float = field(default_factory=time.time)

    @property
    def cache_key(self) -> str:
        return f"{self.entity_type}:{self.entity_id}"


class EventDrivenInvalidationCache:
    """
    Cache that receives invalidation events from a pub/sub channel.
    On write to DB, publisher sends event → this cache deletes entry.
    """

    def __init__(self, ttl_s: float = 300.0):
        self._store  : Dict[str, Any] = {}
        self._ttls   : Dict[str, float] = {}
        self.ttl_s   = ttl_s
        self.hits    = 0
        self.misses  = 0
        self.invalidations = 0
        self._log    : List[str] = []

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            self.misses += 1
            return None
        if time.time() > self._ttls.get(key, 0):
            del self._store[key]
            self._ttls.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        return self._store[key]

    def set(self, key: str, value: Any):
        self._store[key] = value
        self._ttls[key]  = time.time() + self.ttl_s

    def on_invalidation_event(self, event: InvalidationEvent):
        """Called when pub/sub delivers an invalidation message."""
        key = event.cache_key
        if key in self._store:
            del self._store[key]
            self._ttls.pop(key, None)
            self.invalidations += 1
            self._log.append(f"Invalidated {key} (op={event.operation})")

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ─────────────────────────────────────────────
# VERSIONED CACHE KEYS
# ─────────────────────────────────────────────

class VersionedCache:
    """
    Cache key includes entity version: "user:123:v5"
    On write: increment version in a version store.
    Old cache keys become orphaned (expire via TTL).
    Avoids delete race condition entirely.
    """

    def __init__(self, ttl_s: float = 300.0):
        self._store    : Dict[str, Any]  = {}
        self._expires  : Dict[str, float]= {}
        self._versions : Dict[str, int]  = defaultdict(int)   # entity_key → version
        self.ttl_s     = ttl_s
        self.hits      = 0
        self.misses    = 0
        self.orphan_keys = 0   # old versioned keys still in store

    def _versioned_key(self, entity_type: str, entity_id: str) -> str:
        version = self._versions[f"{entity_type}:{entity_id}"]
        return f"{entity_type}:{entity_id}:v{version}"

    def get(self, entity_type: str, entity_id: str) -> Optional[Any]:
        key = self._versioned_key(entity_type, entity_id)
        if key not in self._store or time.time() > self._expires.get(key, 0):
            self.misses += 1
            return None
        self.hits += 1
        return self._store[key]

    def set(self, entity_type: str, entity_id: str, value: Any):
        key = self._versioned_key(entity_type, entity_id)
        self._store[key]   = value
        self._expires[key] = time.time() + self.ttl_s

    def invalidate(self, entity_type: str, entity_id: str):
        """Increment version — old key becomes orphaned, expires naturally."""
        old_key = self._versioned_key(entity_type, entity_id)
        self._versions[f"{entity_type}:{entity_id}"] += 1
        # Old key still in store but unreachable by new key — orphaned
        if old_key in self._store:
            self.orphan_keys += 1

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ─────────────────────────────────────────────
# TAG-BASED INVALIDATION
# ─────────────────────────────────────────────

class TagBasedCache:
    """
    Entries are tagged with entity types.
    When an entity changes, all tagged entries are invalidated.
    Useful for page fragment caches that depend on multiple entities.
    """

    def __init__(self, ttl_s: float = 300.0):
        self._store    : Dict[str, Any]       = {}
        self._expires  : Dict[str, float]     = {}
        self._tags     : Dict[str, Set[str]]  = defaultdict(set)  # key → tags
        self._tag_index: Dict[str, Set[str]]  = defaultdict(set)  # tag → keys
        self.ttl_s     = ttl_s
        self.hits      = 0
        self.misses    = 0
        self.tag_invalidations = 0

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store or time.time() > self._expires.get(key, 0):
            self.misses += 1
            return None
        self.hits += 1
        return self._store[key]

    def set(self, key: str, value: Any, tags: List[str] = None):
        self._store[key]   = value
        self._expires[key] = time.time() + self.ttl_s
        for tag in (tags or []):
            self._tags[key].add(tag)
            self._tag_index[tag].add(key)

    def invalidate_tag(self, tag: str) -> int:
        keys = list(self._tag_index.get(tag, set()))
        for key in keys:
            self._store.pop(key, None)
            self._expires.pop(key, None)
            self._tags.pop(key, None)
            self._tag_index[tag].discard(key)
        self.tag_invalidations += len(keys)
        return len(keys)

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ─────────────────────────────────────────────
# STALE-WHILE-REVALIDATE
# ─────────────────────────────────────────────

class StaleWhileRevalidateCache:
    """
    Serves stale entry immediately; asynchronously refreshes in background.
    Eliminates revalidation latency at cost of one stale response.
    Used in CDN (Cache-Control: stale-while-revalidate=60).
    """

    def __init__(self, ttl_s: float = 60.0, stale_window_s: float = 30.0):
        self._store      : Dict[str, Dict] = {}
        self.ttl_s       = ttl_s
        self.stale_window= stale_window_s
        self.hits_fresh  = 0
        self.hits_stale  = 0
        self.misses      = 0
        self.revalidations = 0

    def get(self, key: str, loader: Callable) -> Optional[Any]:
        entry = self._store.get(key)
        now   = time.time()

        if entry is None:
            # Full miss — must load synchronously
            self.misses += 1
            val = loader(key)
            self._store[key] = {"value": val, "fetched_at": now}
            return val

        age = now - entry["fetched_at"]

        if age < self.ttl_s:
            # Fresh — serve immediately
            self.hits_fresh += 1
            return entry["value"]

        if age < self.ttl_s + self.stale_window:
            # Stale but within window — serve stale, refresh async
            self.hits_stale += 1
            self._async_revalidate(key, loader)
            return entry["value"]

        # Fully expired — synchronous reload
        self.misses += 1
        val = loader(key)
        self._store[key] = {"value": val, "fetched_at": now}
        return val

    def _async_revalidate(self, key: str, loader: Callable):
        """Background refresh — doesn't block the caller."""
        def refresh():
            val = loader(key)
            self._store[key] = {"value": val, "fetched_at": time.time()}
            self.revalidations += 1
        t = threading.Thread(target=refresh, daemon=True)
        t.start()


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_cache_invalidation():
    print("=" * 65)
    print("CACHE INVALIDATION STRATEGIES")
    print("=" * 65)

    random.seed(42)

    # ── TTL-Based ─────────────────────────────
    print("\n[1] TTL-BASED INVALIDATION")
    print("─" * 55)
    ttl_cache = TTLCache(default_ttl_s=0.2)   # 200ms TTL for demo

    ttl_cache.set("user:1", {"name": "Alice"})
    ttl_cache.set("product:1", {"price": 99.9}, ttl_s=0.1)

    print(f"  SET user:1 (TTL=200ms)  product:1 (TTL=100ms)")
    val = ttl_cache.get("user:1")
    print(f"  GET user:1 immediately: {val is not None} (HIT)")

    time.sleep(0.15)
    val1 = ttl_cache.get("product:1")
    val2 = ttl_cache.get("user:1")
    print(f"  After 150ms: product:1={val1 is not None} (MISS-expired)  "
          f"user:1={val2 is not None} (HIT-still valid)")

    time.sleep(0.1)
    val = ttl_cache.get("user:1")
    print(f"  After 250ms: user:1={val is not None} (MISS-expired)")
    print(f"  Expirations: {ttl_cache.expirations}  hit_ratio: {ttl_cache.hit_ratio:.1%}")

    # ── Event-Driven ──────────────────────────
    print("\n\n[2] EVENT-DRIVEN (ACTIVE) INVALIDATION")
    print("─" * 55)
    ev_cache = EventDrivenInvalidationCache(ttl_s=300.0)

    # Populate cache
    for i in range(5):
        ev_cache.set(f"user:{i}", {"name": f"User{i}", "version": 1})
    print(f"  Cached 5 users  cache_size={len(ev_cache._store)}")

    # DB update → publish invalidation event
    events = [
        InvalidationEvent("user", "1", "update"),
        InvalidationEvent("user", "3", "delete"),
    ]
    for event in events:
        ev_cache.on_invalidation_event(event)
        print(f"  Event: {event.entity_type}:{event.entity_id} {event.operation} → cache key invalidated")

    # Verify
    still_valid = sum(1 for i in range(5) if ev_cache.get(f"user:{i}") is not None)
    print(f"  Valid cache entries after events: {still_valid}/5  "
          f"(invalidations={ev_cache.invalidations})")
    print(f"  user:1 in cache: {ev_cache.get('user:1') is not None}")
    print(f"  user:0 in cache: {ev_cache.get('user:0') is not None}")

    # ── Versioned Keys ────────────────────────
    print("\n\n[3] VERSIONED CACHE KEYS")
    print("─" * 55)
    ver_cache = VersionedCache(ttl_s=300.0)

    ver_cache.set("user", "1", {"name": "Alice", "email": "alice@old.com"})
    val = ver_cache.get("user", "1")
    print(f"  SET user:1:v0  GET returns: name={val['name']}")

    ver_cache.invalidate("user", "1")   # bumps version to v1
    val = ver_cache.get("user", "1")   # looks for user:1:v1 — miss
    print(f"  INVALIDATE user:1 → version bumped to v1")
    print(f"  GET user:1:v1: {val} (MISS — not yet written at v1)")

    ver_cache.set("user", "1", {"name": "Alice", "email": "alice@new.com"})
    val = ver_cache.get("user", "1")
    print(f"  SET user:1:v1 → GET returns: email={val['email']} (fresh)")
    print(f"  Orphaned keys (old versions still in store): {ver_cache.orphan_keys}")
    print(f"  hit_ratio: {ver_cache.hit_ratio:.1%}")

    # ── Tag-Based ─────────────────────────────
    print("\n\n[4] TAG-BASED INVALIDATION")
    print("─" * 55)
    tag_cache = TagBasedCache(ttl_s=300.0)

    # Product listing page depends on product catalog + category tags
    tag_cache.set("page:products:electronics", "<html>...</html>",
                   tags=["product_catalog", "category:electronics"])
    tag_cache.set("page:products:laptops", "<html>...</html>",
                   tags=["product_catalog", "category:laptops", "category:electronics"])
    tag_cache.set("page:homepage", "<html>...</html>",
                   tags=["homepage", "featured_products"])
    tag_cache.set("page:product:laptop-1", "<html>...</html>",
                   tags=["product:laptop-1", "product_catalog"])

    print(f"  Cached 4 page fragments with tags")
    print(f"  Tags in use: product_catalog, category:electronics, category:laptops, homepage, featured_products")

    # Laptop catalog updated — invalidate all pages tagged with "product_catalog"
    invalidated = tag_cache.invalidate_tag("product_catalog")
    print(f"\n  Laptop catalog updated → invalidate tag 'product_catalog'")
    print(f"  Invalidated {invalidated} page fragments")
    print(f"  Homepage still cached: {tag_cache.get('page:homepage') is not None}")
    print(f"  Electronics page removed: {tag_cache.get('page:products:electronics') is None}")

    # ── Stale-While-Revalidate ─────────────────
    print("\n\n[5] STALE-WHILE-REVALIDATE")
    print("─" * 55)
    swr_cache = StaleWhileRevalidateCache(ttl_s=0.1, stale_window_s=0.1)
    load_count = {"n": 0}

    def loader(key: str) -> Dict:
        load_count["n"] += 1
        time.sleep(0.02)   # 20ms load time
        return {"key": key, "version": load_count["n"], "loaded_at": time.time()}

    # First access — synchronous load
    val = swr_cache.get("hot-page", loader)
    print(f"  1st access (cold): version={val['version']} (synchronous load)")

    # Within TTL — fresh hit
    val = swr_cache.get("hot-page", loader)
    print(f"  2nd access (fresh): version={val['version']} hits_fresh={swr_cache.hits_fresh}")

    time.sleep(0.12)   # exceed TTL, enter stale window
    val = swr_cache.get("hot-page", loader)
    print(f"  3rd access (stale window): version={val['version']} (stale served, async refresh)")
    time.sleep(0.05)   # wait for async refresh
    print(f"  Background revalidations: {swr_cache.revalidations}  loads={load_count['n']}")

    # ── Comparison ────────────────────────────
    print("\n\n[6] INVALIDATION STRATEGY COMPARISON")
    print("─" * 55)
    strategies = [
        ("TTL-based",         "Simple", "Stale up to TTL window",    "Any data, staleness ok"),
        ("Event-driven",      "Medium", "Near-realtime",             "Profiles, inventory"),
        ("Write-through",     "Medium", "Immediate (race risk)",     "Hot read-after-write"),
        ("Versioned keys",    "Medium", "Immediate (no race)",       "Microservices fan-out"),
        ("Tag-based",         "Complex","Immediate for tag group",   "Page fragment caches"),
        ("Stale-while-reval","Medium", "One stale response",        "CDN, public content"),
    ]
    print(f"  {'Strategy':<22} {'Complexity':<10} {'Staleness':<22} {'Best Use Case'}")
    print(f"  {'─'*75}")
    for strategy, complexity, staleness, use_case in strategies:
        print(f"  {strategy:<22} {complexity:<10} {staleness:<22} {use_case}")


if __name__ == "__main__":
    demonstrate_cache_invalidation()
