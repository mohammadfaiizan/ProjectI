"""
CONTENT DELIVERY NETWORK (CDN) DESIGN
=======================================

Problem Statement:
Serving static and dynamic content from a single origin server is slow for
global users and costly for bandwidth. CDNs solve this by caching content at
edge nodes distributed worldwide, reducing latency and origin load.

Architecture:
  User (US) ──→ Edge-US ──(cache HIT)──→ User
                │ (cache MISS)
                └──→ Origin Server ──→ Edge-US ──→ User

CDN Types:
  Pull CDN: Edge fetches content from origin on first miss, then caches
  Push CDN: Publisher proactively pushes content to all edges (for known-popular content)

Key CDN Benefits:
  - Lower latency: edge closer to user
  - Reduce origin load: 80-95% of requests served from cache
  - DDoS protection: absorbs volumetric attacks at the edge
  - Bandwidth cost: CDN bandwidth cheaper than origin egress

Cache Control:
  Cache-Control: max-age=86400     → cache for 24 hours
  Cache-Control: no-cache          → always revalidate with origin
  Cache-Control: s-maxage=3600     → CDN caches for 1 hour (even if browser doesn't)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import random
import hashlib


class CDNStrategy(Enum):
    PULL = "pull"   # Lazy: fetch on first miss
    PUSH = "push"   # Eager: pre-populate all edges


@dataclass
class CachedObject:
    key        : str
    content    : bytes
    origin_url : str
    ttl_s      : int
    cached_at  : float = field(default_factory=time.time)
    hit_count  : int   = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.cached_at > self.ttl_s

    @property
    def size_kb(self) -> float:
        return len(self.content) / 1024


@dataclass
class EdgeNode:
    node_id       : str
    region        : str
    cache_size_mb : int = 10_240   # 10 GB
    latency_to_user_ms: float = 5.0

    def __post_init__(self):
        self._cache     : Dict[str, CachedObject] = {}
        self._cache_used = 0.0   # MB
        self.cache_hits  = 0
        self.cache_misses= 0
        self.origin_fetches = 0

    @property
    def hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def get(self, key: str) -> Optional[CachedObject]:
        obj = self._cache.get(key)
        if obj and not obj.is_expired:
            obj.hit_count += 1
            self.cache_hits += 1
            return obj
        if obj and obj.is_expired:
            self._cache_used -= obj.size_kb / 1024
            del self._cache[key]
        self.cache_misses += 1
        return None

    def set(self, obj: CachedObject):
        self._cache[obj.key] = obj
        self._cache_used += obj.size_kb / 1024

    def stats(self):
        print(f"  Edge [{self.region}/{self.node_id}]: "
              f"hits={self.cache_hits}  misses={self.cache_misses}  "
              f"hit_ratio={self.hit_ratio:.1%}  "
              f"cached={len(self._cache)} objects")


class OriginServer:
    """Serves content and tracks how often edges need to fetch from it."""

    def __init__(self, url: str):
        self.url         = url
        self.fetch_count = 0
        self._content    : Dict[str, tuple] = {}   # key → (bytes, ttl, content_type)

    def add_content(self, path: str, content: str, ttl_s: int = 3600,
                    content_type: str = "text/html"):
        self._content[path] = (content.encode(), ttl_s, content_type)

    def fetch(self, path: str) -> Optional[CachedObject]:
        self.fetch_count += 1
        entry = self._content.get(path)
        if not entry:
            return None
        content_bytes, ttl, _ = entry
        return CachedObject(
            key=path, content=content_bytes,
            origin_url=f"{self.url}{path}", ttl_s=ttl
        )


class CDNRouter:
    """Routes client requests to the nearest edge node."""

    def __init__(self, edges: List[EdgeNode]):
        self.edges = edges

    def nearest(self, client_region: str) -> EdgeNode:
        """Return edge node for the given client region."""
        for edge in self.edges:
            if edge.region == client_region:
                return edge
        return self.edges[0]   # fallback


class CacheInvalidator:
    """Invalidates cached objects across all edge nodes."""

    def __init__(self, edges: List[EdgeNode]):
        self.edges = edges

    def invalidate(self, key: str):
        count = 0
        for edge in self.edges:
            if key in edge._cache:
                del edge._cache[key]
                count += 1
        print(f"  🗑  Invalidated '{key}' from {count} edge nodes")

    def purge_all(self, pattern: str):
        """Purge all keys matching a prefix pattern."""
        total = 0
        for edge in self.edges:
            keys = [k for k in list(edge._cache.keys()) if k.startswith(pattern)]
            for k in keys:
                del edge._cache[k]
                total += 1
        print(f"  🗑  Purged {total} objects matching '{pattern}*' across all edges")


class CDNMetrics:
    """Aggregated metrics across all edges."""

    def __init__(self, origin: OriginServer, edges: List[EdgeNode]):
        self.origin = origin
        self.edges  = edges

    def total_requests(self) -> int:
        return sum(e.cache_hits + e.cache_misses for e in self.edges)

    def total_hits(self) -> int:
        return sum(e.cache_hits for e in self.edges)

    def overall_hit_ratio(self) -> float:
        t = self.total_requests()
        return self.total_hits() / t if t else 0.0

    def origin_offload_pct(self) -> float:
        return (1 - self.origin.fetch_count / max(1, self.total_requests())) * 100

    def report(self):
        print(f"\n  CDN METRICS:")
        print(f"    Total requests     : {self.total_requests()}")
        print(f"    Cache hits         : {self.total_hits()}")
        print(f"    Overall hit ratio  : {self.overall_hit_ratio():.1%}")
        print(f"    Origin fetches     : {self.origin.fetch_count}")
        print(f"    Origin offload     : {self.origin_offload_pct():.1f}%")
        for edge in self.edges:
            edge.stats()


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_cdn_design():
    print("=" * 65)
    print("CONTENT DELIVERY NETWORK (CDN) DESIGN")
    print("=" * 65)

    # ── Set up Origin ─────────────────────────
    origin = OriginServer("https://origin.example.com")
    origin.add_content("/index.html",      "<html>Homepage</html>",     ttl_s=3600)
    origin.add_content("/style.css",       "body { font-size: 16px; }", ttl_s=86400)
    origin.add_content("/logo.png",        "PNG_BYTES_HERE" * 100,      ttl_s=604800)  # 7 days
    origin.add_content("/api/products",    '{"products": [...]}',        ttl_s=60)      # 1 min
    origin.add_content("/api/user/profile",'{"user": {...}}',            ttl_s=0)       # no cache

    # ── Set up Edge Nodes ─────────────────────
    edges = [
        EdgeNode("edge-us-1", "us",   cache_size_mb=51200, latency_to_user_ms=5),
        EdgeNode("edge-eu-1", "eu",   cache_size_mb=51200, latency_to_user_ms=8),
        EdgeNode("edge-ap-1", "asia", cache_size_mb=51200, latency_to_user_ms=12),
    ]

    router    = CDNRouter(edges)
    invalidator = CacheInvalidator(edges)
    metrics   = CDNMetrics(origin, edges)

    # ── Pull CDN: Serve requests ──────────────
    print("\n[1] PULL CDN SIMULATION (1000 requests from 3 regions)")
    print("─" * 55)

    # Simulate request distribution: 60% US, 30% EU, 10% Asia
    client_regions = (["us"] * 60 + ["eu"] * 30 + ["asia"] * 10)
    paths = ["/index.html", "/style.css", "/logo.png", "/api/products",
             "/index.html", "/style.css", "/index.html", "/style.css",
             "/logo.png", "/api/user/profile"]

    random.seed(42)
    for i in range(100):
        region = random.choice(client_regions)
        path   = random.choice(paths)
        edge   = router.nearest(region)

        cached = edge.get(path)
        if cached:
            pass   # served from cache, no origin fetch
        else:
            obj = origin.fetch(path)
            if obj and obj.ttl_s > 0:
                edge.set(obj)
                edge.origin_fetches += 1

    metrics.report()

    # ── Push CDN: Pre-populate ────────────────
    print("\n\n[2] PUSH CDN — PRE-WARM ALL EDGES")
    print("─" * 55)
    popular_paths = ["/index.html", "/style.css", "/logo.png"]
    for path in popular_paths:
        obj = origin.fetch(path)
        if obj:
            for edge in edges:
                edge.set(obj)
            print(f"  📤 Pushed '{path}' to all {len(edges)} edges")

    print(f"\n  First request for /logo.png (all edges already warm):")
    for edge in edges:
        cached = edge.get("/logo.png")
        print(f"  {edge.region}: {'✅ cache HIT' if cached else '❌ miss'}")

    # ── Cache Invalidation ────────────────────
    print("\n\n[3] CACHE INVALIDATION")
    print("─" * 55)
    invalidator.invalidate("/index.html")
    invalidator.purge_all("/api/")

    # ── Cache-Control Guide ───────────────────
    print("\n\n[4] CACHE-CONTROL HEADER GUIDE")
    print("─" * 55)
    cache_headers = [
        ("/static/app.js",       "public, max-age=31536000, immutable",
         "1 year — hashed filename, never changes"),
        ("/style.css",           "public, max-age=86400",
         "24 hours — stable static asset"),
        ("/index.html",          "public, max-age=3600",
         "1 hour — layout may update occasionally"),
        ("/api/products",        "public, s-maxage=60, max-age=0",
         "CDN caches 60s; browser doesn't cache"),
        ("/api/user/profile",    "private, no-store",
         "Personal data — never cache anywhere"),
        ("/api/auth/token",      "no-cache, no-store",
         "Auth tokens must never be cached"),
    ]
    print(f"  {'Path':<30} {'Cache-Control':<40} {'Reason'}")
    print(f"  {'─'*100}")
    for path, header, reason in cache_headers:
        print(f"  {path:<30} {header:<40} {reason}")

    # ── CDN Use Cases ─────────────────────────
    print("\n\n[5] WHAT TO PUT ON CDN")
    print("─" * 55)
    do_cdn = [
        "Static assets (JS, CSS, images, fonts)",
        "Video/audio files (HLS segments, MP4)",
        "API responses that are public + cacheable (product listings)",
        "HTML pages (with short TTL for dynamic sites)",
        "Software downloads, firmware files",
    ]
    dont_cdn = [
        "Private/user-specific data (profile, inbox)",
        "Real-time data (stock prices, live scores)",
        "POST/PUT/DELETE requests (mutations)",
        "Auth tokens, session data",
        "Server-side rendered dynamic HTML with user data",
    ]
    print("  ✅ Cache on CDN:")
    for item in do_cdn:
        print(f"    • {item}")
    print("\n  ❌ Do NOT cache on CDN:")
    for item in dont_cdn:
        print(f"    • {item}")


if __name__ == "__main__":
    demonstrate_cdn_design()
