"""
CDN AS CACHE (EDGE CACHING)
=============================

Problem Statement:
Your API server in us-east-1 responds in 5ms locally but 200ms for a user
in Tokyo (round-trip latency). For static and semi-static content, a CDN
caches responses at edge nodes globally, serving users from the nearest PoP
in <20ms regardless of origin location.

CDN Architecture:
  Origin Server (your app) → CDN Edge Nodes (PoPs worldwide) → End Users
  100+ edge nodes in major cities: NYC, London, Tokyo, Singapore, São Paulo.

  Pull Model (default):
    CDN fetches from origin on first miss. Cached at edge for TTL.
    No proactive push. Simple. Works for any content.

  Push Model:
    You upload content to CDN directly (S3 → CloudFront distribution).
    Good for large static assets deployed infrequently.

Cache-Control Headers:
  Cache-Control: public, max-age=86400, s-maxage=3600
    public    : CDN may cache this response
    max-age   : browser TTL (seconds)
    s-maxage  : CDN/shared cache TTL (overrides max-age for CDNs)

  Cache-Control: private, no-store
    private : CDN must not cache (user-specific data: session, cart)
    no-store: never cache anywhere

  Vary: Accept-Language, Accept-Encoding
    Different cached versions per header combination.
    CDN maintains separate cache entries per Vary dimension.

Surrogate-Key / Cache-Tag (Fastly / Cloudflare):
  Tag cached responses: Surrogate-Key: product-123 category-electronics
  On product update: purge all responses tagged product-123 instantly.
  Enables surgical cache invalidation without full purge.

Cache Hit Ratio at CDN:
  Target: > 95% for static assets, > 70% for cacheable API responses.
  Low ratio causes: high Vary cardinality, no-cache headers, short TTL.

CDN Caching Strategy by Content Type:
  HTML pages   : s-maxage=60, stale-while-revalidate=300
  JS/CSS/Images: s-maxage=31536000 (1yr), immutable (hash in filename)
  API responses: s-maxage=30 if public, private if user-specific
  Videos       : s-maxage=86400, streaming via byte-range caching
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import time
import random
import hashlib
from collections import defaultdict


# ─────────────────────────────────────────────
# CACHE-CONTROL HEADER PARSER
# ─────────────────────────────────────────────

@dataclass
class CacheDirectives:
    """Parsed Cache-Control header directives."""
    public           : bool = False
    private          : bool = False
    no_cache         : bool = False
    no_store         : bool = False
    max_age          : Optional[int] = None   # browser TTL
    s_maxage         : Optional[int] = None   # CDN TTL
    stale_while_reval: Optional[int] = None
    immutable        : bool = False

    @property
    def cdn_ttl_s(self) -> Optional[int]:
        if self.no_store or self.private or self.no_cache:
            return None   # CDN must not cache
        return self.s_maxage or self.max_age

    @property
    def is_cacheable_by_cdn(self) -> bool:
        return self.cdn_ttl_s is not None and self.cdn_ttl_s > 0

    @classmethod
    def parse(cls, header: str) -> "CacheDirectives":
        d = cls()
        for part in header.replace(" ", "").split(","):
            if part == "public":       d.public = True
            elif part == "private":    d.private = True
            elif part == "no-cache":   d.no_cache = True
            elif part == "no-store":   d.no_store = True
            elif part == "immutable":  d.immutable = True
            elif part.startswith("max-age="):
                d.max_age = int(part.split("=")[1])
            elif part.startswith("s-maxage="):
                d.s_maxage = int(part.split("=")[1])
            elif part.startswith("stale-while-revalidate="):
                d.stale_while_reval = int(part.split("=")[1])
        return d


# ─────────────────────────────────────────────
# CDN EDGE NODE
# ─────────────────────────────────────────────

@dataclass
class CachedResponse:
    body         : bytes
    status_code  : int
    headers      : Dict[str, str]
    cached_at    : float
    ttl_s        : int
    surrogate_keys: List[str] = field(default_factory=list)
    cache_key    : str = ""

    @property
    def is_expired(self) -> bool:
        return time.time() - self.cached_at > self.ttl_s

    @property
    def age_s(self) -> float:
        return time.time() - self.cached_at

    @property
    def ttl_remaining_s(self) -> float:
        return max(0.0, self.ttl_s - self.age_s)


class CDNEdgeNode:
    """
    Simulates a CDN Point of Presence (PoP) edge node.
    Caches origin responses keyed by URL + Vary headers.
    Supports surrogate-key based purging.
    """

    def __init__(self, name: str, location: str, origin_latency_ms: float):
        self.name             = name
        self.location         = location
        self.origin_latency   = origin_latency_ms
        self._cache           : Dict[str, CachedResponse] = {}
        self._surrogate_index : Dict[str, List[str]]      = defaultdict(list)
        self.hits             = 0
        self.misses           = 0
        self.origin_fetches   = 0
        self.purges           = 0

    def _make_cache_key(self, url: str, vary_headers: Dict[str, str]) -> str:
        vary_str = ",".join(f"{k}={v}" for k, v in sorted(vary_headers.items()))
        return hashlib.md5(f"{url}|{vary_str}".encode()).hexdigest()[:12]

    def request(self, url: str, vary_headers: Dict[str, str],
                origin_fetch_fn) -> Tuple[CachedResponse, str]:
        """
        Handle request: check cache, on miss fetch from origin.
        Returns (response, cache_status): cache_status is HIT or MISS.
        """
        key   = self._make_cache_key(url, vary_headers)
        entry = self._cache.get(key)

        if entry and not entry.is_expired:
            self.hits += 1
            return entry, "HIT"

        # Cache miss — fetch from origin
        self.misses       += 1
        self.origin_fetches += 1
        response = origin_fetch_fn(url)
        response.cache_key = key

        # Determine if cacheable
        cc = CacheDirectives.parse(response.headers.get("Cache-Control", ""))
        if cc.is_cacheable_by_cdn:
            self._cache[key] = response
            for sk in response.surrogate_keys:
                self._surrogate_index[sk].append(key)

        return response, "MISS"

    def purge_by_url(self, url: str, vary_headers: Dict[str, str] = None) -> bool:
        key = self._make_cache_key(url, vary_headers or {})
        if key in self._cache:
            del self._cache[key]
            self.purges += 1
            return True
        return False

    def purge_by_surrogate_key(self, surrogate_key: str) -> int:
        """Purge all responses tagged with surrogate_key."""
        keys = self._surrogate_index.pop(surrogate_key, [])
        purged = 0
        for key in keys:
            if key in self._cache:
                del self._cache[key]
                purged += 1
        self.purges += purged
        return purged

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def cache_size(self) -> int:
        return len(self._cache)

    def stats(self) -> str:
        return (f"{self.name} ({self.location}): "
                f"hits={self.hits} misses={self.misses} "
                f"hit_ratio={self.hit_ratio:.1%} "
                f"origin_fetches={self.origin_fetches}")


# ─────────────────────────────────────────────
# ORIGIN SERVER
# ─────────────────────────────────────────────

class OriginServer:
    """Simulates the origin application server."""

    def __init__(self):
        self.requests_served = 0
        self._content = {
            "/":               ("<html>Home</html>",              "text/html",       "public, s-maxage=60, stale-while-revalidate=300"),
            "/static/app.js":  ("function app(){}",               "application/js",  "public, max-age=31536000, immutable"),
            "/api/products":   ('{"products":[...]}',             "application/json","public, s-maxage=30"),
            "/api/user/me":    ('{"name":"Alice"}',               "application/json","private, no-store"),
            "/images/hero.jpg":("(binary image data)",            "image/jpeg",      "public, max-age=86400"),
            "/api/cart":       ('{"items":[]}',                   "application/json","private, no-cache"),
        }

    def handle(self, url: str) -> CachedResponse:
        self.requests_served += 1
        time.sleep(random.uniform(10, 30) / 1000)   # 10-30ms origin latency

        body, content_type, cc_header = self._content.get(
            url, ('<html>404</html>', 'text/html', 'no-store')
        )
        cc = CacheDirectives.parse(cc_header)

        # Extract surrogate keys from content type
        surrogate_keys = []
        if "/api/products" in url:
            surrogate_keys = ["product_catalog", "product_list"]
        elif "/images/" in url:
            surrogate_keys = ["media", "images"]

        return CachedResponse(
            body=body.encode(),
            status_code=200,
            headers={"Content-Type": content_type, "Cache-Control": cc_header},
            cached_at=time.time(),
            ttl_s=cc.cdn_ttl_s or 0,
            surrogate_keys=surrogate_keys,
        )


# ─────────────────────────────────────────────
# CDN NETWORK SIMULATOR
# ─────────────────────────────────────────────

class CDNNetwork:
    """Simulates a CDN network with multiple edge nodes."""

    def __init__(self, origin: OriginServer):
        self.origin    = origin
        self.edge_nodes: Dict[str, CDNEdgeNode] = {}

    def add_edge(self, name: str, location: str, origin_latency_ms: float):
        self.edge_nodes[name] = CDNEdgeNode(name, location, origin_latency_ms)

    def route_request(self, url: str, user_region: str,
                      vary_headers: Dict[str, str] = None) -> Tuple[str, str, float]:
        """Route user request to nearest edge node."""
        # Simple routing: pick first matching region
        node = None
        for name, edge in self.edge_nodes.items():
            if user_region.lower() in edge.location.lower():
                node = edge
                break
        if not node:
            node = next(iter(self.edge_nodes.values()))

        start  = time.perf_counter()
        resp, status = node.request(url, vary_headers or {}, self.origin.handle)
        latency_ms   = (time.perf_counter() - start) * 1000

        return status, node.name, latency_ms

    def purge_surrogate_key(self, surrogate_key: str) -> int:
        """Purge key from ALL edge nodes (broadcast invalidation)."""
        total = 0
        for node in self.edge_nodes.values():
            total += node.purge_by_surrogate_key(surrogate_key)
        return total

    def global_hit_ratio(self) -> float:
        total_hits   = sum(n.hits   for n in self.edge_nodes.values())
        total_misses = sum(n.misses for n in self.edge_nodes.values())
        return total_hits / max(1, total_hits + total_misses)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_cdn_cache():
    print("=" * 65)
    print("CDN AS CACHE (EDGE CACHING)")
    print("=" * 65)

    random.seed(42)

    # ── Cache-Control Parsing ─────────────────
    print("\n[1] CACHE-CONTROL HEADER ANALYSIS")
    print("─" * 55)
    headers = [
        ("public, max-age=86400, s-maxage=3600",         "Static HTML, longer CDN TTL"),
        ("public, max-age=31536000, immutable",           "Versioned JS/CSS (fingerprinted)"),
        ("public, s-maxage=30, stale-while-revalidate=60","Short-lived API response"),
        ("private, no-store",                             "User-specific data (cart, profile)"),
        ("no-cache",                                      "Must revalidate with origin"),
        ("public, max-age=86400",                         "Images (browser=CDN=1 day)"),
    ]
    print(f"  {'Cache-Control':<45} {'CDN TTL':<10} {'Cacheable':<10} {'Description'}")
    print(f"  {'─'*90}")
    for header, desc in headers:
        cc = CacheDirectives.parse(header)
        ttl_str = f"{cc.cdn_ttl_s}s" if cc.cdn_ttl_s else "No"
        print(f"  {header:<45} {ttl_str:<10} {str(cc.is_cacheable_by_cdn):<10} {desc}")

    # ── CDN Network Simulation ────────────────
    print("\n\n[2] CDN NETWORK — EDGE NODE ROUTING")
    print("─" * 55)
    origin = OriginServer()
    cdn    = CDNNetwork(origin)
    cdn.add_edge("edge-us-east",  "us-east",    5.0)
    cdn.add_edge("edge-europe",   "europe",     120.0)
    cdn.add_edge("edge-asia",     "asia",       250.0)

    # Simulate requests from different regions
    requests = [
        ("/", "us-east"),
        ("/static/app.js", "us-east"),
        ("/api/products", "us-east"),
        ("/api/user/me", "us-east"),    # private — not cached
        ("/", "europe"),
        ("/api/products", "europe"),
        ("/", "asia"),
        ("/static/app.js", "asia"),
    ]

    print(f"  {'URL':<25} {'Region':<10} {'Status':<6} {'Edge Node':<18} {'Latency'}")
    print(f"  {'─'*70}")
    for url, region in requests:
        status, edge, latency = cdn.route_request(url, region)
        print(f"  {url:<25} {region:<10} {status:<6} {edge:<18} {latency:.1f}ms")

    # Repeat requests — should hit cache
    print(f"\n  Repeating same requests (should be cache hits):")
    for url, region in requests[:4]:
        status, edge, latency = cdn.route_request(url, region)
        print(f"  {url:<25} {status:<6} {latency:.2f}ms")

    print(f"\n  Origin requests: {origin.requests_served} (rest served from edge cache)")
    print(f"  Global hit ratio: {cdn.global_hit_ratio():.1%}")

    for name, node in cdn.edge_nodes.items():
        print(f"  {node.stats()}")

    # ── Surrogate Key Purge ────────────────────
    print("\n\n[3] SURROGATE KEY CACHE PURGE")
    print("─" * 55)
    print("  Product catalog updated → purge all tagged responses")
    print("  Surrogate key: 'product_catalog'")

    # Access products from multiple edges to populate cache
    for region in ["us-east", "europe", "asia"]:
        cdn.route_request("/api/products", region)

    purged = cdn.purge_surrogate_key("product_catalog")
    print(f"  Purged {purged} cached responses across all edge nodes")
    print(f"  Next request will fetch fresh from origin")

    # ── Content Strategy ──────────────────────
    print("\n\n[4] CDN CACHING STRATEGY BY CONTENT TYPE")
    print("─" * 55)
    strategy = [
        ("HTML pages",          "s-maxage=60",      "stale-while-revalidate=300", "Serve slightly stale, refresh in bg"),
        ("JS/CSS (hashed)",     "max-age=31536000", "immutable",                 "Never expires — hash changes on deploy"),
        ("Images (hashed)",     "max-age=31536000", "immutable",                 "Same as JS/CSS — content-addressed"),
        ("Images (non-hashed)", "max-age=86400",    "",                          "1-day cache, purge on update"),
        ("Public API (list)",   "s-maxage=30",      "stale-while-revalidate=60", "Fresh but allow brief staleness"),
        ("Private API",         "private, no-store","",                          "Never cache — user-specific"),
        ("Videos (HLS)",        "max-age=86400",    "",                          "Byte-range caching per segment"),
    ]
    print(f"  {'Content Type':<24} {'Cache-Control':<16} {'Extra':<26} {'Note'}")
    print(f"  {'─'*80}")
    for ctype, cc, extra, note in strategy:
        print(f"  {ctype:<24} {cc:<16} {extra:<26} {note}")

    # ── CDN Hit Ratio Impact ──────────────────
    print("\n\n[5] CDN HIT RATIO — ORIGIN LOAD REDUCTION")
    print("─" * 55)
    scenarios = [
        (0.99, 100_000, "Static asset CDN (JS/CSS/Images)"),
        (0.90, 100_000, "Typical web app with CDN"),
        (0.70, 100_000, "API responses with moderate cache"),
        (0.50, 100_000, "Low-TTL or high-cardinality API"),
        (0.20, 100_000, "Mostly private/user-specific content"),
    ]
    print(f"  {'Hit Ratio':<12} {'Total QPS':<12} {'Origin QPS':<12} {'Origin Load Reduction':<24} {'Use Case'}")
    print(f"  {'─'*80}")
    for ratio, qps, use_case in scenarios:
        origin_qps = int(qps * (1 - ratio))
        reduction  = ratio
        print(f"  {ratio:.0%}         {qps:<12,} {origin_qps:<12,} {reduction:.0%}                     {use_case}")


if __name__ == "__main__":
    demonstrate_cdn_cache()
