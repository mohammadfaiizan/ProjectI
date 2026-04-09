"""
SYSTEM DESIGN: URL SHORTENER (like bit.ly / TinyURL)
======================================================

Problem Statement:
Design a service that takes a long URL and returns a short alias.
When users visit the short URL, they are redirected to the original URL.

Functional Requirements:
  - Shorten a URL: POST /shorten → returns short_code
  - Redirect: GET /{code} → 301/302 to original URL
  - Optional: custom alias, expiry, analytics (click count)

Non-Functional Requirements:
  - 100M URLs created per day → ~1150 writes/sec
  - 10:1 read:write ratio → 11500 redirects/sec
  - URL must persist for ≥ 5 years
  - Low latency: redirect < 10ms (after cache)
  - High availability: 99.99%

Estimation:
  Writes: 100M/day = 1150/sec
  Reads:  1B/day   = 11500/sec
  Storage per URL: code(7B) + url(200B) + ts(8B) = ~215B
  5 years × 100M/day × 365 × 215B = ~39TB

Short Code Generation:
  Option 1: MD5(url)[:7] → base62 encode.
             Collision risk: Birthday paradox with 3.5T codes.
  Option 2: Auto-increment ID → base62 encode.
             Predictable (sequential), easy, no collision.
  Option 3: Random 7-char base62 (62^7 = 3.5T codes).
             Check DB for collision on each insert.
  Option 4: Pre-generated pool of codes (like Snowflake for IDs).

Base62 Encoding:
  Characters: 0-9, a-z, A-Z (62 chars)
  7 chars → 62^7 = 3,521,614,606,208 possible codes
  At 1150 writes/sec, lasts 97,000 years

Redirect Type:
  301 Permanent: Browser caches → no server hit on repeat visits.
                 Good for CDN offload; bad for analytics.
  302 Temporary: No browser cache → every visit hits server.
                 Good for analytics (every click tracked).

Data Model:
  Table: urls
    short_code VARCHAR(10) PK
    original_url TEXT
    user_id  BIGINT (nullable)
    created_at TIMESTAMP
    expires_at TIMESTAMP (nullable)
    click_count BIGINT DEFAULT 0

Scaling Strategy:
  Read path:   Cache short_code → url in Redis (TTL = expiry - now).
               CDN caches 301 responses.
  Write path:  Single-leader DB with replicas for reads.
  DB:          MySQL/Postgres with code as PK.
               Partition by short_code if > 1TB.
  Counter:     Approximate click counts in Redis; flush to DB periodically.

Custom Aliases:
  Allow users to specify a code (e.g., bit.ly/my-link).
  Reserve namespace; validate against reserved words (api, admin, health).

Rate Limiting:
  Free tier: 1000 shortens/day per IP.
  Auth users: 100k/day.
  Prevents abuse / spam URLs.
"""

from __future__ import annotations

import hashlib
import time
import random
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ─────────────────────────────────────────────
# BASE62 CODEC
# ─────────────────────────────────────────────

BASE62_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def to_base62(num: int) -> str:
    if num == 0:
        return BASE62_CHARS[0]
    chars = []
    while num:
        chars.append(BASE62_CHARS[num % 62])
        num //= 62
    return "".join(reversed(chars))

def from_base62(s: str) -> int:
    num = 0
    for c in s:
        num = num * 62 + BASE62_CHARS.index(c)
    return num


# ─────────────────────────────────────────────
# URL RECORD
# ─────────────────────────────────────────────

@dataclass
class URLRecord:
    short_code:   str
    original_url: str
    user_id:      Optional[str]
    created_at:   float
    expires_at:   Optional[float]
    click_count:  int = 0

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at


# ─────────────────────────────────────────────
# ID GENERATOR (auto-increment simulation)
# ─────────────────────────────────────────────

class AtomicCounter:
    def __init__(self, start: int = 100_000_000):
        self._val  = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._val += 1
            return self._val


# ─────────────────────────────────────────────
# URL STORE (database layer)
# ─────────────────────────────────────────────

class URLStore:
    """In-memory URL database."""

    def __init__(self):
        self._by_code: Dict[str, URLRecord]    = {}
        self._by_url:  Dict[str, URLRecord]    = {}   # dedup
        self._counter  = AtomicCounter()

    def create(self, original_url: str, user_id: Optional[str] = None,
               custom_code: Optional[str] = None,
               ttl_days: Optional[int] = None) -> URLRecord:

        # Dedup: same URL → same code (within same user)
        if original_url in self._by_url and custom_code is None:
            return self._by_url[original_url]

        code = custom_code or to_base62(self._counter.next())

        if code in self._by_code:
            raise ValueError(f"Code {code!r} already taken")

        expires = time.time() + ttl_days * 86400 if ttl_days else None
        rec = URLRecord(code, original_url, user_id, time.time(), expires)
        self._by_code[code] = rec
        self._by_url[original_url] = rec
        return rec

    def get(self, short_code: str) -> Optional[URLRecord]:
        return self._by_code.get(short_code)

    def increment_clicks(self, short_code: str):
        rec = self._by_code.get(short_code)
        if rec:
            rec.click_count += 1

    def stats(self) -> Dict:
        return {
            "total_urls": len(self._by_code),
            "total_clicks": sum(r.click_count for r in self._by_code.values()),
        }


# ─────────────────────────────────────────────
# CACHE LAYER
# ─────────────────────────────────────────────

class URLCache:
    """Redis-like LRU cache for short_code → original_url."""

    def __init__(self, max_size: int = 10_000):
        self._store:    Dict[str, Tuple[str, Optional[float]]] = {}
        self._max_size  = max_size
        self._hits      = 0
        self._misses    = 0

    def get(self, code: str) -> Optional[str]:
        entry = self._store.get(code)
        if entry is None:
            self._misses += 1
            return None
        url, expires = entry
        if expires and time.time() > expires:
            del self._store[code]
            self._misses += 1
            return None
        self._hits += 1
        return url

    def set(self, code: str, url: str, expires_at: Optional[float] = None):
        if len(self._store) >= self._max_size:
            # Evict a random entry (simplified LRU)
            evict = next(iter(self._store))
            del self._store[evict]
        self._store[code] = (url, expires_at)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


# ─────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────

class RateLimiter:
    """Sliding window rate limiter per key."""

    def __init__(self, max_requests: int, window_s: float):
        self._max   = max_requests
        self._win   = window_s
        self._store: Dict[str, List[float]] = defaultdict(list)
        self._lock  = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now    = time.time()
        cutoff = now - self._win
        with self._lock:
            ts_list = self._store[key]
            # Remove old timestamps
            self._store[key] = [t for t in ts_list if t > cutoff]
            if len(self._store[key]) >= self._max:
                return False
            self._store[key].append(now)
            return True


# ─────────────────────────────────────────────
# URL SHORTENER SERVICE
# ─────────────────────────────────────────────

RESERVED_CODES = {"api", "admin", "health", "metrics", "static", "docs"}

class URLShortener:
    """
    URL Shortener service combining store, cache, and rate limiter.
    """

    BASE_URL = "https://sho.rt/"

    def __init__(self):
        self._store   = URLStore()
        self._cache   = URLCache(max_size=1_000_000)
        self._rl      = RateLimiter(max_requests=100, window_s=3600)

    def shorten(self, original_url: str,
                user_id: Optional[str] = None,
                custom_code: Optional[str] = None,
                ttl_days: Optional[int] = None) -> Tuple[str, str]:
        """
        Returns (short_code, short_url).
        Raises ValueError on rate limit or invalid input.
        """
        key = user_id or "anonymous"

        if not self._rl.is_allowed(key):
            raise ValueError("Rate limit exceeded")

        if not original_url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL: must start with http:// or https://")

        if custom_code and custom_code in RESERVED_CODES:
            raise ValueError(f"Reserved code: {custom_code}")

        rec = self._store.create(original_url, user_id, custom_code, ttl_days)
        self._cache.set(rec.short_code, original_url, rec.expires_at)
        return rec.short_code, self.BASE_URL + rec.short_code

    def resolve(self, short_code: str) -> Optional[str]:
        """
        Returns original URL for redirect, or None if not found/expired.
        Increments click counter.
        """
        # Try cache first
        url = self._cache.get(short_code)
        if url:
            self._store.increment_clicks(short_code)
            return url

        # Cache miss → DB
        rec = self._store.get(short_code)
        if rec is None or rec.is_expired():
            return None

        # Populate cache
        self._cache.set(short_code, rec.original_url, rec.expires_at)
        rec.click_count += 1
        return rec.original_url

    def analytics(self, short_code: str) -> Optional[Dict]:
        rec = self._store.get(short_code)
        if not rec:
            return None
        return {
            "short_code":   rec.short_code,
            "original_url": rec.original_url[:60] + "...",
            "click_count":  rec.click_count,
            "created_at":   rec.created_at,
            "expires_at":   rec.expires_at,
        }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_url_shortener():
    print("=" * 65)
    print("SYSTEM DESIGN: URL SHORTENER")
    print("=" * 65)

    svc = URLShortener()

    # ── Basic Shortening ──────────────────────
    print("\n[1] URL SHORTENING")
    print("─" * 55)

    urls = [
        ("https://www.example.com/very/long/path/to/article?id=123&utm_source=newsletter", None, None, None),
        ("https://github.com/user/repo/blob/main/src/very/deep/file.py",                  "alice", None, 30),
        ("https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms", None, "gdocs", None),
    ]

    for url, uid, custom, ttl in urls:
        try:
            code, short = svc.shorten(url, uid, custom, ttl)
            ttl_str = f" (expires {ttl}d)" if ttl else ""
            print(f"  {short}{ttl_str}")
            print(f"    → {url[:60]}...")
        except ValueError as e:
            print(f"  ERROR: {e}")

    # ── Redirect ──────────────────────────────
    print("\n[2] REDIRECT RESOLUTION")
    print("─" * 55)

    test_codes = ["gdocs", "invalid123"]
    for code in ["1DFDfIp", "1DFDfIq", "gdocs", "invalid123"]:
        url = svc.resolve(code)
        print(f"  GET /{code} → "
              f"{'302 ' + url[:50] if url else '404 Not Found'}")

    # ── Cache Hit Rate ─────────────────────────
    print("\n[3] CACHE PERFORMANCE")
    print("─" * 55)

    # Simulate 1000 reads (80/20: 80% hit existing codes)
    code1, _ = svc.shorten("https://hot-article.com/post/1")
    code2, _ = svc.shorten("https://hot-article.com/post/2")

    for _ in range(800):   # hot URLs
        svc.resolve(code1)
        svc.resolve(code2)
    for _ in range(200):   # cold URLs (cache miss)
        svc.resolve(f"nonexistent_{random.randint(0,999)}")

    print(f"  Cache hit rate: {svc._cache.hit_rate*100:.1f}%")
    print(f"  Cache size:     {len(svc._cache._store)} entries")

    # ── Rate Limiting ─────────────────────────
    print("\n[4] RATE LIMITING")
    print("─" * 55)

    rl = RateLimiter(max_requests=5, window_s=60)
    key = "test_user"
    for i in range(7):
        allowed = rl.is_allowed(key)
        print(f"  Request {i+1}: {'ALLOWED' if allowed else 'RATE LIMITED'}")

    # ── Base62 Codec ──────────────────────────
    print("\n[5] BASE62 ENCODING")
    print("─" * 55)

    for num in [1, 100_000_000, 999_999_999, 3_521_614_606_207]:
        code = to_base62(num)
        back = from_base62(code)
        print(f"  {num:>20} → {code:<10} → {back}")

    print(f"\n  7-char base62 capacity: {62**7:,} unique codes")
    print(f"  At 1150 writes/sec: lasts {62**7 / 1150 / 86400 / 365:.0f} years")

    # ── Analytics ─────────────────────────────
    print("\n[6] ANALYTICS")
    print("─" * 55)

    for code in ["1DFDfIp", "1DFDfIq", "gdocs"]:
        info = svc.analytics(code)
        if info:
            print(f"  {info['short_code']}: {info['click_count']} clicks")

    # ── Architecture Summary ───────────────────
    print("\n[7] ARCHITECTURE SUMMARY")
    print("─" * 55)

    arch = [
        ("Write path",  "API server → Postgres (code as PK) → Redis cache"),
        ("Read path",   "API server → Redis (L1) → Postgres (L2) → 302 redirect"),
        ("Scalability", "Read replicas for Postgres; Redis cluster for cache"),
        ("CDN",         "Cloudflare caches 301 responses at edge"),
        ("Sharding",    "Shard DB by code prefix if > 1TB"),
        ("Analytics",   "Click events → Kafka → Flink → ClickHouse"),
        ("Expiry",      "Cron job deletes expired URLs; Redis TTL auto-evicts"),
        ("HA",          "Multi-region with active-active Postgres + CRDT counter"),
    ]
    for component, detail in arch:
        print(f"  {component:<15} {detail}")

    # ── Stats ─────────────────────────────────
    print("\n[8] STORE STATISTICS")
    print("─" * 55)

    stats = svc._store.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    demonstrate_url_shortener()
