"""
URL Shortener - Working Python Implementation
Demonstrates: base62 encoding, LRU cache, collision resolution,
              rate limiting (token bucket), analytics, expiry, custom aliases.
No external dependencies — standard library only.
"""

import hashlib
import time
import collections
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Base62 Encoder / Decoder
# ---------------------------------------------------------------------------
BASE62_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE = 62
SHORT_CODE_LENGTH = 7


def encode_base62(num: int) -> str:
    """Convert a non-negative integer to a base62 string (right-padded to 7 chars)."""
    if num == 0:
        return BASE62_CHARS[0] * SHORT_CODE_LENGTH
    chars = []
    while num:
        chars.append(BASE62_CHARS[num % BASE])
        num //= BASE
    # Pad to fixed length
    while len(chars) < SHORT_CODE_LENGTH:
        chars.append(BASE62_CHARS[0])
    return "".join(reversed(chars))


def decode_base62(s: str) -> int:
    """Convert a base62 string back to an integer."""
    num = 0
    for char in s:
        num = num * BASE + BASE62_CHARS.index(char)
    return num


# ---------------------------------------------------------------------------
# LRU Cache  (O(1) get/put using OrderedDict)
# ---------------------------------------------------------------------------
class LRUCache:
    """
    Least Recently Used cache.
    Evicts least recently used entry when capacity is exceeded.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: collections.OrderedDict = collections.OrderedDict()

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)   # mark as recently used
        return self.cache[key]

    def put(self, key: str, value) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # evict LRU item

    def delete(self, key: str) -> None:
        self.cache.pop(key, None)

    def __len__(self):
        return len(self.cache)


# ---------------------------------------------------------------------------
# Rate Limiter  (Token Bucket per user per day)
# ---------------------------------------------------------------------------
class RateLimiter:
    """
    Per-user daily rate limiter using a simple counter.
    Resets at midnight (UTC).
    """

    def __init__(self, default_daily_limit: int = 100):
        self.default_limit = default_daily_limit
        # { user_id: {"count": int, "date": str, "limit": int} }
        self._buckets: Dict[str, dict] = {}

    def _today(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d")

    def set_limit(self, user_id: str, limit: int) -> None:
        """Override limit for a specific user (e.g., pro tier)."""
        bucket = self._buckets.setdefault(user_id, {"count": 0, "date": self._today()})
        bucket["limit"] = limit

    def is_allowed(self, user_id: str) -> Tuple[bool, int]:
        """
        Returns (allowed: bool, remaining: int).
        Resets counter if date has changed.
        """
        today = self._today()
        if user_id not in self._buckets:
            self._buckets[user_id] = {"count": 0, "date": today, "limit": self.default_limit}
        bucket = self._buckets[user_id]
        # Reset on new day
        if bucket["date"] != today:
            bucket["count"] = 0
            bucket["date"] = today
        limit = bucket.get("limit", self.default_limit)
        if bucket["count"] >= limit:
            return False, 0
        return True, limit - bucket["count"]

    def consume(self, user_id: str) -> bool:
        """Consume one token. Returns False if limit exceeded."""
        allowed, _ = self.is_allowed(user_id)
        if allowed:
            self._buckets[user_id]["count"] += 1
        return allowed


# ---------------------------------------------------------------------------
# Analytics Tracker
# ---------------------------------------------------------------------------
class AnalyticsTracker:
    """
    Tracks click events per short code.
    In production this would stream to Kafka -> ClickHouse.
    """

    def __init__(self):
        # { short_code: { "total": int, "by_day": {date: count}, "referrers": Counter } }
        self._data: Dict[str, dict] = {}

    def record_click(self, short_code: str, referrer: str = "", country: str = ""):
        if short_code not in self._data:
            self._data[short_code] = {
                "total": 0,
                "by_day": collections.Counter(),
                "referrers": collections.Counter(),
                "countries": collections.Counter(),
            }
        rec = self._data[short_code]
        rec["total"] += 1
        rec["by_day"][datetime.utcnow().strftime("%Y-%m-%d")] += 1
        if referrer:
            rec["referrers"][referrer] += 1
        if country:
            rec["countries"][country] += 1

    def get_stats(self, short_code: str) -> Optional[dict]:
        rec = self._data.get(short_code)
        if not rec:
            return None
        return {
            "total_clicks": rec["total"],
            "clicks_by_day": dict(rec["by_day"]),
            "top_referrers": rec["referrers"].most_common(5),
            "top_countries": rec["countries"].most_common(5),
        }


# ---------------------------------------------------------------------------
# URL Record
# ---------------------------------------------------------------------------
class URLRecord:
    def __init__(
        self,
        short_code: str,
        long_url: str,
        user_id: str,
        created_at: datetime,
        expires_at: Optional[datetime] = None,
        is_custom: bool = False,
    ):
        self.short_code = short_code
        self.long_url = long_url
        self.user_id = user_id
        self.created_at = created_at
        self.expires_at = expires_at
        self.is_custom = is_custom
        self.is_active = True

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def __repr__(self):
        return f"URLRecord(code={self.short_code}, url={self.long_url[:40]}...)"


# ---------------------------------------------------------------------------
# URL Shortener — Core System
# ---------------------------------------------------------------------------
class URLShortener:
    """
    Core URL shortener service.
    - Base62 encoding with distributed counter
    - LRU cache for hot URL lookup
    - Rate limiting per user
    - Custom alias support
    - Expiry handling
    - Click analytics
    - Collision resolution
    """

    BASE_URL = "https://sho.rt/"

    def __init__(self, cache_capacity: int = 1000):
        # Simulated DB: { short_code -> URLRecord }
        self._db: Dict[str, URLRecord] = {}
        # Reverse index for idempotency: long_url -> short_code
        self._reverse_db: Dict[str, str] = {}

        # Distributed counter (Redis INCR in prod)
        self._counter = 1_000_000  # start high to avoid trivial codes

        # LRU cache (Redis in prod)
        self._cache = LRUCache(capacity=cache_capacity)

        # Rate limiter
        self._rate_limiter = RateLimiter(default_daily_limit=100)

        # Analytics
        self._analytics = AnalyticsTracker()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_code(self) -> str:
        """Atomically increment counter and return base62 code."""
        code = encode_base62(self._counter)
        self._counter += 1
        return code

    def _generate_hash_code(self, long_url: str, attempt: int = 0) -> str:
        """
        Hash-based code generation (alternative path).
        Uses MD5 of url+attempt, takes 7 chars of base62-encoded digest.
        """
        digest = hashlib.md5(f"{long_url}{attempt}".encode()).hexdigest()
        # Convert hex to int, then to base62
        num = int(digest[:12], 16)  # first 12 hex chars -> 48-bit int
        return encode_base62(num % (BASE ** SHORT_CODE_LENGTH))

    def _code_exists(self, code: str) -> bool:
        """Check if a code is already used (cache first, then DB)."""
        if self._cache.get(code) is not None:
            return True
        return code in self._db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shorten(
        self,
        long_url: str,
        user_id: str = "anonymous",
        custom_alias: Optional[str] = None,
        expires_in_days: Optional[int] = None,
    ) -> dict:
        """
        Create a short URL.
        Returns dict with short_url, short_code, or raises ValueError.
        """
        # 1. Rate limit check
        if not self._rate_limiter.consume(user_id):
            return {"error": "daily rate limit exceeded", "code": 429}

        # 2. Idempotency — return existing code if same URL was shortened before
        if long_url in self._reverse_db and custom_alias is None:
            existing_code = self._reverse_db[long_url]
            record = self._db.get(existing_code)
            if record and not record.is_expired() and record.is_active:
                return {
                    "short_url": self.BASE_URL + existing_code,
                    "short_code": existing_code,
                    "long_url": long_url,
                    "created_at": record.created_at.isoformat(),
                    "existing": True,
                }

        # 3. Determine short code
        if custom_alias:
            code = custom_alias
            if self._code_exists(code):
                return {"error": f"alias '{code}' is already taken", "code": 409}
            is_custom = True
        else:
            # Counter-based: O(1), no collision possible
            code = self._next_code()
            is_custom = False

        # 4. Compute expiry
        expires_at = None
        if expires_in_days is not None:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # 5. Persist to DB
        record = URLRecord(
            short_code=code,
            long_url=long_url,
            user_id=user_id,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            is_custom=is_custom,
        )
        self._db[code] = record
        self._reverse_db[long_url] = code

        # 6. Write to cache
        self._cache.put(code, record)

        return {
            "short_url": self.BASE_URL + code,
            "short_code": code,
            "long_url": long_url,
            "created_at": record.created_at.isoformat(),
            "expires_at": expires_at.isoformat() if expires_at else None,
        }

    def resolve(
        self,
        short_code: str,
        referrer: str = "",
        country: str = "",
        use_301: bool = False,
    ) -> dict:
        """
        Resolve a short code to its long URL.
        Logs analytics asynchronously (simulated here as synchronous).
        Returns dict with long_url and redirect_type, or error.
        """
        # 1. Cache lookup (L1)
        record = self._cache.get(short_code)

        # 2. DB lookup on cache miss (L2)
        if record is None:
            record = self._db.get(short_code)
            if record:
                self._cache.put(short_code, record)  # populate cache

        # 3. Not found
        if record is None:
            return {"error": "short URL not found", "code": 404}

        # 4. Expiry check
        if record.is_expired():
            return {"error": "short URL has expired", "code": 410}

        # 5. Deactivated
        if not record.is_active:
            return {"error": "short URL has been deleted", "code": 404}

        # 6. Log analytics (async in production via Kafka)
        self._analytics.record_click(short_code, referrer=referrer, country=country)

        # 7. Return redirect info
        redirect_code = 301 if use_301 else 302
        return {
            "long_url": record.long_url,
            "redirect_type": redirect_code,
            "short_code": short_code,
        }

    def delete(self, short_code: str, user_id: str) -> dict:
        """Soft-delete a short URL (mark inactive)."""
        record = self._db.get(short_code)
        if record is None:
            return {"error": "not found", "code": 404}
        if record.user_id != user_id:
            return {"error": "forbidden", "code": 403}
        record.is_active = False
        self._cache.delete(short_code)  # invalidate cache
        return {"success": True, "short_code": short_code}

    def get_analytics(self, short_code: str, user_id: str) -> dict:
        """Return analytics for a short code (owner only)."""
        record = self._db.get(short_code)
        if record is None:
            return {"error": "not found", "code": 404}
        if record.user_id != user_id:
            return {"error": "forbidden", "code": 403}
        stats = self._analytics.get_stats(short_code)
        if stats is None:
            return {"short_code": short_code, "total_clicks": 0}
        return {"short_code": short_code, **stats}

    def set_user_limit(self, user_id: str, daily_limit: int) -> None:
        """Configure custom daily limit for a user (e.g., pro tier)."""
        self._rate_limiter.set_limit(user_id, daily_limit)

    def get_cache_stats(self) -> dict:
        return {"cache_size": len(self._cache), "db_size": len(self._db)}


# ---------------------------------------------------------------------------
# Demo / Simulation
# ---------------------------------------------------------------------------
def run_demo():
    print("=" * 60)
    print("URL SHORTENER DEMO")
    print("=" * 60)

    shortener = URLShortener(cache_capacity=500)

    # --- Basic shortening ---
    print("\n[1] Basic URL shortening")
    result = shortener.shorten(
        "https://www.example.com/some/very/long/path?utm_source=twitter&utm_campaign=summer2024",
        user_id="user_001",
    )
    print(f"  Short URL : {result['short_url']}")
    print(f"  Short Code: {result['short_code']}")
    print(f"  Expires   : {result['expires_at']}")

    code1 = result["short_code"]

    # --- Idempotency: same URL returns same code ---
    print("\n[2] Idempotency check (same URL -> same code)")
    result2 = shortener.shorten(
        "https://www.example.com/some/very/long/path?utm_source=twitter&utm_campaign=summer2024",
        user_id="user_001",
    )
    print(f"  Same code returned: {result2.get('existing', False)}")
    print(f"  Code: {result2['short_code']}  (matches: {result2['short_code'] == code1})")

    # --- Custom alias ---
    print("\n[3] Custom alias")
    result3 = shortener.shorten(
        "https://www.company.com/landing-page",
        user_id="user_002",
        custom_alias="my-brand",
    )
    print(f"  Custom URL: {result3['short_url']}")

    # Attempt to re-use same alias
    result4 = shortener.shorten(
        "https://www.other.com/page",
        user_id="user_003",
        custom_alias="my-brand",
    )
    print(f"  Conflict result: {result4.get('error')}")

    # --- URL with expiry ---
    print("\n[4] URL with expiry (1 day)")
    result5 = shortener.shorten(
        "https://www.flash-sale.com/deal?id=999",
        user_id="user_001",
        expires_in_days=1,
    )
    print(f"  Short URL  : {result5['short_url']}")
    print(f"  Expires at : {result5['expires_at']}")

    # --- Resolve URL (302 redirect) ---
    print("\n[5] Resolve URL")
    resolved = shortener.resolve(code1, referrer="twitter.com", country="US")
    print(f"  Long URL      : {resolved['long_url'][:60]}...")
    print(f"  Redirect type : {resolved['redirect_type']}")

    resolved2 = shortener.resolve("NONEXISTENT")
    print(f"  Not-found code: {resolved2['code']}")

    # --- Analytics ---
    print("\n[6] Analytics (simulate multiple clicks)")
    for country in ["US", "US", "US", "GB", "IN", "US", "CA"]:
        shortener.resolve(code1, referrer="twitter.com", country=country)
    for _ in range(3):
        shortener.resolve(code1, referrer="google.com", country="US")

    stats = shortener.get_analytics(code1, user_id="user_001")
    print(f"  Total clicks  : {stats['total_clicks']}")
    print(f"  Top countries : {stats['top_countries']}")
    print(f"  Top referrers : {stats['top_referrers']}")

    # --- Rate limiting ---
    print("\n[7] Rate limiting (free tier: 100/day)")
    heavy_user = "user_heavy"
    shortener.set_user_limit(heavy_user, 5)  # test with limit=5
    for i in range(7):
        res = shortener.shorten(f"https://example.com/page/{i}", user_id=heavy_user)
        status = res.get("error", "OK")
        print(f"  Request {i+1}: {status}")

    # --- Delete URL ---
    print("\n[8] Delete URL")
    del_result = shortener.delete(code1, user_id="user_001")
    print(f"  Delete result: {del_result}")
    resolve_deleted = shortener.resolve(code1)
    print(f"  Resolve after delete: {resolve_deleted.get('error')}")

    # --- Base62 encoder/decoder test ---
    print("\n[9] Base62 encode/decode verification")
    for n in [0, 1, 61, 62, 3521614606207]:  # 3.5T is 62^7 - 1
        encoded = encode_base62(n)
        decoded = decode_base62(encoded)
        print(f"  {n:>20} -> {encoded} -> {decoded}  (match: {decoded == n})")

    # --- Cache stats ---
    print("\n[10] Cache stats")
    print(f"  {shortener.get_cache_stats()}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
