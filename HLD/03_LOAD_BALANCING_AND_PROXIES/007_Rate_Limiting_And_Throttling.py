"""
RATE LIMITING AND THROTTLING
==============================

Problem Statement:
Without rate limiting, a single abusive client or misconfigured bot can
overwhelm your API, degrade service for all users, or drain your budget
(LLM APIs, SMS). Rate limiting protects availability and fairness.

Rate Limiting Algorithms:
  Fixed Window   : N requests per window (cheap, bursting at boundaries)
  Sliding Window : N requests in last T seconds (smooth, more memory)
  Token Bucket   : tokens refill at rate R; burst up to capacity B
  Leaky Bucket   : queue + constant drain rate (smooths bursts)
  Sliding Log    : store each request timestamp; most accurate

When to Rate Limit:
  Per-user     : prevent individual abuse
  Per-IP       : protect against DDoS before auth
  Per-endpoint : expensive operations (search, ML inference)
  Global       : protect total system capacity

Response Codes:
  429 Too Many Requests
  Headers: X-RateLimit-Limit, X-RateLimit-Remaining, Retry-After
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
from collections import deque
import time
import math


class RateLimitAlgorithm(Enum):
    FIXED_WINDOW   = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET   = "token_bucket"
    LEAKY_BUCKET   = "leaky_bucket"
    SLIDING_LOG    = "sliding_log"


@dataclass
class RateLimitResult:
    allowed       : bool
    remaining     : int
    limit         : int
    reset_after_s : float
    retry_after_s : float = 0.0

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "X-RateLimit-Limit"    : str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset"    : str(int(time.time() + self.reset_after_s)),
            **({"Retry-After": str(int(self.retry_after_s))} if not self.allowed else {})
        }


# ─────────────────────────────────────────────
# FIXED WINDOW
# ─────────────────────────────────────────────

class FixedWindow:
    """
    N requests per fixed window (e.g., 100 req/min).
    Problem: boundary bursting — 200 requests in 2 seconds spanning two windows.
    """

    def __init__(self, limit: int, window_s: float):
        self.limit    = limit
        self.window_s = window_s
        self._counts  : Dict[str, int]   = {}
        self._windows : Dict[str, float] = {}   # key → window start time

    def check(self, key: str) -> RateLimitResult:
        now = time.time()
        window_start = self._windows.get(key, 0)

        if now - window_start >= self.window_s:
            self._windows[key] = now
            self._counts[key]  = 0

        count = self._counts.get(key, 0)
        if count < self.limit:
            self._counts[key] = count + 1
            remaining = self.limit - self._counts[key]
            reset_in  = self.window_s - (now - self._windows[key])
            return RateLimitResult(True, remaining, self.limit, reset_in)
        else:
            reset_in = self.window_s - (now - self._windows[key])
            return RateLimitResult(False, 0, self.limit, reset_in, retry_after_s=reset_in)


# ─────────────────────────────────────────────
# TOKEN BUCKET
# ─────────────────────────────────────────────

class TokenBucket:
    """
    Tokens added at rate R per second, capacity B.
    Each request consumes 1 token. Burst up to B requests.
    Most common algorithm for APIs (AWS API GW, Nginx).
    """

    def __init__(self, rate: float, burst: int):
        self.rate     = rate    # tokens per second
        self.burst    = burst   # max tokens (burst capacity)
        self._buckets : Dict[str, Dict] = {}

    def _get_bucket(self, key: str) -> Dict:
        if key not in self._buckets:
            self._buckets[key] = {"tokens": float(self.burst), "last_refill": time.time()}
        return self._buckets[key]

    def _refill(self, bucket: Dict):
        now     = time.time()
        elapsed = now - bucket["last_refill"]
        refill  = elapsed * self.rate
        bucket["tokens"]      = min(self.burst, bucket["tokens"] + refill)
        bucket["last_refill"] = now

    def check(self, key: str, cost: int = 1) -> RateLimitResult:
        bucket = self._get_bucket(key)
        self._refill(bucket)
        tokens = bucket["tokens"]

        if tokens >= cost:
            bucket["tokens"] -= cost
            remaining = int(bucket["tokens"])
            return RateLimitResult(True, remaining, self.burst, 0.0)
        else:
            wait = (cost - tokens) / self.rate
            return RateLimitResult(False, 0, self.burst, 1.0 / self.rate, retry_after_s=wait)

    def get_tokens(self, key: str) -> float:
        b = self._get_bucket(key)
        self._refill(b)
        return b["tokens"]


# ─────────────────────────────────────────────
# LEAKY BUCKET
# ─────────────────────────────────────────────

class LeakyBucket:
    """
    Requests enter a queue; processed at constant rate.
    Smooths bursts into a steady flow. Good for outbound rate limiting.
    """

    def __init__(self, rate: float, capacity: int):
        self.rate     = rate       # requests per second to drain
        self.capacity = capacity   # max queue depth
        self._queues  : Dict[str, Dict] = {}

    def _get_queue(self, key: str) -> Dict:
        if key not in self._queues:
            self._queues[key] = {"count": 0, "last_drain": time.time()}
        return self._queues[key]

    def _drain(self, q: Dict):
        now     = time.time()
        elapsed = now - q["last_drain"]
        drained = elapsed * self.rate
        q["count"]      = max(0, q["count"] - drained)
        q["last_drain"] = now

    def check(self, key: str) -> RateLimitResult:
        q = self._get_queue(key)
        self._drain(q)

        if q["count"] < self.capacity:
            q["count"] += 1
            remaining = int(self.capacity - q["count"])
            return RateLimitResult(True, remaining, self.capacity, 0.0)
        else:
            wait = (q["count"] - self.capacity + 1) / self.rate
            return RateLimitResult(False, 0, self.capacity, wait, retry_after_s=wait)


# ─────────────────────────────────────────────
# SLIDING WINDOW LOG
# ─────────────────────────────────────────────

class SlidingWindowLog:
    """
    Store timestamp of every request.
    Count requests in [now - window, now].
    Most accurate; memory = O(limit) per key.
    """

    def __init__(self, limit: int, window_s: float):
        self.limit    = limit
        self.window_s = window_s
        self._logs    : Dict[str, Deque[float]] = {}

    def check(self, key: str) -> RateLimitResult:
        now = time.time()
        if key not in self._logs:
            self._logs[key] = deque()

        log = self._logs[key]
        # Remove old entries
        cutoff = now - self.window_s
        while log and log[0] < cutoff:
            log.popleft()

        if len(log) < self.limit:
            log.append(now)
            remaining = self.limit - len(log)
            reset_in  = (log[0] + self.window_s) - now if log else self.window_s
            return RateLimitResult(True, remaining, self.limit, reset_in)
        else:
            oldest    = log[0]
            retry_at  = oldest + self.window_s - now
            return RateLimitResult(False, 0, self.limit, retry_at, retry_after_s=retry_at)


# ─────────────────────────────────────────────
# RATE LIMITER MIDDLEWARE
# ─────────────────────────────────────────────

class RateLimiterMiddleware:
    """
    Multi-tier rate limiter: checks global, per-IP, per-user limits.
    First limit that fires → 429.
    """

    def __init__(self):
        self._limiters : List[Dict] = []
        self.blocked   = 0
        self.allowed   = 0

    def add_limiter(self, name: str, limiter, key_extractor):
        self._limiters.append({"name": name, "limiter": limiter,
                                "key": key_extractor})

    def check(self, user_id: str, ip: str, endpoint: str) -> RateLimitResult:
        context = {"user_id": user_id, "ip": ip, "endpoint": endpoint}
        for layer in self._limiters:
            key    = layer["key"](context)
            result = layer["limiter"].check(key)
            if not result.allowed:
                self.blocked += 1
                print(f"  RateLimit [{layer['name']}]: key={key} → 429")
                return result
        self.allowed += 1
        return RateLimitResult(True, 0, 0, 0.0)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_rate_limiting():
    print("=" * 65)
    print("RATE LIMITING AND THROTTLING")
    print("=" * 65)

    # ── Fixed Window ──────────────────────────
    print("\n[1] FIXED WINDOW (5 req / 10 sec)")
    print("─" * 55)
    fw = FixedWindow(limit=5, window_s=10.0)
    print("  Sending 8 requests rapidly:")
    for i in range(8):
        result = fw.check("user-alice")
        icon   = "✅" if result.allowed else "🚫"
        print(f"  req {i+1}: {icon}  remaining={result.remaining}  "
              + (f"retry_after={result.retry_after_s:.1f}s" if not result.allowed else ""))

    # ── Token Bucket ──────────────────────────
    print("\n\n[2] TOKEN BUCKET (rate=2/s, burst=5)")
    print("─" * 55)
    tb = TokenBucket(rate=2.0, burst=5)
    print("  Burst: 7 rapid requests (should allow 5, block 2):")
    for i in range(7):
        result = tb.check("user-bob")
        icon   = "✅" if result.allowed else "🚫"
        print(f"  req {i+1}: {icon}  tokens_left={tb.get_tokens('user-bob'):.2f}  "
              + (f"retry_in={result.retry_after_s:.2f}s" if not result.allowed else ""))

    # ── Leaky Bucket ──────────────────────────
    print("\n\n[3] LEAKY BUCKET (rate=2/s, capacity=4)")
    print("─" * 55)
    lb = LeakyBucket(rate=2.0, capacity=4)
    print("  Burst of 6 requests:")
    for i in range(6):
        result = lb.check("user-carol")
        icon   = "✅" if result.allowed else "🚫"
        print(f"  req {i+1}: {icon}  queue_remaining={result.remaining}  "
              + (f"retry_in={result.retry_after_s:.2f}s" if not result.allowed else ""))

    # ── Sliding Window Log ────────────────────
    print("\n\n[4] SLIDING WINDOW LOG (5 req / 10 sec)")
    print("─" * 55)
    swl = SlidingWindowLog(limit=5, window_s=10.0)
    print("  Sending 7 requests:")
    for i in range(7):
        result = swl.check("user-dave")
        icon   = "✅" if result.allowed else "🚫"
        print(f"  req {i+1}: {icon}  remaining={result.remaining}  "
              + (f"retry_in={result.retry_after_s:.1f}s" if not result.allowed else ""))

    # ── Multi-tier Middleware ─────────────────
    print("\n\n[5] MULTI-TIER RATE LIMITING")
    print("─" * 55)
    mw = RateLimiterMiddleware()
    mw.add_limiter("global",      TokenBucket(100.0, 200),   lambda c: "global")
    mw.add_limiter("per-ip",      FixedWindow(10, 60.0),     lambda c: f"ip:{c['ip']}")
    mw.add_limiter("per-user",    TokenBucket(2.0, 5),       lambda c: f"user:{c['user_id']}")
    mw.add_limiter("per-endpoint",FixedWindow(3, 10.0),      lambda c: f"ep:{c['endpoint']}")

    print("  Sending 8 requests from alice (user limit=2/s burst=5):")
    for i in range(8):
        result = mw.check("alice", "1.2.3.4", "/api/search")
        icon   = "✅" if result.allowed else "🚫"
        print(f"  req {i+1}: {icon}")

    print(f"\n  Middleware totals: allowed={mw.allowed}  blocked={mw.blocked}")

    # ── Algorithm Comparison ──────────────────
    print("\n\n[6] ALGORITHM COMPARISON")
    print("─" * 55)
    rows = [
        ("Fixed Window",     "O(1)",  "Simple, fast",          "Burst at window boundary"),
        ("Token Bucket",     "O(1)",  "Allows controlled burst","Slight complexity"),
        ("Leaky Bucket",     "O(1)",  "Smooths to steady rate", "No burst allowed"),
        ("Sliding Window",   "O(N)",  "No boundary burst",      "More memory (O(limit))"),
        ("Sliding Log",      "O(N)",  "Most accurate",          "Highest memory/CPU"),
    ]
    print(f"  {'Algorithm':<20} {'Memory':<8} {'Pros':<30} {'Cons'}")
    print(f"  {'─'*75}")
    for algo, mem, pros, cons in rows:
        print(f"  {algo:<20} {mem:<8} {pros:<30} {cons}")

    print("\n\n[7] RATE LIMIT RESPONSE HEADERS")
    print("─" * 55)
    result = tb.check("user-example")
    print("  Example headers when rate limiting:")
    for k, v in result.headers.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    demonstrate_rate_limiting()
