"""
SYSTEM DESIGN: DISTRIBUTED RATE LIMITER
=========================================

Problem Statement:
Design a rate limiter that enforces request quotas across multiple
distributed API servers. Must handle millions of requests per second
with low latency overhead.

Functional Requirements:
  - Allow/deny requests based on configurable limits
  - Multiple limit types: per-user, per-IP, per-API-key, per-endpoint
  - Multiple time windows: per-second, per-minute, per-day
  - Return remaining quota and reset time in response headers

Non-Functional Requirements:
  - Decision latency: < 1ms overhead (not on hot path)
  - Accurate within ~0.1% across distributed nodes
  - Support 10M concurrent users × multiple limit tiers

Rate Limiting Algorithms:

  1. Fixed Window Counter:
     Count requests in [0-60s], [60-120s] windows.
     Pro: simple, O(1) space.
     Con: burst at window boundary (2× limit in 2s).

  2. Sliding Window Log:
     Timestamp log per user. Count logs in [now-60s, now].
     Pro: exact.
     Con: O(N) memory per user (N = requests in window).

  3. Sliding Window Counter:
     Hybrid: fixed window + weight current vs previous window.
     estimate = prev_count × (1 - elapsed/window) + curr_count
     Pro: O(1) space, ~97% accurate.
     Con: approximation.

  4. Token Bucket:
     Bucket refills at rate R per second. Max capacity B.
     Each request consumes 1 token.
     Pro: handles bursts up to B. Smooth long-term rate.
     Con: need to track refill time.

  5. Leaky Bucket:
     Queue requests; process at fixed rate.
     Pro: smooths traffic (constant output rate).
     Con: can't handle legitimate bursts.

Distributed Rate Limiting:
  Centralized:  Redis INCR + EXPIRE. Accurate but Redis is bottleneck.
  Local + Sync: Each node keeps local counter + sync periodically.
                Allows short over-counting during sync gap.
  Token Bucket: Redis with atomic Lua script for token bucket.

Redis Lua for Atomic Rate Check:
  local tokens = redis.call('GET', key) or RATE_LIMIT
  if tokens > 0 then
    redis.call('DECRBY', key, 1)
    return 1   -- allow
  else
    return 0   -- deny
  end

Rate Limit Headers:
  X-RateLimit-Limit:     total allowed
  X-RateLimit-Remaining: remaining in window
  X-RateLimit-Reset:     Unix timestamp of window reset
  Retry-After:           seconds to wait (on 429)
"""

from __future__ import annotations

import time
import math
import threading
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque


# ─────────────────────────────────────────────
# RATE LIMIT RESULT
# ─────────────────────────────────────────────

@dataclass
class RateLimitResult:
    allowed:      bool
    limit:        int
    remaining:    int
    reset_at:     float    # unix timestamp
    retry_after:  Optional[float] = None

    def headers(self) -> Dict[str, str]:
        h = {
            "X-RateLimit-Limit":     str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset":     str(int(self.reset_at)),
        }
        if self.retry_after is not None:
            h["Retry-After"] = str(int(math.ceil(self.retry_after)))
        return h


# ─────────────────────────────────────────────
# 1. FIXED WINDOW COUNTER
# ─────────────────────────────────────────────

class FixedWindowCounter:
    """Simple counter per time window. Resets at window boundary."""

    def __init__(self, limit: int, window_s: int):
        self._limit   = limit
        self._window  = window_s
        self._store:  Dict[str, Tuple[int, int]] = {}  # key → (count, window_id)
        self._lock    = threading.Lock()

    def check(self, key: str) -> RateLimitResult:
        now       = time.time()
        window_id = int(now / self._window)
        reset_at  = (window_id + 1) * self._window

        with self._lock:
            stored_count, stored_window = self._store.get(key, (0, window_id))
            if stored_window != window_id:
                stored_count = 0   # new window

            if stored_count >= self._limit:
                return RateLimitResult(False, self._limit, 0, reset_at,
                                       retry_after=reset_at - now)
            stored_count += 1
            self._store[key] = (stored_count, window_id)
            return RateLimitResult(True, self._limit,
                                   self._limit - stored_count, reset_at)


# ─────────────────────────────────────────────
# 2. SLIDING WINDOW LOG
# ─────────────────────────────────────────────

class SlidingWindowLog:
    """Exact sliding window. Stores timestamp per request."""

    def __init__(self, limit: int, window_s: int):
        self._limit  = limit
        self._window = window_s
        self._logs:  Dict[str, deque] = defaultdict(deque)
        self._lock   = threading.Lock()

    def check(self, key: str) -> RateLimitResult:
        now    = time.time()
        cutoff = now - self._window

        with self._lock:
            log = self._logs[key]
            # Remove expired timestamps
            while log and log[0] <= cutoff:
                log.popleft()

            if len(log) >= self._limit:
                retry = log[0] - cutoff if log else 0
                return RateLimitResult(False, self._limit, 0,
                                       now + self._window, retry_after=retry)
            log.append(now)
            return RateLimitResult(True, self._limit,
                                   self._limit - len(log),
                                   now + self._window)


# ─────────────────────────────────────────────
# 3. SLIDING WINDOW COUNTER (hybrid)
# ─────────────────────────────────────────────

class SlidingWindowCounter:
    """
    Approximate sliding window using two fixed windows.
    estimate = prev_count × (1 - elapsed_frac) + curr_count
    ~97% accurate; O(1) space.
    """

    def __init__(self, limit: int, window_s: int):
        self._limit  = limit
        self._window = window_s
        # key → (prev_count, prev_window_id, curr_count, curr_window_id)
        self._store: Dict[str, Tuple[int, int, int, int]] = {}
        self._lock   = threading.Lock()

    def check(self, key: str) -> RateLimitResult:
        now       = time.time()
        window_id = int(now / self._window)
        elapsed   = now - window_id * self._window
        elapsed_frac = elapsed / self._window
        reset_at  = (window_id + 1) * self._window

        with self._lock:
            entry = self._store.get(key, (0, window_id - 1, 0, window_id))
            prev_count, prev_wid, curr_count, curr_wid = entry

            # Slide windows if needed
            if curr_wid != window_id:
                if curr_wid == window_id - 1:
                    prev_count = curr_count
                    prev_wid   = curr_wid
                else:
                    prev_count = 0
                    prev_wid   = window_id - 1
                curr_count = 0
                curr_wid   = window_id

            # Estimate requests in sliding window
            estimate = prev_count * (1.0 - elapsed_frac) + curr_count

            if estimate >= self._limit:
                return RateLimitResult(False, self._limit, 0, reset_at,
                                       retry_after=reset_at - now)
            curr_count += 1
            self._store[key] = (prev_count, prev_wid, curr_count, curr_wid)
            remaining = max(0, int(self._limit - estimate - 1))
            return RateLimitResult(True, self._limit, remaining, reset_at)


# ─────────────────────────────────────────────
# 4. TOKEN BUCKET
# ─────────────────────────────────────────────

class TokenBucket:
    """
    Tokens refill at rate R/sec. Max capacity B.
    Supports burst up to B.
    """

    def __init__(self, rate: float, capacity: int):
        self._rate     = rate      # tokens per second
        self._capacity = capacity
        self._store:   Dict[str, Tuple[float, float]] = {}  # key → (tokens, last_refill)
        self._lock     = threading.Lock()

    def check(self, key: str, cost: int = 1) -> RateLimitResult:
        now = time.time()

        with self._lock:
            tokens, last = self._store.get(key, (float(self._capacity), now))
            # Refill
            elapsed = now - last
            tokens  = min(self._capacity, tokens + elapsed * self._rate)

            if tokens >= cost:
                tokens -= cost
                self._store[key] = (tokens, now)
                remaining = int(tokens)
                reset_at  = now + (self._capacity - tokens) / self._rate
                return RateLimitResult(True, self._capacity, remaining, reset_at)
            else:
                wait = (cost - tokens) / self._rate
                self._store[key] = (tokens, now)
                return RateLimitResult(False, self._capacity, 0,
                                       now + wait, retry_after=wait)


# ─────────────────────────────────────────────
# 5. LEAKY BUCKET
# ─────────────────────────────────────────────

class LeakyBucket:
    """
    Requests are queued; processed at fixed rate.
    If queue full → deny.
    """

    def __init__(self, rate: float, queue_size: int):
        self._rate       = rate
        self._queue_size = queue_size
        # key → (last_drain_time, queue_count)
        self._store: Dict[str, Tuple[float, int]] = {}
        self._lock   = threading.Lock()

    def check(self, key: str) -> RateLimitResult:
        now = time.time()

        with self._lock:
            last_drain, queue_count = self._store.get(key, (now, 0))
            elapsed = now - last_drain
            # Drain queue
            drained    = int(elapsed * self._rate)
            queue_count = max(0, queue_count - drained)
            last_drain  = now if drained > 0 else last_drain

            if queue_count >= self._queue_size:
                wait = (queue_count - self._queue_size + 1) / self._rate
                self._store[key] = (last_drain, queue_count)
                return RateLimitResult(False, self._queue_size, 0,
                                       now + wait, retry_after=wait)
            queue_count += 1
            self._store[key] = (last_drain, queue_count)
            remaining = self._queue_size - queue_count
            return RateLimitResult(True, self._queue_size, remaining,
                                   now + queue_count / self._rate)


# ─────────────────────────────────────────────
# MULTI-TIER RATE LIMITER
# ─────────────────────────────────────────────

@dataclass
class TierConfig:
    name:     str
    limit:    int
    window_s: int


class MultiTierRateLimiter:
    """
    Checks multiple tiers: per-second, per-minute, per-day.
    Request denied if ANY tier is exceeded.
    """

    def __init__(self, tiers: List[TierConfig]):
        self._limiters = [
            (tier, SlidingWindowCounter(tier.limit, tier.window_s))
            for tier in tiers
        ]

    def check(self, key: str) -> RateLimitResult:
        most_restrictive = None
        for tier, limiter in self._limiters:
            result = limiter.check(key)
            if not result.allowed:
                return result
            if most_restrictive is None or result.remaining < most_restrictive.remaining:
                most_restrictive = result
        return most_restrictive or RateLimitResult(True, 0, 0, time.time())


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_rate_limiter():
    print("=" * 65)
    print("SYSTEM DESIGN: DISTRIBUTED RATE LIMITER")
    print("=" * 65)

    # ── Algorithm Comparison ──────────────────
    print("\n[1] ALGORITHM BEHAVIOR (10 requests/min limit)")
    print("─" * 55)

    limit  = 10
    window = 60

    algorithms = [
        ("FixedWindow",    FixedWindowCounter(limit, window)),
        ("SlidingLog",     SlidingWindowLog(limit, window)),
        ("SlidingCounter", SlidingWindowCounter(limit, window)),
    ]

    for name, alg in algorithms:
        results = []
        key = f"user_{name}"
        for _ in range(13):
            r = alg.check(key)
            results.append("✓" if r.allowed else "✗")
        print(f"  {name:<18}: {''.join(results)}")
        print(f"    {'':18}  [first 10 allowed, then denied]")

    # ── Token Bucket (burst) ──────────────────
    print("\n[2] TOKEN BUCKET (rate=5/sec, capacity=10)")
    print("─" * 55)

    tb = TokenBucket(rate=5.0, capacity=10)
    key = "api_user"

    # Burst of 10 requests immediately
    burst_results = []
    for _ in range(12):
        r = tb.check(key)
        burst_results.append("✓" if r.allowed else f"✗(wait {r.retry_after:.2f}s)")
    print(f"  Immediate burst (12 req): {' '.join(burst_results[:12])}")
    print(f"  (first 10 from full bucket; 11th denied; refill at 5/sec)")

    # ── Leaky Bucket ──────────────────────────
    print("\n[3] LEAKY BUCKET (rate=3/sec, queue=5)")
    print("─" * 55)

    lb = LeakyBucket(rate=3.0, queue_size=5)
    key = "lb_user"
    results = []
    for _ in range(8):
        r = lb.check(key)
        results.append("✓" if r.allowed else "✗")
    print(f"  8 immediate requests: {' '.join(results)}")
    print(f"  (queue fills to 5; extras denied; drains at 3/sec)")

    # ── Rate Limit Headers ────────────────────
    print("\n[4] HTTP RATE LIMIT HEADERS")
    print("─" * 55)

    sw  = SlidingWindowCounter(100, 60)
    key = "api-key-xyz"
    for _ in range(15):
        sw.check(key)
    result = sw.check(key)
    for header, value in result.headers().items():
        print(f"  {header}: {value}")

    # ── Multi-Tier ────────────────────────────
    print("\n[5] MULTI-TIER RATE LIMITER")
    print("─" * 55)

    tiers = [
        TierConfig("per_second", 10,     1),
        TierConfig("per_minute", 300,   60),
        TierConfig("per_day",    10000, 86400),
    ]
    multi = MultiTierRateLimiter(tiers)
    key   = "premium_user"

    # Normal rate (within all tiers)
    for _ in range(8):
        multi.check(key)
    r = multi.check(key)
    print(f"  After 9 requests (per-second limit=10):")
    print(f"    allowed={r.allowed}  remaining={r.remaining}")

    # Burst to hit per-second limit
    for _ in range(5):
        multi.check(key)
    r = multi.check(key)
    print(f"\n  After 14 requests in 1s (hits per_second limit=10):")
    print(f"    allowed={r.allowed}  limit={r.limit}  remaining={r.remaining}")

    # ── Fixed Window Boundary Bug ─────────────
    print("\n[6] FIXED WINDOW BOUNDARY PROBLEM")
    print("─" * 55)

    print("  Fixed window: limit=10 per minute")
    print("  t=0:59: user sends 10 requests → OK (window 0)")
    print("  t=1:01: user sends 10 requests → OK (window 1)")
    print("  Result: 20 requests in 2 seconds — 2× limit!")
    print("  Sliding window fixes this: estimate blends prev+curr windows")

    # ── Algorithm Comparison Table ─────────────
    print("\n[7] ALGORITHM COMPARISON")
    print("─" * 55)

    print(f"  {'Algorithm':<22} {'Accuracy':>10}  {'Space':>8}  {'Burst Support'}")
    print("  " + "─" * 60)
    comparison = [
        ("Fixed Window",        "~97%",   "O(1)",    "Yes (boundary bug)"),
        ("Sliding Window Log",  "100%",   "O(N)",    "No"),
        ("Sliding Window Ctr",  "~97%",   "O(1)",    "No"),
        ("Token Bucket",        "100%",   "O(1)",    "Yes (up to capacity)"),
        ("Leaky Bucket",        "100%",   "O(1)",    "No (smooths output)"),
    ]
    for name, accuracy, space, burst in comparison:
        print(f"  {name:<22} {accuracy:>10}  {space:>8}  {burst}")

    # ── Redis Distributed Implementation ──────
    print("\n[8] DISTRIBUTED RATE LIMITER (Redis)")
    print("─" * 55)

    notes = [
        ("Storage",      "Redis sorted set per key; ZADD + ZREMRANGEBYSCORE"),
        ("Atomicity",    "Lua script for read-modify-write (no race conditions)"),
        ("TTL",          "Set key TTL = window_s; auto-cleanup on expiry"),
        ("Cluster",      "Shard by user_id: hash_slot(user_id) → Redis shard"),
        ("Failopen",     "If Redis unavailable: allow request (don't block)"),
        ("Overhead",     "~1-2ms for Redis round trip; use local cache for <0.1ms"),
        ("Hot key",      "Celebrity user: local token bucket + Redis for sync"),
    ]
    for aspect, detail in notes:
        print(f"  {aspect:<14} {detail}")


if __name__ == "__main__":
    demonstrate_rate_limiter()
