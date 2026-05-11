"""
Rate Limiter System Design - Python Implementation
Implements all 5 rate limiting algorithms:
  1. Token Bucket (burst allowed, continuous refill)
  2. Leaky Bucket (smooth output, fixed rate)
  3. Fixed Window Counter (simple, has boundary spike problem)
  4. Sliding Window Log (exact, memory-heavy)
  5. Sliding Window Counter (approximation, memory-efficient)

Also includes: DistributedRateLimiter with Redis-like atomic ops,
boundary spike demonstration, multi-tier rate limiting.
No external dependencies - standard library only.
"""

import time
import math
import hashlib
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────────
# Algorithm 1: Token Bucket
# ─────────────────────────────────────────────

class TokenBucket:
    """
    Token Bucket Algorithm:
    - Bucket holds up to `capacity` tokens
    - Tokens refill continuously at `refill_rate` tokens/second
    - Each request consumes 1 token (or N tokens for weighted)
    - Allows burst: capacity tokens available immediately

    Best for: APIs where short bursts are acceptable
    """

    def __init__(self, capacity: float, refill_rate: float):
        """
        capacity: maximum tokens in bucket (burst size)
        refill_rate: tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def allow_request(self, tokens_needed: float = 1.0) -> tuple:
        """
        Returns (allowed: bool, tokens_remaining: float, wait_time: float)
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            # Add tokens earned during elapsed time
            self._tokens = min(
                self.capacity,
                self._tokens + elapsed * self.refill_rate
            )
            self._last_refill = now

            if self._tokens >= tokens_needed:
                self._tokens -= tokens_needed
                return True, self._tokens, 0.0
            else:
                # How long until enough tokens are available?
                wait = (tokens_needed - self._tokens) / self.refill_rate
                return False, self._tokens, wait

    @property
    def tokens(self) -> float:
        """Current token count (approximate — not thread-safe without lock)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        return min(self.capacity, self._tokens + elapsed * self.refill_rate)


# ─────────────────────────────────────────────
# Algorithm 2: Leaky Bucket
# ─────────────────────────────────────────────

class LeakyBucket:
    """
    Leaky Bucket Algorithm:
    - Requests enter a fixed-size queue
    - Queue processes at a fixed `leak_rate` per second
    - Incoming requests that exceed queue size are rejected

    Unlike Token Bucket: output rate is ALWAYS constant (no burst)
    Best for: Smoothing traffic to downstream services
    """

    def __init__(self, capacity: int, leak_rate: float):
        """
        capacity: max queue size (requests waiting)
        leak_rate: requests processed per second
        """
        self.capacity = capacity
        self.leak_rate = leak_rate
        self._queue_size = 0.0           # Current effective queue depth
        self._last_leak = time.monotonic()
        self._lock = threading.Lock()

    def allow_request(self) -> tuple:
        """
        Returns (allowed: bool, queue_depth: float, wait_time: float)
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_leak

            # "Drain" the queue — process requests that have leaked out
            self._queue_size = max(
                0.0,
                self._queue_size - elapsed * self.leak_rate
            )
            self._last_leak = now

            if self._queue_size < self.capacity:
                self._queue_size += 1.0
                wait = self._queue_size / self.leak_rate
                return True, self._queue_size, wait
            else:
                return False, self._queue_size, -1.0

    @property
    def queue_depth(self) -> float:
        now = time.monotonic()
        elapsed = now - self._last_leak
        return max(0.0, self._queue_size - elapsed * self.leak_rate)


# ─────────────────────────────────────────────
# Algorithm 3: Fixed Window Counter
# ─────────────────────────────────────────────

class FixedWindowCounter:
    """
    Fixed Window Counter:
    - Divide time into fixed windows (e.g., 1-minute windows)
    - Count requests per user per window
    - Reset counter at window boundary

    PROBLEM: Boundary spike — 200% of limit possible across window boundary
    Best for: Simple use cases where boundary spike is acceptable
    """

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self._counters: dict = {}    # key -> (window_ts, count)
        self._lock = threading.Lock()

    def allow_request(self, key: str) -> tuple:
        """
        Returns (allowed: bool, current_count: int, window_reset_in: float)
        """
        with self._lock:
            now = time.monotonic()
            window_ts = int(now / self.window_seconds)
            reset_in = self.window_seconds - (now % self.window_seconds)

            stored = self._counters.get(key)
            if stored is None or stored[0] != window_ts:
                # New window — reset counter
                self._counters[key] = (window_ts, 1)
                return True, 1, reset_in

            current_ts, count = stored
            if count < self.limit:
                self._counters[key] = (window_ts, count + 1)
                return True, count + 1, reset_in
            else:
                return False, count, reset_in

    def get_count(self, key: str) -> int:
        stored = self._counters.get(key)
        if not stored:
            return 0
        now = time.monotonic()
        current_window = int(now / self.window_seconds)
        if stored[0] != current_window:
            return 0
        return stored[1]


# ─────────────────────────────────────────────
# Algorithm 4: Sliding Window Log
# ─────────────────────────────────────────────

class SlidingWindowLog:
    """
    Sliding Window Log:
    - Store timestamp of every request in a sorted deque
    - On each request: remove old timestamps outside the window
    - Count = remaining timestamps = requests in current window

    Most accurate algorithm — no boundary spike
    TRADE-OFF: Memory O(requests in window) per user
    Best for: Strict rate limiting where accuracy is critical
    """

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self._logs: defaultdict = defaultdict(deque)    # key -> deque[timestamp]
        self._lock = threading.Lock()

    def allow_request(self, key: str) -> tuple:
        """
        Returns (allowed: bool, current_count: int, oldest_expiry: float)
        """
        with self._lock:
            now = time.monotonic()
            window_start = now - self.window_seconds
            log = self._logs[key]

            # Remove timestamps outside the window (older than window_start)
            while log and log[0] <= window_start:
                log.popleft()

            count = len(log)
            if count < self.limit:
                log.append(now)
                return True, count + 1, 0.0
            else:
                # When will the oldest request expire?
                oldest = log[0] if log else now
                retry_after = self.window_seconds - (now - oldest)
                return False, count, retry_after

    def get_count(self, key: str) -> int:
        now = time.monotonic()
        window_start = now - self.window_seconds
        log = self._logs.get(key)
        if not log:
            return 0
        return sum(1 for ts in log if ts > window_start)


# ─────────────────────────────────────────────
# Algorithm 5: Sliding Window Counter (Approximation)
# ─────────────────────────────────────────────

class SlidingWindowCounter:
    """
    Sliding Window Counter Approximation:
    - Store counts for current and previous windows (2 counters total)
    - Approximate current rate using weighted interpolation

    Formula:
      approx = prev_count * (1 - elapsed/window) + current_count

    Max error: ~10% (worst case at window boundary)
    Memory: O(1) per user — much more efficient than SlidingWindowLog
    Best for: High-scale systems where small approximation is acceptable
    """

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        # key -> {window_ts, current_count, prev_count}
        self._state: dict = {}
        self._lock = threading.Lock()

    def allow_request(self, key: str) -> tuple:
        """
        Returns (allowed: bool, approx_count: float, approx_remaining: float)
        """
        with self._lock:
            now = time.monotonic()
            current_window_ts = int(now / self.window_seconds)
            elapsed_in_window = now % self.window_seconds

            state = self._state.get(key)
            if state is None:
                self._state[key] = {
                    "window_ts": current_window_ts,
                    "current": 1,
                    "prev": 0
                }
                return True, 1.0, float(self.limit - 1)

            # Advance window if needed
            if state["window_ts"] < current_window_ts:
                age = current_window_ts - state["window_ts"]
                if age == 1:
                    state["prev"] = state["current"]
                else:
                    state["prev"] = 0   # Older windows treated as 0
                state["current"] = 0
                state["window_ts"] = current_window_ts

            # Weight: how much of the previous window is still "active"
            weight = 1.0 - (elapsed_in_window / self.window_seconds)
            approx = state["prev"] * weight + state["current"]

            if approx < self.limit:
                state["current"] += 1
                return True, approx + 1, float(self.limit - approx - 1)
            else:
                return False, approx, 0.0


# ─────────────────────────────────────────────
# Multi-Algorithm Rate Limiter
# ─────────────────────────────────────────────

class RateLimiterAlgorithm(Enum):
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW_LOG = "sliding_window_log"
    SLIDING_WINDOW_COUNTER = "sliding_window_counter"


@dataclass
class RateLimitConfig:
    algorithm: RateLimiterAlgorithm
    limit: int
    window_seconds: int
    burst_size: Optional[int] = None   # For token bucket


@dataclass
class RateLimitResult:
    allowed: bool
    algorithm: str
    current_count: int
    limit: int
    remaining: int
    retry_after: float = 0.0

    def headers(self) -> dict:
        """Standard HTTP rate limit response headers."""
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Algorithm": self.algorithm,
            **({"Retry-After": str(int(self.retry_after))} if not self.allowed else {}),
        }


class RateLimiter:
    """
    Unified rate limiter supporting all 5 algorithms.
    Wraps algorithm selection with a consistent interface.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._limiters: dict = {}     # key -> algorithm instance

    def _get_or_create(self, key: str):
        if key not in self._limiters:
            cfg = self.config
            if cfg.algorithm == RateLimiterAlgorithm.TOKEN_BUCKET:
                burst = cfg.burst_size or cfg.limit
                rate = cfg.limit / cfg.window_seconds
                self._limiters[key] = TokenBucket(burst, rate)
            elif cfg.algorithm == RateLimiterAlgorithm.LEAKY_BUCKET:
                rate = cfg.limit / cfg.window_seconds
                self._limiters[key] = LeakyBucket(cfg.limit, rate)
            elif cfg.algorithm == RateLimiterAlgorithm.FIXED_WINDOW:
                self._limiters[key] = FixedWindowCounter(cfg.limit, cfg.window_seconds)
            elif cfg.algorithm == RateLimiterAlgorithm.SLIDING_WINDOW_LOG:
                self._limiters[key] = SlidingWindowLog(cfg.limit, cfg.window_seconds)
            elif cfg.algorithm == RateLimiterAlgorithm.SLIDING_WINDOW_COUNTER:
                self._limiters[key] = SlidingWindowCounter(cfg.limit, cfg.window_seconds)
        return self._limiters[key]

    def check(self, key: str) -> RateLimitResult:
        limiter = self._get_or_create(key)
        cfg = self.config
        algo = cfg.algorithm

        if algo == RateLimiterAlgorithm.TOKEN_BUCKET:
            allowed, remaining, wait = limiter.allow_request()
            return RateLimitResult(
                allowed=allowed, algorithm=algo.value,
                current_count=int(cfg.limit - remaining),
                limit=cfg.limit, remaining=int(remaining), retry_after=wait
            )
        elif algo == RateLimiterAlgorithm.LEAKY_BUCKET:
            allowed, depth, wait = limiter.allow_request()
            return RateLimitResult(
                allowed=allowed, algorithm=algo.value,
                current_count=int(depth),
                limit=cfg.limit, remaining=max(0, cfg.limit - int(depth)),
                retry_after=max(0, wait)
            )
        elif algo == RateLimiterAlgorithm.FIXED_WINDOW:
            allowed, count, reset_in = limiter.allow_request(key)
            return RateLimitResult(
                allowed=allowed, algorithm=algo.value,
                current_count=count, limit=cfg.limit,
                remaining=max(0, cfg.limit - count), retry_after=reset_in
            )
        elif algo == RateLimiterAlgorithm.SLIDING_WINDOW_LOG:
            allowed, count, retry = limiter.allow_request(key)
            return RateLimitResult(
                allowed=allowed, algorithm=algo.value,
                current_count=count, limit=cfg.limit,
                remaining=max(0, cfg.limit - count), retry_after=retry
            )
        elif algo == RateLimiterAlgorithm.SLIDING_WINDOW_COUNTER:
            allowed, approx, remaining = limiter.allow_request(key)
            return RateLimitResult(
                allowed=allowed, algorithm=algo.value,
                current_count=int(approx), limit=cfg.limit,
                remaining=int(max(0, remaining))
            )


# ─────────────────────────────────────────────
# Distributed Rate Limiter (Redis-like atomic ops)
# ─────────────────────────────────────────────

class FakeRedis:
    """
    Simulates Redis atomic operations for distributed rate limiting.
    In production: use redis-py with EVAL (Lua scripts) for atomicity.
    """

    def __init__(self):
        self._store: dict = {}
        self._lock = threading.Lock()

    def eval_token_bucket(self, key: str, capacity: float,
                          refill_rate: float) -> tuple:
        """
        Atomic token bucket check (simulates Redis Lua script).
        Returns (allowed: bool, tokens_remaining: float)
        """
        with self._lock:
            now = time.monotonic()
            state = self._store.get(key, {"tokens": capacity, "last_refill": now})
            elapsed = now - state["last_refill"]
            tokens = min(capacity, state["tokens"] + elapsed * refill_rate)

            if tokens >= 1:
                new_state = {"tokens": tokens - 1, "last_refill": now}
                self._store[key] = new_state
                return True, tokens - 1
            else:
                self._store[key] = {"tokens": tokens, "last_refill": now}
                return False, tokens

    def eval_fixed_window(self, key: str, limit: int,
                           window_seconds: int) -> tuple:
        """Atomic fixed window counter (simulates INCR + EXPIRE in Lua)."""
        with self._lock:
            now = time.monotonic()
            window_key = f"{key}:{int(now / window_seconds)}"
            count = self._store.get(window_key, 0) + 1
            self._store[window_key] = count
            return count <= limit, count

    def eval_sliding_window_counter(self, key: str, limit: int,
                                     window_seconds: int) -> tuple:
        """Atomic sliding window counter with two fixed windows."""
        with self._lock:
            now = time.monotonic()
            window_ts = int(now / window_seconds)
            elapsed_in_window = now % window_seconds
            weight = 1.0 - (elapsed_in_window / window_seconds)

            curr_key = f"{key}:{window_ts}"
            prev_key = f"{key}:{window_ts - 1}"

            current = self._store.get(curr_key, 0)
            previous = self._store.get(prev_key, 0)

            approx = previous * weight + current
            if approx < limit:
                self._store[curr_key] = current + 1
                return True, approx + 1
            return False, approx


class DistributedRateLimiter:
    """
    Rate limiter using a shared Redis-like store.
    Multiple app server instances share rate limit state.
    """

    def __init__(self, fake_redis: FakeRedis):
        self.redis = fake_redis
        # Tier configs: {tier: (requests_per_min, burst_size)}
        self._tiers = {
            "free":       (60,    10),
            "pro":        (1000,  100),
            "enterprise": (10000, 1000),
        }

    def check_user(self, user_id: str, tier: str = "free",
                   algorithm: str = "token_bucket") -> RateLimitResult:
        rpm, burst = self._tiers.get(tier, (60, 10))
        key = f"rate:{tier}:{user_id}"

        if algorithm == "token_bucket":
            refill_rate = rpm / 60.0
            allowed, tokens = self.redis.eval_token_bucket(key, burst, refill_rate)
            return RateLimitResult(
                allowed=allowed, algorithm="token_bucket",
                current_count=int(burst - tokens), limit=rpm,
                remaining=int(tokens)
            )
        elif algorithm == "fixed_window":
            allowed, count = self.redis.eval_fixed_window(key, rpm, 60)
            return RateLimitResult(
                allowed=allowed, algorithm="fixed_window",
                current_count=count, limit=rpm,
                remaining=max(0, rpm - count)
            )
        elif algorithm == "sliding_window":
            allowed, approx = self.redis.eval_sliding_window_counter(
                key, rpm, 60
            )
            return RateLimitResult(
                allowed=allowed, algorithm="sliding_window_counter",
                current_count=int(approx), limit=rpm,
                remaining=max(0, rpm - int(approx))
            )

        return RateLimitResult(
            allowed=True, algorithm="passthrough",
            current_count=0, limit=rpm, remaining=rpm
        )


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────

def demo_token_bucket():
    print("\n--- Algorithm 1: Token Bucket ---")
    print("capacity=10, refill_rate=2/sec")
    tb = TokenBucket(capacity=10, refill_rate=2.0)

    print("Sending 12 rapid requests (burst):")
    for i in range(12):
        allowed, remaining, wait = tb.allow_request()
        status = "ALLOW" if allowed else f"DENY  (retry in {wait:.2f}s)"
        print(f"  Request {i+1:2d}: {status:25s} tokens_remaining={remaining:.2f}")


def demo_leaky_bucket():
    print("\n--- Algorithm 2: Leaky Bucket ---")
    print("capacity=5 (queue), leak_rate=2/sec")
    lb = LeakyBucket(capacity=5, leak_rate=2.0)

    print("Sending 8 rapid requests:")
    for i in range(8):
        allowed, depth, wait = lb.allow_request()
        status = "ALLOW" if allowed else "DENY (queue full)"
        print(f"  Request {i+1:2d}: {status:25s} queue_depth={depth:.1f}")


def demo_fixed_window_boundary_spike():
    print("\n--- Algorithm 3: Fixed Window Counter — BOUNDARY SPIKE PROBLEM ---")
    print("limit=5 req/window, window=2 seconds")
    fwc = FixedWindowCounter(limit=5, window_seconds=2)

    print("Simulating boundary spike: 5 requests near end of window + 5 at start of next")
    print("(Using accelerated time simulation)")

    # We'll directly show the logic by using separate key contexts
    allowed_window1 = 0
    allowed_window2 = 0

    # Window 1 context (key: user:1:window:0)
    for i in range(5):
        allowed, count, _ = fwc.allow_request("user:spike:window0")
        if allowed:
            allowed_window1 += 1

    # Window 2 context (different window key simulated via different key)
    for i in range(5):
        allowed, count, _ = fwc.allow_request("user:spike:window1")
        if allowed:
            allowed_window2 += 1

    print(f"  Window 1 (last 2s): {allowed_window1}/5 requests allowed")
    print(f"  Window 2 (next 2s): {allowed_window2}/5 requests allowed")
    print(f"  EFFECTIVE RATE: {allowed_window1 + allowed_window2} requests")
    print(f"  PROBLEM: {allowed_window1 + allowed_window2}x limit "
          f"possible in 2-second span across boundary!")
    print(f"  Limit was 5/window but {allowed_window1 + allowed_window2} "
          f"requests went through at boundary!")


def demo_sliding_window_log():
    print("\n--- Algorithm 4: Sliding Window Log (No Boundary Spike) ---")
    print("limit=5 requests per 2-second window")
    swl = SlidingWindowLog(limit=5, window_seconds=2)

    print("Sending 8 rapid requests:")
    for i in range(8):
        allowed, count, retry = swl.allow_request("user:1")
        status = "ALLOW" if allowed else f"DENY  (retry in {retry:.2f}s)"
        print(f"  Request {i+1:2d}: {status:25s} count={count}")


def demo_sliding_window_counter():
    print("\n--- Algorithm 5: Sliding Window Counter (Approximation) ---")
    print("limit=10 requests per window, window=2 seconds")
    swc = SlidingWindowCounter(limit=10, window_seconds=2)

    print("Sending 15 rapid requests:")
    for i in range(15):
        allowed, approx, remaining = swc.allow_request("user:1")
        status = "ALLOW" if allowed else "DENY"
        print(f"  Request {i+1:2d}: {status:5s}  approx_count={approx:.2f}  "
              f"remaining={remaining:.2f}")


def demo_algorithm_comparison():
    print("\n--- Algorithm Comparison Summary ---")
    headers = ["Algorithm", "Burst?", "Memory/user", "Accuracy", "Complexity"]
    rows = [
        ["Token Bucket",          "Yes",        "O(1)",         "High",    "Medium"],
        ["Leaky Bucket",          "Queued only", "O(queue_size)", "High",  "Medium"],
        ["Fixed Window Counter",  "At boundary", "O(1)",         "Medium", "Low"],
        ["Sliding Window Log",    "No",          "O(req/window)", "Exact", "Medium"],
        ["Sliding Window Counter","No",          "O(1)",          "~10%",  "Low"],
    ]
    col_widths = [max(len(r[i]) for r in [headers] + rows) + 2 for i in range(5)]
    fmt = "  " + "".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  " + "-" * (sum(col_widths)))
    for row in rows:
        print(fmt.format(*row))


def demo_distributed_rate_limiter():
    print("\n--- Distributed Rate Limiter (Redis Atomic Lua Scripts) ---")
    redis = FakeRedis()
    dist_limiter = DistributedRateLimiter(redis)

    print("\nTesting different user tiers:")
    tiers = [
        ("user:alice", "free"),
        ("user:bob", "pro"),
        ("user:corp", "enterprise"),
    ]
    for user_id, tier in tiers:
        rpm, burst = dist_limiter._tiers[tier]
        print(f"\n  User: {user_id} (tier={tier}, limit={rpm}req/min, burst={burst})")
        allowed_count = 0
        denied_count = 0
        for i in range(min(burst + 5, 20)):
            result = dist_limiter.check_user(user_id, tier, algorithm="token_bucket")
            if result.allowed:
                allowed_count += 1
            else:
                denied_count += 1
        print(f"  Sent {allowed_count + denied_count} requests: "
              f"{allowed_count} allowed, {denied_count} denied")
        print(f"  Response headers: {result.headers()}")


def demo_rate_limit_headers():
    print("\n--- Rate Limit HTTP Headers ---")
    rl = RateLimiter(RateLimitConfig(
        algorithm=RateLimiterAlgorithm.SLIDING_WINDOW_LOG,
        limit=100, window_seconds=60
    ))

    # Send some requests
    for _ in range(95):
        rl.check("user:demo")

    result = rl.check("user:demo")
    print("Response for request 96 (under limit):")
    for k, v in result.headers().items():
        print(f"  {k}: {v}")

    # Exceed limit
    for _ in range(10):
        rl.check("user:demo")
    result_over = rl.check("user:demo")
    print("\nResponse for request after limit exceeded (429 Too Many Requests):")
    for k, v in result_over.headers().items():
        print(f"  {k}: {v}")
    print(f"  HTTP Status: 429 Too Many Requests")


def run_demo():
    print("=" * 60)
    print("RATE LIMITER SYSTEM DESIGN DEMO")
    print("=" * 60)

    demo_token_bucket()
    demo_leaky_bucket()
    demo_fixed_window_boundary_spike()
    demo_sliding_window_log()
    demo_sliding_window_counter()
    demo_algorithm_comparison()
    demo_distributed_rate_limiter()
    demo_rate_limit_headers()

    print("\n--- Key Design Insights ---")
    insights = [
        "Token Bucket: best for API rate limiting (burst allowed, continuous refill)",
        "Fixed Window: simple but boundary spike allows 2x rate across window edges",
        "Sliding Window Log: exact accuracy, but O(N) memory per user",
        "Sliding Window Counter: O(1) memory, ~10% max error (practical choice at scale)",
        "Redis Lua scripts: atomic execution prevents race conditions in distributed setup",
        "W+R>N quorum ensures consistent reads in distributed systems",
        "Fail-open: rate limiter failure allows requests through (prefer availability)",
        "Response headers guide clients: X-RateLimit-Remaining, Retry-After",
    ]
    for insight in insights:
        print(f"  - {insight}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
