"""
BACKOFF AND RETRY STRATEGIES
================================

Problem Statement:
Services fail transiently: network glitch, overloaded database, brief timeout.
Retrying immediately adds load to an already-struggling system → thundering herd.
Smart retry strategies absorb transient failures without making things worse.

Core Concepts:

  1. Exponential Backoff:
     delay = base * (2 ^ attempt)
     Each retry waits twice as long as the previous.
     Reduces load during extended outages. Gives service time to recover.

  2. Jitter:
     Add random component to backoff delay.
     Without jitter: all retrying clients sync their retries → periodic spikes.
     Full jitter: delay = random(0, base * 2^attempt)
     Equal jitter: delay = base*2^attempt/2 + random(0, base*2^attempt/2)
     Decorrelated jitter (best spread): delay = random(base, prev_delay * 3)

  3. Retry Budget / Max Attempts:
     Don't retry forever. Set max_attempts (3-5 typical).
     After max_attempts → fail fast, return error to caller.
     Prevents cascading failures from excessive retry load.

  4. Retry on Which Errors:
     Retry: timeout, 503, 429 (rate limited), 500 (transient server error).
     Don't retry: 400, 401, 403, 404, 409 (permanent/client errors).
     Never retry: 422 (invalid payload — will always fail).

  5. Circuit Breaker (complementary):
     After N failures → open circuit → fail fast without retrying.
     Retry only when circuit is half-open (probe for recovery).
     See 010_Circuit_Breaker_And_Bulkhead.py.

  6. Deadline Propagation:
     Each retry consumes time from the original request deadline.
     Don't retry if remaining deadline < estimated operation time.
     Prevents stacking retries past the point of usefulness.

  7. Idempotency First:
     Only retry operations that are idempotent (safe to repeat).
     Non-idempotent: use idempotency keys first (see 016_Idempotency).

AWS SDK, gRPC, and most production clients implement exponential backoff + jitter.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import random
import math
import threading


# ─────────────────────────────────────────────
# BACKOFF STRATEGIES
# ─────────────────────────────────────────────

class BackoffStrategy:
    """Pluggable delay calculator."""

    def delay(self, attempt: int) -> float:
        raise NotImplementedError


class FixedBackoff(BackoffStrategy):
    def __init__(self, delay_s: float):
        self._delay = delay_s

    def delay(self, attempt: int) -> float:
        return self._delay


class ExponentialBackoff(BackoffStrategy):
    def __init__(self, base_s: float = 0.1, multiplier: float = 2.0,
                 max_s: float = 60.0):
        self.base       = base_s
        self.multiplier = multiplier
        self.max_s      = max_s

    def delay(self, attempt: int) -> float:
        return min(self.max_s, self.base * (self.multiplier ** attempt))


class ExponentialBackoffWithJitter(BackoffStrategy):
    """Full jitter: delay = random(0, exponential_delay). AWS recommendation."""

    def __init__(self, base_s: float = 0.1, max_s: float = 60.0):
        self.base  = base_s
        self.max_s = max_s

    def delay(self, attempt: int) -> float:
        exp_delay = min(self.max_s, self.base * (2 ** attempt))
        return random.uniform(0, exp_delay)


class DecorrelatedJitterBackoff(BackoffStrategy):
    """
    Decorrelated jitter: spread retries better than full jitter.
    delay = random(base, prev_delay * 3)
    """

    def __init__(self, base_s: float = 0.1, max_s: float = 60.0):
        self.base  = base_s
        self.max_s = max_s
        self._prev = base_s

    def delay(self, attempt: int) -> float:
        new_delay  = random.uniform(self.base, self._prev * 3)
        self._prev = min(new_delay, self.max_s)
        return self._prev


# ─────────────────────────────────────────────
# RETRY POLICY
# ─────────────────────────────────────────────

@dataclass
class RetryResult:
    succeeded    : bool
    attempts     : int
    total_time_s : float
    last_error   : Optional[str] = None
    result       : Any           = None


class RetryPolicy:
    """
    Configurable retry policy: max_attempts, backoff, retryable errors.
    """

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 409, 422}

    def __init__(self, max_attempts: int = 3,
                 backoff: BackoffStrategy = None,
                 retryable_exceptions: Tuple = (IOError, TimeoutError),
                 deadline_s: Optional[float] = None):
        self.max_attempts       = max_attempts
        self.backoff            = backoff or ExponentialBackoffWithJitter()
        self.retryable_exceptions = retryable_exceptions
        self.deadline_s         = deadline_s

    def is_retryable_error(self, exc: Exception) -> bool:
        if isinstance(exc, HttpError):
            return exc.status_code in self.RETRYABLE_STATUS_CODES
        return isinstance(exc, self.retryable_exceptions)

    def execute(self, fn: Callable, *args, **kwargs) -> RetryResult:
        t0      = time.time()
        attempt = 0
        last_error = None

        while attempt < self.max_attempts:
            # Check deadline
            if self.deadline_s and (time.time() - t0) > self.deadline_s:
                return RetryResult(
                    succeeded=False, attempts=attempt,
                    total_time_s=time.time() - t0,
                    last_error="deadline_exceeded",
                )

            attempt += 1
            try:
                result = fn(*args, **kwargs)
                return RetryResult(
                    succeeded=True, attempts=attempt,
                    total_time_s=time.time() - t0,
                    result=result,
                )
            except Exception as exc:
                last_error = str(exc)
                if not self.is_retryable_error(exc):
                    return RetryResult(
                        succeeded=False, attempts=attempt,
                        total_time_s=time.time() - t0,
                        last_error=f"non_retryable: {exc}",
                    )
                if attempt < self.max_attempts:
                    delay = self.backoff.delay(attempt - 1)
                    time.sleep(delay)

        return RetryResult(
            succeeded=False, attempts=attempt,
            total_time_s=time.time() - t0,
            last_error=last_error,
        )


class HttpError(Exception):
    def __init__(self, status_code: int, message: str = ""):
        super().__init__(message)
        self.status_code = status_code


# ─────────────────────────────────────────────
# THUNDERING HERD SIMULATION
# ─────────────────────────────────────────────

class RetrySimulator:
    """
    Simulates N clients retrying after a failure.
    Shows how jitter spreads retries vs simultaneous spikes without jitter.
    """

    def __init__(self, n_clients: int):
        self.n_clients   = n_clients
        self._retry_times: List[float] = []
        self._lock       = threading.Lock()

    def simulate(self, strategy: BackoffStrategy, n_retries: int = 3) -> Dict:
        """Returns retry distribution per second bucket."""
        base_time = 1000.0   # abstract time
        self._retry_times.clear()

        for _ in range(self.n_clients):
            t = base_time
            for attempt in range(n_retries):
                t += strategy.delay(attempt)
                with self._lock:
                    self._retry_times.append(t - base_time)   # offset from start

        # Bucket by second
        buckets: Dict[int, int] = {}
        for t in self._retry_times:
            bucket = int(t)
            buckets[bucket] = buckets.get(bucket, 0) + 1

        return dict(sorted(buckets.items()))


# ─────────────────────────────────────────────
# RETRY BUDGET
# ─────────────────────────────────────────────

class RetryBudget:
    """
    Limits the total retry rate to a fraction of normal request rate.
    Prevents retry storms from cascading.
    """

    def __init__(self, budget_fraction: float = 0.1, window_s: float = 1.0):
        self.budget_fraction = budget_fraction
        self.window_s        = window_s
        self._requests       : List[float] = []
        self._retries        : List[float] = []
        self._lock           = threading.Lock()

    def record_request(self):
        with self._lock:
            self._requests.append(time.time())

    def can_retry(self) -> bool:
        with self._lock:
            now    = time.time()
            cutoff = now - self.window_s
            self._requests = [t for t in self._requests if t > cutoff]
            self._retries  = [t for t in self._retries  if t > cutoff]
            max_retries = max(1, int(len(self._requests) * self.budget_fraction))
            if len(self._retries) < max_retries:
                self._retries.append(now)
                return True
            return False


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_retry_strategies():
    print("=" * 65)
    print("BACKOFF AND RETRY STRATEGIES")
    print("=" * 65)

    random.seed(42)

    # ── Backoff Delay Profiles ────────────────────
    print("\n[1] BACKOFF DELAY PROFILES (attempt 0-4)")
    print("─" * 55)

    strategies = [
        ("Fixed (100ms)",     FixedBackoff(0.1)),
        ("Exponential",       ExponentialBackoff(base_s=0.1, max_s=10.0)),
        ("Exp + Full Jitter", ExponentialBackoffWithJitter(base_s=0.1, max_s=10.0)),
        ("Decorrelated Jitter", DecorrelatedJitterBackoff(base_s=0.1, max_s=10.0)),
    ]

    print(f"  {'Strategy':<24} " + "  ".join(f"{'att '+str(i):<8}" for i in range(5)))
    print(f"  {'─'*65}")
    for name, strategy in strategies:
        delays = [strategy.delay(i) for i in range(5)]
        delay_str = "  ".join(f"{d*1000:>6.1f}ms" for d in delays)
        print(f"  {name:<24} {delay_str}")

    # ── Retry Policy: Retryable vs Non-Retryable ──
    print("\n\n[2] RETRY POLICY — RETRYABLE vs PERMANENT ERRORS")
    print("─" * 55)

    policy = RetryPolicy(
        max_attempts = 3,
        backoff      = ExponentialBackoffWithJitter(base_s=0.001, max_s=0.1),
    )

    # Transient failure (500) → retries
    attempt_count = [0]
    def flaky_service():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise HttpError(503, "Service temporarily unavailable")
        return "OK"

    result = policy.execute(flaky_service)
    print(f"  Transient error (503): succeeded={result.succeeded} "
          f"attempts={result.attempts} time={result.total_time_s*1000:.1f}ms")

    # Permanent failure (400) → no retry
    def invalid_request():
        raise HttpError(400, "Bad request")

    result2 = policy.execute(invalid_request)
    print(f"  Permanent error (400): succeeded={result2.succeeded} "
          f"attempts={result2.attempts} (1 = no retry)")

    # Max attempts exceeded
    def always_fails():
        raise HttpError(503, "Still down")

    result3 = policy.execute(always_fails)
    print(f"  Max retries (503 always): succeeded={result3.succeeded} "
          f"attempts={result3.attempts}")

    # ── Thundering Herd Comparison ────────────────
    print("\n\n[3] THUNDERING HERD — NO JITTER vs JITTER")
    print("─" * 55)

    sim = RetrySimulator(n_clients=20)

    no_jitter_dist = sim.simulate(ExponentialBackoff(base_s=0.1, max_s=10.0))
    jitter_dist    = sim.simulate(ExponentialBackoffWithJitter(base_s=0.1, max_s=10.0))

    print(f"  20 clients retry after failure (exponential base=100ms):")
    print(f"  {'Second':<10} {'No Jitter':<30} {'With Jitter'}")
    print(f"  {'─'*55}")
    all_buckets = sorted(set(list(no_jitter_dist.keys()) + list(jitter_dist.keys())))
    for bucket in all_buckets[:8]:
        nj = no_jitter_dist.get(bucket, 0)
        j  = jitter_dist.get(bucket, 0)
        nj_bar = "█" * nj
        j_bar  = "█" * j
        print(f"  {bucket}s{'':<8} {nj_bar:<30} {j_bar} ({nj} vs {j})")

    print(f"  → Jitter spreads retries; no jitter creates synchronized spikes")

    # ── Retry Budget ──────────────────────────────
    print("\n\n[4] RETRY BUDGET — CAPPING RETRY RATE")
    print("─" * 55)

    budget = RetryBudget(budget_fraction=0.1, window_s=0.5)

    # Simulate 50 requests
    for _ in range(50):
        budget.record_request()

    allowed = sum(1 for _ in range(20) if budget.can_retry())
    print(f"  50 requests, retry_budget=10%, window=500ms")
    print(f"  Retries allowed (of 20 attempted): {allowed}/{20} "
          f"(~{50*0.1:.0f} max)")
    print(f"  → Prevents retry storm from exceeding {int(50*0.1)} retries/window")

    # ── Retry Decision Guide ──────────────────────
    print("\n\n[5] RETRY DECISION GUIDE")
    print("─" * 55)
    rows = [
        ("HTTP 429 (rate limit)",  "Yes",  "Retry after Retry-After header delay"),
        ("HTTP 503 (unavailable)", "Yes",  "Retry with exponential backoff"),
        ("HTTP 500 (server error)","Conditional", "Retry if idempotent operation"),
        ("HTTP 400 (bad request)", "No",   "Client error — will always fail"),
        ("HTTP 401 (unauthorized)","No",   "Fix auth before retrying"),
        ("HTTP 404 (not found)",   "No",   "Resource gone — retry won't help"),
        ("Network timeout",        "Yes",  "Retry with backoff"),
        ("Connection refused",     "Yes",  "Server restarting — retry"),
        ("SSL/TLS error",          "No",   "Config/cert issue — retry useless"),
    ]
    print(f"  {'Error':<30} {'Retry?':<14} {'Guidance'}")
    print(f"  {'─'*72}")
    for error, should_retry, guidance in rows:
        print(f"  {error:<30} {should_retry:<14} {guidance}")

    print("\n\n[6] RETRY STRATEGY SELECTION")
    print("─" * 55)
    selections = [
        ("Single client, few callers", "Fixed backoff (simpler)"),
        ("Many clients, shared resource", "Exponential + full jitter"),
        ("Real-time, latency sensitive", "Max 1-2 retries, tight deadline"),
        ("Background job, eventual", "Decorrelated jitter, high max retries"),
        ("Complex workflow", "Retry at step level + saga compensation"),
    ]
    for scenario, strategy in selections:
        print(f"  {scenario:<36} → {strategy}")


if __name__ == "__main__":
    demonstrate_retry_strategies()
