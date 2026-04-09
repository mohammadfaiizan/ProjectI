"""
FAULT TOLERANCE AND RESILIENCE
================================

Problem Statement:
In large-scale distributed systems, failures are not exceptional — they are
expected. Hardware fails, networks partition, processes crash. A resilient
system anticipates failures and degrades gracefully rather than failing
catastrophically.

Key Patterns:
- Redundancy         : Multiple copies of critical components
- Retry with backoff : Automatic retry with increasing delay
- Circuit Breaker    : Stop calling a failing service to allow recovery
- Bulkhead           : Isolate failures to one pool/partition
- Graceful Degradation: Return partial/cached results instead of errors
- Timeout            : Don't wait forever; fail fast

Failure Taxonomy:
- Crash failure    : Node stops and does not resume
- Omission failure : Node drops messages
- Timing failure   : Node responds too slowly
- Byzantine failure: Node behaves arbitrarily (most complex)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import random
import time


class FailureType(Enum):
    CRASH             = "crash"
    NETWORK_PARTITION = "network_partition"
    SLOW_RESPONSE     = "slow_response"
    MEMORY_OOM        = "memory_oom"
    DISK_FULL         = "disk_full"


class CircuitState(Enum):
    CLOSED    = "CLOSED"     # Normal: requests pass through
    OPEN      = "OPEN"       # Failing: requests fast-fail
    HALF_OPEN = "HALF_OPEN"  # Testing: let one request through


@dataclass
class FailureScenario:
    failure_type    : FailureType
    probability     : float    # 0.0 – 1.0
    recovery_time_s : float


# ─────────────────────────────────────────────
# RETRY WITH BACKOFF
# ─────────────────────────────────────────────

class ExponentialBackoff:
    """Retry with exponential backoff + jitter (AWS recommendation)."""

    def __init__(self, base_ms: float = 100, max_ms: float = 30_000,
                 max_retries: int = 5, jitter: bool = True):
        self.base_ms    = base_ms
        self.max_ms     = max_ms
        self.max_retries= max_retries
        self.jitter     = jitter

    def delay_ms(self, attempt: int) -> float:
        delay = min(self.base_ms * (2 ** attempt), self.max_ms)
        if self.jitter:
            delay = random.uniform(0, delay)
        return delay

    def execute(self, fn: Callable, label: str = "operation") -> Optional[any]:
        for attempt in range(self.max_retries + 1):
            try:
                result = fn()
                if attempt > 0:
                    print(f"  ✅ [{label}] Succeeded on attempt {attempt + 1}")
                return result
            except Exception as e:
                if attempt == self.max_retries:
                    print(f"  ❌ [{label}] Failed after {self.max_retries + 1} attempts: {e}")
                    return None
                delay = self.delay_ms(attempt)
                print(f"  ↺  [{label}] Attempt {attempt + 1} failed ({e}). "
                      f"Retrying in {delay:.0f}ms…")
        return None


# ─────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────

class CircuitBreaker:
    """
    Three-state circuit breaker:
      CLOSED    → requests pass through normally
      OPEN      → requests fast-fail without hitting the service
      HALF_OPEN → one test request; close on success, reopen on failure
    """

    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout_s: float = 5.0):
        self.name              = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout  = recovery_timeout_s
        self.state             = CircuitState.CLOSED
        self.failure_count     = 0
        self.last_failure_time : float = 0.0
        self.requests_blocked  = 0
        self.state_history     : List[str] = ["CLOSED"]

    def _transition(self, new_state: CircuitState):
        if new_state != self.state:
            print(f"  [CB:{self.name}] {self.state.value} → {new_state.value}")
            self.state = new_state
            self.state_history.append(new_state.value)

    def call(self, fn: Callable) -> Optional[any]:
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)
            else:
                self.requests_blocked += 1
                raise Exception(f"Circuit OPEN — fast-fail ({self.name})")

        try:
            result = fn()
            if self.state == CircuitState.HALF_OPEN:
                self.failure_count = 0
                self._transition(CircuitState.CLOSED)
            return result
        except Exception as e:
            self.failure_count    += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self._transition(CircuitState.OPEN)
            raise


# ─────────────────────────────────────────────
# BULKHEAD
# ─────────────────────────────────────────────

class BulkheadPool:
    """
    Thread-pool bulkhead: each downstream service gets a fixed concurrency budget.
    If one service is slow, it cannot exhaust all threads.
    """

    def __init__(self, name: str, max_concurrent: int):
        self.name           = name
        self.max_concurrent = max_concurrent
        self.current        = 0
        self.rejected       = 0
        self.completed      = 0

    def acquire(self) -> bool:
        if self.current >= self.max_concurrent:
            self.rejected += 1
            return False
        self.current += 1
        return True

    def release(self):
        self.current = max(0, self.current - 1)
        self.completed += 1

    def stats(self):
        print(f"  Bulkhead [{self.name}]: "
              f"current={self.current}  "
              f"completed={self.completed}  "
              f"rejected={self.rejected}")


# ─────────────────────────────────────────────
# FAULT TOLERANT SYSTEM
# ─────────────────────────────────────────────

class FaultTolerantSystem:
    """Simulates a system applying multiple resilience patterns."""

    def __init__(self, failure_rate: float = 0.1):
        self.failure_rate = failure_rate
        self.total        = 0
        self.successes    = 0
        self.failures     = 0

    def _unreliable_service(self):
        if random.random() < self.failure_rate:
            raise ConnectionError("Service temporarily unavailable")
        return "OK"

    def call_without_resilience(self, n: int) -> Dict:
        for _ in range(n):
            self.total += 1
            try:
                self._unreliable_service()
                self.successes += 1
            except Exception:
                self.failures += 1
        return {"total": self.total, "success": self.successes,
                "failure": self.failures, "failure_rate_pct": self.failures/self.total*100}

    def call_with_retry(self, n: int, max_retries: int = 3) -> Dict:
        successes = failures = 0
        for _ in range(n):
            for attempt in range(max_retries + 1):
                try:
                    self._unreliable_service()
                    successes += 1
                    break
                except Exception:
                    if attempt == max_retries:
                        failures += 1
        total = successes + failures
        return {"total": total, "success": successes,
                "failure": failures, "failure_rate_pct": failures/total*100 if total else 0}


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_fault_tolerance_and_resilience():
    print("=" * 65)
    print("FAULT TOLERANCE AND RESILIENCE")
    print("=" * 65)
    random.seed(42)

    # ── Without vs With Retry ─────────────────
    print("\n[1] RETRY IMPACT (10% base failure rate, 1000 requests)")
    print("─" * 55)
    sys1 = FaultTolerantSystem(failure_rate=0.10)
    r_no_retry = sys1.call_without_resilience(1000)
    sys2 = FaultTolerantSystem(failure_rate=0.10)
    r_retry    = sys2.call_with_retry(1000, max_retries=3)
    print(f"  Without retry : {r_no_retry['failure_rate_pct']:.1f}% failure rate")
    print(f"  With 3 retries: {r_retry['failure_rate_pct']:.1f}% failure rate")
    print(f"  (0.10^4 = {0.1**4*100:.4f}% residual with 3 retries)")

    # ── Exponential Backoff ───────────────────
    print("\n\n[2] EXPONENTIAL BACKOFF WITH JITTER")
    print("─" * 55)
    backoff = ExponentialBackoff(base_ms=100, max_ms=10_000, max_retries=5)
    print("  Delay schedule (ms) for 6 attempts:")
    for attempt in range(6):
        print(f"    Attempt {attempt}: {backoff.delay_ms(attempt):.0f} ms (jitter)")

    call_count = {"n": 0}
    def flaky():
        call_count["n"] += 1
        if call_count["n"] < 4:
            raise TimeoutError("timeout")
        return "data_returned"

    result = backoff.execute(flaky, "DB read")
    print(f"  Result: {result}")

    # ── Circuit Breaker ───────────────────────
    print("\n\n[3] CIRCUIT BREAKER SIMULATION")
    print("─" * 55)
    cb    = CircuitBreaker("payment-svc", failure_threshold=3, recovery_timeout_s=0.1)
    state = {"healthy": True, "fail_until": 10}

    def payment_service():
        if not state["healthy"]:
            raise ConnectionError("payment service down")
        return "payment_ok"

    results = []
    for i in range(20):
        if i == 3:
            state["healthy"] = False
            print(f"\n  [tick {i}] 🔴 Payment service goes down")
        if i == 12:
            state["healthy"] = True
            print(f"\n  [tick {i}] 🟢 Payment service recovered")
        try:
            r = cb.call(payment_service)
            results.append(f"tick {i}: ✅ {r}")
        except Exception as e:
            results.append(f"tick {i}: ❌ {e}")

    for r in results:
        print(f"  {r}")

    print(f"\n  CB state history: {' → '.join(cb.state_history)}")
    print(f"  Requests blocked (fast-fail): {cb.requests_blocked}")

    # ── Bulkhead ──────────────────────────────
    print("\n\n[4] BULKHEAD ISOLATION")
    print("─" * 55)
    payment_pool   = BulkheadPool("payment-svc",   max_concurrent=5)
    inventory_pool = BulkheadPool("inventory-svc", max_concurrent=5)

    print("  Scenario: inventory-svc becomes slow, acquires all threads")
    for _ in range(5):
        inventory_pool.acquire()   # inventory is slow, holds 5 slots

    print("  Now: 10 payment requests — still handled (separate pool):")
    for i in range(10):
        acquired = payment_pool.acquire()
        if acquired:
            payment_pool.release()
            print(f"  payment request {i+1}: ✅ served")
        else:
            print(f"  payment request {i+1}: ❌ rejected (pool exhausted)")

    payment_pool.stats()
    inventory_pool.stats()

    # ── Graceful Degradation ──────────────────
    print("\n\n[5] GRACEFUL DEGRADATION PATTERNS")
    print("─" * 55)
    patterns = [
        ("Recommendations", "ML service down", "Return most popular items (cached)"),
        ("Search",          "Elasticsearch slow","Return pre-cached top results"),
        ("User profile",    "DB overloaded",   "Return cached profile from Redis"),
        ("Feed",            "Fan-out stalled", "Return 24h old timeline from cache"),
        ("Payment",         "Processor down",  "Queue payment, return 'processing'"),
    ]
    print(f"  {'Feature':<20} {'Failure':<25} {'Degraded Response'}")
    print(f"  {'─'*70}")
    for feature, failure, degraded in patterns:
        print(f"  {feature:<20} {failure:<25} {degraded}")


if __name__ == "__main__":
    demonstrate_fault_tolerance_and_resilience()
