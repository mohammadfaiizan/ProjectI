"""
CIRCUIT BREAKER AND BULKHEAD PATTERNS
========================================

Problem Statement:
In a microservices system, services call other services.
If Service B is slow or down, calls from Service A pile up waiting.
Threads exhaust. Memory fills. A fails too. Cascading failure.
Circuit Breaker and Bulkhead are two defensive patterns that prevent this.

CIRCUIT BREAKER:
  Inspired by electrical circuit breakers — cuts power to prevent fire.
  Three states:

  CLOSED (normal):
    All calls pass through to the downstream service.
    Failures are counted in a sliding window.
    If failure_rate >= threshold AND min_calls reached → trip to OPEN.

  OPEN (broken):
    ALL calls fail immediately without touching the downstream service.
    Fast-fail. Caller gets error immediately instead of waiting.
    After reset_timeout passes → move to HALF_OPEN.

  HALF_OPEN (probing):
    Allow a limited number of probe calls through.
    If probe calls succeed → CLOSED (service recovered).
    If probe calls fail → OPEN again (not yet recovered).

  Key parameters:
    failure_threshold:  % of calls that must fail to trip (e.g., 50%)
    min_requests:       minimum calls before evaluating (avoid tripping on 1/1 = 100%)
    reset_timeout_s:    how long to stay OPEN before probing
    success_threshold:  how many successes in HALF_OPEN before CLOSED

BULKHEAD:
  Named after ship compartments — if one fills with water, others stay dry.
  Isolates resource pools per downstream dependency.
  Without bulkhead: all threads in a shared pool drain due to one slow dependency.
  With bulkhead: each dependency gets its own thread pool / semaphore.

  Thread pool bulkhead:
    Inventory service gets pool of 5 threads.
    Payment service gets pool of 5 threads.
    If inventory is slow and fills its 5, payment still has its own 5.
    System degrades partially rather than completely.

  Semaphore bulkhead:
    Limits concurrent calls without a separate thread pool.
    Lighter-weight. Still isolates concurrent call limits per dependency.

Combining Both:
  Circuit Breaker: stops CALLING a degraded dependency (fail fast).
  Bulkhead: limits HOW MANY concurrent calls can be in-flight.
  Use together: bulkhead limits concurrent requests; CB cuts calls when failures spike.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import time
import threading
import random
import collections


# ─────────────────────────────────────────────
# SLIDING WINDOW (for failure rate calculation)
# ─────────────────────────────────────────────

class SlidingWindowCounter:
    """Track success/failure counts in a fixed-size sliding window."""

    def __init__(self, size: int = 10):
        self.size    = size
        self._window : collections.deque = collections.deque(maxlen=size)
        self._lock   = threading.Lock()

    def record(self, success: bool):
        with self._lock:
            self._window.append(1 if success else 0)

    def failure_rate(self) -> float:
        with self._lock:
            if not self._window:
                return 0.0
            return 1.0 - (sum(self._window) / len(self._window))

    def count(self) -> int:
        return len(self._window)

    def reset(self):
        with self._lock:
            self._window.clear()


# ─────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpen(Exception):
    pass


@dataclass
class CircuitBreakerConfig:
    failure_threshold   : float = 0.5    # 50% failure rate trips the breaker
    min_requests        : int   = 5      # need at least 5 calls before evaluating
    reset_timeout_s     : float = 2.0    # stay OPEN for 2s before probing
    probe_count         : int   = 2      # allow 2 probe calls in HALF_OPEN
    success_threshold   : int   = 2      # 2 successes in HALF_OPEN → CLOSED
    window_size         : int   = 10     # sliding window size


class CircuitBreaker:
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name          = name
        self.cfg           = config or CircuitBreakerConfig()
        self._state        = CircuitState.CLOSED
        self._window       = SlidingWindowCounter(self.cfg.window_size)
        self._opened_at    : Optional[float] = None
        self._probe_calls  = 0
        self._probe_success= 0
        self._lock         = threading.Lock()
        self._state_log    : List[Tuple[float, str]] = []
        self._total_calls  = 0
        self._blocked_calls= 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._get_state()

    def _get_state(self) -> CircuitState:
        if (self._state == CircuitState.OPEN and
                self._opened_at is not None and
                time.time() - self._opened_at >= self.cfg.reset_timeout_s):
            self._transition(CircuitState.HALF_OPEN)
            self._probe_calls   = 0
            self._probe_success = 0
        return self._state

    def _transition(self, new_state: CircuitState):
        if self._state != new_state:
            self._state_log.append((time.time(), f"{self._state.value}→{new_state.value}"))
            self._state = new_state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self._lock:
            state = self._get_state()
            self._total_calls += 1

            if state == CircuitState.OPEN:
                self._blocked_calls += 1
                raise CircuitBreakerOpen(
                    f"Circuit '{self.name}' is OPEN — request blocked (fast fail)")

            if state == CircuitState.HALF_OPEN:
                if self._probe_calls >= self.cfg.probe_count:
                    self._blocked_calls += 1
                    raise CircuitBreakerOpen(
                        f"Circuit '{self.name}' HALF_OPEN — probe slots exhausted")
                self._probe_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerOpen:
            raise
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            self._window.record(True)
            if self._state == CircuitState.HALF_OPEN:
                self._probe_success += 1
                if self._probe_success >= self.cfg.success_threshold:
                    self._window.reset()
                    self._transition(CircuitState.CLOSED)

    def _on_failure(self):
        with self._lock:
            self._window.record(False)
            if self._state == CircuitState.HALF_OPEN:
                self._opened_at = time.time()
                self._transition(CircuitState.OPEN)
            elif (self._state == CircuitState.CLOSED and
                  self._window.count() >= self.cfg.min_requests and
                  self._window.failure_rate() >= self.cfg.failure_threshold):
                self._opened_at = time.time()
                self._transition(CircuitState.OPEN)

    def summary(self) -> Dict:
        with self._lock:
            return {
                "name"          : self.name,
                "state"         : self._get_state().value,
                "total_calls"   : self._total_calls,
                "blocked_calls" : self._blocked_calls,
                "failure_rate"  : f"{self._window.failure_rate():.0%}",
                "window_count"  : self._window.count(),
                "transitions"   : [(t, s) for t, s in self._state_log],
            }


# ─────────────────────────────────────────────
# BULKHEAD (semaphore-based)
# ─────────────────────────────────────────────

class BulkheadFull(Exception):
    pass


class SemaphoreBulkhead:
    """
    Limits concurrent in-flight calls to a dependency.
    If the semaphore is full → reject immediately (fast fail).
    """

    def __init__(self, name: str, max_concurrent: int):
        self.name           = name
        self.max_concurrent = max_concurrent
        self._semaphore     = threading.Semaphore(max_concurrent)
        self._lock          = threading.Lock()
        self._active        = 0
        self._rejected      = 0
        self._completed     = 0

    def call(self, func: Callable, *args, **kwargs) -> Any:
        acquired = self._semaphore.acquire(blocking=False)
        if not acquired:
            with self._lock:
                self._rejected += 1
            raise BulkheadFull(
                f"Bulkhead '{self.name}' full ({self.max_concurrent} concurrent max)")
        try:
            with self._lock:
                self._active += 1
            return func(*args, **kwargs)
        finally:
            self._semaphore.release()
            with self._lock:
                self._active    = max(0, self._active - 1)
                self._completed += 1

    def summary(self) -> Dict:
        with self._lock:
            return {
                "name"          : self.name,
                "max_concurrent": self.max_concurrent,
                "active"        : self._active,
                "completed"     : self._completed,
                "rejected"      : self._rejected,
            }


# ─────────────────────────────────────────────
# DOWNSTREAM SERVICE SIMULATORS
# ─────────────────────────────────────────────

def make_flaky_call(fail_rate: float, latency_ms: float = 20):
    """Returns a callable that fails at the given rate."""
    def call():
        time.sleep(latency_ms / 1000)
        if random.random() < fail_rate:
            raise ConnectionError("downstream service error")
        return {"status": "ok"}
    return call


def make_slow_call(latency_ms: float):
    """Returns a callable that takes a long time (simulates slow downstream)."""
    def call():
        time.sleep(latency_ms / 1000)
        return {"status": "ok"}
    return call


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_circuit_breaker_and_bulkhead():
    print("=" * 65)
    print("CIRCUIT BREAKER AND BULKHEAD PATTERNS")
    print("=" * 65)

    # ── 1. Circuit breaker lifecycle ─────────────
    print("\n[1] CIRCUIT BREAKER STATE TRANSITIONS")
    print("─" * 55)

    cfg = CircuitBreakerConfig(
        failure_threshold=0.5, min_requests=4,
        reset_timeout_s=0.3, success_threshold=2,
        probe_count=3, window_size=6
    )
    cb = CircuitBreaker("inventory-service", cfg)

    # Phase 1: CLOSED with failures
    print("  Phase 1: Sending failing calls to trip the breaker...")
    always_fail = make_flaky_call(1.0, latency_ms=5)
    for i in range(6):
        try:
            cb.call(always_fail)
        except (ConnectionError, CircuitBreakerOpen) as e:
            err_type = "BLOCKED" if isinstance(e, CircuitBreakerOpen) else "FAILED"
            print(f"    Call {i+1}: {err_type:<8} state={cb.state.value}")

    # Phase 2: OPEN — all calls fast-fail
    print(f"\n  Phase 2: Circuit OPEN — sending more calls (should all fast-fail)...")
    for i in range(3):
        try:
            cb.call(always_fail)
        except CircuitBreakerOpen:
            print(f"    Call {i+1}: BLOCKED (fast fail) state={cb.state.value}")

    # Phase 3: HALF_OPEN — wait for timeout, send probe
    print(f"\n  Phase 3: Waiting {cfg.reset_timeout_s}s → HALF_OPEN probe...")
    time.sleep(cfg.reset_timeout_s + 0.05)
    always_succeed = make_flaky_call(0.0, latency_ms=5)
    for i in range(4):
        try:
            cb.call(always_succeed)
            print(f"    Probe {i+1}: SUCCESS  state={cb.state.value}")
        except CircuitBreakerOpen:
            print(f"    Probe {i+1}: BLOCKED  state={cb.state.value}")

    s = cb.summary()
    print(f"\n  Final state: {s['state']}")
    print(f"  Total calls: {s['total_calls']}  Blocked: {s['blocked_calls']}")
    print(f"  Transitions: {[t[1] for t in s['transitions']]}")

    # ── 2. Bulkhead isolation ─────────────────────
    print("\n\n[2] BULKHEAD — THREAD ISOLATION PER DEPENDENCY")
    print("─" * 55)

    inv_bulkhead = SemaphoreBulkhead("inventory-svc", max_concurrent=3)
    pay_bulkhead = SemaphoreBulkhead("payment-svc",   max_concurrent=3)

    slow_call = make_slow_call(latency_ms=200)   # simulates slow inventory

    results = {"inv_ok": 0, "inv_full": 0, "pay_ok": 0}
    lock     = threading.Lock()

    def try_inventory():
        try:
            inv_bulkhead.call(slow_call)
            with lock: results["inv_ok"] += 1
        except BulkheadFull:
            with lock: results["inv_full"] += 1

    def try_payment():
        try:
            pay_bulkhead.call(make_flaky_call(0.0, 10))
            with lock: results["pay_ok"] += 1
        except BulkheadFull:
            pass

    # Saturate inventory bulkhead with 6 slow calls
    inv_threads = [threading.Thread(target=try_inventory) for _ in range(6)]
    # Simultaneously try to use payment (separate bulkhead — unaffected)
    pay_threads = [threading.Thread(target=try_payment) for _ in range(5)]

    all_threads = inv_threads + pay_threads
    for t in all_threads: t.start()
    for t in all_threads: t.join()

    print(f"  Inventory bulkhead (max 3 concurrent, all slow calls):")
    print(f"    Completed: {results['inv_ok']}   Rejected: {results['inv_full']}")
    print(f"  Payment bulkhead (max 3 concurrent, fast calls):")
    print(f"    Completed: {results['pay_ok']}")
    print(f"\n  → Inventory congestion did NOT affect payment availability.")
    print(f"    Without bulkhead, all threads would drain for both services.")

    print(f"\n  Bulkhead summaries:")
    for bh in [inv_bulkhead, pay_bulkhead]:
        s = bh.summary()
        print(f"    {s['name']:<18} completed={s['completed']} "
              f"rejected={s['rejected']}")

    # ── 3. Combined: CB + Bulkhead ────────────────
    print("\n\n[3] COMBINED: CIRCUIT BREAKER + BULKHEAD")
    print("─" * 55)
    print("  Bulkhead limits concurrent calls (queue doesn't build up).")
    print("  Circuit Breaker stops calling entirely after failure threshold.")

    cb2 = CircuitBreaker("payment-v2",
                         CircuitBreakerConfig(failure_threshold=0.6,
                                              min_requests=3,
                                              reset_timeout_s=0.5,
                                              window_size=5))
    bh2 = SemaphoreBulkhead("payment-v2", max_concurrent=2)

    flaky = make_flaky_call(0.7, latency_ms=5)
    results2 = {"ok": 0, "cb_open": 0, "bh_full": 0, "svc_err": 0}

    def combined_call():
        try:
            bh2.call(lambda: cb2.call(flaky))
            with lock: results2["ok"] += 1
        except CircuitBreakerOpen:
            with lock: results2["cb_open"] += 1
        except BulkheadFull:
            with lock: results2["bh_full"] += 1
        except ConnectionError:
            with lock: results2["svc_err"] += 1

    threads = [threading.Thread(target=combined_call) for _ in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()

    print(f"  20 concurrent calls against flaky service (70% fail, max 2 concurrent):")
    print(f"    OK:              {results2['ok']}")
    print(f"    CB blocked:      {results2['cb_open']}")
    print(f"    Bulkhead full:   {results2['bh_full']}")
    print(f"    Service errors:  {results2['svc_err']}")

    # ── 4. Configuration guide ────────────────────
    print("\n\n[4] CIRCUIT BREAKER CONFIGURATION GUIDE")
    print("─" * 55)
    rows = [
        ("failure_threshold", "50%",   "Trip at 50% failure rate in window"),
        ("min_requests",      "10",    "Need 10 calls before evaluating (avoid noise)"),
        ("reset_timeout_s",   "30s",   "Stay open 30s before probing"),
        ("probe_count",       "5",     "Allow 5 probe calls in HALF_OPEN"),
        ("success_threshold", "3",     "3 probe successes → CLOSED"),
        ("window_size",       "20",    "Evaluate last 20 calls"),
    ]
    print(f"  {'Parameter':<22} {'Typical':<10} {'Purpose'}")
    print(f"  {'─'*60}")
    for param, typical, purpose in rows:
        print(f"  {param:<22} {typical:<10} {purpose}")

    # ── 5. Pattern summary ────────────────────────
    print("\n\n[5] CIRCUIT BREAKER vs BULKHEAD")
    print("─" * 55)
    comparisons = [
        ("Problem solved",   "Cascading failure on degraded dep.",  "Thread exhaustion from one slow dep."),
        ("Mechanism",        "Count failures; block if threshold",  "Limit concurrent calls with semaphore"),
        ("Effect when full", "Fast fail: immediate error response", "Reject: immediate error response"),
        ("Recovery",         "Automatic after reset_timeout probe", "Automatic when in-flight calls finish"),
        ("Granularity",      "Per downstream service",              "Per downstream service"),
        ("Use together?",    "Yes — CB cuts calls; BH limits concurrency",  "Yes"),
    ]
    print(f"  {'Aspect':<20} {'Circuit Breaker':<38} {'Bulkhead'}")
    print(f"  {'─'*80}")
    for aspect, cb_desc, bh_desc in comparisons:
        print(f"  {aspect:<20} {cb_desc:<38} {bh_desc}")


if __name__ == "__main__":
    demonstrate_circuit_breaker_and_bulkhead()
