"""
BACKPRESSURE AND FLOW CONTROL
================================

Problem Statement:
When a producer generates data faster than a consumer can process it,
the queue/buffer grows unboundedly → OOM, latency spikes, cascading failures.
Backpressure is the mechanism by which a consumer signals to its upstream
to slow down or stop until the consumer can catch up.

Without backpressure:
  Producer 10k msg/s → Consumer 2k msg/s → Buffer fills → OOM or data loss.

With backpressure:
  Producer 10k msg/s → Consumer 2k msg/s → Buffer near full → Producer slows to 2k msg/s.

Flow Control Strategies:
  1. Blocking: consumer blocks producer (synchronous, same process). Simple.
     Risk: deadlock if both sides block on each other.

  2. Rate Limiting (Token Bucket / Leaky Bucket): producer throttled at source.
     Decoupled from consumer. Works across processes/services.

  3. Credit-Based Flow Control: consumer grants N credits to producer.
     Producer sends at most N messages. Consumer replenishes credits as it processes.
     Used by: AMQP (prefetch count), gRPC flow control, TCP sliding window.

  4. Adaptive: producer monitors queue depth or latency, adjusts rate dynamically.
     Simple heuristic: if queue > 80%, halve rate. If queue < 20%, double rate.

  5. Bounded Queue + Rejection: producer gets "queue full" error, must retry/drop.
     Explicit backpressure signal. Consumer controls its own destiny.
     Overflow policies: block, drop oldest, drop newest, reject.

  6. Reactive Streams: standardised protocol (Java: RxJava, Project Reactor).
     request(N) → publisher sends at most N items → consumer calls request(N) again.

Buffer Overflow Policies:
  DROP_OLDEST: discard head of queue (oldest data). Good for metrics/events.
  DROP_NEWEST: reject incoming message (newest data). Simple, producer-controlled.
  BLOCK:       producer waits until space available. Applies backpressure upstream.
  RESIZE:      grow the buffer (dangerous: may hide the problem).

Key Metric: Queue depth (backlog).
  If depth trending up → consumer slower than producer → investigate.
  If depth trending down → consumer faster, catching up.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque
from enum import Enum
import time
import threading
import random
import math


# ─────────────────────────────────────────────
# OVERFLOW POLICIES
# ─────────────────────────────────────────────

class OverflowPolicy(Enum):
    BLOCK       = "block"
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    REJECT      = "reject"


# ─────────────────────────────────────────────
# BOUNDED QUEUE WITH BACKPRESSURE
# ─────────────────────────────────────────────

class BoundedQueue:
    """
    Fixed-capacity queue with configurable overflow policy.
    Signals backpressure to producers via blocking or rejection.
    """

    def __init__(self, capacity: int, policy: OverflowPolicy = OverflowPolicy.BLOCK):
        self.capacity   = capacity
        self.policy     = policy
        self._queue     : deque = deque()
        self._lock      = threading.Lock()
        self._not_full  = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)
        # Stats
        self.enqueued   = 0
        self.dropped    = 0
        self.rejected   = 0
        self.blocked_ms = 0.0

    def put(self, item: Any, timeout_s: float = 5.0) -> bool:
        """Returns True if item enqueued, False if dropped/rejected."""
        with self._not_full:
            if len(self._queue) >= self.capacity:
                if self.policy == OverflowPolicy.BLOCK:
                    start = time.time()
                    result = self._not_full.wait_for(
                        lambda: len(self._queue) < self.capacity,
                        timeout=timeout_s)
                    self.blocked_ms += (time.time() - start) * 1000
                    if not result:
                        self.rejected += 1
                        return False
                elif self.policy == OverflowPolicy.DROP_OLDEST:
                    self._queue.popleft()
                    self.dropped += 1
                elif self.policy == OverflowPolicy.DROP_NEWEST:
                    self.dropped += 1
                    return False
                elif self.policy == OverflowPolicy.REJECT:
                    self.rejected += 1
                    return False

            self._queue.append(item)
            self.enqueued += 1
            self._not_empty.notify()
            return True

    def get(self, timeout_s: float = 1.0) -> Optional[Any]:
        with self._not_empty:
            if not self._queue:
                self._not_empty.wait_for(lambda: bool(self._queue), timeout=timeout_s)
            if not self._queue:
                return None
            item = self._queue.popleft()
            self._not_full.notify()
            return item

    def depth(self) -> int:
        return len(self._queue)

    def utilization(self) -> float:
        return len(self._queue) / self.capacity

    def stats(self) -> Dict:
        return {
            "depth"      : self.depth(),
            "utilization": f"{self.utilization()*100:.1f}%",
            "enqueued"   : self.enqueued,
            "dropped"    : self.dropped,
            "rejected"   : self.rejected,
            "blocked_ms" : f"{self.blocked_ms:.1f}",
        }


# ─────────────────────────────────────────────
# TOKEN BUCKET RATE LIMITER (producer-side)
# ─────────────────────────────────────────────

class TokenBucket:
    """
    Allows bursts up to `capacity` tokens.
    Tokens refill at `rate` tokens/second.
    Each message costs 1 token. If empty, producer blocks or drops.
    """

    def __init__(self, rate: float, capacity: float):
        self.rate     = rate       # tokens per second
        self.capacity = capacity   # burst size
        self._tokens  = capacity
        self._last    = time.time()
        self._lock    = threading.Lock()
        self.consumed = 0
        self.dropped  = 0

    def _refill(self):
        now    = time.time()
        added  = (now - self._last) * self.rate
        self._tokens = min(self.capacity, self._tokens + added)
        self._last   = now

    def consume(self, block: bool = True, timeout_s: float = 1.0) -> bool:
        deadline = time.time() + timeout_s
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    self.consumed += 1
                    return True
            if not block or time.time() > deadline:
                self.dropped += 1
                return False
            time.sleep(0.001)

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


# ─────────────────────────────────────────────
# CREDIT-BASED FLOW CONTROL
# ─────────────────────────────────────────────

class CreditBasedChannel:
    """
    Consumer grants credits to producer. Producer sends at most `credits` messages.
    Consumer replenishes credits after processing (like AMQP prefetch_count).
    """

    def __init__(self, initial_credits: int = 10):
        self._credits     = initial_credits
        self._lock        = threading.Lock()
        self._credits_cv  = threading.Condition(self._lock)
        self._buffer      : deque = deque()
        self.sent         = 0
        self.received     = 0

    def send(self, item: Any, timeout_s: float = 2.0) -> bool:
        """Producer: wait for credit, then send."""
        with self._credits_cv:
            acquired = self._credits_cv.wait_for(
                lambda: self._credits > 0, timeout=timeout_s)
            if not acquired:
                return False
            self._credits -= 1
            self._buffer.append(item)
            self.sent += 1
            return True

    def receive(self) -> Optional[Any]:
        """Consumer: receive item, process it, then replenish credit."""
        with self._lock:
            if not self._buffer:
                return None
            item = self._buffer.popleft()
            self.received += 1
        return item

    def ack(self, count: int = 1):
        """Consumer: signal that N items were processed — replenish credits."""
        with self._credits_cv:
            self._credits += count
            self._credits_cv.notify_all()

    def credit_available(self) -> int:
        return self._credits


# ─────────────────────────────────────────────
# ADAPTIVE RATE CONTROLLER
# ─────────────────────────────────────────────

class AdaptiveRateController:
    """
    Producer monitors queue depth. Adjusts send rate to match consumer throughput.
    Doubles rate when queue is low; halves when queue is near full.
    """

    def __init__(self, initial_rate: float, min_rate: float, max_rate: float,
                 queue: BoundedQueue):
        self.rate      = initial_rate
        self.min_rate  = min_rate
        self.max_rate  = max_rate
        self.queue     = queue
        self._history  : List[Tuple[float, float]] = []   # (time, rate)

    def adjust(self):
        util = self.queue.utilization()
        if util > 0.8:
            self.rate = max(self.min_rate, self.rate * 0.5)
        elif util < 0.2:
            self.rate = min(self.max_rate, self.rate * 2.0)
        self._history.append((time.time(), self.rate))

    def sleep_between_sends(self):
        time.sleep(1.0 / self.rate)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_backpressure():
    print("=" * 65)
    print("BACKPRESSURE AND FLOW CONTROL")
    print("=" * 65)

    random.seed(5)

    # ── Bounded Queue Overflow Policies ───────────
    print("\n[1] BOUNDED QUEUE — OVERFLOW POLICIES")
    print("─" * 55)

    for policy in [OverflowPolicy.DROP_NEWEST, OverflowPolicy.DROP_OLDEST]:
        q = BoundedQueue(capacity=5, policy=policy)
        # Burst 10 items into capacity-5 queue
        for i in range(10):
            q.put(f"item-{i}", timeout_s=0.001)
        print(f"  {policy.value:<14}: enqueued={q.enqueued} "
              f"dropped={q.dropped} depth={q.depth()}")
        # Show which items survived (drop_oldest should have 5-9; drop_newest 0-4)
        items = []
        while True:
            item = q.get(timeout_s=0.01)
            if item is None:
                break
            items.append(item)
        print(f"    Surviving items: {items}")

    # BLOCK policy — measure backpressure wait
    q_block = BoundedQueue(capacity=3, policy=OverflowPolicy.BLOCK)
    consumer_started = threading.Event()

    def slow_consumer():
        consumer_started.set()
        for _ in range(6):
            q_block.get(timeout_s=1.0)
            time.sleep(0.02)   # 50 msg/s consumer

    t = threading.Thread(target=slow_consumer, daemon=True)
    t.start()
    consumer_started.wait()

    for i in range(6):
        q_block.put(f"m{i}", timeout_s=2.0)   # fast producer

    t.join(timeout=2.0)
    s = q_block.stats()
    print(f"\n  BLOCK policy : enqueued={s['enqueued']} "
          f"rejected={s['rejected']} blocked_ms={s['blocked_ms']}")

    # ── Token Bucket Rate Limiting ────────────────
    print("\n\n[2] TOKEN BUCKET — PRODUCER RATE LIMITING")
    print("─" * 55)

    bucket = TokenBucket(rate=1000, capacity=100)   # 1000 msg/s, burst 100
    sent, dropped = 0, 0
    t0 = time.time()
    for _ in range(200):
        if bucket.consume(block=False):
            sent += 1
        else:
            dropped += 1
    elapsed = (time.time() - t0) * 1000

    print(f"  Attempted 200 sends in {elapsed:.1f}ms")
    print(f"  Sent={bucket.consumed} Dropped={bucket.dropped}")
    print(f"  Token bucket available after burst: {bucket.available:.1f}")

    # Refill and try again
    time.sleep(0.05)   # 50ms → 50 tokens refilled
    refilled = bucket.available
    print(f"  After 50ms sleep: {refilled:.1f} tokens available (rate=1000/s)")

    # ── Credit-Based Flow Control ─────────────────
    print("\n\n[3] CREDIT-BASED FLOW CONTROL (AMQP-STYLE)")
    print("─" * 55)

    channel      = CreditBasedChannel(initial_credits=5)
    produced     = []
    processed    = []
    stop_flag    = threading.Event()

    def producer_thread():
        for i in range(15):
            ok = channel.send(f"msg-{i}", timeout_s=2.0)
            if ok:
                produced.append(i)

    def consumer_thread():
        while len(processed) < 15:
            item = channel.receive()
            if item:
                time.sleep(0.01)   # processing time
                processed.append(item)
                channel.ack(1)     # release credit

    pt = threading.Thread(target=producer_thread, daemon=True)
    ct = threading.Thread(target=consumer_thread, daemon=True)
    ct.start()
    pt.start()
    pt.join(timeout=3.0)
    ct.join(timeout=3.0)

    print(f"  Produced: {len(produced)}  Processed: {len(processed)}")
    print(f"  Credits remaining: {channel.credit_available()}")
    print(f"  → Producer was throttled to consumer's processing speed")

    # ── Adaptive Rate Control ─────────────────────
    print("\n\n[4] ADAPTIVE RATE CONTROLLER — QUEUE DEPTH FEEDBACK")
    print("─" * 55)

    q_adapt    = BoundedQueue(capacity=20, policy=OverflowPolicy.DROP_NEWEST)
    controller = AdaptiveRateController(
        initial_rate = 100.0,
        min_rate     = 10.0,
        max_rate     = 500.0,
        queue        = q_adapt,
    )

    # Simulate queue depth varying: full → empty → full
    scenarios = [
        (18, "high utilization — rate should decrease"),
        (2,  "low utilization — rate should increase"),
        (15, "moderate-high — rate should decrease"),
    ]
    print(f"  {'Queue depth':>12} {'Utilization':>12} {'Rate Before':>12} {'Rate After':>12}")
    for depth, desc in scenarios:
        q_adapt._queue = deque(range(depth))   # set queue depth
        rate_before = controller.rate
        controller.adjust()
        print(f"  depth={depth:<6} util={q_adapt.utilization()*100:5.1f}%  "
              f"rate {rate_before:>8.1f} → {controller.rate:>8.1f}  ({desc})")

    # ── Strategy Comparison ───────────────────────
    print("\n\n[5] BACKPRESSURE STRATEGIES")
    print("─" * 55)
    rows = [
        ("Blocking queue",       "Simple, synchronous", "Same-process, in-memory queues"),
        ("Token bucket",         "Producer throttled at source", "API rate limiting, ingestion"),
        ("Credit-based",         "Consumer controls send rate exactly", "AMQP prefetch, gRPC"),
        ("Adaptive rate",        "Dynamic feedback loop", "Variable-load pipelines"),
        ("Drop oldest",          "Metrics/events (freshness > completeness)", "Monitoring, telemetry"),
        ("Drop newest",          "Reject when full (producer notified)", "Request queues"),
    ]
    print(f"  {'Strategy':<22} {'Mechanism':<35} {'Use Case'}")
    print(f"  {'─'*80}")
    for strategy, mech, use in rows:
        print(f"  {strategy:<22} {mech:<35} {use}")


if __name__ == "__main__":
    demonstrate_backpressure()
