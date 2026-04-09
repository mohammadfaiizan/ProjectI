"""
DEAD LETTER QUEUE (DLQ) DESIGN
================================

Problem Statement:
In any messaging system, some messages will fail to process:
- Malformed payload (schema mismatch, null fields)
- Transient errors (DB down, timeout) — should be retried
- Permanent errors (business logic violation, invalid data) — should NOT be retried
- Poison pills: messages that crash the consumer every time → block the entire queue

Without a DLQ: failed messages are either silently dropped (at-most-once) or
cause the consumer to retry forever, blocking all subsequent messages (queue stall).

DLQ Design:
  1. Retry Policy: retry N times with exponential backoff before moving to DLQ.
  2. Dead Letter: message is moved to a separate DLQ queue after max retries.
  3. DLQ Headers: preserve original queue, failure reason, retry count, timestamps.
  4. DLQ Processing: monitoring, alerting, manual inspection, replay.
  5. Replay: re-enqueue fixed messages from DLQ back to original queue.

Exponential Backoff with Jitter:
  base_delay * (2 ^ attempt) + random_jitter
  Prevents thundering herd on retry storms.

Retry vs DLQ Routing:
  Error Type              Action
  ─────────────────────────────────────────────────────
  Transient (timeout, 5xx) → Retry with backoff
  Permanent (validation)   → DLQ immediately (no retry)
  Poison pill (crash)      → DLQ after max_retries
  Expired TTL              → DLQ or discard

DLQ Patterns:
  - Per-queue DLQ: each queue has its own DLQ (AWS SQS model)
  - Global DLQ: all failures go to one DLQ with original-queue tag
  - Tiered DLQ: DLQ-1 (1 retry), DLQ-2 (no more retries), alert
  - Replay lane: separate queue for operator-triggered replay

Alerting:
  DLQ depth > threshold → PagerDuty alert.
  DLQ growth rate > X msgs/min → escalate.
  Monitor DLQ age of oldest message.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict, deque
from enum import Enum
import time
import uuid
import threading
import random
import math


# ─────────────────────────────────────────────
# MESSAGE WITH RETRY METADATA
# ─────────────────────────────────────────────

class FailureType(Enum):
    TRANSIENT  = "transient"    # retry-able
    PERMANENT  = "permanent"    # go to DLQ immediately
    UNKNOWN    = "unknown"

@dataclass
class Message:
    msg_id       : str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    payload      : Any   = None
    # Retry tracking
    retry_count  : int   = 0
    first_enqueue: float = field(default_factory=time.time)
    last_attempt : float = 0.0
    # DLQ metadata (set when moved to DLQ)
    original_queue: str  = ""
    failure_reason: str  = ""
    failure_type  : str  = ""


# ─────────────────────────────────────────────
# RETRY POLICY
# ─────────────────────────────────────────────

@dataclass
class RetryPolicy:
    max_retries    : int   = 3
    base_delay_s   : float = 0.01    # 10 ms base (scaled for demo)
    max_delay_s    : float = 0.1
    jitter_factor  : float = 0.2     # ±20% jitter

    def next_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter."""
        exp_delay = self.base_delay_s * (2 ** attempt)
        capped    = min(exp_delay, self.max_delay_s)
        jitter    = capped * self.jitter_factor * random.uniform(-1, 1)
        return max(0.0, capped + jitter)

    def should_retry(self, msg: Message) -> bool:
        return msg.retry_count < self.max_retries


# ─────────────────────────────────────────────
# DEAD LETTER QUEUE
# ─────────────────────────────────────────────

class DeadLetterQueue:
    def __init__(self, name: str = "dlq"):
        self.name      = name
        self._messages : deque = deque()
        self._lock     = threading.Lock()
        self.total_received = 0

    def enqueue(self, msg: Message):
        with self._lock:
            self._messages.append(msg)
            self.total_received += 1

    def drain(self) -> List[Message]:
        with self._lock:
            msgs = list(self._messages)
            self._messages.clear()
            return msgs

    def peek_all(self) -> List[Message]:
        with self._lock:
            return list(self._messages)

    def depth(self) -> int:
        return len(self._messages)

    def summary(self) -> Dict[str, int]:
        """Count failures by reason."""
        counts: Dict[str, int] = defaultdict(int)
        for m in self.peek_all():
            counts[m.failure_reason] += 1
        return dict(counts)


# ─────────────────────────────────────────────
# QUEUE WITH RETRY + DLQ
# ─────────────────────────────────────────────

class QueueWithDLQ:
    """
    Message queue that retries transient failures with exponential backoff,
    and routes persistent failures to the dead letter queue.
    """

    def __init__(self, name: str, dlq: DeadLetterQueue,
                 retry_policy: RetryPolicy = None):
        self.name         = name
        self.dlq          = dlq
        self.policy       = retry_policy or RetryPolicy()
        self._queue       : deque = deque()
        self._retry_heap  : List[Tuple[float, Message]] = []  # (deliver_at, msg)
        self._lock        = threading.Lock()
        # Stats
        self.processed_ok   = 0
        self.retried        = 0
        self.dlq_sent       = 0

    def publish(self, msg: Message):
        with self._lock:
            msg.original_queue = self.name
            self._queue.append(msg)

    def _move_to_dlq(self, msg: Message, reason: str):
        msg.failure_reason = reason
        msg.failure_type   = FailureType.PERMANENT.value
        self.dlq.enqueue(msg)
        self.dlq_sent += 1

    def process(self, handler: Callable[[Message], None],
                 classify_error: Callable[[Exception], FailureType]):
        """Process one message. Returns True if message was available."""
        msg = None
        with self._lock:
            # Check retry heap first
            now = time.time()
            ready = [m for deliver_at, m in self._retry_heap if deliver_at <= now]
            not_ready = [(d, m) for d, m in self._retry_heap if d > now]
            self._retry_heap = not_ready

            if ready:
                msg = ready[0]
            elif self._queue:
                msg = self._queue.popleft()

        if msg is None:
            return False

        msg.last_attempt = time.time()
        try:
            handler(msg)
            self.processed_ok += 1
        except Exception as exc:
            failure_type = classify_error(exc)
            if failure_type == FailureType.PERMANENT:
                self._move_to_dlq(msg, str(exc))
            elif self.policy.should_retry(msg):
                msg.retry_count += 1
                delay = self.policy.next_delay(msg.retry_count)
                deliver_at = time.time() + delay
                with self._lock:
                    self._retry_heap.append((deliver_at, msg))
                self.retried += 1
            else:
                # Max retries exhausted
                self._move_to_dlq(msg, f"max_retries_exceeded: {exc}")

        return True

    def depth(self) -> int:
        return len(self._queue)

    def pending_retries(self) -> int:
        return len(self._retry_heap)

    def stats(self) -> Dict:
        return {
            "queue_depth"   : self.depth(),
            "pending_retries": self.pending_retries(),
            "processed_ok"  : self.processed_ok,
            "retried"       : self.retried,
            "dlq_sent"      : self.dlq_sent,
        }


# ─────────────────────────────────────────────
# DLQ REPLAY
# ─────────────────────────────────────────────

class DLQReplayService:
    """
    Operator tool: inspect DLQ, optionally transform messages, re-enqueue to origin.
    """

    def __init__(self, dlq: DeadLetterQueue):
        self.dlq       = dlq
        self.replayed  = 0
        self.discarded = 0

    def replay_all(self, target_queue: "QueueWithDLQ",
                   transform: Callable[[Message], Optional[Message]] = None):
        """Drain DLQ and re-publish to target queue after optional transform."""
        messages = self.dlq.drain()
        for msg in messages:
            if transform:
                msg = transform(msg)
            if msg is None:
                self.discarded += 1
                continue
            # Reset retry state for fresh attempt
            msg.retry_count   = 0
            msg.failure_reason = ""
            msg.failure_type   = ""
            target_queue.publish(msg)
            self.replayed += 1

    def replay_filtered(self, target_queue: "QueueWithDLQ",
                        predicate: Callable[[Message], bool]):
        """Replay only messages matching a predicate."""
        all_messages = self.dlq.drain()
        for msg in all_messages:
            if predicate(msg):
                msg.retry_count   = 0
                msg.failure_reason = ""
                target_queue.publish(msg)
                self.replayed += 1
            else:
                # Put back in DLQ
                self.dlq.enqueue(msg)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

class TransientError(Exception):
    pass

class PermanentError(Exception):
    pass

def error_classifier(exc: Exception) -> FailureType:
    if isinstance(exc, TransientError):
        return FailureType.TRANSIENT
    return FailureType.PERMANENT


def demonstrate_dlq():
    print("=" * 65)
    print("DEAD LETTER QUEUE DESIGN")
    print("=" * 65)

    random.seed(13)

    # ── Retry with Exponential Backoff ────────────
    print("\n[1] RETRY POLICY — EXPONENTIAL BACKOFF")
    print("─" * 55)
    policy = RetryPolicy(max_retries=4, base_delay_s=0.01, max_delay_s=0.1)
    print(f"  Backoff delays per attempt:")
    for attempt in range(5):
        delays = [policy.next_delay(attempt) * 1000 for _ in range(5)]
        avg = sum(delays) / len(delays)
        print(f"    attempt {attempt}: avg={avg:.1f}ms  "
              f"range=[{min(delays):.1f}, {max(delays):.1f}]ms")

    # ── Processing with DLQ ───────────────────────
    print("\n\n[2] QUEUE WITH DLQ — TRANSIENT vs PERMANENT FAILURES")
    print("─" * 55)

    dlq   = DeadLetterQueue("order-queue-dlq")
    queue = QueueWithDLQ("order-queue", dlq, RetryPolicy(max_retries=3))

    # Define handler: fails on specific payloads
    transient_fails: Dict[str, int] = defaultdict(int)

    def handler(msg: Message):
        payload = msg.payload
        if payload.get("type") == "invalid":
            raise PermanentError("invalid payload schema")
        if payload.get("type") == "flaky" and transient_fails[msg.msg_id] < 2:
            transient_fails[msg.msg_id] += 1
            raise TransientError("DB timeout")
        # success

    # Publish messages
    scenarios = [
        {"type": "normal",  "order_id": "ORD-001"},
        {"type": "normal",  "order_id": "ORD-002"},
        {"type": "invalid", "order_id": "ORD-003"},   # permanent fail
        {"type": "flaky",   "order_id": "ORD-004"},   # fails twice, then succeeds
        {"type": "normal",  "order_id": "ORD-005"},
        {"type": "invalid", "order_id": "ORD-006"},   # permanent fail
    ]
    for s in scenarios:
        queue.publish(Message(payload=s))

    # Process until queue + retries drained
    max_rounds = 30
    for _ in range(max_rounds):
        processed = queue.process(handler, error_classifier)
        if not processed and queue.pending_retries() == 0:
            break
        time.sleep(0.005)

    # Drain remaining retries
    for _ in range(50):
        queue.process(handler, error_classifier)
        if queue.pending_retries() == 0 and queue.depth() == 0:
            break
        time.sleep(0.005)

    s = queue.stats()
    print(f"  Queue stats:")
    print(f"    Processed OK   : {s['processed_ok']}")
    print(f"    Retried        : {s['retried']}")
    print(f"    Sent to DLQ    : {s['dlq_sent']}")
    print(f"    Queue depth    : {s['queue_depth']}")
    print(f"  DLQ depth: {dlq.depth()}")
    print(f"  DLQ failure summary: {dlq.summary()}")

    # ── DLQ Inspection ────────────────────────────
    print("\n\n[3] DLQ INSPECTION — OPERATOR VIEW")
    print("─" * 55)
    for msg in dlq.peek_all():
        order_id = msg.payload.get("order_id", "?")
        print(f"  msg={msg.msg_id} order={order_id} "
              f"retries={msg.retry_count} reason='{msg.failure_reason}'")

    # ── DLQ Replay ────────────────────────────────
    print("\n\n[4] DLQ REPLAY — RE-ENQUEUE AFTER FIX")
    print("─" * 55)
    replay_queue = QueueWithDLQ("order-queue-replay", DeadLetterQueue())
    replayer     = DLQReplayService(dlq)

    # Fix the messages before replay (transform invalid → normal)
    def fix_invalid(msg: Message) -> Optional[Message]:
        if msg.payload.get("type") == "invalid":
            msg.payload = {**msg.payload, "type": "normal"}  # simulated fix
        return msg

    replayer.replay_all(replay_queue, transform=fix_invalid)
    print(f"  DLQ drained. Replayed: {replayer.replayed}  Discarded: {replayer.discarded}")
    print(f"  Replay queue depth: {replay_queue.depth()}")
    print(f"  DLQ depth after replay: {dlq.depth()}")

    # ── DLQ Monitoring ────────────────────────────
    print("\n\n[5] DLQ MONITORING THRESHOLDS")
    print("─" * 55)
    thresholds = [
        ("DLQ depth > 100",      "Page on-call — high failure rate"),
        ("DLQ age > 1 hour",     "Alert — messages accumulating unprocessed"),
        ("DLQ growth > 10/min",  "Escalate — processing may be fully broken"),
        ("Same error_type > 80%","Likely config or deployment issue"),
        ("New error_type spike",  "Alert — new bug introduced"),
    ]
    for threshold, action in thresholds:
        print(f"  {threshold:<30} → {action}")

    # ── Pattern Summary ───────────────────────────
    print("\n\n[6] DLQ DESIGN DECISIONS")
    print("─" * 55)
    decisions = [
        ("Max retries",          "3-5 for transient; 0 for permanent (invalid schema)"),
        ("Backoff",              "Exponential + jitter to avoid retry storms"),
        ("Error classification", "Catch exception type or HTTP status to route"),
        ("DLQ per queue",        "Easier to trace; know which pipeline failed"),
        ("DLQ TTL",              "Keep DLQ messages 7-30 days for investigation"),
        ("Replay strategy",      "Bulk replay after fix; filtered replay for subsets"),
        ("DLQ poison pill",      "Inspect + discard; don't replay blindly"),
    ]
    for decision, guideline in decisions:
        print(f"  {decision:<24} {guideline}")


if __name__ == "__main__":
    demonstrate_dlq()
