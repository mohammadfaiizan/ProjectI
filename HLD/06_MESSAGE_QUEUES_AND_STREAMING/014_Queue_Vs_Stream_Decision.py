"""
QUEUE vs STREAM — DECISION FRAMEWORK
======================================

Problem Statement:
"Should I use a message queue (SQS, RabbitMQ) or a streaming platform (Kafka, Kinesis)?"
The answer depends on your access patterns, retention needs, consumer model, and ordering requirements.
Choosing wrong leads to either over-engineering or missing critical capabilities.

Message Queue Characteristics:
  - Broker deletes message once acknowledged (consumed).
  - Consumer-driven: consumer pulls/receives; message disappears on ACK.
  - At-most-once or at-least-once delivery.
  - No inherent ordering across messages (some queues offer FIFO at lower throughput).
  - Replay impossible (message gone after consumption).
  - Best for: task distribution, work queues, command dispatch, job scheduling.
  - Examples: AWS SQS, RabbitMQ, ActiveMQ, Celery tasks.

Stream Characteristics:
  - Broker retains messages for a configurable retention period (days/forever).
  - Consumer-tracked: consumer maintains offset; broker doesn't know what's consumed.
  - Multiple consumer groups can each independently read the same data.
  - Ordered within partition/shard.
  - Replay: consumer can reset offset to re-read past events.
  - Best for: event sourcing, audit logs, analytics, fan-out to multiple systems.
  - Examples: Apache Kafka, AWS Kinesis, Azure Event Hubs, Redpanda.

Decision Matrix:
  Question                              Queue           Stream
  ──────────────────────────────────────────────────────────────
  Message consumed by one consumer?     ✓               ✗ (use consumer group)
  Multiple consumers same data?         ✗ (fan-out MQ)  ✓ (multiple groups)
  Need to replay past messages?         ✗               ✓
  Need ordering?                        Limited          ✓ (per partition)
  High throughput (millions/s)?         SQS ~3k/s        Kafka ~millions/s
  Retention after consume?              ✗               ✓
  Push notifications?                   ✓ (AMQP push)   Pull (consumer polls)
  Complex routing (topic/fanout)?       ✓ (RabbitMQ)    Limited
  Long-term analytics / data lake?      ✗               ✓

Hybrid Approach:
  Kafka + SQS: Kafka as backbone (event log, fan-out), SQS for task dispatch.
  SNS → SQS: SNS for fan-out, SQS for competing consumer work queues.
  Kafka topic → consumer group per service → each group as a queue.

Key Questions to Drive the Decision:
  1. Does the message need to exist after it's processed?
     No → Queue. Yes → Stream.
  2. Will multiple independent services consume the same message?
     No → Queue. Yes → Stream (or SNS/fan-out).
  3. Do you need to re-process past messages (replay)?
     No → Queue. Yes → Stream.
  4. Is strict per-key ordering required?
     No → Queue. Yes → Stream (partition by key).
  5. Is this a task/job (do once) or an event (record what happened)?
     Task → Queue. Event → Stream.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import time
import uuid
import threading
from collections import defaultdict, deque


# ─────────────────────────────────────────────
# SIMULATED QUEUE (SQS-like)
# ─────────────────────────────────────────────

class SimpleQueue:
    """
    Message queue: messages deleted on ACK.
    Competing consumers — each message consumed by exactly one worker.
    No replay, no offset.
    """

    def __init__(self, name: str):
        self.name       = name
        self._messages  : deque = deque()
        self._inflight  : Dict[str, Any] = {}
        self._lock      = threading.Lock()
        self.published  = 0
        self.acked      = 0
        self.nacked     = 0

    def send(self, msg_id: str, payload: Any):
        with self._lock:
            self._messages.append((msg_id, payload))
            self.published += 1

    def receive(self) -> Optional[Tuple[str, Any]]:
        """Returns (receipt_handle, payload) or None. Message in-flight until ack."""
        with self._lock:
            if not self._messages:
                return None
            msg_id, payload = self._messages.popleft()
            receipt = str(uuid.uuid4())[:8]
            self._inflight[receipt] = (msg_id, payload)
            return receipt, payload

    def ack(self, receipt: str):
        """Message processed — delete it."""
        with self._lock:
            self._inflight.pop(receipt, None)
            self.acked += 1

    def nack(self, receipt: str):
        """Message failed — return to queue."""
        with self._lock:
            item = self._inflight.pop(receipt, None)
            if item:
                self._messages.appendleft(item)
                self.nacked += 1

    def depth(self) -> int:
        return len(self._messages)


# ─────────────────────────────────────────────
# SIMULATED STREAM (Kafka-like)
# ─────────────────────────────────────────────

class StreamPartition:
    """Append-only log. Consumers track their own offset."""

    def __init__(self):
        self._log: List[Tuple[int, Any]] = []

    def append(self, payload: Any) -> int:
        offset = len(self._log)
        self._log.append((offset, payload))
        return offset

    def read(self, from_offset: int, max_records: int = 10) -> List[Tuple[int, Any]]:
        return self._log[from_offset: from_offset + max_records]

    def end_offset(self) -> int:
        return len(self._log)


class SimpleStream:
    """
    Stream: messages retained, consumers track offsets independently.
    Multiple consumer groups each get all messages.
    Replay possible by resetting offset.
    """

    def __init__(self, name: str, n_partitions: int = 3):
        self.name        = name
        self._partitions = [StreamPartition() for _ in range(n_partitions)]
        self._offsets    : Dict[str, List[int]] = {}   # group_id → [offset per partition]
        self.published   = 0

    def publish(self, key: str, payload: Any) -> Tuple[int, int]:
        """Route by key hash to partition."""
        partition_idx = hash(key) % len(self._partitions)
        offset = self._partitions[partition_idx].append(payload)
        self.published += 1
        return partition_idx, offset

    def subscribe(self, group_id: str, from_beginning: bool = False):
        if group_id not in self._offsets:
            if from_beginning:
                self._offsets[group_id] = [0] * len(self._partitions)
            else:
                # Start from end (only new messages)
                self._offsets[group_id] = [p.end_offset()
                                            for p in self._partitions]

    def poll(self, group_id: str, max_per_partition: int = 5) -> List[Any]:
        """Return new messages since last poll for this group."""
        if group_id not in self._offsets:
            return []
        results = []
        for i, partition in enumerate(self._partitions):
            offset   = self._offsets[group_id][i]
            records  = partition.read(offset, max_per_partition)
            for rec_offset, payload in records:
                results.append(payload)
            if records:
                self._offsets[group_id][i] = records[-1][0] + 1
        return results

    def seek_to_beginning(self, group_id: str):
        """Replay all messages from offset 0."""
        self._offsets[group_id] = [0] * len(self._partitions)

    def lag(self, group_id: str) -> int:
        """How many unread messages for this consumer group."""
        if group_id not in self._offsets:
            return sum(p.end_offset() for p in self._partitions)
        total = 0
        for i, partition in enumerate(self._partitions):
            total += partition.end_offset() - self._offsets[group_id][i]
        return total


# ─────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────

@dataclass
class UseCase:
    name           : str
    multi_consumer : bool    # multiple independent consumers?
    replay_needed  : bool    # need to re-read past messages?
    ordering_needed: bool    # per-key ordering required?
    retention_after : bool   # keep message after processing?
    is_task        : bool    # work to do (task) vs event that happened?


def recommend(uc: UseCase) -> Tuple[str, List[str]]:
    score_stream = 0
    score_queue  = 0
    reasons      = []

    if uc.multi_consumer:
        score_stream += 2
        reasons.append("Multiple consumers → Stream (each group independent)")
    else:
        score_queue += 1
        reasons.append("Single consumer → Queue (simpler)")

    if uc.replay_needed:
        score_stream += 3
        reasons.append("Replay required → Stream only")
    if uc.retention_after:
        score_stream += 2
        reasons.append("Retain after consume → Stream")
    if uc.ordering_needed:
        score_stream += 1
        reasons.append("Ordering required → Stream (partition by key)")
    if uc.is_task:
        score_queue += 2
        reasons.append("Task/job semantics → Queue (done-once, delete-on-ack)")
    else:
        score_stream += 1
        reasons.append("Event semantics → Stream (record what happened)")

    recommendation = "Stream (Kafka/Kinesis)" if score_stream > score_queue else "Queue (SQS/RabbitMQ)"
    return recommendation, reasons


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_queue_vs_stream():
    print("=" * 65)
    print("QUEUE vs STREAM — DECISION FRAMEWORK")
    print("=" * 65)

    # ── Queue Demo ────────────────────────────────
    print("\n[1] MESSAGE QUEUE — TASK DISTRIBUTION")
    print("─" * 55)
    queue = SimpleQueue("image-resize")
    for i in range(8):
        queue.send(f"msg-{i}", {"image": f"img-{i:02d}.jpg"})

    print(f"  Published: {queue.published}")
    worker_work : Dict[str, List] = defaultdict(list)

    for worker_id in ["W1", "W2", "W3"]:
        while queue.depth() > 0:
            item = queue.receive()
            if not item:
                break
            receipt, payload = item
            worker_work[worker_id].append(payload["image"])
            queue.ack(receipt)

    for worker_id, work in worker_work.items():
        print(f"  {worker_id}: {work}")
    print(f"  Each image processed exactly once. Remaining: {queue.depth()}")
    print(f"  → No replay possible. Queue is now empty.")

    # ── Stream Demo ───────────────────────────────
    print("\n\n[2] STREAM — MULTIPLE CONSUMERS + REPLAY")
    print("─" * 55)
    stream = SimpleStream("order-events", n_partitions=3)

    # Publish 9 order events
    for i in range(9):
        stream.publish(f"user-{i % 3}", {"order_id": f"ORD-{i:03d}", "amount": (i+1)*10})

    # Two independent consumer groups subscribe
    stream.subscribe("billing-group",   from_beginning=True)
    stream.subscribe("analytics-group", from_beginning=True)

    billing_msgs   = stream.poll("billing-group")
    analytics_msgs = stream.poll("analytics-group")
    print(f"  billing-group consumed  : {len(billing_msgs)} messages")
    print(f"  analytics-group consumed: {len(analytics_msgs)} messages")
    print(f"  Both groups got all {stream.published} events independently")

    # Replay: billing re-processes from beginning
    stream.seek_to_beginning("billing-group")
    replayed = stream.poll("billing-group")
    print(f"\n  After seek_to_beginning: billing replayed {len(replayed)} events")
    print(f"  → Stream still has all data. Replay is free.")

    # Consumer lag
    stream.subscribe("new-audit-group", from_beginning=True)
    print(f"\n  New 'audit-group' joined mid-stream: lag={stream.lag('new-audit-group')} msgs")
    stream.poll("new-audit-group")
    print(f"  After first poll: lag={stream.lag('new-audit-group')} msgs")

    # ── Decision Framework ────────────────────────
    print("\n\n[3] DECISION FRAMEWORK — USE CASE ANALYSIS")
    print("─" * 55)
    use_cases = [
        UseCase("Send one email per user",
                multi_consumer=False, replay_needed=False,
                ordering_needed=False, retention_after=False, is_task=True),
        UseCase("Order events → billing + inventory + analytics",
                multi_consumer=True, replay_needed=False,
                ordering_needed=False, retention_after=True, is_task=False),
        UseCase("Resize uploaded images",
                multi_consumer=False, replay_needed=False,
                ordering_needed=False, retention_after=False, is_task=True),
        UseCase("Event sourcing write model",
                multi_consumer=True, replay_needed=True,
                ordering_needed=True, retention_after=True, is_task=False),
        UseCase("Audit log all API calls",
                multi_consumer=True, replay_needed=True,
                ordering_needed=False, retention_after=True, is_task=False),
        UseCase("Charge credit card once per order",
                multi_consumer=False, replay_needed=False,
                ordering_needed=False, retention_after=False, is_task=True),
    ]

    for uc in use_cases:
        rec, reasons = recommend(uc)
        print(f"\n  Use case: '{uc.name}'")
        print(f"  → Recommendation: {rec}")
        for r in reasons:
            print(f"     • {r}")

    # ── Comparison Table ──────────────────────────
    print("\n\n[4] QUEUE vs STREAM COMPARISON")
    print("─" * 55)
    rows = [
        ("Retention",        "Deleted on ACK",             "Retained (configurable TTL)"),
        ("Multiple consumers","Fan-out (copy to each)",    "Consumer groups (shared log)"),
        ("Replay",           "Not possible",               "Possible (seek offset)"),
        ("Ordering",         "FIFO (limited)",             "Per-partition (guaranteed)"),
        ("Throughput",       "10k-100k msg/s",             "Millions msg/s (Kafka)"),
        ("Complexity",       "Lower",                      "Higher"),
        ("Best for",         "Tasks, jobs, commands",      "Events, logs, analytics"),
        ("Examples",         "SQS, RabbitMQ, Celery",      "Kafka, Kinesis, Redpanda"),
    ]
    print(f"  {'Aspect':<20} {'Queue':<30} {'Stream'}")
    print(f"  {'─'*72}")
    for aspect, queue_val, stream_val in rows:
        print(f"  {aspect:<20} {queue_val:<30} {stream_val}")

    print("\n\n[5] QUICK DECISION RULE")
    print("─" * 55)
    rules = [
        ("Need replay?",                      "Yes → Stream"),
        ("Multiple independent consumers?",   "Yes → Stream"),
        ("Retain after processing?",          "Yes → Stream"),
        ("Task/job — do once, delete?",       "Yes → Queue"),
        ("Complex routing (type/headers)?",   "Yes → RabbitMQ"),
        ("Highest throughput (>500k/s)?",     "Yes → Kafka/Kinesis"),
    ]
    for question, answer in rules:
        print(f"  {question:<42} {answer}")


if __name__ == "__main__":
    demonstrate_queue_vs_stream()
