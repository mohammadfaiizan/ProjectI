"""
MESSAGE QUEUE FUNDAMENTALS
============================

Problem Statement:
Service A calls Service B synchronously. If B is slow, A waits. If B is down,
A fails. Message queues decouple producer from consumer: A puts a message on
the queue and moves on; B processes it when ready. This enables async
communication, load leveling, and fault tolerance.

Core Concepts:

  Producer: Creates and sends messages to the queue.
  Consumer: Reads and processes messages from the queue.
  Queue   : Buffer that holds messages until consumed.
  Broker  : The server managing the queue (RabbitMQ, Kafka, SQS).

Message Delivery Semantics:
  At-most-once  : Fire and forget. Messages may be lost but never duplicated.
                  Use: metrics, analytics (loss is ok).
  At-least-once : Message delivered ≥ 1 time. Duplicates possible.
                  Use: payment events (idempotent consumer required).
  Exactly-once  : Delivered exactly once. Most complex, highest overhead.
                  Use: financial transactions (Kafka transactions).

Messaging Patterns:
  Point-to-Point (Queue):
    One producer → Queue → One consumer (competing consumers for scaling).
    Work queue pattern: distribute tasks to N workers.

  Publish-Subscribe (Topic):
    One producer → Topic → Many consumers (each gets a copy).
    Fan-out pattern: order event → inventory + billing + notification.

Queue Properties:
  Durability  : Messages persisted to disk (survive broker restart).
  Ordering    : FIFO (most queues), partitioned ordering (Kafka).
  TTL         : Message expires if not consumed within N seconds.
  DLQ         : Dead Letter Queue — where failed/expired messages go.
  Backpressure: Consumer slow → queue fills → producer throttles.

Benefits of Message Queues:
  Decoupling   : Producer and consumer evolve independently.
  Load Leveling: Queue absorbs traffic spikes; consumers process at their rate.
  Retry        : Failed messages requeued for retry.
  Fan-out      : Single event triggers multiple downstream handlers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
import time
import uuid
import threading
import random
from collections import defaultdict, deque
from enum import Enum


class DeliverySemantics(Enum):
    AT_MOST_ONCE  = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE  = "exactly_once"


class MessageStatus(Enum):
    PENDING     = "pending"
    IN_FLIGHT   = "in_flight"
    ACKED       = "acked"
    NACKED      = "nacked"
    DEAD        = "dead"


@dataclass
class Message:
    msg_id     : str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    payload    : Any = None
    topic      : str = ""
    producer_id: str = ""
    created_at : float = field(default_factory=time.time)
    ttl_s      : Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    status     : MessageStatus = MessageStatus.PENDING
    headers    : Dict[str, str] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.ttl_s is None:
            return False
        return time.time() - self.created_at > self.ttl_s

    @property
    def age_s(self) -> float:
        return time.time() - self.created_at


# ─────────────────────────────────────────────
# BASIC QUEUE
# ─────────────────────────────────────────────

class MessageQueue:
    """
    Basic FIFO message queue.
    Supports: publish, consume, ack, nack, DLQ, TTL, competing consumers.
    """

    def __init__(self, name: str, max_size: int = 10_000,
                 ack_timeout_s: float = 30.0,
                 max_retries: int = 3,
                 semantics: DeliverySemantics = DeliverySemantics.AT_LEAST_ONCE):
        self.name        = name
        self.max_size    = max_size
        self.ack_timeout = ack_timeout_s
        self.max_retries = max_retries
        self.semantics   = semantics
        self._queue      : deque = deque()
        self._in_flight  : Dict[str, tuple] = {}   # msg_id → (msg, deliver_time)
        self._dead_letter : deque = deque()
        self._lock       = threading.Lock()
        self._event      = threading.Event()
        self.published   = 0
        self.delivered   = 0
        self.acked       = 0
        self.nacked      = 0
        self.dead        = 0
        self.expired     = 0

    def publish(self, msg: Message) -> bool:
        with self._lock:
            if len(self._queue) >= self.max_size:
                return False   # Backpressure: queue full
            if msg.is_expired:
                return False
            msg.status = MessageStatus.PENDING
            self._queue.append(msg)
            self.published += 1
        self._event.set()
        return True

    def consume(self, timeout_s: float = 1.0) -> Optional[Message]:
        """
        Pull next message.
        At-most-once: message removed from queue immediately (no ack needed).
        At-least-once: message moved to in-flight; needs ack to confirm.
        """
        deadline = time.time() + timeout_s
        while True:
            with self._lock:
                # Re-enqueue expired in-flight messages (for at-least-once retry)
                if self.semantics == DeliverySemantics.AT_LEAST_ONCE:
                    self._requeue_expired_in_flight()

                # Skip expired queued messages
                while self._queue and self._queue[0].is_expired:
                    expired = self._queue.popleft()
                    self._dead_letter.append(expired)
                    self.expired += 1

                if self._queue:
                    msg = self._queue.popleft()
                    msg.status = MessageStatus.IN_FLIGHT
                    self.delivered += 1

                    if self.semantics == DeliverySemantics.AT_MOST_ONCE:
                        # Remove permanently — fire and forget
                        return msg
                    else:
                        # Keep in in-flight map until acked
                        self._in_flight[msg.msg_id] = (msg, time.time())
                        return msg

            if time.time() > deadline:
                return None
            self._event.wait(timeout=min(0.05, deadline - time.time()))
            self._event.clear()

    def ack(self, msg_id: str) -> bool:
        """Consumer signals successful processing."""
        with self._lock:
            if msg_id in self._in_flight:
                msg, _ = self._in_flight.pop(msg_id)
                msg.status = MessageStatus.ACKED
                self.acked += 1
                return True
        return False

    def nack(self, msg_id: str, requeue: bool = True):
        """Consumer signals failed processing."""
        with self._lock:
            if msg_id in self._in_flight:
                msg, _ = self._in_flight.pop(msg_id)
                msg.retry_count += 1
                self.nacked += 1
                if requeue and msg.retry_count <= self.max_retries:
                    msg.status = MessageStatus.PENDING
                    self._queue.appendleft(msg)   # re-enqueue at front
                else:
                    msg.status = MessageStatus.DEAD
                    self._dead_letter.append(msg)
                    self.dead += 1

    def _requeue_expired_in_flight(self):
        """Re-enqueue in-flight messages that timed out (for at-least-once)."""
        now = time.time()
        to_requeue = [(mid, msg, t) for mid, (msg, t) in self._in_flight.items()
                      if now - t > self.ack_timeout]
        for mid, msg, _ in to_requeue:
            del self._in_flight[mid]
            msg.retry_count += 1
            if msg.retry_count <= self.max_retries:
                msg.status = MessageStatus.PENDING
                self._queue.appendleft(msg)
            else:
                msg.status = MessageStatus.DEAD
                self._dead_letter.append(msg)
                self.dead += 1

    @property
    def depth(self) -> int:
        return len(self._queue)

    @property
    def in_flight_count(self) -> int:
        return len(self._in_flight)

    @property
    def dlq_size(self) -> int:
        return len(self._dead_letter)

    def stats(self) -> str:
        return (f"Queue[{self.name}]: depth={self.depth} "
                f"published={self.published} delivered={self.delivered} "
                f"acked={self.acked} nacked={self.nacked} "
                f"dead={self.dead} expired={self.expired}")


# ─────────────────────────────────────────────
# TOPIC (PUB/SUB)
# ─────────────────────────────────────────────

@dataclass
class Subscription:
    sub_id      : str
    consumer_id : str
    handler     : Callable
    filter_fn   : Optional[Callable] = None   # filter messages


class Topic:
    """
    Pub/Sub topic: each subscriber gets a copy of every published message.
    Fan-out pattern.
    """

    def __init__(self, name: str):
        self.name          = name
        self._subscriptions: Dict[str, Subscription] = {}
        self._message_count= 0
        self._lock         = threading.Lock()

    def subscribe(self, consumer_id: str, handler: Callable,
                  filter_fn: Callable = None) -> str:
        sub_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._subscriptions[sub_id] = Subscription(sub_id, consumer_id, handler, filter_fn)
        return sub_id

    def unsubscribe(self, sub_id: str):
        with self._lock:
            self._subscriptions.pop(sub_id, None)

    def publish(self, msg: Message) -> int:
        """Deliver message to all subscribers. Returns number of deliveries."""
        self._message_count += 1
        deliveries = 0
        with self._lock:
            subs = list(self._subscriptions.values())
        for sub in subs:
            if sub.filter_fn and not sub.filter_fn(msg):
                continue   # filtered out for this subscriber
            try:
                sub.handler(msg)
                deliveries += 1
            except Exception:
                pass
        return deliveries

    @property
    def subscriber_count(self) -> int:
        return len(self._subscriptions)

    @property
    def total_messages(self) -> int:
        return self._message_count


# ─────────────────────────────────────────────
# COMPETING CONSUMERS (Work Queue)
# ─────────────────────────────────────────────

class WorkerPool:
    """
    N consumer threads competing for messages from a single queue.
    Scale workers up/down to match queue depth.
    """

    def __init__(self, queue: MessageQueue, n_workers: int,
                 process_fn: Callable[[Message], bool],
                 simulate_failure_rate: float = 0.0):
        self.queue         = queue
        self.n_workers     = n_workers
        self.process_fn    = process_fn
        self.failure_rate  = simulate_failure_rate
        self._threads      : List[threading.Thread] = []
        self._stop_event   = threading.Event()
        self.processed     = 0
        self.failed        = 0

    def start(self):
        for i in range(self.n_workers):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            t.start()
            self._threads.append(t)

    def _worker(self, worker_id: int):
        while not self._stop_event.is_set():
            msg = self.queue.consume(timeout_s=0.1)
            if not msg:
                continue
            # Simulate processing failure
            if random.random() < self.failure_rate:
                self.queue.nack(msg.msg_id)
                self.failed += 1
                continue
            # Process message
            success = self.process_fn(msg)
            if success:
                self.queue.ack(msg.msg_id)
                self.processed += 1
            else:
                self.queue.nack(msg.msg_id)
                self.failed += 1

    def stop(self):
        self._stop_event.set()


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_message_queue_fundamentals():
    print("=" * 65)
    print("MESSAGE QUEUE FUNDAMENTALS")
    print("=" * 65)

    random.seed(42)

    # ── Basic Queue (At-Least-Once) ────────────
    print("\n[1] BASIC QUEUE — AT-LEAST-ONCE DELIVERY")
    print("─" * 55)
    queue = MessageQueue("orders", max_size=1000, ack_timeout_s=5.0,
                          semantics=DeliverySemantics.AT_LEAST_ONCE)

    # Publish messages
    for i in range(5):
        msg = Message(payload={"order_id": i, "amount": i * 100}, topic="orders")
        queue.publish(msg)
    print(f"  Published 5 messages. Queue depth: {queue.depth}")

    # Consume and ack
    print(f"  Processing messages:")
    for _ in range(5):
        msg = queue.consume(timeout_s=0.5)
        if msg:
            print(f"    Consumed: order_id={msg.payload['order_id']}  msg_id={msg.msg_id}")
            queue.ack(msg.msg_id)

    print(f"  {queue.stats()}")

    # ── Retry on Failure ──────────────────────
    print("\n\n[2] NACK AND RETRY")
    print("─" * 55)
    queue2 = MessageQueue("retries", max_retries=2, semantics=DeliverySemantics.AT_LEAST_ONCE)
    msg    = Message(payload="failing-job")
    queue2.publish(msg)

    attempts = []
    for attempt in range(4):
        m = queue2.consume(timeout_s=0.1)
        if m:
            attempts.append(attempt + 1)
            print(f"  Attempt #{attempt+1}: consumed msg (retry_count={m.retry_count}) → NACK")
            queue2.nack(m.msg_id, requeue=True)
        else:
            print(f"  Attempt #{attempt+1}: no message (exhausted retries → DLQ)")
            break

    print(f"  DLQ size: {queue2.dlq_size}  (max_retries={queue2.max_retries})")

    # ── Pub/Sub ───────────────────────────────
    print("\n\n[3] PUB/SUB — FAN-OUT PATTERN")
    print("─" * 55)
    topic = Topic("order.created")

    # Multiple subscribers
    received_by : Dict[str, List] = defaultdict(list)

    def make_handler(service: str):
        def handler(msg: Message):
            received_by[service].append(msg.payload)
        return handler

    topic.subscribe("inventory-service",    make_handler("inventory"))
    topic.subscribe("billing-service",      make_handler("billing"))
    topic.subscribe("notification-service", make_handler("notification"))

    # Publish one event
    order_msg = Message(payload={"order_id": "ORD-99", "amount": 299.99},
                         topic="order.created")
    deliveries = topic.publish(order_msg)

    print(f"  Published 1 event → delivered to {deliveries} subscribers")
    for service, msgs in received_by.items():
        print(f"    {service}: received {len(msgs)} message(s) = {msgs}")

    # ── Work Queue (Competing Consumers) ───────
    print("\n\n[4] COMPETING CONSUMERS (WORK QUEUE)")
    print("─" * 55)
    work_queue  = MessageQueue("tasks", semantics=DeliverySemantics.AT_LEAST_ONCE)
    job_results : List[str] = []
    result_lock = threading.Lock()

    def process_task(msg: Message) -> bool:
        time.sleep(random.uniform(1, 5) / 1000)   # simulate work
        with result_lock:
            job_results.append(msg.payload["task"])
        return True

    # Start 3 competing workers
    pool = WorkerPool(work_queue, n_workers=3, process_fn=process_task)
    pool.start()

    # Publish 20 tasks
    for i in range(20):
        work_queue.publish(Message(payload={"task": f"task-{i}"}))
    print(f"  Published 20 tasks. Workers: {pool.n_workers}")

    time.sleep(0.3)   # let workers process
    pool.stop()

    print(f"  Processed: {pool.processed}  Failed: {pool.failed}")
    print(f"  Remaining in queue: {work_queue.depth}")
    print(f"  Work distributed across 3 competing consumers")

    # ── Backpressure ──────────────────────────
    print("\n\n[5] BACKPRESSURE — QUEUE FULL")
    print("─" * 55)
    small_queue = MessageQueue("bounded", max_size=5)
    accepted = rejected = 0
    for i in range(10):
        msg = Message(payload=f"msg-{i}")
        if small_queue.publish(msg):
            accepted += 1
        else:
            rejected += 1
    print(f"  Published 10 messages to queue(max_size=5): "
          f"accepted={accepted}  rejected={rejected} (backpressure)")

    # ── Delivery Semantics ─────────────────────
    print("\n\n[6] DELIVERY SEMANTICS COMPARISON")
    print("─" * 55)
    semantics_table = [
        ("At-most-once",  "Fire and forget (no ack)",       "May lose messages", "Metrics, logging"),
        ("At-least-once", "Ack-based retry on failure",     "Duplicates possible","Orders, payments (w/ idempotency)"),
        ("Exactly-once",  "Txn + dedup on consumer side",   "Complex, slow",     "Financial ledgers"),
    ]
    print(f"  {'Semantic':<18} {'Mechanism':<35} {'Risk':<22} {'Use Case'}")
    print(f"  {'─'*85}")
    for sem, mech, risk, use_case in semantics_table:
        print(f"  {sem:<18} {mech:<35} {risk:<22} {use_case}")

    # ── Queue vs Direct Call ───────────────────
    print("\n\n[7] ASYNC QUEUE vs SYNCHRONOUS CALL")
    print("─" * 55)
    comparison = [
        ("Coupling",     "Tight (caller waits)",    "Loose (decouple)"),
        ("Failure mode", "Cascade failure",          "Buffer absorbs spike"),
        ("Latency",      "Sum of all service latency","Producer unblocked"),
        ("Throughput",   "Min(all services)",         "Max(producer rate)"),
        ("Ordering",     "Implicit (call stack)",    "FIFO or partitioned"),
        ("Replay",       "Not possible",             "Yes (with retention)"),
        ("Tracing",      "Easy (stack trace)",       "Needs correlation ID"),
    ]
    print(f"  {'Aspect':<16} {'Synchronous Call':<28} {'Message Queue'}")
    print(f"  {'─'*65}")
    for aspect, sync, async_mq in comparison:
        print(f"  {aspect:<16} {sync:<28} {async_mq}")


if __name__ == "__main__":
    demonstrate_message_queue_fundamentals()
