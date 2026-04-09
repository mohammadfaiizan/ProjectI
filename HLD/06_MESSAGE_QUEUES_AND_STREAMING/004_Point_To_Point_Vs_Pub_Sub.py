"""
POINT-TO-POINT vs PUB-SUB
============================

Problem Statement:
Two fundamental messaging models exist for async communication:
1. Point-to-Point (Queue): one producer, one consumer gets each message.
2. Publish-Subscribe (Topic): one producer, many consumers each get every message.
Choosing wrong creates coupling, missed events, or duplicated work.

Point-to-Point (Queue):
  Producer → [Queue] → Consumer A or Consumer B (competing consumers)
  Each message consumed by exactly ONE consumer.
  Used for: work distribution, task queues, load balancing.
  Examples: SQS, RabbitMQ queues, Kafka partition (within consumer group).

  Benefits:
  ✓ Natural load balancing (add consumers to scale throughput)
  ✓ Each task done exactly once (no duplication of work)
  ✓ Simple: no coordination between consumers

  Drawbacks:
  ✗ Can't broadcast to multiple processors
  ✗ Adding a new consumer type requires the producer to know about it

Publish-Subscribe (Topic):
  Producer → [Topic] → Consumer A (gets copy) + Consumer B (gets copy) + Consumer C
  Each message delivered to ALL subscribers independently.
  Used for: event fan-out, decoupled notification, event-driven architecture.
  Examples: SNS, Kafka topics (multiple consumer groups), Redis Pub/Sub.

  Benefits:
  ✓ Decoupled: producer doesn't know about consumers
  ✓ Add consumers without changing producer (Open/Closed principle)
  ✓ Each consumer builds its own view (inventory, billing, analytics)

  Drawbacks:
  ✗ All consumers must be available (or miss messages if no persistence)
  ✗ Duplicate work if processing is shared responsibility

Hybrid Pattern (Competing Consumer Groups):
  Kafka supports both simultaneously:
  Topic → Consumer Group A (one consumer in group gets each message)
         → Consumer Group B (one consumer in group gets each message)
  Each group = pub/sub delivery. Within group = point-to-point.

Choosing:
  Use P2P when: work needs to happen exactly once (email send, charge card).
  Use Pub/Sub when: event needs to trigger multiple independent actions.
  Use both when: scale within each action (competing consumers per group).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
import time
import uuid
import threading
import random
from collections import defaultdict, deque
from enum import Enum


# ─────────────────────────────────────────────
# SHARED MESSAGE MODEL
# ─────────────────────────────────────────────

@dataclass
class Event:
    event_id  : str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: str = ""
    payload   : Any = None
    timestamp : float = field(default_factory=time.time)
    producer  : str = ""


# ─────────────────────────────────────────────
# POINT-TO-POINT QUEUE
# ─────────────────────────────────────────────

class P2PQueue:
    """Point-to-point queue: competing consumers, each message consumed once."""

    def __init__(self, name: str):
        self.name      = name
        self._messages : deque = deque()
        self._lock     = threading.Lock()
        self._event    = threading.Event()
        self.published = 0
        self.consumed  = 0

    def publish(self, event: Event):
        with self._lock:
            self._messages.append(event)
            self.published += 1
        self._event.set()

    def consume(self, timeout_s: float = 0.5) -> Optional[Event]:
        """One consumer gets this message. Others will not see it."""
        deadline = time.time() + timeout_s
        while True:
            with self._lock:
                if self._messages:
                    self.consumed += 1
                    return self._messages.popleft()
            if time.time() > deadline:
                return None
            self._event.wait(timeout=min(0.01, deadline - time.time()))
            self._event.clear()

    def depth(self) -> int:
        return len(self._messages)


# ─────────────────────────────────────────────
# PUB-SUB TOPIC
# ─────────────────────────────────────────────

class Subscriber:
    def __init__(self, subscriber_id: str, filter_fn: Callable = None):
        self.subscriber_id = subscriber_id
        self.filter_fn     = filter_fn
        self._inbox        : deque = deque()
        self._lock         = threading.Lock()
        self.received      = 0

    def deliver(self, event: Event):
        if self.filter_fn and not self.filter_fn(event):
            return
        with self._lock:
            self._inbox.append(event)
            self.received += 1

    def poll(self) -> Optional[Event]:
        with self._lock:
            return self._inbox.popleft() if self._inbox else None

    def poll_all(self) -> List[Event]:
        with self._lock:
            msgs = list(self._inbox)
            self._inbox.clear()
            return msgs

    def inbox_size(self) -> int:
        return len(self._inbox)


class PubSubTopic:
    """Pub-Sub topic: each subscriber gets their own copy of every event."""

    def __init__(self, name: str, persistent: bool = True):
        self.name        = name
        self.persistent  = persistent   # if True, late subscribers get past events
        self._subscribers: Dict[str, Subscriber] = {}
        self._history    : List[Event] = []   # for persistent delivery
        self._lock       = threading.Lock()
        self.published   = 0

    def subscribe(self, subscriber: Subscriber) -> str:
        with self._lock:
            self._subscribers[subscriber.subscriber_id] = subscriber
            if self.persistent:
                # Deliver backlog to new subscriber
                for event in self._history:
                    subscriber.deliver(event)
        return subscriber.subscriber_id

    def unsubscribe(self, subscriber_id: str):
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def publish(self, event: Event) -> int:
        with self._lock:
            if self.persistent:
                self._history.append(event)
            subs = list(self._subscribers.values())
        deliveries = 0
        for sub in subs:
            sub.deliver(event)
            deliveries += 1
        self.published += 1
        return deliveries

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# ─────────────────────────────────────────────
# HYBRID: COMPETING CONSUMER GROUPS (Kafka-style)
# ─────────────────────────────────────────────

class ConsumerGroup:
    """
    Kafka-style consumer group: Pub/Sub between groups, P2P within groups.
    Topic → Group A (P2P within group) + Group B (P2P within group)
    """

    def __init__(self, group_id: str):
        self.group_id  = group_id
        self._queue    = P2PQueue(f"group-{group_id}")
        self.consumers : List[str] = []

    def add_consumer(self, consumer_id: str):
        self.consumers.append(consumer_id)

    def deliver(self, event: Event):
        """All groups get event (pub/sub). Within group: round-robin (P2P)."""
        self._queue.publish(event)

    def consume(self, consumer_id: str, timeout_s: float = 0.1) -> Optional[Event]:
        """Consumer within group competes for next message."""
        return self._queue.consume(timeout_s)

    def depth(self) -> int:
        return self._queue.depth()


class HybridTopic:
    """
    Kafka-style topic: multiple consumer groups, each group gets all events.
    Within each group, consumers compete (exactly-once delivery per group).
    """

    def __init__(self, name: str):
        self.name     = name
        self._groups  : Dict[str, ConsumerGroup] = {}
        self.published= 0

    def create_group(self, group_id: str) -> ConsumerGroup:
        group = ConsumerGroup(group_id)
        self._groups[group_id] = group
        return group

    def publish(self, event: Event) -> int:
        """Deliver to all consumer groups."""
        self.published += 1
        deliveries = 0
        for group in self._groups.values():
            group.deliver(event)
            deliveries += 1
        return deliveries

    def stats(self) -> Dict[str, Dict]:
        return {gid: {"depth": g.depth(), "consumers": len(g.consumers)}
                for gid, g in self._groups.items()}


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_p2p_vs_pubsub():
    print("=" * 65)
    print("POINT-TO-POINT vs PUB-SUB")
    print("=" * 65)

    random.seed(42)

    # ── P2P: Work Queue ───────────────────────
    print("\n[1] POINT-TO-POINT — COMPETING CONSUMERS")
    print("─" * 55)
    work_queue  = P2PQueue("image-resize")
    worker_log  : Dict[str, List] = defaultdict(list)
    stop_event  = threading.Event()

    def worker(worker_id: str):
        while not stop_event.is_set():
            event = work_queue.consume(timeout_s=0.2)
            if event:
                time.sleep(random.uniform(1, 5) / 1000)   # simulate work
                worker_log[worker_id].append(event.payload["image"])

    # Start 3 competing workers
    threads = [threading.Thread(target=worker, args=(f"worker-{i}",), daemon=True)
               for i in range(3)]
    for t in threads:
        t.start()

    # Publish 12 image resize tasks
    for i in range(12):
        event = Event(event_type="image.resize",
                       payload={"image": f"img-{i:02d}.jpg", "size": "800x600"})
        work_queue.publish(event)
        time.sleep(0.005)

    time.sleep(0.3)   # let workers process
    stop_event.set()

    print(f"  Published: {work_queue.published}  Consumed: {work_queue.consumed}")
    print(f"  Work distribution (competing consumers):")
    for worker_id, images in worker_log.items():
        bar = "█" * len(images)
        print(f"    {worker_id}: {len(images)} tasks  {bar}  {images}")
    print(f"  Each image processed exactly once. Remaining: {work_queue.depth()}")

    # ── Pub/Sub: Event Fan-out ─────────────────
    print("\n\n[2] PUB-SUB — ORDER EVENT FAN-OUT")
    print("─" * 55)
    order_topic = PubSubTopic("order.created", persistent=True)

    inventory_sub    = Subscriber("inventory-service")
    billing_sub      = Subscriber("billing-service")
    notification_sub = Subscriber("notification-service",
                                   filter_fn=lambda e: e.payload.get("amount", 0) > 100)

    order_topic.subscribe(inventory_sub)
    order_topic.subscribe(billing_sub)
    order_topic.subscribe(notification_sub)

    # Publish order events
    for i in range(5):
        amount = random.randint(50, 300)
        event  = Event(event_type="order.created",
                        payload={"order_id": f"ORD-{i:03d}", "amount": amount},
                        producer="order-service")
        n = order_topic.publish(event)
        print(f"  Published order ORD-{i:03d} amt=${amount} → {n} subscribers")

    print(f"\n  Messages per subscriber:")
    for sub in [inventory_sub, billing_sub, notification_sub]:
        events = sub.poll_all()
        print(f"    {sub.subscriber_id}: {len(events)} events "
              f"(filter active={'amount>100' if sub.filter_fn else 'none'})")

    # Late subscriber — gets backlog (persistent mode)
    analytics_sub = Subscriber("analytics-service")
    order_topic.subscribe(analytics_sub)
    backlog = analytics_sub.poll_all()
    print(f"\n  Late subscriber (analytics) joined → received {len(backlog)} backlogged events")

    # ── Hybrid: Kafka Consumer Groups ─────────
    print("\n\n[3] HYBRID — COMPETING CONSUMER GROUPS (KAFKA-STYLE)")
    print("─" * 55)
    events_topic = HybridTopic("user.events")

    order_grp  = events_topic.create_group("order-processors")
    order_grp.add_consumer("order-worker-1")
    order_grp.add_consumer("order-worker-2")

    analytics_grp = events_topic.create_group("analytics")
    analytics_grp.add_consumer("analytics-worker-1")

    notify_grp = events_topic.create_group("notifications")
    notify_grp.add_consumer("notify-worker-1")

    # Publish 6 user events
    for i in range(6):
        event = Event(event_type="user.purchase",
                       payload={"user_id": f"u-{i}", "product": f"p-{i}"})
        n = events_topic.publish(event)

    print(f"  Published 6 events to topic. Stats:")
    for gid, stats in events_topic.stats().items():
        print(f"    Group '{gid}': depth={stats['depth']} consumers={stats['consumers']}")

    # Each group consumes independently
    print(f"\n  Consuming (each group gets all 6 events):")
    for gid, group in events_topic._groups.items():
        consumed = 0
        while True:
            msg = group.consume("worker", timeout_s=0.05)
            if not msg:
                break
            consumed += 1
        print(f"    {gid}: consumed {consumed} events")

    print(f"\n  → Order group: distributed among 2 workers (P2P within group)")
    print(f"  → All 3 groups got all 6 events (pub/sub between groups)")

    # ── When to Use ───────────────────────────
    print("\n\n[4] PATTERN SELECTION GUIDE")
    print("─" * 55)
    guide = [
        ("Send email to user",           "P2P",      "Email should be sent once, not 3 times"),
        ("Order → update inventory",     "P2P",      "Inventory update once per order"),
        ("Order → billing+notif+inv",    "Pub/Sub",  "Multiple independent actions needed"),
        ("Resize uploaded images",       "P2P",      "Each image processed by one worker"),
        ("Audit log all API calls",      "Pub/Sub",  "Multiple consumers (SIEM, analytics)"),
        ("Scale email workers",          "P2P",      "Add workers, each gets different emails"),
        ("Add new downstream service",   "Pub/Sub",  "Subscribe without changing producer"),
        ("Distribute parallel work",     "P2P",      "Work done once, not N times"),
    ]
    print(f"  {'Use Case':<38} {'Pattern':<10} {'Reason'}")
    print(f"  {'─'*75}")
    for use_case, pattern, reason in guide:
        print(f"  {use_case:<38} {pattern:<10} {reason}")

    # ── Pattern Summary ───────────────────────
    print("\n\n[5] P2P vs PUB-SUB SUMMARY")
    print("─" * 55)
    comparison = [
        ("Message delivery", "One consumer gets it",     "All subscribers get it"),
        ("Scaling",          "Add competing workers",    "Add consumer groups"),
        ("Coupling",         "Producer-consumer paired", "Fully decoupled"),
        ("Durability",       "Until consumed",           "Until subscription lasts"),
        ("Best for",         "Tasks, jobs, work queues", "Events, notifications, audit"),
        ("Tools",            "SQS, RabbitMQ queue",      "SNS, Redis Pub/Sub, Kafka"),
    ]
    print(f"  {'Aspect':<20} {'Point-to-Point':<30} {'Pub-Sub'}")
    print(f"  {'─'*70}")
    for aspect, p2p, ps in comparison:
        print(f"  {aspect:<20} {p2p:<30} {ps}")


if __name__ == "__main__":
    demonstrate_p2p_vs_pubsub()
