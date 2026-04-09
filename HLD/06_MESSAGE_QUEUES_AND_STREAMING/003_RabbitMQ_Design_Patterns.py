"""
RABBITMQ DESIGN PATTERNS
===========================

Problem Statement:
RabbitMQ implements the AMQP protocol with powerful routing logic.
Unlike Kafka (append-only log), RabbitMQ is a traditional message broker:
messages are routed through exchanges to queues based on routing rules,
then consumed and deleted. Best for task queues, RPC, and complex routing.

RabbitMQ Core Components:

  Exchange  : Messages are published to exchanges, not queues directly.
              Exchange routes message to one or more queues based on type + bindings.
  Binding   : Rule connecting exchange to a queue (routing key pattern).
  Queue     : Buffer where messages wait for consumers.
  Routing Key: String metadata attached to messages; exchange uses it for routing.

Exchange Types:

  Direct Exchange:
    Message → queue whose binding key exactly matches routing key.
    Use: task routing (routing_key="email" → email queue).

  Fanout Exchange:
    Message → ALL bound queues (ignores routing key).
    Use: broadcast (one event → multiple independent consumers).

  Topic Exchange:
    Pattern matching on routing key using * (one word) and # (zero+words).
    "order.*" matches "order.placed", "order.shipped".
    "#.error" matches "app.service.error".
    Use: log aggregation, fine-grained event routing.

  Headers Exchange:
    Route based on message headers (key-value), not routing key.
    Use: complex routing logic based on multiple attributes.

Patterns:

  Work Queue (Competing Consumers):
    Single queue, multiple consumers. Tasks distributed across workers.
    Fair dispatch: prefetch=1 ensures workers get one at a time.

  Publish/Subscribe:
    Fanout exchange → N queues (one per subscriber).
    Each subscriber gets all messages independently.

  Routing:
    Direct/Topic exchange → route to specific queues based on content.

  RPC (Request-Reply):
    Client publishes to RPC queue with reply_to and correlation_id headers.
    Server processes and replies to reply_to queue.
    Client correlates response by correlation_id.

  Dead Letter Exchange (DLX):
    Queue with x-dead-letter-exchange.
    Failed/expired messages routed to DLX → DLQ for investigation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
import time
import uuid
import threading
import random
import fnmatch
from collections import defaultdict, deque
from enum import Enum


class ExchangeType(Enum):
    DIRECT  = "direct"
    FANOUT  = "fanout"
    TOPIC   = "topic"
    HEADERS = "headers"


# ─────────────────────────────────────────────
# AMQP MESSAGE
# ─────────────────────────────────────────────

@dataclass
class AMQPMessage:
    body          : Any
    routing_key   : str = ""
    exchange      : str = ""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    reply_to      : str = ""
    headers       : Dict[str, str] = field(default_factory=dict)
    expiration_ms : Optional[int] = None
    delivery_mode : int = 2   # 1=transient, 2=persistent
    timestamp     : float = field(default_factory=time.time)
    priority      : int = 0
    delivery_tag  : int = 0

    @property
    def is_expired(self) -> bool:
        if self.expiration_ms is None:
            return False
        return (time.time() - self.timestamp) * 1000 > self.expiration_ms


# ─────────────────────────────────────────────
# RABBITMQ QUEUE
# ─────────────────────────────────────────────

class RabbitQueue:
    """
    AMQP queue: holds messages, supports ack/nack, DLX routing, TTL.
    """

    def __init__(self, name: str, durable: bool = True,
                 max_length: int = None, ttl_ms: int = None,
                 dead_letter_exchange: str = None,
                 dead_letter_routing_key: str = None):
        self.name        = name
        self.durable     = durable
        self.max_length  = max_length
        self.ttl_ms      = ttl_ms
        self.dlx         = dead_letter_exchange
        self.dlx_key     = dead_letter_routing_key or name
        self._messages   : deque = deque()
        self._in_flight  : Dict[int, AMQPMessage] = {}
        self._lock       = threading.Lock()
        self._tag_counter= 0
        self._dlq        : deque = deque()
        self.delivered   = 0
        self.acked       = 0
        self.nacked      = 0
        self.dead        = 0

    def enqueue(self, msg: AMQPMessage) -> bool:
        with self._lock:
            if self.max_length and len(self._messages) >= self.max_length:
                return False
            if msg.is_expired or (self.ttl_ms and
                                   (time.time() - msg.timestamp) * 1000 > self.ttl_ms):
                self._dlq.append(msg)
                self.dead += 1
                return False
            self._messages.append(msg)
            return True

    def dequeue(self) -> Optional[AMQPMessage]:
        with self._lock:
            if not self._messages:
                return None
            msg = self._messages.popleft()
            if msg.is_expired:
                self._dlq.append(msg)
                self.dead += 1
                return None
            self._tag_counter += 1
            msg.delivery_tag = self._tag_counter
            self._in_flight[msg.delivery_tag] = msg
            self.delivered += 1
            return msg

    def basic_ack(self, delivery_tag: int) -> bool:
        with self._lock:
            msg = self._in_flight.pop(delivery_tag, None)
            if msg:
                self.acked += 1
                return True
        return False

    def basic_nack(self, delivery_tag: int, requeue: bool = True):
        with self._lock:
            msg = self._in_flight.pop(delivery_tag, None)
            if msg:
                self.nacked += 1
                if requeue:
                    self._messages.appendleft(msg)
                else:
                    self._dlq.append(msg)
                    self.dead += 1

    def depth(self) -> int:
        return len(self._messages)

    def in_flight(self) -> int:
        return len(self._in_flight)


# ─────────────────────────────────────────────
# EXCHANGES
# ─────────────────────────────────────────────

@dataclass
class Binding:
    queue_name  : str
    routing_key : str   # for direct/topic; empty for fanout
    headers_match: Dict[str, str] = field(default_factory=dict)  # for headers exchange


class Exchange:
    """Base exchange: routes messages to queues based on bindings."""

    def __init__(self, name: str, exchange_type: ExchangeType, durable: bool = True):
        self.name     = name
        self.type     = exchange_type
        self.durable  = durable
        self._bindings: List[Binding] = []

    def bind(self, queue_name: str, routing_key: str = "",
              headers: Dict[str, str] = None):
        self._bindings.append(Binding(queue_name, routing_key, headers or {}))

    def get_queues(self, msg: AMQPMessage) -> List[str]:
        """Return queue names that should receive this message."""
        raise NotImplementedError


class DirectExchange(Exchange):
    def __init__(self, name: str):
        super().__init__(name, ExchangeType.DIRECT)

    def get_queues(self, msg: AMQPMessage) -> List[str]:
        return [b.queue_name for b in self._bindings
                if b.routing_key == msg.routing_key]


class FanoutExchange(Exchange):
    def __init__(self, name: str):
        super().__init__(name, ExchangeType.FANOUT)

    def get_queues(self, msg: AMQPMessage) -> List[str]:
        return [b.queue_name for b in self._bindings]


class TopicExchange(Exchange):
    """
    Routing key pattern matching:
    * = exactly one word
    # = zero or more words
    Example: "order.*" matches "order.placed", "order.shipped"
    """

    def __init__(self, name: str):
        super().__init__(name, ExchangeType.TOPIC)

    def _match(self, pattern: str, routing_key: str) -> bool:
        # Convert AMQP topic pattern to fnmatch glob
        glob = pattern.replace("#", "**").replace("*", "[^.]*")
        # Custom match: # matches zero or more .words
        parts_p = pattern.split(".")
        parts_k = routing_key.split(".")
        return self._amqp_match(parts_p, parts_k)

    def _amqp_match(self, pattern: List[str], key: List[str]) -> bool:
        if not pattern and not key:
            return True
        if pattern and pattern[0] == "#":
            # # matches zero or more words
            return (self._amqp_match(pattern[1:], key) or
                    (key and self._amqp_match(pattern, key[1:])))
        if not pattern or not key:
            return False
        if pattern[0] == "*" or pattern[0] == key[0]:
            return self._amqp_match(pattern[1:], key[1:])
        return False

    def get_queues(self, msg: AMQPMessage) -> List[str]:
        return [b.queue_name for b in self._bindings
                if self._match(b.routing_key, msg.routing_key)]


class HeadersExchange(Exchange):
    def __init__(self, name: str):
        super().__init__(name, ExchangeType.HEADERS)

    def get_queues(self, msg: AMQPMessage) -> List[str]:
        matched = []
        for b in self._bindings:
            if all(msg.headers.get(k) == v for k, v in b.headers_match.items()):
                matched.append(b.queue_name)
        return matched


# ─────────────────────────────────────────────
# RABBITMQ BROKER
# ─────────────────────────────────────────────

class RabbitMQBroker:
    """
    Simulates a RabbitMQ broker.
    Manages exchanges, queues, bindings, and message routing.
    """

    def __init__(self):
        self._exchanges : Dict[str, Exchange]    = {}
        self._queues    : Dict[str, RabbitQueue] = {}
        self.routed     = 0
        self.unrouted   = 0

    def declare_exchange(self, exchange: Exchange):
        self._exchanges[exchange.name] = exchange

    def declare_queue(self, queue: RabbitQueue):
        self._queues[queue.name] = queue

    def bind(self, exchange_name: str, queue_name: str,
             routing_key: str = "", headers: Dict[str, str] = None):
        exchange = self._exchanges.get(exchange_name)
        if exchange:
            exchange.bind(queue_name, routing_key, headers)

    def publish(self, exchange_name: str, msg: AMQPMessage) -> int:
        exchange = self._exchanges.get(exchange_name)
        if not exchange:
            self.unrouted += 1
            return 0
        msg.exchange = exchange_name
        target_queues = exchange.get_queues(msg)
        delivered = 0
        for q_name in target_queues:
            queue = self._queues.get(q_name)
            if queue and queue.enqueue(msg):
                delivered += 1
        if delivered == 0:
            self.unrouted += 1
        else:
            self.routed += 1
        return delivered

    def basic_get(self, queue_name: str) -> Optional[AMQPMessage]:
        queue = self._queues.get(queue_name)
        return queue.dequeue() if queue else None

    def basic_ack(self, queue_name: str, delivery_tag: int):
        queue = self._queues.get(queue_name)
        if queue:
            queue.basic_ack(delivery_tag)

    def basic_nack(self, queue_name: str, delivery_tag: int, requeue: bool = True):
        queue = self._queues.get(queue_name)
        if queue:
            queue.basic_nack(delivery_tag, requeue)

    def queue_stats(self) -> Dict[str, Dict]:
        return {
            name: {"depth": q.depth(), "in_flight": q.in_flight(),
                   "acked": q.acked, "dead": q.dead}
            for name, q in self._queues.items()
        }


# ─────────────────────────────────────────────
# RPC PATTERN
# ─────────────────────────────────────────────

class RPCClient:
    """RPC over RabbitMQ: send request, await reply on correlation ID."""

    def __init__(self, broker: RabbitMQBroker, rpc_exchange: str):
        self.broker      = broker
        self.rpc_exchange= rpc_exchange
        self.reply_queue = f"rpc.reply.{uuid.uuid4().hex[:6]}"
        broker.declare_queue(RabbitQueue(self.reply_queue, durable=False))
        self._pending    : Dict[str, Any] = {}

    def call(self, routing_key: str, payload: Any, timeout_s: float = 2.0) -> Optional[Any]:
        corr_id = str(uuid.uuid4())[:8]
        msg     = AMQPMessage(
            body=payload, routing_key=routing_key,
            reply_to=self.reply_queue, correlation_id=corr_id
        )
        self.broker.publish(self.rpc_exchange, msg)

        # Wait for reply
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            reply = self.broker.basic_get(self.reply_queue)
            if reply and reply.correlation_id == corr_id:
                return reply.body
            time.sleep(0.01)
        return None   # timeout


class RPCServer:
    """Processes RPC requests and publishes replies."""

    def __init__(self, broker: RabbitMQBroker, request_queue: str):
        self.broker        = broker
        self.request_queue = request_queue
        self.processed     = 0

    def process_one(self, handler: Callable) -> bool:
        msg = self.broker.basic_get(self.request_queue)
        if not msg:
            return False
        result = handler(msg.body)
        if msg.reply_to:
            reply = AMQPMessage(body=result, correlation_id=msg.correlation_id)
            self.broker.publish("", reply)   # direct to queue
            # Simplified: directly enqueue reply
            reply_q = self.broker._queues.get(msg.reply_to)
            if reply_q:
                reply_q.enqueue(reply)
        self.broker.basic_ack(self.request_queue, msg.delivery_tag)
        self.processed += 1
        return True


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_rabbitmq():
    print("=" * 65)
    print("RABBITMQ DESIGN PATTERNS")
    print("=" * 65)

    broker = RabbitMQBroker()

    # ── Direct Exchange (Work Queue) ──────────
    print("\n[1] DIRECT EXCHANGE — WORK QUEUE")
    print("─" * 55)
    direct_ex = DirectExchange("tasks")
    broker.declare_exchange(direct_ex)
    broker.declare_queue(RabbitQueue("email-queue"))
    broker.declare_queue(RabbitQueue("sms-queue"))
    broker.bind("tasks", "email-queue", routing_key="email")
    broker.bind("tasks", "sms-queue",   routing_key="sms")

    # Publish
    for i in range(3):
        broker.publish("tasks", AMQPMessage(body=f"Email job {i}", routing_key="email"))
        broker.publish("tasks", AMQPMessage(body=f"SMS job {i}",   routing_key="sms"))

    print(f"  Published 3 email + 3 SMS tasks")
    stats = broker.queue_stats()
    for q, s in stats.items():
        print(f"  {q}: depth={s['depth']}")

    # Consume email
    print(f"\n  Processing email queue:")
    for _ in range(3):
        msg = broker.basic_get("email-queue")
        if msg:
            print(f"    Processed: {msg.body}")
            broker.basic_ack("email-queue", msg.delivery_tag)

    # ── Fanout Exchange (Pub/Sub) ─────────────
    print("\n\n[2] FANOUT EXCHANGE — ORDER EVENT FAN-OUT")
    print("─" * 55)
    fanout_ex = FanoutExchange("order.events")
    broker.declare_exchange(fanout_ex)
    broker.declare_queue(RabbitQueue("inventory-q"))
    broker.declare_queue(RabbitQueue("billing-q"))
    broker.declare_queue(RabbitQueue("notification-q"))
    broker.bind("order.events", "inventory-q")
    broker.bind("order.events", "billing-q")
    broker.bind("order.events", "notification-q")

    order_msg = AMQPMessage(body={"order_id": "ORD-123", "amount": 299.99})
    n_delivered = broker.publish("order.events", order_msg)
    print(f"  Published 1 order event → delivered to {n_delivered} queues")
    for q in ["inventory-q", "billing-q", "notification-q"]:
        msg = broker.basic_get(q)
        if msg:
            print(f"  {q}: received order_id={msg.body.get('order_id', msg.body)}")

    # ── Topic Exchange (Pattern Routing) ──────
    print("\n\n[3] TOPIC EXCHANGE — PATTERN ROUTING")
    print("─" * 55)
    topic_ex = TopicExchange("logs")
    broker.declare_exchange(topic_ex)
    broker.declare_queue(RabbitQueue("all-errors"))
    broker.declare_queue(RabbitQueue("order-logs"))
    broker.declare_queue(RabbitQueue("critical-all"))
    broker.bind("logs", "all-errors",   routing_key="*.error")
    broker.bind("logs", "order-logs",   routing_key="order.#")
    broker.bind("logs", "critical-all", routing_key="#.critical")

    log_events = [
        ("order.placed",    "order placed"),
        ("order.error",     "payment failed"),
        ("service.error",   "DB timeout"),
        ("order.critical",  "inventory gone"),
        ("auth.critical",   "breach detected"),
        ("info.message",    "health check ok"),
    ]
    print(f"  Routing keys and target queues:")
    for rk, body in log_events:
        msg = AMQPMessage(body=body, routing_key=rk)
        n   = broker.publish("logs", msg)
        print(f"    {rk:<25} → {n} queue(s)")

    print(f"\n  Queue depths after routing:")
    for q in ["all-errors", "order-logs", "critical-all"]:
        q_obj = broker._queues.get(q)
        print(f"    {q}: {q_obj.depth() if q_obj else 0} messages")

    # ── RPC Pattern ───────────────────────────
    print("\n\n[4] RPC PATTERN (request-reply)")
    print("─" * 55)
    rpc_direct = DirectExchange("rpc")
    broker.declare_exchange(rpc_direct)
    broker.declare_queue(RabbitQueue("calculator-rpc"))
    broker.bind("rpc", "calculator-rpc", routing_key="calculator")

    rpc_server = RPCServer(broker, "calculator-rpc")
    rpc_client = RPCClient(broker, "rpc")

    # Send RPC requests
    for a, b in [(10, 5), (100, 7), (42, 8)]:
        rpc_client.broker.publish("rpc",
            AMQPMessage(body={"a": a, "b": b, "op": "add"},
                        routing_key="calculator",
                        reply_to=rpc_client.reply_queue,
                        correlation_id="corr-" + str(a)))

    # Server processes
    def calculator(body):
        return {"result": body["a"] + body["b"]}

    for _ in range(3):
        rpc_server.process_one(calculator)

    print(f"  RPC server processed {rpc_server.processed} requests")

    # ── DLQ Pattern ───────────────────────────
    print("\n\n[5] DEAD LETTER QUEUE (DLQ)")
    print("─" * 55)
    dlq = RabbitQueue("payment.dlq")
    payment_q = RabbitQueue("payment.process", max_length=5,
                             dead_letter_exchange="dlx",
                             dead_letter_routing_key="payment.failed")
    broker.declare_queue(dlq)
    broker.declare_queue(payment_q)

    # Publish 8 (max_length=5 → some go to DLQ)
    for i in range(8):
        broker.declare_queue(payment_q)
        payment_q.enqueue(AMQPMessage(body=f"payment-{i}"))

    # Nack 2 messages (simulating failure → DLQ)
    for _ in range(2):
        msg = payment_q.dequeue()
        if msg:
            payment_q.basic_nack(msg.delivery_tag, requeue=False)

    print(f"  payment.process: depth={payment_q.depth()}  dead={payment_q.dead}")
    print(f"  DLQ depth: {dlq.depth()}")

    # ── Exchange Comparison ────────────────────
    print("\n\n[6] EXCHANGE TYPE COMPARISON")
    print("─" * 55)
    types = [
        ("Direct",   "Exact routing key match",     "Work queues, task routing"),
        ("Fanout",   "All bound queues (broadcast)", "Event fan-out, pub/sub"),
        ("Topic",    "Pattern match (*, #)",         "Log routing, fine-grained events"),
        ("Headers",  "Header key-value match",       "Complex routing by attributes"),
    ]
    print(f"  {'Type':<10} {'Routing Logic':<32} {'Use Case'}")
    print(f"  {'─'*60}")
    for t, logic, use_case in types:
        print(f"  {t:<10} {logic:<32} {use_case}")


if __name__ == "__main__":
    demonstrate_rabbitmq()
