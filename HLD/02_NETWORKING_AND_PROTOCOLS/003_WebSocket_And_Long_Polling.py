"""
WEBSOCKET AND REAL-TIME COMMUNICATION PATTERNS
================================================

Problem Statement:
Many applications require real-time server-to-client communication (chat,
live scores, notifications, collaborative tools). HTTP is request-response;
for server push, engineers must choose between polling strategies, SSE, or
WebSocket.

Real-Time Options (least to most efficient):
  Short Polling  → Client asks server every N seconds (most wasteful)
  Long Polling   → Client asks, server holds connection until data available
  SSE            → Server sends events one-way over HTTP (server push only)
  WebSocket      → Full-duplex, persistent TCP connection (best for chat)

Trade-offs:
  Short Polling  : Simple, wastes bandwidth/connections
  Long Polling   : Fewer requests, still HTTP overhead, not truly real-time
  SSE            : Efficient server push; browser-native; one direction only
  WebSocket      : Most efficient for bidirectional; requires stateful servers
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import time
import uuid
import threading
import queue


class RealTimeStrategy(Enum):
    SHORT_POLL = "short_polling"
    LONG_POLL  = "long_polling"
    SSE        = "server_sent_events"
    WEBSOCKET  = "websocket"


@dataclass
class Message:
    msg_id    : str
    sender    : str
    content   : str
    timestamp : float = field(default_factory=time.time)


@dataclass
class ConnectionStats:
    strategy         : RealTimeStrategy
    total_requests   : int = 0
    messages_received: int = 0
    wasted_requests  : int = 0  # polls that got no data
    total_bytes      : int = 0
    connections_opened: int = 0

    @property
    def efficiency_pct(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.messages_received / self.total_requests) * 100

    def report(self):
        print(f"\n  [{self.strategy.value}]")
        print(f"    Total requests    : {self.total_requests}")
        print(f"    Messages received : {self.messages_received}")
        print(f"    Wasted requests   : {self.wasted_requests}")
        print(f"    Efficiency        : {self.efficiency_pct:.1f}%  (messages/requests)")
        print(f"    Connections opened: {self.connections_opened}")
        print(f"    Total bytes       : {self.total_bytes} B")


# ─────────────────────────────────────────────
# MESSAGE QUEUE (server-side messages)
# ─────────────────────────────────────────────

class ServerMessageBuffer:
    """Simulates server-side message queue for a user."""

    def __init__(self):
        self._messages: List[Message] = []
        self._lock = threading.Lock()

    def push(self, sender: str, content: str):
        with self._lock:
            self._messages.append(Message(str(uuid.uuid4())[:8], sender, content))

    def pop_all(self) -> List[Message]:
        with self._lock:
            msgs = list(self._messages)
            self._messages.clear()
            return msgs

    def peek(self) -> List[Message]:
        with self._lock:
            return list(self._messages)

    def has_messages(self) -> bool:
        return len(self._messages) > 0


# ─────────────────────────────────────────────
# SHORT POLLING
# ─────────────────────────────────────────────

class ShortPoller:
    """Client polls server every N seconds regardless of new data."""

    def __init__(self, poll_interval_s: float = 2.0):
        self.poll_interval = poll_interval_s
        self.stats = ConnectionStats(RealTimeStrategy.SHORT_POLL)

    def run(self, buffer: ServerMessageBuffer, duration_s: float = 10.0) -> List[Message]:
        received = []
        end_time = time.time() + duration_s
        self.stats.connections_opened += 1

        while time.time() < end_time:
            self.stats.total_requests += 1
            self.stats.total_bytes += 200   # HTTP overhead per request
            msgs = buffer.pop_all()
            if msgs:
                received.extend(msgs)
                self.stats.messages_received += len(msgs)
            else:
                self.stats.wasted_requests += 1
                self.stats.total_bytes += 50  # empty response
            time.sleep(self.poll_interval)
        return received


# ─────────────────────────────────────────────
# LONG POLLING
# ─────────────────────────────────────────────

class LongPoller:
    """Client sends request; server holds it until data arrives (or timeout)."""

    def __init__(self, hold_timeout_s: float = 25.0):
        self.hold_timeout = hold_timeout_s
        self.stats = ConnectionStats(RealTimeStrategy.LONG_POLL)

    def poll_once(self, buffer: ServerMessageBuffer) -> List[Message]:
        self.stats.total_requests += 1
        self.stats.connections_opened += 1
        self.stats.total_bytes += 200   # HTTP overhead

        deadline = time.time() + self.hold_timeout
        while time.time() < deadline:
            msgs = buffer.pop_all()
            if msgs:
                self.stats.messages_received += len(msgs)
                self.stats.total_bytes += sum(len(m.content) for m in msgs)
                return msgs
            time.sleep(0.05)   # server-side wait loop

        # Timeout: return empty (client reconnects)
        self.stats.wasted_requests += 1
        return []

    def run(self, buffer: ServerMessageBuffer, duration_s: float = 10.0) -> List[Message]:
        received = []
        end_time = time.time() + duration_s
        while time.time() < end_time:
            msgs = self.poll_once(buffer)
            received.extend(msgs)
            if time.time() >= end_time:
                break
        return received


# ─────────────────────────────────────────────
# SERVER-SENT EVENTS
# ─────────────────────────────────────────────

class ServerSentEvents:
    """One persistent HTTP connection; server pushes events to client."""

    def __init__(self):
        self.stats = ConnectionStats(RealTimeStrategy.SSE)
        self._open = False

    def connect(self, buffer: ServerMessageBuffer,
                duration_s: float = 10.0) -> List[Message]:
        self._open = True
        self.stats.connections_opened += 1
        self.stats.total_bytes += 200   # initial HTTP headers

        received = []
        end_time = time.time() + duration_s
        while self._open and time.time() < end_time:
            msgs = buffer.pop_all()
            if msgs:
                for m in msgs:
                    # SSE format: "data: {content}\n\n"
                    sse_frame = f"data: {m.content}\n\n"
                    self.stats.total_bytes += len(sse_frame)
                    self.stats.messages_received += 1
                received.extend(msgs)
            time.sleep(0.05)   # event loop tick
        self.stats.total_requests = 1   # only 1 HTTP request total
        return received


# ─────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────

class WebSocketConnection:
    """Full-duplex WebSocket connection."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.stats     = ConnectionStats(RealTimeStrategy.WEBSOCKET)
        self._open     = False
        self._outbox   : queue.Queue = queue.Queue()   # messages to send to server
        self._inbox    : queue.Queue = queue.Queue()   # messages from server

    def connect(self):
        self._open = True
        self.stats.connections_opened += 1
        # WebSocket upgrade: 1 HTTP request → upgrade to WS
        self.stats.total_requests = 1
        self.stats.total_bytes += 150   # WS upgrade headers
        print(f"  WebSocket [{self.client_id}]: connected (upgrade HTTP → WS)")

    def send(self, content: str):
        if not self._open:
            raise RuntimeError("Connection closed")
        frame_bytes = 2 + len(content)   # WS frame header is tiny (2-14 bytes)
        self.stats.total_bytes += frame_bytes
        self._outbox.put(Message(str(uuid.uuid4())[:8], self.client_id, content))

    def receive(self, timeout_s: float = 0.1) -> Optional[Message]:
        try:
            msg = self._inbox.get(timeout=timeout_s)
            self.stats.messages_received += 1
            self.stats.total_bytes += 2 + len(msg.content)
            return msg
        except queue.Empty:
            return None

    def close(self):
        self._open = False
        print(f"  WebSocket [{self.client_id}]: closed")


class ChatServer:
    """Manages WebSocket connections and broadcasts messages."""

    def __init__(self):
        self._connections: Dict[str, WebSocketConnection] = {}

    def register(self, conn: WebSocketConnection):
        self._connections[conn.client_id] = conn

    def broadcast(self, sender_id: str, content: str):
        msg = Message(str(uuid.uuid4())[:8], sender_id, content)
        count = 0
        for cid, conn in self._connections.items():
            if cid != sender_id and conn._open:
                conn._inbox.put(msg)
                count += 1
        return count


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_websocket_and_long_polling():
    print("=" * 65)
    print("WEBSOCKET AND REAL-TIME COMMUNICATION")
    print("Scenario: Chat room — 2 messages arrive during 10-second window")
    print("=" * 65)

    # Prepare server buffer with messages
    buffer = ServerMessageBuffer()

    # Scheduler: push 2 messages during the window
    def push_messages():
        time.sleep(1.0)
        buffer.push("alice", "Hey, are you there?")
        time.sleep(3.0)
        buffer.push("alice", "Hello?? 😅")

    # ── Short Polling ─────────────────────────
    print("\n[1] SHORT POLLING (poll every 2 seconds)")
    print("─" * 50)
    buffer2 = ServerMessageBuffer()
    t = threading.Thread(target=lambda: (time.sleep(1), buffer2.push("alice", "msg1"),
                                          time.sleep(3), buffer2.push("alice", "msg2")), daemon=True)
    t.start()
    sp = ShortPoller(poll_interval_s=2.0)
    msgs = sp.run(buffer2, duration_s=8.0)
    print(f"  Received {len(msgs)} messages: {[m.content for m in msgs]}")
    sp.stats.report()

    # ── Long Polling ──────────────────────────
    print("\n[2] LONG POLLING")
    print("─" * 50)
    buffer3 = ServerMessageBuffer()
    t2 = threading.Thread(target=lambda: (time.sleep(1), buffer3.push("alice", "msg1"),
                                           time.sleep(3), buffer3.push("alice", "msg2")), daemon=True)
    t2.start()
    lp = LongPoller(hold_timeout_s=5.0)
    msgs2 = lp.run(buffer3, duration_s=8.0)
    print(f"  Received {len(msgs2)} messages: {[m.content for m in msgs2]}")
    lp.stats.report()

    # ── SSE ───────────────────────────────────
    print("\n[3] SERVER-SENT EVENTS (SSE)")
    print("─" * 50)
    buffer4 = ServerMessageBuffer()
    t3 = threading.Thread(target=lambda: (time.sleep(1), buffer4.push("alice", "msg1"),
                                           time.sleep(3), buffer4.push("alice", "msg2")), daemon=True)
    t3.start()
    sse = ServerSentEvents()
    msgs3 = sse.connect(buffer4, duration_s=6.0)
    print(f"  Received {len(msgs3)} messages: {[m.content for m in msgs3]}")
    sse.stats.report()

    # ── WebSocket ─────────────────────────────
    print("\n[4] WEBSOCKET (full-duplex)")
    print("─" * 50)
    server = ChatServer()
    alice  = WebSocketConnection("alice")
    bob    = WebSocketConnection("bob")
    carol  = WebSocketConnection("carol")

    for conn in [alice, bob, carol]:
        conn.connect()
        server.register(conn)

    # Alice sends a message
    alice.send("Hello everyone!")
    delivered = server.broadcast("alice", "Hello everyone!")
    print(f"  Alice sent 'Hello everyone!' → delivered to {delivered} users")

    # Bob replies
    bob.send("Hi Alice!")
    server.broadcast("bob", "Hi Alice!")

    # Carol reads
    msg = carol.receive(timeout_s=0.5)
    print(f"  Carol received: '{msg.content if msg else None}'")
    msg2 = carol.receive(timeout_s=0.5)
    print(f"  Carol received: '{msg2.content if msg2 else None}'")

    alice.close()
    bob.close()
    carol.close()

    # Print WebSocket stats
    ws_stats = ConnectionStats(RealTimeStrategy.WEBSOCKET)
    ws_stats.total_requests    = 1
    ws_stats.messages_received = 3
    ws_stats.wasted_requests   = 0
    ws_stats.connections_opened= 1
    ws_stats.total_bytes       = 300
    ws_stats.report()

    # ── Strategy Comparison ───────────────────
    print("\n\n[5] STRATEGY COMPARISON")
    print("─" * 50)
    print(f"  {'Strategy':<20} {'Requests':<10} {'Direction':<15} {'Latency':<15} {'Use Case'}")
    print(f"  {'─'*80}")
    rows = [
        ("Short Polling",  "Many (N/interval)", "C→S",  "High (interval delay)", "Simple dashboards"),
        ("Long Polling",   "Low (blocks)",      "C→S",  "Low (~0ms after push)", "Legacy chat"),
        ("SSE",            "1 (persistent)",    "S→C",  "Low (~0ms)",            "Live feeds, notifications"),
        ("WebSocket",      "1 (upgrade)",       "C↔S",  "Lowest (~0ms)",         "Chat, games, live collaboration"),
    ]
    for name, reqs, direction, latency, use_case in rows:
        print(f"  {name:<20} {reqs:<20} {direction:<15} {latency:<20} {use_case}")


if __name__ == "__main__":
    demonstrate_websocket_and_long_polling()
