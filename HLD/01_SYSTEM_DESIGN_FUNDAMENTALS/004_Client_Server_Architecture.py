"""
CLIENT-SERVER ARCHITECTURE
===========================

Problem Statement:
The client-server model is the foundation of virtually every web system.
Understanding the request-response cycle, connection types, and how to
scale a server to handle thousands of concurrent clients is essential.

Architecture:
    Client ──(HTTP Request)──► Server
    Client ◄─(HTTP Response)── Server

Key Concepts:
- Request-Response Cycle : Client sends request, server processes, returns response
- Stateless HTTP         : Each request is independent; no memory of prior requests
- Connection Types       : Short-lived (HTTP/1.0), Keep-Alive (HTTP/1.1), Multiplexed (HTTP/2)
- Thick vs Thin Client   : Where computation lives (client-side vs server-side)
- Connection Pooling     : Reusing expensive TCP connections
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import random
import uuid


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class HTTPMethod(Enum):
    GET    = "GET"
    POST   = "POST"
    PUT    = "PUT"
    DELETE = "DELETE"
    PATCH  = "PATCH"


class StatusCode(Enum):
    OK                  = 200
    CREATED             = 201
    NO_CONTENT          = 204
    BAD_REQUEST         = 400
    UNAUTHORIZED        = 401
    FORBIDDEN           = 403
    NOT_FOUND           = 404
    TOO_MANY_REQUESTS   = 429
    INTERNAL_SERVER_ERROR = 500
    SERVICE_UNAVAILABLE = 503


class ConnectionType(Enum):
    SHORT_LIVED  = "close"          # HTTP/1.0 — new TCP per request
    KEEP_ALIVE   = "keep-alive"     # HTTP/1.1 — reuse TCP, one request at a time
    MULTIPLEXED  = "multiplexed"    # HTTP/2   — multiple concurrent on one TCP


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class Request:
    method    : HTTPMethod
    path      : str
    client_id : str
    headers   : Dict[str, str] = field(default_factory=dict)
    body      : Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class Response:
    status      : StatusCode
    body        : str
    headers     : Dict[str, str] = field(default_factory=dict)
    latency_ms  : float = 0.0
    served_by   : str = ""


# ─────────────────────────────────────────────
# SERVER CLASSES
# ─────────────────────────────────────────────

class ConnectionPool:
    """
    Manages a pool of reusable connections to the same server.
    Avoids the overhead of TCP handshake on every request.
    """

    def __init__(self, server_host: str, max_size: int = 10):
        self.server_host = server_host
        self.max_size    = max_size
        self._idle  : List[str] = []          # connection IDs available
        self._active: Dict[str, str] = {}     # conn_id → client_id
        self._total_created = 0
        self._reuses        = 0

    def _create_connection(self) -> str:
        conn_id = f"conn-{self._total_created + 1}"
        self._total_created += 1
        return conn_id

    def acquire(self, client_id: str) -> Optional[str]:
        if self._idle:
            conn_id = self._idle.pop()
            self._reuses += 1
        elif len(self._active) < self.max_size:
            conn_id = self._create_connection()
        else:
            return None   # pool exhausted
        self._active[conn_id] = client_id
        return conn_id

    def release(self, conn_id: str):
        if conn_id in self._active:
            del self._active[conn_id]
            self._idle.append(conn_id)

    def stats(self) -> Dict:
        return {
            "total_created"   : self._total_created,
            "reuses"          : self._reuses,
            "active"          : len(self._active),
            "idle"            : len(self._idle),
            "reuse_ratio_pct" : round(self._reuses / max(1, self._total_created + self._reuses) * 100, 1),
        }


class Server:
    """
    A simulated HTTP server with route handlers and basic load tracking.
    """

    def __init__(self, name: str, processing_ms: float = 20.0):
        self.name          = name
        self.processing_ms = processing_ms
        self.requests_served = 0
        self.error_count     = 0
        self._routes: Dict[str, callable] = {}

    def register(self, path: str, handler: callable):
        self._routes[path] = handler

    def handle(self, request: Request) -> Response:
        self.requests_served += 1
        start = time.perf_counter()

        handler = self._routes.get(request.path)
        if handler is None:
            self.error_count += 1
            latency = (time.perf_counter() - start) * 1000
            return Response(StatusCode.NOT_FOUND, f"404 Not Found: {request.path}",
                            latency_ms=latency, served_by=self.name)

        try:
            body = handler(request)
            status = StatusCode.OK
        except Exception as e:
            body = str(e)
            status = StatusCode.INTERNAL_SERVER_ERROR
            self.error_count += 1

        latency = (time.perf_counter() - start) * 1000 + self.processing_ms
        return Response(status, body, latency_ms=round(latency, 2), served_by=self.name)


class LoadBalancedServer:
    """
    Round-robin load balancer across multiple backend server instances.
    """

    def __init__(self, servers: List[Server]):
        self.servers = servers
        self._index  = 0
        self._request_log: List[tuple] = []

    def handle(self, request: Request) -> Response:
        server = self.servers[self._index % len(self.servers)]
        self._index += 1
        response = server.handle(request)
        self._request_log.append((request.client_id, server.name, response.status.value))
        return response

    def stats(self):
        print("\nLoad Balancer Distribution:")
        counts = {}
        for _, srv, _ in self._request_log:
            counts[srv] = counts.get(srv, 0) + 1
        for srv, count in counts.items():
            bar = "█" * count
            print(f"  {srv:<12}: {bar} ({count})")


# ─────────────────────────────────────────────
# CLIENT CLASS
# ─────────────────────────────────────────────

class Client:
    """HTTP client that uses a connection pool to communicate with a server."""

    def __init__(self, client_id: str, pool: ConnectionPool, lb: LoadBalancedServer):
        self.client_id = client_id
        self.pool      = pool
        self.lb        = lb
        self.latencies : List[float] = []

    def send(self, method: HTTPMethod, path: str, body: str = None) -> Response:
        conn_id = self.pool.acquire(self.client_id)
        if conn_id is None:
            return Response(StatusCode.SERVICE_UNAVAILABLE, "Connection pool exhausted")

        request  = Request(method, path, self.client_id, body=body)
        response = self.lb.handle(request)
        self.latencies.append(response.latency_ms)
        self.pool.release(conn_id)
        return response

    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_client_server_architecture():
    print("=" * 60)
    print("CLIENT-SERVER ARCHITECTURE DEMONSTRATION")
    print("=" * 60)

    # ── Build 3 backend servers ───────────────
    def user_handler(req: Request) -> str:
        return f'{{"user_id": "u123", "name": "Alice", "email": "alice@example.com"}}'

    def product_handler(req: Request) -> str:
        return f'{{"product_id": "p456", "name": "Laptop", "price": 999.99}}'

    def order_handler(req: Request) -> str:
        return f'{{"order_id": "o789", "status": "shipped", "items": 3}}'

    servers = []
    for i in range(1, 4):
        s = Server(f"app-server-{i}", processing_ms=random.uniform(10, 30))
        s.register("/api/user",    user_handler)
        s.register("/api/product", product_handler)
        s.register("/api/order",   order_handler)
        servers.append(s)

    lb   = LoadBalancedServer(servers)
    pool = ConnectionPool("app-cluster", max_size=5)

    # ── Simulate 12 client requests ──────────
    print("\n[Simulating 12 requests from 4 clients across 3 routes]")
    print("─" * 55)
    clients = [Client(f"client-{i}", pool, lb) for i in range(1, 5)]
    routes  = ["/api/user", "/api/product", "/api/order"]

    for i, (client, route) in enumerate(zip(clients * 3, routes * 4)):
        resp = client.send(HTTPMethod.GET, route)
        print(f"  {client.client_id} → {route:<15} [{resp.status.value}] "
              f"{resp.latency_ms:.1f}ms via {resp.served_by}")

    # ── Also try a bad route ──────────────────
    print("\n[Testing 404 path]")
    resp = clients[0].send(HTTPMethod.GET, "/api/missing")
    print(f"  client-1 → /api/missing [{resp.status.value}] {resp.body}")

    # ── Connection Pool Stats ─────────────────
    pool_stats = pool.stats()
    print(f"\nConnection Pool Stats (max_size={pool.max_size}):")
    for k, v in pool_stats.items():
        print(f"  {k:<22}: {v}")

    # ── Load Balancer Distribution ────────────
    lb.stats()

    # ── Latency Stats per Client ──────────────
    print("\nClient Latency Summary:")
    for client in clients:
        if client.latencies:
            print(f"  {client.client_id}: avg={client.avg_latency():.1f}ms  "
                  f"requests={len(client.latencies)}")

    # ── Conceptual Comparison ─────────────────
    print("\nConnection Type Trade-offs:")
    comparisons = [
        ("HTTP/1.0 (short-lived)", "New TCP per request", "Simple", "High latency (TCP handshake each time)"),
        ("HTTP/1.1 (keep-alive)",  "Reuse TCP connection","Lower latency", "Head-of-line blocking"),
        ("HTTP/2 (multiplexed)",   "Many streams per TCP","Lowest latency, parallel", "Complex to implement"),
    ]
    for name, mechanism, pro, con in comparisons:
        print(f"\n  {name}")
        print(f"    Mechanism : {mechanism}")
        print(f"    Pro       : {pro}")
        print(f"    Con       : {con}")


if __name__ == "__main__":
    demonstrate_client_server_architecture()
