"""
SYSTEM DESIGN: API GATEWAY
============================

Problem Statement:
Design an API Gateway that acts as the single entry point for client
requests, handling cross-cutting concerns so individual microservices
don't need to implement them.

Functional Requirements:
  - Route requests to downstream services
  - Authentication / Authorization
  - Rate limiting (per-user, per-API-key, per-endpoint)
  - Request/Response transformation
  - Load balancing across service instances
  - Circuit breaking (fail fast when service is down)
  - Caching (cache GET responses)
  - Logging, metrics, tracing

Non-Functional Requirements:
  - < 1ms overhead for routing (excluding downstream)
  - Handle 1M+ RPS
  - 99.99% availability
  - Horizontal scalability

Why API Gateway:
  Without: Each service duplicates auth, rate limiting, CORS, logging.
  With:    Cross-cutting concerns in one place; services focus on business logic.

Gateway vs BFF (Backend for Frontend):
  API Gateway: generic, handles all clients.
  BFF:         client-specific. Mobile BFF aggregates 3 service calls into 1.
  Use both: BFF sits behind API Gateway.

Common Products:
  Kong (open-source, Lua plugins), AWS API Gateway, nginx, Envoy,
  Traefik, Google Cloud Apigee, Azure API Management.

Routing Patterns:
  Simple:       /api/users → user-service
  Path prefix:  /v1/orders → order-service-v1
  Host-based:   api.example.com → main; admin.example.com → admin-service
  Weight-based: 90% → stable, 10% → canary (for gradual rollout)

Circuit Breaker States:
  CLOSED:    Requests flow normally. Count failures.
  OPEN:      Requests fail fast (no downstream call). After timeout → HALF_OPEN.
  HALF_OPEN: Allow N test requests. If pass → CLOSED; if fail → OPEN.

Load Balancing:
  Round Robin: equal distribution.
  Least Connections: route to instance with fewest active requests.
  Weighted Round Robin: instance_a weight=3, instance_b weight=1.
  Consistent Hashing: same client → same upstream (session affinity).
  IP Hash: hash client IP → upstream (simple session affinity).
"""

from __future__ import annotations

import time
import uuid
import random
import threading
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict, deque


# ─────────────────────────────────────────────
# REQUEST / RESPONSE
# ─────────────────────────────────────────────

@dataclass
class Request:
    request_id:  str
    method:      str
    path:        str
    headers:     Dict[str, str]
    body:        Optional[str]
    client_ip:   str
    timestamp:   float = field(default_factory=time.time)
    ctx:         Dict[str, Any] = field(default_factory=dict)   # gateway context


@dataclass
class Response:
    status_code: int
    headers:     Dict[str, str]
    body:        str
    upstream:    Optional[str] = None   # which upstream served it

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300


# ─────────────────────────────────────────────
# UPSTREAM SERVICE INSTANCE
# ─────────────────────────────────────────────

@dataclass
class ServiceInstance:
    instance_id: str
    host:        str
    port:        int
    weight:      int = 1
    active_conns: int = 0
    healthy:     bool = True

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


# ─────────────────────────────────────────────
# LOAD BALANCER
# ─────────────────────────────────────────────

class LBStrategy(Enum):
    ROUND_ROBIN       = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM            = "random"
    IP_HASH           = "ip_hash"


class LoadBalancer:
    def __init__(self, strategy: LBStrategy = LBStrategy.ROUND_ROBIN):
        self._strategy = strategy
        self._counter  = 0
        self._lock     = threading.Lock()

    def pick(self, instances: List[ServiceInstance],
             client_ip: str = "0.0.0.0") -> Optional[ServiceInstance]:
        healthy = [i for i in instances if i.healthy]
        if not healthy:
            return None

        if self._strategy == LBStrategy.ROUND_ROBIN:
            with self._lock:
                idx = self._counter % len(healthy)
                self._counter += 1
            return healthy[idx]

        if self._strategy == LBStrategy.LEAST_CONNECTIONS:
            return min(healthy, key=lambda i: i.active_conns)

        if self._strategy == LBStrategy.RANDOM:
            return random.choice(healthy)

        if self._strategy == LBStrategy.IP_HASH:
            h   = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
            idx = h % len(healthy)
            return healthy[idx]

        return healthy[0]


# ─────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────

class CBState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """
    Per-upstream circuit breaker.
    CLOSED → OPEN after threshold failures in window.
    OPEN → HALF_OPEN after timeout.
    HALF_OPEN → CLOSED on success, OPEN on failure.
    """

    upstream:          str
    failure_threshold: int   = 5
    window_s:          float = 60.0
    open_timeout_s:    float = 30.0
    half_open_max:     int   = 3

    state:             CBState = field(default_factory=lambda: CBState.CLOSED)
    failure_count:     int = 0
    last_failure_ts:   float = 0.0
    opened_at:         float = 0.0
    half_open_count:   int = 0
    _lock:             object = field(default_factory=threading.Lock)

    def allow(self) -> bool:
        """Should this request be allowed through?"""
        with self._lock:
            now = time.time()

            if self.state == CBState.CLOSED:
                return True

            if self.state == CBState.OPEN:
                if now - self.opened_at >= self.open_timeout_s:
                    self.state = CBState.HALF_OPEN
                    self.half_open_count = 0
                    return True
                return False

            if self.state == CBState.HALF_OPEN:
                if self.half_open_count < self.half_open_max:
                    self.half_open_count += 1
                    return True
                return False

        return True

    def record_success(self):
        with self._lock:
            if self.state == CBState.HALF_OPEN:
                self.state         = CBState.CLOSED
                self.failure_count = 0
            elif self.state == CBState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        with self._lock:
            now = time.time()
            if self.state == CBState.HALF_OPEN:
                self.state    = CBState.OPEN
                self.opened_at = now
                return

            # Clear old failures outside window
            if now - self.last_failure_ts > self.window_s:
                self.failure_count = 0

            self.failure_count  += 1
            self.last_failure_ts = now

            if self.failure_count >= self.failure_threshold:
                self.state     = CBState.OPEN
                self.opened_at = now


# ─────────────────────────────────────────────
# ROUTE TABLE
# ─────────────────────────────────────────────

@dataclass
class Route:
    path_prefix:  str
    service_name: str
    strip_prefix: bool = False
    require_auth: bool = True
    rate_limit:   Optional[int] = None   # requests per minute
    cache_ttl_s:  Optional[float] = None


class RouteTable:
    def __init__(self):
        self._routes: List[Route] = []

    def add(self, route: Route):
        self._routes.append(route)
        # Longer prefixes first (most specific match)
        self._routes.sort(key=lambda r: -len(r.path_prefix))

    def match(self, path: str) -> Optional[Route]:
        for route in self._routes:
            if path.startswith(route.path_prefix):
                return route
        return None


# ─────────────────────────────────────────────
# AUTH MIDDLEWARE
# ─────────────────────────────────────────────

class AuthMiddleware:
    def __init__(self):
        self._valid_tokens: Dict[str, str] = {}   # token → user_id

    def register(self, token: str, user_id: str):
        self._valid_tokens[token] = user_id

    def authenticate(self, req: Request) -> Tuple[bool, Optional[str]]:
        auth_header = req.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return False, None
        token   = auth_header[7:]
        user_id = self._valid_tokens.get(token)
        return (True, user_id) if user_id else (False, None)


# ─────────────────────────────────────────────
# SIMPLE RATE LIMITER (token bucket)
# ─────────────────────────────────────────────

class GatewayRateLimiter:
    def __init__(self):
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock    = threading.Lock()

    def check(self, key: str, rate_per_min: int) -> bool:
        rate_per_s = rate_per_min / 60
        now = time.time()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(rate_per_min), now))
            elapsed = now - last
            tokens  = min(rate_per_min, tokens + elapsed * rate_per_s)
            if tokens < 1:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1, now)
            return True


# ─────────────────────────────────────────────
# RESPONSE CACHE
# ─────────────────────────────────────────────

class ResponseCache:
    def __init__(self):
        self._store: Dict[str, Tuple[Response, float]] = {}

    def get(self, key: str) -> Optional[Response]:
        entry = self._store.get(key)
        if not entry:
            return None
        resp, expires = entry
        if time.time() > expires:
            del self._store[key]
            return None
        return resp

    def set(self, key: str, resp: Response, ttl_s: float):
        self._store[key] = (resp, time.time() + ttl_s)

    def cache_key(self, req: Request) -> str:
        return f"{req.method}:{req.path}:{req.headers.get('Authorization','')}"


# ─────────────────────────────────────────────
# SIMULATED UPSTREAM
# ─────────────────────────────────────────────

def simulate_upstream_call(instance: ServiceInstance,
                            req: Request, fail_rate: float = 0.05) -> Response:
    """Simulate a call to an upstream service."""
    r = random.random()
    if r < fail_rate:
        return Response(500, {}, f'{{"error":"upstream_error"}}', instance.address)
    latency = random.uniform(5, 50)  # 5-50ms
    time.sleep(latency / 1000)
    return Response(200, {"Content-Type": "application/json"},
                    f'{{"data": "response from {instance.address}", '
                    f'"path": "{req.path}"}}',
                    instance.address)


# ─────────────────────────────────────────────
# API GATEWAY
# ─────────────────────────────────────────────

class APIGateway:
    def __init__(self):
        self._routes       = RouteTable()
        self._auth         = AuthMiddleware()
        self._rate_limiter = GatewayRateLimiter()
        self._cache        = ResponseCache()
        self._lb           = LoadBalancer(LBStrategy.LEAST_CONNECTIONS)
        self._services:    Dict[str, List[ServiceInstance]] = defaultdict(list)
        self._breakers:    Dict[str, CircuitBreaker]        = {}
        self._metrics      = defaultdict(int)

    def register_service(self, service_name: str, instances: List[ServiceInstance]):
        self._services[service_name] = instances
        self._breakers[service_name] = CircuitBreaker(service_name)

    def add_route(self, route: Route):
        self._routes.add(route)

    def handle(self, req: Request) -> Response:
        self._metrics["total_requests"] += 1

        # 1. Route matching
        route = self._routes.match(req.path)
        if not route:
            self._metrics["not_found"] += 1
            return Response(404, {}, '{"error":"not_found"}')

        # 2. Authentication
        if route.require_auth:
            ok, user_id = self._auth.authenticate(req)
            if not ok:
                self._metrics["auth_failed"] += 1
                return Response(401, {}, '{"error":"unauthorized"}')
            req.ctx["user_id"] = user_id

        # 3. Rate limiting
        key = req.ctx.get("user_id", req.client_ip)
        limit = route.rate_limit or 1000
        if not self._rate_limiter.check(key, limit):
            self._metrics["rate_limited"] += 1
            return Response(429, {"Retry-After": "60"},
                            '{"error":"rate_limit_exceeded"}')

        # 4. Cache check (GET only)
        if req.method == "GET" and route.cache_ttl_s:
            cached = self._cache.get(self._cache.cache_key(req))
            if cached:
                self._metrics["cache_hits"] += 1
                cached.headers["X-Cache"] = "HIT"
                return cached

        # 5. Circuit breaker
        cb = self._breakers.get(route.service_name)
        if cb and not cb.allow():
            self._metrics["circuit_open"] += 1
            return Response(503, {"Retry-After": "30"},
                            '{"error":"service_unavailable"}')

        # 6. Load balance + upstream call
        instances = self._services.get(route.service_name, [])
        instance  = self._lb.pick(instances, req.client_ip)
        if not instance:
            return Response(503, {}, '{"error":"no_instances"}')

        instance.active_conns += 1
        try:
            resp = simulate_upstream_call(instance, req)
            if cb:
                if resp.is_success:
                    cb.record_success()
                else:
                    cb.record_failure()
        except Exception:
            if cb:
                cb.record_failure()
            resp = Response(502, {}, '{"error":"bad_gateway"}')
        finally:
            instance.active_conns -= 1

        # 7. Cache response
        if req.method == "GET" and route.cache_ttl_s and resp.is_success:
            self._cache.set(self._cache.cache_key(req), resp, route.cache_ttl_s)

        # 8. Add gateway headers
        resp.headers["X-Request-ID"]  = req.request_id
        resp.headers["X-Upstream"]    = instance.address
        resp.headers["X-Cache"]       = "MISS"

        self._metrics["success" if resp.is_success else "errors"] += 1
        return resp


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_api_gateway():
    print("=" * 65)
    print("SYSTEM DESIGN: API GATEWAY")
    print("=" * 65)

    random.seed(42)
    gw = APIGateway()

    # ── Register Services ─────────────────────
    print("\n[1] SERVICE REGISTRATION")
    print("─" * 55)

    services = {
        "user-service": [
            ServiceInstance("u1", "10.0.1.1", 8080, weight=2),
            ServiceInstance("u2", "10.0.1.2", 8080, weight=1),
        ],
        "order-service": [
            ServiceInstance("o1", "10.0.2.1", 8081),
            ServiceInstance("o2", "10.0.2.2", 8081),
            ServiceInstance("o3", "10.0.2.3", 8081),
        ],
        "product-service": [
            ServiceInstance("p1", "10.0.3.1", 8082),
        ],
    }
    for svc_name, instances in services.items():
        gw.register_service(svc_name, instances)
        print(f"  {svc_name}: {len(instances)} instance(s)")

    # ── Routes ────────────────────────────────
    print("\n[2] ROUTE TABLE")
    print("─" * 55)

    routes = [
        Route("/api/users",    "user-service",    require_auth=True,  rate_limit=100),
        Route("/api/orders",   "order-service",   require_auth=True,  rate_limit=50),
        Route("/api/products", "product-service", require_auth=False, cache_ttl_s=30.0),
        Route("/health",       "user-service",    require_auth=False, rate_limit=1000),
    ]
    for route in routes:
        gw.add_route(route)
        print(f"  {route.path_prefix:<18} → {route.service_name:<20} "
              f"auth={route.require_auth}  limit={route.rate_limit}/min")

    # ── Auth Tokens ───────────────────────────
    gw._auth.register("token_alice_123", "user_alice")
    gw._auth.register("token_bob_456",   "user_bob")

    # ── Test Requests ─────────────────────────
    print("\n[3] REQUEST HANDLING")
    print("─" * 55)

    test_cases = [
        ("GET",  "/api/products",    {},                                 "10.0.0.1", "No auth (public)"),
        ("GET",  "/api/users/42",    {"Authorization": "Bearer token_alice_123"}, "10.0.0.1", "Authenticated"),
        ("GET",  "/api/users/43",    {},                                 "10.0.0.2", "Missing auth"),
        ("POST", "/api/orders",      {"Authorization": "Bearer token_bob_456"}, "10.0.0.3", "Order create"),
        ("GET",  "/api/unknown",     {},                                 "10.0.0.1", "No route"),
        ("GET",  "/api/products",    {},                                 "10.0.0.1", "Cache hit (2nd req)"),
    ]

    for method, path, headers, ip, desc in test_cases:
        req = Request(uuid.uuid4().hex[:8], method, path, headers, None, ip)
        resp = gw.handle(req)
        cached = resp.headers.get("X-Cache", "")
        print(f"  [{method:<4}] {path:<20} {desc:<25} → {resp.status_code} "
              f"{cached}")

    # ── Rate Limiting ─────────────────────────
    print("\n[4] RATE LIMITING")
    print("─" * 55)

    rl_gw = APIGateway()
    rl_gw.register_service("user-service", [ServiceInstance("u1", "10.0.1.1", 8080)])
    rl_gw.add_route(Route("/api/users", "user-service", require_auth=False, rate_limit=3))
    rl_gw._auth.register("tok", "user")

    results = []
    for i in range(6):
        req  = Request(str(i), "GET", "/api/users", {}, None, "1.2.3.4")
        resp = rl_gw.handle(req)
        results.append(resp.status_code)
    print(f"  Rate limit=3/min, 6 requests: {results}")

    # ── Circuit Breaker ───────────────────────
    print("\n[5] CIRCUIT BREAKER")
    print("─" * 55)

    cb = CircuitBreaker("test-svc", failure_threshold=3, open_timeout_s=0.5)
    states = []

    # 3 failures → OPEN
    for _ in range(3):
        cb.allow()
        cb.record_failure()
        states.append(cb.state.value)

    # OPEN → fast fail
    for _ in range(2):
        allowed = cb.allow()
        states.append(f"allow={allowed}({cb.state.value})")

    print(f"  States after failures + open:")
    for s in states:
        print(f"    {s}")

    # Wait for timeout → HALF_OPEN
    time.sleep(0.6)
    cb.allow()
    states.append(f"after_timeout={cb.state.value}")
    cb.record_success()
    states.append(f"after_success={cb.state.value}")
    print(f"  After timeout + success: {states[-2:]}")

    # ── Gateway Metrics ───────────────────────
    print("\n[6] GATEWAY METRICS")
    print("─" * 55)

    for k, v in gw._metrics.items():
        print(f"  {k}: {v}")

    # ── Architecture ──────────────────────────
    print("\n[7] API GATEWAY PATTERNS")
    print("─" * 55)

    patterns = [
        ("Authentication",  "JWT verification; OAuth token introspection"),
        ("Rate Limiting",   "Token bucket per user/IP in Redis"),
        ("Load Balancing",  "Least connections + health check probes"),
        ("Circuit Breaker", "Per-upstream; CLOSED→OPEN→HALF_OPEN"),
        ("Caching",         "GET responses cached in Redis; cache-control headers"),
        ("Observability",   "Trace ID injected; latency per route; error rates"),
        ("Canary",          "Weight-based routing: 95% stable, 5% new"),
        ("Protocol",        "HTTP/2, gRPC, WebSocket → proxy to upstream"),
        ("mTLS",            "Mutual TLS between gateway and upstream services"),
        ("Products",        "Kong, AWS API GW, Envoy, Nginx Plus, Traefik"),
    ]
    for name, detail in patterns:
        print(f"  {name:<18} {detail}")


if __name__ == "__main__":
    demonstrate_api_gateway()
