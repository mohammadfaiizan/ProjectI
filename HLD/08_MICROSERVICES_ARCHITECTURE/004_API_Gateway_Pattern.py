"""
API GATEWAY PATTERN
=====================

Problem Statement:
Clients (web, mobile, 3rd party) shouldn't call 20 microservices directly.
Each service has its own auth, rate limits, versioning, and URL scheme.
An API Gateway is a single entry point that handles cross-cutting concerns
before routing to the appropriate backend service.

What an API Gateway Does:
  1. Authentication / Authorization:
     Validate JWT or API key before request reaches any service.
     Services trust the gateway — no redundant auth logic in each service.

  2. Rate Limiting:
     Per-client or per-endpoint throttling.
     Prevents abuse; protects backends from overload.

  3. Routing:
     Map public URLs to internal service URLs.
     /api/orders → order-service:8081/orders
     /api/users  → user-service:8082/users

  4. Request Transformation:
     Rename fields, add/remove headers, transform payload format.
     Shields clients from internal API changes.

  5. Fan-out / Aggregation:
     One client request → parallel calls to multiple backend services.
     Aggregate results before returning. Reduces client round trips.
     (Also the core of the BFF pattern.)

  6. Correlation ID Injection:
     Gateway stamps every inbound request with X-Correlation-Id.
     Propagated to all downstream services for distributed tracing.

  7. Response Caching:
     Cache GET responses for static/slow-changing data.

Trade-offs:
  Pro:  Single place for cross-cutting concerns; simpler clients.
  Con:  Single point of failure (needs high availability, multiple instances).
  Con:  Can become a bottleneck if business logic leaks into the gateway.
  Rule: Gateway handles infrastructure concerns; business logic stays in services.

API Gateway vs Service Mesh:
  Gateway:      North-South traffic (external → internal). Layer 7.
  Service Mesh: East-West traffic (service → service). mTLS, retries, tracing.
  They complement each other; not mutually exclusive.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import uuid
import threading
import re


# ─────────────────────────────────────────────
# ROUTE REGISTRY
# ─────────────────────────────────────────────

@dataclass
class Route:
    path_pattern  : str               # e.g. "/api/orders/{id}"
    method        : str               # GET / POST / PUT / DELETE
    backend       : str               # logical service name
    backend_path  : str               # internal path
    requires_auth : bool = True
    rate_limit_rpm: int  = 100        # requests per minute per client
    cache_ttl_s   : int  = 0          # 0 = no cache
    fan_out       : Optional[List[str]] = None  # service names to fan out to


class RouteRegistry:
    def __init__(self):
        self._routes: List[Route] = []

    def register(self, route: Route):
        self._routes.append(route)

    def match(self, path: str, method: str) -> Optional[Tuple[Route, Dict[str, str]]]:
        """Return (route, path_params) or None."""
        for route in self._routes:
            params = self._match_pattern(route.path_pattern, path)
            if params is not None and route.method == method.upper():
                return route, params
        return None

    def _match_pattern(self, pattern: str, path: str) -> Optional[Dict[str, str]]:
        regex = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", pattern) + "$"
        m = re.match(regex, path)
        if m:
            return m.groupdict()
        return None


# ─────────────────────────────────────────────
# RATE LIMITER (token bucket, per client)
# ─────────────────────────────────────────────

class TokenBucketRateLimiter:
    """Per-client token bucket. Thread-safe."""

    def __init__(self, rpm: int):
        self.rpm         = rpm
        self._buckets    : Dict[str, Dict] = {}
        self._lock       = threading.Lock()

    def _get_bucket(self, client_id: str, rpm: int) -> Dict:
        with self._lock:
            if client_id not in self._buckets:
                self._buckets[client_id] = {
                    "tokens"      : float(rpm),
                    "last_refill" : time.time(),
                    "rpm"         : rpm,
                }
            return self._buckets[client_id]

    def is_allowed(self, client_id: str, rpm: int) -> bool:
        with self._lock:
            bucket = self._get_bucket(client_id, rpm)
            now    = time.time()
            elapsed = now - bucket["last_refill"]
            # Refill tokens based on elapsed time
            refill = elapsed * (rpm / 60.0)
            bucket["tokens"]      = min(rpm, bucket["tokens"] + refill)
            bucket["last_refill"] = now

            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            return False


# ─────────────────────────────────────────────
# JWT AUTH (stub)
# ─────────────────────────────────────────────

@dataclass
class AuthContext:
    client_id : str
    user_id   : str
    scopes    : List[str]
    valid     : bool


def validate_jwt(token: str) -> AuthContext:
    """Stub JWT validator. In production: verify RS256 signature."""
    if not token or token == "invalid":
        return AuthContext("", "", [], valid=False)
    # Simulated: token encodes client:user:scope1,scope2
    parts = token.split(":")
    if len(parts) == 3:
        return AuthContext(parts[0], parts[1], parts[2].split(","), valid=True)
    return AuthContext("anon", "anon", ["read"], valid=True)


# ─────────────────────────────────────────────
# BACKEND SERVICES (stub)
# ─────────────────────────────────────────────

class BackendService:
    def __init__(self, name: str, base_latency_ms: float = 20):
        self.name           = name
        self.base_latency   = base_latency_ms
        self._call_count    = 0

    def handle(self, path: str, method: str, headers: Dict, body: Any) -> Dict:
        import random
        self._call_count += 1
        time.sleep((self.base_latency + random.uniform(0, 10)) / 1000)
        return {
            "service"  : self.name,
            "path"     : path,
            "status"   : 200,
            "data"     : f"{self.name} response for {path}",
            "call_num" : self._call_count,
        }


# ─────────────────────────────────────────────
# RESPONSE CACHE
# ─────────────────────────────────────────────

class ResponseCache:
    def __init__(self):
        self._store : Dict[str, Tuple[Any, float]] = {}  # key → (value, expires_at)

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry and time.time() < entry[1]:
            return entry[0]
        return None

    def put(self, key: str, value: Any, ttl_s: int):
        self._store[key] = (value, time.time() + ttl_s)

    def size(self) -> int:
        return len(self._store)


# ─────────────────────────────────────────────
# API GATEWAY
# ─────────────────────────────────────────────

@dataclass
class GatewayRequest:
    path          : str
    method        : str
    headers       : Dict[str, str]
    body          : Any = None

    @property
    def auth_token(self) -> str:
        auth = self.headers.get("Authorization", "")
        return auth.replace("Bearer ", "") if auth.startswith("Bearer ") else auth


@dataclass
class GatewayResponse:
    status          : int
    body            : Any
    headers         : Dict[str, str] = field(default_factory=dict)
    correlation_id  : str = ""
    total_ms        : float = 0.0


class ApiGateway:
    def __init__(self):
        self.registry     = RouteRegistry()
        self.rate_limiter = TokenBucketRateLimiter(rpm=60)
        self.cache        = ResponseCache()
        self._services    : Dict[str, BackendService] = {}
        self._access_log  : List[Dict] = []

    def register_service(self, name: str, service: BackendService):
        self._services[name] = service

    def handle(self, req: GatewayRequest) -> GatewayResponse:
        start = time.time()
        corr_id = req.headers.get("X-Correlation-Id") or str(uuid.uuid4())[:8]
        req.headers["X-Correlation-Id"] = corr_id

        # ── Route matching ──────────────────────
        match = self.registry.match(req.path, req.method)
        if match is None:
            return self._resp(404, {"error": "route not found"}, corr_id, start)

        route, path_params = match

        # ── Auth ────────────────────────────────
        if route.requires_auth:
            auth = validate_jwt(req.auth_token)
            if not auth.valid:
                return self._resp(401, {"error": "unauthorized"}, corr_id, start)
            req.headers["X-User-Id"]   = auth.user_id
            req.headers["X-Client-Id"] = auth.client_id
        else:
            auth = AuthContext("anon", "anon", ["read"], True)

        # ── Rate limiting ───────────────────────
        if not self.rate_limiter.is_allowed(auth.client_id, route.rate_limit_rpm):
            return self._resp(429, {"error": "rate limit exceeded"}, corr_id, start)

        # ── Cache check (GET only) ──────────────
        cache_key = f"{auth.client_id}:{req.method}:{req.path}"
        if req.method == "GET" and route.cache_ttl_s > 0:
            cached = self.cache.get(cache_key)
            if cached:
                resp = self._resp(200, cached, corr_id, start)
                resp.headers["X-Cache"] = "HIT"
                return resp

        # ── Fan-out or single call ──────────────
        if route.fan_out:
            body = self._fan_out(route.fan_out, req, corr_id)
        else:
            svc  = self._services.get(route.backend)
            if svc is None:
                return self._resp(503, {"error": f"service {route.backend} unavailable"},
                                  corr_id, start)
            body = svc.handle(route.backend_path, req.method, req.headers, req.body)

        # ── Cache store ─────────────────────────
        if req.method == "GET" and route.cache_ttl_s > 0:
            self.cache.put(cache_key, body, route.cache_ttl_s)

        resp = self._resp(200, body, corr_id, start)
        resp.headers["X-Cache"] = "MISS"
        self._log(req, resp, auth.client_id)
        return resp

    def _fan_out(self, services: List[str], req: GatewayRequest,
                 corr_id: str) -> Dict:
        """Call multiple backend services in parallel, aggregate results."""
        results  = {}
        lock     = threading.Lock()
        threads  = []

        def call_service(name: str):
            svc = self._services.get(name)
            if svc:
                result = svc.handle(req.path, req.method, req.headers, req.body)
                with lock:
                    results[name] = result

        for svc_name in services:
            t = threading.Thread(target=call_service, args=(svc_name,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return {"aggregated": results, "correlation_id": corr_id}

    def _resp(self, status: int, body: Any,
              corr_id: str, start: float) -> GatewayResponse:
        ms = (time.time() - start) * 1000
        return GatewayResponse(status, body,
                               {"X-Correlation-Id": corr_id},
                               corr_id, ms)

    def _log(self, req: GatewayRequest, resp: GatewayResponse, client_id: str):
        self._access_log.append({
            "path"      : req.path,
            "method"    : req.method,
            "status"    : resp.status,
            "client"    : client_id,
            "ms"        : resp.total_ms,
            "corr_id"   : resp.correlation_id,
        })


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_api_gateway():
    print("=" * 65)
    print("API GATEWAY PATTERN")
    print("=" * 65)

    # Build gateway
    gw = ApiGateway()

    # Register backend services
    order_svc  = BackendService("order-service",   latency_ms := 30)
    user_svc   = BackendService("user-service",    20)
    product_svc= BackendService("product-service", 15)
    gw.register_service("order-service",   order_svc)
    gw.register_service("user-service",    user_svc)
    gw.register_service("product-service", product_svc)

    # Register routes
    gw.registry.register(Route("/api/orders",           "POST", "order-service",   "/orders",    True,  60))
    gw.registry.register(Route("/api/orders/{id}",      "GET",  "order-service",   "/orders",    True,  100, cache_ttl_s=5))
    gw.registry.register(Route("/api/users/{id}",       "GET",  "user-service",    "/users",     True,  100, cache_ttl_s=10))
    gw.registry.register(Route("/api/health",           "GET",  "order-service",   "/health",    False, 1000))
    gw.registry.register(Route("/api/dashboard/{uid}",  "GET",  "order-service",   "/dashboard", True,  50,
                                fan_out=["order-service", "user-service", "product-service"]))

    # ── 1. Auth rejection ─────────────────────────
    print("\n[1] AUTHENTICATION — REJECT INVALID TOKEN")
    print("─" * 55)
    resp = gw.handle(GatewayRequest("/api/orders", "POST",
                                    {"Authorization": "Bearer invalid"}))
    print(f"  Invalid token → status={resp.status}  body={resp.body}")

    resp = gw.handle(GatewayRequest("/api/orders", "POST",
                                    {"Authorization": "Bearer web-client:user123:read,write"},
                                    body={"items": [{"sku": "A1", "qty": 2}]}))
    print(f"  Valid token   → status={resp.status}  corr_id={resp.correlation_id}")

    # ── 2. Rate limiting ──────────────────────────
    print("\n\n[2] RATE LIMITING — PER CLIENT")
    print("─" * 55)
    # Use a route with very low limit
    gw.registry.register(Route("/api/slow-route", "GET", "order-service", "/slow",
                               True, rate_limit_rpm=2))
    token  = "cli:user1:read"
    allowed = blocked = 0
    for _ in range(5):
        r = gw.handle(GatewayRequest("/api/slow-route", "GET",
                                     {"Authorization": f"Bearer {token}"}))
        if r.status == 200: allowed += 1
        else:               blocked += 1

    print(f"  5 rapid requests with limit=2rpm: allowed={allowed} blocked={blocked}")
    print(f"  Rate limit exceeded → 429 Too Many Requests")

    # ── 3. Route not found ────────────────────────
    print("\n\n[3] ROUTING — UNKNOWN PATH → 404")
    print("─" * 55)
    resp = gw.handle(GatewayRequest("/api/unknown/path", "GET",
                                    {"Authorization": "Bearer web:u1:read"}))
    print(f"  /api/unknown/path → status={resp.status}  body={resp.body}")

    # ── 4. Fan-out aggregation ────────────────────
    print("\n\n[4] FAN-OUT — ONE REQUEST, THREE BACKEND CALLS IN PARALLEL")
    print("─" * 55)
    t0   = time.time()
    resp = gw.handle(GatewayRequest("/api/dashboard/user99", "GET",
                                    {"Authorization": "Bearer web:user99:read"}))
    elapsed = (time.time() - t0) * 1000
    print(f"  Dashboard request: status={resp.status}  total={elapsed:.1f}ms")
    agg = resp.body.get("aggregated", {})
    for svc_name, data in agg.items():
        print(f"    [{svc_name}] {data['data']}")
    print(f"  Fan-out to {len(agg)} services ran in parallel; "
          f"total ≈ max(latencies), not sum.")

    # ── 5. Cache ──────────────────────────────────
    print("\n\n[5] RESPONSE CACHING")
    print("─" * 55)
    token = "web:u1:read"
    r1 = gw.handle(GatewayRequest("/api/orders/ord-001", "GET",
                                  {"Authorization": f"Bearer {token}"}))
    r2 = gw.handle(GatewayRequest("/api/orders/ord-001", "GET",
                                  {"Authorization": f"Bearer {token}"}))
    print(f"  Request 1: status={r1.status}  cache={r1.headers.get('X-Cache','?')}  "
          f"ms={r1.total_ms:.2f}")
    print(f"  Request 2: status={r2.status}  cache={r2.headers.get('X-Cache','?')}  "
          f"ms={r2.total_ms:.2f}")
    print(f"  Cache HIT is faster (no backend call).")

    # ── 6. Correlation ID propagation ────────────
    print("\n\n[6] CORRELATION ID — REQUEST TRACING")
    print("─" * 55)
    req = GatewayRequest("/api/users/u42", "GET",
                         {"Authorization": "Bearer web:u42:read",
                          "X-Correlation-Id": "trace-ABCD1234"})
    resp = gw.handle(req)
    print(f"  Client sends X-Correlation-Id: trace-ABCD1234")
    print(f"  Gateway preserves it: {resp.correlation_id}")
    print(f"  Propagated to all downstream services in headers.")

    # ── 7. Access log ─────────────────────────────
    print("\n\n[7] ACCESS LOG (last 4 entries)")
    print("─" * 55)
    for entry in gw._access_log[-4:]:
        print(f"  {entry['method']:<5} {entry['path']:<30} "
              f"status={entry['status']} client={entry['client']} "
              f"ms={entry['ms']:.1f}")

    # ── 8. Gateway principles ─────────────────────
    print("\n\n[8] API GATEWAY RESPONSIBILITIES")
    print("─" * 55)
    responsibilities = [
        ("Authentication",      "Validate JWT/API key — once, at the edge"),
        ("Rate limiting",       "Per-client throttle; protect backends"),
        ("Routing",             "Public URL → internal service URL"),
        ("Request transform",   "Add headers, rename fields, strip secrets"),
        ("Fan-out/aggregate",   "Reduce client round trips"),
        ("Correlation IDs",     "Stamp every request for distributed tracing"),
        ("Caching",             "Cache GET responses at the edge"),
        ("NOT business logic",  "Gateway stays infrastructure-only"),
    ]
    for name, desc in responsibilities:
        print(f"  {name:<22} {desc}")


if __name__ == "__main__":
    demonstrate_api_gateway()
