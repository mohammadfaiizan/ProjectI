"""
API Gateway - Core Implementation
Demonstrates: middleware chain pattern, JWT auth, token-bucket rate limiting,
circuit breaker state machine, response caching, request routing,
structured logging. Standard library only.
"""

import hashlib
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

@dataclass
class Request:
    method:     str           # GET, POST, PUT, DELETE
    path:       str
    headers:    Dict[str, str] = field(default_factory=dict)
    body:       Optional[str] = None
    query:      Dict[str, str] = field(default_factory=dict)
    client_ip:  str = "0.0.0.0"
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    received_at: float = field(default_factory=time.time)


@dataclass
class Response:
    status_code: int
    body:        Any = None
    headers:     Dict[str, str] = field(default_factory=dict)
    from_cache:  bool = False

    def is_error(self) -> bool:
        return self.status_code >= 400


# ---------------------------------------------------------------------------
# Context (request-scoped bag of values)
# ---------------------------------------------------------------------------

class RequestContext:
    def __init__(self, request: Request):
        self.request  = request
        self.response: Optional[Response] = None
        self.user_id:  Optional[str] = None
        self.client_id: Optional[str] = None
        self.upstream:  Optional[str] = None
        self.log_data:  Dict[str, Any] = {}
        self.start_time = time.time()

    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


# ---------------------------------------------------------------------------
# Route Registry
# ---------------------------------------------------------------------------

@dataclass
class Route:
    path_prefix: str
    service:     str
    instances:   List[str]
    methods:     List[str] = field(default_factory=lambda: ["GET","POST","PUT","DELETE"])
    cache_ttl:   int = 0        # seconds; 0 = no cache
    auth_required: bool = True
    version:     str = "v1"
    canary_pct:  int = 0        # 0-100; percentage of traffic to canary
    canary_service: Optional[str] = None

    def match(self, method: str, path: str) -> bool:
        return method in self.methods and path.startswith(self.path_prefix)

    def select_instance(self) -> str:
        """Simple round-robin (real: weighted RR with health checks)."""
        if not self.instances:
            raise RuntimeError(f"No instances for service {self.service}")
        idx = int(time.time() * 1000) % len(self.instances)
        return self.instances[idx]

    def should_use_canary(self) -> bool:
        if not self.canary_service or self.canary_pct == 0:
            return False
        return (int(time.time() * 1000) % 100) < self.canary_pct


class RouteRegistry:
    def __init__(self):
        self._routes: List[Route] = []

    def register(self, route: Route):
        self._routes.append(route)

    def match(self, method: str, path: str) -> Optional[Route]:
        """Return the most specific matching route."""
        candidates = [r for r in self._routes if r.match(method, path)]
        if not candidates:
            return None
        # Most specific = longest path_prefix
        return max(candidates, key=lambda r: len(r.path_prefix))


# ---------------------------------------------------------------------------
# Token Bucket Rate Limiter
# ---------------------------------------------------------------------------

@dataclass
class BucketState:
    tokens:         float
    capacity:       float
    refill_rate:    float   # tokens per second
    last_refill_ts: float


class RateLimiter:
    """
    Token bucket implementation.
    In production: atomic operations via Redis Lua script.
    """

    DEFAULT_TIERS = {
        "free":       (100,   100 / 60),    # capacity, tokens/sec
        "pro":        (10000, 10000 / 60),
        "enterprise": (1000000, 1000000 / 60),
        "ip":         (1000,  1000 / 60),   # 1000 req/min per IP (generous for demo)
    }

    def __init__(self):
        self._buckets: Dict[str, BucketState] = {}
        # Instance-level copy so demo overrides don't bleed across tests
        self.TIERS = dict(self.DEFAULT_TIERS)

    def _get_bucket(self, key: str, tier: str) -> BucketState:
        if key not in self._buckets:
            capacity, rate = self.TIERS.get(tier, self.TIERS["free"])
            self._buckets[key] = BucketState(
                tokens=capacity, capacity=capacity,
                refill_rate=rate, last_refill_ts=time.time()
            )
        return self._buckets[key]

    def allow(self, key: str, tier: str = "free",
              tokens_needed: float = 1.0) -> Tuple[bool, Dict]:
        """
        Returns (allowed, metadata).
        Atomically checks and deducts tokens.
        """
        now = time.time()
        bucket = self._get_bucket(key, tier)

        # Refill tokens based on elapsed time
        elapsed = now - bucket.last_refill_ts
        earned = elapsed * bucket.refill_rate
        bucket.tokens = min(bucket.capacity, bucket.tokens + earned)
        bucket.last_refill_ts = now

        remaining = bucket.tokens - tokens_needed
        if remaining >= 0:
            bucket.tokens = remaining
            retry_after = 0
            allowed = True
        else:
            # How long until we have enough tokens?
            deficit = tokens_needed - bucket.tokens
            retry_after = int(deficit / bucket.refill_rate) + 1
            allowed = False

        return allowed, {
            "limit": int(bucket.capacity),
            "remaining": max(0, int(bucket.tokens)),
            "retry_after": retry_after,
        }

    def bucket_count(self) -> int:
        return len(self._buckets)


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED     = "CLOSED"
    OPEN       = "OPEN"
    HALF_OPEN  = "HALF_OPEN"


class CircuitBreaker:
    """
    Per-service circuit breaker with rolling window failure tracking.
    States: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
    """

    def __init__(self, service: str, failure_threshold: float = 0.5,
                 min_requests: int = 5, open_timeout: float = 10.0):
        self.service            = service
        self.failure_threshold  = failure_threshold  # 50%
        self.min_requests       = min_requests
        self.open_timeout       = open_timeout        # seconds before HALF_OPEN probe
        self.state              = CircuitState.CLOSED
        self._window: deque     = deque(maxlen=20)    # last 20 results (True=success)
        self._open_since: float = 0.0
        self._probe_allowed     = True

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self._open_since >= self.open_timeout:
                self.state = CircuitState.HALF_OPEN
                self._probe_allowed = True
                print(f"  [CB:{self.service}] OPEN -> HALF_OPEN (probe allowed)")
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            if self._probe_allowed:
                self._probe_allowed = False
                return True
            return False
        return False

    def record_success(self):
        self._window.append(True)
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self._window.clear()
            print(f"  [CB:{self.service}] HALF_OPEN -> CLOSED (probe succeeded)")

    def record_failure(self):
        self._window.append(False)
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self._open_since = time.time()
            self._probe_allowed = False
            print(f"  [CB:{self.service}] HALF_OPEN -> OPEN (probe failed)")
            return

        if len(self._window) >= self.min_requests:
            failure_rate = self._window.count(False) / len(self._window)
            if failure_rate >= self.failure_threshold and self.state == CircuitState.CLOSED:
                self.state = CircuitState.OPEN
                self._open_since = time.time()
                print(f"  [CB:{self.service}] CLOSED -> OPEN "
                      f"(failure rate={failure_rate:.0%})")

    def status(self) -> Dict:
        total = len(self._window)
        failures = self._window.count(False)
        return {
            "state": self.state.value,
            "failure_rate": f"{failures/total:.0%}" if total else "0%",
            "window_size": total,
        }


# ---------------------------------------------------------------------------
# Response Cache
# ---------------------------------------------------------------------------

class ResponseCache:
    """
    LRU cache with TTL support.
    Key: hash of (method, path, sorted_query_params).
    In production: Redis with cache-control header parsing.
    """

    def __init__(self, max_size: int = 1000):
        self._store:    Dict[str, Tuple[Response, float]] = {}  # key -> (resp, expiry)
        self._max_size  = max_size
        self._hits      = 0
        self._misses    = 0

    def _cache_key(self, request: Request) -> str:
        sorted_query = sorted(request.query.items())
        raw = f"{request.method}:{request.path}:{sorted_query}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, request: Request) -> Optional[Response]:
        if request.method != "GET":
            return None  # Only cache GET
        key = self._cache_key(request)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        response, expiry = entry
        if time.time() > expiry:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        cached = Response(
            status_code=response.status_code,
            body=response.body,
            headers=dict(response.headers),
            from_cache=True
        )
        return cached

    def set(self, request: Request, response: Response, ttl: int):
        if request.method != "GET" or ttl <= 0 or response.is_error():
            return
        if len(self._store) >= self._max_size:
            # Evict oldest entry (simple eviction)
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        key = self._cache_key(request)
        self._store[key] = (response, time.time() + ttl)

    def invalidate(self, path_prefix: str):
        """Invalidate all cache entries matching a path prefix."""
        # In practice: tag-based invalidation or explicit key tracking
        to_delete = [k for k in self._store]  # simplified: clear all on write
        for k in to_delete:
            del self._store[k]

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def stats(self) -> Dict:
        return {"hits": self._hits, "misses": self._misses,
                "hit_rate": f"{self.hit_rate:.1%}", "size": len(self._store)}


# ---------------------------------------------------------------------------
# Middleware Base Class
# ---------------------------------------------------------------------------

Handler = Callable[[RequestContext], Response]

class Middleware:
    name: str = "base"

    def handle(self, ctx: RequestContext, next_handler: Handler) -> Response:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Middleware Implementations
# ---------------------------------------------------------------------------

class LoggingMiddleware(Middleware):
    name = "logging"

    def handle(self, ctx: RequestContext, next_handler: Handler) -> Response:
        req = ctx.request
        ctx.log_data.update({"request_id": req.request_id, "method": req.method,
                              "path": req.path, "client_ip": req.client_ip})
        response = next_handler(ctx)
        print(f"  [LOG] {req.method} {req.path} -> {response.status_code} "
              f"({ctx.elapsed_ms():.1f}ms) "
              f"{'[CACHE HIT]' if response.from_cache else ''}")
        return response


class AuthMiddleware(Middleware):
    """Validates JWT or API key. Injects user_id into context."""
    name = "auth"

    # Simulate a small API key registry (in production: Redis lookup)
    VALID_API_KEYS = {
        "key_free_user1":       ("client_001", "free"),
        "key_pro_user2":        ("client_002", "pro"),
        "key_enterprise_corp":  ("client_003", "enterprise"),
    }
    # Simulate valid JWT tokens (in production: RS256 signature verification)
    VALID_JWT_SUBJECTS = {"jwt_alice", "jwt_bob", "jwt_admin"}

    def handle(self, ctx: RequestContext, next_handler: Handler) -> Response:
        req = ctx.request

        # Check if route requires auth; skip auth for public routes and /health
        if req.path.startswith("/health"):
            return next_handler(ctx)
        # Look up route to check auth_required flag
        from_registry = ctx.log_data.get("_registry_checked")
        # Simple path-based public route bypass (production: check route config)
        PUBLIC_PREFIXES = ("/api/products",)
        if any(req.path.startswith(p) for p in PUBLIC_PREFIXES):
            ctx.client_id = "anonymous"
            ctx.request.headers["X-Client-Tier"] = "free"
            return next_handler(ctx)

        api_key = req.headers.get("X-API-Key") or req.query.get("api_key")
        auth_header = req.headers.get("Authorization", "")

        if api_key:
            if api_key in self.VALID_API_KEYS:
                ctx.client_id, tier = self.VALID_API_KEYS[api_key]
                ctx.log_data["client_id"] = ctx.client_id
                ctx.log_data["auth_tier"] = tier
                ctx.request.headers["X-Client-Tier"] = tier
                return next_handler(ctx)
            return Response(401, {"error": "Invalid API key"},
                            {"WWW-Authenticate": "ApiKey"})

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            # Simulate JWT validation (real: decode + verify RS256 signature)
            if token in self.VALID_JWT_SUBJECTS:
                ctx.user_id = token
                ctx.request.headers["X-User-Id"] = token
                ctx.request.headers.pop("Authorization", None)  # strip before forwarding
                return next_handler(ctx)
            return Response(401, {"error": "Invalid or expired JWT"},
                            {"WWW-Authenticate": 'Bearer realm="api"'})

        return Response(401, {"error": "Authentication required"},
                        {"WWW-Authenticate": "Bearer, ApiKey"})


class RateLimitMiddleware(Middleware):
    name = "rate_limit"

    def __init__(self, limiter: RateLimiter):
        self._limiter = limiter

    def handle(self, ctx: RequestContext, next_handler: Handler) -> Response:
        req = ctx.request
        tier = req.headers.get("X-Client-Tier", "free")
        key = ctx.client_id or ctx.user_id or req.client_ip

        # IP-level check (DDoS protection)
        ip_allowed, ip_meta = self._limiter.allow(f"ip:{req.client_ip}", "ip")
        if not ip_allowed:
            return Response(429, {"error": "IP rate limit exceeded"},
                            {"Retry-After": str(ip_meta["retry_after"]),
                             "X-RateLimit-Limit": str(ip_meta["limit"])})

        # Client/API-key-level check
        allowed, meta = self._limiter.allow(f"client:{key}", tier)
        headers = {
            "X-RateLimit-Limit":     str(meta["limit"]),
            "X-RateLimit-Remaining": str(meta["remaining"]),
        }
        if not allowed:
            headers["Retry-After"] = str(meta["retry_after"])
            return Response(429, {"error": "Rate limit exceeded",
                                  "retry_after_seconds": meta["retry_after"]}, headers)

        response = next_handler(ctx)
        response.headers.update(headers)
        return response


class CacheMiddleware(Middleware):
    name = "cache"

    def __init__(self, cache: ResponseCache, route_registry: RouteRegistry):
        self._cache    = cache
        self._registry = route_registry

    def handle(self, ctx: RequestContext, next_handler: Handler) -> Response:
        req = ctx.request

        # Try cache hit (GET only)
        cached = self._cache.get(req)
        if cached:
            cached.headers["X-Cache"] = "HIT"
            return cached

        response = next_handler(ctx)

        # Cache successful GET responses
        if req.method == "GET" and not response.is_error():
            route = self._registry.match(req.method, req.path)
            ttl = route.cache_ttl if route else 0
            if ttl > 0:
                self._cache.set(req, response, ttl)
                response.headers["X-Cache"] = "MISS"

        # Invalidate cache on mutations
        if req.method in ("POST", "PUT", "DELETE") and not response.is_error():
            self._cache.invalidate(req.path)

        return response


class RouterMiddleware(Middleware):
    name = "router"

    def __init__(self, registry: RouteRegistry):
        self._registry = registry

    def handle(self, ctx: RequestContext, next_handler: Handler) -> Response:
        req = ctx.request
        route = self._registry.match(req.method, req.path)
        if not route:
            return Response(404, {"error": f"No route for {req.method} {req.path}"})

        # Canary routing
        if route.should_use_canary():
            ctx.upstream = route.canary_service
        else:
            ctx.upstream = route.select_instance()

        ctx.request.headers["X-Upstream"] = ctx.upstream
        return next_handler(ctx)


class CircuitBreakerMiddleware(Middleware):
    name = "circuit_breaker"

    def __init__(self, breakers: Dict[str, CircuitBreaker]):
        self._breakers = breakers

    def handle(self, ctx: RequestContext, next_handler: Handler) -> Response:
        service = ctx.upstream
        if not service:
            return next_handler(ctx)

        # Get or create circuit breaker for this upstream
        if service not in self._breakers:
            self._breakers[service] = CircuitBreaker(service)
        cb = self._breakers[service]

        if not cb.allow_request():
            return Response(503, {"error": f"Service unavailable: {service}",
                                  "circuit_state": "OPEN"},
                            {"X-Circuit-Breaker": "OPEN",
                             "Retry-After": str(int(cb.open_timeout))})

        response = next_handler(ctx)

        if response.status_code >= 500:
            cb.record_failure()
        else:
            cb.record_success()

        response.headers["X-Circuit-Breaker"] = cb.state.value
        return response


# ---------------------------------------------------------------------------
# Upstream Simulator (simulates actual microservice calls)
# ---------------------------------------------------------------------------

class UpstreamSimulator(Middleware):
    """Simulates forwarding to a backend service. Replace with actual HTTP call."""
    name = "upstream"

    # Service-specific failure rates for demo
    FAILURE_RATES: Dict[str, float] = {}

    def handle(self, ctx: RequestContext, next_handler: Handler) -> Response:
        req = ctx.request
        upstream = ctx.upstream or "unknown"
        path = req.path

        # Simulate service-specific failure rate
        service_name = upstream.split(":")[0] if ":" in upstream else upstream
        fail_rate = self.FAILURE_RATES.get(service_name, 0.0)
        import random
        if random.random() < fail_rate:
            return Response(500, {"error": "Internal service error"})

        # Simulate typical responses
        if path.startswith("/api/users"):
            return Response(200, {"user_id": "u123", "name": "Alice", "email": "alice@example.com"})
        if path.startswith("/api/orders"):
            return Response(200, {"order_id": "o456", "status": "PROCESSING", "total": 9999})
        if path.startswith("/api/products"):
            return Response(200, {"products": [{"id": "p1", "name": "Widget", "price": 1999}]})
        if path == "/health":
            return Response(200, {"status": "healthy"})
        return Response(200, {"message": "OK", "path": path})


# ---------------------------------------------------------------------------
# API Gateway (Assembles Middleware Chain)
# ---------------------------------------------------------------------------

class APIGateway:

    def __init__(self):
        self.registry = RouteRegistry()
        self.cache    = ResponseCache(max_size=500)
        self.limiter  = RateLimiter()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        upstream = UpstreamSimulator()

        self._pipeline: List[Middleware] = [
            LoggingMiddleware(),
            AuthMiddleware(),
            RateLimitMiddleware(self.limiter),
            CacheMiddleware(self.cache, self.registry),
            RouterMiddleware(self.registry),
            CircuitBreakerMiddleware(self.circuit_breakers),
            upstream,
        ]

    def register_route(self, route: Route):
        self.registry.register(route)

    def handle_request(self, request: Request) -> Response:
        """Entry point: execute middleware chain for an incoming request."""
        ctx = RequestContext(request)
        response = self._execute_chain(ctx, 0)
        response.headers["X-Request-Id"] = request.request_id
        response.headers["X-Gateway-Latency-Ms"] = f"{ctx.elapsed_ms():.2f}"
        return response

    def _execute_chain(self, ctx: RequestContext, index: int) -> Response:
        if index >= len(self._pipeline):
            return Response(502, {"error": "End of middleware chain — no handler"})
        middleware = self._pipeline[index]
        return middleware.handle(ctx, lambda c: self._execute_chain(c, index + 1))


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def setup_gateway() -> APIGateway:
    gw = APIGateway()
    gw.register_route(Route(
        path_prefix="/api/users", service="user-service",
        instances=["10.0.1.1:8080", "10.0.1.2:8080"],
        cache_ttl=30, auth_required=True
    ))
    gw.register_route(Route(
        path_prefix="/api/orders", service="order-service",
        instances=["10.0.2.1:8080"],
        cache_ttl=0, auth_required=True
    ))
    gw.register_route(Route(
        path_prefix="/api/products", service="product-service",
        instances=["10.0.3.1:8080", "10.0.3.2:8080", "10.0.3.3:8080"],
        cache_ttl=300, auth_required=False
    ))
    gw.register_route(Route(
        path_prefix="/health", service="gateway",
        instances=["localhost"], methods=["GET"],
        auth_required=False, cache_ttl=0
    ))
    return gw


def demo_auth_and_routing():
    print("=== Auth & Routing Demo ===")
    gw = setup_gateway()

    # Valid API key request
    r1 = Request("GET", "/api/users/123", headers={"X-API-Key": "key_pro_user2"})
    resp = gw.handle_request(r1)
    print(f"Valid API key: {resp.status_code} | RateLimit-Remaining: "
          f"{resp.headers.get('X-RateLimit-Remaining')}")

    # Invalid API key
    r2 = Request("GET", "/api/users/123", headers={"X-API-Key": "invalid_key"})
    resp = gw.handle_request(r2)
    print(f"Invalid API key: {resp.status_code} | {resp.body}")

    # Valid JWT
    r3 = Request("GET", "/api/orders/456",
                 headers={"Authorization": "Bearer jwt_alice"})
    resp = gw.handle_request(r3)
    print(f"Valid JWT: {resp.status_code}")

    # No auth on public route (products)
    r4 = Request("GET", "/api/products", headers={})
    resp = gw.handle_request(r4)
    print(f"Public route (no auth): {resp.status_code}")

    # Unknown route
    r5 = Request("GET", "/api/unknown", headers={"X-API-Key": "key_free_user1"})
    resp = gw.handle_request(r5)
    print(f"Unknown route: {resp.status_code} | {resp.body}")


def demo_rate_limiting():
    print("\n=== Rate Limiting Demo ===")
    gw = setup_gateway()

    # Override free tier to tiny limit for demo
    gw.limiter.TIERS["free"] = (3, 3 / 60)

    print("Sending 5 requests with free tier key (limit=3)...")
    for i in range(5):
        r = Request("GET", "/api/products", headers={"X-API-Key": "key_free_user1"},
                    client_ip="1.2.3.4")
        resp = gw.handle_request(r)
        remaining = resp.headers.get("X-RateLimit-Remaining", "?")
        retry_after = resp.headers.get("Retry-After", "")
        msg = f"  Request {i+1}: {resp.status_code} | remaining={remaining}"
        if retry_after:
            msg += f" | retry_after={retry_after}s"
        print(msg)


def demo_cache():
    print("\n=== Response Cache Demo ===")
    gw = setup_gateway()
    headers = {"X-API-Key": "key_pro_user2"}

    r1 = Request("GET", "/api/products", headers=headers)
    resp1 = gw.handle_request(r1)
    print(f"First request (MISS): status={resp1.status_code}, cache={resp1.headers.get('X-Cache')}")

    r2 = Request("GET", "/api/products", headers=headers)
    resp2 = gw.handle_request(r2)
    print(f"Second request (HIT): status={resp2.status_code}, cache={resp2.headers.get('X-Cache')}")
    print(f"Cache stats: {gw.cache.stats()}")


def demo_circuit_breaker():
    print("\n=== Circuit Breaker Demo ===")
    gw = setup_gateway()

    # Inject failure rate for order service (100% failure)
    # Key matches upstream.split(":")[0] where upstream = "10.0.2.1:8080"
    UpstreamSimulator.FAILURE_RATES["10.0.2.1"] = 1.0

    # Pre-create the circuit breaker for the order service upstream
    order_upstream = "10.0.2.1:8080"
    cb = CircuitBreaker(order_upstream, failure_threshold=0.5, min_requests=5,
                        open_timeout=2.0)
    gw.circuit_breakers[order_upstream] = cb

    print("Sending 8 requests to 100%-failing order service "
          "(trips at 5 failures with 50% threshold)...")

    for i in range(8):
        # Create fresh headers dict each time — auth middleware strips Authorization
        r = Request("GET", "/api/orders/123",
                    headers={"Authorization": "Bearer jwt_alice"},
                    client_ip=f"10.1.1.{i+1}")
        resp = gw.handle_request(r)
        cb_header = resp.headers.get("X-Circuit-Breaker", "?")
        status_label = {200: "OK", 500: "FAIL", 503: "OPEN-FAST-FAIL"}.get(
            resp.status_code, str(resp.status_code))
        print(f"  Request {i+1}: {resp.status_code} ({status_label}) | "
              f"CB={cb_header} | {cb.status()}")

    # Restore service health and advance time past open_timeout
    UpstreamSimulator.FAILURE_RATES["10.0.2.1"] = 0.0
    cb._open_since = time.time() - cb.open_timeout - 0.1  # fast-forward timer

    print("\nService recovered. Next request triggers HALF_OPEN probe...")
    r = Request("GET", "/api/orders/123",
                headers={"Authorization": "Bearer jwt_alice"},
                client_ip="10.1.1.99")
    resp = gw.handle_request(r)
    print(f"  Probe result: status={resp.status_code}, "
          f"CB={resp.headers.get('X-Circuit-Breaker')} (expected: CLOSED after success)")


def demo_middleware_chain():
    print("\n=== Middleware Chain Flow Demo ===")
    gw = setup_gateway()
    print("Middleware pipeline order:")
    for i, mw in enumerate(gw._pipeline):
        print(f"  {i+1}. {mw.name}")
    print()
    r = Request("GET", "/api/users/me",
                headers={"Authorization": "Bearer jwt_bob"},
                client_ip="192.168.1.1")
    resp = gw.handle_request(r)
    print(f"Final response: {resp.status_code}")
    print(f"Response headers: {json.dumps(resp.headers, indent=4)}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_auth_and_routing()
    demo_rate_limiting()
    demo_cache()
    demo_circuit_breaker()
    demo_middleware_chain()
