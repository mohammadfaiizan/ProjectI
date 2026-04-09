"""
API GATEWAY DESIGN
===================

Problem Statement:
Microservices expose dozens of internal APIs. Clients shouldn't talk to each
service directly — this creates coupling, duplicates cross-cutting concerns
(auth, rate limiting, logging), and exposes internal topology. An API Gateway
is the single entry point that handles all of this centrally.

API Gateway Responsibilities:
  1. Routing        → path-based routing to downstream services
  2. Authentication → validate JWT/API keys before forwarding
  3. Rate Limiting  → prevent abuse (per-user or per-IP quotas)
  4. Request/Response Transformation → adapt payloads between versions
  5. Aggregation    → combine multiple service calls into one response (BFF)
  6. Logging/Tracing → centralized observability
  7. Circuit Breaking → stop cascading failures

Popular API Gateways:
  AWS API Gateway, Kong, Apigee, Traefik, Nginx Plus, Envoy
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import time
import hashlib
import uuid


class AuthMethod(Enum):
    JWT       = "jwt"
    API_KEY   = "api_key"
    NONE      = "none"


class RateLimitPolicy(Enum):
    PER_USER  = "per_user"
    PER_IP    = "per_ip"
    GLOBAL    = "global"


@dataclass
class GatewayRequest:
    request_id : str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    client_ip  : str = "0.0.0.0"
    method     : str = "GET"
    path       : str = "/"
    headers    : Dict[str, str] = field(default_factory=dict)
    body       : Dict = field(default_factory=dict)
    user_id    : Optional[str] = None


@dataclass
class GatewayResponse:
    status_code : int
    body        : Any
    headers     : Dict[str, str] = field(default_factory=dict)
    latency_ms  : float = 0.0
    served_by   : str = "gateway"


@dataclass
class Route:
    path_prefix     : str
    service_url     : str
    service_name    : str
    auth_required   : bool = True
    rate_limit_rpm  : int  = 1000    # requests per minute
    strip_prefix    : bool = True    # remove path prefix when forwarding


# ─────────────────────────────────────────────
# AUTH MIDDLEWARE
# ─────────────────────────────────────────────

class AuthMiddleware:
    """Validates JWT tokens and API keys."""

    # Simulated valid tokens
    VALID_TOKENS = {
        "Bearer token-alice" : "alice",
        "Bearer token-bob"   : "bob",
        "Bearer token-admin" : "admin",
    }
    VALID_API_KEYS = {
        "key-partner-1": "partner-A",
        "key-partner-2": "partner-B",
    }

    def authenticate(self, req: GatewayRequest) -> Optional[str]:
        """Returns user_id if authenticated, None if rejected."""
        auth_header = req.headers.get("Authorization", "")
        api_key     = req.headers.get("X-API-Key", "")

        if auth_header in self.VALID_TOKENS:
            return self.VALID_TOKENS[auth_header]
        if api_key in self.VALID_API_KEYS:
            return self.VALID_API_KEYS[api_key]
        return None


# ─────────────────────────────────────────────
# RATE LIMITER (Token Bucket)
# ─────────────────────────────────────────────

class RateLimiter:
    """
    Token bucket rate limiter.
    Each user/IP gets a bucket that refills at `rate_rpm` tokens/minute.
    """

    def __init__(self, rate_rpm: int = 60, burst: int = 10):
        self.rate_rpm  = rate_rpm
        self.burst     = burst
        self._buckets  : Dict[str, Dict] = {}

    def _get_bucket(self, key: str) -> Dict:
        if key not in self._buckets:
            self._buckets[key] = {"tokens": float(self.burst), "last_refill": time.time()}
        return self._buckets[key]

    def _refill(self, bucket: Dict):
        now     = time.time()
        elapsed = now - bucket["last_refill"]
        refill  = elapsed * (self.rate_rpm / 60.0)
        bucket["tokens"]      = min(self.burst, bucket["tokens"] + refill)
        bucket["last_refill"] = now

    def allow(self, key: str) -> bool:
        bucket = self._get_bucket(key)
        self._refill(bucket)
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False

    def remaining(self, key: str) -> int:
        bucket = self._get_bucket(key)
        self._refill(bucket)
        return int(bucket["tokens"])


# ─────────────────────────────────────────────
# REQUEST TRANSFORMER
# ─────────────────────────────────────────────

class RequestTransformer:
    """
    Adapts requests between API versions or formats.
    Example: v1 clients send snake_case; v2 services expect camelCase.
    """

    def transform_v1_to_v2(self, body: Dict) -> Dict:
        """Convert snake_case keys to camelCase for v2 service."""
        def to_camel(key: str) -> str:
            parts = key.split("_")
            return parts[0] + "".join(p.capitalize() for p in parts[1:])
        return {to_camel(k): v for k, v in body.items()}

    def add_correlation_id(self, req: GatewayRequest):
        req.headers["X-Correlation-ID"] = req.request_id
        req.headers["X-Gateway-Time"]   = str(time.time())


# ─────────────────────────────────────────────
# DOWNSTREAM SERVICE (mock)
# ─────────────────────────────────────────────

class DownstreamService:
    def __init__(self, name: str, healthy: bool = True):
        self.name    = name
        self.healthy = healthy
        self.calls   = 0

    def call(self, method: str, path: str, body: Dict = None) -> GatewayResponse:
        self.calls += 1
        if not self.healthy:
            raise ConnectionError(f"{self.name} is down")
        return GatewayResponse(
            status_code=200,
            body={"service": self.name, "path": path, "result": "ok"},
            headers={"X-Service": self.name}
        )


# ─────────────────────────────────────────────
# API GATEWAY
# ─────────────────────────────────────────────

class APIGateway:
    """
    Central entry point for all client requests.
    Handles: routing, auth, rate limiting, transformation, logging.
    """

    def __init__(self):
        self.routes       : List[Route] = []
        self.services     : Dict[str, DownstreamService] = {}
        self.auth         = AuthMiddleware()
        self.rate_limiter = RateLimiter(rate_rpm=10, burst=5)
        self.transformer  = RequestTransformer()
        self._access_log  : List[Dict] = []
        self.request_count = 0
        self.auth_failures = 0
        self.rate_limited  = 0

    def register_route(self, route: Route, service: DownstreamService):
        self.routes.append(route)
        self.services[route.service_name] = service

    def _match_route(self, path: str) -> Optional[Route]:
        for route in self.routes:
            if path.startswith(route.path_prefix):
                return route
        return None

    def _log(self, req: GatewayRequest, resp: GatewayResponse, route: Optional[Route]):
        entry = {
            "request_id" : req.request_id,
            "method"     : req.method,
            "path"       : req.path,
            "user"       : req.user_id,
            "status"     : resp.status_code,
            "latency_ms" : resp.latency_ms,
            "service"    : route.service_name if route else "gateway",
        }
        self._access_log.append(entry)

    def handle(self, req: GatewayRequest) -> GatewayResponse:
        self.request_count += 1
        start = time.perf_counter()

        # 1. Route matching
        route = self._match_route(req.path)
        if not route:
            resp = GatewayResponse(404, {"error": "No route matched"})
            self._log(req, resp, None)
            return resp

        # 2. Authentication
        if route.auth_required:
            user_id = self.auth.authenticate(req)
            if not user_id:
                self.auth_failures += 1
                resp = GatewayResponse(401, {"error": "Unauthorized"})
                self._log(req, resp, route)
                return resp
            req.user_id = user_id

        # 3. Rate limiting (per user or per IP)
        rate_key = req.user_id or req.client_ip
        if not self.rate_limiter.allow(rate_key):
            self.rate_limited += 1
            remaining = self.rate_limiter.remaining(rate_key)
            resp = GatewayResponse(
                429, {"error": "Rate limit exceeded", "retry_after": "60s"},
                headers={"X-RateLimit-Remaining": str(remaining)}
            )
            self._log(req, resp, route)
            return resp

        # 4. Request transformation
        self.transformer.add_correlation_id(req)
        if req.body:
            req.body = self.transformer.transform_v1_to_v2(req.body)

        # 5. Forward to downstream service
        service = self.services.get(route.service_name)
        forward_path = req.path[len(route.path_prefix):] if route.strip_prefix else req.path

        try:
            resp = service.call(req.method, forward_path, req.body)
        except ConnectionError as e:
            resp = GatewayResponse(502, {"error": str(e)})

        resp.latency_ms = round((time.perf_counter() - start) * 1000, 2)
        resp.headers["X-Request-ID"] = req.request_id

        self._log(req, resp, route)
        return resp

    def report(self):
        print(f"\n  API Gateway Metrics:")
        print(f"    Total requests : {self.request_count}")
        print(f"    Auth failures  : {self.auth_failures}")
        print(f"    Rate limited   : {self.rate_limited}")
        print(f"\n  Downstream Service Calls:")
        for name, svc in self.services.items():
            print(f"    {name}: {svc.calls} calls")
        print(f"\n  Access Log (last 5):")
        print(f"  {'ID':<10} {'Method':<6} {'Path':<25} {'User':<12} {'Status':<7} {'ms'}")
        print(f"  {'─'*70}")
        for entry in self._access_log[-5:]:
            print(f"  {entry['request_id']:<10} {entry['method']:<6} {entry['path']:<25} "
                  f"{str(entry['user']):<12} {entry['status']:<7} {entry['latency_ms']}")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_api_gateway():
    print("=" * 65)
    print("API GATEWAY DESIGN")
    print("=" * 65)

    # ── Setup ─────────────────────────────────
    gateway = APIGateway()

    # Register routes and services
    gateway.register_route(
        Route("/api/v1/users",    "http://user-service:8001",    "user-service",    auth_required=True,  rate_limit_rpm=100),
        DownstreamService("user-service")
    )
    gateway.register_route(
        Route("/api/v1/orders",   "http://order-service:8002",   "order-service",   auth_required=True,  rate_limit_rpm=50),
        DownstreamService("order-service")
    )
    gateway.register_route(
        Route("/api/v1/products", "http://product-service:8003", "product-service", auth_required=False, rate_limit_rpm=500),
        DownstreamService("product-service")
    )
    gateway.register_route(
        Route("/health",          "http://health-service:8004",  "health-service",  auth_required=False, rate_limit_rpm=9999),
        DownstreamService("health-service")
    )

    # ── Test Requests ──────────────────────────
    print("\n[1] ROUTING AND AUTH")
    print("─" * 55)
    requests = [
        GatewayRequest(client_ip="1.1.1.1", method="GET",  path="/api/v1/users/123",
                       headers={"Authorization": "Bearer token-alice"}),
        GatewayRequest(client_ip="1.1.1.2", method="GET",  path="/api/v1/users/456",
                       headers={}),  # no auth → 401
        GatewayRequest(client_ip="1.1.1.3", method="GET",  path="/api/v1/products",
                       headers={}),  # no auth needed
        GatewayRequest(client_ip="1.1.1.4", method="POST", path="/api/v1/orders",
                       headers={"Authorization": "Bearer token-bob"},
                       body={"user_id": "u2", "product_id": "p1", "quantity": 2}),
        GatewayRequest(client_ip="1.1.1.5", method="GET",  path="/unknown/path",
                       headers={"Authorization": "Bearer token-alice"}),  # 404
    ]
    for req in requests:
        resp = gateway.handle(req)
        print(f"  {req.method:<5} {req.path:<30} → {resp.status_code}  "
              f"user={req.user_id}  {resp.latency_ms}ms")

    # ── Rate Limiting ─────────────────────────
    print("\n\n[2] RATE LIMITING (burst=5, limit=10 RPM)")
    print("─" * 55)
    for i in range(8):
        req = GatewayRequest(
            client_ip="9.9.9.9", method="GET", path="/api/v1/products",
            headers={}
        )
        # Override rate limiter key
        resp = gateway.handle(req)
        print(f"  Request {i+1}: {resp.status_code}  "
              f"remaining={gateway.rate_limiter.remaining('9.9.9.9')}")

    # ── Request Transformation ────────────────
    print("\n\n[3] REQUEST TRANSFORMATION (snake_case → camelCase)")
    print("─" * 55)
    t = RequestTransformer()
    v1_body = {"user_id": "u1", "first_name": "Alice", "last_name": "Smith",
               "is_active": True, "created_at": "2024-01-01"}
    v2_body = t.transform_v1_to_v2(v1_body)
    print(f"  v1 payload: {v1_body}")
    print(f"  v2 payload: {v2_body}")

    # ── Gateway Report ─────────────────────────
    gateway.report()

    # ── Architecture ───────────────────────────
    print("\n\n[4] API GATEWAY RESPONSIBILITIES")
    print("─" * 55)
    responsibilities = [
        ("Routing",          "Map path /api/v1/users → user-service:8001"),
        ("Authentication",   "Validate JWT, API keys before forwarding"),
        ("Rate Limiting",    "Per-user/IP token bucket, return 429"),
        ("Load Balancing",   "Distribute across service instances"),
        ("SSL Termination",  "Decrypt HTTPS; backends get plain HTTP"),
        ("Request Transform","v1→v2 field mapping, header injection"),
        ("Response Agg.",    "BFF: merge user+orders+products in 1 response"),
        ("Logging/Tracing",  "Centralized access log, correlation IDs"),
        ("Circuit Breaking", "Stop forwarding to unhealthy services"),
        ("Caching",          "Cache GET responses, reduce backend calls"),
    ]
    for resp, detail in responsibilities:
        print(f"  • {resp:<22} {detail}")

    print("\n\n[5] GATEWAY vs SERVICE MESH")
    print("─" * 55)
    print("  API Gateway : North-South traffic (client → services)")
    print("  Service Mesh: East-West traffic (service ↔ service)")
    print("  Often used together: Gateway at edge, Mesh inside cluster")


if __name__ == "__main__":
    demonstrate_api_gateway()
