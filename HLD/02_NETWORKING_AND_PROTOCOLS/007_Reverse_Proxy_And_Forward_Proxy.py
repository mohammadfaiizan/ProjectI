"""
REVERSE PROXY AND FORWARD PROXY
=================================

Problem Statement:
Proxies sit between clients and servers, adding indirection that enables
security, load distribution, caching, and anonymity. Engineers must know
when to use forward vs reverse proxies and how tools like Nginx work.

Forward Proxy (client-side):
  Client → Forward Proxy → Internet → Server
  - Hides client identity (anonymity)
  - Content filtering (corporate firewalls)
  - Bypass geo-restrictions
  - Cache outbound requests

Reverse Proxy (server-side):
  Client → Reverse Proxy → Backend Servers
  - SSL/TLS termination
  - Load balancing
  - Static asset caching
  - DDoS protection
  - Single public IP for many backends

Popular Tools:
  Nginx, HAProxy, Traefik, Envoy, Cloudflare
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import random
import hashlib


class ProxyType(Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


class BackendHealth(Enum):
    HEALTHY   = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass
class ProxyRequest:
    client_ip  : str
    destination: str
    method     : str
    path       : str
    headers    : Dict[str, str] = field(default_factory=dict)
    body       : str = ""

    @property
    def is_https(self) -> bool:
        return self.destination.startswith("https://")


@dataclass
class ProxyResponse:
    status_code  : int
    body         : str
    headers      : Dict[str, str] = field(default_factory=dict)
    served_by    : str = ""
    cached       : bool = False
    ssl_terminated: bool = False
    latency_ms   : float = 0.0


@dataclass
class BackendServer:
    server_id   : str
    address     : str
    port        : int
    health      : BackendHealth = BackendHealth.HEALTHY
    active_conn : int = 0
    total_served: int = 0

    def handle(self, path: str) -> Tuple[int, str]:
        self.active_conn += 1
        self.total_served += 1
        # Simulate response
        time.sleep(0.001)
        self.active_conn -= 1
        return 200, f"Response from {self.server_id}: {path}"


# ─────────────────────────────────────────────
# FORWARD PROXY
# ─────────────────────────────────────────────

class ForwardProxy:
    """
    Client-side proxy that forwards requests on behalf of the client.
    Hides client IP; enables filtering and caching.
    """

    BLOCKED_DOMAINS = {"social-media.com", "gambling.net", "malware.bad"}

    def __init__(self, proxy_ip: str = "10.0.0.1"):
        self.proxy_ip     = proxy_ip
        self._cache       : Dict[str, ProxyResponse] = {}
        self.requests_made = 0
        self.blocked_count = 0
        self.cache_hits    = 0

    def _is_blocked(self, destination: str) -> bool:
        for blocked in self.BLOCKED_DOMAINS:
            if blocked in destination:
                return True
        return False

    def _cache_key(self, req: ProxyRequest) -> str:
        return hashlib.md5(f"{req.destination}{req.path}".encode()).hexdigest()

    def forward(self, req: ProxyRequest) -> ProxyResponse:
        self.requests_made += 1

        # Content filtering
        if self._is_blocked(req.destination):
            self.blocked_count += 1
            return ProxyResponse(
                status_code=403,
                body=f"Blocked by proxy policy: {req.destination}",
                served_by=self.proxy_ip
            )

        # Cache check (GET only)
        if req.method == "GET":
            key = self._cache_key(req)
            if key in self._cache:
                self.cache_hits += 1
                resp = self._cache[key]
                resp.cached = True
                print(f"  ForwardProxy: cache HIT for {req.destination}{req.path}")
                return resp

        # Forward request — client IP is hidden; proxy IP is seen by server
        start = time.perf_counter()
        print(f"  ForwardProxy: {req.client_ip} → proxy({self.proxy_ip}) → {req.destination}{req.path}")
        print(f"    Server sees: X-Forwarded-For: {self.proxy_ip}  (client hidden)")

        resp = ProxyResponse(
            status_code=200,
            body=f"Content from {req.destination}{req.path}",
            headers={"Via": f"1.1 {self.proxy_ip} (ForwardProxy)"},
            served_by=self.proxy_ip,
            latency_ms=round((time.perf_counter() - start) * 1000 + 20, 2)
        )

        if req.method == "GET":
            self._cache[self._cache_key(req)] = resp

        return resp

    def report(self):
        print(f"\n  ForwardProxy Stats:")
        print(f"    Requests   : {self.requests_made}")
        print(f"    Blocked    : {self.blocked_count}")
        print(f"    Cache hits : {self.cache_hits}")


# ─────────────────────────────────────────────
# SSL TERMINATOR
# ─────────────────────────────────────────────

class SSLTerminator:
    """
    Terminates TLS at the proxy; backends receive plain HTTP.
    Offloads CPU-intensive crypto from application servers.
    """

    def __init__(self, cert_domain: str):
        self.cert_domain    = cert_domain
        self.certs_verified = 0
        self.tls_terminated = 0

    def terminate(self, req: ProxyRequest) -> ProxyRequest:
        if req.is_https:
            self.tls_terminated += 1
            print(f"  SSLTerminator: TLS terminated for {req.destination}")
            print(f"    Proxy decrypts → backend receives plain HTTP")
            req.headers["X-Forwarded-Proto"] = "https"
            req.destination = req.destination.replace("https://", "http://")
        return req


# ─────────────────────────────────────────────
# REVERSE PROXY
# ─────────────────────────────────────────────

class ReverseProxy:
    """
    Server-side proxy. Clients talk to the proxy; backends are hidden.
    Provides: SSL termination, load balancing, caching, rate limiting.
    """

    def __init__(self, public_ip: str = "203.0.113.1"):
        self.public_ip    = public_ip
        self.backends     : List[BackendServer] = []
        self._rr_index    = 0
        self._cache       : Dict[str, ProxyResponse] = {}
        self.ssl_terminator = SSLTerminator("example.com")
        self.requests_total = 0
        self.cache_hits     = 0

    def add_backend(self, backend: BackendServer):
        self.backends.append(backend)

    def _healthy_backends(self) -> List[BackendServer]:
        return [b for b in self.backends if b.health == BackendHealth.HEALTHY]

    def _round_robin(self) -> Optional[BackendServer]:
        healthy = self._healthy_backends()
        if not healthy:
            return None
        backend = healthy[self._rr_index % len(healthy)]
        self._rr_index += 1
        return backend

    def _least_conn(self) -> Optional[BackendServer]:
        healthy = self._healthy_backends()
        if not healthy:
            return None
        return min(healthy, key=lambda b: b.active_conn)

    def _cache_key(self, path: str) -> str:
        return hashlib.md5(path.encode()).hexdigest()

    def handle(self, req: ProxyRequest, use_least_conn: bool = False) -> ProxyResponse:
        self.requests_total += 1
        start = time.perf_counter()

        # SSL termination
        req = self.ssl_terminator.terminate(req)

        # Cache for GET requests (static assets)
        if req.method == "GET" and req.path.startswith("/static/"):
            key = self._cache_key(req.path)
            if key in self._cache:
                self.cache_hits += 1
                resp = self._cache[key]
                resp.cached = True
                print(f"  ReverseProxy: static asset cache HIT {req.path}")
                return resp

        # Route to backend
        backend = self._least_conn() if use_least_conn else self._round_robin()
        if not backend:
            return ProxyResponse(502, "Bad Gateway — no healthy backends",
                                  served_by=self.public_ip)

        req.headers["X-Real-IP"]       = req.client_ip
        req.headers["X-Forwarded-For"] = req.client_ip
        req.headers["Host"]            = backend.address

        status, body = backend.handle(req.path)
        latency = round((time.perf_counter() - start) * 1000, 2)

        resp = ProxyResponse(
            status_code=status, body=body,
            headers={"X-Served-By": backend.server_id,
                     "X-Proxy": self.public_ip},
            served_by=backend.server_id,
            ssl_terminated=True, latency_ms=latency
        )

        # Cache static assets
        if req.method == "GET" and req.path.startswith("/static/"):
            self._cache[self._cache_key(req.path)] = resp

        print(f"  ReverseProxy: {req.client_ip} → proxy → {backend.server_id}  "
              f"({latency:.1f}ms, ssl={resp.ssl_terminated})")
        return resp

    def health_check(self):
        """Mark backends as unhealthy if simulated check fails."""
        for backend in self.backends:
            # Simulate: backend-3 goes down
            if backend.server_id == "backend-3":
                backend.health = BackendHealth.UNHEALTHY
                print(f"  HealthCheck: {backend.server_id} → UNHEALTHY (removed from pool)")
            else:
                print(f"  HealthCheck: {backend.server_id} → HEALTHY")

    def report(self):
        print(f"\n  ReverseProxy Stats:")
        print(f"    Total requests  : {self.requests_total}")
        print(f"    Static cache hits: {self.cache_hits}")
        print(f"    SSL terminated  : {self.ssl_terminator.tls_terminated}")
        print(f"\n  Backend Distribution:")
        for b in self.backends:
            status = "✅" if b.health == BackendHealth.HEALTHY else "❌"
            print(f"    {status} {b.server_id}: {b.total_served} requests  "
                  f"active_conn={b.active_conn}")


# ─────────────────────────────────────────────
# NGINX CONFIG REPRESENTATION
# ─────────────────────────────────────────────

class NginxConfig:
    """Generates representative Nginx config snippets."""

    @staticmethod
    def reverse_proxy_config() -> str:
        return """
  # Nginx as Reverse Proxy (simplified)
  upstream backend_pool {
      least_conn;                          # LB algorithm
      server 10.0.1.1:8080;
      server 10.0.1.2:8080;
      server 10.0.1.3:8080;
      keepalive 32;                        # keep connections alive
  }

  server {
      listen 443 ssl http2;
      server_name example.com;

      ssl_certificate     /etc/ssl/example.com.crt;
      ssl_certificate_key /etc/ssl/example.com.key;

      # Static asset caching
      location /static/ {
          proxy_cache STATIC;
          proxy_cache_valid 200 1d;
          add_header X-Cache-Status $upstream_cache_status;
          proxy_pass http://backend_pool;
      }

      # API: no cache, pass to backends
      location /api/ {
          proxy_pass         http://backend_pool;
          proxy_set_header   X-Real-IP        $remote_addr;
          proxy_set_header   X-Forwarded-For  $proxy_add_x_forwarded_for;
          proxy_set_header   X-Forwarded-Proto $scheme;
          proxy_set_header   Host             $host;
          proxy_read_timeout 30s;
      }
  }"""

    @staticmethod
    def forward_proxy_config() -> str:
        return """
  # Nginx as Forward Proxy (corporate outbound)
  server {
      listen 8080;
      resolver 8.8.8.8;

      location / {
          proxy_pass $scheme://$host$request_uri;
          proxy_set_header Host $host;
          # Blocks social media
          if ($host ~* "(facebook|twitter|instagram)\\.com") {
              return 403 "Blocked by policy";
          }
      }
  }"""


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_proxies():
    print("=" * 65)
    print("REVERSE PROXY AND FORWARD PROXY")
    print("=" * 65)

    # ── Forward Proxy ─────────────────────────
    print("\n[1] FORWARD PROXY — CLIENT ANONYMITY & FILTERING")
    print("─" * 55)
    fwd = ForwardProxy(proxy_ip="10.100.0.1")

    requests = [
        ProxyRequest("192.168.1.5", "https://news.example.com", "GET", "/headlines"),
        ProxyRequest("192.168.1.6", "https://social-media.com",  "GET", "/feed"),      # blocked
        ProxyRequest("192.168.1.7", "https://news.example.com", "GET", "/headlines"),  # cached
        ProxyRequest("192.168.1.8", "https://api.partner.com",  "POST","/data"),
    ]
    for req in requests:
        resp = fwd.forward(req)
        print(f"  {req.client_ip} → {req.destination}{req.path} → {resp.status_code} "
              f"{'(cached)' if resp.cached else ''}")
    fwd.report()

    # ── Reverse Proxy ─────────────────────────
    print("\n\n[2] REVERSE PROXY — SSL TERMINATION + LOAD BALANCING")
    print("─" * 55)
    proxy = ReverseProxy(public_ip="203.0.113.10")
    for i in range(1, 4):
        proxy.add_backend(BackendServer(f"backend-{i}", f"10.0.1.{i}", 8080))

    # Simulate health check — backend-3 goes down
    print("\n  Health checks:")
    proxy.health_check()

    # Send requests — distributed across healthy backends
    print("\n  Routing requests:")
    test_requests = [
        ProxyRequest("1.2.3.4", "https://example.com", "GET",  "/api/users"),
        ProxyRequest("1.2.3.5", "https://example.com", "GET",  "/api/orders"),
        ProxyRequest("1.2.3.6", "https://example.com", "POST", "/api/checkout"),
        ProxyRequest("1.2.3.7", "https://example.com", "GET",  "/static/app.js"),  # cached
        ProxyRequest("1.2.3.8", "https://example.com", "GET",  "/static/app.js"),  # cache HIT
        ProxyRequest("1.2.3.9", "https://example.com", "GET",  "/api/users"),
    ]
    for req in test_requests:
        proxy.handle(req)

    proxy.report()

    # ── Nginx Config ──────────────────────────
    print("\n\n[3] NGINX CONFIGURATION EXAMPLES")
    print("─" * 55)
    print("  Reverse Proxy Config:")
    print(NginxConfig.reverse_proxy_config())
    print("\n  Forward Proxy Config:")
    print(NginxConfig.forward_proxy_config())

    # ── Comparison ────────────────────────────
    print("\n\n[4] FORWARD vs REVERSE PROXY COMPARISON")
    print("─" * 55)
    rows = [
        ("Who deploys it", "Client / corporate IT",   "Server / ops team"),
        ("Who it serves",  "Client (hides client)",   "Server (hides servers)"),
        ("Client knows?",  "Yes — configured by user","No — transparent"),
        ("SSL",            "Can intercept HTTPS (MitM)","Terminates at proxy"),
        ("Load balance",   "No",                       "Yes"),
        ("Caching",        "Outbound cache",           "Inbound cache"),
        ("DDoS protection","No",                       "Yes — absorbs at edge"),
        ("Use cases",      "VPN, filtering, anonymity","Nginx, Cloudflare, CDN"),
    ]
    print(f"  {'Aspect':<22} {'Forward Proxy':<28} {'Reverse Proxy'}")
    print(f"  {'─'*70}")
    for aspect, fwd_v, rev_v in rows:
        print(f"  {aspect:<22} {fwd_v:<28} {rev_v}")


if __name__ == "__main__":
    demonstrate_proxies()
