"""
LAYER 4 VS LAYER 7 LOAD BALANCING
=====================================

Problem Statement:
Not all load balancers are equal. L4 LBs route at the TCP/UDP level —
blindly forwarding byte streams. L7 LBs understand HTTP and can make
intelligent routing decisions based on content. The trade-off is
flexibility vs raw throughput.

Layer 4 (Transport Layer):
  - Routes based on: IP address, TCP/UDP port, protocol
  - Does NOT look inside the packet payload
  - Very fast — minimal processing overhead
  - Supports TCP, UDP, TLS pass-through
  - No content-based routing
  - Examples: AWS NLB, HAProxy in TCP mode

Layer 7 (Application Layer):
  - Routes based on: URL path, HTTP method, headers, cookies, body
  - Parses and understands HTTP/HTTPS
  - Supports: A/B testing, canary deployments, API versioning
  - Can rewrite URLs, add headers, cache responses
  - Higher latency than L4 (~1ms extra)
  - Examples: AWS ALB, Nginx, Traefik, Envoy

When to Use:
  L4: Gaming (UDP), DNS, low-latency financial systems, non-HTTP protocols
  L7: HTTP APIs, microservices routing, A/B testing, content-based splitting
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib
import time
import random


class RoutingDecision(Enum):
    BACKEND_A  = "backend_a"
    BACKEND_B  = "backend_b"
    REJECT     = "reject"
    REDIRECT   = "redirect"


@dataclass
class TCPPacket:
    """L4 packet — only IP/port visible to L4 LB."""
    src_ip   : str
    src_port : int
    dst_ip   : str
    dst_port : int
    payload  : bytes = b""   # L4 LB does not read this


@dataclass
class HTTPRequest:
    """L7 request — full HTTP content visible to L7 LB."""
    src_ip    : str
    method    : str
    host      : str
    path      : str
    headers   : Dict[str, str] = field(default_factory=dict)
    body      : str = ""
    version   : str = "HTTP/1.1"

    @property
    def user_agent(self) -> str:
        return self.headers.get("User-Agent", "")

    @property
    def auth_token(self) -> str:
        return self.headers.get("Authorization", "")

    @property
    def cookie(self) -> str:
        return self.headers.get("Cookie", "")


@dataclass
class Backend:
    backend_id : str
    address    : str
    port       : int
    requests   : int = 0

    def handle(self, info: str = "") -> Tuple[int, str]:
        self.requests += 1
        return 200, f"OK from {self.backend_id}"


# ─────────────────────────────────────────────
# L4 LOAD BALANCER
# ─────────────────────────────────────────────

class L4LoadBalancer:
    """
    Routes TCP/UDP traffic based solely on IP:port.
    Cannot read HTTP headers, cookies, or URL paths.
    """

    def __init__(self, vip: str, vport: int, protocol: str = "tcp"):
        self.vip        = vip
        self.vport      = vport
        self.protocol   = protocol
        self.backends   : List[Backend] = []
        self._rr_index  = 0
        self.packets_handled = 0

    def add_backend(self, backend: Backend):
        self.backends.append(backend)

    def route(self, packet: TCPPacket) -> Backend:
        """
        L4 routing: only src_ip, src_port, dst_port used.
        Cannot make content-based decisions.
        """
        self.packets_handled += 1
        # 4-tuple hash for connection persistence (same connection → same backend)
        h = hash((packet.src_ip, packet.src_port, packet.dst_ip, packet.dst_port))
        backend = self.backends[abs(h) % len(self.backends)]
        return backend

    def report(self):
        print(f"\n  L4 LoadBalancer [{self.vip}:{self.vport}]:")
        print(f"    Packets handled : {self.packets_handled}")
        for b in self.backends:
            print(f"    {b.backend_id}: {b.requests} connections")


# ─────────────────────────────────────────────
# L7 LOAD BALANCER
# ─────────────────────────────────────────────

class L7LoadBalancer:
    """
    Routes HTTP traffic based on full request content.
    Supports: path routing, header routing, A/B testing, rewrites.
    """

    def __init__(self, name: str = "L7-LB"):
        self.name         = name
        self._routes      : List[Dict] = []   # ordered rules
        self.backends     : Dict[str, Backend] = {}
        self.requests_handled = 0
        self.routing_log  : List[str] = []

    def add_backend(self, backend: Backend):
        self.backends[backend.backend_id] = backend

    def add_path_route(self, path_prefix: str, backend_id: str):
        self._routes.append({"type": "path", "prefix": path_prefix, "target": backend_id})

    def add_header_route(self, header: str, value: str, backend_id: str):
        self._routes.append({"type": "header", "header": header, "value": value, "target": backend_id})

    def add_canary_route(self, path_prefix: str, canary_id: str, stable_id: str, canary_pct: int):
        self._routes.append({"type": "canary", "prefix": path_prefix,
                              "canary": canary_id, "stable": stable_id, "pct": canary_pct})

    def route(self, req: HTTPRequest) -> Tuple[Optional[Backend], str]:
        self.requests_handled += 1
        for rule in self._routes:
            rtype = rule["type"]

            if rtype == "path" and req.path.startswith(rule["prefix"]):
                backend = self.backends.get(rule["target"])
                reason  = f"path {rule['prefix']} → {rule['target']}"
                self.routing_log.append(f"{req.path} → {reason}")
                return backend, reason

            elif rtype == "header":
                if req.headers.get(rule["header"]) == rule["value"]:
                    backend = self.backends.get(rule["target"])
                    reason  = f"header {rule['header']}={rule['value']} → {rule['target']}"
                    return backend, reason

            elif rtype == "canary" and req.path.startswith(rule["prefix"]):
                # Route by percentage
                h = int(hashlib.md5(req.src_ip.encode()).hexdigest(), 16) % 100
                if h < rule["pct"]:
                    target  = rule["canary"]
                    reason  = f"canary {rule['pct']}% split → {target}"
                else:
                    target  = rule["stable"]
                    reason  = f"canary {100 - rule['pct']}% → {target}"
                backend = self.backends.get(target)
                return backend, reason

        return None, "no matching route"

    def report(self):
        print(f"\n  L7 LoadBalancer [{self.name}]:")
        print(f"    Requests : {self.requests_handled}")
        for bid, b in self.backends.items():
            print(f"    {bid}: {b.requests} requests")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_l4_vs_l7():
    print("=" * 65)
    print("LAYER 4 VS LAYER 7 LOAD BALANCING")
    print("=" * 65)

    random.seed(42)

    # ── L4 LB ─────────────────────────────────
    print("\n[1] L4 LOAD BALANCER (TCP routing)")
    print("─" * 55)
    l4lb = L4LoadBalancer("10.0.0.1", 443, "tcp")
    l4lb.add_backend(Backend("tcp-1", "10.0.1.1", 8080))
    l4lb.add_backend(Backend("tcp-2", "10.0.1.2", 8080))
    l4lb.add_backend(Backend("tcp-3", "10.0.1.3", 8080))

    print("  Routing TCP connections (only src_ip:port used — no content):")
    packets = [
        TCPPacket("5.5.5.1", 52001, "10.0.0.1", 443),
        TCPPacket("5.5.5.2", 52002, "10.0.0.1", 443),
        TCPPacket("5.5.5.3", 52003, "10.0.0.1", 443),
        TCPPacket("5.5.5.4", 52004, "10.0.0.1", 443),
        TCPPacket("5.5.5.1", 52001, "10.0.0.1", 443),  # same connection → same backend
    ]
    for pkt in packets:
        backend = l4lb.route(pkt)
        print(f"  {pkt.src_ip}:{pkt.src_port} → {backend.backend_id}  "
              f"(routing: IP+port hash only)")
    l4lb.report()

    # ── L7 LB ─────────────────────────────────
    print("\n\n[2] L7 LOAD BALANCER (HTTP content routing)")
    print("─" * 55)
    l7lb = L7LoadBalancer("api-gateway")

    # Add backends
    for bid, host in [("user-svc",    "10.0.2.1"),
                       ("product-svc", "10.0.2.2"),
                       ("order-svc",   "10.0.2.3"),
                       ("order-v2",    "10.0.2.4"),
                       ("admin-svc",   "10.0.2.5")]:
        l7lb.add_backend(Backend(bid, host, 8080))

    # Add routing rules
    l7lb.add_header_route("X-Internal-User", "true", "admin-svc")
    l7lb.add_path_route("/api/v1/users",    "user-svc")
    l7lb.add_path_route("/api/v1/products", "product-svc")
    l7lb.add_canary_route("/api/v1/orders", "order-v2", "order-svc", canary_pct=20)

    print("  Routing rules:")
    print("    Header X-Internal-User=true → admin-svc")
    print("    /api/v1/users               → user-svc")
    print("    /api/v1/products            → product-svc")
    print("    /api/v1/orders              → 20% order-v2 (canary), 80% order-svc")

    requests = [
        HTTPRequest("1.1.1.1", "GET",  "api.example.com", "/api/v1/users/123"),
        HTTPRequest("1.1.1.2", "GET",  "api.example.com", "/api/v1/products"),
        HTTPRequest("1.1.1.3", "POST", "api.example.com", "/api/v1/orders",
                    headers={"Authorization": "Bearer tok"}),
        HTTPRequest("1.1.1.4", "POST", "api.example.com", "/api/v1/orders"),
        HTTPRequest("1.1.1.5", "POST", "api.example.com", "/api/v1/orders"),
        HTTPRequest("1.1.1.6", "GET",  "api.example.com", "/api/v1/users/456",
                    headers={"X-Internal-User": "true"}),
        HTTPRequest("1.1.1.7", "GET",  "api.example.com", "/unknown"),
    ]

    print(f"\n  {'Path':<30} {'Reason':<35} {'Backend'}")
    print(f"  {'─'*75}")
    for req in requests:
        backend, reason = l7lb.route(req)
        backend.handle() if backend else None
        print(f"  {req.path:<30} {reason:<35} {backend.backend_id if backend else '❌ 503'}")

    l7lb.report()

    # ── Comparison ────────────────────────────
    print("\n\n[3] L4 vs L7 COMPARISON")
    print("─" * 55)
    rows = [
        ("OSI layer",           "4 (Transport)",             "7 (Application)"),
        ("Routing info",        "IP + port only",            "URL, headers, cookies, body"),
        ("Latency overhead",    "~0.1ms (minimal)",          "~1-2ms (parses HTTP)"),
        ("Throughput",          "Highest (multi-Gbps)",      "High (but CPU intensive)"),
        ("SSL/TLS",             "Pass-through only",         "Full termination"),
        ("Session persistence", "IP hash / TCP connection",  "Cookie, header, URL param"),
        ("Content routing",     "❌ No",                     "✅ Path, header, method"),
        ("A/B testing",         "❌ No",                     "✅ Yes"),
        ("Health checks",       "TCP connect",               "HTTP GET /health"),
        ("WebSocket",           "✅ Native (TCP)",           "✅ With upgrade header"),
        ("UDP support",         "✅ Native",                 "❌ HTTP only"),
        ("Examples",            "AWS NLB, HAProxy TCP",      "AWS ALB, Nginx, Traefik"),
        ("Use case",            "DNS, gaming, raw TCP",      "HTTP APIs, microservices"),
    ]
    print(f"  {'Aspect':<24} {'L4':<28} {'L7'}")
    print(f"  {'─'*75}")
    for aspect, l4, l7 in rows:
        print(f"  {aspect:<24} {l4:<28} {l7}")


if __name__ == "__main__":
    demonstrate_l4_vs_l7()
