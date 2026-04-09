"""
HTTP / HTTPS FUNDAMENTALS
==========================

Problem Statement:
HTTP is the foundation of web communication. Engineers must understand the
differences between HTTP/1.0, HTTP/1.1, HTTP/2, and HTTP/3 to make informed
decisions about performance, connection management, and protocol selection.

HTTP Evolution:
  HTTP/1.0 → New TCP connection per request
  HTTP/1.1 → Keep-alive: reuse TCP; but head-of-line blocking
  HTTP/2   → Multiplexing: many requests on one TCP; header compression (HPACK)
  HTTP/3   → QUIC (UDP-based): eliminates TCP head-of-line blocking

HTTPS:
  TLS handshake: ClientHello → ServerHello → Certificate → Key Exchange → Finished
  TLS 1.3 reduces handshake to 1 round-trip (was 2 in TLS 1.2)

Key Headers:
  Cache-Control, Content-Type, Authorization, Connection, Transfer-Encoding
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import time
import hashlib


class HTTPVersion(Enum):
    HTTP_1_0 = "HTTP/1.0"
    HTTP_1_1 = "HTTP/1.1"
    HTTP_2   = "HTTP/2"
    HTTP_3   = "HTTP/3"


class HTTPMethod(Enum):
    GET     = "GET"
    POST    = "POST"
    PUT     = "PUT"
    DELETE  = "DELETE"
    PATCH   = "PATCH"
    HEAD    = "HEAD"
    OPTIONS = "OPTIONS"


class StatusCode(Enum):
    OK                    = 200
    CREATED               = 201
    NO_CONTENT            = 204
    NOT_MODIFIED          = 304
    BAD_REQUEST           = 400
    UNAUTHORIZED          = 401
    FORBIDDEN             = 403
    NOT_FOUND             = 404
    METHOD_NOT_ALLOWED    = 405
    TOO_MANY_REQUESTS     = 429
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY           = 502
    SERVICE_UNAVAILABLE   = 503


@dataclass
class HTTPHeader:
    name : str
    value: str


@dataclass
class HTTPRequest:
    method   : HTTPMethod
    url      : str
    version  : HTTPVersion = HTTPVersion.HTTP_1_1
    headers  : Dict[str, str] = field(default_factory=dict)
    body     : Optional[str]  = None

    def add_header(self, name: str, value: str):
        self.headers[name] = value
        return self

    def display(self):
        print(f"  {self.method.value} {self.url} {self.version.value}")
        for k, v in self.headers.items():
            print(f"  {k}: {v}")
        if self.body:
            print(f"\n  {self.body}")


@dataclass
class HTTPResponse:
    status   : StatusCode
    version  : HTTPVersion = HTTPVersion.HTTP_1_1
    headers  : Dict[str, str] = field(default_factory=dict)
    body     : str = ""
    latency_ms: float = 0.0

    def display(self):
        print(f"  {self.version.value} {self.status.value} {self.status.name}")
        for k, v in self.headers.items():
            print(f"  {k}: {v}")
        if self.body:
            print(f"\n  {self.body[:120]}")


# ─────────────────────────────────────────────
# TLS HANDSHAKE SIMULATION
# ─────────────────────────────────────────────

class TLSHandshake:
    """Simulates TLS 1.3 handshake steps."""

    def __init__(self, hostname: str):
        self.hostname  = hostname
        self.session_key: Optional[str] = None
        self._steps: List[str] = []

    def _log(self, msg: str):
        self._steps.append(msg)
        print(f"  TLS: {msg}")

    def perform(self) -> bool:
        self._log(f"[1] ClientHello → {self.hostname} (TLS 1.3, cipher suites, random nonce)")
        time.sleep(0.01)
        self._log(f"[2] ServerHello ← (chosen cipher: TLS_AES_256_GCM_SHA384, server random)")
        time.sleep(0.01)
        self._log(f"[3] Certificate ← (server cert: CN={self.hostname}, CA=DigiCert)")
        time.sleep(0.005)
        self._log(f"[4] Client verifies cert chain (checks CA signature)")
        self._log(f"[5] Key Exchange: ECDHE (client + server compute shared secret)")
        self.session_key = hashlib.sha256(f"{self.hostname}-session".encode()).hexdigest()[:16]
        self._log(f"[6] Finished: both sides derive session keys (symmetric AES)")
        self._log(f"[7] ✅ Handshake complete — encrypted channel established")
        return True


# ─────────────────────────────────────────────
# CONNECTION MANAGER
# ─────────────────────────────────────────────

class ConnectionManager:
    """Simulates connection handling for different HTTP versions."""

    def __init__(self, version: HTTPVersion):
        self.version      = version
        self.connections  = 0
        self.requests     = 0
        self.total_latency= 0.0
        self._active_conns= []

    def _tcp_handshake_ms(self) -> float:
        return 30.0   # simulated RTT for TCP SYN-SYNACK-ACK

    def _tls_handshake_ms(self) -> float:
        return 15.0   # TLS 1.3: 1 RTT = ~15ms

    def _process_ms(self) -> float:
        return 5.0    # simulated server processing

    def send_request(self, request: HTTPRequest) -> HTTPResponse:
        self.requests += 1
        latency = self._process_ms()

        if self.version == HTTPVersion.HTTP_1_0:
            # New TCP connection per request
            self.connections += 1
            latency += self._tcp_handshake_ms()
            if request.url.startswith("https"):
                latency += self._tls_handshake_ms()

        elif self.version == HTTPVersion.HTTP_1_1:
            # Keep-alive: reuse TCP after first connection
            if not self._active_conns:
                self.connections += 1
                latency += self._tcp_handshake_ms()
                if request.url.startswith("https"):
                    latency += self._tls_handshake_ms()
                self._active_conns.append("conn-1")

        elif self.version in (HTTPVersion.HTTP_2, HTTPVersion.HTTP_3):
            # Single connection, multiplexed
            if not self._active_conns:
                self.connections += 1
                latency += self._tcp_handshake_ms()
                if request.url.startswith("https"):
                    latency += self._tls_handshake_ms()
                self._active_conns.append("conn-1")
            # HTTP/2: headers compressed (30% smaller), multiplexed (no HoL blocking)

        self.total_latency += latency
        return HTTPResponse(
            status=StatusCode.OK,
            version=self.version,
            headers={"Content-Type": "application/json", "Connection": "keep-alive"},
            body='{"status": "ok"}',
            latency_ms=round(latency, 1)
        )

    def report(self, n_requests: int):
        avg = self.total_latency / n_requests if n_requests else 0
        print(f"\n  [{self.version.value}] {n_requests} requests:")
        print(f"    TCP connections opened : {self.connections}")
        print(f"    Total latency          : {self.total_latency:.0f} ms")
        print(f"    Avg latency/request    : {avg:.1f} ms")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_http_https_fundamentals():
    print("=" * 65)
    print("HTTP / HTTPS FUNDAMENTALS")
    print("=" * 65)

    # ── Request / Response ────────────────────
    print("\n[1] HTTP REQUEST / RESPONSE STRUCTURE")
    print("─" * 50)
    req = HTTPRequest(HTTPMethod.POST, "https://api.example.com/users")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer eyJhbGci...")
    req.add_header("Accept", "application/json")
    req.body = '{"name": "Alice", "email": "alice@example.com"}'
    print("  REQUEST:")
    req.display()

    resp = HTTPResponse(
        StatusCode.CREATED,
        headers={"Content-Type": "application/json", "Location": "/users/u123",
                 "Cache-Control": "no-store"},
        body='{"user_id": "u123", "name": "Alice"}'
    )
    print("\n  RESPONSE:")
    resp.display()

    # ── TLS Handshake ─────────────────────────
    print("\n\n[2] TLS 1.3 HANDSHAKE")
    print("─" * 50)
    tls = TLSHandshake("api.example.com")
    tls.perform()

    # ── HTTP Version Comparison ───────────────
    print("\n\n[3] HTTP VERSION COMPARISON (10 requests per version)")
    print("─" * 50)
    url = "https://api.example.com/data"
    for version in [HTTPVersion.HTTP_1_0, HTTPVersion.HTTP_1_1, HTTPVersion.HTTP_2]:
        mgr = ConnectionManager(version)
        for i in range(10):
            r = HTTPRequest(HTTPMethod.GET, url, version)
            mgr.send_request(r)
        mgr.report(10)

    # ── Common Headers Guide ──────────────────
    print("\n\n[4] IMPORTANT HTTP HEADERS")
    print("─" * 50)
    headers = [
        ("Cache-Control",      "max-age=3600",      "Cache for 1 hour"),
        ("Cache-Control",      "no-cache",          "Must revalidate with server"),
        ("Cache-Control",      "no-store",          "Never cache (e.g., bank data)"),
        ("Authorization",      "Bearer <jwt>",      "Auth token"),
        ("Content-Type",       "application/json",  "Body format"),
        ("Accept",             "application/json",  "Expected response format"),
        ("Connection",         "keep-alive",        "Reuse TCP connection"),
        ("Transfer-Encoding",  "chunked",           "Stream response in chunks"),
        ("ETag",               '"abc123"',          "Resource version for conditional GET"),
        ("If-None-Match",      '"abc123"',          "Return 304 if unchanged"),
        ("X-RateLimit-Remaining","99",              "Rate limit header"),
        ("Retry-After",        "60",                "Wait 60s before retry (429 response)"),
    ]
    print(f"  {'Header':<25} {'Example Value':<25} {'Purpose'}")
    print(f"  {'─'*75}")
    for name, value, purpose in headers:
        print(f"  {name:<25} {value:<25} {purpose}")

    # ── Status Codes Guide ────────────────────
    print("\n\n[5] KEY STATUS CODES")
    print("─" * 50)
    codes = [
        (200, "OK",                   "Standard success"),
        (201, "Created",              "Resource created (POST)"),
        (204, "No Content",           "Success, no body (DELETE)"),
        (304, "Not Modified",         "Cache hit — use cached version"),
        (400, "Bad Request",          "Client error — invalid input"),
        (401, "Unauthorized",         "Missing/invalid auth token"),
        (403, "Forbidden",            "Authenticated but no permission"),
        (404, "Not Found",            "Resource does not exist"),
        (429, "Too Many Requests",    "Rate limited — check Retry-After"),
        (500, "Internal Server Error","Server bug — log and alert"),
        (502, "Bad Gateway",          "Upstream service returned error"),
        (503, "Service Unavailable",  "Server overloaded or down"),
    ]
    for code, name, meaning in codes:
        print(f"  {code} {name:<25} {meaning}")


if __name__ == "__main__":
    demonstrate_http_https_fundamentals()
