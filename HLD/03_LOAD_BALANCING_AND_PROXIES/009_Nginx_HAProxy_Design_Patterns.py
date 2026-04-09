"""
NGINX AND HAPROXY DESIGN PATTERNS
====================================

Problem Statement:
Nginx and HAProxy are the two most widely deployed reverse proxies and
load balancers. Understanding their design patterns, configuration idioms,
and performance characteristics is essential for production system design.

Nginx:
  - Event-driven, async architecture (handles 10k+ connections per worker)
  - Excellent at serving static files directly (bypasses app servers)
  - Rich L7 features: URL rewriting, caching, SSL termination, streaming
  - Rate limiting, geo-blocking, A/B testing built-in
  - Acts as API Gateway with lua/njs scripting

HAProxy:
  - Purpose-built TCP/HTTP load balancer — faster for pure LB
  - Excellent L4 and L7 support; industry standard for HA setups
  - Rich stats dashboard, hot-reload without dropping connections
  - Advanced health checks: layer 4, layer 7, custom scripts
  - ACL-based routing (Access Control Lists)

Nginx vs HAProxy:
  Nginx:    static files, caching, web server + LB + reverse proxy
  HAProxy:  pure LB performance, L4+L7, TCP proxying, advanced health checks
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import random


class ProxyEngine(Enum):
    NGINX   = "nginx"
    HAPROXY = "haproxy"


class NginxDirective(Enum):
    UPSTREAM       = "upstream"
    SERVER         = "server"
    LOCATION       = "location"
    PROXY_PASS     = "proxy_pass"
    RATE_LIMIT     = "limit_req"
    CACHE          = "proxy_cache"


@dataclass
class UpstreamServer:
    address     : str
    port        : int
    weight      : int = 1
    max_fails   : int = 3
    fail_timeout: int = 30   # seconds
    backup      : bool = False

    def __str__(self):
        parts = [f"server {self.address}:{self.port}"]
        if self.weight != 1:
            parts.append(f"weight={self.weight}")
        if self.max_fails != 3:
            parts.append(f"max_fails={self.max_fails}")
        if self.fail_timeout != 30:
            parts.append(f"fail_timeout={self.fail_timeout}s")
        if self.backup:
            parts.append("backup")
        return " ".join(parts) + ";"


@dataclass
class LocationBlock:
    path        : str
    proxy_pass  : Optional[str] = None
    return_code : Optional[int] = None
    directives  : List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# NGINX CONFIG GENERATOR
# ─────────────────────────────────────────────

class NginxConfigGenerator:
    """Generates Nginx configurations for common patterns."""

    @staticmethod
    def upstream_block(name: str, algorithm: str,
                        servers: List[UpstreamServer]) -> str:
        lines = [f"upstream {name} {{"]
        if algorithm != "round_robin":
            lines.append(f"    {algorithm};")
        lines.append(f"    keepalive 32;")
        for s in servers:
            lines.append(f"    {s}")
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def rate_limit_zone(zone_name: str, key: str,
                         size: str, rate: str) -> str:
        return f"limit_req_zone {key} zone={zone_name}:{size} rate={rate};"

    @staticmethod
    def api_server_block(domain: str, upstream: str,
                          rate_zone: str = None) -> str:
        rate_conf = ""
        if rate_zone:
            rate_conf = f"""
        limit_req zone={rate_zone} burst=20 nodelay;
        limit_req_status 429;"""
        return f"""server {{
    listen 443 ssl http2;
    server_name {domain};

    ssl_certificate     /etc/ssl/certs/{domain}.crt;
    ssl_certificate_key /etc/ssl/private/{domain}.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         ECDHE+AESGCM:ECDHE+CHACHA20:!RC4:!MD5;
    ssl_session_cache   shared:SSL:10m;
    ssl_session_timeout 1d;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options           DENY;
    add_header X-Content-Type-Options    nosniff;

    # Static assets — serve from disk, bypass app server
    location /static/ {{
        root /var/www/{domain};
        expires 1y;
        add_header Cache-Control "public, immutable";
    }}

    # API — proxy to upstream{rate_conf}
    location /api/ {{
        proxy_pass         http://{upstream};
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 30s;
        proxy_connect_timeout 5s;
    }}

    # Health check (internal only)
    location /nginx_status {{
        stub_status;
        allow 10.0.0.0/8;
        deny  all;
    }}

    # Redirect HTTP → HTTPS
    error_page 497 https://$host$request_uri;
}}

server {{
    listen 80;
    server_name {domain};
    return 301 https://$host$request_uri;
}}"""

    @staticmethod
    def cache_config() -> str:
        return """
# Proxy cache zone (in shared memory + disk)
proxy_cache_path /var/cache/nginx
    levels=1:2
    keys_zone=api_cache:10m
    max_size=1g
    inactive=60m
    use_temp_path=off;

location /api/public/ {
    proxy_cache         api_cache;
    proxy_cache_valid   200 60s;
    proxy_cache_valid   404 10s;
    proxy_cache_key     $host$uri$is_args$args;
    proxy_cache_use_stale error timeout updating;
    add_header X-Cache-Status $upstream_cache_status;
    proxy_pass http://backend_pool;
}"""


# ─────────────────────────────────────────────
# HAPROXY CONFIG GENERATOR
# ─────────────────────────────────────────────

class HAProxyConfigGenerator:
    @staticmethod
    def full_config(backends: List[UpstreamServer]) -> str:
        backend_lines = "\n".join(
            f"    server {s.address.replace('.', '-')}-{s.port} "
            f"{s.address}:{s.port} "
            f"check weight {s.weight}"
            + (" backup" if s.backup else "")
            for s in backends
        )
        return f"""global
    maxconn 100000
    log /dev/log local0
    stats socket /run/haproxy/admin.sock mode 660 level admin
    tune.ssl.default-dh-param 2048

defaults
    mode    http
    log     global
    option  httplog
    option  dontlognull
    option  forwardfor
    option  http-server-close
    timeout connect 5s
    timeout client  30s
    timeout server  30s
    retries 3

frontend http-in
    bind *:80
    redirect scheme https code 301 if !{{ ssl_fc }}

frontend https-in
    bind *:443 ssl crt /etc/ssl/certs/bundle.pem
    default_backend app-pool

    # ACL-based routing
    acl is_api   path_beg /api/
    acl is_admin path_beg /admin/
    use_backend  api-pool    if is_api
    use_backend  admin-pool  if is_admin

backend app-pool
    balance     leastconn
    option      httpchk GET /health
    http-check  expect status 200
{backend_lines}

backend api-pool
    balance     roundrobin
    option      httpchk GET /api/health
    http-check  expect status 200
    server api-1 10.0.2.1:8080 check
    server api-2 10.0.2.2:8080 check

backend admin-pool
    balance     first        # all to first; overflow to next
    option      httpchk GET /admin/health
    server adm-1 10.0.3.1:9090 check

listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 10s
    stats auth admin:secret"""


# ─────────────────────────────────────────────
# PERFORMANCE SIMULATOR
# ─────────────────────────────────────────────

class ProxyPerformanceSimulator:
    """Simulates throughput and connection handling."""

    # Benchmarks (approximate, requests/sec on modern hardware)
    THROUGHPUT = {
        ProxyEngine.NGINX:   {"rps_http": 100_000, "rps_https": 25_000,
                               "conc_connections": 10_000,
                               "static_file_mbps": 10_000},
        ProxyEngine.HAPROXY: {"rps_http": 150_000, "rps_https": 30_000,
                               "conc_connections": 50_000,
                               "static_file_mbps": "N/A (not file server)"},
    }

    @classmethod
    def compare(cls):
        print(f"  {'Metric':<30} {'Nginx':<20} {'HAProxy'}")
        print(f"  {'─'*65}")
        metrics = [
            ("HTTP req/sec",     "100K",     "150K"),
            ("HTTPS req/sec",    "25K",      "30K"),
            ("Concurrent conns", "10K/worker","50K"),
            ("Static files",     "Yes (fast)","No"),
            ("Caching",          "Yes",      "No"),
            ("Web server",       "Yes",      "No"),
            ("L4 TCP proxy",     "Basic",    "Excellent"),
            ("Hot reload",       "Yes",      "Yes (no drops)"),
            ("Stats dashboard",  "stub_status","Full dashboard :8404"),
            ("Scripting",        "Lua / njs","Lua"),
            ("Config syntax",    "Declarative","ACL-based"),
        ]
        for metric, nginx, haproxy in metrics:
            print(f"  {metric:<30} {nginx:<20} {haproxy}")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_nginx_haproxy():
    print("=" * 65)
    print("NGINX AND HAPROXY DESIGN PATTERNS")
    print("=" * 65)

    servers = [
        UpstreamServer("10.0.1.1", 8080, weight=2),
        UpstreamServer("10.0.1.2", 8080, weight=2),
        UpstreamServer("10.0.1.3", 8080, weight=1),
        UpstreamServer("10.0.1.4", 8080, backup=True),
    ]

    # ── Nginx Upstream Block ──────────────────
    print("\n[1] NGINX UPSTREAM BLOCK")
    print("─" * 55)
    print(NginxConfigGenerator.upstream_block(
        "backend_pool", "least_conn", servers))

    # ── Rate Limit Zone ───────────────────────
    print("\n\n[2] NGINX RATE LIMITING CONFIG")
    print("─" * 55)
    print(NginxConfigGenerator.rate_limit_zone(
        "api_limit", "$binary_remote_addr", "10m", "10r/s"))
    print(NginxConfigGenerator.rate_limit_zone(
        "user_limit", "$http_x_user_id", "20m", "100r/m"))

    # ── Server Block ──────────────────────────
    print("\n\n[3] NGINX SERVER BLOCK (SSL + reverse proxy)")
    print("─" * 55)
    print(NginxConfigGenerator.api_server_block(
        "api.example.com", "backend_pool", rate_zone="api_limit"))

    # ── Cache Config ──────────────────────────
    print("\n\n[4] NGINX CACHING CONFIG")
    print("─" * 55)
    print(NginxConfigGenerator.cache_config())

    # ── HAProxy Config ────────────────────────
    print("\n\n[5] HAPROXY FULL CONFIG")
    print("─" * 55)
    print(HAProxyConfigGenerator.full_config(servers))

    # ── Performance Comparison ────────────────
    print("\n\n[6] NGINX vs HAPROXY PERFORMANCE")
    print("─" * 55)
    ProxyPerformanceSimulator.compare()

    # ── Decision Guide ────────────────────────
    print("\n\n[7] WHEN TO USE WHICH")
    print("─" * 55)
    guide = [
        ("Serving static files",     "Nginx",    "Built-in efficient file serving"),
        ("Pure TCP load balancing",  "HAProxy",  "Purpose-built, better L4 perf"),
        ("Web server + LB",          "Nginx",    "One tool for both"),
        ("Complex HTTP routing",     "HAProxy",  "Rich ACL system"),
        ("SSL/TLS termination",      "Either",   "Both excellent"),
        ("API caching",              "Nginx",    "proxy_cache built-in"),
        ("Real-time LB stats",       "HAProxy",  "Superior dashboard"),
        ("Kubernetes Ingress",        "Nginx",    "nginx-ingress-controller"),
        ("Zero-downtime deploys",    "HAProxy",  "Hot-reload with socket"),
    ]
    print(f"  {'Use Case':<32} {'Winner':<12} Reason")
    print(f"  {'─'*70}")
    for use_case, winner, reason in guide:
        print(f"  {use_case:<32} {winner:<12} {reason}")


if __name__ == "__main__":
    demonstrate_nginx_haproxy()
