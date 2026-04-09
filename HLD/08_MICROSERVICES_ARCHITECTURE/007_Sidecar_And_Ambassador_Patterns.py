"""
SIDECAR AND AMBASSADOR PATTERNS
==================================

Problem Statement:
Cross-cutting concerns (logging, mTLS, retries, metrics, service discovery)
must not be duplicated in every service's business logic.
Two patterns solve this by externalizing these concerns into a co-deployed proxy.

Sidecar Pattern:
  A secondary container/process deployed alongside the main service container,
  in the same pod/VM. It intercepts ALL inbound traffic to the main service.
  The main service doesn't know the sidecar exists.

  Responsibilities of a Sidecar:
    - Mutual TLS (mTLS): terminate TLS, verify peer certificate.
    - Inbound logging: log every request before it reaches the service.
    - Metrics collection: measure latency, error rates per endpoint.
    - Header injection: add X-Forwarded-For, correlation IDs.
    - Circuit breaking for inbound calls.
    - Health check endpoint exposure.

  Key insight: Business service code has ZERO networking boilerplate.
  All networking concerns live in the sidecar (e.g., Envoy, Linkerd proxy).

Ambassador Pattern:
  A proxy for OUTBOUND calls from the service to external dependencies.
  The service calls "localhost:port" and the ambassador handles the rest.

  Responsibilities of an Ambassador:
    - Retry logic: retry transient failures with backoff.
    - Circuit breaking on outbound calls.
    - Service discovery: resolve service name to current IP.
    - Connection pooling.
    - Timeout enforcement.

  Key insight: Service just calls localhost. Ambassador handles discovery,
  retries, and circuit breaking without service awareness.

Sidecar vs Ambassador:
  Sidecar:    Handles INBOUND traffic (external → this service).
  Ambassador: Handles OUTBOUND traffic (this service → external service).
  In practice, Envoy/Istio acts as both (ingress + egress proxy).

Service Mesh:
  When every service has a sidecar proxy, you have a service mesh.
  Control plane (Istio/Linkerd) pushes config to all sidecars.
  Data plane: all the sidecar proxies handling actual traffic.
  You get: mTLS everywhere, distributed tracing, traffic management —
  without changing any application code.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import time
import uuid
import threading
import random


# ─────────────────────────────────────────────
# REQUEST / RESPONSE TYPES
# ─────────────────────────────────────────────

@dataclass
class Request:
    path          : str
    method        : str
    headers       : Dict[str, str] = field(default_factory=dict)
    body          : Any = None
    request_id    : str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class Response:
    status  : int
    body    : Any
    headers : Dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0


# ─────────────────────────────────────────────
# ACTUAL BUSINESS SERVICE (knows nothing about networking)
# ─────────────────────────────────────────────

class BusinessService:
    """Pure business logic. No auth, no logging, no retry — that's the sidecar's job."""

    def __init__(self, name: str):
        self.name = name

    def handle(self, req: Request) -> Response:
        # Simulate processing
        time.sleep(0.008)
        return Response(
            status = 200,
            body   = {"service": self.name, "result": f"processed {req.path}",
                      "echo_headers": {k: v for k, v in req.headers.items()
                                       if k.startswith("X-")}},
        )


# ─────────────────────────────────────────────
# SIDECAR PROXY (inbound)
# ─────────────────────────────────────────────

@dataclass
class TLSCertificate:
    common_name  : str
    issuer       : str
    valid        : bool


class SidecarProxy:
    """
    Intercepts all inbound requests before they reach the business service.
    Handles: mTLS validation, logging, metrics, header injection.
    The service only sees clean, enriched requests.
    """

    def __init__(self, service: BusinessService):
        self._service      = service
        self._request_log  : List[Dict] = []
        self._metrics      = {"requests": 0, "errors": 0, "total_ms": 0.0}
        self._require_mtls = False

    def enable_mtls(self):
        self._require_mtls = True

    def intercept(self, req: Request,
                  peer_cert: Optional[TLSCertificate] = None) -> Response:
        start = time.time()

        # ── mTLS validation ──────────────────────
        if self._require_mtls:
            if peer_cert is None or not peer_cert.valid:
                self._log(req, 401, start)
                return Response(401, {"error": "mTLS: no valid peer certificate"})

        # ── Inject standard headers ──────────────
        req.headers.setdefault("X-Correlation-Id", str(uuid.uuid4())[:8])
        req.headers["X-Forwarded-Via"] = f"sidecar/{self._service.name}"
        req.headers["X-Request-Start"] = str(time.time())

        # ── Forward to business service ──────────
        try:
            resp = self._service.handle(req)
        except Exception as e:
            self._metrics["errors"] += 1
            self._log(req, 500, start)
            return Response(500, {"error": str(e)})

        elapsed = (time.time() - start) * 1000
        resp.latency_ms = elapsed
        resp.headers["X-Served-By"]      = self._service.name
        resp.headers["X-Correlation-Id"] = req.headers["X-Correlation-Id"]

        # ── Metrics update ───────────────────────
        self._metrics["requests"]  += 1
        self._metrics["total_ms"]  += elapsed
        if resp.status >= 500:
            self._metrics["errors"] += 1

        self._log(req, resp.status, start)
        return resp

    def _log(self, req: Request, status: int, start: float):
        self._request_log.append({
            "req_id" : req.request_id,
            "path"   : req.path,
            "method" : req.method,
            "status" : status,
            "ms"     : round((time.time() - start) * 1000, 2),
            "corr_id": req.headers.get("X-Correlation-Id", "—"),
        })

    def metrics_summary(self) -> Dict:
        reqs = self._metrics["requests"]
        return {
            "total_requests" : reqs,
            "error_count"    : self._metrics["errors"],
            "error_rate"     : f"{self._metrics['errors'] / max(reqs,1):.1%}",
            "avg_latency_ms" : round(self._metrics["total_ms"] / max(reqs, 1), 2),
        }


# ─────────────────────────────────────────────
# AMBASSADOR PROXY (outbound)
# ─────────────────────────────────────────────

class ServiceRegistry:
    """Simulates service discovery (Consul/K8s DNS)."""

    def __init__(self):
        self._registry : Dict[str, str] = {}

    def register(self, name: str, address: str):
        self._registry[name] = address

    def resolve(self, name: str) -> Optional[str]:
        return self._registry.get(name)


class DownstreamService:
    """Simulates a downstream microservice with configurable failure rate."""

    def __init__(self, name: str, fail_rate: float = 0.0, latency_ms: float = 20):
        self.name       = name
        self.fail_rate  = fail_rate
        self.latency_ms = latency_ms
        self.call_count = 0

    def call(self, path: str) -> Dict:
        self.call_count += 1
        time.sleep(self.latency_ms / 1000)
        if random.random() < self.fail_rate:
            raise ConnectionError(f"{self.name} transient failure")
        return {"from": self.name, "path": path, "call_num": self.call_count}


class AmbassadorProxy:
    """
    Proxy for outbound calls from a service to its dependencies.
    The service calls self.ambassador.call("inventory-service", "/reserve")
    and the ambassador handles discovery, retries, and circuit breaking.
    """

    def __init__(self, registry: ServiceRegistry,
                 max_retries: int = 3,
                 retry_delay_ms: float = 50,
                 timeout_s: float = 1.0):
        self.registry       = registry
        self.max_retries    = max_retries
        self.retry_delay_ms = retry_delay_ms
        self.timeout_s      = timeout_s
        self._call_log      : List[Dict] = []
        self._downstream    : Dict[str, DownstreamService] = {}

    def register_downstream(self, name: str, svc: DownstreamService):
        self._downstream[name] = svc
        self.registry.register(name, f"10.0.0.{len(self._downstream)}:8080")

    def call(self, service_name: str, path: str) -> Dict:
        """Service calls this; ambassador handles everything else."""
        address = self.registry.resolve(service_name)
        if address is None:
            raise RuntimeError(f"Service '{service_name}' not found in registry")

        svc      = self._downstream.get(service_name)
        attempts = 0
        last_err = None

        while attempts < self.max_retries:
            attempts += 1
            try:
                result = svc.call(path)
                self._call_log.append({
                    "service"  : service_name,
                    "path"     : path,
                    "attempt"  : attempts,
                    "success"  : True,
                })
                return result
            except ConnectionError as e:
                last_err = e
                self._call_log.append({
                    "service"  : service_name,
                    "path"     : path,
                    "attempt"  : attempts,
                    "success"  : False,
                    "error"    : str(e),
                })
                if attempts < self.max_retries:
                    time.sleep(self.retry_delay_ms / 1000 * attempts)  # backoff

        raise RuntimeError(
            f"Ambassador: '{service_name}' failed after {attempts} attempts: {last_err}")

    def call_stats(self) -> Dict[str, Dict]:
        stats: Dict[str, Dict] = {}
        for entry in self._call_log:
            svc = entry["service"]
            if svc not in stats:
                stats[svc] = {"total": 0, "success": 0, "retries": 0}
            stats[svc]["total"] += 1
            if entry["success"]:
                stats[svc]["success"] += 1
            if entry["attempt"] > 1:
                stats[svc]["retries"] += 1
        return stats


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_sidecar_and_ambassador():
    print("=" * 65)
    print("SIDECAR AND AMBASSADOR PATTERNS")
    print("=" * 65)

    # ── 1. Sidecar: basic interception ───────────
    print("\n[1] SIDECAR PROXY — INBOUND INTERCEPTION")
    print("─" * 55)

    biz_service = BusinessService("order-service")
    sidecar     = SidecarProxy(biz_service)

    req1 = Request("/orders/42", "GET")
    resp1 = sidecar.intercept(req1)
    print(f"  Request:  GET /orders/42  (no headers from client)")
    print(f"  Response: status={resp1.status}  latency={resp1.latency_ms:.1f}ms")
    print(f"  Headers injected by sidecar:")
    for k, v in resp1.headers.items():
        print(f"    {k}: {v}")
    print(f"  Headers sidecar added to request (visible to service):")
    for k, v in req1.headers.items():
        print(f"    {k}: {v}")

    # ── 2. Sidecar: mTLS enforcement ──────────────
    print("\n\n[2] SIDECAR — mTLS VALIDATION")
    print("─" * 55)
    sidecar.enable_mtls()

    resp_no_cert = sidecar.intercept(Request("/orders", "GET"))
    print(f"  No cert:       status={resp_no_cert.status}  "
          f"body={resp_no_cert.body}")

    valid_cert   = TLSCertificate("inventory-service", "internal-ca", valid=True)
    resp_with_cert = sidecar.intercept(Request("/orders", "GET"), valid_cert)
    print(f"  Valid cert:    status={resp_with_cert.status}  "
          f"body={resp_with_cert.body.get('result', '')}")

    invalid_cert = TLSCertificate("unknown-service", "external-ca", valid=False)
    resp_bad_cert= sidecar.intercept(Request("/orders", "GET"), invalid_cert)
    print(f"  Invalid cert:  status={resp_bad_cert.status}  "
          f"body={resp_bad_cert.body}")

    # ── 3. Sidecar: metrics ───────────────────────
    print("\n\n[3] SIDECAR — COLLECTED METRICS")
    print("─" * 55)
    sidecar.enable_mtls()
    cert = TLSCertificate("web-service", "internal-ca", valid=True)
    for i in range(10):
        sidecar.intercept(Request(f"/orders/{i}", "GET"), cert)

    m = sidecar.metrics_summary()
    print(f"  Total requests : {m['total_requests']}")
    print(f"  Error count    : {m['error_count']}")
    print(f"  Error rate     : {m['error_rate']}")
    print(f"  Avg latency    : {m['avg_latency_ms']}ms")
    print(f"  (Collected by sidecar; business service wrote zero metrics code)")

    # ── 4. Ambassador: outbound with retry ────────
    print("\n\n[4] AMBASSADOR PROXY — OUTBOUND WITH RETRY")
    print("─" * 55)
    registry   = ServiceRegistry()
    ambassador = AmbassadorProxy(registry, max_retries=3, retry_delay_ms=10)

    reliable_svc = DownstreamService("payment-service",   fail_rate=0.0, latency_ms=10)
    flaky_svc    = DownstreamService("inventory-service", fail_rate=0.6, latency_ms=10)

    ambassador.register_downstream("payment-service",   reliable_svc)
    ambassador.register_downstream("inventory-service", flaky_svc)

    print("  Calling reliable payment-service (0% failure):")
    result = ambassador.call("payment-service", "/charge")
    print(f"    Result: {result}")

    print(f"\n  Calling flaky inventory-service (60% failure, 3 retries):")
    success = fail_count = 0
    for _ in range(10):
        try:
            ambassador.call("inventory-service", "/reserve")
            success += 1
        except RuntimeError:
            fail_count += 1

    stats = ambassador.call_stats()
    print(f"    10 attempts: success={success} total_failures={fail_count}")
    inv_stats = stats.get("inventory-service", {})
    print(f"    Retry stats: {inv_stats}")
    print(f"    (Service just called ambassador.call(); all retry logic is external)")

    # ── 5. Sidecar access log ─────────────────────
    print("\n\n[5] SIDECAR ACCESS LOG (last 3 entries)")
    print("─" * 55)
    for entry in sidecar._request_log[-3:]:
        print(f"  {entry['method']:<5} {entry['path']:<20} "
              f"status={entry['status']} ms={entry['ms']} "
              f"corr={entry['corr_id']}")

    # ── 6. Comparison table ───────────────────────
    print("\n\n[6] SIDECAR vs AMBASSADOR COMPARISON")
    print("─" * 55)
    rows = [
        ("Direction",      "Inbound (external → service)",      "Outbound (service → dependency)"),
        ("Concerns",       "mTLS, logging, metrics, header inj.","Retry, circuit break, discovery"),
        ("Service sees",   "Clean enriched request",             "Just calls localhost:port"),
        ("Technology",     "Envoy, Linkerd proxy, NGINX",        "Envoy egress, Ambassador (project)"),
        ("Deployed as",    "Co-located container in same pod",   "Co-located container in same pod"),
        ("Service Mesh",   "Sidecar = data plane of mesh",       "Also part of data plane"),
    ]
    print(f"  {'Attribute':<18} {'Sidecar':<38} {'Ambassador'}")
    print(f"  {'─'*75}")
    for attr, sid, amb in rows:
        print(f"  {attr:<18} {sid:<38} {amb}")


if __name__ == "__main__":
    demonstrate_sidecar_and_ambassador()
