"""
SYNTHETIC MONITORING
======================

Problem Statement:
Passive monitoring (logs, metrics) only captures real user traffic.
You won't know a feature is broken if no one uses it at 3 AM.
Synthetic monitoring probes your system continuously using scripted
transactions, detecting issues before users do.

Types of Synthetic Monitoring:
  Uptime / Availability:   Simple HTTP ping every 1-5 min from multiple PoPs.
                           Alert if HTTP ≠ 2xx or latency > threshold.
                           Tools: Pingdom, UptimeRobot, AWS Route53 health checks.

  API Probes:              Multi-step HTTP sequences. Auth → POST → validate response.
                           Validates: status code, response schema, specific fields.
                           Tools: Postman monitors, Runscope, AWS CloudWatch Synthetics.

  Browser/E2E Tests:       Full Chromium headless browser scripted with Playwright.
                           Measures Core Web Vitals: LCP, FID/INP, CLS.
                           Tools: Datadog Synthetics, Grafana k6, Checkly.

  DNS / Certificate Probes: Check DNS resolution latency, TLS cert expiry.
                           Alert 30 days before cert expiry.

  Third-party APIs:        Probe external dependencies (Stripe, Twilio, Sendgrid).
                           Catch upstream degradation before it impacts users.

Key Metrics:
  TTFB:    Time To First Byte — server processing + network.
  FCP:     First Contentful Paint — first content rendered.
  LCP:     Largest Contentful Paint — SLO target: < 2.5s.
  CLS:     Cumulative Layout Shift — SLO target: < 0.1.
  TLS:     Certificate validation time; days until expiry.

Multi-location Probing:
  Run probes from 5-10 geographically distributed PoPs.
  Distinguish: global outage vs regional CDN failure.
  If only Frankfurt probe fails → CDN routing issue, not app bug.

Alerting Design:
  1 failure in 1 location → info (might be transient).
  2 consecutive failures in 1 location → warning.
  1 failure in 3+ locations → critical (global outage).
  Always re-verify before paging: transient blips cause alert fatigue.
"""

from __future__ import annotations

import time
import uuid
import random
import json
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum


# ─────────────────────────────────────────────
# PROBE STATUS
# ─────────────────────────────────────────────

class ProbeStatus(Enum):
    PASS    = "pass"
    FAIL    = "fail"
    TIMEOUT = "timeout"
    ERROR   = "error"


# ─────────────────────────────────────────────
# SIMULATED HTTP CLIENT
# ─────────────────────────────────────────────

@dataclass
class FakeResponse:
    status_code: int
    body:        str
    latency_ms:  float
    headers:     Dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.body)


class SimulatedHTTP:
    """
    Mock HTTP client for synthetic probes.
    Injects failures based on fault_map.
    """

    def __init__(self, fault_map: Optional[Dict[str, Any]] = None):
        self._faults = fault_map or {}
        self._rng    = random.Random(42)

    def get(self, url: str, timeout_s: float = 5.0,
            headers: Optional[Dict] = None) -> FakeResponse:
        return self._request("GET", url, timeout_s)

    def post(self, url: str, body: str, timeout_s: float = 5.0,
             headers: Optional[Dict] = None) -> FakeResponse:
        return self._request("POST", url, timeout_s, body)

    def _request(self, method: str, url: str, timeout_s: float,
                 body: Optional[str] = None) -> FakeResponse:
        fault = self._faults.get(url)
        base_lat = self._rng.uniform(20, 80)

        if fault == "timeout":
            time.sleep(0.01)   # simulate partial wait
            raise TimeoutError(f"Request to {url} timed out")
        if fault == "error":
            raise ConnectionError(f"Connection refused: {url}")
        if fault == "slow":
            base_lat = self._rng.uniform(800, 1500)
        if fault == "500":
            return FakeResponse(500, '{"error":"internal server error"}', base_lat)

        # Simulate sensible responses based on URL pattern
        if "/health" in url:
            return FakeResponse(200, '{"status":"ok"}', base_lat)
        if "/login" in url or "/auth" in url:
            token = hashlib.sha256(b"user:pass").hexdigest()[:32]
            return FakeResponse(200, json.dumps({"token": token, "user_id": 42}), base_lat)
        if "/api/products" in url:
            return FakeResponse(200, json.dumps({
                "items": [{"id": 1, "name": "Widget", "price": 9.99}],
                "total": 1,
            }), base_lat)
        if "/api/checkout" in url:
            return FakeResponse(201, json.dumps({"order_id": "ord-123", "status": "created"}), base_lat)
        return FakeResponse(200, '{"ok":true}', base_lat)


# ─────────────────────────────────────────────
# ASSERTION
# ─────────────────────────────────────────────

@dataclass
class Assertion:
    name:  str
    check: Callable[[FakeResponse], bool]
    desc:  str

    def evaluate(self, resp: FakeResponse) -> Tuple[bool, str]:
        try:
            ok = self.check(resp)
            return ok, ("" if ok else f"FAILED: {self.desc}")
        except Exception as e:
            return False, f"EXCEPTION: {e}"


# ─────────────────────────────────────────────
# PROBE STEP
# ─────────────────────────────────────────────

@dataclass
class ProbeStep:
    name:       str
    method:     str
    url:        str
    body:       Optional[str]      = None
    headers:    Dict[str, str]     = field(default_factory=dict)
    assertions: List[Assertion]    = field(default_factory=list)
    timeout_s:  float              = 5.0
    # Optional: extract values from response to use in subsequent steps
    extracts:   Dict[str, str]     = field(default_factory=dict)  # var → jq-like path


@dataclass
class StepResult:
    step_name:  str
    status:     ProbeStatus
    latency_ms: float
    errors:     List[str]    = field(default_factory=list)
    extracted:  Dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────
# SYNTHETIC PROBE RUNNER
# ─────────────────────────────────────────────

@dataclass
class ProbeResult:
    probe_name:   str
    location:     str
    status:       ProbeStatus
    total_ms:     float
    step_results: List[StepResult]
    timestamp:    float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return self.status == ProbeStatus.PASS


class SyntheticProbe:
    """
    Multi-step API probe.
    Each step can extract values (tokens, IDs) for use in subsequent steps.
    """

    def __init__(self, name: str, steps: List[ProbeStep],
                 location: str = "us-east-1"):
        self.name     = name
        self.steps    = steps
        self.location = location

    def run(self, http: SimulatedHTTP) -> ProbeResult:
        start = time.time()
        step_results: List[StepResult] = []
        variables: Dict[str, str] = {}
        overall = ProbeStatus.PASS

        for step in self.steps:
            # Substitute variables in URL and body
            url  = step.url.format(**variables)
            body = (step.body.format(**variables) if step.body else None)
            hdrs = {k: v.format(**variables) for k, v in step.headers.items()}

            t0 = time.time()
            errors: List[str] = []
            extracted: Dict[str, str] = {}
            status = ProbeStatus.PASS

            try:
                if step.method == "GET":
                    resp = http.get(url, step.timeout_s, hdrs)
                else:
                    resp = http.post(url, body or "", step.timeout_s, hdrs)

                # Check assertions
                for assertion in step.assertions:
                    ok, err = assertion.evaluate(resp)
                    if not ok:
                        errors.append(err)
                        status = ProbeStatus.FAIL

                # Extract variables
                if resp.status_code < 400:
                    try:
                        data = resp.json()
                        for var_name, path in step.extracts.items():
                            # Simple dot-path extraction: "token" → data["token"]
                            parts = path.split(".")
                            val   = data
                            for part in parts:
                                if isinstance(val, dict):
                                    val = val.get(part, "")
                            extracted[var_name] = str(val)
                            variables[var_name] = str(val)
                    except Exception:
                        pass

            except TimeoutError as e:
                status = ProbeStatus.TIMEOUT
                errors.append(str(e))
            except Exception as e:
                status = ProbeStatus.ERROR
                errors.append(str(e))

            lat = (time.time() - t0) * 1000
            sr  = StepResult(step.name, status, lat, errors, extracted)
            step_results.append(sr)

            if status != ProbeStatus.PASS:
                overall = status
                break   # stop on first failure

        total_ms = (time.time() - start) * 1000
        return ProbeResult(self.name, self.location, overall, total_ms, step_results)


# ─────────────────────────────────────────────
# UPTIME MONITOR
# ─────────────────────────────────────────────

@dataclass
class UptimeRecord:
    timestamp:   float
    location:    str
    latency_ms:  float
    status:      ProbeStatus
    status_code: Optional[int]


class UptimeMonitor:
    """Simple ping-based uptime monitor across multiple locations."""

    def __init__(self, url: str, locations: List[str]):
        self._url       = url
        self._locations = locations
        self._records:  List[UptimeRecord] = []

    def check_all(self, http: SimulatedHTTP) -> List[UptimeRecord]:
        results = []
        for loc in self._locations:
            t0 = time.time()
            try:
                resp = http.get(self._url)
                lat  = (time.time() - t0) * 1000
                status = ProbeStatus.PASS if resp.status_code < 400 else ProbeStatus.FAIL
                r = UptimeRecord(t0, loc, lat, status, resp.status_code)
            except TimeoutError:
                r = UptimeRecord(t0, loc, 5000, ProbeStatus.TIMEOUT, None)
            except Exception:
                r = UptimeRecord(t0, loc, 0, ProbeStatus.ERROR, None)
            self._records.append(r)
            results.append(r)
        return results

    def availability(self, window_checks: int = 100) -> float:
        recent = self._records[-window_checks:]
        if not recent: return 1.0
        passed = sum(1 for r in recent if r.status == ProbeStatus.PASS)
        return passed / len(recent)

    def avg_latency_ms(self, window_checks: int = 100) -> float:
        recent = [r for r in self._records[-window_checks:]
                  if r.status == ProbeStatus.PASS]
        if not recent: return 0.0
        return sum(r.latency_ms for r in recent) / len(recent)


# ─────────────────────────────────────────────
# TLS CERTIFICATE CHECKER
# ─────────────────────────────────────────────

@dataclass
class CertStatus:
    domain:      str
    days_until_expiry: int
    is_valid:    bool
    issuer:      str

    @property
    def needs_renewal(self) -> bool:
        return self.days_until_expiry < 30

    @property
    def is_critical(self) -> bool:
        return self.days_until_expiry < 7


class TLSProbe:
    """Simulates TLS certificate expiry checking."""

    def __init__(self, certs: Dict[str, int]):
        """certs: {domain: days_remaining}"""
        self._certs = certs

    def check(self, domain: str) -> CertStatus:
        days = self._certs.get(domain, 365)
        return CertStatus(
            domain            = domain,
            days_until_expiry = days,
            is_valid          = days > 0,
            issuer            = "Let's Encrypt" if days < 90 else "DigiCert",
        )


# ─────────────────────────────────────────────
# MULTI-LOCATION ALERT EVALUATOR
# ─────────────────────────────────────────────

@dataclass
class SyntheticAlert:
    severity:    str
    probe_name:  str
    message:     str
    locations:   List[str]


class MultiLocationAlerter:
    """
    Alert only when failures occur in multiple locations
    to avoid false alerts from transient regional issues.
    """

    def __init__(self, global_threshold: int = 3):
        self._threshold = global_threshold

    def evaluate(self, results: List[ProbeResult]) -> Optional[SyntheticAlert]:
        failed = [r for r in results if not r.passed]
        passed = [r for r in results if r.passed]

        if not failed:
            return None

        # All locations fail → critical global outage
        if len(failed) == len(results):
            return SyntheticAlert(
                "critical",
                results[0].probe_name,
                f"Global outage: all {len(results)} locations failing",
                [r.location for r in failed],
            )

        # Multiple (≥ threshold) locations fail → critical
        if len(failed) >= self._threshold:
            return SyntheticAlert(
                "critical",
                results[0].probe_name,
                f"{len(failed)}/{len(results)} locations failing",
                [r.location for r in failed],
            )

        # 1-2 locations fail → warning
        return SyntheticAlert(
            "warning",
            results[0].probe_name,
            f"{len(failed)} location(s) failing: {[r.location for r in failed]}",
            [r.location for r in failed],
        )


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_synthetic():
    print("=" * 65)
    print("SYNTHETIC MONITORING")
    print("=" * 65)

    # ── Multi-step API Probe ──────────────────
    print("\n[1] MULTI-STEP API PROBE (E-commerce checkout flow)")
    print("─" * 55)

    checkout_probe = SyntheticProbe("checkout_flow", steps=[
        ProbeStep(
            name="health_check",
            method="GET",
            url="https://api.example.com/health",
            assertions=[
                Assertion("status_200",
                          lambda r: r.status_code == 200, "Expected HTTP 200"),
                Assertion("latency_ok",
                          lambda r: r.latency_ms < 200, "Latency > 200ms"),
            ],
        ),
        ProbeStep(
            name="authenticate",
            method="POST",
            url="https://api.example.com/auth/login",
            body='{"username":"synthetic_user","password":"test_pass"}',
            assertions=[
                Assertion("status_200", lambda r: r.status_code == 200, "Auth failed"),
                Assertion("has_token",
                          lambda r: "token" in r.json(), "No token in response"),
            ],
            extracts={"token": "token"},
        ),
        ProbeStep(
            name="browse_products",
            method="GET",
            url="https://api.example.com/api/products",
            headers={"Authorization": "Bearer {token}"},
            assertions=[
                Assertion("status_200", lambda r: r.status_code == 200, "Products fail"),
                Assertion("has_items",
                          lambda r: len(r.json().get("items", [])) > 0, "No products"),
            ],
        ),
        ProbeStep(
            name="checkout",
            method="POST",
            url="https://api.example.com/api/checkout",
            body='{"product_id":1,"qty":1}',
            headers={"Authorization": "Bearer {token}"},
            assertions=[
                Assertion("status_201", lambda r: r.status_code == 201, "Checkout failed"),
                Assertion("has_order_id",
                          lambda r: "order_id" in r.json(), "No order_id"),
            ],
            extracts={"order_id": "order_id"},
        ),
    ])

    http = SimulatedHTTP()
    result = checkout_probe.run(http)

    print(f"  Probe: {result.probe_name}  |  Status: {result.status.value.upper()}")
    print(f"  Total time: {result.total_ms:.1f}ms")
    for sr in result.step_results:
        icon = "✓" if sr.status == ProbeStatus.PASS else "✗"
        print(f"    [{icon}] {sr.step_name:<20} {sr.latency_ms:>6.1f}ms  "
              f"{sr.status.value}")
        if sr.extracted:
            print(f"         extracted: {sr.extracted}")
        for err in sr.errors:
            print(f"         ERR: {err}")

    # ── Probe with failure injection ──────────
    print("\n[2] PROBE WITH FAILURE INJECTION")
    print("─" * 55)

    # Inject a 500 on checkout endpoint
    http_faulty = SimulatedHTTP(fault_map={
        "https://api.example.com/api/checkout": "500"
    })
    result_faulty = checkout_probe.run(http_faulty)

    print(f"  Probe status: {result_faulty.status.value.upper()}")
    for sr in result_faulty.step_results:
        icon = "✓" if sr.status == ProbeStatus.PASS else "✗"
        print(f"    [{icon}] {sr.step_name:<20} {sr.latency_ms:>6.1f}ms  "
              f"{sr.status.value}")
        for err in sr.errors:
            print(f"         ERR: {err}")

    # ── Uptime Monitor ────────────────────────
    print("\n[3] UPTIME MONITOR (multi-location)")
    print("─" * 55)

    locations = ["us-east-1", "eu-west-1", "ap-south-1", "us-west-2"]
    monitor   = UptimeMonitor("https://api.example.com/health", locations)

    # Simulate 20 check rounds
    random.seed(7)
    for round_n in range(20):
        # Simulate occasional failure in one region
        faults = {}
        if round_n in (5, 6):
            faults["https://api.example.com/health"] = "slow"

        http_r = SimulatedHTTP(faults)
        records = monitor.check_all(http_r)

    avail  = monitor.availability()
    avg_lat = monitor.avg_latency_ms()
    print(f"  URL: https://api.example.com/health")
    print(f"  20 check rounds × 4 locations = {len(monitor._records)} checks")
    print(f"  Availability: {avail*100:.2f}%")
    print(f"  Avg latency:  {avg_lat:.1f}ms")

    # ── Multi-location Alert Evaluation ───────
    print("\n[4] MULTI-LOCATION ALERT EVALUATION")
    print("─" * 55)

    alerter = MultiLocationAlerter(global_threshold=3)

    scenarios = [
        ("All healthy",
         [ProbeResult("api", loc, ProbeStatus.PASS, 50, []) for loc in locations]),
        ("1 location fails (EU)",
         [ProbeResult("api", loc,
                      ProbeStatus.FAIL if loc == "eu-west-1" else ProbeStatus.PASS,
                      50, []) for loc in locations]),
        ("3 locations fail (global outage)",
         [ProbeResult("api", loc,
                      ProbeStatus.FAIL if loc != "ap-south-1" else ProbeStatus.PASS,
                      50, []) for loc in locations]),
        ("All fail",
         [ProbeResult("api", loc, ProbeStatus.FAIL, 50, []) for loc in locations]),
    ]

    for label, results in scenarios:
        alert = alerter.evaluate(results)
        if alert:
            print(f"  [{label}] → [{alert.severity.upper()}] {alert.message}")
        else:
            print(f"  [{label}] → OK (no alert)")

    # ── TLS Certificate Checks ────────────────
    print("\n[5] TLS CERTIFICATE EXPIRY CHECKS")
    print("─" * 55)

    tls = TLSProbe({
        "api.example.com":    365,
        "cdn.example.com":    28,      # needs renewal
        "legacy.example.com": 4,       # critical!
        "admin.example.com":  90,
    })

    domains = ["api.example.com", "cdn.example.com", "legacy.example.com", "admin.example.com"]
    print(f"  {'Domain':<25} {'Days Left':>10} {'Issuer':<15}  {'Status'}")
    print("  " + "─" * 65)
    for domain in domains:
        cert = tls.check(domain)
        if cert.is_critical:
            status = "CRITICAL — renew now!"
        elif cert.needs_renewal:
            status = "WARNING — renew soon"
        else:
            status = "OK"
        print(f"  {cert.domain:<25} {cert.days_until_expiry:>10}d "
              f"{cert.issuer:<15}  {status}")

    # ── Probe Design Guidelines ───────────────
    print("\n[6] SYNTHETIC MONITORING BEST PRACTICES")
    print("─" * 55)

    practices = [
        "Run from 5+ geographically distributed locations",
        "Use dedicated synthetic user accounts (not real users)",
        "Run critical flows every 1 min; secondary flows every 5-15 min",
        "Alert only on consecutive failures (2 failures = warn, 3 = critical)",
        "Alert globally only when 3+ locations fail (avoids regional noise)",
        "Include TLS cert check with 30-day expiry alert",
        "Validate response schema, not just status code",
        "Measure all steps: auth latency, search latency, checkout latency",
        "Store probe results as metrics for SLO error budget calculation",
    ]
    for i, p in enumerate(practices, 1):
        print(f"  {i}. {p}")


if __name__ == "__main__":
    demonstrate_synthetic()
