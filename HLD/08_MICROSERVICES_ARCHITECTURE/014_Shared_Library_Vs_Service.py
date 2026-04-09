"""
SHARED LIBRARY VS SHARED SERVICE
====================================

Problem Statement:
In a microservices system, common logic appears in many services:
  - Date/time utilities, currency formatting
  - JWT validation, request signing
  - Retry logic, circuit breaker implementation
  - Audit logging, metrics emission

Two options: package as a shared library (code dependency) or
extract as a shared service (network dependency).
Wrong choice leads to: tight version coupling (library) or
network overhead and single-point-of-failure (service).

SHARED LIBRARY:
  What it is:  A package published to a package registry (PyPI, npm, Maven).
               Services import it as a code dependency.
  Pro:
    - No network hop; pure in-process call.
    - No availability concern (no extra service to deploy).
    - Low latency.
  Con:
    - Every consumer must update to get bug fixes (version drift).
    - All services using v1.0 are broken by a bug in v1.0 — coordinated upgrade required.
    - Language-locked: Python library can't be used by Go service.
  Best for:
    - Utility code (crypto, formatting, validation).
    - Framework components (retry logic, CB implementation).
    - Code that changes rarely.

SHARED SERVICE:
  What it is:  A standalone microservice exposing an API.
               Other services call it via HTTP/gRPC.
  Pro:
    - Language-agnostic: any service can call it.
    - Independent deployment: fix bugs without touching consumers.
    - Centralized logic (e.g., pricing rules, tax calculation).
  Con:
    - Network hop on every call (latency + availability risk).
    - Becomes a critical dependency — must be highly available.
    - Risk of becoming a bottleneck (high fan-in).
  Best for:
    - Business logic shared across languages.
    - Logic that changes frequently and needs independent deployment.
    - Stateful shared functionality (e.g., rate limiter, token store).

Decision Matrix:
  Use LIBRARY if:
    - Logic is stateless and pure utility.
    - Single language ecosystem.
    - Low update frequency.
    - Performance-critical path.
  Use SERVICE if:
    - Logic is stateful or requires persistent store.
    - Shared across language boundaries.
    - Business rules that change frequently.
    - Need independent deployment and rollback.

Semantic Versioning for Libraries:
  MAJOR.MINOR.PATCH
    MAJOR: breaking change (consumers must update call sites).
    MINOR: new features, backwards compatible.
    PATCH: bug fixes, backwards compatible.
  Lockfile (requirements.txt, poetry.lock): pins exact versions.
  Prevents surprise upgrades but requires explicit update PRs.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import time
import re
import threading


# ─────────────────────────────────────────────
# SEMANTIC VERSION
# ─────────────────────────────────────────────

@dataclass
class SemanticVersion:
    major: int
    minor: int
    patch: int

    @staticmethod
    def parse(s: str) -> "SemanticVersion":
        m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", s.strip())
        if not m:
            raise ValueError(f"Invalid semver: {s}")
        return SemanticVersion(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible_with(self, required: "SemanticVersion") -> bool:
        """Semantic compatibility: same major, >= minor.patch."""
        if self.major != required.major:
            return False
        if self.minor < required.minor:
            return False
        if self.minor == required.minor and self.patch < required.patch:
            return False
        return True

    def bump_major(self) -> "SemanticVersion":
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor, self.patch + 1)


# ─────────────────────────────────────────────
# PACKAGE REGISTRY (simulated PyPI)
# ─────────────────────────────────────────────

@dataclass
class LibraryRelease:
    name     : str
    version  : SemanticVersion
    changelog: str
    code     : Callable       # the actual implementation


class PackageRegistry:
    """Simulates a package registry (PyPI, Nexus, Artifactory)."""

    def __init__(self):
        self._packages : Dict[str, List[LibraryRelease]] = {}

    def publish(self, release: LibraryRelease):
        self._packages.setdefault(release.name, []).append(release)
        print(f"  [Registry] Published {release.name}=={release.version}")

    def install(self, name: str, version_constraint: str) -> Optional[LibraryRelease]:
        """Install best matching version. constraint: '>=1.2.0', '==1.3.0', '^1.0.0'."""
        releases = self._packages.get(name, [])
        if not releases:
            return None
        required = SemanticVersion.parse(
            version_constraint.lstrip("><=^~").strip())
        compatible = [r for r in releases
                      if r.version.is_compatible_with(required)]
        if not compatible:
            return None
        return max(compatible, key=lambda r: (r.version.minor, r.version.patch))

    def latest(self, name: str) -> Optional[LibraryRelease]:
        releases = self._packages.get(name, [])
        if not releases:
            return None
        return max(releases, key=lambda r: (r.version.major,
                                             r.version.minor, r.version.patch))

    def all_versions(self, name: str) -> List[str]:
        return [str(r.version) for r in self._packages.get(name, [])]


# ─────────────────────────────────────────────
# SHARED LIBRARY EXAMPLE
# ─────────────────────────────────────────────

def auth_lib_v1_0_0(token: str) -> Dict:
    """v1.0.0: basic validation, no expiry check."""
    if not token or len(token) < 10:
        raise ValueError("Invalid token")
    return {"user_id": "extracted-from-token", "valid": True}


def auth_lib_v1_1_0(token: str) -> Dict:
    """v1.1.0: adds expiry check (backwards compatible)."""
    if not token or len(token) < 10:
        raise ValueError("Invalid token")
    if token.startswith("expired:"):
        raise ValueError("Token expired")
    return {"user_id": "extracted-from-token", "valid": True, "expiry_checked": True}


def auth_lib_v2_0_0(token: str, audience: str = "") -> Dict:
    """v2.0.0: BREAKING — added required audience parameter."""
    if not audience:
        raise ValueError("v2: audience is required (BREAKING CHANGE)")
    return {"user_id": "extracted-from-token", "valid": True,
            "audience": audience, "expiry_checked": True}


# ─────────────────────────────────────────────
# SERVICE CONSUMING A SHARED LIBRARY
# ─────────────────────────────────────────────

class ServiceUsingLibrary:
    """A microservice that depends on a shared library (pinned version)."""

    def __init__(self, name: str, library_version: str, registry: PackageRegistry):
        self.name     = name
        self._release = registry.install("common-auth", library_version)
        if self._release is None:
            raise RuntimeError(f"Cannot install common-auth {library_version}")
        self._version = self._release.version
        self._validate = self._release.code

    def handle_request(self, token: str) -> Dict:
        try:
            claims = self._validate(token)
            return {"service": self.name, "status": "ok", "claims": claims,
                    "lib_version": str(self._version)}
        except Exception as e:
            return {"service": self.name, "status": "error", "error": str(e)}


# ─────────────────────────────────────────────
# SHARED SERVICE EXAMPLE
# ─────────────────────────────────────────────

class SharedAuthService:
    """
    Shared functionality as a SERVICE (not a library).
    All consumers call this via API. One deployment.
    Bug fixes are deployed once; all consumers instantly benefit.
    """

    def __init__(self):
        self._version  = "3.1.0"
        self._call_log : List[Dict] = []
        self._lock     = threading.Lock()

    def validate_token(self, token: str, audience: str,
                       caller: str) -> Dict:
        """Network call from consumer service to this shared service."""
        time.sleep(0.003)   # 3ms network hop
        with self._lock:
            self._call_log.append({"caller": caller, "ts": time.time()})

        if not token:
            return {"valid": False, "error": "no token"}
        if token.startswith("expired:"):
            return {"valid": False, "error": "token expired"}
        return {"valid": True, "user_id": "u-from-token",
                "audience": audience, "service_version": self._version}

    def stats(self) -> Dict:
        with self._lock:
            callers: Dict[str, int] = {}
            for entry in self._call_log:
                callers[entry["caller"]] = callers.get(entry["caller"], 0) + 1
            return {"total_calls": len(self._call_log), "by_caller": callers}


# ─────────────────────────────────────────────
# DECISION MATRIX
# ─────────────────────────────────────────────

@dataclass
class SharedCodeCandidate:
    name             : str
    description      : str
    is_stateless     : bool
    is_single_language: bool
    change_frequency : str    # low / medium / high
    is_performance_critical: bool
    needs_central_store: bool

    def recommendation(self) -> Tuple[str, str]:
        score_library = 0
        score_service = 0

        if self.is_stateless            : score_library += 3
        if self.is_single_language      : score_library += 2
        if self.change_frequency == "low": score_library += 2
        if self.is_performance_critical : score_library += 3
        if not self.needs_central_store : score_library += 2

        if not self.is_stateless        : score_service += 3
        if not self.is_single_language  : score_service += 3
        if self.change_frequency == "high": score_service += 2
        if self.needs_central_store     : score_service += 3
        if self.change_frequency == "medium": score_service += 1

        if score_library >= score_service:
            return "Shared Library", f"library_score={score_library} > service_score={score_service}"
        return "Shared Service", f"service_score={score_service} > library_score={score_library}"


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_shared_library_vs_service():
    print("=" * 65)
    print("SHARED LIBRARY VS SHARED SERVICE")
    print("=" * 65)

    registry = PackageRegistry()

    # ── 1. Publish library versions ───────────────
    print("\n[1] PUBLISHING SHARED LIBRARY TO REGISTRY")
    print("─" * 55)
    registry.publish(LibraryRelease("common-auth", SemanticVersion.parse("1.0.0"),
                                    "Initial release", auth_lib_v1_0_0))
    registry.publish(LibraryRelease("common-auth", SemanticVersion.parse("1.1.0"),
                                    "Added expiry check", auth_lib_v1_1_0))
    registry.publish(LibraryRelease("common-auth", SemanticVersion.parse("2.0.0"),
                                    "BREAKING: audience param required", auth_lib_v2_0_0))
    print(f"  Available versions: {registry.all_versions('common-auth')}")

    # ── 2. Services pinned to different versions ──
    print("\n\n[2] SERVICES PINNED TO DIFFERENT LIBRARY VERSIONS")
    print("─" * 55)
    svc_a = ServiceUsingLibrary("order-service",    ">=1.0.0", registry)  # gets 1.1.0
    svc_b = ServiceUsingLibrary("payment-service",  ">=1.0.0", registry)  # gets 1.1.0
    svc_c = ServiceUsingLibrary("user-service",     ">=1.0.0", registry)  # gets 1.1.0

    print(f"  Services installed versions:")
    for svc in [svc_a, svc_b, svc_c]:
        print(f"    {svc.name}: common-auth=={svc._version}")

    print(f"\n  Token validation with valid token:")
    for svc in [svc_a, svc_b, svc_c]:
        r = svc.handle_request("valid-token-12345678")
        print(f"    {svc.name}: {r['status']}  lib={r['lib_version']}")

    # ── 3. Version drift problem ──────────────────
    print("\n\n[3] VERSION DRIFT — THE SHARED LIBRARY PROBLEM")
    print("─" * 55)
    print("  Bug found in v1.0.0: expiry not checked!")
    print("  Fix shipped in v1.1.0. But some services are still on v1.0.0.")

    # Simulate a service still on old version
    old_svc = ServiceUsingLibrary("legacy-service", ">=1.0.0", registry)
    print(f"\n  legacy-service pinned to: {old_svc._version}")

    # v2.0.0 is a breaking change
    print(f"\n  v2.0.0 has breaking change: audience param required.")
    print(f"  Services on v1.x.x will BREAK if they upgrade to v2.")
    # Test the v2 code directly
    try:
        auth_lib_v2_0_0("valid-token-12345678")
    except ValueError as e:
        print(f"  v2 without audience: {e}")
    result = auth_lib_v2_0_0("valid-token-12345678", audience="api-gateway")
    print(f"  v2 with audience: {result}")
    print(f"  → ALL services must update call sites before upgrading to v2.")

    # ── 4. Shared service approach ────────────────
    print("\n\n[4] SHARED SERVICE APPROACH")
    print("─" * 55)
    auth_service = SharedAuthService()

    consumers = ["order-service", "payment-service", "user-service", "legacy-service"]
    print(f"  All services call the shared auth service (v{auth_service._version}):")
    for svc in consumers:
        result = auth_service.validate_token("valid-token-abc", "internal", caller=svc)
        print(f"    {svc:<22} → valid={result['valid']}  "
              f"svc_version={result.get('service_version')}")

    print(f"\n  Fix deployed to auth-service v3.2.0 — all consumers instantly benefit.")
    print(f"  No library update PRs. No coordinated upgrades.")

    stats = auth_service.stats()
    print(f"\n  Calls received by shared auth service: {stats['total_calls']}")
    print(f"  Per caller: {stats['by_caller']}")
    print(f"  ⚠ Shared service is now a critical dependency (single point of failure).")

    # ── 5. Latency comparison ──────────────────────
    print("\n\n[5] LATENCY: LIBRARY (IN-PROCESS) vs SERVICE (NETWORK)")
    print("─" * 55)
    N = 100

    t0 = time.time()
    for _ in range(N):
        auth_lib_v1_1_0("valid-token-12345678")
    lib_ms = (time.time() - t0) * 1000

    t0 = time.time()
    for _ in range(N):
        auth_service.validate_token("valid-token-abc", "internal", "test")
    svc_ms = (time.time() - t0) * 1000

    print(f"  {N} calls — shared library: {lib_ms:.2f}ms total "
          f"({lib_ms/N:.3f}ms/call)")
    print(f"  {N} calls — shared service: {svc_ms:.2f}ms total "
          f"({svc_ms/N:.2f}ms/call, incl. 3ms simulated latency)")
    print(f"  Library is ~{svc_ms/max(lib_ms,0.001):.0f}x faster per call.")

    # ── 6. Decision matrix ────────────────────────
    print("\n\n[6] DECISION MATRIX — LIBRARY OR SERVICE?")
    print("─" * 55)
    candidates = [
        SharedCodeCandidate("JWT validation",     "Validate JWT tokens",
                            True, True, "low", True, False),
        SharedCodeCandidate("Tax calculation",    "Country-specific tax rules",
                            True, False, "high", False, False),
        SharedCodeCandidate("Rate limiter",       "Distributed rate limiting",
                            False, False, "medium", False, True),
        SharedCodeCandidate("Date formatting",    "Format dates per locale",
                            True, True, "low", True, False),
        SharedCodeCandidate("Pricing engine",     "Dynamic pricing rules",
                            True, False, "high", False, False),
        SharedCodeCandidate("Audit logger",       "Centralized audit trail",
                            False, False, "low", False, True),
    ]
    print(f"  {'Candidate':<22} {'Stateless':<11} {'SingleLang':<11} "
          f"{'Change':<9} {'PerfCrit':<9} {'Recommendation'}")
    print(f"  {'─'*78}")
    for c in candidates:
        rec, reason = c.recommendation()
        print(f"  {c.name:<22} {str(c.is_stateless):<11} "
              f"{str(c.is_single_language):<11} "
              f"{c.change_frequency:<9} {str(c.is_performance_critical):<9} {rec}")

    # ── 7. Semantic versioning guide ──────────────
    print("\n\n[7] SEMANTIC VERSIONING STRATEGY")
    print("─" * 55)
    v = SemanticVersion.parse("1.4.2")
    print(f"  Current: {v}")
    print(f"    Patch release (bug fix):        {v.bump_patch()}  "
          f"(consumers don't need to change code)")
    print(f"    Minor release (new feature):    {v.bump_minor()}  "
          f"(backwards compatible; consumers can upgrade safely)")
    print(f"    Major release (breaking change):{v.bump_major()}  "
          f"(consumers MUST update call sites)")

    print(f"\n  Compatibility check:")
    installed = SemanticVersion.parse("1.5.0")
    requirements = [("1.3.0", True), ("1.6.0", False), ("2.0.0", False)]
    for req_str, expected in requirements:
        req = SemanticVersion.parse(req_str)
        compat = installed.is_compatible_with(req)
        print(f"    installed={installed}  requires>={req_str}  "
              f"compatible={compat}  ({'correct' if compat == expected else 'check'})")


if __name__ == "__main__":
    demonstrate_shared_library_vs_service()
