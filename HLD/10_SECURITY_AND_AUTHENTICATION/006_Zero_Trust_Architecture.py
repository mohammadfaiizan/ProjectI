"""
ZERO TRUST ARCHITECTURE
=========================

Problem Statement:
Traditional perimeter security trusts everything inside the network.
But: employees with laptops, cloud workloads, SaaS apps mean there's
no clear perimeter. Zero Trust assumes breach: verify every request.

Zero Trust Principles (NIST SP 800-207):
  1. Verify explicitly:       Always authenticate and authorize.
                              Use all available data: identity, location, device.
  2. Least privilege:         Grant minimum needed permissions. Just-in-time access.
  3. Assume breach:           Segment networks. Encrypt all traffic.
                              Collect telemetry. Detect anomalies.

vs Traditional Perimeter:
  Perimeter: "castle and moat" — trust inside, distrust outside.
  Zero Trust: no implicit trust based on network location.
              Even internal services must authenticate each other.

Zero Trust Components:
  Policy Engine:     Evaluates access requests. Grants/denies.
  Policy Enforcer:   Proxy/gateway that enforces policy decisions.
  Identity Provider: Source of user/device identity.
  Device Trust:      Is the device managed? Patched? Compliant?
  Network Micro-Segmentation: Isolate workloads. Default deny.
  Service Mesh (mTLS): Encrypt all inter-service traffic.

Access Request Context:
  User identity:  Who? (SSO + MFA).
  Device posture: Is the device managed? Enrolled in MDM?
                  Is antivirus up to date? OS patch level?
  Location:       Expected location? Anomalous travel?
  Behavior:       Normal usage patterns? Time of day?
  Resource:       What's being accessed? Classification level?
  Risk score:     Combined risk from all signals.

Continuous Verification:
  Not just at login — re-verify throughout session.
  Privileged access: step-up auth (MFA again for sensitive ops).
  Short-lived credentials: tokens expire quickly (15 min).
  Just-in-time (JIT) access: approve elevated access on demand.

BeyondCorp (Google's implementation):
  Any employee can work from any network.
  Access granted based on device state + user credentials (not VPN).
  Device inventory database + certificate issuance.
  Access proxy intercepts all traffic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import time
import uuid
import secrets
import hashlib


# ─────────────────────────────────────────────
# DEVICE TRUST
# ─────────────────────────────────────────────

class DeviceComplianceStatus(Enum):
    COMPLIANT     = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    UNKNOWN       = "UNKNOWN"


@dataclass
class DevicePosture:
    device_id          : str
    os_type            : str     # "windows", "macos", "linux", "ios", "android"
    os_version         : str
    is_managed         : bool    # enrolled in MDM/Intune
    is_encrypted       : bool    # disk encrypted
    antivirus_current  : bool    # AV signatures up to date
    os_patch_current   : bool    # OS patches applied
    last_seen          : float   # timestamp of last compliance check
    certificate_hash   : str     # device certificate fingerprint

    def compliance_status(self) -> DeviceComplianceStatus:
        if not self.is_managed:
            return DeviceComplianceStatus.NON_COMPLIANT
        if not all([self.is_encrypted, self.antivirus_current, self.os_patch_current]):
            return DeviceComplianceStatus.NON_COMPLIANT
        if (time.time() - self.last_seen) > 86400:   # not checked in 24h
            return DeviceComplianceStatus.UNKNOWN
        return DeviceComplianceStatus.COMPLIANT

    def trust_score(self) -> int:
        """0-100 device trust score."""
        score = 0
        if self.is_managed       : score += 30
        if self.is_encrypted     : score += 20
        if self.antivirus_current: score += 20
        if self.os_patch_current : score += 20
        if (time.time() - self.last_seen) < 3600:  score += 10  # recent check
        return score


# ─────────────────────────────────────────────
# USER IDENTITY + RISK
# ─────────────────────────────────────────────

@dataclass
class UserIdentity:
    user_id        : str
    email          : str
    roles          : Set[str]
    mfa_verified   : bool
    mfa_type       : str    # "totp", "webauthn", "sms"
    last_mfa_at    : float
    ip_address     : str
    country        : str
    usual_countries: Set[str]
    risk_signals   : List[str] = field(default_factory=list)

    def location_risk(self) -> str:
        if self.country not in self.usual_countries:
            return "HIGH"
        return "LOW"

    def mfa_freshness_ok(self, max_age_s: float = 3600) -> bool:
        return (time.time() - self.last_mfa_at) < max_age_s

    def identity_trust_score(self) -> int:
        score = 50  # base
        if self.mfa_verified:                  score += 20
        if self.mfa_type == "webauthn":        score += 10  # phishing-resistant
        if self.mfa_freshness_ok():            score += 10
        if self.location_risk() == "HIGH":     score -= 30
        if "impossible_travel" in self.risk_signals: score -= 40
        if "known_bad_ip" in self.risk_signals:      score -= 50
        return max(0, min(100, score))


# ─────────────────────────────────────────────
# RESOURCE CLASSIFICATION
# ─────────────────────────────────────────────

class ResourceSensitivity(Enum):
    PUBLIC       = 1
    INTERNAL     = 2
    CONFIDENTIAL = 3
    RESTRICTED   = 4   # PII, financial, health


@dataclass
class Resource:
    resource_id    : str
    name           : str
    sensitivity    : ResourceSensitivity
    required_roles : Set[str]
    min_trust_score: int    # combined device+identity score required
    require_mfa    : bool
    allowed_networks: Set[str]  # "corporate", "vpn", "any"


# ─────────────────────────────────────────────
# ZERO TRUST POLICY ENGINE
# ─────────────────────────────────────────────

@dataclass
class AccessRequest:
    user     : UserIdentity
    device   : DevicePosture
    resource : Resource
    action   : str
    timestamp: float = field(default_factory=time.time)


@dataclass
class PolicyDecision:
    allowed     : bool
    risk_level  : str    # LOW, MEDIUM, HIGH
    trust_score : int
    reasons     : List[str]
    step_up_required: bool = False   # additional MFA needed


class ZeroTrustPolicyEngine:
    """
    Evaluates access requests using all available context signals.
    Core of a Zero Trust system.
    """

    def evaluate(self, request: AccessRequest) -> PolicyDecision:
        reasons = []
        device_score   = request.device.trust_score()
        identity_score = request.user.identity_trust_score()
        combined_score = (device_score + identity_score) // 2

        allowed = True
        step_up = False

        # 1. Role check
        if not request.resource.required_roles.intersection(request.user.roles):
            if "admin" not in request.user.roles:
                allowed = False
                reasons.append("DENY: user role insufficient")

        # 2. MFA requirement
        if request.resource.require_mfa and not request.user.mfa_verified:
            allowed = False
            reasons.append("DENY: MFA required")
        elif request.resource.require_mfa and not request.user.mfa_freshness_ok():
            step_up = True
            reasons.append("STEP_UP: MFA stale")

        # 3. Device compliance
        compliance = request.device.compliance_status()
        if compliance == DeviceComplianceStatus.NON_COMPLIANT:
            if request.resource.sensitivity.value >= ResourceSensitivity.CONFIDENTIAL.value:
                allowed = False
                reasons.append("DENY: device non-compliant for sensitive resource")
        elif compliance == DeviceComplianceStatus.UNKNOWN:
            reasons.append("WARN: device compliance unknown")

        # 4. Trust score check
        if combined_score < request.resource.min_trust_score:
            allowed = False
            reasons.append(f"DENY: trust score {combined_score} < required {request.resource.min_trust_score}")

        # 5. Location anomaly
        if request.user.location_risk() == "HIGH":
            if request.resource.sensitivity.value >= ResourceSensitivity.CONFIDENTIAL.value:
                allowed = False
                reasons.append("DENY: high-risk location for confidential resource")
            else:
                step_up = True
                reasons.append("STEP_UP: unusual location detected")

        # 6. Risk signals
        if "impossible_travel" in request.user.risk_signals:
            allowed = False
            reasons.append("DENY: impossible travel detected")

        if not reasons:
            reasons.append("ALLOW: all checks passed")

        risk_level = "HIGH" if not allowed else \
                     ("MEDIUM" if step_up else "LOW")

        return PolicyDecision(
            allowed=allowed, risk_level=risk_level,
            trust_score=combined_score, reasons=reasons,
            step_up_required=step_up and allowed,
        )


# ─────────────────────────────────────────────
# MICRO-SEGMENTATION
# ─────────────────────────────────────────────

class NetworkSegment:
    """Simulates micro-segment with allow/deny rules (default deny)."""

    def __init__(self, name: str):
        self.name  = name
        self._rules: List[Tuple[str, str, str]] = []  # (src, dst, action)

    def allow(self, src: str, dst: str):
        self._rules.append((src, dst, "ALLOW"))

    def deny(self, src: str, dst: str):
        self._rules.append((src, dst, "DENY"))

    def can_reach(self, src: str, dst: str) -> bool:
        for rule_src, rule_dst, action in self._rules:
            if (rule_src == src or rule_src == "*") and \
               (rule_dst == dst or rule_dst == "*"):
                return action == "ALLOW"
        return False  # default deny


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_zero_trust():
    print("=" * 65)
    print("ZERO TRUST ARCHITECTURE")
    print("=" * 65)

    engine = ZeroTrustPolicyEngine()

    # Define resources
    internal_api = Resource(
        resource_id="api-1", name="Internal API",
        sensitivity=ResourceSensitivity.INTERNAL,
        required_roles={"engineer", "operator"},
        min_trust_score=50, require_mfa=False,
        allowed_networks={"corporate", "vpn", "any"},
    )
    payroll_db = Resource(
        resource_id="db-payroll", name="Payroll Database",
        sensitivity=ResourceSensitivity.RESTRICTED,
        required_roles={"hr-admin", "finance"},
        min_trust_score=80, require_mfa=True,
        allowed_networks={"corporate", "vpn"},
    )

    # ── Scenario 1: Normal Access ──────────────────
    print("\n[1] NORMAL ACCESS — COMPLIANT DEVICE + FRESH MFA")
    print("─" * 55)

    alice_device = DevicePosture(
        device_id="dev-alice", os_type="macos", os_version="14.0",
        is_managed=True, is_encrypted=True, antivirus_current=True,
        os_patch_current=True, last_seen=time.time(),
        certificate_hash=hashlib.sha256(b"alice-device-cert").hexdigest()
    )
    alice_id = UserIdentity(
        user_id="alice", email="alice@corp.com",
        roles={"engineer"}, mfa_verified=True, mfa_type="webauthn",
        last_mfa_at=time.time() - 600,  # 10 min ago
        ip_address="10.0.0.42", country="US",
        usual_countries={"US", "CA"},
    )
    req1 = AccessRequest(alice_id, alice_device, internal_api, "GET")
    d1   = engine.evaluate(req1)
    print(f"  Device trust score:   {alice_device.trust_score()}/100")
    print(f"  Identity trust score: {alice_id.identity_trust_score()}/100")
    print(f"  Combined:             {d1.trust_score}/100")
    print(f"  Decision: {'ALLOW' if d1.allowed else 'DENY'}  risk={d1.risk_level}")
    for r in d1.reasons:
        print(f"    {r}")

    # ── Scenario 2: Non-Compliant Device ──────────
    print("\n\n[2] NON-COMPLIANT DEVICE + CONFIDENTIAL RESOURCE")
    print("─" * 55)

    bad_device = DevicePosture(
        device_id="dev-personal", os_type="windows", os_version="10.0",
        is_managed=False, is_encrypted=False, antivirus_current=False,
        os_patch_current=False, last_seen=time.time() - 3600 * 48,
        certificate_hash=""
    )
    bob_id = UserIdentity(
        user_id="bob", email="bob@corp.com",
        roles={"hr-admin"}, mfa_verified=True, mfa_type="totp",
        last_mfa_at=time.time() - 7200,  # 2h ago (stale)
        ip_address="1.2.3.4", country="US",
        usual_countries={"US"},
    )
    req2 = AccessRequest(bob_id, bad_device, payroll_db, "SELECT")
    d2   = engine.evaluate(req2)
    print(f"  Device trust score:   {bad_device.trust_score()}/100")
    print(f"  Compliance:           {bad_device.compliance_status().value}")
    print(f"  Decision: {'ALLOW' if d2.allowed else 'DENY'}  risk={d2.risk_level}")
    for r in d2.reasons:
        print(f"    {r}")

    # ── Scenario 3: Anomalous Location ────────────
    print("\n\n[3] IMPOSSIBLE TRAVEL / RISK SIGNALS")
    print("─" * 55)

    suspicious = UserIdentity(
        user_id="carol", email="carol@corp.com",
        roles={"engineer"}, mfa_verified=True, mfa_type="totp",
        last_mfa_at=time.time() - 300,
        ip_address="5.6.7.8", country="RU",  # unusual country
        usual_countries={"US"},
        risk_signals=["impossible_travel"],
    )
    req3 = AccessRequest(suspicious, alice_device, internal_api, "GET")
    d3   = engine.evaluate(req3)
    print(f"  Country: {suspicious.country} (usual: {suspicious.usual_countries})")
    print(f"  Risk signals: {suspicious.risk_signals}")
    print(f"  Decision: {'ALLOW' if d3.allowed else 'DENY'}  risk={d3.risk_level}")
    for r in d3.reasons:
        print(f"    {r}")

    # ── Micro-Segmentation ────────────────────────
    print("\n\n[4] NETWORK MICRO-SEGMENTATION (default deny)")
    print("─" * 55)

    segment = NetworkSegment("production")
    segment.allow("api-gateway",   "user-service")
    segment.allow("api-gateway",   "order-service")
    segment.allow("user-service",  "user-db")
    segment.allow("order-service", "order-db")
    # No direct access allowed between services

    checks = [
        ("api-gateway",   "user-service",  True),
        ("api-gateway",   "user-db",       False),  # direct DB access denied
        ("user-service",  "order-db",      False),  # cross-service DB denied
        ("order-service", "order-db",      True),
        ("user-service",  "user-db",       True),
    ]
    for src, dst, expected in checks:
        result = segment.can_reach(src, dst)
        status = "OK" if result == expected else "UNEXPECTED"
        print(f"  {src:<20} → {dst:<16}: {'ALLOW' if result else 'DENY'} [{status}]")

    # ── Zero Trust Principles ─────────────────────
    print("\n\n[5] ZERO TRUST IMPLEMENTATION CHECKLIST")
    print("─" * 55)

    checklist = [
        ("Identity verification", "Every request authenticated via IdP + MFA"),
        ("Device posture",        "MDM compliance check before every access"),
        ("Least privilege",       "Role-based; JIT elevation for sensitive ops"),
        ("Continuous auth",       "Re-verify on behavior change; short token TTL"),
        ("Micro-segmentation",    "Default-deny network rules between services"),
        ("mTLS everywhere",       "Encrypt all east-west traffic in service mesh"),
        ("Full audit trail",      "Log all access decisions with full context"),
        ("Risk-based step-up",    "Anomaly → prompt for additional verification"),
        ("No implicit trust",     "Internal network = untrusted = verify anyway"),
        ("Just-in-time access",   "Approve elevated access per-request, time-limited"),
    ]
    for principle, implementation in checklist:
        print(f"  {principle:<24} {implementation}")


if __name__ == "__main__":
    demonstrate_zero_trust()
