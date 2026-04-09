"""
NETWORK SECURITY FUNDAMENTALS
================================

Problem Statement:
Every public-facing system is constantly under attack — DDoS, injection,
man-in-the-middle, credential stuffing, and more. Engineers must understand
the network security layers and design systems that are secure by default.

Layers of Network Security:
  1. TLS/mTLS      → encrypt data in transit
  2. Firewalls      → restrict inbound/outbound traffic by port/IP
  3. VPC/Subnets    → network segmentation isolates blast radius
  4. WAF            → block L7 attacks (SQLi, XSS, LFI)
  5. DDoS Protection→ rate limiting, IP reputation, anycast scrubbing
  6. Zero Trust     → never trust the network; verify every request

OWASP Top 10 Relevant to APIs:
  A01 - Broken Access Control
  A02 - Cryptographic Failures (weak TLS, unencrypted data)
  A03 - Injection (SQLi, command injection)
  A05 - Security Misconfiguration
  A07 - Identification and Authentication Failures
  A09 - Security Logging and Monitoring Failures
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import re
import hashlib
import time
import ipaddress


class ThreatType(Enum):
    SQL_INJECTION   = "sql_injection"
    XSS             = "xss"
    COMMAND_INJ     = "command_injection"
    PATH_TRAVERSAL  = "path_traversal"
    DDOS            = "ddos"
    BRUTE_FORCE     = "brute_force"
    CSRF            = "csrf"
    MITM            = "mitm"


class FirewallAction(Enum):
    ALLOW = "allow"
    DENY  = "deny"
    LOG   = "log"


@dataclass
class NetworkRequest:
    src_ip   : str
    dst_port : int
    protocol : str = "tcp"
    payload  : str = ""
    method   : str = "GET"
    path     : str = "/"
    headers  : Dict[str, str] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    threat_type : ThreatType
    src_ip      : str
    payload     : str
    severity    : str   # low / medium / high / critical
    blocked     : bool  = True
    timestamp   : float = field(default_factory=time.time)


# ─────────────────────────────────────────────
# TLS INSPECTOR
# ─────────────────────────────────────────────

class TLSInspector:
    """Reports TLS configuration quality."""

    WEAK_CIPHERS = {
        "RC4", "DES", "3DES", "NULL", "EXPORT", "MD5", "SHA1"
    }
    STRONG_CIPHERS = {
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "TLS_AES_128_GCM_SHA256",
        "ECDHE-RSA-AES256-GCM-SHA384",
    }

    def audit(self, tls_version: str, cipher: str, cert_expiry_days: int) -> Dict:
        issues = []
        if tls_version in ("TLSv1.0", "TLSv1.1", "SSLv3"):
            issues.append(f"Weak TLS version: {tls_version} — upgrade to TLS 1.2+")
        if any(weak in cipher for weak in self.WEAK_CIPHERS):
            issues.append(f"Weak cipher suite: {cipher}")
        if cert_expiry_days < 30:
            issues.append(f"Certificate expires in {cert_expiry_days} days — renew now!")
        grade = "A" if not issues else ("B" if len(issues) == 1 else "F")
        return {"tls_version": tls_version, "cipher": cipher,
                "cert_expiry_days": cert_expiry_days,
                "grade": grade, "issues": issues}


# ─────────────────────────────────────────────
# FIREWALL
# ─────────────────────────────────────────────

@dataclass
class FirewallRule:
    rule_id     : str
    src_cidr    : str       # "0.0.0.0/0" = any
    dst_port    : int       # -1 = any
    protocol    : str       # tcp/udp/any
    action      : FirewallAction
    description : str = ""

    def matches(self, req: NetworkRequest) -> bool:
        try:
            network = ipaddress.ip_network(self.src_cidr, strict=False)
            ip_match = ipaddress.ip_address(req.src_ip) in network
        except ValueError:
            ip_match = (self.src_cidr == "0.0.0.0/0")
        port_match = (self.dst_port == -1 or self.dst_port == req.dst_port)
        proto_match = (self.protocol == "any" or self.protocol == req.protocol)
        return ip_match and port_match and proto_match


class Firewall:
    """Stateless packet filter — rules evaluated in order (first match wins)."""

    def __init__(self, name: str):
        self.name   = name
        self.rules  : List[FirewallRule] = []
        self.allowed = 0
        self.denied  = 0

    def add_rule(self, rule: FirewallRule):
        self.rules.append(rule)

    def evaluate(self, req: NetworkRequest) -> FirewallAction:
        for rule in self.rules:
            if rule.matches(req):
                if rule.action == FirewallAction.ALLOW:
                    self.allowed += 1
                else:
                    self.denied += 1
                return rule.action
        # Default deny
        self.denied += 1
        return FirewallAction.DENY

    def show_rules(self):
        print(f"\n  Firewall: {self.name}")
        print(f"  {'#':<4} {'CIDR':<18} {'Port':<6} {'Proto':<6} {'Action':<8} Description")
        print(f"  {'─'*65}")
        for i, r in enumerate(self.rules, 1):
            print(f"  {i:<4} {r.src_cidr:<18} {r.dst_port:<6} {r.protocol:<6} "
                  f"{r.action.value:<8} {r.description}")


# ─────────────────────────────────────────────
# WAF (Web Application Firewall)
# ─────────────────────────────────────────────

class WAF:
    """
    L7 inspection — detects OWASP Top 10 attack patterns.
    Analyzes request path, headers, and body.
    """

    SQL_PATTERNS = [
        r"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\s",
        r"(?i)OR\s+1\s*=\s*1",
        r"(?i)'.*--",
        r"(?i);.*DROP",
    ]
    XSS_PATTERNS = [
        r"(?i)<script[^>]*>",
        r"(?i)javascript:",
        r"(?i)on\w+\s*=",
        r"(?i)<img[^>]+src[^>]+onerror",
    ]
    PATH_TRAVERSAL = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%252e%252e",
    ]
    CMD_INJECTION = [
        r"(?i)(;|\||\|\|)\s*(ls|cat|rm|wget|curl|bash|sh)\b",
        r"(?i)\$\(",
        r"(?i)`[^`]+`",
    ]

    def __init__(self):
        self.events : List[SecurityEvent] = []
        self.blocked = 0
        self.allowed = 0

    def _check(self, patterns: List[str], text: str) -> bool:
        return any(re.search(p, text) for p in patterns)

    def inspect(self, req: NetworkRequest) -> Optional[SecurityEvent]:
        full_input = f"{req.path} {req.payload} {' '.join(req.headers.values())}"

        checks = [
            (self.SQL_PATTERNS,   ThreatType.SQL_INJECTION,  "critical"),
            (self.XSS_PATTERNS,   ThreatType.XSS,            "high"),
            (self.PATH_TRAVERSAL, ThreatType.PATH_TRAVERSAL, "high"),
            (self.CMD_INJECTION,  ThreatType.COMMAND_INJ,    "critical"),
        ]
        for patterns, threat, severity in checks:
            if self._check(patterns, full_input):
                event = SecurityEvent(
                    threat_type=threat, src_ip=req.src_ip,
                    payload=full_input[:80], severity=severity, blocked=True
                )
                self.events.append(event)
                self.blocked += 1
                return event

        self.allowed += 1
        return None


# ─────────────────────────────────────────────
# DDOS PROTECTOR
# ─────────────────────────────────────────────

class DDoSProtector:
    """
    Rate limiting + IP reputation to absorb volumetric attacks.
    Strategies: connection rate limits, SYN cookies, IP blocklist.
    """

    def __init__(self, rate_limit_rps: int = 100, ban_threshold: int = 500):
        self.rate_limit_rps  = rate_limit_rps
        self.ban_threshold   = ban_threshold
        self._ip_counters    : Dict[str, int] = {}
        self._banned_ips     : Set[str] = set()
        self._window_start   = time.time()
        self.blocked_total   = 0

    def _reset_window(self):
        if time.time() - self._window_start > 1.0:
            self._ip_counters = {}
            self._window_start = time.time()

    def check(self, src_ip: str) -> bool:
        """Returns True if request should be allowed."""
        self._reset_window()
        if src_ip in self._banned_ips:
            self.blocked_total += 1
            return False
        self._ip_counters[src_ip] = self._ip_counters.get(src_ip, 0) + 1
        if self._ip_counters[src_ip] > self.ban_threshold:
            self._banned_ips.add(src_ip)
            print(f"  DDoS: IP {src_ip} BANNED ({self._ip_counters[src_ip]} req/s)")
            self.blocked_total += 1
            return False
        if self._ip_counters[src_ip] > self.rate_limit_rps:
            self.blocked_total += 1
            return False
        return True


# ─────────────────────────────────────────────
# ZERO TRUST VERIFIER
# ─────────────────────────────────────────────

class ZeroTrustVerifier:
    """
    Never trust the network; verify every request with:
    - Identity (who are you? — JWT / mTLS cert)
    - Device posture (is the device compliant?)
    - Context (IP, time, location anomalies)
    - Least privilege (do you have permission for THIS resource?)
    """

    TRUSTED_CERTS = {"cert-svc-a", "cert-svc-b", "cert-admin"}
    HIGH_RISK_COUNTRIES = {"XX", "ZZ"}   # example — not real

    def verify(self, identity: str, device_posture: str,
               resource: str, action: str) -> Dict:
        checks = {}
        checks["identity_valid"]  = identity in self.TRUSTED_CERTS
        checks["device_compliant"]= device_posture == "compliant"
        checks["action_allowed"]  = action in ("read",) or identity == "cert-admin"
        checks["country_allowed"] = True   # simplified

        allowed = all(checks.values())
        return {"allowed": allowed, "checks": checks,
                "identity": identity, "resource": resource, "action": action}


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_network_security():
    print("=" * 65)
    print("NETWORK SECURITY FUNDAMENTALS")
    print("=" * 65)

    # ── TLS Audit ─────────────────────────────
    print("\n[1] TLS CONFIGURATION AUDIT")
    print("─" * 55)
    inspector = TLSInspector()
    configs = [
        ("TLSv1.3", "TLS_AES_256_GCM_SHA384", 180),
        ("TLSv1.1", "RC4-MD5",                 5),
        ("TLSv1.2", "ECDHE-RSA-AES256-GCM-SHA384", 45),
    ]
    for version, cipher, days in configs:
        result = inspector.audit(version, cipher, days)
        grade  = result["grade"]
        issues = result["issues"]
        print(f"  Grade {grade}: {version} / {cipher[:30]:<30} expiry={days}d")
        for issue in issues:
            print(f"    ⚠  {issue}")

    # ── Firewall ──────────────────────────────
    print("\n\n[2] FIREWALL RULES")
    print("─" * 55)
    fw = Firewall("ingress-sg")
    fw.add_rule(FirewallRule("r1", "10.0.0.0/8",   443,  "tcp", FirewallAction.ALLOW, "Internal HTTPS"))
    fw.add_rule(FirewallRule("r2", "0.0.0.0/0",    443,  "tcp", FirewallAction.ALLOW, "Public HTTPS"))
    fw.add_rule(FirewallRule("r3", "0.0.0.0/0",    80,   "tcp", FirewallAction.ALLOW, "HTTP redirect"))
    fw.add_rule(FirewallRule("r4", "10.0.0.0/8",   5432, "tcp", FirewallAction.ALLOW, "DB internal only"))
    fw.add_rule(FirewallRule("r5", "0.0.0.0/0",    5432, "tcp", FirewallAction.DENY,  "Block DB from internet"))
    fw.add_rule(FirewallRule("r6", "0.0.0.0/0",    22,   "tcp", FirewallAction.DENY,  "Block SSH from internet"))
    fw.add_rule(FirewallRule("r7", "0.0.0.0/0",    -1,   "any", FirewallAction.DENY,  "Default deny all"))
    fw.show_rules()

    test_reqs = [
        NetworkRequest("10.0.1.5",  443,  "tcp"),
        NetworkRequest("1.2.3.4",   443,  "tcp"),
        NetworkRequest("1.2.3.4",   5432, "tcp"),   # blocked
        NetworkRequest("1.2.3.4",   22,   "tcp"),   # blocked
        NetworkRequest("10.0.1.5",  5432, "tcp"),   # internal DB OK
    ]
    print(f"\n  Evaluating requests:")
    for req in test_reqs:
        action = fw.evaluate(req)
        icon   = "✅" if action == FirewallAction.ALLOW else "🚫"
        print(f"  {icon} {req.src_ip:<15} port={req.dst_port:<6} → {action.value}")

    # ── WAF ───────────────────────────────────
    print("\n\n[3] WAF — ATTACK DETECTION")
    print("─" * 55)
    waf = WAF()
    attacks = [
        NetworkRequest("1.2.3.1", 443, payload="' OR 1=1--",           path="/login"),
        NetworkRequest("1.2.3.2", 443, payload="<script>alert(1)</script>", path="/comment"),
        NetworkRequest("1.2.3.3", 443, payload="",                     path="/files/../../etc/passwd"),
        NetworkRequest("1.2.3.4", 443, payload="test; rm -rf /",       path="/run"),
        NetworkRequest("1.2.3.5", 443, payload="normal product search", path="/search?q=laptop"),
    ]
    for req in attacks:
        event = waf.inspect(req)
        if event:
            print(f"  🚫 BLOCKED [{event.threat_type.value}] from {req.src_ip}  "
                  f"severity={event.severity}  payload='{req.payload[:40]}'")
        else:
            print(f"  ✅ ALLOWED  from {req.src_ip}  path={req.path}")

    # ── DDoS ──────────────────────────────────
    print("\n\n[4] DDoS PROTECTION")
    print("─" * 55)
    ddos = DDoSProtector(rate_limit_rps=5, ban_threshold=10)
    print("  Simulating 15 rapid requests from one IP:")
    for i in range(15):
        allowed = ddos.check("192.168.0.100")
        print(f"  req {i+1:02d}: {'✅ allowed' if allowed else '🚫 blocked'}")

    # ── Zero Trust ────────────────────────────
    print("\n\n[5] ZERO TRUST VERIFICATION")
    print("─" * 55)
    zt = ZeroTrustVerifier()
    cases = [
        ("cert-svc-a",  "compliant",     "/api/data",  "read"),
        ("cert-svc-a",  "compliant",     "/api/admin", "write"),   # no write perm
        ("cert-unknown","compliant",     "/api/data",  "read"),    # bad cert
        ("cert-admin",  "compliant",     "/api/admin", "write"),   # admin ok
        ("cert-svc-b",  "non-compliant", "/api/data",  "read"),    # bad device
    ]
    for identity, posture, resource, action in cases:
        result = zt.verify(identity, posture, resource, action)
        icon   = "✅" if result["allowed"] else "❌"
        failed = [k for k, v in result["checks"].items() if not v]
        print(f"  {icon} {identity:<20} {action:<6} {resource:<16} "
              + (f"DENIED: {failed}" if failed else "GRANTED"))

    # ── Security Checklist ────────────────────
    print("\n\n[6] NETWORK SECURITY CHECKLIST")
    print("─" * 55)
    checklist = [
        ("TLS 1.2+ everywhere",            "Encrypt data in transit; disable TLS 1.0/1.1"),
        ("Certificates auto-renewed",       "Use cert-manager; alert 30 days before expiry"),
        ("Firewall default-deny",           "Allow only required ports; block everything else"),
        ("VPC segmentation",               "DB in private subnet; no direct internet access"),
        ("WAF on public endpoints",         "Block SQLi, XSS, LFI before reaching app"),
        ("DDoS scrubbing",                 "Cloudflare/AWS Shield absorbs volumetric attacks"),
        ("mTLS between services",           "Service mesh ensures every service hop is encrypted"),
        ("Zero Trust network policy",       "Kubernetes NetworkPolicy — pod-to-pod allow-list"),
        ("Secret management",              "Vault/AWS Secrets Manager — never hardcode secrets"),
        ("Security logging",               "Log all denied requests; alert on anomalies"),
    ]
    for item, detail in checklist:
        print(f"  • {item:<35} {detail}")


if __name__ == "__main__":
    demonstrate_network_security()
