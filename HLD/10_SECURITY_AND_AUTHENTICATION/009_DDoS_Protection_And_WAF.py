"""
DDOS PROTECTION AND WEB APPLICATION FIREWALL (WAF)
=====================================================

Problem Statement:
Distributed Denial of Service (DDoS) attacks overwhelm systems with traffic,
making them unavailable to legitimate users. WAFs filter malicious requests
at the application layer.

DDoS Attack Types:
  Volume-based:    Flood bandwidth with UDP/ICMP. Measured in Gbps.
                   Example: DNS amplification, SSDP reflection.
  Protocol-based:  Exploit protocol weaknesses. SYN flood, Ping of Death.
                   Measured in Mpps (million packets per second).
  Application-layer (L7): HTTP flood, slow HTTP (Slowloris).
                   Measured in RPS (requests per second).
                   Harder to detect: looks like legitimate traffic.

DDoS Mitigation Layers:
  ISP/Upstream:   Scrubbing centers. BGP routing to absorb traffic.
  CDN:            Absorb volumetric attacks. Anycast routing.
                  Cloudflare, Akamai, AWS CloudFront.
  Cloud DDoS:     AWS Shield, GCP Cloud Armor, Azure DDoS Protection.
  Network:        Rate limit at firewall/load balancer. SYN cookies.
  Application:    Rate limiting, CAPTCHAs, IP reputation, behavioral analysis.

WAF (Web Application Firewall):
  Inspects HTTP/HTTPS traffic at Layer 7.
  Blocks: SQLi, XSS, CSRF, path traversal, known attack signatures.
  Types:
    Allowlist-based: only permit known good traffic (strict).
    Blocklist-based: block known attack signatures (OWASP CRS).
    Hybrid: both.
  OWASP ModSecurity Core Rule Set (CRS): standard WAF ruleset.
  Products: AWS WAF, Cloudflare WAF, ModSecurity, Imperva.

Rate Limiting vs DDoS Protection:
  Rate limiting: per-IP or per-user request limits. Simple, fast.
  DDoS protection: behavioral analysis, traffic shaping, CAPTCHAs,
                   IP reputation, geoblocking.

Anycast / CDN:
  Traffic routed to nearest CDN PoP (Point of Presence).
  Attack traffic absorbed across hundreds of PoPs globally.
  Origin server protected behind CDN (hide real IP).

Detection Signals:
  Request volume spike. High error rate. Unusual user agents.
  Geographic anomaly (traffic from unexpected countries).
  Payload anomaly (unusual headers, large body, slow sends).
  IP reputation: known bad actors, Tor exit nodes, botnets.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
import re
import time
import hashlib
import ipaddress
import secrets
from collections import defaultdict, deque
import threading


# ─────────────────────────────────────────────
# IP REPUTATION
# ─────────────────────────────────────────────

class IPReputationLevel(Enum):
    CLEAN   = "CLEAN"
    SUSPECT = "SUSPECT"
    KNOWN_BAD = "KNOWN_BAD"
    TOR_EXIT  = "TOR_EXIT"


class IPReputationDB:
    def __init__(self):
        self._reputation: Dict[str, IPReputationLevel] = {}
        self._blocklists : Set[str] = set()

    def add_known_bad(self, ip: str):
        self._reputation[ip] = IPReputationLevel.KNOWN_BAD
        self._blocklists.add(ip)

    def add_tor_exit(self, ip: str):
        self._reputation[ip] = IPReputationLevel.TOR_EXIT

    def get_reputation(self, ip: str) -> IPReputationLevel:
        # Check exact match
        if ip in self._reputation:
            return self._reputation[ip]
        # Check if in reserved/private ranges
        try:
            addr = ipaddress.ip_address(ip)
            if addr.is_private or addr.is_loopback:
                return IPReputationLevel.CLEAN
        except ValueError:
            pass
        return IPReputationLevel.CLEAN

    def is_blocked(self, ip: str) -> bool:
        return ip in self._blocklists


# ─────────────────────────────────────────────
# WAF RULES
# ─────────────────────────────────────────────

class WafAction(Enum):
    ALLOW   = "ALLOW"
    BLOCK   = "BLOCK"
    CAPTCHA = "CAPTCHA"
    LOG     = "LOG"


@dataclass
class WafRule:
    rule_id    : str
    name       : str
    priority   : int        # lower = checked first
    action     : WafAction
    conditions : List[Dict] # [{field, pattern, op}]

    def matches(self, request: "HttpRequest") -> bool:
        for cond in self.conditions:
            field   = cond["field"]
            pattern = cond.get("pattern", "")
            op      = cond.get("op", "contains")

            value = self._get_field(request, field)
            if value is None:
                return False

            if op == "contains":
                if pattern.lower() not in value.lower():
                    return False
            elif op == "regex":
                if not re.search(pattern, value, re.IGNORECASE):
                    return False
            elif op == "==":
                if value != pattern:
                    return False
            elif op == "startswith":
                if not value.lower().startswith(pattern.lower()):
                    return False
        return True

    def _get_field(self, req: "HttpRequest", field: str) -> Optional[str]:
        if field == "uri":       return req.path
        if field == "method":    return req.method
        if field == "body":      return req.body or ""
        if field == "user_agent":return req.headers.get("User-Agent", "")
        if field.startswith("header:"): return req.headers.get(field[7:], "")
        if field.startswith("query:"):  return req.query_params.get(field[6:], "")
        return None


# ─────────────────────────────────────────────
# HTTP REQUEST
# ─────────────────────────────────────────────

@dataclass
class HttpRequest:
    ip           : str
    method       : str
    path         : str
    headers      : Dict[str, str]
    body         : Optional[str] = None
    query_params : Dict[str, str] = field(default_factory=dict)
    timestamp    : float = field(default_factory=time.time)


@dataclass
class WafDecision:
    action      : WafAction
    rule_id     : Optional[str]
    reason      : str
    blocked     : bool


# ─────────────────────────────────────────────
# WAF ENGINE
# ─────────────────────────────────────────────

class WAFEngine:
    """
    Web Application Firewall with:
    - OWASP-like rule matching.
    - IP reputation filtering.
    - Rate limiting per IP.
    - Anomaly detection.
    """

    def __init__(self, ip_db: IPReputationDB):
        self._rules      : List[WafRule] = []
        self._ip_db      = ip_db
        self._request_log: Dict[str, deque] = defaultdict(lambda: deque())
        self._blocked_ips: Set[str] = set()
        self._lock       = threading.Lock()
        self._audit_log  : List[Dict] = []
        self._rate_limit  = 100    # req/min per IP
        self._load_default_rules()

    def _load_default_rules(self):
        """Load OWASP CRS-like rules."""
        default_rules = [
            WafRule("SQL-001", "SQL Injection - Basic", 10, WafAction.BLOCK,
                    [{"field": "body", "op": "regex",
                      "pattern": r"(union|select|drop|insert|update|delete|exec).*(\(|where|from)"}]),
            WafRule("SQL-002", "SQL Injection - Comments", 10, WafAction.BLOCK,
                    [{"field": "body", "op": "regex", "pattern": r"(--|#|/\*).*"}]),
            WafRule("XSS-001", "XSS - Script Tag", 10, WafAction.BLOCK,
                    [{"field": "body", "op": "regex",
                      "pattern": r"<script[\s\S]*?>[\s\S]*?</script>"}]),
            WafRule("XSS-002", "XSS - Event Handler", 10, WafAction.BLOCK,
                    [{"field": "body", "op": "regex",
                      "pattern": r"on(click|load|error|mouseover)\s*="}]),
            WafRule("PT-001", "Path Traversal", 10, WafAction.BLOCK,
                    [{"field": "uri", "op": "contains", "pattern": "../"}]),
            WafRule("UA-001", "Scanner User-Agent", 20, WafAction.BLOCK,
                    [{"field": "user_agent", "op": "regex",
                      "pattern": r"(sqlmap|nikto|nmap|masscan|zgrab|nuclei)"}]),
            WafRule("METH-001", "Dangerous Method", 30, WafAction.BLOCK,
                    [{"field": "method", "op": "regex",
                      "pattern": r"^(TRACE|TRACK|DEBUG)$"}]),
        ]
        self._rules.extend(default_rules)

    def add_rule(self, rule: WafRule):
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)

    def inspect(self, request: HttpRequest) -> WafDecision:
        # 1. IP blocklist
        if request.ip in self._blocked_ips:
            return WafDecision(WafAction.BLOCK, "IP-BLOCK", "IP permanently blocked", True)

        # 2. IP reputation
        rep = self._ip_db.get_reputation(request.ip)
        if rep == IPReputationLevel.KNOWN_BAD:
            return WafDecision(WafAction.BLOCK, "REP-001", f"Known bad IP: {request.ip}", True)
        if rep == IPReputationLevel.TOR_EXIT:
            return WafDecision(WafAction.CAPTCHA, "REP-002", "Tor exit node", False)

        # 3. Rate limiting
        if not self._check_rate_limit(request.ip):
            return WafDecision(WafAction.BLOCK, "RL-001",
                               f"Rate limit exceeded ({self._rate_limit} req/min)", True)

        # 4. Rule matching
        for rule in sorted(self._rules, key=lambda r: r.priority):
            if rule.matches(request):
                self._log_event(request, rule.rule_id, rule.name, rule.action)
                return WafDecision(rule.action, rule.rule_id, rule.name,
                                    rule.action == WafAction.BLOCK)

        # 5. Anomaly scoring (simplified)
        score = self._anomaly_score(request)
        if score >= 50:
            return WafDecision(WafAction.CAPTCHA, "ANOM-001",
                                f"High anomaly score: {score}", False)

        self._log_event(request, None, "allowed", WafAction.ALLOW)
        return WafDecision(WafAction.ALLOW, None, "passed_all_checks", False)

    def _check_rate_limit(self, ip: str) -> bool:
        now    = time.time()
        cutoff = now - 60
        with self._lock:
            log = self._request_log[ip]
            while log and log[0] < cutoff:
                log.popleft()
            if len(log) >= self._rate_limit:
                return False
            log.append(now)
            return True

    def _anomaly_score(self, req: HttpRequest) -> int:
        score = 0
        ua    = req.headers.get("User-Agent", "")
        if not ua:                        score += 20
        if len(req.body or "") > 100_000: score += 15  # unusually large body
        if req.path.count(".."):          score += 25
        if not req.headers.get("Accept"): score += 10
        return score

    def _log_event(self, req: HttpRequest, rule_id: Optional[str],
                    reason: str, action: WafAction):
        self._audit_log.append({
            "ts"     : time.time(),
            "ip"     : req.ip,
            "method" : req.method,
            "path"   : req.path,
            "rule"   : rule_id,
            "action" : action.value,
            "reason" : reason,
        })

    def block_ip(self, ip: str):
        self._blocked_ips.add(ip)

    def stats(self) -> Dict:
        from collections import Counter
        actions = Counter(e["action"] for e in self._audit_log)
        return {"total_requests": len(self._audit_log),
                "blocked": actions.get("BLOCK", 0),
                "allowed": actions.get("ALLOW", 0),
                "captcha": actions.get("CAPTCHA", 0)}


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_ddos_waf():
    print("=" * 65)
    print("DDOS PROTECTION AND WAF")
    print("=" * 65)

    ip_db = IPReputationDB()
    ip_db.add_known_bad("1.2.3.4")
    ip_db.add_tor_exit("5.5.5.5")

    waf = WAFEngine(ip_db)

    # ── Normal Requests ───────────────────────────
    print("\n[1] NORMAL REQUESTS")
    print("─" * 55)

    normal_requests = [
        HttpRequest("10.0.0.1", "GET", "/api/users/42",
                    {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}),
        HttpRequest("10.0.0.2", "POST", "/api/login",
                    {"User-Agent": "Chrome/120", "Accept": "*/*"},
                    body='{"username":"alice","password":"secret"}'),
    ]
    for req in normal_requests:
        d = waf.inspect(req)
        print(f"  {req.method} {req.path}: {d.action.value} ({d.reason})")

    # ── Attack Requests ───────────────────────────
    print("\n\n[2] ATTACK REQUESTS — WAF BLOCKING")
    print("─" * 55)

    attack_requests = [
        ("SQL Injection", HttpRequest("192.168.1.1", "POST", "/api/users",
                          {"User-Agent": "curl/7.0", "Accept": "*/*"},
                          body="username=admin' OR '1'='1'; DROP TABLE users; --")),
        ("XSS Attack", HttpRequest("192.168.1.1", "POST", "/api/comment",
                        {"User-Agent": "Mozilla/5.0", "Accept": "*/*"},
                        body='<script>alert("xss")</script>')),
        ("Path Traversal", HttpRequest("192.168.1.1", "GET", "../../etc/passwd",
                            {"User-Agent": "curl", "Accept": "*/*"})),
        ("Scanner UA", HttpRequest("192.168.1.2", "GET", "/",
                        {"User-Agent": "sqlmap/1.7 (https://sqlmap.org)"})),
        ("Known Bad IP", HttpRequest("1.2.3.4", "GET", "/api/data",
                          {"User-Agent": "Mozilla/5.0", "Accept": "*/*"})),
        ("Tor Exit Node", HttpRequest("5.5.5.5", "GET", "/admin",
                           {"User-Agent": "Mozilla/5.0", "Accept": "*/*"})),
    ]
    for label, req in attack_requests:
        d = waf.inspect(req)
        print(f"  {label:<20}: {d.action.value:<8} rule={d.rule_id} reason={d.reason}")

    # ── Rate Limiting ─────────────────────────────
    print("\n\n[3] RATE LIMITING (DDoS L7)")
    print("─" * 55)

    waf._rate_limit = 5   # low limit for demo
    attacker_ip     = "99.99.99.99"
    results = []
    for i in range(8):
        req = HttpRequest(attacker_ip, "GET", "/api/products",
                          {"User-Agent": "bot/1.0", "Accept": "*/*"})
        d = waf.inspect(req)
        results.append(d.action.value)

    print(f"  8 requests from {attacker_ip} (limit=5/min):")
    for i, action in enumerate(results, 1):
        print(f"    Req {i}: {action}")

    # ── WAF Stats ─────────────────────────────────
    print("\n\n[4] WAF STATISTICS")
    print("─" * 55)

    stats = waf.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ── DDoS Mitigation Layers ────────────────────
    print("\n\n[5] DDOS MITIGATION LAYERS")
    print("─" * 55)

    layers = [
        ("ISP/Upstream",      "BGP blackholing, scrubbing centers (>1Tbps capacity)"),
        ("CDN/Anycast",       "Absorb volumetric attacks; distribute traffic globally"),
        ("Cloud DDoS",        "AWS Shield Advanced, GCP Cloud Armor, Cloudflare"),
        ("Network layer",     "SYN cookies, connection rate limits, IP reputation"),
        ("WAF (L7)",          "Rule-based + ML anomaly detection at HTTP layer"),
        ("Rate limiting",     "Token bucket per IP; 429 with Retry-After"),
        ("CAPTCHA",           "Challenge suspicious traffic before serving"),
        ("IP geoblocking",    "Block countries with no legitimate traffic (optional)"),
    ]
    for layer, mitigation in layers:
        print(f"  {layer:<22} {mitigation}")


if __name__ == "__main__":
    demonstrate_ddos_waf()
