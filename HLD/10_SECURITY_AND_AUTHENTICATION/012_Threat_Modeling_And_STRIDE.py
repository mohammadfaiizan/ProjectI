"""
THREAT MODELING AND STRIDE
============================

Problem Statement:
Security must be designed in, not bolted on. Threat modeling is a
structured process to identify, analyze, and mitigate threats early
in the design phase.

Threat Modeling Process:
  1. Decompose the system: DFD (Data Flow Diagram).
  2. Identify threats: use STRIDE categories.
  3. Rate threats: DREAD or CVSS scoring.
  4. Mitigate: countermeasures for each threat.
  5. Validate: verify mitigations work.

DFD Elements:
  External Entity: user, external service (outside trust boundary).
  Process:         application code, service, function.
  Data Store:      database, cache, file, queue.
  Data Flow:       arrows showing data movement.
  Trust Boundary:  line between different trust levels.

STRIDE (Microsoft):
  S - Spoofing:       Pretending to be someone you're not.
                      Mitigation: Authentication (MFA, certificates).
  T - Tampering:      Modifying data in transit or at rest.
                      Mitigation: Integrity (HMAC, digital signatures, TLS).
  R - Repudiation:    Denying you did something.
                      Mitigation: Non-repudiation (audit logs, digital signatures).
  I - Information Disclosure: Exposing private data.
                      Mitigation: Confidentiality (encryption, access control).
  D - Denial of Service: Making service unavailable.
                      Mitigation: Availability (rate limiting, redundancy).
  E - Elevation of Privilege: Gaining more access than granted.
                      Mitigation: Authorization (least privilege, RBAC).

DREAD Scoring (5 factors, 0-10 each):
  Damage potential:   How bad if exploited?
  Reproducibility:    How easy to reproduce?
  Exploitability:     How easy to exploit?
  Affected users:     How many affected?
  Discoverability:    How easy to discover?
  Score = avg(D+R+E+A+D). Higher = more critical.

CVSS (Common Vulnerability Scoring System):
  Industry-standard scoring. 0-10.
  Vector string: AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H = 10.0 (critical).

Attack Trees:
  Root: attacker's goal (e.g., "steal user PII").
  Children: ways to achieve goal (AND/OR nodes).
  Leaves: specific attacks.
  Used to analyze which countermeasures are most effective.

PASTA (Process for Attack Simulation and Threat Analysis):
  7-stage methodology. Business risk driven.
  Stage 1: Define objectives. Stage 7: Enumerate and score attack scenarios.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import json


# ─────────────────────────────────────────────
# STRIDE CATEGORIES
# ─────────────────────────────────────────────

class STRIDECategory(Enum):
    SPOOFING              = "Spoofing"
    TAMPERING             = "Tampering"
    REPUDIATION           = "Repudiation"
    INFORMATION_DISCLOSURE= "Information Disclosure"
    DENIAL_OF_SERVICE     = "Denial of Service"
    ELEVATION_OF_PRIVILEGE= "Elevation of Privilege"


class ThreatSeverity(Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    INFO     = "INFO"


# ─────────────────────────────────────────────
# DFD ELEMENTS
# ─────────────────────────────────────────────

class DFDElementType(Enum):
    EXTERNAL_ENTITY = "External Entity"
    PROCESS         = "Process"
    DATA_STORE      = "Data Store"
    DATA_FLOW       = "Data Flow"


@dataclass
class DFDElement:
    element_id: str
    name      : str
    element_type: DFDElementType
    trust_level: int   # 0 = untrusted (internet), higher = more trusted
    description: str = ""


@dataclass
class DataFlow:
    flow_id   : str
    source_id : str
    dest_id   : str
    data_desc : str
    protocol  : str
    encrypted : bool = False
    authenticated: bool = False


@dataclass
class TrustBoundary:
    boundary_id: str
    name       : str
    elements   : Set[str]   # element IDs inside boundary


# ─────────────────────────────────────────────
# THREAT
# ─────────────────────────────────────────────

@dataclass
class Threat:
    threat_id   : str
    name        : str
    category    : STRIDECategory
    description : str
    target      : str               # element_id or flow_id
    attack_vector: str
    dread_scores: Dict[str, int]    # D,R,E,A,D each 0-10
    mitigations : List[str]
    status      : str = "OPEN"      # OPEN, MITIGATED, ACCEPTED, FALSE_POSITIVE
    cve         : Optional[str] = None

    @property
    def dread_score(self) -> float:
        scores = list(self.dread_scores.values())
        return sum(scores) / len(scores) if scores else 0

    @property
    def severity(self) -> ThreatSeverity:
        s = self.dread_score
        if s >= 8:  return ThreatSeverity.CRITICAL
        if s >= 6:  return ThreatSeverity.HIGH
        if s >= 4:  return ThreatSeverity.MEDIUM
        if s >= 2:  return ThreatSeverity.LOW
        return ThreatSeverity.INFO


# ─────────────────────────────────────────────
# THREAT MODEL
# ─────────────────────────────────────────────

class ThreatModel:
    """
    Threat model for a system: DFD + STRIDE threats + mitigations.
    """

    def __init__(self, system_name: str):
        self.system_name  = system_name
        self._elements    : Dict[str, DFDElement]   = {}
        self._flows       : Dict[str, DataFlow]     = {}
        self._boundaries  : List[TrustBoundary]     = []
        self._threats     : List[Threat]            = []

    def add_element(self, element: DFDElement):
        self._elements[element.element_id] = element

    def add_flow(self, flow: DataFlow):
        self._flows[flow.flow_id] = flow

    def add_boundary(self, boundary: TrustBoundary):
        self._boundaries.append(boundary)

    def add_threat(self, threat: Threat):
        self._threats.append(threat)

    def auto_analyze_threats(self):
        """
        Automatically generate STRIDE threats based on DFD elements and flows.
        """
        generated = []

        # Analyze data flows
        for flow in self._flows.values():
            src = self._elements.get(flow.source_id)
            dst = self._elements.get(flow.dest_id)

            if not flow.encrypted:
                generated.append(Threat(
                    threat_id=f"T-{len(self._threats)+len(generated)+1:03d}",
                    name=f"Unencrypted flow: {flow.flow_id}",
                    category=STRIDECategory.INFORMATION_DISCLOSURE,
                    description=f"Data flow '{flow.data_desc}' is not encrypted",
                    target=flow.flow_id,
                    attack_vector="Network sniffing / MITM",
                    dread_scores={"D":7,"R":7,"E":5,"A":8,"D2":7},
                    mitigations=["Enforce TLS 1.2+", "Certificate validation"],
                ))

            if not flow.authenticated:
                generated.append(Threat(
                    threat_id=f"T-{len(self._threats)+len(generated)+1:03d}",
                    name=f"Unauthenticated flow: {flow.flow_id}",
                    category=STRIDECategory.SPOOFING,
                    description=f"Flow '{flow.data_desc}' has no authentication",
                    target=flow.flow_id,
                    attack_vector="Request forgery / impersonation",
                    dread_scores={"D":8,"R":6,"E":5,"A":8,"D2":5},
                    mitigations=["Require API keys or JWT", "mTLS for internal services"],
                ))

        # Analyze data stores
        for elem in self._elements.values():
            if elem.element_type == DFDElementType.DATA_STORE and elem.trust_level < 3:
                generated.append(Threat(
                    threat_id=f"T-{len(self._threats)+len(generated)+1:03d}",
                    name=f"Data store tampering: {elem.name}",
                    category=STRIDECategory.TAMPERING,
                    description=f"Data store '{elem.name}' may lack integrity controls",
                    target=elem.element_id,
                    attack_vector="SQL injection / direct DB access",
                    dread_scores={"D":8,"R":5,"E":4,"A":9,"D2":4},
                    mitigations=["Parameterized queries", "DB access controls",
                                  "Audit all writes"],
                ))

        self._threats.extend(generated)
        return generated

    def threats_by_severity(self) -> Dict[ThreatSeverity, List[Threat]]:
        result: Dict[ThreatSeverity, List[Threat]] = {}
        for t in self._threats:
            if t.severity not in result:
                result[t.severity] = []
            result[t.severity].append(t)
        return result

    def risk_summary(self) -> Dict:
        by_cat: Dict[str, int] = {}
        by_sev: Dict[str, int] = {}
        open_threats = [t for t in self._threats if t.status == "OPEN"]
        for t in self._threats:
            by_cat[t.category.value] = by_cat.get(t.category.value, 0) + 1
            by_sev[t.severity.value] = by_sev.get(t.severity.value, 0) + 1
        return {
            "total_threats"  : len(self._threats),
            "open_threats"   : len(open_threats),
            "mitigated"      : len(self._threats) - len(open_threats),
            "by_category"    : by_cat,
            "by_severity"    : by_sev,
        }


# ─────────────────────────────────────────────
# ATTACK TREE NODE
# ─────────────────────────────────────────────

@dataclass
class AttackNode:
    goal     : str
    node_type: str          # "OR" or "AND" or "LEAF"
    children : List["AttackNode"] = field(default_factory=list)
    cost     : float = 0    # attacker effort
    probability: float = 0  # probability of success

    def add_child(self, child: "AttackNode"):
        self.children.append(child)

    def feasibility(self) -> float:
        """Estimate attack feasibility (0-1)."""
        if self.node_type == "LEAF":
            return self.probability
        if self.node_type == "OR":
            return max((c.feasibility() for c in self.children), default=0)
        if self.node_type == "AND":
            result = 1.0
            for c in self.children:
                result *= c.feasibility()
            return result
        return 0

    def min_cost(self) -> float:
        """Minimum attacker cost to succeed."""
        if self.node_type == "LEAF":
            return self.cost
        if self.node_type == "OR":
            return min((c.min_cost() for c in self.children), default=float("inf"))
        if self.node_type == "AND":
            return sum(c.min_cost() for c in self.children)
        return float("inf")

    def print_tree(self, indent: int = 0):
        prefix = "  " * indent
        print(f"{prefix}[{self.node_type}] {self.goal} "
              f"(p={self.feasibility():.0%}, cost={self.min_cost():.0f})")
        for child in self.children:
            child.print_tree(indent + 1)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_threat_modeling():
    print("=" * 65)
    print("THREAT MODELING AND STRIDE")
    print("=" * 65)

    # ── Build DFD for a Payment System ────────────
    print("\n[1] DFD — PAYMENT PROCESSING SYSTEM")
    print("─" * 55)

    tm = ThreatModel("Payment Processing System")

    elements = [
        DFDElement("user",       "Browser/Mobile App", DFDElementType.EXTERNAL_ENTITY, 0),
        DFDElement("api-gw",     "API Gateway",         DFDElementType.PROCESS,         2),
        DFDElement("payment-svc","Payment Service",     DFDElementType.PROCESS,         3),
        DFDElement("db",         "Payment Database",    DFDElementType.DATA_STORE,      3),
        DFDElement("card-proc",  "Card Processor (ext)",DFDElementType.EXTERNAL_ENTITY, 1),
    ]
    for e in elements:
        tm.add_element(e)
        print(f"  {e.element_type.value:<18} '{e.name}' trust_level={e.trust_level}")

    flows = [
        DataFlow("F1", "user",        "api-gw",      "Payment request", "HTTPS", True,  False),
        DataFlow("F2", "api-gw",      "payment-svc", "Validated request","HTTP",  False, True),  # internal, unencrypted
        DataFlow("F3", "payment-svc", "db",           "Store transaction","TCP",   False, True),
        DataFlow("F4", "payment-svc", "card-proc",    "Card authorization","HTTPS",True, True),
    ]
    for f in flows:
        tm.add_flow(f)

    tm.add_boundary(TrustBoundary("B1", "Internet", {"user"}))
    tm.add_boundary(TrustBoundary("B2", "DMZ", {"api-gw"}))
    tm.add_boundary(TrustBoundary("B3", "Internal", {"payment-svc", "db"}))

    # ── Manual STRIDE Threats ─────────────────────
    print("\n\n[2] STRIDE THREAT ANALYSIS")
    print("─" * 55)

    manual_threats = [
        Threat("T-001", "JWT Spoofing",
               STRIDECategory.SPOOFING,
               "Attacker forges JWT to impersonate other user",
               "api-gw", "alg=none attack or weak signing key",
               {"D":8,"R":6,"E":5,"A":9,"D2":4},
               ["Validate alg claim", "Use RS256", "Short JWT TTL"],
               status="MITIGATED"),
        Threat("T-002", "SQL Injection in Payment DB",
               STRIDECategory.TAMPERING,
               "SQLi via payment form fields",
               "db", "Malicious input in card fields",
               {"D":9,"R":8,"E":7,"A":9,"D2":7},
               ["Parameterized queries", "Input validation", "WAF"],
               status="MITIGATED"),
        Threat("T-003", "Card Data Skimming",
               STRIDECategory.INFORMATION_DISCLOSURE,
               "Internal attacker reads card numbers from DB",
               "db", "Rogue DBA / compromised service account",
               {"D":10,"R":4,"E":3,"A":9,"D2":5},
               ["Encrypt card data at rest", "Field-level encryption",
                "Tokenization (never store PAN)"],
               status="OPEN"),
        Threat("T-004", "Payment Service Flood",
               STRIDECategory.DENIAL_OF_SERVICE,
               "Payment API flooded to prevent transactions",
               "payment-svc", "Botnet HTTP flood",
               {"D":7,"R":8,"E":7,"A":7,"D2":6},
               ["Rate limiting per user", "AWS Shield", "WAF"],
               status="OPEN"),
        Threat("T-005", "No Audit Log for Payments",
               STRIDECategory.REPUDIATION,
               "User denies making payment; no non-repudiation",
               "payment-svc", "User claims transaction was unauthorized",
               {"D":6,"R":5,"E":3,"A":7,"D2":4},
               ["Immutable audit log", "Digital signature on transactions"],
               status="OPEN"),
    ]
    for t in manual_threats:
        tm.add_threat(t)

    # Auto-generate threats from DFD
    auto = tm.auto_analyze_threats()

    all_threats = tm._threats
    print(f"  {'ID':<8} {'Category':<26} {'Severity':<10} {'Score':>6} {'Status':<12} {'Name'}")
    print(f"  {'─'*80}")
    for t in sorted(all_threats, key=lambda x: -x.dread_score):
        print(f"  {t.threat_id:<8} {t.category.value:<26} "
              f"{t.severity.value:<10} {t.dread_score:>5.1f}  "
              f"{t.status:<12} {t.name[:30]}")

    # ── Risk Summary ──────────────────────────────
    print("\n\n[3] RISK SUMMARY")
    print("─" * 55)

    summary = tm.risk_summary()
    print(f"  Total threats: {summary['total_threats']}")
    print(f"  Open: {summary['open_threats']}  Mitigated: {summary['mitigated']}")
    print(f"  By severity: {summary['by_severity']}")
    print(f"  By category:")
    for cat, count in sorted(summary['by_category'].items()):
        print(f"    {cat}: {count}")

    # ── Attack Tree ───────────────────────────────
    print("\n\n[4] ATTACK TREE — STEAL PAYMENT DATA")
    print("─" * 55)

    root = AttackNode("Steal payment data", "OR")

    # Branch 1: Attack database directly
    db_attack = AttackNode("Compromise DB directly", "OR")
    db_attack.add_child(AttackNode("SQL Injection", "LEAF", probability=0.3, cost=5))
    db_attack.add_child(AttackNode("Stolen DBA credentials", "LEAF", probability=0.2, cost=20))

    # Branch 2: Attack in transit
    network_attack = AttackNode("Intercept traffic", "AND")
    network_attack.add_child(AttackNode("MITM position", "LEAF", probability=0.4, cost=30))
    network_attack.add_child(AttackNode("Break TLS (requires)", "LEAF", probability=0.05, cost=1000))

    # Branch 3: Compromise payment service
    app_attack = AttackNode("Compromise payment service", "OR")
    app_attack.add_child(AttackNode("RCE via CVE", "LEAF", probability=0.1, cost=50))
    app_attack.add_child(AttackNode("Supply chain compromise", "LEAF", probability=0.05, cost=200))

    root.add_child(db_attack)
    root.add_child(network_attack)
    root.add_child(app_attack)

    root.print_tree()

    print(f"\n  Overall attack feasibility: {root.feasibility():.1%}")
    print(f"  Minimum attacker cost:       {root.min_cost():.0f} units")

    # ── STRIDE Summary ────────────────────────────
    print("\n\n[5] STRIDE MITIGATION SUMMARY")
    print("─" * 55)

    stride_mitigations = [
        (STRIDECategory.SPOOFING,              "MFA, JWT RS256, API keys, mTLS"),
        (STRIDECategory.TAMPERING,             "HMAC signatures, TLS, input validation, WORM logs"),
        (STRIDECategory.REPUDIATION,           "Signed audit logs, digital signatures, NTP"),
        (STRIDECategory.INFORMATION_DISCLOSURE,"Encryption at rest+transit, access control"),
        (STRIDECategory.DENIAL_OF_SERVICE,     "Rate limiting, circuit breakers, CDN, DDoS protection"),
        (STRIDECategory.ELEVATION_OF_PRIVILEGE,"Least privilege, RBAC, no privilege inheritance"),
    ]
    for cat, mitigation in stride_mitigations:
        print(f"  {cat.value:<28} {mitigation}")


if __name__ == "__main__":
    demonstrate_threat_modeling()
