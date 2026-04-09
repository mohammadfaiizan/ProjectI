"""
AUDIT LOGGING AND COMPLIANCE
================================

Problem Statement:
Regulated industries (finance, healthcare, government) require:
  - Immutable audit trails of all access and changes.
  - Ability to prove "who accessed what, when, from where."
  - Retention for years (SOX: 7 years, HIPAA: 6 years, GDPR: varies).
  - Alerting on suspicious patterns.

What to Audit:
  Authentication: login attempts, success/failure, MFA events.
  Authorization:  permission checks, access denied.
  Data access:    read/write/delete of sensitive records (PII, financial).
  Admin actions:  user management, role changes, config changes.
  System events:  service start/stop, config changes.

Audit Log Properties:
  Immutable:   Cannot be modified or deleted (write-once storage).
  Tamper-evident: Hash chain or WORM storage.
  Complete:    No gaps in event sequence (sequence numbers).
   Timestamped:Accurate timestamps (NTP-synced, UTC).
  Structured:  Machine-readable (JSON) for analysis.
  Contextual:  IP, user agent, session ID, correlation ID.

SIEM (Security Information and Event Management):
  Aggregates logs from all systems.
  Correlates events to detect attacks.
  Products: Splunk, Elastic SIEM, Sumo Logic, AWS Security Hub.

Compliance Frameworks:
  SOC 2:      Security, availability, confidentiality. SaaS companies.
  HIPAA:      Healthcare. PHI access controls + audit logs.
  PCI DSS:    Payment card data. Log all access to cardholder data.
  GDPR:       EU data privacy. Right to access/erasure. Consent logs.
  SOX:        Sarbanes-Oxley. Financial records integrity. 7-year retention.
  ISO 27001:  Information security management. ISMS.

Log Integrity:
  Hash chain: each log record includes hash of previous record.
             Tampering with any record breaks the chain.
  WORM storage: Write Once Read Many. S3 Object Lock, Glacier.
  Append-only log: no UPDATE or DELETE on audit table.
  Merkle tree: efficient integrity verification across log segments.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import hashlib
import json
import time
import uuid
import re
import threading
from collections import defaultdict


# ─────────────────────────────────────────────
# AUDIT EVENT TYPES
# ─────────────────────────────────────────────

class AuditEventType(Enum):
    AUTH_LOGIN_SUCCESS  = "auth.login.success"
    AUTH_LOGIN_FAILURE  = "auth.login.failure"
    AUTH_LOGOUT         = "auth.logout"
    AUTH_MFA_SUCCESS    = "auth.mfa.success"
    AUTH_MFA_FAILURE    = "auth.mfa.failure"
    AUTHZ_PERMIT        = "authz.permit"
    AUTHZ_DENY          = "authz.deny"
    DATA_READ           = "data.read"
    DATA_WRITE          = "data.write"
    DATA_DELETE         = "data.delete"
    ADMIN_ROLE_GRANT    = "admin.role.grant"
    ADMIN_ROLE_REVOKE   = "admin.role.revoke"
    ADMIN_CONFIG_CHANGE = "admin.config.change"
    SYSTEM_START        = "system.start"
    SYSTEM_STOP         = "system.stop"


@dataclass
class AuditEvent:
    event_id     : str
    event_type   : AuditEventType
    timestamp    : float
    user_id      : Optional[str]
    ip_address   : str
    user_agent   : str
    session_id   : Optional[str]
    correlation_id: Optional[str]
    resource     : Optional[str]
    action       : Optional[str]
    outcome      : str          # "success", "failure"
    metadata     : Dict[str, Any] = field(default_factory=dict)
    prev_hash    : str = ""     # hash of previous event (chain)
    event_hash   : str = field(default="", init=False)

    def __post_init__(self):
        self.event_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = json.dumps({
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "resource": self.resource,
            "outcome": self.outcome,
            "prev_hash": self.prev_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "event_id"     : self.event_id,
            "event_type"   : self.event_type.value,
            "timestamp"    : self.timestamp,
            "ts_iso"       : time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime(self.timestamp)),
            "user_id"      : self.user_id,
            "ip_address"   : self.ip_address,
            "session_id"   : self.session_id,
            "correlation_id": self.correlation_id,
            "resource"     : self.resource,
            "action"       : self.action,
            "outcome"      : self.outcome,
            "metadata"     : self.metadata,
            "prev_hash"    : self.prev_hash,
            "event_hash"   : self.event_hash,
        }


# ─────────────────────────────────────────────
# AUDIT LOGGER (immutable chain)
# ─────────────────────────────────────────────

class ImmutableAuditLog:
    """
    Append-only audit log with hash chain for tamper evidence.
    Each event includes hash of previous event → any modification breaks chain.
    """

    def __init__(self):
        self._events   : List[AuditEvent] = []
        self._seq      = 0
        self._lock     = threading.Lock()
        self._last_hash = "0" * 64   # genesis hash

    def log(self, event_type: AuditEventType, user_id: str = None,
            ip_address: str = "unknown", user_agent: str = "unknown",
            session_id: str = None, correlation_id: str = None,
            resource: str = None, action: str = None,
            outcome: str = "success", metadata: Dict = None) -> AuditEvent:

        with self._lock:
            self._seq += 1
            event = AuditEvent(
                event_id=f"evt-{self._seq:08d}-{uuid.uuid4().hex[:8]}",
                event_type=event_type,
                timestamp=time.time(),
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=session_id,
                correlation_id=correlation_id,
                resource=resource,
                action=action,
                outcome=outcome,
                metadata=metadata or {},
                prev_hash=self._last_hash,
            )
            self._events.append(event)
            self._last_hash = event.event_hash
            return event

    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """Verify hash chain integrity. Returns (valid, first_broken_index)."""
        if not self._events:
            return True, None
        prev_hash = "0" * 64
        for i, event in enumerate(self._events):
            # Recompute expected hash
            expected = AuditEvent(
                event_id=event.event_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                user_id=event.user_id,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                session_id=event.session_id,
                correlation_id=event.correlation_id,
                resource=event.resource,
                action=event.action,
                outcome=event.outcome,
                metadata=event.metadata,
                prev_hash=prev_hash,
            ).event_hash

            if event.prev_hash != prev_hash:
                return False, i
            if event.event_hash != expected:
                return False, i
            prev_hash = event.event_hash
        return True, None

    def query(self, user_id: str = None, event_type: AuditEventType = None,
               resource: str = None, since_ts: float = None,
               limit: int = 100) -> List[AuditEvent]:
        results = self._events
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if resource:
            results = [e for e in results if e.resource == resource]
        if since_ts:
            results = [e for e in results if e.timestamp >= since_ts]
        return results[-limit:]

    def total_events(self) -> int:
        return len(self._events)


# ─────────────────────────────────────────────
# ANOMALY DETECTION (SIEM-like)
# ─────────────────────────────────────────────

class AuditAnomalyDetector:
    """
    Rule-based anomaly detection on audit events.
    Simplified SIEM correlation rules.
    """

    def __init__(self):
        self._failed_logins : Dict[str, List[float]] = defaultdict(list)
        self._alerts        : List[Dict] = []

    def analyze(self, event: AuditEvent):
        self._check_brute_force(event)
        self._check_impossible_travel(event)
        self._check_mass_data_access(event)
        self._check_privilege_escalation(event)

    def _check_brute_force(self, event: AuditEvent):
        if event.event_type != AuditEventType.AUTH_LOGIN_FAILURE:
            return
        key = event.user_id or event.ip_address
        now = time.time()
        self._failed_logins[key] = [t for t in self._failed_logins[key]
                                      if now - t < 300]  # 5-min window
        self._failed_logins[key].append(now)
        if len(self._failed_logins[key]) >= 5:
            self._alert("BRUTE_FORCE", f"5+ failed logins in 5min: {key}",
                         "HIGH", event)

    def _check_impossible_travel(self, event: AuditEvent):
        if event.event_type == AuditEventType.AUTH_LOGIN_SUCCESS:
            country = event.metadata.get("country")
            last_country = event.metadata.get("prev_country")
            last_ts      = event.metadata.get("prev_login_ts")
            if country and last_country and country != last_country and last_ts:
                time_diff_h = (event.timestamp - last_ts) / 3600
                if time_diff_h < 2:
                    self._alert("IMPOSSIBLE_TRAVEL",
                                f"Login from {country} then {last_country} in {time_diff_h:.1f}h",
                                "HIGH", event)

    def _check_mass_data_access(self, event: AuditEvent):
        if event.event_type == AuditEventType.DATA_READ:
            count = event.metadata.get("records_accessed", 0)
            if count > 10000:
                self._alert("MASS_DATA_EXPORT",
                             f"User {event.user_id} accessed {count} records",
                             "MEDIUM", event)

    def _check_privilege_escalation(self, event: AuditEvent):
        if event.event_type == AuditEventType.ADMIN_ROLE_GRANT:
            if event.user_id == event.metadata.get("target_user"):
                self._alert("SELF_PRIVILEGE_ESCALATION",
                             f"User {event.user_id} granted themselves admin",
                             "CRITICAL", event)

    def _alert(self, alert_type: str, message: str,
                severity: str, event: AuditEvent):
        self._alerts.append({
            "alert_type" : alert_type,
            "message"    : message,
            "severity"   : severity,
            "event_id"   : event.event_id,
            "timestamp"  : event.timestamp,
            "user_id"    : event.user_id,
        })

    def get_alerts(self) -> List[Dict]:
        return list(self._alerts)


# ─────────────────────────────────────────────
# COMPLIANCE REPORT GENERATOR
# ─────────────────────────────────────────────

class ComplianceReporter:
    def __init__(self, audit_log: ImmutableAuditLog):
        self._log = audit_log

    def pci_dss_report(self, from_ts: float, to_ts: float) -> Dict:
        """PCI DSS Requirement 10: log all access to cardholder data."""
        data_events = self._log.query(
            event_type=AuditEventType.DATA_READ, since_ts=from_ts
        )
        cardholder = [e for e in data_events
                       if "cardholder" in (e.resource or "").lower()
                       or "payment" in (e.resource or "").lower()]
        return {
            "period"          : f"{time.strftime('%Y-%m-%d',time.gmtime(from_ts))} to "
                                f"{time.strftime('%Y-%m-%d',time.gmtime(to_ts))}",
            "total_events"    : len(data_events),
            "cardholder_accesses": len(cardholder),
            "unique_users"    : len({e.user_id for e in cardholder}),
            "compliant"       : True,
            "requirement"     : "PCI DSS 10.2.1: log access to cardholder data",
        }

    def failed_login_summary(self, since_ts: float) -> Dict:
        failures = self._log.query(event_type=AuditEventType.AUTH_LOGIN_FAILURE,
                                    since_ts=since_ts)
        by_user: Dict[str, int] = defaultdict(int)
        by_ip  : Dict[str, int] = defaultdict(int)
        for e in failures:
            by_user[e.user_id or "unknown"] += 1
            by_ip[e.ip_address] += 1
        return {
            "total_failures"  : len(failures),
            "top_targeted_users": sorted(by_user.items(), key=lambda x: -x[1])[:5],
            "top_source_ips"  : sorted(by_ip.items(),  key=lambda x: -x[1])[:5],
        }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_audit():
    print("=" * 65)
    print("AUDIT LOGGING AND COMPLIANCE")
    print("=" * 65)

    audit   = ImmutableAuditLog()
    detector= AuditAnomalyDetector()

    # ── Log Events ────────────────────────────────
    print("\n[1] AUDIT EVENT LOGGING")
    print("─" * 55)

    events_data = [
        (AuditEventType.AUTH_LOGIN_SUCCESS, "alice", "10.0.0.1", "Chrome/120",
         "sess-1", None, None, None, {"country": "US"}),
        (AuditEventType.DATA_READ, "alice", "10.0.0.1", "Chrome/120",
         "sess-1", None, "/api/users/all", "list", {"records_accessed": 50000}),
        (AuditEventType.AUTH_LOGIN_FAILURE, "bob", "5.5.5.5", "curl/7.0",
         None, None, None, None, {"reason": "wrong_password"}),
        (AuditEventType.AUTH_LOGIN_FAILURE, "bob", "5.5.5.5", "curl/7.0",
         None, None, None, None, {}),
        (AuditEventType.AUTH_LOGIN_FAILURE, "bob", "5.5.5.5", "curl/7.0",
         None, None, None, None, {}),
        (AuditEventType.AUTH_LOGIN_FAILURE, "bob", "5.5.5.5", "curl/7.0",
         None, None, None, None, {}),
        (AuditEventType.AUTH_LOGIN_FAILURE, "bob", "5.5.5.5", "curl/7.0",
         None, None, None, None, {}),
        (AuditEventType.ADMIN_ROLE_GRANT, "carol", "10.0.1.1", "Chrome",
         "sess-2", None, None, "grant_admin",
         {"target_user": "carol", "role": "admin"}),
    ]
    logged = []
    for evt_type, uid, ip, ua, sess, corr, res, act, meta in events_data:
        e = audit.log(evt_type, uid, ip, ua, sess, corr, res, act, "success", meta)
        logged.append(e)
        detector.analyze(e)
        print(f"  {e.event_type.value:<30} user={uid}  hash={e.event_hash[:12]}...")

    # ── Chain Verification ────────────────────────
    print("\n\n[2] HASH CHAIN INTEGRITY VERIFICATION")
    print("─" * 55)

    valid, broken = audit.verify_chain()
    print(f"  Chain valid: {valid}  broken_at: {broken}")
    print(f"  Total events: {audit.total_events()}")

    # Simulate tampering
    if audit._events:
        original_hash = audit._events[0].event_hash
        audit._events[0].event_hash = "tampered_hash_" + "0" * 50
        valid2, broken2 = audit.verify_chain()
        print(f"  After tamper: valid={valid2}  broken_at_index={broken2}")
        audit._events[0].event_hash = original_hash   # restore

    # ── Query and Filter ──────────────────────────
    print("\n\n[3] AUDIT LOG QUERY")
    print("─" * 55)

    alice_events = audit.query(user_id="alice")
    print(f"  Alice's events: {len(alice_events)}")
    for e in alice_events:
        print(f"    {e.event_type.value} at {time.strftime('%H:%M:%S', time.gmtime(e.timestamp))}")

    failures = audit.query(event_type=AuditEventType.AUTH_LOGIN_FAILURE)
    print(f"  Failed logins: {len(failures)} (from {failures[0].ip_address if failures else 'N/A'})")

    # ── Anomaly Alerts ────────────────────────────
    print("\n\n[4] SIEM ANOMALY DETECTION")
    print("─" * 55)

    alerts = detector.get_alerts()
    print(f"  Alerts generated: {len(alerts)}")
    for alert in alerts:
        print(f"  [{alert['severity']:<8}] {alert['alert_type']:<28} {alert['message']}")

    # ── Compliance Report ─────────────────────────
    print("\n\n[5] COMPLIANCE REPORT (PCI DSS)")
    print("─" * 55)

    reporter = ComplianceReporter(audit)
    since    = time.time() - 3600
    report   = reporter.pci_dss_report(since, time.time())
    for k, v in report.items():
        print(f"  {k}: {v}")

    login_summary = reporter.failed_login_summary(since)
    print(f"\n  Failed login summary:")
    for k, v in login_summary.items():
        print(f"    {k}: {v}")

    # ── Retention Policy ──────────────────────────
    print("\n\n[6] COMPLIANCE RETENTION REQUIREMENTS")
    print("─" * 55)

    retention = [
        ("PCI DSS",   "1 year online, 3 months immediately available"),
        ("HIPAA",     "6 years. 60-day breach notification."),
        ("SOX",       "7 years for financial records"),
        ("GDPR",      "No longer than necessary; erasure on request"),
        ("SOC 2",     "Evidence of controls for 12-month period"),
        ("ISO 27001", "Defined in ISMS policy (typically 1-3 years)"),
    ]
    for framework, requirement in retention:
        print(f"  {framework:<12} {requirement}")


if __name__ == "__main__":
    demonstrate_audit()
