"""
ALERTING AND ON-CALL MANAGEMENT
=================================

Problem Statement:
Metrics and logs are useless if no one is notified when things break.
An alerting system turns signals into actionable pages; on-call management
ensures the right person responds at the right time.

Alert Lifecycle:
  Pending → Firing → Resolved
  Pending:  Condition is true but hasn't been for `for` duration yet.
  Firing:   Condition has been true long enough to page.
  Resolved: Condition is no longer true.

Alertmanager (Prometheus):
  Groups related alerts → one notification per group.
  Routes by label: severity=critical → PagerDuty; severity=warning → Slack.
  Inhibition: Suppress child alerts when parent fires
              (e.g. node down → suppress all pod alerts on that node).
  Silences:   Time-bounded mutes (maintenance windows).
  Receivers:  PagerDuty, Slack, email, webhook, OpsGenie.

Alert Design Principles:
  - Every alert must be actionable: if you can't do anything, it's noise.
  - Alert on symptoms, not causes (high error rate, not "disk might fill").
  - Use for: to avoid flapping on transient spikes.
  - Include runbook URL in every alert.
  - Set appropriate severity: critical=wake-up-now, warning=next-business-day.

On-Call Rotation:
  Schedule: who is on-call for which time window.
  Escalation policy: Page primary → if no ack in 5min → page secondary → manager.
  Handoff: primary hands off to next person in rotation.
  Burnout prevention: max 1 incident/night, no back-to-back on-call weeks.

PagerDuty / OpsGenie Concepts:
  Incident: Created when alert fires. Has urgency (high/low), priority, status.
  Ack:      Engineer acknowledges; stops escalation; 30min to resolve.
  Resolve:  Incident closed; duration/MTTR recorded.
  Services: Groups of alerts; each has its own escalation policy.
  Teams:    Engineers; can be members of multiple services.

MTTR / MTTD / MTTF:
  MTTD: Mean Time To Detect — alert fires latency after issue starts.
  MTTA: Mean Time To Acknowledge — p50 time between alert and ack.
  MTTR: Mean Time To Resolve — p50 incident duration.
  MTTF: Mean Time To Failure — how long between incidents.
  Availability = MTTF / (MTTF + MTTR)
"""

from __future__ import annotations

import time
import uuid
import heapq
import random
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set
from enum import Enum


# ─────────────────────────────────────────────
# ALERT SEVERITY AND STATE
# ─────────────────────────────────────────────

class Severity(Enum):
    CRITICAL = "critical"   # wake up engineer now
    HIGH     = "high"       # page within 5 min
    WARNING  = "warning"    # slack/email, no page
    INFO     = "info"       # dashboard only


class AlertState(Enum):
    INACTIVE = "inactive"
    PENDING  = "pending"
    FIRING   = "firing"
    RESOLVED = "resolved"


# ─────────────────────────────────────────────
# ALERT RULE
# ─────────────────────────────────────────────

@dataclass
class AlertRule:
    name:       str
    severity:   Severity
    expr_fn:    Callable[[], float]   # returns > 0 when condition true
    for_secs:   float                 # must be true this long before firing
    labels:     Dict[str, str]        # team, service, etc.
    annotations: Dict[str, str] = field(default_factory=dict)  # summary, runbook

    @property
    def runbook(self) -> str:
        return self.annotations.get("runbook", "https://wiki/runbooks/" + self.name)


# ─────────────────────────────────────────────
# FIRING ALERT INSTANCE
# ─────────────────────────────────────────────

@dataclass
class AlertInstance:
    id:          str
    rule:        AlertRule
    state:       AlertState
    first_seen:  float          # when condition first became true
    fired_at:    Optional[float]
    resolved_at: Optional[float]
    value:       float

    def duration_s(self) -> Optional[float]:
        if self.fired_at and self.resolved_at:
            return self.resolved_at - self.fired_at
        return None


# ─────────────────────────────────────────────
# ALERT GROUP (Alertmanager grouping)
# ─────────────────────────────────────────────

@dataclass
class AlertGroup:
    """Group of related alerts routed to the same receiver."""
    group_key:  str
    labels:     Dict[str, str]
    alerts:     List[AlertInstance]
    receiver:   str
    wait_until: float   # group_wait before first notification


# ─────────────────────────────────────────────
# INHIBITION RULE
# ─────────────────────────────────────────────

@dataclass
class InhibitionRule:
    """
    Suppress alerts matching target_matcher when an alert
    matching source_matcher is firing.
    Example: node_down suppresses all pod alerts on that node.
    """
    source_matcher: Dict[str, str]   # e.g. {"alertname": "NodeDown"}
    target_matcher: Dict[str, str]   # e.g. {"severity": "warning"}
    equal:          List[str]        # must match both: e.g. ["instance"]

    def should_inhibit(self, source: AlertInstance,
                       target: AlertInstance) -> bool:
        for k, v in self.source_matcher.items():
            if source.rule.labels.get(k) != v:
                return False
        for k, v in self.target_matcher.items():
            if target.rule.labels.get(k) != v:
                return False
        for eq_label in self.equal:
            if source.rule.labels.get(eq_label) != target.rule.labels.get(eq_label):
                return False
        return True


# ─────────────────────────────────────────────
# SILENCE
# ─────────────────────────────────────────────

@dataclass
class Silence:
    id:         str
    matchers:   Dict[str, str]   # label=value that must match
    start:      float
    end:        float
    created_by: str
    comment:    str

    def is_active(self, ts: Optional[float] = None) -> bool:
        ts = ts or time.time()
        return self.start <= ts <= self.end

    def matches(self, alert: AlertInstance) -> bool:
        if not self.is_active():
            return False
        for k, v in self.matchers.items():
            if alert.rule.labels.get(k) != v:
                return False
        return True


# ─────────────────────────────────────────────
# RECEIVER
# ─────────────────────────────────────────────

@dataclass
class Notification:
    receiver:    str
    alert_ids:   List[str]
    severity:    Severity
    message:     str
    sent_at:     float = field(default_factory=time.time)


class Receiver:
    def __init__(self, name: str, severity_filter: Set[Severity]):
        self.name             = name
        self.severity_filter  = severity_filter
        self.notifications:   List[Notification] = []

    def send(self, group: AlertGroup):
        severities = {a.rule.severity for a in group.alerts}
        if not severities & self.severity_filter:
            return
        msg = (f"[{self.name.upper()}] {len(group.alerts)} alert(s) | "
               f"group={group.group_key} | "
               f"receiver={group.receiver}")
        notif = Notification(
            receiver  = self.name,
            alert_ids = [a.id for a in group.alerts],
            severity  = max(severities, key=lambda s: s.value),
            message   = msg,
        )
        self.notifications.append(notif)
        return notif


# ─────────────────────────────────────────────
# ALERTMANAGER
# ─────────────────────────────────────────────

class Alertmanager:
    """
    Routes, deduplicates, groups, and delivers alerts.
    """

    def __init__(self,
                 group_wait_s:    float = 30.0,
                 group_interval_s: float = 300.0,
                 repeat_interval_s: float = 3600.0):
        self._group_wait      = group_wait_s
        self._group_interval  = group_interval_s
        self._repeat_interval = repeat_interval_s
        self._receivers:   Dict[str, Receiver]      = {}
        self._routes:      List[Dict]               = []   # matchers → receiver
        self._inhibitions: List[InhibitionRule]     = []
        self._silences:    List[Silence]            = []
        self._active_alerts: Dict[str, AlertInstance] = {}
        self._lock         = threading.Lock()

    def add_receiver(self, receiver: Receiver):
        self._receivers[receiver.name] = receiver

    def add_route(self, matchers: Dict[str, str], receiver_name: str,
                  group_by: Optional[List[str]] = None):
        self._routes.append({
            "matchers":  matchers,
            "receiver":  receiver_name,
            "group_by":  group_by or ["alertname"],
        })

    def add_inhibition(self, rule: InhibitionRule):
        self._inhibitions.append(rule)

    def add_silence(self, silence: Silence):
        self._silences.append(silence)

    def _route_alert(self, alert: AlertInstance) -> str:
        for route in self._routes:
            match = all(
                alert.rule.labels.get(k) == v
                for k, v in route["matchers"].items()
            )
            if match:
                return route["receiver"]
        return "default"

    def _is_inhibited(self, target: AlertInstance) -> bool:
        firing = [a for a in self._active_alerts.values()
                  if a.state == AlertState.FIRING]
        for rule in self._inhibitions:
            for source in firing:
                if rule.should_inhibit(source, target):
                    return True
        return False

    def _is_silenced(self, alert: AlertInstance) -> bool:
        return any(s.matches(alert) for s in self._silences)

    def receive(self, alert: AlertInstance):
        """Called when alert state changes."""
        with self._lock:
            self._active_alerts[alert.id] = alert

            if alert.state != AlertState.FIRING:
                return

            if self._is_silenced(alert):
                return
            if self._is_inhibited(alert):
                return

            receiver_name = self._route_alert(alert)
            receiver      = self._receivers.get(receiver_name)
            if receiver:
                group = AlertGroup(
                    group_key  = f"{receiver_name}:{alert.rule.labels.get('service','unknown')}",
                    labels     = alert.rule.labels,
                    alerts     = [alert],
                    receiver   = receiver_name,
                    wait_until = time.time() + self._group_wait,
                )
                notif = receiver.send(group)
                return notif

    def active_alerts(self) -> List[AlertInstance]:
        return [a for a in self._active_alerts.values()
                if a.state == AlertState.FIRING]


# ─────────────────────────────────────────────
# ALERT ENGINE (evaluates rules periodically)
# ─────────────────────────────────────────────

class AlertEngine:
    def __init__(self, alertmanager: Alertmanager,
                 eval_interval_s: float = 15.0):
        self._am        = alertmanager
        self._rules:    List[AlertRule]              = []
        self._states:   Dict[str, AlertInstance]     = {}
        self._eval_interval = eval_interval_s

    def add_rule(self, rule: AlertRule):
        self._rules.append(rule)

    def evaluate(self) -> List[AlertInstance]:
        """Evaluate all rules once. Returns newly fired/resolved."""
        changed = []
        now     = time.time()

        for rule in self._rules:
            try:
                val = rule.expr_fn()
            except Exception:
                val = 0.0

            inst = self._states.get(rule.name)

            if val > 0:
                if inst is None or inst.state == AlertState.INACTIVE:
                    inst = AlertInstance(
                        id          = uuid.uuid4().hex[:8],
                        rule        = rule,
                        state       = AlertState.PENDING,
                        first_seen  = now,
                        fired_at    = None,
                        resolved_at = None,
                        value       = val,
                    )
                    self._states[rule.name] = inst

                inst.value = val

                if inst.state == AlertState.PENDING:
                    elapsed = now - inst.first_seen
                    if elapsed >= rule.for_secs:
                        inst.state    = AlertState.FIRING
                        inst.fired_at = now
                        changed.append(inst)
                        self._am.receive(inst)

            else:
                if inst and inst.state in (AlertState.PENDING, AlertState.FIRING):
                    inst.state       = AlertState.RESOLVED
                    inst.resolved_at = now
                    changed.append(inst)
                    self._am.receive(inst)
                    self._states[rule.name] = None

        return changed


# ─────────────────────────────────────────────
# ON-CALL SCHEDULE
# ─────────────────────────────────────────────

@dataclass
class OnCallShift:
    engineer: str
    start:    float   # epoch seconds
    end:      float


class OnCallSchedule:
    """Weekly rotating on-call schedule."""

    def __init__(self, rotation: List[str], shift_duration_s: float = 7 * 86400):
        self._rotation = rotation
        self._shift    = shift_duration_s

    def current_on_call(self, ts: Optional[float] = None) -> str:
        ts    = ts or time.time()
        index = int(ts / self._shift) % len(self._rotation)
        return self._rotation[index]

    def next_shift_in_s(self, ts: Optional[float] = None) -> float:
        ts         = ts or time.time()
        shift_start = int(ts / self._shift) * self._shift
        return shift_start + self._shift - ts

    def upcoming_shifts(self, n: int = 4, ts: Optional[float] = None
                        ) -> List[OnCallShift]:
        ts = ts or time.time()
        shifts = []
        for i in range(n):
            start_abs = (int(ts / self._shift) + i) * self._shift
            eng       = self._rotation[(int(ts / self._shift) + i) % len(self._rotation)]
            shifts.append(OnCallShift(eng, start_abs, start_abs + self._shift))
        return shifts


# ─────────────────────────────────────────────
# ESCALATION POLICY
# ─────────────────────────────────────────────

@dataclass
class EscalationLevel:
    contact:      str         # engineer name or team
    delay_s:      float       # wait this long before escalating further
    notify_via:   List[str]   # ["pagerduty", "sms", "email"]


class EscalationPolicy:
    def __init__(self, name: str, levels: List[EscalationLevel]):
        self.name   = name
        self.levels = levels

    def escalation_chain(self) -> List[str]:
        return [f"t+{l.delay_s:.0f}s → {l.contact} via {'/'.join(l.notify_via)}"
                for l in self.levels]


# ─────────────────────────────────────────────
# INCIDENT METRICS
# ─────────────────────────────────────────────

@dataclass
class Incident:
    id:          str
    title:       str
    severity:    Severity
    created_at:  float
    acked_at:    Optional[float] = None
    resolved_at: Optional[float] = None

    @property
    def mttd_s(self) -> float:
        """Time from alert fire to incident creation (proxy for MTTD)."""
        return 0.0    # would be fire_time - issue_start in real system

    @property
    def mtta_s(self) -> Optional[float]:
        if self.acked_at:
            return self.acked_at - self.created_at
        return None

    @property
    def mttr_s(self) -> Optional[float]:
        if self.resolved_at:
            return self.resolved_at - self.created_at
        return None


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_alerting():
    print("=" * 65)
    print("ALERTING AND ON-CALL MANAGEMENT")
    print("=" * 65)

    # ── Setup Alertmanager ────────────────────
    am = Alertmanager(group_wait_s=0, group_interval_s=60)

    pagerduty = Receiver("pagerduty", {Severity.CRITICAL, Severity.HIGH})
    slack     = Receiver("slack",     {Severity.WARNING, Severity.INFO,
                                       Severity.HIGH, Severity.CRITICAL})
    am.add_receiver(pagerduty)
    am.add_receiver(slack)

    # Routes: critical → pagerduty; else → slack
    am.add_route({"severity": "critical"}, "pagerduty")
    am.add_route({"severity": "high"},     "pagerduty")
    am.add_route({},                       "slack")

    # Inhibition: if NodeDown fires, suppress pod-level warnings
    am.add_inhibition(InhibitionRule(
        source_matcher={"alertname": "NodeDown"},
        target_matcher={"severity": "warning"},
        equal=["zone"],
    ))

    # ── Alert Rules ───────────────────────────
    print("\n[1] ALERT RULES")
    print("─" * 55)

    # Simulated metrics
    _error_rate  = [0.0]
    _latency_p99 = [0.0]
    _disk_free   = [0.40]

    rules = [
        AlertRule(
            "HighErrorRate", Severity.CRITICAL,
            expr_fn=lambda: max(0, _error_rate[0] - 0.05),
            for_secs=0,
            labels={"service": "api", "team": "backend", "severity": "critical",
                    "alertname": "HighErrorRate"},
            annotations={"summary": "Error rate > 5%",
                         "runbook": "https://wiki/runbooks/high-error-rate"},
        ),
        AlertRule(
            "HighLatency", Severity.HIGH,
            expr_fn=lambda: max(0, _latency_p99[0] - 1.0),
            for_secs=0,
            labels={"service": "api", "team": "backend", "severity": "high",
                    "alertname": "HighLatency"},
            annotations={"summary": "p99 latency > 1s"},
        ),
        AlertRule(
            "DiskFilling", Severity.WARNING,
            expr_fn=lambda: max(0, 0.15 - _disk_free[0]),
            for_secs=0,
            labels={"service": "storage", "team": "infra", "severity": "warning",
                    "alertname": "DiskFilling"},
            annotations={"summary": "Disk < 15% free"},
        ),
    ]

    engine = AlertEngine(am)
    for rule in rules:
        engine.add_rule(rule)
        print(f"  [{rule.severity.value:<10}] {rule.name:<20} for={rule.for_secs}s  "
              f"runbook={rule.runbook}")

    # ── Evaluate: no issues ───────────────────
    print("\n[2] EVALUATION CYCLE — NO ISSUES")
    print("─" * 55)

    changed = engine.evaluate()
    print(f"  Active alerts: {len(am.active_alerts())} (expected 0)")

    # ── Inject errors ─────────────────────────
    print("\n[3] ERROR RATE SPIKES — ALERT FIRES")
    print("─" * 55)

    _error_rate[0]  = 0.12   # 12% errors
    _latency_p99[0] = 1.8    # 1.8s

    changed = engine.evaluate()
    for inst in changed:
        if inst.state == AlertState.FIRING:
            print(f"  FIRING: {inst.rule.name}  severity={inst.rule.severity.value}  "
                  f"value={inst.value:.3f}")

    print(f"\n  PagerDuty notifications: {len(pagerduty.notifications)}")
    for notif in pagerduty.notifications:
        print(f"    [{notif.severity.value}] {notif.message}")

    print(f"\n  Slack notifications:     {len(slack.notifications)}")

    # ── Silence maintenance ────────────────────
    print("\n[4] SILENCE (maintenance window)")
    print("─" * 55)

    silence = Silence(
        id         = "sil-001",
        matchers   = {"team": "infra"},
        start      = time.time() - 1,
        end        = time.time() + 3600,
        created_by = "alice",
        comment    = "DB maintenance window",
    )
    am.add_silence(silence)

    _disk_free[0] = 0.05    # disk almost full — would trigger DiskFilling
    prev_count    = len(slack.notifications)
    changed       = engine.evaluate()
    new_slack     = len(slack.notifications) - prev_count

    print(f"  Silence active for team=infra")
    print(f"  DiskFilling fires? new_slack_notifications={new_slack} (0 = silenced)")

    # ── On-Call Schedule ──────────────────────
    print("\n[5] ON-CALL ROTATION")
    print("─" * 55)

    schedule = OnCallSchedule(
        ["alice", "bob", "charlie", "diana"],
        shift_duration_s=7 * 86400,
    )

    # Use a fixed reference time for reproducible output
    ref_ts = 1700000000.0
    print(f"  Current on-call: {schedule.current_on_call(ref_ts)}")
    print(f"  Next shift in:   {schedule.next_shift_in_s(ref_ts)/3600:.1f} hours")
    print("\n  Upcoming shifts:")
    for shift in schedule.upcoming_shifts(4, ref_ts):
        print(f"    {shift.engineer:<10} (next {7:.0f}d window)")

    # ── Escalation Policy ─────────────────────
    print("\n[6] ESCALATION POLICY")
    print("─" * 55)

    policy = EscalationPolicy("backend-critical", [
        EscalationLevel("alice (primary)",   0,   ["pagerduty", "sms"]),
        EscalationLevel("bob (secondary)",  300,  ["pagerduty", "sms"]),
        EscalationLevel("eng-manager",      900,  ["pagerduty", "phone"]),
    ])
    print(f"  Policy: {policy.name}")
    for step in policy.escalation_chain():
        print(f"    {step}")

    # ── Incident Metrics ──────────────────────
    print("\n[7] INCIDENT METRICS (MTTD / MTTA / MTTR)")
    print("─" * 55)

    now = time.time()
    incidents = [
        Incident("INC-001", "Database latency spike", Severity.CRITICAL,
                 now - 1800, now - 1780, now - 1500),   # 20s ack, 5min resolve
        Incident("INC-002", "Payment service 500s",   Severity.CRITICAL,
                 now - 3600, now - 3540, now - 3000),   # 60s ack, 10min resolve
        Incident("INC-003", "Auth service degraded",  Severity.HIGH,
                 now - 7200, now - 7000, now - 6300),   # 200s ack, 15min resolve
    ]

    print(f"  {'ID':<10} {'Title':<30} {'MTTA':>8}  {'MTTR':>8}")
    print("  " + "─" * 60)
    for inc in incidents:
        mtta = f"{inc.mtta_s:.0f}s" if inc.mtta_s else "—"
        mttr = f"{inc.mttr_s/60:.1f}m" if inc.mttr_s else "—"
        print(f"  {inc.id:<10} {inc.title:<30} {mtta:>8}  {mttr:>8}")

    mttrs = [i.mttr_s for i in incidents if i.mttr_s]
    mttas = [i.mtta_s for i in incidents if i.mtta_s]
    print(f"\n  p50 MTTA: {sorted(mttas)[len(mttas)//2]:.0f}s")
    print(f"  p50 MTTR: {sorted(mttrs)[len(mttrs)//2]/60:.1f}m")


if __name__ == "__main__":
    demonstrate_alerting()
