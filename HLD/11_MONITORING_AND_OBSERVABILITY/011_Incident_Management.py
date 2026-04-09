"""
INCIDENT MANAGEMENT
=====================

Problem Statement:
When systems fail, confusion and lack of process make outages longer.
Incident management provides a structured response: detect, alert,
coordinate, resolve, and learn.

Incident Severity Levels:
  SEV1 / P1:  Complete service outage. All hands on deck. Customer impact.
              Examples: payment service down, login broken for all users.
  SEV2 / P2:  Significant degradation. >10% users affected.
              Examples: search returning errors, checkout slow.
  SEV3 / P3:  Minor issue. <1% users affected, workaround exists.
  SEV4 / P4:  Cosmetic / non-functional issues. Fix in next release.

Incident Lifecycle:
  Detected → Acknowledged → Investigating → Identified → Fixing → Resolved → Post-mortem

Roles:
  Incident Commander (IC):  Coordinates response. Makes decisions.
                            One IC per incident.
  Technical Lead (TL):      Diagnoses and implements fixes.
  Communications Lead (CL): Updates status page, emails, stakeholders.
  Subject Matter Expert (SME): Service owner, database expert, etc.

Status Page:
  Public-facing page with per-component status and incident history.
  Components: API, Database, CDN, Authentication, Payment.
  Statuses: Operational, Degraded, Partial Outage, Major Outage.
  Tools: Statuspage.io (Atlassian), Instatus, Cachet (open-source).

Post-Mortem (Blameless):
  Written within 48h of resolution.
  Sections: summary, timeline, root cause, contributing factors,
            detection (MTTD), resolution (MTTR), action items.
  Key principle: BLAMELESS. Systems fail; people make mistakes.
  Goal: improve systems and processes, not assign blame.
  Action items: specific, assignable, time-bound. Not vague.

Chaos Engineering:
  Proactively test resilience: kill random pods (Chaos Monkey),
  inject latency (Chaos Mesh), cut network segments.
  "GameDay": scheduled failure injection with on-call team.
  Reduces MTTD/MTTR by building muscle memory.

MTTR Reduction Techniques:
  - Runbooks: step-by-step remediation for common failures.
  - Automated rollback: new deploy → if error rate spikes → auto-revert.
  - Feature flags: disable problematic features without deploys.
  - Pre-baked dashboards: link from alert → relevant Grafana dashboard.
  - War room channel: dedicated Slack channel per incident.
"""

from __future__ import annotations

import time
import uuid
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


# ─────────────────────────────────────────────
# SEVERITY AND STATUS
# ─────────────────────────────────────────────

class IncidentSeverity(Enum):
    SEV1 = 1   # Critical: complete outage
    SEV2 = 2   # High: significant degradation
    SEV3 = 3   # Medium: minor impact
    SEV4 = 4   # Low: cosmetic

    @property
    def label(self) -> str:
        labels = {1: "SEV1 (Critical)", 2: "SEV2 (High)",
                  3: "SEV3 (Medium)",   4: "SEV4 (Low)"}
        return labels[self.value]

    @property
    def response_time_min(self) -> int:
        return {1: 5, 2: 15, 3: 60, 4: 480}[self.value]


class IncidentStatus(Enum):
    DETECTED      = "detected"
    ACKNOWLEDGED  = "acknowledged"
    INVESTIGATING = "investigating"
    IDENTIFIED    = "identified"
    FIXING        = "fixing"
    RESOLVED      = "resolved"
    CLOSED        = "closed"


# ─────────────────────────────────────────────
# COMPONENT STATUS (for status page)
# ─────────────────────────────────────────────

class ComponentStatus(Enum):
    OPERATIONAL    = "operational"
    DEGRADED       = "degraded"
    PARTIAL_OUTAGE = "partial_outage"
    MAJOR_OUTAGE   = "major_outage"

    @property
    def icon(self) -> str:
        return {
            "operational":    "✓",
            "degraded":       "~",
            "partial_outage": "!",
            "major_outage":   "✗",
        }[self.value]


# ─────────────────────────────────────────────
# INCIDENT TIMELINE EVENT
# ─────────────────────────────────────────────

@dataclass
class TimelineEvent:
    timestamp:  float
    author:     str
    event_type: str   # note, status_change, role_assigned, action_taken
    content:    str

    def format(self) -> str:
        elapsed = int(time.time() - self.timestamp)
        return (f"[{self.event_type.upper()}] by {self.author}: {self.content}")


# ─────────────────────────────────────────────
# INCIDENT
# ─────────────────────────────────────────────

@dataclass
class Incident:
    id:          str
    title:       str
    severity:    IncidentSeverity
    status:      IncidentStatus    = IncidentStatus.DETECTED
    created_at:  float             = field(default_factory=time.time)
    acked_at:    Optional[float]   = None
    resolved_at: Optional[float]   = None
    commander:   Optional[str]     = None
    tech_lead:   Optional[str]     = None
    comms_lead:  Optional[str]     = None
    affected_components: List[str] = field(default_factory=list)
    timeline:    List[TimelineEvent] = field(default_factory=list)
    action_items: List["ActionItem"] = field(default_factory=list)
    root_cause:  Optional[str]     = None
    slack_channel: Optional[str]   = None

    @property
    def mtta_s(self) -> Optional[float]:
        return (self.acked_at - self.created_at) if self.acked_at else None

    @property
    def mttr_s(self) -> Optional[float]:
        return (self.resolved_at - self.created_at) if self.resolved_at else None

    def add_event(self, author: str, event_type: str, content: str):
        self.timeline.append(TimelineEvent(time.time(), author, event_type, content))

    def transition(self, new_status: IncidentStatus, author: str):
        old = self.status
        self.status = new_status
        self.add_event(author, "status_change",
                       f"{old.value} → {new_status.value}")

        if new_status == IncidentStatus.ACKNOWLEDGED and not self.acked_at:
            self.acked_at = time.time()
        if new_status == IncidentStatus.RESOLVED and not self.resolved_at:
            self.resolved_at = time.time()


# ─────────────────────────────────────────────
# ACTION ITEM
# ─────────────────────────────────────────────

@dataclass
class ActionItem:
    id:          str
    title:       str
    owner:       str
    due_date:    float
    priority:    str   # immediate, short_term, long_term
    status:      str   = "open"   # open, in_progress, done
    ticket_url:  Optional[str] = None


# ─────────────────────────────────────────────
# STATUS PAGE
# ─────────────────────────────────────────────

@dataclass
class StatusUpdate:
    timestamp: float
    author:    str
    message:   str
    status:    ComponentStatus


class StatusPage:
    """
    Public-facing status page.
    Tracks per-component status and incident history.
    """

    def __init__(self, components: List[str]):
        self._components:  Dict[str, ComponentStatus] = {
            c: ComponentStatus.OPERATIONAL for c in components
        }
        self._updates:     Dict[str, List[StatusUpdate]] = {c: [] for c in components}
        self._incidents:   List[Incident] = []

    def update_component(self, component: str, status: ComponentStatus,
                         author: str, message: str):
        self._components[component] = status
        self._updates[component].append(
            StatusUpdate(time.time(), author, message, status))

    def overall_status(self) -> ComponentStatus:
        statuses = list(self._components.values())
        if ComponentStatus.MAJOR_OUTAGE in statuses:
            return ComponentStatus.MAJOR_OUTAGE
        if ComponentStatus.PARTIAL_OUTAGE in statuses:
            return ComponentStatus.PARTIAL_OUTAGE
        if ComponentStatus.DEGRADED in statuses:
            return ComponentStatus.DEGRADED
        return ComponentStatus.OPERATIONAL

    def post_incident(self, incident: Incident):
        self._incidents.append(incident)

    def render(self) -> str:
        lines = ["=== STATUS PAGE ==="]
        overall = self.overall_status()
        lines.append(f"Overall: {overall.icon} {overall.value.upper()}")
        lines.append("")
        lines.append("Components:")
        for comp, status in self._components.items():
            lines.append(f"  {status.icon} {comp:<20} {status.value}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# POST-MORTEM
# ─────────────────────────────────────────────

@dataclass
class PostMortem:
    incident_id:     str
    title:           str
    severity:        IncidentSeverity
    summary:         str
    timeline:        List[str]           # human-readable timeline entries
    root_cause:      str
    contributing_factors: List[str]
    what_went_well:  List[str]
    action_items:    List[ActionItem]
    mttd_s:          float
    mttr_s:          float

    def render(self) -> str:
        lines = [
            f"POST-MORTEM: {self.title}",
            f"Severity: {self.severity.label}",
            f"",
            f"SUMMARY",
            f"  {self.summary}",
            f"",
            f"KEY METRICS",
            f"  MTTD: {self.mttd_s:.0f}s ({self.mttd_s/60:.1f}m)",
            f"  MTTR: {self.mttr_s:.0f}s ({self.mttr_s/60:.1f}m)",
            f"",
            f"TIMELINE",
        ]
        for t in self.timeline:
            lines.append(f"  {t}")
        lines += [
            f"",
            f"ROOT CAUSE",
            f"  {self.root_cause}",
            f"",
            f"CONTRIBUTING FACTORS",
        ]
        for f in self.contributing_factors:
            lines.append(f"  - {f}")
        lines += [
            f"",
            f"WHAT WENT WELL",
        ]
        for w in self.what_went_well:
            lines.append(f"  + {w}")
        lines += [
            f"",
            f"ACTION ITEMS",
        ]
        for ai in self.action_items:
            due = time.strftime("%Y-%m-%d", time.localtime(ai.due_date))
            lines.append(f"  [{ai.priority.upper()}] {ai.title}  "
                         f"owner={ai.owner}  due={due}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# INCIDENT MANAGER
# ─────────────────────────────────────────────

class IncidentManager:
    def __init__(self, status_page: StatusPage):
        self._sp         = status_page
        self._incidents: Dict[str, Incident] = {}

    def declare(self, title: str, severity: IncidentSeverity,
                components: List[str]) -> Incident:
        inc = Incident(
            id                  = f"INC-{uuid.uuid4().hex[:6].upper()}",
            title               = title,
            severity            = severity,
            affected_components = components,
            slack_channel       = f"#incident-{uuid.uuid4().hex[:6]}",
        )
        self._incidents[inc.id] = inc
        for comp in components:
            status = (ComponentStatus.MAJOR_OUTAGE
                      if severity == IncidentSeverity.SEV1
                      else ComponentStatus.DEGRADED)
            self._sp.update_component(comp, status, "incident-bot",
                                      f"Incident {inc.id} declared")
        inc.add_event("system", "incident_created",
                      f"Incident declared: {title}")
        return inc

    def assign_roles(self, inc: Incident, commander: str,
                     tech_lead: str, comms_lead: str):
        inc.commander  = commander
        inc.tech_lead  = tech_lead
        inc.comms_lead = comms_lead
        inc.add_event("system", "role_assigned",
                      f"IC={commander} TL={tech_lead} CL={comms_lead}")

    def resolve(self, inc: Incident, root_cause: str, author: str):
        inc.root_cause = root_cause
        inc.transition(IncidentStatus.RESOLVED, author)
        for comp in inc.affected_components:
            self._sp.update_component(
                comp, ComponentStatus.OPERATIONAL, author,
                f"Incident {inc.id} resolved")
        self._sp.post_incident(inc)

    def write_postmortem(self, inc: Incident) -> PostMortem:
        mttd = (inc.acked_at or inc.created_at) - inc.created_at
        mttr = (inc.resolved_at or time.time()) - inc.created_at

        return PostMortem(
            incident_id   = inc.id,
            title         = inc.title,
            severity      = inc.severity,
            summary       = (f"Incident {inc.id}: {inc.title}. "
                             f"Duration: {mttr/60:.1f}min. "
                             f"Root cause: {inc.root_cause}"),
            timeline      = [f"{ev.format()}" for ev in inc.timeline],
            root_cause    = inc.root_cause or "Under investigation",
            contributing_factors = [
                "No alerting on p99 latency (only error rate)",
                "Runbook was out of date",
                "On-call engineer was not familiar with this service",
            ],
            what_went_well = [
                "Alert fired within 3 minutes of incident start",
                "Rollback was automated and took < 2 minutes",
                "Status page was updated within 5 minutes",
            ],
            action_items  = inc.action_items,
            mttd_s        = mttd,
            mttr_s        = mttr,
        )

    def open_incidents(self) -> List[Incident]:
        return [i for i in self._incidents.values()
                if i.status not in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED)]


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_incident():
    print("=" * 65)
    print("INCIDENT MANAGEMENT")
    print("=" * 65)

    # ── Setup ─────────────────────────────────
    components = ["API Gateway", "Authentication", "Payment", "Database", "CDN"]
    sp  = StatusPage(components)
    mgr = IncidentManager(sp)

    # ── Declare Incident ──────────────────────
    print("\n[1] INCIDENT DECLARED (SEV1: Payment Outage)")
    print("─" * 55)

    inc = mgr.declare(
        "Payment service returning 500 for all checkout requests",
        IncidentSeverity.SEV1,
        components=["Payment", "API Gateway"],
    )
    print(f"  Incident ID:   {inc.id}")
    print(f"  Severity:      {inc.severity.label}")
    print(f"  Slack channel: {inc.slack_channel}")
    print(f"  Response time: within {inc.severity.response_time_min}min")

    # ── Assign Roles ──────────────────────────
    print("\n[2] ROLE ASSIGNMENT")
    print("─" * 55)

    mgr.assign_roles(inc, "alice (IC)", "bob (TL)", "charlie (CL)")
    inc.add_event("alice", "note",
                  "War room open in Zoom. IC alice leading.")
    inc.add_event("bob", "action_taken",
                  "Pulling logs from payment-service pods")
    inc.add_event("charlie", "note",
                  "Status page updated: Payment MAJOR_OUTAGE")

    print(f"  IC: {inc.commander}")
    print(f"  TL: {inc.tech_lead}")
    print(f"  CL: {inc.comms_lead}")

    # ── Investigation ─────────────────────────
    print("\n[3] INVESTIGATION TIMELINE")
    print("─" * 55)

    inc.transition(IncidentStatus.INVESTIGATING, "alice")
    inc.add_event("bob", "note",
                  "Found: DB connection pool exhausted (max=100, current=100)")
    inc.add_event("bob", "note",
                  "Root cause: new deploy at 14:32 added N+1 query in checkout")
    inc.transition(IncidentStatus.IDENTIFIED, "bob")

    # ── Fix and Resolve ───────────────────────
    inc.add_event("bob", "action_taken", "Rolling back deploy v2.5.1 → v2.5.0")
    inc.transition(IncidentStatus.FIXING, "bob")
    inc.add_event("bob", "note",
                  "Rollback complete. Payment errors dropping.")
    inc.add_event("alice", "note",
                  "Error rate back to 0%. Resolving incident.")

    # Simulate elapsed time
    inc.acked_at    = inc.created_at + 180   # 3min to ack
    inc.resolved_at = inc.created_at + 1500  # 25min to resolve

    mgr.resolve(inc, "N+1 query in checkout introduced by v2.5.1 deploy "
                "exhausted DB connection pool", "alice")

    print(f"  Status:       {inc.status.value}")
    print(f"  Root cause:   {inc.root_cause}")
    print(f"  MTTA:         {inc.mtta_s:.0f}s ({inc.mtta_s/60:.1f}min)")
    print(f"  MTTR:         {inc.mttr_s:.0f}s ({inc.mttr_s/60:.1f}min)")

    # ── Status Page ───────────────────────────
    print("\n[4] STATUS PAGE")
    print("─" * 55)

    print(sp.render())

    # ── Post-Mortem ───────────────────────────
    print("\n[5] POST-MORTEM")
    print("─" * 55)

    # Add action items
    now = time.time()
    inc.action_items = [
        ActionItem(
            id="AI-001", title="Add N+1 query linter to CI pipeline",
            owner="bob", due_date=now + 7*86400,
            priority="immediate", ticket_url="https://jira/PLAT-1234"),
        ActionItem(
            id="AI-002", title="Add DB connection pool metric alert",
            owner="alice", due_date=now + 3*86400,
            priority="immediate"),
        ActionItem(
            id="AI-003", title="Update payment service runbook",
            owner="charlie", due_date=now + 14*86400,
            priority="short_term"),
        ActionItem(
            id="AI-004", title="Implement automated rollback on error rate spike",
            owner="bob", due_date=now + 30*86400,
            priority="long_term"),
    ]

    pm = mgr.write_postmortem(inc)
    print(pm.render())

    # ── Incident Severity Guide ───────────────
    print("\n\n[6] SEVERITY GUIDELINES")
    print("─" * 55)

    examples = [
        (IncidentSeverity.SEV1, "Payment down for all users",     "5min",  "IC + TL + CL required"),
        (IncidentSeverity.SEV2, "Search returning errors for 20%","15min", "TL required"),
        (IncidentSeverity.SEV3, "Bulk export slow (>10s)",        "1h",    "On-call engineer"),
        (IncidentSeverity.SEV4, "Admin UI typo",                  "8h",    "Ticket only, no page"),
    ]
    print(f"  {'Sev':<8} {'Example':<40} {'Response':>8}  {'Roles'}")
    print("  " + "─" * 80)
    for sev, example, resp, roles in examples:
        print(f"  {sev.value.value:<8} {example:<40} {resp:>8}  {roles}")

    # ── MTTR Improvement Techniques ───────────
    print("\n[7] MTTR REDUCTION TECHNIQUES")
    print("─" * 55)

    techniques = [
        ("Runbooks",           "Step-by-step guides for common failures"),
        ("Auto rollback",      "Error spike after deploy → auto-revert"),
        ("Feature flags",      "Disable features without deploys"),
        ("Pre-built dashboards","Alert → Grafana link in notification"),
        ("War room channel",   "Dedicated Slack channel per incident"),
        ("GameDay drills",     "Scheduled failure injection to build muscle memory"),
        ("On-call shadowing",  "New engineers shadow for 2 rotations before primary"),
        ("Runbook automation", "Runbooks as code (Ansible/scripts)"),
    ]
    for tech, desc in techniques:
        print(f"  {tech:<22} {desc}")


if __name__ == "__main__":
    demonstrate_incident()
