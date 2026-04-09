"""
SYSTEM DESIGN METHODOLOGY
===========================

Problem Statement:
A structured framework for approaching any system design problem — whether
in a 45-minute interview or a multi-day architecture session. Without a
framework, designers jump to solutions before understanding the problem.

45-Minute Interview Framework:
  ┌──────────────────────────────────────────────────────┐
  │  Phase              │ Time  │ Goal                   │
  ├──────────────────────────────────────────────────────┤
  │  1. Requirements    │  5min │ Clarify scope           │
  │  2. Estimation      │  5min │ Scale/capacity numbers  │
  │  3. High-Level HLD  │ 10min │ Major components        │
  │  4. Deep Dive       │ 15min │ Critical component      │
  │  5. Trade-offs      │ 10min │ Justify decisions       │
  └──────────────────────────────────────────────────────┘

Common Mistakes:
  ❌ Jumping to database choice without understanding scale
  ❌ No capacity estimation
  ❌ Ignoring non-functional requirements
  ❌ Over-engineering for day-1 traffic
  ❌ Not discussing trade-offs
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time


class InterviewPhase(Enum):
    REQUIREMENTS = ("Requirements",  5)
    ESTIMATION   = ("Estimation",    5)
    HLD          = ("High-Level Design", 10)
    DEEP_DIVE    = ("Deep Dive",     15)
    TRADE_OFFS   = ("Trade-Offs",    10)

    def __init__(self, label: str, budget_min: int):
        self.label      = label
        self.budget_min = budget_min


@dataclass
class DesignNote:
    phase   : InterviewPhase
    content : str
    is_key  : bool = False   # mark important decisions


@dataclass
class ComponentDecision:
    component  : str
    chosen     : str
    rejected   : List[str]
    reason     : str


class DesignFramework:
    """
    Structured step-by-step framework for each design phase.
    """

    PHASE_QUESTIONS = {
        InterviewPhase.REQUIREMENTS: [
            "What are the core features? (ask interviewer to prioritise)",
            "How many users? DAU? Writes/reads per day?",
            "What is the expected latency SLA?",
            "What is the availability requirement?",
            "Any special constraints: mobile-first? offline support? compliance?",
        ],
        InterviewPhase.ESTIMATION: [
            "Calculate write QPS = daily_writes / 86400",
            "Calculate read QPS = daily_reads  / 86400 (usually read:write = 10:1 to 100:1)",
            "Estimate storage: bytes_per_record × records_per_day × retention_days",
            "Estimate bandwidth: bytes_per_request × QPS",
            "Estimate cache size: 20% of daily data (80/20 rule)",
        ],
        InterviewPhase.HLD: [
            "Draw the main components: clients, LB, services, DB, cache, queue",
            "Define APIs: key endpoints with request/response",
            "Choose database type (SQL vs NoSQL) based on scale + access pattern",
            "Add caching layer (Redis) for read-heavy paths",
            "Add message queue for async operations",
        ],
        InterviewPhase.DEEP_DIVE: [
            "Pick the most interesting / hardest component to deep-dive",
            "Discuss data model: schema, partition keys, indexes",
            "Walk through critical path: request → response step by step",
            "Address the hardest non-functional requirement",
            "Handle edge cases: what happens when X fails?",
        ],
        InterviewPhase.TRADE_OFFS: [
            "Consistency vs Availability — what did you choose and why?",
            "Discuss alternative approaches you considered and rejected",
            "Scale bottlenecks: what breaks first at 10× traffic?",
            "Cost vs performance trade-offs",
            "What would you do differently with more time?",
        ],
    }

    @classmethod
    def guide(cls, phase: InterviewPhase):
        print(f"\n  PHASE: {phase.label} ({phase.budget_min} min budget)")
        for q in cls.PHASE_QUESTIONS[phase]:
            print(f"    • {q}")


class InterviewCoach:
    """Provides real-time coaching and flags common mistakes."""

    COMMON_MISTAKES = [
        ("Jumping straight to DB choice",       "Requirements",  "Ask about scale first — SQL or NoSQL depends on it"),
        ("No capacity estimation",              "Estimation",    "Always do quick math: QPS, storage, bandwidth"),
        ("Designing for day-1 traffic only",    "HLD",           "Ask about 5× growth scenario"),
        ("Ignoring cache layer",                "HLD",           "Add Redis for any read-heavy path"),
        ("Not handling failure cases",          "Deep Dive",     "What happens when DB primary goes down?"),
        ("No trade-off discussion",             "Trade-Offs",    "Always compare your choice with alternatives"),
        ("Over-engineering early",              "HLD",           "Start simple, scale only what's needed"),
    ]

    @classmethod
    def print_mistakes(cls):
        print("\n  COMMON INTERVIEW MISTAKES:")
        for mistake, phase, fix in cls.COMMON_MISTAKES:
            print(f"  ❌ {mistake}")
            print(f"     → [{phase}] {fix}")


class SystemDesignInterview:
    """
    Simulates a structured system design interview session.
    """

    def __init__(self, system: str, total_minutes: int = 45):
        self.system         = system
        self.total_minutes  = total_minutes
        self.notes          : List[DesignNote] = []
        self.decisions      : List[ComponentDecision] = []
        self._start_time    = time.time()
        self._phase_times   : Dict[str, float] = {}

    def _elapsed_min(self) -> float:
        return (time.time() - self._start_time) / 60.0

    def run_phase(self, phase: InterviewPhase, notes: List[str],
                  time_spent_min: float = None):
        t = time_spent_min or phase.budget_min
        elapsed = sum(self._phase_times.values())
        self._phase_times[phase.label] = t
        budget_ok = "✅" if t <= phase.budget_min else "⚠ over budget"
        print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  [{phase.label.upper()}]  {t}min  {budget_ok}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for note in notes:
            is_key = note.startswith("*")
            clean  = note.lstrip("*").strip()
            self.notes.append(DesignNote(phase, clean, is_key))
            prefix = "  ⭐ " if is_key else "  • "
            print(f"{prefix}{clean}")

    def decide(self, component: str, chosen: str,
               rejected: List[str], reason: str):
        self.decisions.append(ComponentDecision(component, chosen, rejected, reason))

    def summary(self):
        total_used = sum(self._phase_times.values())
        print(f"\n  {'='*55}")
        print(f"  DESIGN SUMMARY: {self.system}")
        print(f"  {'='*55}")
        print(f"\n  Time usage:")
        for phase, t in self._phase_times.items():
            bar = "█" * int(t)
            print(f"    {phase:<20} {bar} {t}min")
        print(f"    {'TOTAL':<20} {total_used}min / {self.total_minutes}min")

        print(f"\n  Key Decisions ({len(self.decisions)}):")
        for d in self.decisions:
            print(f"    ▶ {d.component}: chose '{d.chosen}'")
            print(f"      Reason   : {d.reason}")
            print(f"      Rejected : {', '.join(d.rejected)}")


def demonstrate_system_design_methodology():
    print("=" * 65)
    print("SYSTEM DESIGN METHODOLOGY")
    print("System: Design WhatsApp (45-minute interview)")
    print("=" * 65)

    # ── Phase Guide ───────────────────────────
    print("\n[FRAMEWORK QUICK REFERENCE]")
    for phase in InterviewPhase:
        DesignFramework.guide(phase)

    # ── Simulated Interview Session ───────────
    print("\n\n" + "=" * 65)
    print("SIMULATED INTERVIEW: Design WhatsApp")
    print("=" * 65)

    session = SystemDesignInterview("WhatsApp", 45)

    session.run_phase(InterviewPhase.REQUIREMENTS, [
        "Users send 1:1 messages (text, images, video)",
        "Group chats (up to 256 members)",
        "* Delivery receipts: sent ✓, delivered ✓✓, read ✓✓ (blue)",
        "Online/last-seen presence",
        "Push notifications when offline",
        "End-to-end encryption",
        "OUT OF SCOPE: Stories, Payments, Business API",
    ], time_spent_min=5)

    session.run_phase(InterviewPhase.ESTIMATION, [
        "2B users, 100M DAU, 100B messages/day",
        "* Read QPS ≈ 1.16M/sec (100B / 86400)",
        "Write QPS ≈ same (each send = one write)",
        "Storage: 100B msgs × 100 bytes = 10TB/day → 3.65PB/year",
        "Media: 30% msgs have image (300KB avg) → 3PB/day (need CDN!)",
        "Connections: 100M DAU × keep-alive WebSocket = 100M connections",
    ], time_spent_min=5)

    session.run_phase(InterviewPhase.HLD, [
        "Client → WebSocket Gateway → Chat Service → Message DB",
        "Chat Service publishes to Message Queue (Kafka)",
        "Fan-out service delivers to recipient's device or inbox",
        "Redis: online presence, recent message cache",
        "Object Store (S3): images and video",
        "Push Notification Service (APNS/FCM) for offline users",
        "* Separate read path (timeline) from write path (send)",
    ], time_spent_min=10)

    session.run_phase(InterviewPhase.DEEP_DIVE, [
        "* Deep dive: Message Delivery Flow",
        "Send: Client → WebSocket → Chat Service → Cassandra (write) → Kafka",
        "Deliver: Kafka consumer → check if recipient online → push or queue",
        "Online delivery: WebSocket push to recipient",
        "Offline delivery: store in inbox table; push notification via APNS/FCM",
        "Receipt flow: client sends ack → update message status in Cassandra",
        "* Message ordering: Cassandra partition by (chat_id), cluster by (timestamp, msg_id)",
        "Group chat: fan-out to all members' inboxes via Kafka consumer group",
        "E2E encryption: Diffie-Hellman key exchange on session start; server stores ciphertext only",
    ], time_spent_min=15)

    session.run_phase(InterviewPhase.TRADE_OFFS, [
        "* WebSocket vs HTTP polling: WebSocket = 1 persistent conn; polling = 10K req/min per user",
        "Cassandra vs MySQL: Cassandra handles 1M writes/sec at scale; MySQL would need heavy sharding",
        "Push vs pull fan-out for group chats: push (fan-out on write) for small groups; pull for large",
        "E2E encryption limits server features (spam detection, backup search)",
        "Bottleneck at 10× scale: WebSocket gateway → solve with connection sharding by user_id",
    ], time_spent_min=10)

    session.decide("Database",       "Cassandra",  ["MySQL", "DynamoDB"],   "Write-heavy (1M writes/sec), partition by chat_id, ordered reads")
    session.decide("Real-time layer","WebSocket",  ["Long-polling", "SSE"], "Full-duplex, efficient for 100M concurrent connections")
    session.decide("Message Queue",  "Kafka",      ["RabbitMQ", "SQS"],     "High throughput, replay, partitioned by chat_id for ordering")
    session.decide("Media Storage",  "S3 + CDN",   ["DB blobs", "HDFS"],   "S3 for durability, CDN for global low-latency delivery")

    session.summary()

    # ── Common Mistakes ───────────────────────
    print()
    InterviewCoach.print_mistakes()


if __name__ == "__main__":
    demonstrate_system_design_methodology()
