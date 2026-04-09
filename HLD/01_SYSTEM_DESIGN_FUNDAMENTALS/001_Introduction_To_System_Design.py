"""
INTRODUCTION TO SYSTEM DESIGN
==============================

Problem Statement:
Understand what High Level Design (HLD) is, its core goals, the structured
design process used in interviews and production systems, and the key metrics
that define a well-designed large-scale system.

Key Concepts:
- Scalability   : System handles growing load without degradation
- Availability  : System is operational (measured in "nines")
- Reliability   : System produces correct results consistently
- Maintainability: System is easy to change, debug, and extend
- Design Process: Clarify → Estimate → Design → Deep Dive → Trade-offs

Design Process (45-min Interview):
  [5 min]  Clarify requirements
  [5 min]  Capacity estimation
  [10 min] High-level architecture
  [15 min] Deep dive on critical components
  [10 min] Trade-offs and alternatives
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class DesignGoal(Enum):
    SCALABILITY     = "scalability"
    AVAILABILITY    = "availability"
    RELIABILITY     = "reliability"
    MAINTAINABILITY = "maintainability"
    PERFORMANCE     = "performance"
    CONSISTENCY     = "consistency"


class DesignStep(Enum):
    REQUIREMENTS = ("requirements", 5)
    ESTIMATION   = ("estimation",   5)
    HLD          = ("high_level_design", 10)
    DEEP_DIVE    = ("deep_dive",    15)
    TRADE_OFFS   = ("trade_offs",   10)

    def __init__(self, label: str, budget_min: int):
        self.label      = label
        self.budget_min = budget_min


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class DesignDecision:
    area       : str
    choice     : str
    alternatives: List[str]
    rationale  : str


@dataclass
class DesignPhaseResult:
    step    : DesignStep
    notes   : List[str] = field(default_factory=list)
    duration_sec: float = 0.0


# ─────────────────────────────────────────────
# CORE CLASSES
# ─────────────────────────────────────────────

class SystemDesignGoalEvaluator:
    """
    Scores a proposed design against each design goal (1–5 scale).
    """

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.scores: Dict[DesignGoal, int] = {}

    def score(self, goal: DesignGoal, value: int, reason: str = ""):
        assert 1 <= value <= 5, "Score must be between 1 and 5"
        self.scores[goal] = value
        if reason:
            print(f"  [{goal.value.upper()}] score={value}/5 — {reason}")

    def overall_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def report(self):
        print(f"\n{'='*50}")
        print(f"Design Evaluation: {self.system_name}")
        print(f"{'='*50}")
        for goal, score in self.scores.items():
            bar = "█" * score + "░" * (5 - score)
            print(f"  {goal.value:<18} [{bar}] {score}/5")
        print(f"  {'OVERALL':<18} {self.overall_score():.1f}/5")


class RequirementsGatherer:
    """
    Guides through clarifying questions for a given system.
    """

    QUESTION_TEMPLATES = {
        "scale"      : "How many daily active users (DAU) do we expect?",
        "geography"  : "Is this a global system or region-specific?",
        "consistency": "Is strong consistency required, or is eventual consistency acceptable?",
        "availability": "What is the target availability SLA? (99.9% / 99.99% / 99.999%)",
        "latency"    : "What is the acceptable read/write latency (p99)?",
        "storage"    : "How much data will we store, and for how long?",
        "auth"       : "Do we need authentication and authorization?",
        "monetization": "Are there any compliance or regulatory requirements?",
    }

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.answers: Dict[str, str] = {}

    def ask(self, topic: str, answer: str):
        question = self.QUESTION_TEMPLATES.get(topic, topic)
        self.answers[topic] = answer
        print(f"  Q: {question}")
        print(f"  A: {answer}")

    def summary(self):
        print(f"\nRequirements Summary for [{self.system_name}]:")
        for topic, answer in self.answers.items():
            print(f"  • {topic:<14}: {answer}")


class DesignSession:
    """
    Simulates a structured system design session, tracking each phase.
    """

    def __init__(self, system_name: str, total_minutes: int = 45):
        self.system_name   = system_name
        self.total_minutes = total_minutes
        self.phases: List[DesignPhaseResult] = []
        self.decisions: List[DesignDecision] = []

    def run_phase(self, step: DesignStep, notes: List[str]) -> DesignPhaseResult:
        result = DesignPhaseResult(step=step, notes=notes)
        self.phases.append(result)
        print(f"\n[PHASE] {step.label.upper().replace('_', ' ')} ({step.budget_min} min budget)")
        for note in notes:
            print(f"  • {note}")
        return result

    def add_decision(self, area: str, choice: str, alternatives: List[str], rationale: str):
        decision = DesignDecision(area, choice, alternatives, rationale)
        self.decisions.append(decision)

    def print_decisions(self):
        print(f"\n{'─'*50}")
        print("KEY DESIGN DECISIONS:")
        for d in self.decisions:
            print(f"  ▶ {d.area}")
            print(f"    Chose   : {d.choice}")
            print(f"    Alt     : {', '.join(d.alternatives)}")
            print(f"    Rationale: {d.rationale}")

    def summary(self):
        print(f"\n{'='*50}")
        print(f"SESSION COMPLETE: {self.system_name}")
        print(f"Phases covered : {len(self.phases)}/{len(DesignStep)}")
        print(f"Decisions made : {len(self.decisions)}")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_introduction_to_system_design():
    print("=" * 60)
    print("INTRODUCTION TO SYSTEM DESIGN")
    print("System: URL Shortener (like TinyURL)")
    print("=" * 60)

    # ── Step 1: Requirements ──────────────────
    session = DesignSession("URL Shortener")
    gatherer = RequirementsGatherer("URL Shortener")

    session.run_phase(DesignStep.REQUIREMENTS, [
        "Shorten a long URL → short alias",
        "Redirect short URL → original URL",
        "Custom aliases (optional)",
        "Analytics: click counts per URL",
        "URL expiry support",
    ])

    gatherer.ask("scale",       "100M URLs stored, 1B redirects/day (write:read = 1:10)")
    gatherer.ask("latency",     "Redirect must be <50ms p99 globally")
    gatherer.ask("availability","99.99% uptime — redirects are business critical")
    gatherer.ask("consistency", "Eventual consistency OK for analytics; strong for redirects")
    gatherer.ask("storage",     "~500 bytes/URL × 100M = ~50GB metadata")
    gatherer.summary()

    # ── Step 2: Estimation ────────────────────
    session.run_phase(DesignStep.ESTIMATION, [
        "Writes : 100M URLs / (365 * 86400) ≈ 3 writes/sec",
        "Reads  : 1B redirects/day ≈ 11,574 reads/sec (≈ 12K QPS)",
        "Storage: 100M * 500B ≈ 50 GB (fits on a single DB with sharding room)",
        "Bandwidth: 12K * 500B ≈ 6 MB/s outbound",
        "Cache  : 20% URLs get 80% traffic → cache 20M entries ≈ 10 GB RAM",
    ])

    # ── Step 3: High-Level Design ─────────────
    session.run_phase(DesignStep.HLD, [
        "Client → Load Balancer → URL Service (stateless)",
        "URL Service → writes to Primary DB, reads from Cache / Read Replica",
        "Cache (Redis) for hot redirects: 12K QPS, 1ms latency",
        "Object Storage (S3) for analytics event logs",
        "CDN edge: redirect before hitting origin for cached short URLs",
    ])

    # ── Step 4: Deep Dive ─────────────────────
    session.run_phase(DesignStep.DEEP_DIVE, [
        "Key generation: Base62 encode 7 chars = 62^7 ≈ 3.5 trillion unique URLs",
        "Collision handling: check DB before insert (optimistic) or pre-generate keys",
        "DB: NoSQL (Cassandra/DynamoDB) — key-value access pattern, massive scale",
        "Cache strategy: cache-aside, TTL=24h, LRU eviction",
        "Analytics: write click events to Kafka → batch aggregate to analytics DB",
    ])

    # ── Step 5: Trade-offs ────────────────────
    session.run_phase(DesignStep.TRADE_OFFS, [
        "Base62 vs MD5 hash: Base62 predictable length; MD5 needs collision handling",
        "SQL vs NoSQL: SQL easier joins; NoSQL better at 100M+ rows key-value access",
        "Push vs pull CDN: pull CDN simpler ops; push better for known popular URLs",
        "Async analytics vs sync: async loses <0.01% events but writes stay <10ms",
    ])

    # ── Decisions ─────────────────────────────
    session.add_decision(
        "Key Generation", "Base62 counter (7 chars)",
        ["MD5 hash", "UUID", "Random bytes"],
        "Predictable length, no collision by design with distributed counter"
    )
    session.add_decision(
        "Primary Database", "DynamoDB (NoSQL key-value)",
        ["PostgreSQL", "MySQL", "Cassandra"],
        "Access pattern is pure key-value; NoSQL scales horizontally with no schema"
    )
    session.add_decision(
        "Caching", "Redis cache-aside (TTL 24h)",
        ["Memcached", "In-process cache", "No cache"],
        "Handles 80% of read traffic (12K QPS → 2.4K to DB); Redis supports cluster"
    )

    session.print_decisions()
    session.summary()

    # ── Goal Evaluation ───────────────────────
    print("\n" + "─" * 50)
    print("EVALUATING DESIGN GOALS:")
    evaluator = SystemDesignGoalEvaluator("URL Shortener")
    evaluator.score(DesignGoal.SCALABILITY,     5, "Stateless services + NoSQL + Redis scales horizontally")
    evaluator.score(DesignGoal.AVAILABILITY,    5, "Multi-AZ DB + replicated Redis + CDN redundancy")
    evaluator.score(DesignGoal.RELIABILITY,     4, "Exactly-once key generation needs careful coordination")
    evaluator.score(DesignGoal.PERFORMANCE,     5, "Cache hit → <5ms; DB read → <20ms p99")
    evaluator.score(DesignGoal.MAINTAINABILITY, 4, "Stateless services easy to update; analytics pipeline adds complexity")
    evaluator.score(DesignGoal.CONSISTENCY,     4, "Strong consistency for redirects; eventual for analytics is fine")
    evaluator.report()


if __name__ == "__main__":
    demonstrate_introduction_to_system_design()
