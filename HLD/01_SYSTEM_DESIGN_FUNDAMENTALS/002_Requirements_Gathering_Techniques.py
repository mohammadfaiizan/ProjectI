"""
REQUIREMENTS GATHERING TECHNIQUES
==================================

Problem Statement:
Before designing any system, engineers must precisely define WHAT the system
must do (functional) and HOW WELL it must do it (non-functional). Poor
requirement gathering is the #1 reason system designs fail in interviews and
in production.

Key Concepts:
- Functional Requirements   : Features the system must provide (user-facing behaviours)
- Non-Functional Requirements: Quality attributes (scale, latency, availability, durability)
- Constraints               : Technical or business limits (budget, team size, regulations)
- Out-of-Scope              : Explicitly what we are NOT building (prevents scope creep)
- Clarifying Questions      : How to extract requirements in an interview or meeting
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class RequirementType(Enum):
    FUNCTIONAL     = "functional"
    NON_FUNCTIONAL = "non_functional"
    CONSTRAINT     = "constraint"
    OUT_OF_SCOPE   = "out_of_scope"


class Priority(Enum):
    MUST_HAVE  = "P0 - Must Have"
    SHOULD_HAVE= "P1 - Should Have"
    NICE_TO_HAVE="P2 - Nice to Have"


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class Requirement:
    req_type   : RequirementType
    description: str
    priority   : Priority = Priority.MUST_HAVE
    notes      : str = ""


@dataclass
class ClarifyingQuestion:
    category : str          # e.g. "scale", "consistency", "latency"
    question : str
    why      : str          # why this question matters architecturally
    answer   : Optional[str] = None


# ─────────────────────────────────────────────
# CORE CLASSES
# ─────────────────────────────────────────────

class RequirementsDocument:
    """
    Collects and organises all requirements for a system.
    """

    def __init__(self, system_name: str):
        self.system_name   = system_name
        self.requirements : List[Requirement] = []

    def add(self, req_type: RequirementType, description: str,
            priority: Priority = Priority.MUST_HAVE, notes: str = "") -> None:
        self.requirements.append(Requirement(req_type, description, priority, notes))

    def functional(self, description: str, priority: Priority = Priority.MUST_HAVE, notes: str = ""):
        self.add(RequirementType.FUNCTIONAL, description, priority, notes)

    def non_functional(self, description: str, priority: Priority = Priority.MUST_HAVE, notes: str = ""):
        self.add(RequirementType.NON_FUNCTIONAL, description, priority, notes)

    def constraint(self, description: str, notes: str = ""):
        self.add(RequirementType.CONSTRAINT, description, Priority.MUST_HAVE, notes)

    def out_of_scope(self, description: str):
        self.add(RequirementType.OUT_OF_SCOPE, description, Priority.NICE_TO_HAVE)

    def summary(self):
        sections = {rt: [] for rt in RequirementType}
        for r in self.requirements:
            sections[r.req_type].append(r)

        print(f"\n{'='*60}")
        print(f"REQUIREMENTS DOCUMENT: {self.system_name}")
        print(f"{'='*60}")

        labels = {
            RequirementType.FUNCTIONAL    : "✅ Functional Requirements",
            RequirementType.NON_FUNCTIONAL: "⚡ Non-Functional Requirements",
            RequirementType.CONSTRAINT    : "🔒 Constraints",
            RequirementType.OUT_OF_SCOPE  : "❌ Out of Scope",
        }

        for rt, reqs in sections.items():
            if not reqs:
                continue
            print(f"\n{labels[rt]}:")
            for r in reqs:
                tag = f"[{r.priority.value}] " if rt != RequirementType.OUT_OF_SCOPE else ""
                print(f"  • {tag}{r.description}")
                if r.notes:
                    print(f"      ↳ {r.notes}")


class ScopeDefiner:
    """
    Helps distinguish in-scope from out-of-scope features using a priority matrix.
    """

    def __init__(self):
        self.in_scope  : List[str] = []
        self.out_scope : List[str] = []

    def include(self, feature: str):
        self.in_scope.append(feature)

    def exclude(self, feature: str):
        self.out_scope.append(feature)

    def print_scope(self):
        print("\nSCOPE MATRIX:")
        print(f"  {'IN SCOPE':<40}  {'OUT OF SCOPE'}")
        print(f"  {'─'*38}  {'─'*38}")
        max_len = max(len(self.in_scope), len(self.out_scope))
        for i in range(max_len):
            inc = self.in_scope[i]  if i < len(self.in_scope)  else ""
            exc = self.out_scope[i] if i < len(self.out_scope) else ""
            print(f"  ✅ {inc:<37}  ❌ {exc}")


class ClarifyingQuestionBank:
    """
    Bank of standard clarifying questions categorised by architecture concern.
    """

    QUESTIONS: List[ClarifyingQuestion] = [
        ClarifyingQuestion(
            "scale",
            "How many daily active users (DAU) do we expect?",
            "Drives QPS, DB size, number of servers needed"
        ),
        ClarifyingQuestion(
            "read/write ratio",
            "What is the expected read-to-write ratio?",
            "Determines if we need read replicas and aggressive caching"
        ),
        ClarifyingQuestion(
            "latency",
            "What is the acceptable p99 latency for reads and writes?",
            "Affects caching strategy, DB choice, and CDN usage"
        ),
        ClarifyingQuestion(
            "availability",
            "What is the uptime SLA? (99.9% / 99.99% / 99.999%)",
            "Drives redundancy: multi-AZ, multi-region, active-active"
        ),
        ClarifyingQuestion(
            "consistency",
            "Do we need strong consistency or is eventual consistency acceptable?",
            "Strong = synchronous replication (slower); Eventual = async (faster)"
        ),
        ClarifyingQuestion(
            "data retention",
            "How long should data be stored? Any archival requirements?",
            "Affects storage tier (hot SSD vs cold S3 Glacier)"
        ),
        ClarifyingQuestion(
            "geography",
            "Is this a single-region or globally distributed system?",
            "Multi-region adds complexity: data residency, latency, consistency"
        ),
        ClarifyingQuestion(
            "security",
            "Are there regulatory requirements? (GDPR, HIPAA, PCI-DSS)",
            "May mandate encryption at rest, data residency, audit logs"
        ),
    ]

    @classmethod
    def print_all(cls):
        print("\nSTANDARD CLARIFYING QUESTIONS:")
        for i, q in enumerate(cls.QUESTIONS, 1):
            print(f"\n  {i}. [{q.category.upper()}]")
            print(f"     Q: {q.question}")
            print(f"     Why: {q.why}")

    @classmethod
    def answer(cls, category: str, answer: str):
        for q in cls.QUESTIONS:
            if q.category == category:
                q.answer = answer
                return
        raise ValueError(f"Category '{category}' not found")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_requirements_gathering():
    print("=" * 60)
    print("REQUIREMENTS GATHERING: Twitter-like System")
    print("=" * 60)

    # ── Clarifying Questions ──────────────────
    print("\n[STEP 1] Ask Clarifying Questions")
    print("─" * 45)
    bank = ClarifyingQuestionBank()
    answers = [
        ("scale",          "300M DAU, 500M tweets/day"),
        ("read/write ratio","Read:Write = 100:1 (heavy read)"),
        ("latency",        "Timeline load <200ms p99; tweet post <500ms"),
        ("availability",   "99.99% — tweets are time-sensitive, outages hurt"),
        ("consistency",    "Eventual OK — seeing tweet 1s late is acceptable"),
        ("data retention", "Tweets kept forever; media kept 7 years min"),
        ("geography",      "Global — users in US, EU, Asia, LATAM"),
        ("security",       "GDPR for EU users; COPPA for minors"),
    ]
    for category, answer in answers:
        bank.answer(category, answer)
        q = next(q for q in bank.QUESTIONS if q.category == category)
        print(f"  Q [{category}]: {q.question}")
        print(f"  A: {answer}\n")

    # ── Requirements Document ─────────────────
    print("\n[STEP 2] Document Requirements")
    doc = RequirementsDocument("Twitter Clone")

    # Functional
    doc.functional("User can post a tweet (≤280 characters)")
    doc.functional("User can follow / unfollow other users")
    doc.functional("User sees a personalised home timeline (followed users' tweets)")
    doc.functional("User can like, retweet, and reply to tweets")
    doc.functional("User can search tweets and users by keyword / hashtag")
    doc.functional("User can receive notifications (likes, follows, mentions)")
    doc.functional("Trending topics based on recent tweet volume",
                   Priority.SHOULD_HAVE)
    doc.functional("Direct Messages (1:1 and group)",
                   Priority.SHOULD_HAVE)

    # Non-Functional
    doc.non_functional("300M DAU; peak 500K tweets/min (tweet storms, live events)")
    doc.non_functional("Timeline load p99 < 200ms for cached timelines")
    doc.non_functional("99.99% availability (< 52 min downtime/year)")
    doc.non_functional("Eventual consistency for timeline (1-3s propagation acceptable)")
    doc.non_functional("Tweets stored indefinitely; media for ≥ 7 years")
    doc.non_functional("GDPR compliant: right to erasure within 30 days")

    # Constraints
    doc.constraint("Must run on cloud infrastructure (AWS preferred)")
    doc.constraint("Mobile clients are primary consumers (iOS + Android)")
    doc.constraint("Budget: 3 engineering teams (backend, infra, data)")

    # Out of Scope
    doc.out_of_scope("Twitter Blue / paid subscription features")
    doc.out_of_scope("Advertising platform")
    doc.out_of_scope("Video streaming (Twitter Spaces / Live)")
    doc.out_of_scope("Third-party API (Twitter API v2)")

    doc.summary()

    # ── Scope Matrix ──────────────────────────
    print("\n[STEP 3] Define Scope")
    scope = ScopeDefiner()
    scope.include("Tweet CRUD (create, read, delete)")
    scope.include("Follow graph")
    scope.include("Home timeline generation")
    scope.include("Likes, retweets, replies")
    scope.include("Hashtag and user search")
    scope.include("Push notifications")
    scope.exclude("Advertising / Promoted tweets")
    scope.exclude("Twitter Blue subscriptions")
    scope.exclude("Live video (Spaces)")
    scope.exclude("Third-party API access")
    scope.print_scope()

    print("\n✅ Requirements gathering complete. Ready for capacity estimation.")


if __name__ == "__main__":
    demonstrate_requirements_gathering()
