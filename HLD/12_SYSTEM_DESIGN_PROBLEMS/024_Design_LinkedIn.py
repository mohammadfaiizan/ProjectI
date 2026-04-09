"""
LINKEDIN — Professional Networking Platform
============================================

FUNCTIONAL REQUIREMENTS:
- Professional profiles: work history, skills, education, endorsements
- Connection graph: 1st, 2nd, 3rd degree connections
- Feed: posts, articles, job posts, career updates
- Job board: search, apply, track applications
- Messaging: professional DMs
- Skills endorsement and recommendations
- Company pages

NON-FUNCTIONAL REQUIREMENTS:
- 950 M members, 58 M companies, 15 M job listings
- 120 M job applications/month
- Graph traversal for "People You May Know" (2nd-degree BFS)
- Feed generation < 200 ms p99
- Job search < 100 ms p99

ARCHITECTURE:
  ┌──────────┐     ┌────────────┐     ┌──────────────────┐
  │ Client   │────▶│ API GW     │────▶│ Profile Svc      │──▶ MySQL
  └──────────┘     └────────────┘     └──────────────────┘
                         │            ┌──────────────────┐
                         ├───────────▶│ Graph Svc        │──▶ Neptune/GraphDB
                         │            └──────────────────┘
                         │            ┌──────────────────┐
                         ├───────────▶│ Feed Svc         │──▶ Kafka + Redis
                         │            └──────────────────┘
                         │            ┌──────────────────┐
                         └───────────▶│ Job Svc          │──▶ Elasticsearch

KEY DESIGN DECISIONS:
1. CONNECTION GRAPH — stored in a graph database (Neptune/Neo4j) or as
   adjacency lists in MySQL.  BFS limited to 3 hops.
   "People You May Know" (PYMK): shared connections count as features for ranking.

2. DEGREE CALCULATION — user's 1st degree connections stored in Redis SET.
   For 2nd degree: fan out 1st degree → union their connections → exclude 1st + self.
   Capped at 500 for performance.

3. FEED ALGORITHM — LinkedIn Feed uses reinforcement learning:
   - Content features: post type, author, company, hashtags
   - User features: job function, seniority, industry
   - Engagement signals: dwell time, likes, comments, shares, applies
   EdgeRank-style weighted score with recency decay.

4. JOB MATCHING — two-way matching:
   - Job → candidates: skills overlap, title similarity, location, years of exp
   - Member → jobs: explicit search + "Easy Apply" recommendations
   Elasticsearch with field boosting: title^3, skills^2, location^1.

5. SKILLS GRAPH — endorsements are weighted edges; more endorsements = higher skill score.
   Skills normalized to canonical taxonomy (e.g. "JS" → "JavaScript").

6. VIRAL COEFFICIENT — "X also liked this" social proof boosts feed ranking.
"""

from __future__ import annotations
import time
import uuid
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# Profile Models
# ---------------------------------------------------------------------------

class ConnectionDegree(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    OUT_OF_NETWORK = 4


@dataclass
class WorkExperience:
    company: str
    title: str
    start_year: int
    end_year: Optional[int]   # None = current
    description: str = ""
    is_current: bool = False

    @property
    def years(self) -> int:
        end = self.end_year or 2026
        return end - self.start_year


@dataclass
class Education:
    school: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: Optional[int]


@dataclass
class Skill:
    name: str
    endorsement_count: int = 0
    endorsed_by: Set[str] = field(default_factory=set)


@dataclass
class Profile:
    user_id: str
    name: str
    headline: str
    location: str
    industry: str
    summary: str = ""
    experience: List[WorkExperience] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    skills: Dict[str, Skill] = field(default_factory=dict)
    open_to_work: bool = False
    follower_count: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def years_of_experience(self) -> int:
        return sum(e.years for e in self.experience)

    @property
    def current_title(self) -> str:
        for e in reversed(self.experience):
            if e.is_current:
                return f"{e.title} at {e.company}"
        return self.headline


# ---------------------------------------------------------------------------
# Connection Graph
# ---------------------------------------------------------------------------

class ConnectionGraph:
    """
    Adjacency list representation of the professional connection graph.
    In production: Neptune (AWS) or custom sharded MySQL.
    """

    def __init__(self):
        # user_id → set of directly connected user_ids
        self._connections: Dict[str, Set[str]] = defaultdict(set)
        # Pending invitations
        self._pending: Dict[str, Set[str]] = defaultdict(set)  # receiver → senders

    def send_invite(self, from_id: str, to_id: str) -> bool:
        if to_id in self._connections[from_id]:
            return False  # Already connected
        self._pending[to_id].add(from_id)
        return True

    def accept_invite(self, from_id: str, to_id: str) -> bool:
        if from_id not in self._pending[to_id]:
            return False
        self._connections[to_id].add(from_id)
        self._connections[from_id].add(to_id)
        self._pending[to_id].discard(from_id)
        return True

    def remove_connection(self, user_a: str, user_b: str) -> None:
        self._connections[user_a].discard(user_b)
        self._connections[user_b].discard(user_a)

    def connection_degree(self, from_id: str, to_id: str, max_depth: int = 3) -> ConnectionDegree:
        """BFS to find shortest connection path."""
        if from_id == to_id:
            return ConnectionDegree.FIRST
        if to_id in self._connections[from_id]:
            return ConnectionDegree.FIRST

        visited = {from_id}
        queue = deque([(from_id, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in self._connections[current]:
                if neighbor == to_id:
                    degree = depth + 1
                    return [ConnectionDegree.FIRST,
                            ConnectionDegree.SECOND,
                            ConnectionDegree.THIRD][min(degree - 1, 2)]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return ConnectionDegree.OUT_OF_NETWORK

    def first_degree(self, user_id: str) -> Set[str]:
        return set(self._connections[user_id])

    def second_degree(self, user_id: str, limit: int = 500) -> Set[str]:
        """Users 2 hops away (not already connected and not self)."""
        first = self._connections[user_id]
        second = set()
        for conn in first:
            for conn2 in self._connections[conn]:
                if conn2 != user_id and conn2 not in first:
                    second.add(conn2)
                    if len(second) >= limit:
                        return second
        return second

    def mutual_connections(self, user_a: str, user_b: str) -> Set[str]:
        return self._connections[user_a] & self._connections[user_b]

    def connection_count(self, user_id: str) -> int:
        return len(self._connections[user_id])


# ---------------------------------------------------------------------------
# People You May Know (PYMK)
# ---------------------------------------------------------------------------

class PYMKService:
    """Recommends 2nd-degree connections ranked by mutual connections."""

    def __init__(self, graph: ConnectionGraph, profiles: Dict[str, Profile]):
        self._graph = graph
        self._profiles = profiles

    def recommend(self, user_id: str, limit: int = 10) -> List[Tuple[str, int, Profile]]:
        """Returns list of (user_id, mutual_count, profile)."""
        second_degree = self._graph.second_degree(user_id)
        scored = []
        for candidate_id in second_degree:
            profile = self._profiles.get(candidate_id)
            if not profile:
                continue
            mutuals = len(self._graph.mutual_connections(user_id, candidate_id))
            scored.append((candidate_id, mutuals, profile))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]


# ---------------------------------------------------------------------------
# Job Board
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str
    company_id: str
    company_name: str
    title: str
    description: str
    required_skills: List[str]
    preferred_skills: List[str]
    location: str
    remote: bool
    salary_min: Optional[int]   # USD/year
    salary_max: Optional[int]
    experience_years_min: int = 0
    applicant_count: int = 0
    posted_at: float = field(default_factory=time.time)
    is_easy_apply: bool = True


@dataclass
class JobApplication:
    application_id: str
    job_id: str
    applicant_id: str
    status: str = "applied"   # applied | reviewed | interview | offer | rejected
    applied_at: float = field(default_factory=time.time)
    cover_note: str = ""


class JobService:
    def __init__(self, profiles: Dict[str, Profile]):
        self._jobs: Dict[str, Job] = {}
        self._applications: Dict[str, List[JobApplication]] = defaultdict(list)
        self._user_applications: Dict[str, List[str]] = defaultdict(list)
        self._profiles = profiles

    def post_job(self, job: Job) -> Job:
        self._jobs[job.job_id] = job
        return job

    def apply(self, job_id: str, applicant_id: str, cover_note: str = "") -> Optional[JobApplication]:
        if job_id not in self._jobs:
            return None
        # Prevent duplicate application
        for app in self._applications[job_id]:
            if app.applicant_id == applicant_id:
                return None
        application = JobApplication(
            application_id=str(uuid.uuid4())[:8],
            job_id=job_id,
            applicant_id=applicant_id,
            cover_note=cover_note,
        )
        self._applications[job_id].append(application)
        self._user_applications[applicant_id].append(job_id)
        self._jobs[job_id].applicant_count += 1
        return application

    def search_jobs(self, query: str = "", location: str = "", skills: List[str] = None,
                    remote_only: bool = False, limit: int = 20) -> List[Tuple[Job, float]]:
        results = []
        skills_set = set(s.lower() for s in (skills or []))
        tokens = set(query.lower().split()) if query else set()

        for job in self._jobs.values():
            score = 0.0
            # Title match
            job_tokens = set(job.title.lower().split())
            if tokens:
                title_overlap = len(tokens & job_tokens) / len(tokens)
                score += title_overlap * 3.0  # boost title

            # Skills match
            if skills_set:
                req_skills = set(s.lower() for s in job.required_skills)
                pref_skills = set(s.lower() for s in job.preferred_skills)
                req_overlap = len(skills_set & req_skills) / max(len(req_skills), 1)
                pref_overlap = len(skills_set & pref_skills) / max(len(pref_skills), 1)
                score += req_overlap * 2.0 + pref_overlap * 1.0

            # Location match
            if location and location.lower() in job.location.lower():
                score += 1.0
            if remote_only and not job.remote:
                continue

            # Recency boost
            age_days = (time.time() - job.posted_at) / 86400
            score += math.exp(-age_days / 30) * 0.5

            if score > 0 or (not query and not skills):
                results.append((job, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def job_match_score(self, job: Job, profile: Profile) -> float:
        """How well does this candidate match the job?"""
        score = 0.0
        profile_skills = {s.lower() for s in profile.skills.keys()}
        req_skills = {s.lower() for s in job.required_skills}
        pref_skills = {s.lower() for s in job.preferred_skills}

        if req_skills:
            score += len(profile_skills & req_skills) / len(req_skills) * 50
        if pref_skills:
            score += len(profile_skills & pref_skills) / len(pref_skills) * 30

        if profile.years_of_experience >= job.experience_years_min:
            score += 20

        return round(score, 1)

    def user_applications(self, user_id: str) -> List[Tuple[Job, JobApplication]]:
        result = []
        for job_id in self._user_applications.get(user_id, []):
            job = self._jobs.get(job_id)
            for app in self._applications[job_id]:
                if app.applicant_id == user_id:
                    result.append((job, app))
        return result


# ---------------------------------------------------------------------------
# Feed Service
# ---------------------------------------------------------------------------

@dataclass
class FeedPost:
    post_id: str
    author_id: str
    content: str
    post_type: str    # "update" | "article" | "job_share" | "milestone"
    likes: int = 0
    comments: int = 0
    shares: int = 0
    created_at: float = field(default_factory=time.time)


class FeedService:
    def __init__(self, graph: ConnectionGraph):
        self._posts: Dict[str, FeedPost] = {}
        self._user_feed: Dict[str, List[str]] = defaultdict(list)
        self._graph = graph

    def create_post(self, author_id: str, content: str, post_type: str = "update") -> FeedPost:
        post = FeedPost(
            post_id=str(uuid.uuid4())[:10],
            author_id=author_id,
            content=content,
            post_type=post_type,
        )
        self._posts[post.post_id] = post
        # Fan out to connections
        for conn_id in self._graph.first_degree(author_id):
            self._user_feed[conn_id].append(post.post_id)
        return post

    def get_feed(self, user_id: str, limit: int = 20) -> List[FeedPost]:
        post_ids = self._user_feed.get(user_id, [])
        posts = [self._posts[pid] for pid in post_ids if pid in self._posts]
        # LinkedIn feed score: engagement × recency
        def feed_score(p: FeedPost) -> float:
            engagement = math.log1p(p.likes + p.comments * 2 + p.shares * 3)
            age_hours = (time.time() - p.created_at) / 3600
            recency = math.exp(-age_hours / 48)  # 48h half-life
            return engagement * 0.6 + recency * 0.4
        posts.sort(key=feed_score, reverse=True)
        return posts[:limit]

    def like(self, post_id: str) -> None:
        if post_id in self._posts:
            self._posts[post_id].likes += 1


# ---------------------------------------------------------------------------
# Skill Endorsement
# ---------------------------------------------------------------------------

class SkillService:
    def __init__(self, profiles: Dict[str, Profile]):
        self._profiles = profiles

    def add_skill(self, user_id: str, skill_name: str) -> bool:
        profile = self._profiles.get(user_id)
        if not profile:
            return False
        normalized = skill_name.strip().title()
        if normalized not in profile.skills:
            profile.skills[normalized] = Skill(name=normalized)
        return True

    def endorse(self, endorser_id: str, endorsee_id: str, skill_name: str) -> bool:
        """Endorser endorses endorsee's skill."""
        profile = self._profiles.get(endorsee_id)
        if not profile or skill_name not in profile.skills:
            return False
        skill = profile.skills[skill_name]
        if endorser_id in skill.endorsed_by:
            return False  # Already endorsed
        skill.endorsed_by.add(endorser_id)
        skill.endorsement_count += 1
        return True

    def top_skills(self, user_id: str, n: int = 5) -> List[Skill]:
        profile = self._profiles.get(user_id)
        if not profile:
            return []
        return sorted(profile.skills.values(),
                       key=lambda s: s.endorsement_count, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def build_profiles() -> Dict[str, Profile]:
    profiles = {}
    data = [
        ("u_alice", "Alice Chen", "Senior Software Engineer", "San Francisco, CA", "Technology"),
        ("u_bob", "Bob Kumar", "Product Manager", "Seattle, WA", "Technology"),
        ("u_carol", "Carol Smith", "Data Scientist", "New York, NY", "Finance"),
        ("u_dave", "Dave Torres", "Engineering Manager", "San Francisco, CA", "Technology"),
        ("u_eve", "Eve Johnson", "Frontend Developer", "Austin, TX", "Technology"),
        ("u_frank", "Frank Lee", "DevOps Engineer", "Seattle, WA", "Technology"),
    ]
    for uid, name, headline, loc, ind in data:
        p = Profile(uid, name, headline, loc, ind)
        p.experience = [WorkExperience("TechCorp", headline, 2020, None, is_current=True)]
        profiles[uid] = p
    return profiles


def demonstrate_1_connections_and_pymk():
    print("\n=== 1. Connection Graph & PYMK ===")
    profiles = build_profiles()
    graph = ConnectionGraph()

    # Build connections
    pairs = [
        ("u_alice", "u_bob"), ("u_alice", "u_dave"),
        ("u_bob", "u_carol"), ("u_bob", "u_frank"),
        ("u_dave", "u_eve"), ("u_dave", "u_frank"),
        ("u_carol", "u_eve"),
    ]
    for a, b in pairs:
        graph.send_invite(a, b)
        graph.accept_invite(a, b)

    alice_conns = graph.first_degree("u_alice")
    print(f"Alice's connections ({len(alice_conns)}): "
          f"{[profiles[u].name for u in alice_conns]}")

    degree_to_carol = graph.connection_degree("u_alice", "u_carol")
    degree_to_eve = graph.connection_degree("u_alice", "u_eve")
    print(f"\nAlice → Carol: {degree_to_carol.name}")
    print(f"Alice → Eve: {degree_to_eve.name}")

    mutuals = graph.mutual_connections("u_alice", "u_carol")
    print(f"Alice & Carol mutual connections: {[profiles[u].name for u in mutuals]}")

    pymk = PYMKService(graph, profiles)
    recommendations = pymk.recommend("u_alice")
    print(f"\nPYMK for Alice:")
    for uid, mutual_count, prof in recommendations:
        print(f"  {prof.name} — {mutual_count} mutual connections")

    return profiles, graph


def demonstrate_2_job_search_and_match():
    print("\n=== 2. Job Search & Candidate Matching ===")
    profiles, graph = demonstrate_1_connections_and_pymk()

    # Add skills to profiles
    alice = profiles["u_alice"]
    alice.skills = {
        "Python": Skill("Python", 45),
        "Distributed Systems": Skill("Distributed Systems", 22),
        "AWS": Skill("AWS", 18),
        "Java": Skill("Java", 10),
    }
    alice.experience.append(WorkExperience("StartupXYZ", "Backend Engineer", 2017, 2020))

    job_svc = JobService(profiles)

    jobs = [
        Job("j001", "co_001", "DataDriven Inc", "Senior Backend Engineer",
            "Build scalable data pipelines",
            required_skills=["Python", "Distributed Systems"],
            preferred_skills=["Kafka", "Spark", "AWS"],
            location="San Francisco, CA", remote=True,
            salary_min=180000, salary_max=240000, experience_years_min=5),
        Job("j002", "co_002", "CloudNine Corp", "Staff Engineer",
            "Lead cloud infrastructure team",
            required_skills=["AWS", "Terraform"],
            preferred_skills=["Python", "Go"],
            location="Seattle, WA", remote=False,
            salary_min=220000, salary_max=280000, experience_years_min=7),
        Job("j003", "co_003", "StartupABC", "Backend Developer",
            "Build REST APIs",
            required_skills=["Python", "Java"],
            preferred_skills=["Django", "Spring"],
            location="New York, NY", remote=True,
            salary_min=120000, salary_max=160000, experience_years_min=2),
    ]
    for j in jobs:
        job_svc.post_job(j)

    # Search jobs for Alice
    results = job_svc.search_jobs(
        query="Backend Engineer",
        skills=list(alice.skills.keys()),
        remote_only=True
    )
    print(f"Jobs matching Alice's profile (remote, backend):")
    for job, score in results:
        match = job_svc.job_match_score(job, alice)
        print(f"  [{match}% match] {job.title} at {job.company_name} — "
              f"${job.salary_min//1000}K-${job.salary_max//1000}K")

    # Alice applies
    app = job_svc.apply("j001", "u_alice", "I'm very excited about this role!")
    print(f"\nApplied to j001: {app.application_id if app else 'Failed'}")

    apps = job_svc.user_applications("u_alice")
    print(f"Alice's applications: {len(apps)}")


def demonstrate_3_skills_endorsement():
    print("\n=== 3. Skills & Endorsements ===")
    profiles = build_profiles()
    skill_svc = SkillService(profiles)

    alice = profiles["u_alice"]
    for skill in ["Python", "System Design", "AWS", "Go"]:
        skill_svc.add_skill("u_alice", skill)

    # Others endorse Alice
    endorsers = ["u_bob", "u_carol", "u_dave"]
    for uid in endorsers:
        skill_svc.endorse(uid, "u_alice", "Python")
    skill_svc.endorse("u_bob", "u_alice", "System Design")
    skill_svc.endorse("u_carol", "u_alice", "System Design")

    top = skill_svc.top_skills("u_alice")
    print(f"Alice's top skills:")
    for s in top:
        print(f"  {s.name}: {s.endorsement_count} endorsements "
              f"(by {list(s.endorsed_by)[:3]}...)")


def demonstrate_4_feed():
    print("\n=== 4. LinkedIn Feed ===")
    profiles = build_profiles()
    graph = ConnectionGraph()

    for a, b in [("u_alice", "u_bob"), ("u_alice", "u_dave"), ("u_bob", "u_carol")]:
        graph.send_invite(a, b)
        graph.accept_invite(a, b)

    feed_svc = FeedService(graph)

    # Posts from connections
    p1 = feed_svc.create_post("u_bob", "Excited to announce I've joined ProductCo! 🎉",
                               "milestone")
    p2 = feed_svc.create_post("u_dave", "Great article on distributed systems: 10 lessons learned",
                               "article")
    p3 = feed_svc.create_post("u_bob", "Python 4.0 is going to change everything",
                               "update")

    # Engagement
    for _ in range(50):
        feed_svc.like(p1.post_id)
    for _ in range(10):
        feed_svc.like(p2.post_id)
    feed_svc._posts[p2.post_id].comments = 15
    feed_svc.like(p3.post_id)

    alice_feed = feed_svc.get_feed("u_alice")
    print(f"Alice's feed ({len(alice_feed)} posts, sorted by engagement × recency):")
    for p in alice_feed:
        print(f"  [{p.post_type}] ♥{p.likes} 💬{p.comments} | "
              f"'{p.content[:50]}...' by {profiles.get(p.author_id, type('X', (), {'name': p.author_id})()).name if p.author_id in profiles else p.author_id}")


if __name__ == "__main__":
    demonstrate_1_connections_and_pymk()
    demonstrate_2_job_search_and_match()
    demonstrate_3_skills_endorsement()
    demonstrate_4_feed()
