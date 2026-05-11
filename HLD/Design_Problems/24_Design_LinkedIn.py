"""
LinkedIn System Design - Python Implementation
Demonstrates: ConnectionGraph (BFS), PYMK (People You May Know),
              SkillEndorsement, JobSearch (inverted index), FeedGenerator.
No external dependencies - standard library only.
"""

import uuid
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class User:
    id: str
    name: str
    headline: str = ""
    location: str = ""
    company_id: Optional[str] = None
    school_id: Optional[str] = None
    skills: list = field(default_factory=list)
    connection_count: int = 0
    follower_count: int = 0

@dataclass
class Job:
    id: str
    company_id: str
    title: str
    description: str
    location: str
    required_skills: list
    experience_min: int = 0
    experience_max: int = 10
    remote_type: str = "onsite"   # onsite, remote, hybrid
    is_active: bool = True
    posted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class Post:
    id: str
    author_id: str
    content: str
    likes: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class PYMKCandidate:
    user: User
    mutual_connections: int
    score: float


# ---------------------------------------------------------------------------
# 1. ConnectionGraph — BFS-based degree discovery
# ---------------------------------------------------------------------------

class ConnectionGraph:
    """
    Undirected graph of professional connections.
    Stored as adjacency list: user_id -> set of connected user_ids.
    In production: sharded PostgreSQL adjacency table + graph DB for traversal.
    """

    def __init__(self):
        # user_id -> set of 1st-degree connection IDs
        self._adj: dict[str, set] = defaultdict(set)
        # user_id -> set of follower IDs (directed)
        self._followers: dict[str, set] = defaultdict(set)
        # Pending connection requests: (from_id, to_id)
        self._pending: set[tuple] = set()

    def send_request(self, from_id: str, to_id: str) -> None:
        if from_id == to_id:
            raise ValueError("Cannot connect to yourself")
        if to_id in self._adj[from_id]:
            raise ValueError("Already connected")
        self._pending.add((from_id, to_id))

    def accept_request(self, from_id: str, to_id: str) -> None:
        if (from_id, to_id) not in self._pending:
            raise KeyError("No pending request found")
        self._pending.discard((from_id, to_id))
        # Undirected edge: add both directions
        self._adj[from_id].add(to_id)
        self._adj[to_id].add(from_id)

    def follow(self, follower_id: str, target_id: str) -> None:
        self._followers[target_id].add(follower_id)

    def get_connections(self, user_id: str) -> set:
        return self._adj[user_id].copy()

    def get_followers(self, user_id: str) -> set:
        return self._followers[user_id].copy()

    def get_connection_count(self, user_id: str) -> int:
        return len(self._adj[user_id])

    def get_mutual_connections(self, user_a: str, user_b: str) -> set:
        return self._adj[user_a] & self._adj[user_b]

    def get_degree(self, user_id: str, target_id: str) -> int:
        """Returns 1, 2, 3, or -1 if > 3rd degree. Uses BFS."""
        if target_id == user_id:
            return 0
        visited = {user_id}
        queue   = deque([(user_id, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= 3:
                continue
            for neighbor in self._adj[current]:
                if neighbor == target_id:
                    return depth + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return -1

    def get_2nd_degree_connections(self, user_id: str) -> dict[str, int]:
        """
        Returns {candidate_id: mutual_count} for all 2nd degree connections.
        In production: cached in Redis (TTL 1h).
        """
        first_degree = self._adj[user_id]
        mutual_count: dict[str, int] = defaultdict(int)

        for conn in first_degree:
            for second in self._adj[conn]:
                if second != user_id and second not in first_degree:
                    mutual_count[second] += 1

        return dict(mutual_count)

    def is_connected(self, user_a: str, user_b: str) -> bool:
        return user_b in self._adj[user_a]


# ---------------------------------------------------------------------------
# 2. PYMK — People You May Know
# ---------------------------------------------------------------------------

class PYMK:
    """
    Scores candidates for PYMK recommendation.
    Signal weights:
      - mutual connections: +10 per mutual
      - same company:       +25
      - same school:        +15
      - same industry:      +5
    In production: offline Spark job computes PYMK nightly.
    """

    WEIGHTS = {
        "mutual_connection": 10,
        "same_company":      25,
        "same_school":       15,
        "same_industry":     5,
    }

    def __init__(self, graph: ConnectionGraph, users: dict[str, User]):
        self._graph = graph
        self._users = users

    def get_recommendations(
        self, user_id: str, top_n: int = 10
    ) -> list[PYMKCandidate]:
        user = self._users.get(user_id)
        if not user:
            return []

        # Get 2nd degree candidates with mutual counts
        second_degree = self._graph.get_2nd_degree_connections(user_id)
        candidates = []

        for cand_id, mutual_count in second_degree.items():
            cand = self._users.get(cand_id)
            if not cand:
                continue

            score = mutual_count * self.WEIGHTS["mutual_connection"]

            # Same current company
            if user.company_id and user.company_id == cand.company_id:
                score += self.WEIGHTS["same_company"]

            # Same school
            if user.school_id and user.school_id == cand.school_id:
                score += self.WEIGHTS["same_school"]

            candidates.append(PYMKCandidate(
                user=cand,
                mutual_connections=mutual_count,
                score=score,
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_n]


# ---------------------------------------------------------------------------
# 3. SkillEndorsement — top skills per user
# ---------------------------------------------------------------------------

class SkillEndorsement:
    """
    Tracks skill endorsements between connected users.
    Only 1st degree connections can endorse.
    One endorsement per skill per endorser (idempotent).
    """

    def __init__(self, graph: ConnectionGraph):
        self._graph = graph
        # (user_id, skill) -> count
        self._endorsement_counts: dict[tuple, int] = defaultdict(int)
        # endorser_id -> set of (user_id, skill) they've endorsed
        self._endorser_log: dict[str, set] = defaultdict(set)

    def endorse_skill(self, endorser_id: str, user_id: str, skill: str) -> bool:
        """
        Returns True if endorsement added, False if already endorsed or invalid.
        """
        if endorser_id == user_id:
            return False
        if not self._graph.is_connected(endorser_id, user_id):
            raise PermissionError(f"{endorser_id} is not connected to {user_id}")

        key = (user_id, skill)
        if key in self._endorser_log[endorser_id]:
            return False  # Already endorsed

        self._endorsement_counts[key] += 1
        self._endorser_log[endorser_id].add(key)
        return True

    def get_skill_count(self, user_id: str, skill: str) -> int:
        return self._endorsement_counts[(user_id, skill)]

    def get_top_skills(self, user: User, top_n: int = 3) -> list[tuple[str, int]]:
        """Returns top N skills by endorsement count."""
        skills_with_counts = [
            (skill, self._endorsement_counts[(user.id, skill)])
            for skill in user.skills
        ]
        skills_with_counts.sort(key=lambda x: x[1], reverse=True)
        return skills_with_counts[:top_n]


# ---------------------------------------------------------------------------
# 4. JobSearch — inverted index on skills + location
# ---------------------------------------------------------------------------

class JobSearch:
    """
    Inverted index over job required_skills + title tokens.
    In production: Elasticsearch with function_score for personalized ranking.
    """

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        # token -> set of job_ids
        self._skill_index:  dict[str, set] = defaultdict(set)
        self._title_index:  dict[str, set] = defaultdict(set)
        self._location_index: dict[str, set] = defaultdict(set)

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().replace(",", " ").replace("-", " ").split()

    def index_job(self, job: Job) -> None:
        self._jobs[job.id] = job
        for skill in job.required_skills:
            self._skill_index[skill.lower()].add(job.id)
        for token in self._tokenize(job.title):
            self._title_index[token].add(job.id)
        loc_tokens = self._tokenize(job.location)
        for token in loc_tokens:
            self._location_index[token].add(job.id)

    def _skill_match_score(self, user_skills: list, job: Job) -> float:
        """Fraction of required skills the user has."""
        if not job.required_skills:
            return 1.0
        user_set = {s.lower() for s in user_skills}
        req_set  = {s.lower() for s in job.required_skills}
        overlap  = len(user_set & req_set)
        return overlap / len(req_set)

    def search(
        self,
        query: str = "",
        user_skills: list = None,
        location: str = "",
        remote_filter: str = "",    # onsite|remote|hybrid|""
        exp_years: int = 0,
        limit: int = 20,
    ) -> list[dict]:
        user_skills = user_skills or []
        candidate_ids: Optional[set] = None

        # Skill-based filter
        if user_skills:
            skill_matches: dict[str, int] = defaultdict(int)
            for skill in user_skills:
                for jid in self._skill_index.get(skill.lower(), set()):
                    skill_matches[jid] += 1
            if candidate_ids is None:
                candidate_ids = set(skill_matches.keys())
            else:
                candidate_ids &= set(skill_matches.keys())

        # Title/keyword filter
        if query:
            title_matches: set = set()
            for token in self._tokenize(query):
                title_matches |= self._title_index.get(token, set())
            if candidate_ids is None:
                candidate_ids = title_matches
            else:
                candidate_ids &= title_matches

        # Location filter
        if location:
            loc_matches: set = set()
            for token in self._tokenize(location):
                loc_matches |= self._location_index.get(token, set())
            if candidate_ids is None:
                candidate_ids = loc_matches
            else:
                candidate_ids &= loc_matches

        if candidate_ids is None:
            candidate_ids = set(self._jobs.keys())

        results = []
        for jid in candidate_ids:
            job = self._jobs[jid]
            if not job.is_active:
                continue
            if remote_filter and job.remote_type != remote_filter:
                continue
            if exp_years and not (job.experience_min <= exp_years <= job.experience_max):
                continue

            match_score = self._skill_match_score(user_skills, job)
            results.append({
                "job":          job,
                "match_score":  round(match_score, 2),
                "skills_match": int(match_score * len(job.required_skills)),
                "skills_total": len(job.required_skills),
            })

        results.sort(key=lambda r: r["match_score"], reverse=True)
        return results[:limit]


# ---------------------------------------------------------------------------
# 5. FeedGenerator — push-pull hybrid
# ---------------------------------------------------------------------------

class FeedGenerator:
    """
    Simple feed generator mixing:
    - Posts from connections (push for normal users, pull for celebrities)
    - Job recommendations
    - PYMK cards
    Threshold: users with > 1000 followers use pull model.
    """
    CELEBRITY_THRESHOLD = 1000

    def __init__(self, graph: ConnectionGraph, users: dict[str, User]):
        self._graph       = graph
        self._users       = users
        self._posts: list[Post] = []
        # user_id -> pre-computed feed post_ids (push model cache)
        self._feed_cache: dict[str, list] = defaultdict(list)

    def publish_post(self, post: Post) -> None:
        self._posts.append(post)
        author = self._users.get(post.author_id)
        if not author:
            return
        follower_count = self._graph.get_connection_count(post.author_id)
        # Push to all connection feeds if below celebrity threshold
        if follower_count < self.CELEBRITY_THRESHOLD:
            connections = self._graph.get_connections(post.author_id)
            for conn_id in connections:
                self._feed_cache[conn_id].insert(0, post.id)

    def get_feed(self, user_id: str, limit: int = 20) -> list[Post]:
        """Returns feed for user: mix of push cache + pull from large-follower connections."""
        post_map = {p.id: p for p in self._posts}
        seen_ids: set = set()
        result: list[Post] = []

        # 1. Posts from push cache (normal connections)
        for pid in self._feed_cache.get(user_id, []):
            if pid in post_map and pid not in seen_ids:
                result.append(post_map[pid])
                seen_ids.add(pid)

        # 2. Pull from celebrity connections (follower_count >= threshold)
        connections = self._graph.get_connections(user_id)
        for conn_id in connections:
            conn_follower_count = self._graph.get_connection_count(conn_id)
            if conn_follower_count >= self.CELEBRITY_THRESHOLD:
                conn_posts = [p for p in self._posts if p.author_id == conn_id]
                conn_posts.sort(key=lambda p: p.created_at, reverse=True)
                for post in conn_posts[:5]:
                    if post.id not in seen_ids:
                        result.append(post)
                        seen_ids.add(post.id)

        result.sort(key=lambda p: p.created_at, reverse=True)
        return result[:limit]


# ---------------------------------------------------------------------------
# 6. LinkedInSystem — Facade
# ---------------------------------------------------------------------------

class LinkedInSystem:
    def __init__(self):
        self._users: dict[str, User] = {}
        self._graph     = ConnectionGraph()
        self._endorsement = SkillEndorsement(self._graph)
        self._job_search = JobSearch()
        self._pymk       = PYMK(self._graph, self._users)
        self._feed_gen   = FeedGenerator(self._graph, self._users)

    def add_user(self, user: User) -> None:
        self._users[user.id] = user

    def connect(self, from_id: str, to_id: str) -> None:
        self._graph.send_request(from_id, to_id)
        self._graph.accept_request(from_id, to_id)

    def search_people(self, query: str) -> list[User]:
        q = query.lower()
        return [
            u for u in self._users.values()
            if q in u.name.lower() or q in u.headline.lower()
        ]

    def get_2nd_degree_connections(self, user_id: str) -> dict[str, int]:
        return self._graph.get_2nd_degree_connections(user_id)

    def get_people_you_may_know(self, user_id: str, top_n: int = 5) -> list[PYMKCandidate]:
        return self._pymk.get_recommendations(user_id, top_n)

    def post_job(self, job: Job) -> None:
        self._job_search.index_job(job)

    def apply_to_job(self, job_id: str, user_id: str) -> dict:
        return {"application_id": str(uuid.uuid4())[:8], "status": "submitted"}

    def endorse_skill(self, endorser_id: str, user_id: str, skill: str) -> bool:
        return self._endorsement.endorse_skill(endorser_id, user_id, skill)

    def get_feed(self, user_id: str, limit: int = 20) -> list[Post]:
        return self._feed_gen.get_feed(user_id, limit)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    system = LinkedInSystem()

    # Create users
    users = [
        User("u1", "Alice Chen",   "Senior Software Engineer", "San Francisco", company_id="C1", school_id="S1", skills=["Python", "Distributed Systems", "Kafka"]),
        User("u2", "Bob Smith",    "Data Scientist",           "San Francisco", company_id="C1", school_id="S2", skills=["Python", "ML", "SQL"]),
        User("u3", "Carol Jones",  "Engineering Manager",      "Seattle",       company_id="C2", school_id="S1", skills=["Python", "Leadership", "Distributed Systems"]),
        User("u4", "Dave Wilson",  "Backend Engineer",         "New York",      company_id="C3", school_id="S2", skills=["Java", "Kafka", "SQL"]),
        User("u5", "Eve Martinez", "ML Engineer",              "San Francisco", company_id="C2", school_id="S1", skills=["Python", "ML", "PyTorch"]),
    ]
    for u in users:
        system.add_user(u)

    # Build connections
    connections = [("u1", "u2"), ("u1", "u3"), ("u2", "u4"), ("u3", "u5"), ("u4", "u5")]
    for a, b in connections:
        system.connect(a, b)

    # Degree discovery
    print("=== Connection Degrees from u1 ===")
    for uid in ["u2", "u3", "u4", "u5"]:
        deg = system._graph.get_degree("u1", uid)
        mutual = system._graph.get_mutual_connections("u1", uid)
        print(f"  u1 -> {uid} ({system._users[uid].name}): {deg}° | mutual: {len(mutual)}")

    # 2nd degree
    print("\n=== 2nd Degree Connections (u1) ===")
    second_degree = system.get_2nd_degree_connections("u1")
    for uid, mutual_count in sorted(second_degree.items(), key=lambda x: -x[1]):
        user = system._users[uid]
        print(f"  {user.name}: {mutual_count} mutual connections")

    # PYMK
    print("\n=== People You May Know (u1) ===")
    pymk = system.get_people_you_may_know("u1", top_n=3)
    for c in pymk:
        print(f"  {c.user.name}: score={c.score} | mutual={c.mutual_connections} | "
              f"same_company={c.user.company_id == system._users['u1'].company_id}")

    # Endorsements
    print("\n=== Skill Endorsements ===")
    system.endorse_skill("u2", "u1", "Python")
    system.endorse_skill("u3", "u1", "Python")
    system.endorse_skill("u3", "u1", "Distributed Systems")
    top_skills = system._endorsement.get_top_skills(system._users["u1"])
    for skill, count in top_skills:
        print(f"  u1.{skill}: {count} endorsements")

    # Job search
    print("\n=== Job Search ===")
    jobs = [
        Job("J1", "C1", "Senior Python Engineer", "...", "San Francisco, CA",
            ["Python", "Distributed Systems", "Kafka"], 4, 8, "hybrid"),
        Job("J2", "C2", "ML Engineer", "...", "Remote",
            ["Python", "ML", "PyTorch"], 2, 6, "remote"),
        Job("J3", "C3", "Backend Java Engineer", "...", "New York, NY",
            ["Java", "SQL", "REST"], 2, 7, "onsite"),
        Job("J4", "C4", "Data Scientist", "...", "San Francisco, CA",
            ["Python", "ML", "SQL"], 1, 5, "hybrid"),
    ]
    for j in jobs:
        system.post_job(j)

    alice_results = system._job_search.search(
        user_skills=system._users["u1"].skills,
        location="San Francisco",
        limit=5,
    )
    for r in alice_results:
        j = r["job"]
        print(f"  {j.title} @ {j.company_id}: {r['skills_match']}/{r['skills_total']} skills | "
              f"match={r['match_score']:.0%}")

    # Feed
    print("\n=== Feed (u1) ===")
    for uid, content in [("u2", "Excited to share our new model results!"),
                         ("u3", "Great talk at the distributed systems conference"),
                         ("u4", "New blog post: Kafka at scale")]:
        post = Post(str(uuid.uuid4())[:6], uid, content)
        system._feed_gen.publish_post(post)

    feed = system.get_feed("u1")
    for post in feed:
        author = system._users.get(post.author_id, User(post.author_id, "Unknown"))
        print(f"  [{author.name}]: {post.content[:60]}")
