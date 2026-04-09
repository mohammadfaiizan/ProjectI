"""
REDDIT — Social News Aggregation Platform
==========================================

FUNCTIONAL REQUIREMENTS:
- Subreddits: create, subscribe, moderate
- Posts: text, link, image, video, polls
- Voting: upvote/downvote (1 vote per user per post/comment)
- Comments: threaded with unlimited depth
- Ranking algorithms: Hot, Top, New, Controversial, Rising
- Awards: Gold, Silver, Platinum
- User karma (post + comment karma)
- Moderation: AutoModerator, ban, flair, spam removal

NON-FUNCTIONAL REQUIREMENTS:
- 1.5 B posts and comments, 52 M DAU
- 25-50 K submissions/hour during peak events
- Feed generation < 200 ms p99
- Vote counts eventually consistent (OK to lag by seconds)
- Ranking updated every few minutes

ARCHITECTURE:
  Client ──▶ API GW ──▶ Feed Svc ──▶ Ranking Worker (batch)
                    ──▶ Post Svc ──▶ PostgreSQL
                    ──▶ Vote Svc ──▶ Redis (counters) + Cassandra
                    ──▶ Comment Svc ──▶ PostgreSQL (adjacency list)
                    ──▶ Search Svc ──▶ Elasticsearch

KEY DESIGN DECISIONS:
1. VOTE COUNTING — Redis INCR/DECR for real-time counters.
   Periodically flush to persistent store (Cassandra) via background job.
   Vote dedup: Redis SET per post to track who voted (bloom filter for scale).

2. HOT RANKING ALGORITHM — Wilson Score for comments; Reddit Hot Score for posts:
   score = log10(max(|ups - downs|, 1)) + sign(ups - downs) × ts / 45000
   ts = seconds since Reddit epoch (2005-12-08 07:46:43 UTC)
   Creates time-decaying score where newer posts can outrank older popular ones.

3. COMMENT TREE — stored as flat table with parent_id pointer.
   Retrieved via BFS/DFS with depth limit.
   "Best" sort: Wilson Lower Confidence Bound (accounts for few votes).

4. KARMA — computed incrementally on vote events via message queue.
   Cached in user profile; exact count from DB on profile page load.

5. SUBREDDIT FEED — pre-computed for Top 1000 subs; on-demand for rest.
   Worker re-ranks every 15 minutes using hot score on recent posts.

6. AUTOMOD — rule engine running on post creation:
   regex patterns, min karma, link domain blocklist, flair requirements.
"""

from __future__ import annotations
import time
import uuid
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDDIT_EPOCH = 1134028003.0   # 2005-12-08 07:46:43 UTC


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class PostType(Enum):
    TEXT = "text"
    LINK = "link"
    IMAGE = "image"
    VIDEO = "video"
    POLL = "poll"


class VoteDirection(Enum):
    UP = 1
    DOWN = -1
    NONE = 0


@dataclass
class Subreddit:
    subreddit_id: str
    name: str            # e.g. "programming"
    title: str
    description: str
    rules: List[str] = field(default_factory=list)
    moderators: Set[str] = field(default_factory=set)
    subscribers: Set[str] = field(default_factory=set)
    is_nsfw: bool = False
    is_private: bool = False
    created_at: float = field(default_factory=time.time)

    @property
    def subscriber_count(self) -> int:
        return len(self.subscribers)


@dataclass
class Post:
    post_id: str
    subreddit_id: str
    author_id: str
    title: str
    post_type: PostType
    content: str = ""      # text body or URL
    ups: int = 0
    downs: int = 0
    awards: int = 0
    flair: str = ""
    is_stickied: bool = False
    is_locked: bool = False
    is_removed: bool = False
    comment_count: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def score(self) -> int:
        return self.ups - self.downs

    @property
    def upvote_ratio(self) -> float:
        total = self.ups + self.downs
        return self.ups / total if total > 0 else 0.0


@dataclass
class Comment:
    comment_id: str
    post_id: str
    author_id: str
    parent_id: Optional[str]   # None = top-level comment
    body: str
    ups: int = 0
    downs: int = 0
    is_removed: bool = False
    depth: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def score(self) -> int:
        return self.ups - self.downs


@dataclass
class User:
    user_id: str
    username: str
    post_karma: int = 0
    comment_karma: int = 0
    created_at: float = field(default_factory=time.time)
    subscriptions: Set[str] = field(default_factory=set)

    @property
    def total_karma(self) -> int:
        return self.post_karma + self.comment_karma


# ---------------------------------------------------------------------------
# Ranking Algorithms
# ---------------------------------------------------------------------------

class RankingAlgorithm:
    """Collection of Reddit ranking functions."""

    @staticmethod
    def hot_score(ups: int, downs: int, created_at: float) -> float:
        """
        Reddit's Hot algorithm.
        Combines score magnitude with recency.
        """
        score = ups - downs
        order = math.log10(max(abs(score), 1))
        sign = 1 if score > 0 else (-1 if score < 0 else 0)
        seconds = created_at - REDDIT_EPOCH
        return round(sign * order + seconds / 45000, 7)

    @staticmethod
    def wilson_lower_bound(ups: int, downs: int, z: float = 1.96) -> float:
        """
        Wilson Score Lower Confidence Bound for comment ranking ("Best" sort).
        Accounts for uncertainty with few votes.
        z=1.96 corresponds to 95% confidence interval.
        """
        n = ups + downs
        if n == 0:
            return 0.0
        phat = ups / n
        return (phat + z * z / (2 * n) -
                z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

    @staticmethod
    def top_score(ups: int, downs: int) -> int:
        """Top sort: pure score."""
        return ups - downs

    @staticmethod
    def controversial_score(ups: int, downs: int) -> float:
        """
        Controversial: high engagement with balanced up/down votes.
        """
        if downs == 0:
            return ups
        if ups == 0:
            return downs
        magnitude = ups + downs
        balance = min(ups, downs) / max(ups, downs)  # closer to 1 = more controversial
        return magnitude * balance

    @staticmethod
    def rising_score(ups: int, created_at: float) -> float:
        """Rising: recent posts with fast-growing upvotes."""
        age_hours = (time.time() - created_at) / 3600
        if age_hours <= 0:
            age_hours = 0.001
        return ups / age_hours


# ---------------------------------------------------------------------------
# Vote Service
# ---------------------------------------------------------------------------

class VoteService:
    """
    Redis-backed vote counters with dedup tracking.
    Real: Redis INCR/DECR + Bloom filter for dedup.
    """

    def __init__(self):
        # (entity_id, user_id) → VoteDirection (current vote)
        self._votes: Dict[Tuple[str, str], VoteDirection] = {}
        # entity_id → (ups, downs)
        self._counts: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))

    def vote(self, entity_id: str, user_id: str,
             direction: VoteDirection) -> Tuple[int, int]:
        """Apply vote; returns new (ups, downs)."""
        key = (entity_id, user_id)
        old = self._votes.get(key, VoteDirection.NONE)
        ups, downs = self._counts[entity_id]

        # Undo old vote
        if old == VoteDirection.UP:
            ups -= 1
        elif old == VoteDirection.DOWN:
            downs -= 1

        # Apply new vote
        if direction == VoteDirection.UP:
            ups += 1
        elif direction == VoteDirection.DOWN:
            downs += 1

        self._votes[key] = direction
        self._counts[entity_id] = (ups, downs)
        return ups, downs

    def get_counts(self, entity_id: str) -> Tuple[int, int]:
        return self._counts[entity_id]

    def get_user_vote(self, entity_id: str, user_id: str) -> VoteDirection:
        return self._votes.get((entity_id, user_id), VoteDirection.NONE)


# ---------------------------------------------------------------------------
# Post & Comment Service
# ---------------------------------------------------------------------------

class PostService:
    def __init__(self, vote_svc: VoteService):
        self._posts: Dict[str, Post] = {}
        self._sub_posts: Dict[str, List[str]] = defaultdict(list)
        self._vote_svc = vote_svc

    def submit(self, subreddit_id: str, author_id: str, title: str,
               post_type: PostType, content: str = "") -> Post:
        post = Post(
            post_id=str(uuid.uuid4())[:12],
            subreddit_id=subreddit_id,
            author_id=author_id,
            title=title,
            post_type=post_type,
            content=content,
            ups=1,  # Auto-upvote by author
        )
        self._posts[post.post_id] = post
        self._sub_posts[subreddit_id].append(post.post_id)
        self._vote_svc.vote(post.post_id, author_id, VoteDirection.UP)
        return post

    def vote(self, post_id: str, user_id: str, direction: VoteDirection) -> Optional[Post]:
        post = self._posts.get(post_id)
        if not post:
            return None
        ups, downs = self._vote_svc.vote(post_id, user_id, direction)
        post.ups = ups
        post.downs = downs
        return post

    def get_feed(self, subreddit_id: str, sort: str = "hot",
                 limit: int = 25) -> List[Post]:
        post_ids = self._sub_posts.get(subreddit_id, [])
        posts = [self._posts[pid] for pid in post_ids if pid in self._posts
                 and not self._posts[pid].is_removed]

        if sort == "hot":
            posts.sort(key=lambda p: RankingAlgorithm.hot_score(p.ups, p.downs, p.created_at),
                       reverse=True)
        elif sort == "top":
            posts.sort(key=lambda p: RankingAlgorithm.top_score(p.ups, p.downs), reverse=True)
        elif sort == "new":
            posts.sort(key=lambda p: p.created_at, reverse=True)
        elif sort == "controversial":
            posts.sort(key=lambda p: RankingAlgorithm.controversial_score(p.ups, p.downs),
                       reverse=True)
        elif sort == "rising":
            posts.sort(key=lambda p: RankingAlgorithm.rising_score(p.ups, p.created_at),
                       reverse=True)

        return posts[:limit]

    def get_post(self, post_id: str) -> Optional[Post]:
        return self._posts.get(post_id)


class CommentService:
    def __init__(self, vote_svc: VoteService, post_svc: PostService):
        self._comments: Dict[str, Comment] = {}
        self._post_comments: Dict[str, List[str]] = defaultdict(list)
        self._vote_svc = vote_svc
        self._post_svc = post_svc

    def add_comment(self, post_id: str, author_id: str, body: str,
                    parent_id: Optional[str] = None) -> Optional[Comment]:
        post = self._post_svc.get_post(post_id)
        if not post or post.is_locked:
            return None

        depth = 0
        if parent_id and parent_id in self._comments:
            depth = self._comments[parent_id].depth + 1

        comment = Comment(
            comment_id=str(uuid.uuid4())[:10],
            post_id=post_id,
            author_id=author_id,
            parent_id=parent_id,
            body=body,
            ups=1,
            depth=depth,
        )
        self._comments[comment.comment_id] = comment
        self._post_comments[post_id].append(comment.comment_id)
        post.comment_count += 1
        self._vote_svc.vote(comment.comment_id, author_id, VoteDirection.UP)
        return comment

    def vote_comment(self, comment_id: str, user_id: str, direction: VoteDirection):
        comment = self._comments.get(comment_id)
        if comment:
            ups, downs = self._vote_svc.vote(comment_id, user_id, direction)
            comment.ups = ups
            comment.downs = downs

    def get_comment_tree(self, post_id: str, sort: str = "best",
                          max_depth: int = 6) -> List[Comment]:
        """Return comments as a flat list with depth info, sorted."""
        comment_ids = self._post_comments.get(post_id, [])
        all_comments = {cid: self._comments[cid] for cid in comment_ids
                        if cid in self._comments and not self._comments[cid].is_removed}

        # Get top-level comments
        top_level = [c for c in all_comments.values() if c.parent_id is None]

        if sort == "best":
            top_level.sort(
                key=lambda c: RankingAlgorithm.wilson_lower_bound(c.ups, c.downs),
                reverse=True
            )
        elif sort == "top":
            top_level.sort(key=lambda c: c.score, reverse=True)
        elif sort == "new":
            top_level.sort(key=lambda c: c.created_at, reverse=True)
        elif sort == "controversial":
            top_level.sort(
                key=lambda c: RankingAlgorithm.controversial_score(c.ups, c.downs),
                reverse=True
            )

        # BFS to build tree
        result = []
        queue = [(c, 0) for c in top_level]
        # Build children index
        children: Dict[str, List[Comment]] = defaultdict(list)
        for c in all_comments.values():
            if c.parent_id:
                children[c.parent_id].append(c)

        def dfs(comment: Comment, depth: int):
            if depth > max_depth:
                return
            result.append(comment)
            kids = children.get(comment.comment_id, [])
            kids.sort(key=lambda c: RankingAlgorithm.wilson_lower_bound(c.ups, c.downs),
                      reverse=True)
            for kid in kids:
                dfs(kid, depth + 1)

        for c in top_level:
            dfs(c, 0)

        return result


# ---------------------------------------------------------------------------
# AutoModerator
# ---------------------------------------------------------------------------

@dataclass
class AutoModRule:
    name: str
    min_account_age_days: int = 0
    min_karma: int = 0
    domain_blocklist: Set[str] = field(default_factory=set)
    required_flair: bool = False
    action: str = "remove"  # "remove" | "report" | "require_flair"


class AutoModerator:
    def __init__(self):
        self._rules: Dict[str, List[AutoModRule]] = defaultdict(list)

    def add_rule(self, subreddit_id: str, rule: AutoModRule):
        self._rules[subreddit_id].append(rule)

    def evaluate(self, post: Post, author: User,
                 account_age_days: int) -> Tuple[bool, str]:
        """Returns (allowed, reason)."""
        for rule in self._rules.get(post.subreddit_id, []):
            if account_age_days < rule.min_account_age_days:
                return False, f"Account too new (need {rule.min_account_age_days} days)"
            if author.total_karma < rule.min_karma:
                return False, f"Karma too low (need {rule.min_karma})"
            if post.post_type == PostType.LINK and rule.domain_blocklist:
                for blocked in rule.domain_blocklist:
                    if blocked in post.content:
                        return False, f"Blocked domain: {blocked}"
        return True, "OK"


# ---------------------------------------------------------------------------
# Karma System
# ---------------------------------------------------------------------------

class KarmaService:
    def __init__(self):
        self._user_karma: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"post": 0, "comment": 0}
        )

    def on_vote(self, entity_id: str, author_id: str,
                entity_type: str, delta: int) -> None:
        """delta = +1 for upvote, -1 for downvote."""
        if entity_type == "post":
            self._user_karma[author_id]["post"] += delta
        elif entity_type == "comment":
            self._user_karma[author_id]["comment"] += delta

    def get_karma(self, user_id: str) -> Dict[str, int]:
        return dict(self._user_karma[user_id])


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demonstrate_1_subreddit_and_posting():
    print("\n=== 1. Subreddits & Post Submission ===")
    vote_svc = VoteService()
    post_svc = PostService(vote_svc)

    sub = Subreddit(
        subreddit_id="sub_prog",
        name="programming",
        title="Computer Programming",
        description="A subreddit for discussion and news about computer programming.",
        subscribers={"u1", "u2", "u3"},
    )

    p1 = post_svc.submit(sub.subreddit_id, "user_alice", "Python 4.0 Released!",
                          PostType.LINK, "https://python.org/news")
    p2 = post_svc.submit(sub.subreddit_id, "user_bob", "Ask me anything about Rust",
                          PostType.TEXT, "I've been writing Rust for 5 years...")
    p3 = post_svc.submit(sub.subreddit_id, "user_carol", "Why I switched from Java to Go",
                          PostType.TEXT, "TL;DR: faster build times and simpler concurrency")

    # Simulate votes
    voters = ["user_dave", "user_eve", "user_frank"]
    for v in voters:
        post_svc.vote(p1.post_id, v, VoteDirection.UP)
    post_svc.vote(p2.post_id, "user_dave", VoteDirection.DOWN)
    post_svc.vote(p3.post_id, "user_dave", VoteDirection.UP)
    post_svc.vote(p3.post_id, "user_eve", VoteDirection.UP)

    print(f"Posts in r/{sub.name}:")
    for p in [p1, p2, p3]:
        print(f"  ↑{p.ups} ↓{p.downs} [{p.score:+d}] '{p.title}'")

    return sub, post_svc, vote_svc, p1, p2, p3


def demonstrate_2_ranking_algorithms(sub, post_svc, vote_svc, p1, p2, p3):
    print("\n=== 2. Ranking Algorithms ===")

    for sort_name in ["hot", "top", "new", "controversial"]:
        feed = post_svc.get_feed(sub.subreddit_id, sort=sort_name)
        print(f"\n{sort_name.upper()} sort:")
        for i, p in enumerate(feed, 1):
            if sort_name == "hot":
                sort_score = RankingAlgorithm.hot_score(p.ups, p.downs, p.created_at)
            elif sort_name == "top":
                sort_score = RankingAlgorithm.top_score(p.ups, p.downs)
            elif sort_name == "controversial":
                sort_score = RankingAlgorithm.controversial_score(p.ups, p.downs)
            else:
                sort_score = p.created_at
            print(f"  #{i} (score={sort_score:.4f}) '{p.title[:40]}'")


def demonstrate_3_comments():
    print("\n=== 3. Threaded Comments ===")
    vote_svc = VoteService()
    post_svc = PostService(vote_svc)
    comment_svc = CommentService(vote_svc, post_svc)

    sub = Subreddit("sub_askreddit", "AskReddit", "Ask Reddit!", "")
    post = post_svc.submit(sub.subreddit_id, "user_alice",
                            "What's your biggest productivity hack?", PostType.TEXT)

    # Top-level comments
    c1 = comment_svc.add_comment(post.post_id, "user_bob",
                                   "Use Pomodoro technique - 25 min focus, 5 min break.")
    c2 = comment_svc.add_comment(post.post_id, "user_carol",
                                   "Just close Twitter. Seriously.")
    c3 = comment_svc.add_comment(post.post_id, "user_dave",
                                   "Sleep 8 hours. Everything else is noise.")

    # Replies
    r1 = comment_svc.add_comment(post.post_id, "user_eve", "This! Changed my life.",
                                   parent_id=c1.comment_id)
    r2 = comment_svc.add_comment(post.post_id, "user_frank", "+1 Twitter is a time sink",
                                   parent_id=c2.comment_id)

    # Votes
    for u in ["u1", "u2", "u3", "u4", "u5"]:
        comment_svc.vote_comment(c1.comment_id, u, VoteDirection.UP)
    comment_svc.vote_comment(c2.comment_id, "u1", VoteDirection.UP)
    comment_svc.vote_comment(c2.comment_id, "u2", VoteDirection.DOWN)
    for u in ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]:
        comment_svc.vote_comment(c3.comment_id, u, VoteDirection.UP)

    tree = comment_svc.get_comment_tree(post.post_id, sort="best")
    print(f"Comment tree (sorted by Wilson 'Best'):")
    for c in tree:
        indent = "  " * c.depth
        wilson = RankingAlgorithm.wilson_lower_bound(c.ups, c.downs)
        print(f"{indent}↑{c.ups} wilson={wilson:.3f} | {c.body[:50]}")


def demonstrate_4_automod():
    print("\n=== 4. AutoModerator ===")
    vote_svc = VoteService()
    post_svc = PostService(vote_svc)
    automod = AutoModerator()

    sub_id = "sub_investing"
    automod.add_rule(sub_id, AutoModRule(
        name="karma_gate",
        min_karma=100,
        min_account_age_days=30,
        domain_blocklist={"spam.com", "click-bait.net"},
    ))

    new_user = User("u_new", "newbie", post_karma=10, comment_karma=5)
    vet_user = User("u_vet", "veteran", post_karma=500, comment_karma=300)

    p_new = Post("p_test1", sub_id, new_user.user_id, "Buy this stock!",
                  PostType.LINK, "https://spam.com/hot-tip")
    p_vet = Post("p_test2", sub_id, vet_user.user_id, "My portfolio analysis",
                  PostType.TEXT, "Here's my Q1 review...")

    allowed1, reason1 = automod.evaluate(p_new, new_user, account_age_days=5)
    allowed2, reason2 = automod.evaluate(p_vet, vet_user, account_age_days=365)

    print(f"New user post: allowed={allowed1}, reason='{reason1}'")
    print(f"Veteran post: allowed={allowed2}, reason='{reason2}'")


def demonstrate_5_wilson_score():
    print("\n=== 5. Wilson Score vs Naive Ranking ===")
    comments = [
        ("Only 1 upvote, new", 1, 0),
        ("100 ups, 0 downs", 100, 0),
        ("100 ups, 1 down", 100, 1),
        ("1000 ups, 100 downs (controversial)", 1000, 100),
        ("10 ups, 5 downs (balanced low)", 10, 5),
    ]

    print(f"{'Comment':<45} {'Score':>6} {'Wilson':>8} {'Naive%':>8}")
    print("-" * 70)
    for label, ups, downs in comments:
        total = ups + downs
        naive_pct = ups / total if total else 0
        wilson = RankingAlgorithm.wilson_lower_bound(ups, downs)
        print(f"{label:<45} {ups - downs:>6} {wilson:>8.3f} {naive_pct:>8.1%}")


if __name__ == "__main__":
    sub, post_svc, vote_svc, p1, p2, p3 = demonstrate_1_subreddit_and_posting()
    demonstrate_2_ranking_algorithms(sub, post_svc, vote_svc, p1, p2, p3)
    demonstrate_3_comments()
    demonstrate_4_automod()
    demonstrate_5_wilson_score()
