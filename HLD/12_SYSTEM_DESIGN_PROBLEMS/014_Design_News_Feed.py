"""
SYSTEM DESIGN: NEWS FEED / ACTIVITY FEED
==========================================

Problem Statement:
Design a personalized news feed that aggregates activities from
connections (Facebook News Feed, LinkedIn feed) and ranks them by relevance.

Functional Requirements:
  - Post content (text, image, link)
  - Like, comment, share posts
  - Follow users / connect with friends
  - Ranked news feed (not purely chronological)
  - Pagination of feed

Non-Functional Requirements:
  - 500M DAU (Facebook scale)
  - 100B feed reads/day → ~1.15M reads/sec
  - Feed load < 500ms (p99)
  - Feed freshness: new posts visible within 30s

Feed Generation (detailed):
  See also: 002_Design_Instagram.py (push/pull)
  Facebook uses a 3-stage pipeline:
    1. Candidate Generation: pull ~5000 stories from followed sources.
    2. Filtering/Integrity: spam, hate speech, blocked users.
    3. Ranking: ML model scores each story (EdgeRank → AI).
    4. Heuristic Adjustments: diversity, promotions, suggestions.

EdgeRank (legacy Facebook ranking):
  score = Σ (affinity × weight × time_decay)
  affinity:   strength of relationship (interaction history).
  weight:     type of action (video > photo > text > like).
  time_decay: e^(-λt) — older stories ranked lower.

Modern Feed Ranking:
  Features: user_engagement_history, post_type, creator_engagement_rate,
            recency, diversity (same creator not shown 3x in a row).
  Two-tower model: user embedding × item embedding → score.
  Served by ML platform: 50ms inference at scale.

Activity Types:
  CREATED: user creates a post
  LIKED:   user likes a post
  COMMENTED: user comments on a post
  SHARED:  user shared a post
  FOLLOWED: user follows another user

Aggregation:
  "Alice and 3 others liked Bob's post"
  Aggregate same activity_type + object_id within 30 minutes.
  Show one aggregated story instead of N individual stories.
"""

from __future__ import annotations

import time
import uuid
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import defaultdict


# ─────────────────────────────────────────────
# ACTIVITY TYPE
# ─────────────────────────────────────────────

class ActivityType(Enum):
    CREATED   = "created"
    LIKED     = "liked"
    COMMENTED = "commented"
    SHARED    = "shared"
    FOLLOWED  = "followed"

    @property
    def weight(self) -> float:
        """EdgeRank weight."""
        return {
            "created":   3.0,
            "shared":    2.5,
            "commented": 2.0,
            "liked":     1.0,
            "followed":  0.5,
        }[self.value]


# ─────────────────────────────────────────────
# POST
# ─────────────────────────────────────────────

class PostType(Enum):
    TEXT  = "text"
    IMAGE = "image"
    VIDEO = "video"
    LINK  = "link"

    @property
    def base_weight(self) -> float:
        return {"video": 3.0, "image": 2.0, "link": 1.5, "text": 1.0}[self.value]


@dataclass
class Post:
    post_id:    str
    author_id:  str
    content:    str
    post_type:  PostType
    created_at: float
    like_count:  int = 0
    comment_count: int = 0
    share_count:   int = 0

    @property
    def engagement_rate(self) -> float:
        total = self.like_count + self.comment_count * 2 + self.share_count * 3
        return math.log1p(total)


# ─────────────────────────────────────────────
# ACTIVITY
# ─────────────────────────────────────────────

@dataclass
class Activity:
    activity_id:   str
    actor_id:      str
    activity_type: ActivityType
    object_id:     str    # post_id or user_id
    object_type:   str    # post / user
    created_at:    float


# ─────────────────────────────────────────────
# FEED STORY (aggregated activity)
# ─────────────────────────────────────────────

@dataclass
class FeedStory:
    story_id:       str
    activity_type:  ActivityType
    actors:         List[str]    # user_ids
    object_id:      str
    object_type:    str
    created_at:     float
    score:          float = 0.0

    def display_text(self, user_store: "UserStore") -> str:
        names = [user_store.name(a) for a in self.actors[:2]]
        if len(self.actors) > 2:
            names.append(f"and {len(self.actors) - 2} others")
        actors_str = ", ".join(names)
        return f"{actors_str} {self.activity_type.value} [{self.object_id[:8]}]"


# ─────────────────────────────────────────────
# USER STORE
# ─────────────────────────────────────────────

class UserStore:
    def __init__(self):
        self._users:   Dict[str, str]      = {}   # user_id → name
        self._follows: Dict[str, Set[str]] = defaultdict(set)
        self._affinity: Dict[Tuple[str,str], float] = {}  # (viewer, creator) → affinity

    def add(self, user_id: str, name: str):
        self._users[user_id] = name

    def name(self, user_id: str) -> str:
        return self._users.get(user_id, user_id[:8])

    def follow(self, follower: str, followee: str):
        self._follows[follower].add(followee)

    def followees(self, user_id: str) -> Set[str]:
        return self._follows.get(user_id, set())

    def set_affinity(self, viewer: str, creator: str, score: float):
        self._affinity[(viewer, creator)] = score

    def affinity(self, viewer: str, creator: str) -> float:
        return self._affinity.get((viewer, creator), 1.0)


# ─────────────────────────────────────────────
# ACTIVITY AGGREGATOR
# ─────────────────────────────────────────────

class ActivityAggregator:
    """
    Groups similar activities into feed stories.
    "Alice and Bob liked the same post" = 1 story, not 2.
    """

    WINDOW_S = 1800   # 30 minutes aggregation window

    def __init__(self):
        # (activity_type, object_id) → FeedStory
        self._stories: Dict[Tuple, FeedStory] = {}

    def ingest(self, activity: Activity) -> FeedStory:
        key = (activity.activity_type, activity.object_id)
        now = time.time()

        story = self._stories.get(key)
        if story and (now - story.created_at) < self.WINDOW_S:
            # Aggregate: add actor if not already there
            if activity.actor_id not in story.actors:
                story.actors.append(activity.actor_id)
                story.created_at = now   # refresh timestamp
        else:
            # New story
            story = FeedStory(
                story_id      = uuid.uuid4().hex[:10],
                activity_type = activity.activity_type,
                actors        = [activity.actor_id],
                object_id     = activity.object_id,
                object_type   = activity.object_type,
                created_at    = now,
            )
            self._stories[key] = story

        return story


# ─────────────────────────────────────────────
# EDGE RANK SCORER
# ─────────────────────────────────────────────

class EdgeRankScorer:
    """
    Classic EdgeRank: affinity × weight × time_decay.
    Modern systems use full ML models.
    """

    DECAY_LAMBDA = 0.1   # controls how fast old stories decay

    def score(self, story: FeedStory, viewer_id: str,
              posts: Dict[str, Post],
              user_store: UserStore) -> float:
        # Time decay: e^(-λ * hours_old)
        hours_old  = (time.time() - story.created_at) / 3600
        time_decay = math.exp(-self.DECAY_LAMBDA * hours_old)

        # Action weight
        weight = story.activity_type.weight

        # Affinity: avg affinity between viewer and actors
        affinities = [user_store.affinity(viewer_id, actor)
                      for actor in story.actors]
        affinity   = sum(affinities) / max(len(affinities), 1)

        # Post type boost
        post = posts.get(story.object_id)
        type_boost = post.post_type.base_weight if post else 1.0

        # Social proof: more actors = higher score
        social_proof = math.log1p(len(story.actors))

        return affinity * weight * type_decay * type_boost * social_proof

    def score(self, story: FeedStory, viewer_id: str,
              posts: Dict[str, Post],
              user_store: UserStore) -> float:
        hours_old  = (time.time() - story.created_at) / 3600
        time_decay = math.exp(-self.DECAY_LAMBDA * max(hours_old, 0))
        weight     = story.activity_type.weight
        affinities = [user_store.affinity(viewer_id, a) for a in story.actors]
        affinity   = sum(affinities) / max(len(affinities), 1)
        post       = posts.get(story.object_id)
        type_boost = post.post_type.base_weight if post else 1.0
        social_proof = math.log1p(len(story.actors))
        return affinity * weight * time_decay * type_boost * social_proof


# ─────────────────────────────────────────────
# FEED SERVICE
# ─────────────────────────────────────────────

class NewsFeedService:
    def __init__(self):
        self._users      = UserStore()
        self._posts:     Dict[str, Post] = {}
        self._activities: List[Activity] = []
        self._aggregator = ActivityAggregator()
        self._scorer     = EdgeRankScorer()
        # Pre-computed feed: user_id → [FeedStory] (sorted by score)
        self._feeds:     Dict[str, List[FeedStory]] = defaultdict(list)

    def create_user(self, name: str) -> str:
        uid = uuid.uuid4().hex[:8]
        self._users.add(uid, name)
        return uid

    def follow(self, follower: str, followee: str):
        self._users.follow(follower, followee)

    def create_post(self, author_id: str, content: str,
                    post_type: PostType = PostType.TEXT) -> Post:
        post = Post(uuid.uuid4().hex[:10], author_id, content,
                    post_type, time.time())
        self._posts[post.post_id] = post
        self._record_activity(author_id, ActivityType.CREATED,
                              post.post_id, "post")
        return post

    def like(self, actor_id: str, post_id: str):
        post = self._posts.get(post_id)
        if post:
            post.like_count += 1
        self._record_activity(actor_id, ActivityType.LIKED, post_id, "post")

    def comment(self, actor_id: str, post_id: str, text: str):
        post = self._posts.get(post_id)
        if post:
            post.comment_count += 1
        self._record_activity(actor_id, ActivityType.COMMENTED, post_id, "post")

    def share(self, actor_id: str, post_id: str):
        post = self._posts.get(post_id)
        if post:
            post.share_count += 1
        self._record_activity(actor_id, ActivityType.SHARED, post_id, "post")

    def _record_activity(self, actor_id: str, activity_type: ActivityType,
                         object_id: str, object_type: str):
        activity = Activity(uuid.uuid4().hex[:8], actor_id, activity_type,
                            object_id, object_type, time.time())
        self._activities.append(activity)
        story = self._aggregator.ingest(activity)

        # Push story to followers' feeds
        author_followers = [uid for uid, followees in self._users._follows.items()
                            if actor_id in followees]
        for fid in author_followers:
            # Avoid duplicates
            if story not in self._feeds[fid]:
                self._feeds[fid].append(story)

    def get_feed(self, user_id: str, limit: int = 20) -> List[FeedStory]:
        stories = self._feeds.get(user_id, [])

        # Score each story
        for story in stories:
            story.score = self._scorer.score(
                story, user_id, self._posts, self._users)

        # Sort by score (descending)
        sorted_stories = sorted(stories, key=lambda s: -s.score)
        return sorted_stories[:limit]


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_news_feed():
    print("=" * 65)
    print("SYSTEM DESIGN: NEWS FEED")
    print("=" * 65)

    random.seed(42)
    svc = NewsFeedService()

    # ── Users ─────────────────────────────────
    print("\n[1] USERS AND CONNECTIONS")
    print("─" * 55)

    alice = svc.create_user("Alice")
    bob   = svc.create_user("Bob")
    carol = svc.create_user("Carol")
    dave  = svc.create_user("Dave")

    svc.follow(alice, bob)
    svc.follow(alice, carol)
    svc.follow(bob, carol)
    svc.follow(bob, alice)

    # Set affinities (Alice closely connected to Bob)
    svc._users.set_affinity(alice, bob,   3.0)
    svc._users.set_affinity(alice, carol, 1.5)
    svc._users.set_affinity(alice, dave,  0.5)

    print(f"  Alice follows: {[svc._users.name(u) for u in svc._users.followees(alice)]}")
    print(f"  Bob follows:   {[svc._users.name(u) for u in svc._users.followees(bob)]}")

    # ── Posts ─────────────────────────────────
    print("\n[2] POSTS")
    print("─" * 55)

    p1 = svc.create_post(bob,   "Just launched my new Python library 🐍 #opensource",
                          PostType.LINK)
    time.sleep(0.001)
    p2 = svc.create_post(carol, "Beautiful sunset today 🌅",
                          PostType.IMAGE)
    time.sleep(0.001)
    p3 = svc.create_post(bob,   "Check out this amazing system design video!",
                          PostType.VIDEO)

    for p in [p1, p2, p3]:
        author = svc._users.name(p.author_id)
        print(f"  @{author}: [{p.post_type.value}] {p.content[:50]}")

    # ── Engagement ────────────────────────────
    print("\n[3] ENGAGEMENT (likes, comments, shares)")
    print("─" * 55)

    # Multiple people engage with Bob's posts
    svc.like(carol, p1.post_id)
    svc.like(dave, p1.post_id)
    svc.like(alice, p1.post_id)
    svc.comment(carol, p1.post_id, "Amazing work!")
    svc.share(alice, p3.post_id)

    print(f"  '{p1.content[:40]}...'")
    print(f"    Likes: {p1.like_count}  Comments: {p1.comment_count}")

    # ── Feed Aggregation ──────────────────────
    print("\n[4] ACTIVITY AGGREGATION")
    print("─" * 55)

    # Show how multiple likes are aggregated
    agg_key = (ActivityType.LIKED, p1.post_id)
    story = svc._aggregator._stories.get(agg_key)
    if story:
        print(f"  Aggregated story for p1 likes:")
        print(f"    actors: {[svc._users.name(a) for a in story.actors]}")
        print(f"    → '{story.display_text(svc._users)}'")

    # ── Ranked Feed ───────────────────────────
    print("\n[5] ALICE'S RANKED NEWS FEED")
    print("─" * 55)

    feed = svc.get_feed(alice, limit=10)
    print(f"  {len(feed)} stories in feed (ranked by EdgeRank score):")
    for i, story in enumerate(feed[:6], 1):
        text = story.display_text(svc._users)
        post = svc._posts.get(story.object_id)
        type_str = f"[{post.post_type.value}]" if post else ""
        print(f"  {i}. [score={story.score:.3f}] {text} {type_str}")

    # ── EdgeRank Factors ──────────────────────
    print("\n[6] EDGERANK SCORING FACTORS")
    print("─" * 55)

    print("  score = affinity × action_weight × time_decay × type_boost × social_proof")
    print()
    for at in ActivityType:
        print(f"  {at.value:<12} weight={at.weight:.1f}")

    print()
    for pt in PostType:
        print(f"  {pt.value:<8} type_boost={pt.base_weight:.1f}")

    # ── Architecture ──────────────────────────
    print("\n[7] NEWS FEED ARCHITECTURE")
    print("─" * 55)

    arch = [
        ("Write path",     "Activity → Kafka → Fan-out service → Redis feed list"),
        ("Read path",      "Fetch pre-computed feed → rank scores → serve"),
        ("Fan-out",        "Push to followers' feeds (except celebrities)"),
        ("Ranking",        "ML model: 50ms inference at read time"),
        ("Aggregation",    "Merge similar stories: dedup by (type, object_id, 30min)"),
        ("Cache",          "Redis: feed:{user_id} sorted set; 800 story limit"),
        ("Pagination",     "Cursor-based: score of last seen story"),
        ("Freshness",      "New posts visible in ~30s via fan-out pipeline"),
        ("ML features",    "Affinity, recency, engagement rate, content type"),
        ("Diversity",      "Re-rank to avoid 3+ consecutive posts from same author"),
    ]
    for component, detail in arch:
        print(f"  {component:<18} {detail}")


if __name__ == "__main__":
    demonstrate_news_feed()
