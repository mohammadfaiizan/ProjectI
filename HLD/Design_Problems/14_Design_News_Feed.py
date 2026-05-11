"""
News Feed System - Core Implementation
Demonstrates: fan-out on write/read hybrid, feed ranking with time decay,
Redis sorted set simulation, cursor-based pagination, follow graph.
Standard library only.
"""

import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CELEBRITY_THRESHOLD = 5000   # followers above this -> pull model
MAX_FEED_DEPTH      = 500    # max post_ids stored per user timeline
FEED_TTL_SECONDS    = 7 * 86400  # 7 days
AFFINITY_DECAY_RATE = 0.1    # lambda for affinity time-decay


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Post:
    post_id:       str
    author_id:     str
    content:       str
    post_type:     str = "TEXT"   # TEXT, IMAGE, VIDEO, LINK
    like_count:    int = 0
    comment_count: int = 0
    share_count:   int = 0
    created_at:    float = field(default_factory=time.time)
    deleted_at:    Optional[float] = None

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    @property
    def engagement_score(self) -> float:
        """Weighted engagement: shares > comments > likes."""
        return self.like_count * 1.0 + self.comment_count * 3.0 + self.share_count * 5.0


@dataclass
class FeedItem:
    """An entry in a user's feed timeline (Redis sorted set equivalent)."""
    post_id: str
    score:   float  # ranking score; higher = shown sooner


# ---------------------------------------------------------------------------
# Redis-Like Sorted Set (Feed Timeline)
# ---------------------------------------------------------------------------

class SortedSet:
    """
    Simulates Redis ZADD/ZREVRANGE/ZRANGEBYSCORE/ZREM operations.
    Internally keeps a list sorted by score (descending).
    """

    def __init__(self):
        self._items: Dict[str, float] = {}  # member -> score

    def zadd(self, member: str, score: float):
        self._items[member] = score

    def zrem(self, member: str):
        self._items.pop(member, None)

    def zrevrange(self, start: int, stop: int) -> List[Tuple[str, float]]:
        """Return members [start:stop] ordered by descending score."""
        sorted_items = sorted(self._items.items(), key=lambda x: -x[1])
        return sorted_items[start:stop + 1] if stop >= 0 else sorted_items[start:]

    def zrevrangebyscore(self, max_score: float, min_score: float = 0.0,
                         limit: int = 20) -> List[Tuple[str, float]]:
        """Range by score (exclusive of max_score for cursor pagination)."""
        result = [
            (m, s) for m, s in self._items.items()
            if min_score <= s < max_score
        ]
        result.sort(key=lambda x: -x[1])
        return result[:limit]

    def zcard(self) -> int:
        return len(self._items)

    def ztrim_to_size(self, max_size: int):
        """Remove lowest-scored items beyond max_size."""
        if len(self._items) > max_size:
            sorted_items = sorted(self._items.items(), key=lambda x: x[1])
            to_remove = len(self._items) - max_size
            for member, _ in sorted_items[:to_remove]:
                del self._items[member]

    def get_score(self, member: str) -> Optional[float]:
        return self._items.get(member)

    def __len__(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------------
# Feed Ranker
# ---------------------------------------------------------------------------

class FeedRanker:
    """
    EdgeRank-inspired ranking formula:
      score = affinity_weight × content_type_weight × time_decay
    Time decay: 1 / (1 + age_hours) or exponential decay
    """

    CONTENT_TYPE_WEIGHTS = {
        "VIDEO":  3.0,
        "IMAGE":  2.0,
        "LINK":   1.5,
        "TEXT":   1.0,
    }
    TIME_DECAY_LAMBDA = 0.05  # e^(-λ × age_hours); smaller = slower decay

    def __init__(self):
        # (viewer_id, author_id) -> affinity score [0.0, 1.0]
        self._affinity: Dict[Tuple[str, str], float] = defaultdict(float)

    def compute_score(self, post: Post, viewer_id: str) -> float:
        """Compute ranking score for a post from viewer's perspective."""
        age_hours = (time.time() - post.created_at) / 3600
        affinity  = self._affinity.get((viewer_id, post.author_id), 0.2)  # default 0.2
        type_wt   = self.CONTENT_TYPE_WEIGHTS.get(post.post_type, 1.0)
        # Engagement boost (normalized; viral posts score higher)
        engagement = min(1.0, post.engagement_score / 100.0)
        # Time decay using exponential decay
        time_decay = math.exp(-self.TIME_DECAY_LAMBDA * age_hours)
        score = affinity * type_wt * (1.0 + engagement) * time_decay
        return round(score, 6)

    def record_interaction(self, viewer_id: str, author_id: str,
                           interaction_type: str):
        """
        Affinity increases based on interaction type.
        Interaction weights: like=1, comment=3, share=5, message=10.
        """
        weights = {"like": 1, "comment": 3, "share": 5, "message": 10}
        w = weights.get(interaction_type, 1)
        key = (viewer_id, author_id)
        # Exponential moving average (gives recency bias)
        current = self._affinity.get(key, 0.0)
        self._affinity[key] = min(1.0, current + w * 0.01)

    def rank(self, posts: List[Post], viewer_id: str) -> List[Tuple[Post, float]]:
        """Sort posts by descending score for a given viewer."""
        scored = [(p, self.compute_score(p, viewer_id)) for p in posts]
        scored.sort(key=lambda x: -x[1])
        return scored


# ---------------------------------------------------------------------------
# Follow Graph
# ---------------------------------------------------------------------------

class FollowGraph:
    """
    In-memory follow graph.
    Production: PostgreSQL source of truth + Redis adjacency sets.
    """

    def __init__(self):
        # user_id -> set of user_ids they follow
        self._following: Dict[str, Set[str]] = defaultdict(set)
        # user_id -> set of user_ids who follow them
        self._followers: Dict[str, Set[str]] = defaultdict(set)

    def follow(self, follower_id: str, followee_id: str) -> bool:
        if followee_id in self._following[follower_id]:
            return False  # Already following
        self._following[follower_id].add(followee_id)
        self._followers[followee_id].add(follower_id)
        return True

    def unfollow(self, follower_id: str, followee_id: str) -> bool:
        if followee_id not in self._following[follower_id]:
            return False
        self._following[follower_id].discard(followee_id)
        self._followers[followee_id].discard(follower_id)
        return True

    def get_followers(self, user_id: str) -> Set[str]:
        return self._followers.get(user_id, set())

    def get_following(self, user_id: str) -> Set[str]:
        return self._following.get(user_id, set())

    def follower_count(self, user_id: str) -> int:
        return len(self._followers.get(user_id, set()))

    def is_celebrity(self, user_id: str) -> bool:
        return self.follower_count(user_id) > CELEBRITY_THRESHOLD


# ---------------------------------------------------------------------------
# Fan-Out Service
# ---------------------------------------------------------------------------

class FanOutService:
    """
    Async fan-out: pushes new post_ids to follower timelines.
    In production: Kafka consumer subscribing to post.created.
    """

    def __init__(self, follow_graph: FollowGraph, timelines: Dict[str, SortedSet],
                 ranker: FeedRanker):
        self._graph     = follow_graph
        self._timelines = timelines
        self._ranker    = ranker

    def fan_out(self, post: Post):
        """Push post to all non-celebrity follower timelines."""
        if self._graph.is_celebrity(post.author_id):
            print(f"  [FAN-OUT] Skipping fan-out for celebrity {post.author_id} "
                  f"({self._graph.follower_count(post.author_id)} followers) — pull model")
            return

        followers = self._graph.get_followers(post.author_id)
        fan_out_count = 0
        for follower_id in followers:
            timeline = self._timelines.setdefault(follower_id, SortedSet())
            # Use author-neutral score for fan-out; viewer-specific re-rank happens at read
            base_score = self._ranker.compute_score(post, follower_id)
            timeline.zadd(post.post_id, base_score)
            timeline.ztrim_to_size(MAX_FEED_DEPTH)
            fan_out_count += 1

        if fan_out_count > 0:
            print(f"  [FAN-OUT] Post {post.post_id[:8]} pushed to "
                  f"{fan_out_count} follower timelines")

    def propagate_delete(self, post_id: str, author_id: str):
        """Remove a deleted post from all follower timelines."""
        followers = self._graph.get_followers(author_id)
        for follower_id in followers:
            if follower_id in self._timelines:
                self._timelines[follower_id].zrem(post_id)
        print(f"  [FAN-OUT] Post {post_id[:8]} removed from "
              f"{len(followers)} timelines")


# ---------------------------------------------------------------------------
# Main News Feed System
# ---------------------------------------------------------------------------

class NewsFeedSystem:

    def __init__(self):
        self._posts:     Dict[str, Post] = {}
        self._users:     Set[str] = set()
        self._mutes:     Dict[str, Set[str]] = defaultdict(set)  # user -> muted users
        self._blocks:    Dict[str, Set[str]] = defaultdict(set)

        # Core components
        self.graph     = FollowGraph()
        self.ranker    = FeedRanker()

        # Timeline cache: user_id -> SortedSet of (post_id, score)
        self._timelines: Dict[str, SortedSet] = {}

        # Author's own posts index: author_id -> SortedSet(post_id, timestamp)
        self._author_posts: Dict[str, SortedSet] = {}

        self.fan_out_svc = FanOutService(self.graph, self._timelines, self.ranker)

    def register_user(self, user_id: str):
        self._users.add(user_id)

    def create_post(self, author_id: str, content: str,
                    post_type: str = "TEXT") -> Post:
        post = Post(
            post_id=str(uuid.uuid4()),
            author_id=author_id,
            content=content,
            post_type=post_type,
        )
        self._posts[post.post_id] = post

        # Update author's own post index
        author_ts = self._author_posts.setdefault(author_id, SortedSet())
        author_ts.zadd(post.post_id, post.created_at)

        # Fan-out to followers (push model for non-celebrities)
        self.fan_out_svc.fan_out(post)

        return post

    def delete_post(self, post_id: str, author_id: str) -> bool:
        post = self._posts.get(post_id)
        if not post or post.author_id != author_id:
            return False
        post.deleted_at = time.time()
        # Propagate delete to follower timelines
        self.fan_out_svc.propagate_delete(post_id, author_id)
        return True

    def follow_user(self, follower_id: str, followee_id: str) -> bool:
        result = self.graph.follow(follower_id, followee_id)
        if result:
            # Warm the follower's feed with recent posts from new followee
            self._backfill_feed(follower_id, followee_id, limit=20)
        return result

    def unfollow_user(self, follower_id: str, followee_id: str) -> bool:
        result = self.graph.unfollow(follower_id, followee_id)
        if result:
            # Remove unfollowed user's posts from timeline
            followee_posts_set = self._author_posts.get(followee_id, SortedSet())
            if follower_id in self._timelines:
                for post_id, _ in followee_posts_set.zrevrange(0, -1):
                    self._timelines[follower_id].zrem(post_id)
        return result

    def get_feed(self, user_id: str, limit: int = 20,
                 cursor: Optional[float] = None) -> Tuple[List[Dict], Optional[float]]:
        """
        Returns (posts_list, next_cursor).
        Hybrid: precomputed timeline + celebrity pull + re-ranking.
        """
        max_score = cursor if cursor is not None else float("inf")

        # 1. Fetch from precomputed timeline (non-celebrities)
        timeline = self._timelines.get(user_id, SortedSet())
        candidate_items = timeline.zrevrangebyscore(max_score, limit=limit * 3)

        # 2. Pull recent posts from followed celebrities (merge at read time)
        celebrity_posts = self._pull_celebrity_posts(user_id, max_posts_per=5)

        # 3. Combine all candidates
        candidate_ids = {post_id for post_id, _ in candidate_items}
        for p in celebrity_posts:
            candidate_ids.add(p.post_id)

        # 4. Resolve posts and filter
        candidates = []
        for post_id in candidate_ids:
            post = self._posts.get(post_id)
            if post and not post.is_deleted:
                if post.author_id not in self._mutes.get(user_id, set()):
                    if post.author_id not in self._blocks.get(user_id, set()):
                        candidates.append(post)

        # 5. Re-rank candidates for this specific viewer
        ranked = self.ranker.rank(candidates, user_id)

        # 6. Apply cursor filter (score strictly less than cursor)
        if cursor is not None:
            ranked = [(p, s) for p, s in ranked if s < cursor]

        # 7. Paginate
        page = ranked[:limit]
        next_cursor = page[-1][1] if len(page) == limit else None

        result = [
            {
                "post_id": p.post_id,
                "author_id": p.author_id,
                "content": p.content[:80],  # truncate for display
                "post_type": p.post_type,
                "like_count": p.like_count,
                "score": round(s, 4),
                "age_minutes": round((time.time() - p.created_at) / 60, 1),
            }
            for p, s in page
        ]
        return result, next_cursor

    def _pull_celebrity_posts(self, user_id: str,
                              max_posts_per: int = 5) -> List[Post]:
        """Pull recent posts from celebrity accounts the user follows."""
        following = self.graph.get_following(user_id)
        celebrity_posts = []
        for followee_id in following:
            if self.graph.is_celebrity(followee_id):
                author_posts = self._author_posts.get(followee_id, SortedSet())
                recent = author_posts.zrevrange(0, max_posts_per - 1)
                for post_id, _ in recent:
                    post = self._posts.get(post_id)
                    if post and not post.is_deleted:
                        celebrity_posts.append(post)
        return celebrity_posts

    def _backfill_feed(self, user_id: str, followee_id: str, limit: int = 20):
        """Add recent posts from a new followee to the user's timeline."""
        author_posts = self._author_posts.get(followee_id, SortedSet())
        recent = author_posts.zrevrange(0, limit - 1)
        timeline = self._timelines.setdefault(user_id, SortedSet())
        for post_id, ts in recent:
            post = self._posts.get(post_id)
            if post and not post.is_deleted:
                score = self.ranker.compute_score(post, user_id)
                timeline.zadd(post_id, score)

    def like_post(self, user_id: str, post_id: str):
        post = self._posts.get(post_id)
        if post:
            post.like_count += 1
            self.ranker.record_interaction(user_id, post.author_id, "like")

    def comment_post(self, user_id: str, post_id: str):
        post = self._posts.get(post_id)
        if post:
            post.comment_count += 1
            self.ranker.record_interaction(user_id, post.author_id, "comment")

    def mute_user(self, user_id: str, target_id: str):
        self._mutes[user_id].add(target_id)

    def block_user(self, user_id: str, target_id: str):
        self._blocks[user_id].add(target_id)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_posts": len(self._posts),
            "total_users": len(self._users),
            "cached_timelines": len(self._timelines),
            "total_timeline_entries": sum(len(t) for t in self._timelines.values()),
        }


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_basic_feed():
    print("=== Basic News Feed Demo ===")
    system = NewsFeedSystem()

    for uid in ["alice", "bob", "carol", "dave"]:
        system.register_user(uid)

    # Setup: alice and bob follow carol; alice follows dave
    system.follow_user("alice", "carol")
    system.follow_user("alice", "dave")
    system.follow_user("bob", "carol")

    # Carol posts 3 times
    p1 = system.create_post("carol", "Hello world! My first post.", "TEXT")
    time.sleep(0.01)  # slight time gap for ordering
    p2 = system.create_post("carol", "Check out this amazing photo!", "IMAGE")
    time.sleep(0.01)
    p3 = system.create_post("dave", "Video tutorial on system design", "VIDEO")

    # Alice likes carol's image post (increases affinity)
    system.like_post("alice", p2.post_id)
    system.like_post("alice", p2.post_id)  # like twice to boost score
    system.comment_post("alice", p2.post_id)

    # Alice's feed
    feed, next_cursor = system.get_feed("alice", limit=10)
    print(f"\nAlice's feed ({len(feed)} posts):")
    for item in feed:
        print(f"  [{item['post_type']:<6}] {item['content'][:50]} "
              f"| score={item['score']:.4f}")

    # Bob's feed (should see carol's posts but not dave's)
    bob_feed, _ = system.get_feed("bob", limit=10)
    print(f"\nBob's feed ({len(bob_feed)} posts, should not include Dave's):")
    for item in bob_feed:
        print(f"  author={item['author_id']:<8} | {item['content'][:50]}")

    print(f"\nSystem stats: {system.stats()}")


def demo_celebrity_hybrid_fanout():
    print("\n=== Celebrity Hybrid Fan-Out Demo ===")
    system = NewsFeedSystem()

    # Create a celebrity (more followers than threshold)
    celebrity_id = "popstar_gaga"
    system.register_user(celebrity_id)
    for i in range(CELEBRITY_THRESHOLD + 10):
        fan_id = f"fan_{i:05d}"
        system.register_user(fan_id)
        system.graph.follow(fan_id, celebrity_id)

    # One regular user follows the celebrity
    system.register_user("regular_alice")
    system.follow_user("regular_alice", celebrity_id)

    # Also add some regular user posts to alice's feed
    system.register_user("regular_bob")
    system.follow_user("regular_alice", "regular_bob")
    system.create_post("regular_bob", "Bob's regular post", "TEXT")

    print(f"Celebrity '{celebrity_id}' has "
          f"{system.graph.follower_count(celebrity_id)} followers "
          f"(threshold={CELEBRITY_THRESHOLD})")
    print(f"Is celebrity: {system.graph.is_celebrity(celebrity_id)}")

    # Celebrity posts — fan-out should be SKIPPED
    celeb_post = system.create_post(celebrity_id, "New album out NOW!", "VIDEO")

    # Alice's feed should include celebrity post via PULL at read time
    feed, _ = system.get_feed("regular_alice", limit=10)
    print(f"\nAlice's feed includes celebrity post: "
          f"{any(item['author_id'] == celebrity_id for item in feed)}")
    for item in feed:
        print(f"  author={item['author_id']:<20} | {item['content'][:50]}")


def demo_cursor_pagination():
    print("\n=== Cursor-Based Pagination Demo ===")
    system = NewsFeedSystem()
    for uid in ["reader", "writer"]:
        system.register_user(uid)
    system.follow_user("reader", "writer")

    # Create 15 posts
    for i in range(15):
        system.create_post("writer", f"Post number {i+1:02d} — some content here", "TEXT")
        time.sleep(0.001)

    print("Fetching feed page by page (5 per page)...")
    cursor = None
    page_num = 0
    total_fetched = 0
    seen_ids = set()

    while True:
        page_num += 1
        feed, cursor = system.get_feed("reader", limit=5, cursor=cursor)
        if not feed:
            break
        total_fetched += len(feed)
        for item in feed:
            seen_ids.add(item["post_id"])
        scores = [item["score"] for item in feed]
        print(f"  Page {page_num}: {len(feed)} posts | "
              f"scores {scores[0]:.4f} -> {scores[-1]:.4f} (desc) | "
              f"next_cursor={round(cursor, 4) if cursor else None}")
        if cursor is None:
            break

    print(f"Total posts fetched: {total_fetched} (no duplicates: {len(seen_ids) == total_fetched})")


def demo_mute_and_delete():
    print("\n=== Mute & Delete Demo ===")
    system = NewsFeedSystem()
    for uid in ["alice", "bob", "carol"]:
        system.register_user(uid)
    system.follow_user("alice", "bob")
    system.follow_user("alice", "carol")

    p_bob = system.create_post("bob", "Bob's post that Alice will mute", "TEXT")
    p_carol = system.create_post("carol", "Carol's visible post", "TEXT")

    feed_before, _ = system.get_feed("alice")
    print(f"Feed before mute: {len(feed_before)} posts")

    # Alice mutes bob
    system.mute_user("alice", "bob")
    feed_after, _ = system.get_feed("alice")
    print(f"Feed after muting Bob: {len(feed_after)} posts "
          f"(Bob's posts excluded: "
          f"{not any(i['author_id']=='bob' for i in feed_after)})")

    # Carol deletes her post
    system.delete_post(p_carol.post_id, "carol")
    feed_after_delete, _ = system.get_feed("alice")
    print(f"Feed after Carol deletes her post: {len(feed_after_delete)} posts")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_basic_feed()
    demo_celebrity_hybrid_fanout()
    demo_cursor_pagination()
    demo_mute_and_delete()
