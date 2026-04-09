"""
SYSTEM DESIGN: INSTAGRAM (Photo Sharing)
==========================================

Problem Statement:
Design a photo/video sharing platform where users can upload media,
follow others, and see a feed of recent posts from people they follow.

Functional Requirements:
  - Upload photo/video
  - Follow / unfollow users
  - View home feed (posts from followed users, ordered by recency)
  - View profile (user's posts)
  - Like and comment on posts
  - Search by hashtag / username

Non-Functional Requirements:
  - 1B daily active users (DAU)
  - 100M photos uploaded/day → ~1150 photo uploads/sec
  - Feed read: 10× writes → 11500 feed reads/sec
  - Photos must be durable (object storage, no loss)
  - Feed read latency < 200ms

Estimation:
  Photos: 1MB average → 100M × 1MB = 100TB/day; 36PB/year
  Metadata per photo: 500B → 100M × 500B = 50GB/day
  Feed DB row: 100 bytes, 10B rows total

Data Model:
  users:     user_id, username, bio, profile_pic_url, follower_count
  posts:     post_id, user_id, media_url, caption, created_at, like_count
  follows:   follower_id, followee_id, created_at
  likes:     post_id, user_id, created_at
  comments:  comment_id, post_id, user_id, text, created_at
  feed:      user_id, post_id, score (pre-computed feed table)

Feed Generation Strategies:
  PUSH (fan-out on write):
    On post: write post_id to each follower's feed table.
    Read: simple SELECT from feed table.
    Problem: celebrity with 100M followers → 100M writes per post.
    Solution: hybrid — push for regular users, pull for celebrities.

  PULL (fan-out on read):
    On read: collect post_ids from all followees, merge, sort by time.
    Problem: user follows 1000 people → 1000 DB queries on each feed load.
    Solution: pre-join with Redis sorted sets.

  HYBRID (Instagram's approach):
    Celebrities (>10K followers): PULL model.
    Regular users:                PUSH model.
    On feed read: merge pre-computed feed + celebrity posts.

Media Storage:
  Upload: client → API server → S3 / GCS (multipart upload).
  CDN: CloudFront / Fastly distributes photos globally.
  Thumbnails: generated asynchronously by worker (Lambda).
  Formats: JPEG for photos, HLS for video.
"""

from __future__ import annotations

import time
import uuid
import heapq
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class User:
    user_id:        str
    username:       str
    follower_count: int = 0
    following_count: int = 0

    @property
    def is_celebrity(self) -> bool:
        return self.follower_count >= 10_000


@dataclass
class Post:
    post_id:    str
    user_id:    str
    media_url:  str
    caption:    str
    created_at: float
    like_count: int = 0

    def to_feed_item(self) -> "FeedItem":
        return FeedItem(self.post_id, self.user_id, self.created_at)


@dataclass
class FeedItem:
    post_id:    str
    author_id:  str
    score:      float    # typically created_at (recency-based)

    def __lt__(self, other: "FeedItem"):
        return self.score > other.score   # max-heap by score


# ─────────────────────────────────────────────
# USER STORE
# ─────────────────────────────────────────────

class UserStore:
    def __init__(self):
        self._users:    Dict[str, User]          = {}
        self._follows:  Dict[str, Set[str]]      = defaultdict(set)  # follower → followees
        self._followers: Dict[str, Set[str]]     = defaultdict(set)  # user → followers

    def create_user(self, username: str, follower_count: int = 0) -> User:
        uid = uuid.uuid4().hex[:12]
        u   = User(uid, username, follower_count)
        self._users[uid] = u
        return u

    def follow(self, follower_id: str, followee_id: str):
        self._follows[follower_id].add(followee_id)
        self._followers[followee_id].add(follower_id)
        self._users[follower_id].following_count += 1
        self._users[followee_id].follower_count  += 1

    def get_followees(self, user_id: str) -> Set[str]:
        return self._follows.get(user_id, set())

    def get_followers(self, user_id: str) -> Set[str]:
        return self._followers.get(user_id, set())

    def get(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)


# ─────────────────────────────────────────────
# POST STORE
# ─────────────────────────────────────────────

class PostStore:
    def __init__(self):
        self._posts:    Dict[str, Post]          = {}
        self._by_user:  Dict[str, List[Post]]    = defaultdict(list)

    def create(self, user_id: str, media_url: str, caption: str) -> Post:
        post = Post(uuid.uuid4().hex[:12], user_id, media_url,
                    caption, time.time())
        self._posts[post.post_id] = post
        self._by_user[user_id].append(post)
        return post

    def get(self, post_id: str) -> Optional[Post]:
        return self._posts.get(post_id)

    def get_user_posts(self, user_id: str, limit: int = 20) -> List[Post]:
        posts = self._by_user.get(user_id, [])
        return sorted(posts, key=lambda p: -p.created_at)[:limit]


# ─────────────────────────────────────────────
# FEED SERVICE (hybrid push/pull)
# ─────────────────────────────────────────────

class FeedService:
    """
    Hybrid feed generation:
    - Regular users: push (fan-out on write) to pre-computed feed
    - Celebrities: pull on read (merged into feed)
    """

    CELEBRITY_THRESHOLD = 10_000
    FEED_MAX_SIZE       = 500   # entries per user feed

    def __init__(self, user_store: UserStore, post_store: PostStore):
        self._users      = user_store
        self._posts      = post_store
        # Pre-computed feed: user_id → sorted list of FeedItems (newest first)
        self._feeds: Dict[str, List[FeedItem]] = defaultdict(list)

    def on_new_post(self, post: Post):
        """
        Fan-out on write for regular users.
        Skip fan-out for celebrities (handled at read time).
        """
        author = self._users.get(post.user_id)
        if not author or author.is_celebrity:
            return   # celebrity → skip fan-out (handled at read)

        # Push to each follower's feed
        followers = self._users.get_followers(post.user_id)
        item      = post.to_feed_item()
        for follower_id in followers:
            feed = self._feeds[follower_id]
            feed.append(item)
            # Keep feed size bounded (evict oldest)
            if len(feed) > self.FEED_MAX_SIZE:
                feed.sort(key=lambda x: -x.score)
                self._feeds[follower_id] = feed[:self.FEED_MAX_SIZE]

    def get_feed(self, user_id: str, limit: int = 20) -> List[Post]:
        """
        Read feed for user:
        1. Load pre-computed feed (pushed items from regular users)
        2. Merge celebrity posts (pulled at read time)
        3. Sort by recency, return top `limit`
        """
        pre_feed = list(self._feeds.get(user_id, []))

        # Pull celebrity posts
        followees = self._users.get_followees(user_id)
        for fid in followees:
            user = self._users.get(fid)
            if user and user.is_celebrity:
                celebrity_posts = self._posts.get_user_posts(fid, limit=10)
                for p in celebrity_posts:
                    pre_feed.append(p.to_feed_item())

        # Merge and sort
        pre_feed.sort(key=lambda x: -x.score)
        top_items = pre_feed[:limit]

        # Hydrate post objects
        result = []
        for item in top_items:
            post = self._posts.get(item.post_id)
            if post:
                result.append(post)
        return result


# ─────────────────────────────────────────────
# LIKE SERVICE
# ─────────────────────────────────────────────

class LikeService:
    def __init__(self, post_store: PostStore):
        self._posts  = post_store
        self._likes: Dict[str, Set[str]] = defaultdict(set)  # post_id → user_ids

    def like(self, post_id: str, user_id: str) -> bool:
        if user_id in self._likes[post_id]:
            return False   # already liked
        self._likes[post_id].add(user_id)
        post = self._posts.get(post_id)
        if post:
            post.like_count += 1
        return True

    def unlike(self, post_id: str, user_id: str) -> bool:
        if user_id not in self._likes[post_id]:
            return False
        self._likes[post_id].discard(user_id)
        post = self._posts.get(post_id)
        if post:
            post.like_count -= 1
        return True

    def like_count(self, post_id: str) -> int:
        return len(self._likes[post_id])


# ─────────────────────────────────────────────
# MEDIA UPLOAD SIMULATION
# ─────────────────────────────────────────────

@dataclass
class MediaUploadResult:
    media_url:     str
    thumbnail_url: str
    size_kb:       int
    format:        str

def simulate_upload(filename: str, size_kb: int) -> MediaUploadResult:
    """Simulates upload to S3 + CDN URL."""
    ext  = filename.split(".")[-1].lower()
    fid  = uuid.uuid4().hex[:16]
    cdn  = f"https://cdn.instagram.com/media/{fid}.{ext}"
    thumb= f"https://cdn.instagram.com/thumbs/{fid}_320x320.jpg"
    return MediaUploadResult(cdn, thumb, size_kb, ext)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_instagram():
    print("=" * 65)
    print("SYSTEM DESIGN: INSTAGRAM")
    print("=" * 65)

    user_store = UserStore()
    post_store = PostStore()
    feed_svc   = FeedService(user_store, post_store)
    like_svc   = LikeService(post_store)

    # ── Create Users ──────────────────────────
    print("\n[1] USERS")
    print("─" * 55)

    alice   = user_store.create_user("alice", follower_count=500)
    bob     = user_store.create_user("bob",   follower_count=200)
    celeb   = user_store.create_user("celebrity_chef", follower_count=5_000_000)

    # Alice and Bob follow the celebrity
    user_store.follow(alice.user_id, celeb.user_id)
    user_store.follow(bob.user_id,   celeb.user_id)
    user_store.follow(alice.user_id, bob.user_id)

    for u in [alice, bob, celeb]:
        print(f"  @{u.username:<20} followers={u.follower_count:>10,}  "
              f"celebrity={u.is_celebrity}")

    # ── Upload Media ──────────────────────────
    print("\n[2] MEDIA UPLOAD")
    print("─" * 55)

    upload = simulate_upload("vacation.jpg", 1024)
    print(f"  Uploaded: {upload.media_url}")
    print(f"  Thumbnail: {upload.thumbnail_url}")
    print(f"  Size: {upload.size_kb}KB  Format: {upload.format}")

    # ── Posts ─────────────────────────────────
    print("\n[3] POSTS AND FEED FAN-OUT")
    print("─" * 55)

    # Bob posts (regular user → push to Alice's feed)
    p1 = post_store.create(bob.user_id, upload.media_url, "My lunch today 🍕")
    feed_svc.on_new_post(p1)
    print(f"  @bob posts → pushed to {len(user_store.get_followers(bob.user_id))} follower feeds")

    # Celebrity posts (no fan-out; pulled on read)
    time.sleep(0.001)
    p2 = post_store.create(celeb.user_id, "https://cdn.example.com/pasta.jpg",
                           "My famous carbonara recipe 👨‍🍳")
    feed_svc.on_new_post(p2)
    print(f"  @celebrity_chef posts → NO fan-out (celebrity); pulled on read")

    # Alice posts
    time.sleep(0.001)
    p3 = post_store.create(alice.user_id, "https://cdn.example.com/sunrise.jpg",
                           "Beautiful sunrise this morning 🌅")
    feed_svc.on_new_post(p3)
    print(f"  @alice posts → pushed to followers")

    # ── Feed Read (Alice) ─────────────────────
    print("\n[4] ALICE'S FEED")
    print("─" * 55)

    feed = feed_svc.get_feed(alice.user_id, limit=10)
    for post in feed:
        author = user_store.get(post.user_id)
        name   = f"@{author.username}" if author else "unknown"
        print(f"  [{name}] {post.caption[:45]}")

    # ── Likes ─────────────────────────────────
    print("\n[5] LIKES")
    print("─" * 55)

    like_svc.like(p2.post_id, alice.user_id)
    like_svc.like(p2.post_id, bob.user_id)
    like_svc.like(p2.post_id, alice.user_id)  # duplicate

    print(f"  @celebrity_chef's post: {like_svc.like_count(p2.post_id)} likes")
    unlike_ok = like_svc.unlike(p2.post_id, alice.user_id)
    print(f"  After @alice unlikes: {like_svc.like_count(p2.post_id)} likes")

    # ── Scalability Design ────────────────────
    print("\n[6] SCALABILITY DESIGN")
    print("─" * 55)

    design = [
        ("Media storage",  "S3 / GCS with CDN (CloudFront) for global distribution"),
        ("Posts DB",       "Cassandra or sharded MySQL; shard by user_id"),
        ("Follows DB",     "Adjacency list in Cassandra: row = follower, cols = followees"),
        ("Feed",           "Redis sorted sets per user (score = timestamp)"),
        ("Celeb posts",    "Pulled at read time; merged with pre-computed feed"),
        ("Like counter",   "Redis incr + periodic flush to DB (approximate ok)"),
        ("Search",         "Elasticsearch for hashtag/username full-text search"),
        ("CDN",            "Photos served from Fastly edge; origin is S3"),
        ("Video",          "HLS transcoded by FFmpeg workers; stored to S3"),
        ("Notifications",  "Kafka → worker → APNs/FCM push notifications"),
    ]
    for component, detail in design:
        print(f"  {component:<18} {detail}")


if __name__ == "__main__":
    demonstrate_instagram()
