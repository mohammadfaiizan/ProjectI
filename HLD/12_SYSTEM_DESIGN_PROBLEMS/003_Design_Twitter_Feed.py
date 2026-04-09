"""
SYSTEM DESIGN: TWITTER / X FEED (Timeline)
============================================

Problem Statement:
Design Twitter's home timeline: a user sees tweets from people they follow,
in reverse-chronological order (or ranked by engagement).

Functional Requirements:
  - Post tweet (text, media, replies)
  - Follow / unfollow users
  - Home timeline (tweets from followed users)
  - Notifications (likes, retweets, mentions)
  - Search (recent tweets, trending topics)

Non-Functional Requirements:
  - 300M MAU, 100M DAU
  - 500M tweets/day → ~5787 writes/sec
  - Timeline reads: 100× tweets → 578700 reads/sec
  - Timeline must load < 300ms (p99)
  - Tweets must be stored indefinitely

Twitter's Actual Architecture (key insights):
  - Fan-out on write to Redis lists (pre-computed timelines)
  - Celebrities (> ~20K followers): fan-out on read
  - Timeline stored in Redis as sorted set (tweet_id as score)
  - Snowflake IDs: time-sortable 64-bit integers
  - Finagle RPC internally; Thrift serialization

Snowflake ID (Twitter's tweet ID):
  Bit layout: [timestamp 41b][machine_id 10b][sequence 12b]
  - Timestamp: ms since epoch (Jan 1, 2010)
  - 2^41 ms = 69 years
  - 2^10 = 1024 machines
  - 2^12 = 4096 IDs/ms/machine
  - IDs sort chronologically (no extra sort needed for timeline)

Timeline Storage (Redis):
  Key:   timeline:{user_id}
  Type:  List (LPUSH + LTRIM to 800 entries)
  On post: LPUSH tweet_id to each follower's timeline

Trending Topics:
  Count hashtag occurrences in sliding 15-min window.
  Redis sorted set: ZINCRBY trends #{hashtag} 1
  ZREVRANGE trends 0 9 → top 10 trending

Retweet Storm / Hotspot:
  Celebrity tweets: don't fan-out → pull at read time.
  Cache tweet object separately; dedup by tweet_id when merging.
"""

from __future__ import annotations

import time
import random
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict, deque


# ─────────────────────────────────────────────
# SNOWFLAKE ID GENERATOR
# ─────────────────────────────────────────────

class SnowflakeGenerator:
    """
    64-bit time-sortable ID.
    Layout: [unused 1b][timestamp 41b][machine_id 10b][sequence 12b]
    """

    EPOCH_MS = 1288834974657   # Twitter epoch: Nov 4 2010

    def __init__(self, machine_id: int = 1):
        if machine_id >= 1024:
            raise ValueError("machine_id must be < 1024")
        self._machine  = machine_id
        self._seq      = 0
        self._last_ms  = -1

    def next_id(self) -> int:
        now_ms = int(time.time() * 1000) - self.EPOCH_MS
        if now_ms == self._last_ms:
            self._seq = (self._seq + 1) & 0xFFF
            if self._seq == 0:
                # Spin until next millisecond
                while now_ms <= self._last_ms:
                    now_ms = int(time.time() * 1000) - self.EPOCH_MS
        else:
            self._seq = 0
        self._last_ms = now_ms
        return (now_ms << 22) | (self._machine << 12) | self._seq

    def timestamp_ms(self, snowflake_id: int) -> int:
        return (snowflake_id >> 22) + self.EPOCH_MS


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class Tweet:
    tweet_id:   int
    user_id:    str
    text:       str
    created_at: float
    retweet_of: Optional[int]  = None
    reply_to:   Optional[int]  = None
    like_count: int = 0
    rt_count:   int = 0
    hashtags:   List[str] = field(default_factory=list)


@dataclass
class TwitterUser:
    user_id:        str
    username:       str
    follower_count: int = 0

    @property
    def is_celebrity(self) -> bool:
        return self.follower_count >= 20_000


# ─────────────────────────────────────────────
# REDIS-LIKE TIMELINE STORE
# ─────────────────────────────────────────────

class TimelineStore:
    """
    Simulates Redis-based pre-computed timelines.
    Each user's timeline is a deque of tweet_ids (newest first).
    """

    MAX_TIMELINE = 800   # entries per user

    def __init__(self):
        self._timelines: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.MAX_TIMELINE))

    def push(self, user_id: str, tweet_id: int):
        """LPUSH tweet_id to user's timeline."""
        tl = self._timelines[user_id]
        tl.appendleft(tweet_id)

    def get(self, user_id: str, limit: int = 20) -> List[int]:
        """Return most recent `limit` tweet_ids."""
        return list(self._timelines[user_id])[:limit]

    def size(self, user_id: str) -> int:
        return len(self._timelines[user_id])


# ─────────────────────────────────────────────
# TRENDING TOPICS
# ─────────────────────────────────────────────

class TrendingTopics:
    """
    Sliding window count of hashtags.
    Uses a time-bucketed counter (1-minute buckets).
    """

    def __init__(self, window_min: int = 15):
        self._window  = window_min
        # bucket → {hashtag: count}
        self._buckets: Dict[int, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int))

    def _bucket(self, ts: float) -> int:
        return int(ts / 60)   # 1-minute buckets

    def record(self, hashtags: List[str], ts: Optional[float] = None):
        ts = ts or time.time()
        b  = self._bucket(ts)
        for ht in hashtags:
            self._buckets[b][ht.lower()] += 1

    def top(self, n: int = 10, ts: Optional[float] = None) -> List[tuple]:
        ts     = ts or time.time()
        b_now  = self._bucket(ts)
        b_min  = b_now - self._window

        counts: Dict[str, int] = defaultdict(int)
        for b, bucket_data in self._buckets.items():
            if b >= b_min:
                for ht, cnt in bucket_data.items():
                    counts[ht] += cnt

        return sorted(counts.items(), key=lambda x: -x[1])[:n]


# ─────────────────────────────────────────────
# TWITTER SERVICE
# ─────────────────────────────────────────────

class TwitterService:
    def __init__(self):
        self._id_gen    = SnowflakeGenerator(machine_id=1)
        self._users:    Dict[str, TwitterUser]  = {}
        self._tweets:   Dict[int, Tweet]        = {}
        self._follows:  Dict[str, Set[str]]     = defaultdict(set)
        self._followers: Dict[str, Set[str]]    = defaultdict(set)
        self._timelines = TimelineStore()
        self._trending  = TrendingTopics()

    def create_user(self, username: str, followers: int = 0) -> TwitterUser:
        uid = f"u_{len(self._users):05d}"
        u   = TwitterUser(uid, username, followers)
        self._users[uid] = u
        return u

    def follow(self, follower_id: str, followee_id: str):
        self._follows[follower_id].add(followee_id)
        self._followers[followee_id].add(follower_id)
        self._users[followee_id].follower_count += 1

    def tweet(self, user_id: str, text: str,
              retweet_of: Optional[int] = None) -> Tweet:
        tweet_id = self._id_gen.next_id()
        hashtags  = [w[1:] for w in text.split() if w.startswith("#")]
        tw        = Tweet(tweet_id, user_id, text, time.time(),
                          retweet_of, hashtags=hashtags)
        self._tweets[tweet_id] = tw

        if hashtags:
            self._trending.record(hashtags)

        # Fan-out
        author = self._users.get(user_id)
        if not author or author.is_celebrity:
            return tw   # celebrity → no fan-out

        followers = self._followers.get(user_id, set())
        for fid in followers:
            self._timelines.push(fid, tweet_id)

        # Also push to author's own timeline
        self._timelines.push(user_id, tweet_id)
        return tw

    def get_timeline(self, user_id: str, limit: int = 20) -> List[Tweet]:
        """
        Hybrid timeline:
        1. Pre-computed feed (non-celebrity followees)
        2. Merge celebrity tweets (pull)
        3. Sort by tweet_id (= chronological via Snowflake)
        """
        tweet_ids = set(self._timelines.get(user_id, limit * 3))

        # Pull celebrity posts
        for fid in self._follows.get(user_id, set()):
            fuser = self._users.get(fid)
            if fuser and fuser.is_celebrity:
                # Get recent celebrity tweets (simulated)
                celeb_tweets = [tid for tid, tw in self._tweets.items()
                                if tw.user_id == fid][-10:]
                tweet_ids.update(celeb_tweets)

        # Hydrate and sort
        tweets = [self._tweets[tid] for tid in tweet_ids if tid in self._tweets]
        tweets.sort(key=lambda t: -t.tweet_id)   # Snowflake sort = time sort
        return tweets[:limit]

    def like(self, tweet_id: int, user_id: str):
        tw = self._tweets.get(tweet_id)
        if tw:
            tw.like_count += 1

    def retweet(self, tweet_id: int, user_id: str) -> Tweet:
        orig = self._tweets.get(tweet_id)
        if not orig:
            raise ValueError("Tweet not found")
        orig.rt_count += 1
        rt_text = f"RT @{self._users.get(orig.user_id, TwitterUser('','?')).username}: {orig.text}"
        return self.tweet(user_id, rt_text, retweet_of=tweet_id)

    def trending(self, n: int = 10) -> List[tuple]:
        return self._trending.top(n)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_twitter():
    print("=" * 65)
    print("SYSTEM DESIGN: TWITTER FEED")
    print("=" * 65)

    svc = TwitterService()

    # ── Snowflake IDs ─────────────────────────
    print("\n[1] SNOWFLAKE ID GENERATION")
    print("─" * 55)

    gen = SnowflakeGenerator(machine_id=42)
    ids = [gen.next_id() for _ in range(5)]
    print(f"  {'ID':<22} {'Timestamp (ms)'}")
    for sid in ids:
        ts_ms = gen.timestamp_ms(sid)
        print(f"  {sid:<22} {ts_ms}")
    print(f"  IDs are monotonically increasing: {ids == sorted(ids)}")

    # ── Users and Follows ─────────────────────
    print("\n[2] USERS AND FOLLOWS")
    print("─" * 55)

    alice  = svc.create_user("alice",   followers=500)
    bob    = svc.create_user("bob",     followers=300)
    elon   = svc.create_user("elonX",  followers=5_000_000)
    taylor = svc.create_user("t_swift", followers=90_000_000)

    svc.follow(alice.user_id, bob.user_id)
    svc.follow(alice.user_id, elon.user_id)
    svc.follow(alice.user_id, taylor.user_id)
    svc.follow(bob.user_id, alice.user_id)

    for u in [alice, bob, elon, taylor]:
        print(f"  @{u.username:<12} followers={u.follower_count:>12,}  celebrity={u.is_celebrity}")

    # ── Tweets and Fan-out ────────────────────
    print("\n[3] TWEETS AND FAN-OUT")
    print("─" * 55)

    t1 = svc.tweet(bob.user_id, "Just had the best coffee ever! #morning #coffee")
    print(f"  @bob tweets → fan-out to {len(svc._followers.get(bob.user_id, set()))} followers")

    t2 = svc.tweet(elon.user_id, "The future of #AI is here. #tech")
    print(f"  @elonX tweets → NO fan-out (celebrity). Pulled on read.")

    t3 = svc.tweet(taylor.user_id, "#Midnights is 1 year old! #swifties")
    print(f"  @t_swift tweets → NO fan-out (celebrity). Pulled on read.")

    t4 = svc.tweet(alice.user_id, "Good morning! Excited for the #weekend")
    print(f"  @alice tweets → fan-out to {len(svc._followers.get(alice.user_id, set()))} followers")

    # ── Timeline Read ─────────────────────────
    print("\n[4] ALICE'S TIMELINE (hybrid)")
    print("─" * 55)

    timeline = svc.get_timeline(alice.user_id, limit=10)
    print(f"  Timeline size: {len(timeline)} tweets")
    for tw in timeline[:6]:
        author = svc._users.get(tw.user_id)
        print(f"  [@{author.username if author else '?':<12}] {tw.text[:55]}")

    # ── Retweet and Like ──────────────────────
    print("\n[5] RETWEET AND LIKE")
    print("─" * 55)

    svc.like(t2.tweet_id, alice.user_id)
    svc.like(t2.tweet_id, bob.user_id)
    rt = svc.retweet(t1.tweet_id, alice.user_id)
    print(f"  @elonX tweet likes: {svc._tweets[t2.tweet_id].like_count}")
    print(f"  @bob tweet RTs:     {svc._tweets[t1.tweet_id].rt_count}")
    print(f"  @alice retweet: {rt.text[:55]}")

    # ── Trending Topics ───────────────────────
    print("\n[6] TRENDING TOPICS")
    print("─" * 55)

    # Add more tweets with hashtags
    extra_hashtags = ["#coffee", "#morning", "#coffee", "#AI", "#AI", "#AI",
                      "#tech", "#weekend", "#swifties", "#swifties", "#swifties",
                      "#swifties", "#coffee", "#AI"]
    random.seed(1)
    for ht in extra_hashtags:
        svc._trending.record([ht.lstrip("#")])

    trends = svc.trending(n=5)
    print("  Top 5 trending:")
    for rank, (tag, count) in enumerate(trends, 1):
        print(f"    #{rank} #{tag:<15} {count} mentions")

    # ── Architecture ──────────────────────────
    print("\n[7] TWITTER ARCHITECTURE")
    print("─" * 55)

    arch = [
        ("Tweet write",    "API → Fanout Service → Redis Timeline (LPUSH)"),
        ("Celebrity",      "No fan-out on write; merged at read time"),
        ("Timeline read",  "Redis sorted set → hydrate from Tweet cache"),
        ("Tweet storage",  "Manhattan (Twitter's distributed KV store) / Cassandra"),
        ("Media",          "BlobStore → CDN (Fastly)"),
        ("Search",         "Earlybird (Lucene-based) for real-time tweet search"),
        ("Trending",       "Redis ZINCRBY per hashtag; sliding 15-min window"),
        ("Notifications",  "EventBus → Notification worker → APNs/FCM"),
        ("IDs",            "Snowflake: 64-bit time-sortable, no sort needed"),
        ("Serving",        "Gizzard (partitioned graph DB) for social graph"),
    ]
    for component, detail in arch:
        print(f"  {component:<18} {detail}")


if __name__ == "__main__":
    demonstrate_twitter()
