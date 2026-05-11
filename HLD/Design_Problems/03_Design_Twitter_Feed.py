"""
Twitter Feed System - Working Python Implementation
Demonstrates: tweet posting, fan-out (hybrid push/pull), timeline generation,
              retweet mechanics, trending topics (sliding window), snowflake IDs,
              @mention and hashtag parsing.
No external dependencies — standard library only.
"""

import time
import heapq
import re
import collections
from datetime import datetime
from typing import Optional, List, Dict, Set, Tuple


# ---------------------------------------------------------------------------
# Snowflake ID Generator
# ---------------------------------------------------------------------------
class SnowflakeIDGenerator:
    """
    Generates 64-bit time-sortable unique IDs.
    Layout: [1-bit zero][41-bit ms timestamp][10-bit machine_id][12-bit sequence]
    This means IDs are monotonically increasing and can be sorted like timestamps.
    """
    EPOCH = 1288834974657  # Twitter's epoch: Nov 4, 2010 in ms

    def __init__(self, machine_id: int = 1):
        assert 0 <= machine_id < 1024, "machine_id must be 0-1023"
        self.machine_id = machine_id
        self._sequence = 0
        self._last_ms = -1

    def generate(self) -> int:
        ms = int(time.time() * 1000) - self.EPOCH
        if ms == self._last_ms:
            self._sequence = (self._sequence + 1) & 0xFFF  # 12 bits = 4096 max
            if self._sequence == 0:
                # Sequence exhausted — wait for next millisecond
                while ms <= self._last_ms:
                    ms = int(time.time() * 1000) - self.EPOCH
        else:
            self._sequence = 0
        self._last_ms = ms
        # Compose the ID
        return (ms << 22) | (self.machine_id << 12) | self._sequence

    @staticmethod
    def extract_timestamp(snowflake_id: int) -> int:
        """Return ms-since-epoch embedded in the snowflake ID."""
        return (snowflake_id >> 22) + SnowflakeIDGenerator.EPOCH


# ---------------------------------------------------------------------------
# Sliding Window Trending Tracker
# ---------------------------------------------------------------------------
class TrendingTracker:
    """
    Tracks hashtag counts using a sliding window approach.
    Buckets by minute; window = last N minutes.
    In production: uses Kafka stream + Redis sorted sets.
    """

    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        # { hashtag: deque of (minute_bucket, count) }
        self._counts: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=window_minutes * 2)
        )

    def _current_bucket(self) -> int:
        """Current minute bucket (integer minutes since epoch)."""
        return int(time.time() // 60)

    def record(self, hashtags: List[str]) -> None:
        bucket = self._current_bucket()
        for tag in hashtags:
            tag = tag.lower()
            dq = self._counts[tag]
            if dq and dq[-1][0] == bucket:
                # Increment current bucket
                old_bucket, old_count = dq[-1]
                dq[-1] = (old_bucket, old_count + 1)
            else:
                dq.append((bucket, 1))

    def get_score(self, hashtag: str) -> int:
        """Sum of counts within the sliding window."""
        cutoff = self._current_bucket() - self.window_minutes
        total = 0
        for bucket, count in self._counts[hashtag.lower()]:
            if bucket >= cutoff:
                total += count
        return total

    def top_k(self, k: int = 10) -> List[Tuple[str, int]]:
        """Return top-K trending hashtags by score."""
        scores = []
        for tag in self._counts:
            score = self.get_score(tag)
            if score > 0:
                scores.append((score, tag))
        # Use min-heap of size K for O(N log K) complexity
        heap: List[Tuple[int, str]] = []
        for score, tag in scores:
            if len(heap) < k:
                heapq.heappush(heap, (score, tag))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, tag))
        return sorted([(tag, score) for score, tag in heap], key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
class Tweet:
    def __init__(
        self,
        tweet_id: int,
        user_id: int,
        content: str,
        retweet_of: Optional[int] = None,
    ):
        self.tweet_id = tweet_id           # Snowflake ID
        self.user_id = user_id
        self.content = content
        self.retweet_of = retweet_of       # original tweet_id if this is a retweet
        self.like_count = 0
        self.retweet_count = 0
        self.hashtags: List[str] = self._extract_hashtags(content)
        self.mentions: List[str] = self._extract_mentions(content)
        self.timestamp = SnowflakeIDGenerator.extract_timestamp(tweet_id)

    @staticmethod
    def _extract_hashtags(content: str) -> List[str]:
        return re.findall(r"#(\w+)", content)

    @staticmethod
    def _extract_mentions(content: str) -> List[str]:
        return re.findall(r"@(\w+)", content)

    def __repr__(self):
        flag = "[RT]" if self.retweet_of else ""
        return f"Tweet({self.tweet_id}, @uid={self.user_id} {flag}: '{self.content[:40]}')"


class User:
    def __init__(self, user_id: int, username: str, follower_count: int = 0):
        self.user_id = user_id
        self.username = username
        self.follower_count = follower_count
        self.following_count = 0
        self.tweet_count = 0

    def is_celebrity(self, threshold: int = 10) -> bool:
        """In production threshold = 1,000,000."""
        return self.follower_count >= threshold

    def __repr__(self):
        return f"User({self.user_id}, @{self.username}, followers={self.follower_count})"


# ---------------------------------------------------------------------------
# Twitter Core System
# ---------------------------------------------------------------------------
class TwitterSystem:
    """
    Core Twitter system implementing:
    - Tweet posting with Snowflake IDs
    - Hybrid fan-out (push for regular, pull for celebrities)
    - Timeline generation (merge pre-computed + celebrity pull)
    - Retweet mechanics
    - Trending topics (sliding window)
    - @mention and hashtag parsing
    """

    CELEBRITY_THRESHOLD = 10          # lowered for demo; prod = 1,000,000
    TIMELINE_MAX_SIZE = 800           # max tweets in Redis timeline
    CELEBRITY_PULL_LIMIT = 50         # how many celebrity tweets to pull on read

    def __init__(self):
        self._id_gen = SnowflakeIDGenerator(machine_id=1)
        self._trending = TrendingTracker(window_minutes=60)

        # Stores (DB tables simulated as dicts)
        self._users: Dict[int, User] = {}
        self._tweets: Dict[int, Tweet] = {}

        # Follow graph
        self._followers: Dict[int, Set[int]] = collections.defaultdict(set)
        self._following: Dict[int, Set[int]] = collections.defaultdict(set)

        # User tweet list: user_id -> sorted list of tweet_ids (newest first)
        self._user_tweets: Dict[int, List[int]] = collections.defaultdict(list)

        # Pre-computed timelines (Redis Sorted Set simulation)
        # timeline[user_id] = min-heap of (-tweet_id, tweet_id)
        # Negative tweet_id so that pop gives smallest (oldest) for trimming
        self._timelines: Dict[int, List] = collections.defaultdict(list)

        # Likes: tweet_id -> set of user_ids
        self._likes: Dict[int, Set[int]] = collections.defaultdict(set)

        # Retweet tracking: tweet_id -> set of user_ids who retweeted
        self._retweets: Dict[int, Set[int]] = collections.defaultdict(set)

        # Search index: keyword/hashtag -> list of tweet_ids
        self._search_index: Dict[str, List[int]] = collections.defaultdict(list)

    # ------------------------------------------------------------------
    # User Management
    # ------------------------------------------------------------------

    def create_user(self, user_id: int, username: str, follower_count: int = 0) -> User:
        user = User(user_id, username, follower_count)
        self._users[user_id] = user
        return user

    # ------------------------------------------------------------------
    # Follow
    # ------------------------------------------------------------------

    def follow(self, follower_id: int, followee_id: int) -> dict:
        if follower_id == followee_id:
            return {"error": "cannot follow yourself"}
        if followee_id in self._following[follower_id]:
            return {"already_following": True}

        self._following[follower_id].add(followee_id)
        self._followers[followee_id].add(follower_id)
        self._users[follower_id].following_count += 1
        self._users[followee_id].follower_count += 1

        # Fan-out recent tweets to follower's timeline if followee is not celebrity
        followee = self._users.get(followee_id)
        if followee and not followee.is_celebrity(self.CELEBRITY_THRESHOLD):
            recent = self._user_tweets[followee_id][:20]
            for tid in recent:
                self._push_to_timeline(follower_id, tid)

        return {"following": True}

    def unfollow(self, follower_id: int, followee_id: int) -> dict:
        self._following[follower_id].discard(followee_id)
        self._followers[followee_id].discard(follower_id)
        if follower_id in self._users:
            self._users[follower_id].following_count = max(
                0, self._users[follower_id].following_count - 1
            )
        if followee_id in self._users:
            self._users[followee_id].follower_count = max(
                0, self._users[followee_id].follower_count - 1
            )
        return {"following": False}

    # ------------------------------------------------------------------
    # Tweet Posting
    # ------------------------------------------------------------------

    def post_tweet(self, user_id: int, content: str) -> dict:
        """
        Post a tweet. Fan-out to follower timelines based on poster's follower count.
        Regular users: push to each follower's timeline (fan-out on write).
        Celebrities: skip push; followers pull on timeline read.
        """
        if user_id not in self._users:
            return {"error": "user not found"}
        if len(content) > 280:
            return {"error": f"tweet too long: {len(content)} chars (max 280)"}

        tweet_id = self._id_gen.generate()
        tweet = Tweet(tweet_id, user_id, content)
        self._tweets[tweet_id] = tweet

        # Store in user's tweet list (newest first)
        self._user_tweets[user_id].insert(0, tweet_id)
        self._users[user_id].tweet_count += 1

        # Index for search
        for word in content.lower().split():
            self._search_index[word].append(tweet_id)
        for tag in tweet.hashtags:
            self._search_index[f"#{tag.lower()}"].append(tweet_id)

        # Update trending
        if tweet.hashtags:
            self._trending.record(tweet.hashtags)

        # Fan-out
        poster = self._users[user_id]
        if not poster.is_celebrity(self.CELEBRITY_THRESHOLD):
            fan_out_count = 0
            for follower_id in self._followers[user_id]:
                self._push_to_timeline(follower_id, tweet_id)
                fan_out_count += 1
        else:
            fan_out_count = 0  # celebrity: pull on read

        return {
            "tweet_id": tweet_id,
            "content": content,
            "hashtags": tweet.hashtags,
            "mentions": tweet.mentions,
            "fan_out_count": fan_out_count,
            "created_at": datetime.utcfromtimestamp(tweet.timestamp / 1000).isoformat(),
        }

    def _push_to_timeline(self, user_id: int, tweet_id: int) -> None:
        """
        Push a tweet_id into a user's timeline.
        Timeline is a max-heap by tweet_id (Snowflake = time-sortable).
        Store as (-tweet_id) so that heappop gives the smallest (oldest) for trimming.
        """
        heapq.heappush(self._timelines[user_id], -tweet_id)
        # Trim to TIMELINE_MAX_SIZE (evict oldest = smallest tweet_id = most negative)
        while len(self._timelines[user_id]) > self.TIMELINE_MAX_SIZE:
            heapq.heappop(self._timelines[user_id])  # pops smallest (oldest)

    # ------------------------------------------------------------------
    # Retweet
    # ------------------------------------------------------------------

    def retweet(self, user_id: int, original_tweet_id: int) -> dict:
        """
        Retweet = create a new tweet referencing the original.
        Fan-out the retweet to the retweeter's followers.
        """
        original = self._tweets.get(original_tweet_id)
        if not original:
            return {"error": "tweet not found"}
        if user_id in self._retweets[original_tweet_id]:
            return {"error": "already retweeted"}

        tweet_id = self._id_gen.generate()
        rt_content = f"RT @{self._users[original.user_id].username}: {original.content}"
        rt = Tweet(tweet_id, user_id, rt_content, retweet_of=original_tweet_id)
        self._tweets[tweet_id] = rt

        # Update original tweet stats
        original.retweet_count += 1
        self._retweets[original_tweet_id].add(user_id)

        # Store in user's tweet list
        self._user_tweets[user_id].insert(0, tweet_id)
        self._users[user_id].tweet_count += 1

        # Fan-out retweet
        poster = self._users[user_id]
        fan_out_count = 0
        if not poster.is_celebrity(self.CELEBRITY_THRESHOLD):
            for follower_id in self._followers[user_id]:
                self._push_to_timeline(follower_id, tweet_id)
                fan_out_count += 1

        return {
            "retweet_id": tweet_id,
            "original_tweet_id": original_tweet_id,
            "fan_out_count": fan_out_count,
        }

    # ------------------------------------------------------------------
    # Timeline (Home Feed)
    # ------------------------------------------------------------------

    def get_timeline(self, user_id: int, limit: int = 20) -> dict:
        """
        Generate home timeline for user.
        Hybrid approach:
          1. Read pre-computed timeline (push from non-celebrities)
          2. Pull latest tweets from celebrity followees
          3. N-way merge and return top `limit` entries
        """
        if user_id not in self._users:
            return {"error": "user not found"}

        # Step 1: Pre-computed timeline (sorted descending by tweet_id)
        pre_computed = sorted(self._timelines[user_id], reverse=True)
        # Convert: stored as negatives, get positives
        pre_ids = [-x for x in sorted(self._timelines[user_id])]  # largest first

        # Step 2: Pull from celebrity followees
        celebrity_ids = []
        for followee_id in self._following[user_id]:
            followee = self._users.get(followee_id)
            if followee and followee.is_celebrity(self.CELEBRITY_THRESHOLD):
                # Pull top N tweets from celebrity
                celeb_tweets = self._user_tweets[followee_id][:self.CELEBRITY_PULL_LIMIT]
                celebrity_ids.extend(celeb_tweets)

        # Step 3: Merge and deduplicate, sort descending (newest first = largest snowflake)
        all_ids = list(set(pre_ids + celebrity_ids))
        all_ids.sort(reverse=True)  # newest first (larger Snowflake ID = newer)

        # Step 4: Paginate
        page_ids = all_ids[:limit]

        # Step 5: Enrich with tweet data
        tweets = []
        for tid in page_ids:
            tweet = self._tweets.get(tid)
            if tweet:
                author = self._users.get(tweet.user_id, {})
                tweets.append({
                    "tweet_id": tid,
                    "user": {
                        "user_id": tweet.user_id,
                        "username": getattr(author, "username", "unknown"),
                        "verified": False,
                    },
                    "content": tweet.content,
                    "like_count": tweet.like_count,
                    "retweet_count": tweet.retweet_count,
                    "hashtags": tweet.hashtags,
                    "mentions": tweet.mentions,
                    "is_retweet": tweet.retweet_of is not None,
                    "created_at": datetime.utcfromtimestamp(
                        tweet.timestamp / 1000
                    ).isoformat(),
                })

        return {
            "timeline": tweets,
            "total_available": len(all_ids),
            "user_id": user_id,
        }

    # ------------------------------------------------------------------
    # Likes
    # ------------------------------------------------------------------

    def like_tweet(self, user_id: int, tweet_id: int) -> dict:
        tweet = self._tweets.get(tweet_id)
        if not tweet:
            return {"error": "tweet not found"}
        if user_id in self._likes[tweet_id]:
            return {"liked": True, "already_liked": True, "like_count": tweet.like_count}
        self._likes[tweet_id].add(user_id)
        tweet.like_count += 1
        return {"liked": True, "like_count": tweet.like_count}

    # ------------------------------------------------------------------
    # Trending
    # ------------------------------------------------------------------

    def get_trending_topics(self, k: int = 10) -> dict:
        """Return top-K trending hashtags by sliding window score."""
        trends = self._trending.top_k(k)
        return {
            "trends": [
                {"hashtag": f"#{tag}", "score": score, "tweet_count": score}
                for tag, score in trends
            ],
            "as_of": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_tweets(self, query: str, limit: int = 20) -> dict:
        """
        Search tweets by keyword or hashtag.
        In production: Elasticsearch with near-real-time indexing.
        """
        terms = query.lower().split()
        if not terms:
            return {"results": [], "query": query}

        # Find tweets matching all terms (AND search)
        candidate_sets = []
        for term in terms:
            ids = set(self._search_index.get(term, []))
            candidate_sets.append(ids)

        if not candidate_sets:
            return {"results": [], "query": query}

        matching_ids = candidate_sets[0]
        for s in candidate_sets[1:]:
            matching_ids &= s  # AND

        # Sort by tweet_id desc (newest first via Snowflake ordering)
        sorted_ids = sorted(matching_ids, reverse=True)[:limit]

        results = []
        for tid in sorted_ids:
            t = self._tweets.get(tid)
            if t:
                author = self._users.get(t.user_id)
                results.append({
                    "tweet_id": tid,
                    "username": getattr(author, "username", "?"),
                    "content": t.content,
                    "like_count": t.like_count,
                })

        return {"results": results, "query": query, "count": len(results)}

    def get_user_tweets(self, user_id: int, limit: int = 20) -> List[dict]:
        """Fetch a user's tweet timeline (profile view)."""
        tweet_ids = self._user_tweets[user_id][:limit]
        result = []
        for tid in tweet_ids:
            t = self._tweets.get(tid)
            if t:
                result.append({"tweet_id": tid, "content": t.content,
                                "like_count": t.like_count, "is_retweet": t.retweet_of is not None})
        return result


# ---------------------------------------------------------------------------
# Demo / Simulation
# ---------------------------------------------------------------------------
def run_demo():
    print("=" * 65)
    print("TWITTER FEED SYSTEM DEMO")
    print("=" * 65)

    tw = TwitterSystem()

    # Create users
    alice   = tw.create_user(1, "alice")
    bob     = tw.create_user(2, "bob")
    charlie = tw.create_user(3, "charlie")
    # Celebrity: pre-seeded with high follower count
    celeb   = tw.create_user(4, "mega_star", follower_count=50)  # 50 > threshold=10

    print(f"\n[1] Created users. Celebrity threshold: {TwitterSystem.CELEBRITY_THRESHOLD}")
    print(f"    mega_star followers={celeb.follower_count}, is_celebrity={celeb.is_celebrity(TwitterSystem.CELEBRITY_THRESHOLD)}")

    # Follow relationships
    print("\n[2] Follow graph setup")
    tw.follow(1, 2)   # alice follows bob
    tw.follow(1, 3)   # alice follows charlie
    tw.follow(2, 3)   # bob follows charlie
    tw.follow(1, 4)   # alice follows celebrity
    tw.follow(2, 4)   # bob follows celebrity
    print(f"    Alice follows: {tw._following[1]}")

    # Snowflake ID demo
    print("\n[3] Snowflake ID demo")
    gen = SnowflakeIDGenerator(machine_id=5)
    ids = [gen.generate() for _ in range(5)]
    print(f"    IDs are monotonically increasing: {ids[0] < ids[1] < ids[2]}")
    for sid in ids:
        ts_ms = SnowflakeIDGenerator.extract_timestamp(sid)
        print(f"    ID={sid}, ts_ms={ts_ms}")

    # Post tweets
    print("\n[4] Posting tweets")
    t1 = tw.post_tweet(2, "Beautiful day! #weather #sunshine")
    t2 = tw.post_tweet(3, "Learning system design #sysdesign #tech")
    t3 = tw.post_tweet(3, "Cassandra vs MySQL for large scale #database #cassandra")
    t4 = tw.post_tweet(4, "New album dropping this Friday! #music #pop #celebrity")
    t5 = tw.post_tweet(2, "Coffee and code #coffee #coding #tech")
    print(f"    Bob's tweet (fan-out to {t1['fan_out_count']} followers): id={t1['tweet_id']}")
    print(f"    Celebrity tweet (fan-out to {t4['fan_out_count']} - skipped): id={t4['tweet_id']}")

    # Check alice's timeline size
    print(f"\n    Alice's pre-computed timeline size: {len(tw._timelines[1])}")

    # Get alice's home timeline
    print("\n[5] Alice's home timeline (hybrid merge)")
    timeline = tw.get_timeline(1, limit=10)
    print(f"    Total available tweets: {timeline['total_available']}")
    for t in timeline["timeline"]:
        flag = "[RT]" if t["is_retweet"] else "   "
        print(f"    {flag} @{t['user']['username']}: \"{t['content'][:55]}\"")

    # Retweet
    print("\n[6] Retweet mechanics")
    rt_result = tw.retweet(1, t2["tweet_id"])
    print(f"    Alice retweeted charlie's tweet: retweet_id={rt_result['retweet_id']}")
    # Try duplicate retweet
    rt_dup = tw.retweet(1, t2["tweet_id"])
    print(f"    Duplicate retweet result: {rt_dup.get('error')}")
    # Verify retweet_count
    print(f"    Original tweet retweet count: {tw._tweets[t2['tweet_id']].retweet_count}")

    # Likes
    print("\n[7] Likes")
    r1 = tw.like_tweet(1, t1["tweet_id"])
    r2 = tw.like_tweet(3, t1["tweet_id"])
    r3 = tw.like_tweet(1, t1["tweet_id"])  # duplicate
    print(f"    After 2 unique likes: count={r2['like_count']}")
    print(f"    Duplicate like: already_liked={r3.get('already_liked')}")

    # Trending topics
    print("\n[8] Trending topics")
    # Post more tweets with trending hashtags
    for _ in range(8):
        tw.post_tweet(3, "Another #tech #sysdesign post!")
    for _ in range(5):
        tw.post_tweet(2, "Music weekend #music #pop")
    trends = tw.get_trending_topics(k=5)
    print("    Top trends:")
    for trend in trends["trends"]:
        print(f"    {trend['hashtag']:25} score={trend['score']}")

    # Search
    print("\n[9] Search")
    sr1 = tw.search_tweets("#sysdesign")
    print(f"    '#sysdesign': {sr1['count']} tweets found")
    sr2 = tw.search_tweets("cassandra database")
    print(f"    'cassandra database': {sr2['count']} tweets found")
    for r in sr2["results"]:
        print(f"    - @{r['username']}: \"{r['content'][:55]}\"")

    # User profile tweets
    print("\n[10] Charlie's tweet history")
    charlie_tweets = tw.get_user_tweets(3, limit=5)
    for t in charlie_tweets:
        print(f"    id={t['tweet_id']}, likes={t['like_count']}: \"{t['content'][:50]}\"")

    print("\n" + "=" * 65)
    print("DEMO COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
