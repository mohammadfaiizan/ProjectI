"""
Reddit System Design - Python Implementation
Demonstrates: HotScoreCalculator, VoteTracker, CommentTree (materialized path),
              SubredditFeed (Redis sorted set sim), KarmaTracker, RedditSystem.
No external dependencies - standard library only.
"""

import math
import uuid
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Post:
    id: str
    subreddit_id: str
    author_id: str
    title: str
    body: str
    post_type: str           # text | link | image
    upvotes: int = 1
    downvotes: int = 0
    comment_count: int = 0
    is_removed: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hot_score: float = 0.0

    @property
    def score(self) -> int:
        return self.upvotes - self.downvotes

@dataclass
class Comment:
    id: str
    post_id: str
    author_id: str
    body: str
    parent_id: Optional[str]
    path: str               # materialized path: "/root/parent/self/"
    depth: int
    upvotes: int = 1
    downvotes: int = 0
    is_removed: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def score(self) -> int:
        return self.upvotes - self.downvotes

@dataclass
class Subreddit:
    id: str
    name: str
    description: str = ""
    subscriber_count: int = 0


# ---------------------------------------------------------------------------
# 1. HotScoreCalculator — Reddit's actual log-scale algorithm
# ---------------------------------------------------------------------------

class HotScoreCalculator:
    """
    Reddit's hot ranking formula:
      score = log10(max(|ups - downs|, 1)) * sign + seconds_since_epoch / 45000

    Key properties:
    - log10 dampens viral runaway (1K votes barely beats 100 votes in rank)
    - Time decay: posts age out of hot in ~12.5 hours
    - Positive posts always rank higher than negative ones
    """
    REDDIT_EPOCH = 1134028003  # Reddit's creation timestamp

    @classmethod
    def compute(cls, ups: int, downs: int, created_at: datetime) -> float:
        net = ups - downs
        order = math.log10(max(abs(net), 1))
        sign  = 1 if net > 0 else (-1 if net < 0 else 0)
        seconds_offset = created_at.timestamp() - cls.REDDIT_EPOCH
        return round(sign * order + seconds_offset / 45000, 7)

    @classmethod
    def decay_estimate_hours(cls) -> float:
        """How many hours until a post's time component adds 1 unit of order."""
        return 45000 / 3600  # ~12.5 hours per log unit


# ---------------------------------------------------------------------------
# 2. VoteTracker — idempotent vote management
# ---------------------------------------------------------------------------

class VoteTracker:
    """
    Tracks user votes per item. Prevents double-voting.
    Supports: upvote, downvote, unvote, vote-change.
    In production: Redis HASH for counters + votes table in PostgreSQL.
    """

    def __init__(self):
        # (user_id, item_id) -> vote_direction: 1=up, -1=down, 0=none
        self._votes: dict[tuple, int] = {}
        # item_id -> (ups, downs)
        self._counters: dict[str, list] = defaultdict(lambda: [0, 0])

    def vote(self, user_id: str, item_id: str, direction: int) -> dict:
        """
        direction: 1=upvote, -1=downvote, 0=unvote
        Returns delta applied to (ups, downs).
        """
        if direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0, or 1")

        key = (user_id, item_id)
        prev = self._votes.get(key, 0)

        if prev == direction:
            return {"delta_ups": 0, "delta_downs": 0, "new_direction": direction}

        counters = self._counters[item_id]
        # Remove previous vote
        if prev == 1:   counters[0] -= 1
        elif prev == -1: counters[1] -= 1
        # Apply new vote
        if direction == 1:   counters[0] += 1
        elif direction == -1: counters[1] += 1

        self._votes[key] = direction
        return {
            "delta_ups":   (1 if direction == 1 else 0) - (1 if prev == 1 else 0),
            "delta_downs": (1 if direction == -1 else 0) - (1 if prev == -1 else 0),
            "new_direction": direction,
            "current_ups":   counters[0],
            "current_downs": counters[1],
        }

    def get_user_vote(self, user_id: str, item_id: str) -> int:
        return self._votes.get((user_id, item_id), 0)

    def get_counts(self, item_id: str) -> tuple[int, int]:
        c = self._counters[item_id]
        return c[0], c[1]

    def get_voters(self, item_id: str) -> list[str]:
        """Return list of users who voted on this item."""
        return [uid for (uid, iid), v in self._votes.items() if iid == item_id and v != 0]


# ---------------------------------------------------------------------------
# 3. CommentTree — materialized path for nested comments
# ---------------------------------------------------------------------------

class CommentTree:
    """
    Stores comments with materialized path for efficient subtree queries.
    Path format: "/root_id/child_id/grandchild_id/"
    Subtree of node X: all comments WHERE path LIKE '/X/%'
    """

    def __init__(self):
        self._comments: dict[str, Comment] = {}
        # post_id -> list of comment_ids (insertion order)
        self._post_comments: dict[str, list] = defaultdict(list)

    def add_comment(
        self,
        post_id: str,
        author_id: str,
        body: str,
        parent_id: Optional[str] = None,
    ) -> Comment:
        cid = str(uuid.uuid4())[:8]

        if parent_id is None:
            path  = f"/{cid}/"
            depth = 0
        else:
            parent = self._comments.get(parent_id)
            if not parent:
                raise KeyError(f"Parent comment {parent_id} not found")
            path  = parent.path + f"{cid}/"
            depth = parent.depth + 1

        comment = Comment(
            id=cid, post_id=post_id, author_id=author_id,
            body=body, parent_id=parent_id, path=path, depth=depth,
        )
        self._comments[cid] = comment
        self._post_comments[post_id].append(cid)
        return comment

    def get_subtree(self, root_comment_id: str) -> list[Comment]:
        """Returns root + all descendants, sorted by path (depth-first)."""
        root = self._comments.get(root_comment_id)
        if not root:
            return []
        prefix = root.path
        results = [
            c for c in self._comments.values()
            if c.path.startswith(prefix)
        ]
        results.sort(key=lambda c: c.path)
        return results

    def get_top_level_comments(self, post_id: str) -> list[Comment]:
        """Returns root-level comments for a post sorted by score."""
        result = [
            self._comments[cid]
            for cid in self._post_comments[post_id]
            if self._comments[cid].depth == 0 and not self._comments[cid].is_removed
        ]
        result.sort(key=lambda c: c.score, reverse=True)
        return result

    def get_children(self, parent_id: str) -> list[Comment]:
        """Direct children of a comment."""
        parent = self._comments.get(parent_id)
        if not parent:
            return []
        expected_depth = parent.depth + 1
        return [
            c for c in self._comments.values()
            if c.parent_id == parent_id and not c.is_removed
        ]

    def render_tree(self, post_id: str, max_depth: int = 5) -> list[dict]:
        """Returns flat list with indentation info for rendering."""
        top_level = self.get_top_level_comments(post_id)
        result = []

        def _traverse(comment: Comment, current_depth: int):
            if current_depth > max_depth:
                return
            result.append({
                "id":      comment.id,
                "author":  comment.author_id,
                "body":    comment.body[:80],
                "score":   comment.score,
                "depth":   comment.depth,
                "indent":  "  " * comment.depth,
            })
            children = self.get_children(comment.id)
            children.sort(key=lambda c: c.score, reverse=True)
            for child in children:
                _traverse(child, current_depth + 1)

        for root in top_level:
            _traverse(root, 0)
        return result


# ---------------------------------------------------------------------------
# 4. SubredditFeed — sorted sets simulating Redis ZADD/ZREVRANGE
# ---------------------------------------------------------------------------

class SubredditFeed:
    """
    Per-subreddit sorted feeds for hot/new/top/rising.
    Simulates Redis sorted sets: O(log N) insert, O(log N + K) range read.
    In production: Redis ZADD + ZREVRANGEBYSCORE.
    """

    def __init__(self):
        # subreddit_id -> {sort_type -> [(score, post_id)]}
        self._feeds: dict[str, dict[str, list]] = defaultdict(
            lambda: {"hot": [], "new": [], "top": []}
        )

    def _insert_sorted(self, lst: list, score: float, post_id: str) -> None:
        """Maintain list sorted by score descending."""
        import bisect
        # Store as (-score, post_id) for ascending bisect on descending scores
        entry = (-score, post_id)
        bisect.insort(lst, entry)

    def update_post(self, subreddit_id: str, post: Post) -> None:
        """Update all feed types for a post."""
        feeds = self._feeds[subreddit_id]
        post_id = post.id

        # Remove existing entry for this post
        for sort_type in ["hot", "new", "top"]:
            feeds[sort_type] = [(s, pid) for s, pid in feeds[sort_type] if pid != post_id]

        # Re-insert with updated scores
        hot_score = HotScoreCalculator.compute(post.upvotes, post.downvotes, post.created_at)
        new_score = post.created_at.timestamp()
        top_score = float(post.score)

        self._insert_sorted(feeds["hot"], hot_score, post_id)
        self._insert_sorted(feeds["new"], new_score, post_id)
        self._insert_sorted(feeds["top"], top_score, post_id)

    def get_feed(
        self, subreddit_id: str, sort: str = "hot", limit: int = 25, offset: int = 0
    ) -> list[str]:
        """Returns list of post_ids in feed order."""
        feed = self._feeds[subreddit_id].get(sort, [])
        return [post_id for _, post_id in feed[offset: offset + limit]]


# ---------------------------------------------------------------------------
# 5. KarmaTracker — post + comment karma per user
# ---------------------------------------------------------------------------

class KarmaTracker:
    """
    Tracks user karma from votes.
    Post karma capped per post (max +1000) to prevent manipulation.
    In production: Kafka stream of vote events → karma service.
    """
    MAX_KARMA_PER_POST = 1000

    def __init__(self):
        self._post_karma: dict[str, int]     = defaultdict(int)   # user_id -> karma
        self._comment_karma: dict[str, int]  = defaultdict(int)
        # item_id -> accumulated karma contribution (for cap enforcement)
        self._post_contributions: dict[str, int] = defaultdict(int)

    def apply_vote_event(
        self, author_id: str, item_id: str, item_type: str, delta: int
    ) -> None:
        """
        Called when a vote changes. delta = change in net score.
        item_type: 'post' or 'comment'
        """
        if item_type == "post":
            current = self._post_contributions[item_id]
            new_total = min(current + delta, self.MAX_KARMA_PER_POST)
            actual_delta = new_total - current
            self._post_contributions[item_id] = new_total
            self._post_karma[author_id] += actual_delta
        elif item_type == "comment":
            self._comment_karma[author_id] += delta

    def get_karma(self, user_id: str) -> dict:
        return {
            "post_karma":    self._post_karma[user_id],
            "comment_karma": self._comment_karma[user_id],
            "total":         self._post_karma[user_id] + self._comment_karma[user_id],
        }


# ---------------------------------------------------------------------------
# 6. RedditSystem — Facade
# ---------------------------------------------------------------------------

class RedditSystem:
    def __init__(self):
        self._posts: dict[str, Post]             = {}
        self._subreddits: dict[str, Subreddit]   = {}
        self._sub_by_name: dict[str, str]        = {}  # name -> id
        self._subscriptions: dict[str, set]      = defaultdict(set)  # user_id -> {sub_id}
        self._vote_tracker  = VoteTracker()
        self._comment_tree  = CommentTree()
        self._feed          = SubredditFeed()
        self._karma         = KarmaTracker()

    # -- Subreddit ----------------------------------------------------------

    def create_subreddit(self, name: str, description: str = "") -> Subreddit:
        sub = Subreddit(id=str(uuid.uuid4())[:6], name=name, description=description)
        self._subreddits[sub.id] = sub
        self._sub_by_name[name.lower()] = sub.id
        return sub

    def subscribe_subreddit(self, user_id: str, subreddit_name: str) -> None:
        sub_id = self._sub_by_name.get(subreddit_name.lower())
        if not sub_id:
            raise KeyError(f"Subreddit r/{subreddit_name} not found")
        self._subscriptions[user_id].add(sub_id)
        self._subreddits[sub_id].subscriber_count += 1

    # -- Posts --------------------------------------------------------------

    def submit_post(
        self,
        subreddit_name: str,
        author_id: str,
        title: str,
        body: str = "",
        post_type: str = "text",
    ) -> Post:
        sub_id = self._sub_by_name.get(subreddit_name.lower())
        if not sub_id:
            raise KeyError(f"Subreddit r/{subreddit_name} not found")
        post = Post(
            id=str(uuid.uuid4())[:8],
            subreddit_id=sub_id,
            author_id=author_id,
            title=title,
            body=body,
            post_type=post_type,
        )
        post.hot_score = HotScoreCalculator.compute(1, 0, post.created_at)
        self._posts[post.id] = post
        self._feed.update_post(sub_id, post)
        return post

    def vote(self, user_id: str, item_id: str, item_type: str, direction: int) -> dict:
        result = self._vote_tracker.vote(user_id, item_id, direction)
        # Update post/comment vote counts and hot score
        if item_type == "post":
            post = self._posts.get(item_id)
            if post:
                post.upvotes   += result["delta_ups"]
                post.downvotes += result["delta_downs"]
                post.hot_score = HotScoreCalculator.compute(
                    post.upvotes, post.downvotes, post.created_at
                )
                self._feed.update_post(post.subreddit_id, post)
                # Karma: net delta
                net_delta = result["delta_ups"] - result["delta_downs"]
                self._karma.apply_vote_event(post.author_id, item_id, "post", net_delta)
        return result

    def add_comment(
        self, post_id: str, author_id: str, body: str, parent_id: str = None
    ) -> Comment:
        if post_id not in self._posts:
            raise KeyError(f"Post {post_id} not found")
        comment = self._comment_tree.add_comment(post_id, author_id, body, parent_id)
        self._posts[post_id].comment_count += 1
        return comment

    # -- Feed ---------------------------------------------------------------

    def get_feed(
        self, subreddit_name: str, sort: str = "hot", limit: int = 25
    ) -> list[Post]:
        sub_id = self._sub_by_name.get(subreddit_name.lower())
        if not sub_id:
            raise KeyError(f"Subreddit r/{subreddit_name} not found")
        post_ids = self._feed.get_feed(sub_id, sort, limit)
        return [self._posts[pid] for pid in post_ids if pid in self._posts]

    def get_home_feed(self, user_id: str, sort: str = "hot", limit: int = 25) -> list[Post]:
        """Merge feeds from all subscribed subreddits."""
        sub_ids = self._subscriptions[user_id]
        all_posts = []
        for sub_id in sub_ids:
            sub = self._subreddits[sub_id]
            posts = self.get_feed(sub.name, sort=sort, limit=10)
            all_posts.extend(posts)
        if sort == "hot":
            all_posts.sort(key=lambda p: p.hot_score, reverse=True)
        elif sort == "new":
            all_posts.sort(key=lambda p: p.created_at, reverse=True)
        elif sort == "top":
            all_posts.sort(key=lambda p: p.score, reverse=True)
        return all_posts[:limit]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    system = RedditSystem()

    # Create subreddits
    prog = system.create_subreddit("programming", "News for programmers")
    ask  = system.create_subreddit("AskReddit", "Ask and answer questions")

    # Subscribe users
    system.subscribe_subreddit("u1", "programming")
    system.subscribe_subreddit("u1", "AskReddit")
    system.subscribe_subreddit("u2", "programming")

    # Submit posts
    p1 = system.submit_post("programming", "u1", "Python 4.0 released!", "Major changes...")
    p2 = system.submit_post("programming", "u2", "Why Rust is the future", "Thread safety...")
    p3 = system.submit_post("AskReddit", "u2", "What is your proudest moment?", "Share yours!")

    # Voting
    print("=== Voting Demo ===")
    r = system.vote("u2", p1.id, "post", 1)
    print(f"  u2 upvoted p1: ups={r['current_ups']} downs={r['current_downs']}")
    r = system.vote("u3", p1.id, "post", 1)
    r = system.vote("u4", p1.id, "post", -1)
    print(f"  p1 score: {system._posts[p1.id].score}, hot_score: {system._posts[p1.id].hot_score:.4f}")

    # Double-vote prevention
    r2 = system.vote("u2", p1.id, "post", 1)
    print(f"  u2 votes again (same dir): delta_ups={r2['delta_ups']} (should be 0)")

    # Vote change
    r3 = system.vote("u2", p1.id, "post", -1)
    print(f"  u2 changes to downvote: delta_ups={r3['delta_ups']}, delta_downs={r3['delta_downs']}")

    # Comments
    print("\n=== Comment Tree ===")
    c1 = system.add_comment(p1.id, "u3", "This is huge news!")
    c2 = system.add_comment(p1.id, "u4", "I'm skeptical...", parent_id=c1.id)
    c3 = system.add_comment(p1.id, "u5", "Why skeptical?", parent_id=c2.id)
    c4 = system.add_comment(p1.id, "u1", "Top-level different thread")

    rendered = system._comment_tree.render_tree(p1.id)
    for row in rendered:
        print(f"  {row['indent']}[{row['author']}] {row['body']} (score: {row['score']})")

    # Subtree
    print(f"\n  Subtree of c1 ({c1.id}): {len(system._comment_tree.get_subtree(c1.id))} comments")

    # Feed
    print("\n=== Subreddit Feed (hot) ===")
    for i, post in enumerate(system.get_feed("programming", sort="hot")):
        print(f"  {i+1}. [{post.hot_score:.4f}] {post.title} (score: {post.score})")

    print("\n=== Subreddit Feed (new) ===")
    for i, post in enumerate(system.get_feed("programming", sort="new")):
        print(f"  {i+1}. {post.title}")

    # Karma
    print("\n=== Karma Tracker ===")
    for uid in ["u1", "u2"]:
        k = system._karma.get_karma(uid)
        print(f"  {uid}: post_karma={k['post_karma']}, comment_karma={k['comment_karma']}, total={k['total']}")

    # Hot score formula demo
    print("\n=== Hot Score Formula ===")
    dt = datetime.now(timezone.utc)
    for ups, downs in [(1, 0), (10, 0), (100, 5), (1000, 200), (100, 90)]:
        score = HotScoreCalculator.compute(ups, downs, dt)
        print(f"  ups={ups:<5} downs={downs:<4} hot_score={score:.4f}")

    # Home feed
    print("\n=== Home Feed (u1) ===")
    for post in system.get_home_feed("u1", sort="hot"):
        sub = system._subreddits[post.subreddit_id]
        print(f"  r/{sub.name}: {post.title} | hot={post.hot_score:.4f}")
