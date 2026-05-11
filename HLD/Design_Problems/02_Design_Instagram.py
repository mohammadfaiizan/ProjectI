"""
Instagram System - Working Python Implementation
Demonstrates: photo posting, follow graph, feed generation (fan-out on write
              with hybrid for celebrities), like/comment system, hashtag search,
              LRU cache for feeds.
No external dependencies — standard library only.
"""

import time
import collections
import heapq
from datetime import datetime
from typing import Optional, List, Dict, Set


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CELEBRITY_THRESHOLD = 10          # In prod: 1,000,000. Low for demo.
FEED_PAGE_SIZE = 20
TIMELINE_MAX_LENGTH = 800         # Max posts in a user's cached timeline


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
class User:
    def __init__(self, user_id: int, username: str):
        self.user_id = user_id
        self.username = username
        self.follower_count = 0
        self.following_count = 0
        self.post_count = 0
        self.created_at = datetime.utcnow()

    def is_celebrity(self) -> bool:
        return self.follower_count >= CELEBRITY_THRESHOLD

    def __repr__(self):
        return f"User({self.user_id}, @{self.username}, followers={self.follower_count})"


class Post:
    _id_counter = 1

    def __init__(self, user_id: int, caption: str, hashtags: List[str], location: str = ""):
        self.post_id = Post._id_counter
        Post._id_counter += 1
        self.user_id = user_id
        self.caption = caption
        self.hashtags = hashtags
        self.location = location
        self.like_count = 0
        self.comment_count = 0
        self.created_at = datetime.utcnow()
        self.timestamp = time.time()          # float for sorting
        self.is_deleted = False
        # Simulated CDN URLs
        self.thumbnail_url = f"https://cdn.example.com/photos/{user_id}/{self.post_id}/thumb.jpg"
        self.medium_url = f"https://cdn.example.com/photos/{user_id}/{self.post_id}/medium.jpg"

    def __repr__(self):
        return f"Post({self.post_id}, user={self.user_id}, '{self.caption[:30]}...')"


class Comment:
    _id_counter = 1

    def __init__(self, post_id: int, user_id: int, content: str, parent_id: Optional[int] = None):
        self.comment_id = Comment._id_counter
        Comment._id_counter += 1
        self.post_id = post_id
        self.user_id = user_id
        self.content = content
        self.parent_id = parent_id
        self.created_at = datetime.utcnow()


# ---------------------------------------------------------------------------
# LRU Cache (for feed caching)
# ---------------------------------------------------------------------------
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: collections.OrderedDict = collections.OrderedDict()

    def get(self, key):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key, value) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def invalidate(self, key) -> None:
        self._cache.pop(key, None)


# ---------------------------------------------------------------------------
# Instagram Core System
# ---------------------------------------------------------------------------
class InstagramSystem:
    """
    Core Instagram system with:
    - User management and follow graph
    - Photo posting
    - Hybrid fan-out feed generation
    - Like and comment system
    - Hashtag search
    - Feed caching
    """

    def __init__(self):
        # Primary data stores
        self._users: Dict[int, User] = {}                  # user_id -> User
        self._posts: Dict[int, Post] = {}                  # post_id -> Post
        self._comments: Dict[int, Comment] = {}            # comment_id -> Comment

        # Follow graph (adjacency lists)
        # followers[user_id] = set of user_ids that follow this user
        self._followers: Dict[int, Set[int]] = collections.defaultdict(set)
        # following[user_id] = set of user_ids this user follows
        self._following: Dict[int, Set[int]] = collections.defaultdict(set)

        # User posts index: user_id -> list of (timestamp, post_id), newest first
        self._user_posts: Dict[int, List] = collections.defaultdict(list)

        # Pre-computed timelines: user_id -> list of (timestamp, post_id)
        # Sorted descending by timestamp (most recent first)
        self._timelines: Dict[int, List] = collections.defaultdict(list)

        # Likes: post_id -> set of user_ids
        self._likes: Dict[int, Set[int]] = collections.defaultdict(set)

        # Hashtag index: hashtag -> list of (timestamp, post_id)
        self._hashtag_index: Dict[str, List] = collections.defaultdict(list)

        # Feed cache (LRU)
        self._feed_cache = LRUCache(capacity=10000)

        # Notification queue (simplified)
        self._notifications: Dict[int, List] = collections.defaultdict(list)

    # ------------------------------------------------------------------
    # User Management
    # ------------------------------------------------------------------

    def create_user(self, user_id: int, username: str) -> User:
        user = User(user_id, username)
        self._users[user_id] = user
        return user

    def get_user(self, user_id: int) -> Optional[User]:
        return self._users.get(user_id)

    # ------------------------------------------------------------------
    # Follow System
    # ------------------------------------------------------------------

    def follow_user(self, follower_id: int, followee_id: int) -> dict:
        """
        Follow a user. For non-celebrity followees, also adds recent posts
        to follower's timeline (fan-out on follow).
        """
        if follower_id == followee_id:
            return {"error": "cannot follow yourself"}
        if follower_id not in self._users or followee_id not in self._users:
            return {"error": "user not found"}

        # Prevent duplicate follow
        if followee_id in self._following[follower_id]:
            return {"already_following": True}

        # Update follow graph
        self._following[follower_id].add(followee_id)
        self._followers[followee_id].add(follower_id)

        # Update counts
        self._users[follower_id].following_count += 1
        self._users[followee_id].follower_count += 1

        # Fan-out recent posts to follower's timeline
        followee = self._users[followee_id]
        if not followee.is_celebrity():
            # Add last 20 posts from followee to follower's timeline
            recent = self._user_posts[followee_id][-20:]
            for ts, post_id in recent:
                self._insert_into_timeline(follower_id, ts, post_id)

        # Notify followee
        self._notifications[followee_id].append({
            "type": "new_follower",
            "from_user": follower_id,
            "timestamp": time.time(),
        })

        # Invalidate follower's cached feed
        self._feed_cache.invalidate(follower_id)

        return {"following": True, "followee": self._users[followee_id].username}

    def unfollow_user(self, follower_id: int, followee_id: int) -> dict:
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
        self._feed_cache.invalidate(follower_id)
        return {"following": False}

    # ------------------------------------------------------------------
    # Post System
    # ------------------------------------------------------------------

    def post_photo(
        self,
        user_id: int,
        caption: str,
        hashtags: Optional[List[str]] = None,
        location: str = "",
    ) -> dict:
        """
        Create a new post and fan it out to followers' timelines.
        Fan-out strategy:
          - Regular users: push to each follower's timeline (fan-out on write)
          - Celebrities: skip push; followers pull on feed read (fan-out on read)
        """
        if user_id not in self._users:
            return {"error": "user not found"}

        hashtags = hashtags or []
        post = Post(user_id, caption, hashtags, location)
        self._posts[post.post_id] = post

        # Update user's post list (newest at end for easy slicing)
        self._user_posts[user_id].append((post.timestamp, post.post_id))
        self._users[user_id].post_count += 1

        # Index hashtags
        for tag in hashtags:
            self._hashtag_index[tag.lower()].append((post.timestamp, post.post_id))

        # Fan-out to followers
        poster = self._users[user_id]
        if not poster.is_celebrity():
            # Push model: write to each follower's timeline
            for follower_id in self._followers[user_id]:
                self._insert_into_timeline(follower_id, post.timestamp, post.post_id)
                self._feed_cache.invalidate(follower_id)
        # Celebrity: don't push — feed service will pull on read

        return {
            "post_id": post.post_id,
            "thumbnail_url": post.thumbnail_url,
            "created_at": post.created_at.isoformat(),
            "fanout_count": len(self._followers[user_id]) if not poster.is_celebrity() else 0,
        }

    def _insert_into_timeline(self, user_id: int, timestamp: float, post_id: int) -> None:
        """
        Insert a post into a user's timeline maintaining sorted order (descending).
        Uses bisect for O(log N) insertion position, O(N) insert (acceptable for demo).
        In production: Redis ZADD with score=timestamp.
        """
        timeline = self._timelines[user_id]
        # Insert maintaining descending order by timestamp
        # We store as (-timestamp, post_id) so heappush gives newest first
        entry = (-timestamp, post_id)
        heapq.heappush(timeline, entry)
        # Trim to max length (LRU-style, oldest entries fall off)
        if len(timeline) > TIMELINE_MAX_LENGTH:
            # Rebuild trimmed heap (keep TIMELINE_MAX_LENGTH newest)
            all_items = sorted(timeline)  # ascending by -ts = newest first
            self._timelines[user_id] = all_items[:TIMELINE_MAX_LENGTH]
            heapq.heapify(self._timelines[user_id])

    # ------------------------------------------------------------------
    # Feed Generation
    # ------------------------------------------------------------------

    def get_feed(self, user_id: int, page: int = 1, limit: int = FEED_PAGE_SIZE) -> dict:
        """
        Generate personalized feed for a user.
        Hybrid approach:
          - Pre-computed timeline contains posts from non-celebrity followees
          - On read, merge with latest posts from celebrity followees
        """
        if user_id not in self._users:
            return {"error": "user not found"}

        # Check cache first
        cache_key = f"feed:{user_id}:page:{page}"
        cached = self._feed_cache.get(cache_key)
        if cached:
            return {"posts": cached, "source": "cache", "page": page}

        # Step 1: Get pre-computed timeline entries
        timeline_heap = list(self._timelines[user_id])
        timeline_entries = sorted(timeline_heap)  # ascending (-ts, post_id) -> newest first

        # Step 2: Pull from celebrity followees (fan-out on read)
        celebrity_entries = []
        for followee_id in self._following[user_id]:
            if followee_id in self._users and self._users[followee_id].is_celebrity():
                # Take last 50 posts from celebrity
                celeb_posts = self._user_posts[followee_id][-50:]
                for ts, post_id in celeb_posts:
                    celebrity_entries.append((-ts, post_id))

        # Step 3: Merge pre-computed + celebrity entries
        all_entries = list(set(timeline_entries + celebrity_entries))
        all_entries.sort()  # sort by -timestamp (newest first)

        # Step 4: Paginate
        start = (page - 1) * limit
        end = start + limit
        page_entries = all_entries[start:end]

        # Step 5: Enrich with post metadata
        feed_posts = []
        for neg_ts, post_id in page_entries:
            post = self._posts.get(post_id)
            if post and not post.is_deleted:
                author = self._users.get(post.user_id, {})
                feed_posts.append({
                    "post_id": post.post_id,
                    "user": {
                        "user_id": post.user_id,
                        "username": getattr(author, "username", "unknown"),
                    },
                    "caption": post.caption,
                    "thumbnail_url": post.thumbnail_url,
                    "like_count": post.like_count,
                    "comment_count": post.comment_count,
                    "hashtags": post.hashtags,
                    "created_at": post.created_at.isoformat(),
                })

        # Cache this feed page
        self._feed_cache.put(cache_key, feed_posts)

        return {
            "posts": feed_posts,
            "source": "generated",
            "page": page,
            "has_more": end < len(all_entries),
        }

    # ------------------------------------------------------------------
    # Like System
    # ------------------------------------------------------------------

    def like_post(self, user_id: int, post_id: int) -> dict:
        """Like a post. Atomic increment on like count (Redis INCR in prod)."""
        post = self._posts.get(post_id)
        if not post or post.is_deleted:
            return {"error": "post not found"}

        if user_id in self._likes[post_id]:
            return {"liked": True, "like_count": post.like_count, "already_liked": True}

        self._likes[post_id].add(user_id)
        post.like_count += 1

        # Notify post owner
        if post.user_id != user_id:
            self._notifications[post.user_id].append({
                "type": "like",
                "from_user": user_id,
                "post_id": post_id,
                "timestamp": time.time(),
            })

        return {"liked": True, "like_count": post.like_count}

    def unlike_post(self, user_id: int, post_id: int) -> dict:
        post = self._posts.get(post_id)
        if not post:
            return {"error": "post not found"}
        self._likes[post_id].discard(user_id)
        post.like_count = max(0, post.like_count - 1)
        return {"liked": False, "like_count": post.like_count}

    # ------------------------------------------------------------------
    # Comment System
    # ------------------------------------------------------------------

    def add_comment(
        self,
        post_id: int,
        user_id: int,
        content: str,
        parent_id: Optional[int] = None,
    ) -> dict:
        post = self._posts.get(post_id)
        if not post or post.is_deleted:
            return {"error": "post not found"}

        comment = Comment(post_id, user_id, content, parent_id)
        self._comments[comment.comment_id] = comment
        post.comment_count += 1

        # Notify post owner
        if post.user_id != user_id:
            self._notifications[post.user_id].append({
                "type": "comment",
                "from_user": user_id,
                "post_id": post_id,
                "comment_id": comment.comment_id,
            })

        return {"comment_id": comment.comment_id, "content": content}

    def get_comments(self, post_id: int, limit: int = 20) -> list:
        result = []
        for c in self._comments.values():
            if c.post_id == post_id:
                result.append({
                    "comment_id": c.comment_id,
                    "user_id": c.user_id,
                    "content": c.content,
                    "parent_id": c.parent_id,
                    "created_at": c.created_at.isoformat(),
                })
        result.sort(key=lambda x: x["created_at"])
        return result[:limit]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_hashtag(self, tag: str, limit: int = 20) -> dict:
        """Search posts by hashtag. In prod: Elasticsearch query."""
        tag = tag.lower().lstrip("#")
        entries = self._hashtag_index.get(tag, [])
        # Sort by timestamp desc (most recent first)
        sorted_entries = sorted(entries, key=lambda x: -x[0])
        post_ids = [pid for _, pid in sorted_entries[:limit]]
        posts = []
        for pid in post_ids:
            post = self._posts.get(pid)
            if post and not post.is_deleted:
                posts.append({
                    "post_id": post.post_id,
                    "thumbnail_url": post.thumbnail_url,
                    "like_count": post.like_count,
                })
        return {"hashtag": tag, "post_count": len(entries), "posts": posts}

    def search_users(self, query: str, limit: int = 10) -> list:
        """Search users by username prefix. In prod: Elasticsearch with n-gram."""
        query = query.lower()
        results = []
        for user in self._users.values():
            if query in user.username.lower():
                results.append({
                    "user_id": user.user_id,
                    "username": user.username,
                    "follower_count": user.follower_count,
                })
        return sorted(results, key=lambda x: -x["follower_count"])[:limit]

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def get_user_profile(self, user_id: int) -> dict:
        user = self._users.get(user_id)
        if not user:
            return {"error": "not found"}
        posts = []
        for ts, pid in sorted(self._user_posts[user_id], reverse=True)[:9]:
            p = self._posts.get(pid)
            if p and not p.is_deleted:
                posts.append({"post_id": pid, "thumbnail_url": p.thumbnail_url})
        return {
            "user_id": user.user_id,
            "username": user.username,
            "follower_count": user.follower_count,
            "following_count": user.following_count,
            "post_count": user.post_count,
            "is_celebrity": user.is_celebrity(),
            "recent_posts": posts,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def run_demo():
    print("=" * 65)
    print("INSTAGRAM SYSTEM DEMO")
    print("=" * 65)

    ig = InstagramSystem()

    # Create users
    alice = ig.create_user(1, "alice")
    bob = ig.create_user(2, "bob")
    charlie = ig.create_user(3, "charlie")
    # Create a celebrity (will exceed CELEBRITY_THRESHOLD followers)
    celeb = ig.create_user(4, "mega_star")

    print(f"\n[1] Created users: {alice}, {bob}, {charlie}, {celeb}")

    # Everyone follows the celebrity
    for uid in [1, 2, 3]:
        # Directly inflate celeb's follower count to trigger celebrity status
        ig._followers[4].add(uid)
        ig._following[uid].add(4)
        ig._users[4].follower_count += 1
        ig._users[uid].following_count += 1

    print(f"\n[2] Celebrity threshold: {CELEBRITY_THRESHOLD}")
    print(f"    mega_star follower count: {celeb.follower_count}")
    print(f"    Is celebrity: {celeb.is_celebrity()}")

    # Follow relationships
    print("\n[3] Setting up follow graph")
    ig.follow_user(1, 2)  # alice follows bob
    ig.follow_user(1, 3)  # alice follows charlie
    ig.follow_user(2, 3)  # bob follows charlie
    print(f"    Alice follows: {ig._following[1]}")
    print(f"    Bob's followers: {ig._followers[2]}")

    # Post photos
    print("\n[4] Posting photos")
    p1 = ig.post_photo(2, "Sunset at the beach #sunset #travel", ["sunset", "travel"])
    p2 = ig.post_photo(2, "Coffee morning #coffee #morning", ["coffee", "morning"])
    p3 = ig.post_photo(3, "Mountain hike #nature #hiking", ["nature", "hiking"])
    p4 = ig.post_photo(3, "City lights at night #city #urban", ["city", "urban"])
    # Celebrity post (fan-out skipped)
    cp1 = ig.post_photo(4, "New album out now! #music #celebrity", ["music", "celebrity"])
    print(f"    Bob's post 1: id={p1['post_id']}, fanout_to={p1['fanout_count']} followers")
    print(f"    Celebrity post: id={cp1['post_id']}, fanout_to={cp1['fanout_count']} (skipped)")

    # Check alice's timeline (pre-computed)
    print(f"\n    Alice's pre-computed timeline size: {len(ig._timelines[1])}")

    # Get feed for alice (merges timeline + celebrity pull)
    print("\n[5] Alice's feed (page 1)")
    feed = ig.get_feed(1, page=1)
    print(f"    Source: {feed['source']}, Posts: {len(feed['posts'])}")
    for p in feed["posts"]:
        print(f"    - Post {p['post_id']} by @{p['user']['username']}: \"{p['caption'][:40]}\"")

    # Cache hit on second request
    feed2 = ig.get_feed(1, page=1)
    print(f"\n    Second feed request source: {feed2['source']}  (should be 'cache')")

    # Likes
    print("\n[6] Likes")
    r1 = ig.like_post(1, p1["post_id"])
    r2 = ig.like_post(3, p1["post_id"])
    r3 = ig.like_post(1, p1["post_id"])  # duplicate like
    print(f"    Like by alice:   {r1}")
    print(f"    Like by charlie: {r2}")
    print(f"    Duplicate like:  already_liked={r3.get('already_liked')}")

    # Comments
    print("\n[7] Comments")
    c1 = ig.add_comment(p1["post_id"], 1, "Gorgeous shot!")
    c2 = ig.add_comment(p1["post_id"], 3, "Where is this?", parent_id=c1["comment_id"])
    comments = ig.get_comments(p1["post_id"])
    for c in comments:
        prefix = "  [reply]" if c["parent_id"] else "  "
        print(f"{prefix} Comment {c['comment_id']}: \"{c['content']}\"")

    # Search
    print("\n[8] Hashtag search")
    results = ig.search_hashtag("sunset")
    print(f"    #sunset: {results['post_count']} posts found")
    results2 = ig.search_hashtag("music")
    print(f"    #music: {results2['post_count']} posts, posts={results2['posts']}")

    print("\n[9] User search")
    user_results = ig.search_users("a")
    for u in user_results:
        print(f"    @{u['username']} - {u['follower_count']} followers")

    # Profile
    print("\n[10] User profile")
    profile = ig.get_user_profile(2)
    print(f"    @{profile['username']}: {profile['post_count']} posts, "
          f"{profile['follower_count']} followers, celebrity={profile['is_celebrity']}")

    # Notifications
    print("\n[11] Notifications for bob (user_id=2)")
    notifs = ig._notifications[2]
    for n in notifs:
        print(f"    {n['type']} from user {n.get('from_user', '?')}")

    print("\n" + "=" * 65)
    print("DEMO COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
