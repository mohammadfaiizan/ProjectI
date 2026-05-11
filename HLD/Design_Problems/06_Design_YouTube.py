"""
YouTube System Design - Python Implementation
Demonstrates: video upload pipeline, processing state machine, adaptive streaming,
view count batching, recommendation engine, search, comments, subscriptions.
No external dependencies - standard library only.
"""

import hashlib
import time
import uuid
import heapq
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# Enums and Constants
# ─────────────────────────────────────────────

class VideoStatus(Enum):
    UPLOADING = "uploading"
    QUEUED = "queued"
    TRANSCODING = "transcoding"
    THUMBNAIL_GEN = "thumbnail_gen"
    UPLOADING_CDN = "uploading_cdn"
    READY = "ready"
    FAILED = "failed"


class Resolution(Enum):
    R360P = "360p"
    R480P = "480p"
    R720P = "720p"
    R1080P = "1080p"
    R4K = "4k"


BITRATE_MAP = {
    Resolution.R360P: 500,
    Resolution.R480P: 1000,
    Resolution.R720P: 2500,
    Resolution.R1080P: 5000,
    Resolution.R4K: 15000,
}

VIEW_FLUSH_INTERVAL = 60   # seconds


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class User:
    user_id: str
    username: str
    channel_name: str
    subscriber_count: int = 0
    subscribers: set = field(default_factory=set)   # user_ids who subscribe


@dataclass
class Video:
    video_id: str
    user_id: str
    title: str
    description: str
    tags: list
    duration_secs: int
    status: VideoStatus = VideoStatus.UPLOADING
    view_count: int = 0
    like_count: int = 0
    dislike_count: int = 0
    hls_manifest: dict = field(default_factory=dict)   # resolution -> s3_key
    thumbnail_url: str = ""
    created_at: float = field(default_factory=time.time)
    category: str = "general"


@dataclass
class Comment:
    comment_id: str
    video_id: str
    user_id: str
    content: str
    parent_id: Optional[str]
    like_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class ProcessingJob:
    job_id: str
    video_id: str
    created_at: float
    attempts: int = 0


# ─────────────────────────────────────────────
# Video Processing Pipeline (State Machine)
# ─────────────────────────────────────────────

class VideoProcessor:
    """
    Simulates a Kafka-consumer-based video transcoding pipeline.
    State machine: QUEUED -> TRANSCODING -> THUMBNAIL_GEN -> UPLOADING_CDN -> READY
    """

    def __init__(self, s3_store: dict):
        self.s3_store = s3_store      # Simulated S3
        self.processing_queue: deque = deque()
        self.failed_jobs: list = []

    def enqueue(self, video_id: str) -> ProcessingJob:
        job = ProcessingJob(
            job_id=str(uuid.uuid4())[:8],
            video_id=video_id,
            created_at=time.time()
        )
        self.processing_queue.append(job)
        return job

    def process_next(self, videos: dict) -> Optional[str]:
        """Process one job from the queue. Returns video_id if success."""
        if not self.processing_queue:
            return None

        job = self.processing_queue.popleft()
        video = videos.get(job.video_id)
        if not video:
            return None

        try:
            # Stage 1: Transcoding
            video.status = VideoStatus.TRANSCODING
            hls_segments = self._transcode(video)

            # Stage 2: Thumbnail generation
            video.status = VideoStatus.THUMBNAIL_GEN
            thumbnail = self._generate_thumbnail(video)

            # Stage 3: Upload to CDN origin (S3)
            video.status = VideoStatus.UPLOADING_CDN
            self._upload_to_s3(video, hls_segments, thumbnail)

            # Stage 4: Ready
            video.status = VideoStatus.READY
            video.hls_manifest = hls_segments
            video.thumbnail_url = thumbnail
            print(f"  [Processor] Video {video.video_id[:8]} processing complete -> READY")
            return video.video_id

        except Exception as e:
            job.attempts += 1
            if job.attempts < 3:
                self.processing_queue.append(job)   # Retry
                video.status = VideoStatus.QUEUED
                print(f"  [Processor] Retry {job.attempts} for video {video.video_id[:8]}")
            else:
                video.status = VideoStatus.FAILED
                self.failed_jobs.append(job)
                print(f"  [Processor] Video {video.video_id[:8]} FAILED after 3 attempts")
            return None

    def _transcode(self, video: Video) -> dict:
        """Simulate FFmpeg transcoding to multiple resolutions."""
        segments = {}
        for resolution in Resolution:
            bitrate = BITRATE_MAP[resolution]
            s3_key = f"videos/{video.video_id}/{resolution.value}/manifest.m3u8"
            # Simulate HLS segment generation
            num_segments = video.duration_secs // 10 + 1
            segments[resolution.value] = {
                "manifest": s3_key,
                "bitrate_kbps": bitrate,
                "segment_count": num_segments,
                "segments": [
                    f"videos/{video.video_id}/{resolution.value}/seg{i:04d}.ts"
                    for i in range(num_segments)
                ]
            }
        return segments

    def _generate_thumbnail(self, video: Video) -> str:
        """Simulate thumbnail generation from video frames."""
        frame_timestamps = [
            int(video.duration_secs * p) for p in [0.1, 0.25, 0.5, 0.75]
        ]
        best_frame = max(frame_timestamps)  # Simplified scoring
        thumbnail_key = f"thumbnails/{video.video_id}/{best_frame}.jpg"
        self.s3_store[thumbnail_key] = f"<thumbnail_data_at_{best_frame}s>"
        return f"https://cdn.example.com/{thumbnail_key}"

    def _upload_to_s3(self, video: Video, segments: dict, thumbnail: str):
        """Simulate uploading HLS segments to S3."""
        for resolution, data in segments.items():
            self.s3_store[data["manifest"]] = f"<HLS manifest for {resolution}>"
            for seg in data["segments"]:
                self.s3_store[seg] = f"<segment data>"


# ─────────────────────────────────────────────
# View Counter with Redis-like Batching
# ─────────────────────────────────────────────

class ViewCounter:
    """
    Simulates Redis INCR with periodic batch flush to DB.
    Pattern: increment in-memory -> flush to DB every N seconds.
    """

    def __init__(self, flush_interval: int = VIEW_FLUSH_INTERVAL):
        self._pending: defaultdict = defaultdict(int)   # video_id -> pending count
        self._last_flush = time.time()
        self._flush_interval = flush_interval

    def increment(self, video_id: str) -> int:
        self._pending[video_id] += 1
        self._maybe_flush()
        return self._pending[video_id]

    def get_pending(self, video_id: str) -> int:
        return self._pending.get(video_id, 0)

    def _maybe_flush(self):
        now = time.time()
        if now - self._last_flush >= self._flush_interval:
            self.flush()

    def flush(self, videos: dict = None) -> dict:
        """Flush all pending counts to DB (or return the batch for simulation)."""
        batch = dict(self._pending)
        if videos:
            for vid_id, count in batch.items():
                if vid_id in videos:
                    videos[vid_id].view_count += count
        self._pending.clear()
        self._last_flush = time.time()
        print(f"  [ViewCounter] Flushed {len(batch)} video view counts to DB")
        return batch


# ─────────────────────────────────────────────
# Inverted Index for Search
# ─────────────────────────────────────────────

class InvertedIndex:
    """Simple inverted index for video search. term -> {video_id: tf_score}"""

    STOP_WORDS = {"a", "an", "the", "is", "in", "on", "at", "to", "for", "of",
                  "and", "or", "but", "with", "by"}

    def __init__(self):
        self._index: defaultdict = defaultdict(dict)   # term -> {video_id: tf}
        self._doc_count = 0

    def add_document(self, video_id: str, title: str, description: str, tags: list):
        text = f"{title} {title} {description} {' '.join(tags)}"  # Title weighted 2x
        terms = self._tokenize(text)
        term_freq = defaultdict(int)
        for term in terms:
            term_freq[term] += 1
        total = len(terms) or 1
        for term, count in term_freq.items():
            tf = count / total
            self._index[term][video_id] = tf
        self._doc_count += 1

    def search(self, query: str, top_k: int = 10) -> list:
        """TF-IDF search. Returns list of (video_id, score) tuples."""
        query_terms = self._tokenize(query)
        scores: defaultdict = defaultdict(float)

        for term in query_terms:
            if term not in self._index:
                continue
            # IDF = log(N / df)
            df = len(self._index[term])
            idf = math.log((self._doc_count + 1) / (df + 1)) + 1
            for video_id, tf in self._index[term].items():
                scores[video_id] += tf * idf

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _tokenize(self, text: str) -> list:
        tokens = text.lower().split()
        cleaned = []
        for t in tokens:
            t = t.strip(".,!?;:'\"")
            if t and t not in self.STOP_WORDS and len(t) > 1:
                cleaned.append(t)
        return cleaned


# ─────────────────────────────────────────────
# Recommendation Engine
# ─────────────────────────────────────────────

class RecommendationEngine:
    """
    Collaborative filtering based on watch history similarity.
    Jaccard similarity between user watch histories to find similar users,
    then recommend videos watched by similar users but not by target user.
    """

    def __init__(self):
        self._watch_history: defaultdict = defaultdict(set)   # user_id -> {video_ids}
        self._video_watchers: defaultdict = defaultdict(set)  # video_id -> {user_ids}

    def record_watch(self, user_id: str, video_id: str):
        self._watch_history[user_id].add(video_id)
        self._video_watchers[video_id].add(user_id)

    def get_recommendations(self, user_id: str, top_k: int = 10) -> list:
        """Return recommended video_ids using collaborative filtering."""
        user_history = self._watch_history.get(user_id, set())
        if not user_history:
            return self._get_trending(top_k)

        # Find similar users by Jaccard similarity
        candidate_users = set()
        for vid in user_history:
            candidate_users.update(self._video_watchers.get(vid, set()))
        candidate_users.discard(user_id)

        # Compute Jaccard similarity
        similarities = []
        for other_id in candidate_users:
            other_history = self._watch_history[other_id]
            intersection = len(user_history & other_history)
            union = len(user_history | other_history)
            if union > 0:
                similarities.append((intersection / union, other_id))

        similarities.sort(reverse=True)
        top_similar = similarities[:20]  # Top 20 similar users

        # Aggregate their watches, exclude already-watched
        candidate_videos: defaultdict = defaultdict(float)
        for sim_score, similar_user_id in top_similar:
            for vid_id in self._watch_history[similar_user_id]:
                if vid_id not in user_history:
                    candidate_videos[vid_id] += sim_score

        sorted_candidates = sorted(candidate_videos.items(),
                                   key=lambda x: x[1], reverse=True)
        return [vid_id for vid_id, _ in sorted_candidates[:top_k]]

    def _get_trending(self, top_k: int) -> list:
        """Fallback: most-watched videos globally."""
        video_scores = [(len(watchers), vid_id)
                        for vid_id, watchers in self._video_watchers.items()]
        video_scores.sort(reverse=True)
        return [vid_id for _, vid_id in video_scores[:top_k]]


# ─────────────────────────────────────────────
# Main YouTube System
# ─────────────────────────────────────────────

class YouTubeSystem:
    """
    Orchestrates all YouTube subsystems.
    Simulates the full flow from upload to stream to recommendation.
    """

    def __init__(self):
        self.users: dict = {}
        self.videos: dict = {}
        self.comments: dict = defaultdict(list)        # video_id -> [Comment]
        self.likes: dict = {}                           # (user_id, video_id) -> bool
        self.s3_store: dict = {}
        self.processor = VideoProcessor(self.s3_store)
        self.view_counter = ViewCounter(flush_interval=60)
        self.search_index = InvertedIndex()
        self.recommender = RecommendationEngine()
        self._kafka_queue: deque = deque()

    # ── User Management ──────────────────────

    def create_user(self, username: str, channel_name: str) -> User:
        user_id = str(uuid.uuid4())[:8]
        user = User(user_id=user_id, username=username, channel_name=channel_name)
        self.users[user_id] = user
        return user

    # ── Video Upload Pipeline ─────────────────

    def upload_video(self, user_id: str, title: str, description: str,
                     tags: list, duration_secs: int) -> Video:
        """
        Simulates chunked upload to S3 + Kafka event -> processing queue.
        """
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")

        video_id = str(uuid.uuid4())[:8]
        video = Video(
            video_id=video_id,
            user_id=user_id,
            title=title,
            description=description,
            tags=tags,
            duration_secs=duration_secs,
        )
        self.videos[video_id] = video

        # Simulate chunked upload
        chunk_size_mb = 5
        num_chunks = math.ceil(duration_secs * 0.002)  # Approx size based on duration
        print(f"  [Upload] Video '{title}' - {num_chunks} chunks uploaded to S3")

        # Publish to Kafka -> processing queue
        video.status = VideoStatus.QUEUED
        job = self.processor.enqueue(video_id)
        self._kafka_queue.append({"event": "video.upload.raw", "video_id": video_id,
                                   "job_id": job.job_id})
        print(f"  [Upload] Published to Kafka: video.upload.raw -> job {job.job_id}")
        return video

    def process_video(self, video_id: str) -> bool:
        """Trigger video processing (normally done by a Kafka consumer worker)."""
        result = self.processor.process_next(self.videos)
        if result == video_id:
            # Index in search after processing
            video = self.videos[video_id]
            self.search_index.add_document(
                video_id, video.title, video.description, video.tags
            )
            print(f"  [Index] Video '{video.title}' indexed for search")
            return True
        return False

    # ── Streaming ─────────────────────────────

    def get_video_stream(self, video_id: str,
                         preferred_quality: str = "auto") -> dict:
        """Returns HLS manifest URL for adaptive streaming."""
        video = self.videos.get(video_id)
        if not video or video.status != VideoStatus.READY:
            return {"error": "Video not available"}

        if preferred_quality == "auto":
            # Return master manifest (client chooses resolution)
            return {
                "video_id": video_id,
                "title": video.title,
                "stream_type": "HLS",
                "master_manifest": f"https://cdn.example.com/videos/{video_id}/master.m3u8",
                "available_qualities": list(video.hls_manifest.keys()),
                "duration_secs": video.duration_secs,
            }
        else:
            quality_data = video.hls_manifest.get(preferred_quality)
            if not quality_data:
                return {"error": f"Quality {preferred_quality} not available"}
            return {
                "video_id": video_id,
                "quality": preferred_quality,
                "bitrate_kbps": quality_data["bitrate_kbps"],
                "manifest_url": f"https://cdn.example.com/{quality_data['manifest']}",
            }

    # ── View Counting ─────────────────────────

    def increment_view_count(self, video_id: str, user_id: str) -> int:
        """
        Increment view via Redis INCR (simulated).
        Also records watch for recommendations.
        """
        if video_id not in self.videos:
            return 0

        pending = self.view_counter.increment(video_id)
        self.recommender.record_watch(user_id, video_id)

        # Real-time count = DB count + pending Redis count
        db_count = self.videos[video_id].view_count
        return db_count + pending

    def flush_view_counts(self):
        """Manually trigger batch flush (normally runs every 60s)."""
        self.view_counter.flush(self.videos)

    # ── Comments ──────────────────────────────

    def add_comment(self, video_id: str, user_id: str,
                    content: str, parent_id: Optional[str] = None) -> Comment:
        """Add comment to video (stored in Cassandra-like partition by video_id)."""
        if video_id not in self.videos:
            raise ValueError("Video not found")

        comment = Comment(
            comment_id=str(uuid.uuid4())[:8],
            video_id=video_id,
            user_id=user_id,
            content=content,
            parent_id=parent_id
        )
        self.comments[video_id].append(comment)
        return comment

    def get_comments(self, video_id: str, limit: int = 20,
                     parent_id: Optional[str] = None) -> list:
        """Paginate comments for a video."""
        all_comments = self.comments.get(video_id, [])
        filtered = [c for c in all_comments if c.parent_id == parent_id]
        return sorted(filtered, key=lambda c: c.created_at, reverse=True)[:limit]

    # ── Likes / Dislikes ──────────────────────

    def like_video(self, user_id: str, video_id: str,
                   action: str = "like") -> dict:
        """Like, dislike, or remove reaction. action: 'like'|'dislike'|'remove'"""
        video = self.videos.get(video_id)
        if not video:
            return {"error": "Video not found"}

        key = (user_id, video_id)
        existing = self.likes.get(key)

        # Remove previous reaction
        if existing is True and existing != (action == "like"):
            video.like_count = max(0, video.like_count - 1)
        elif existing is False and existing != (action == "dislike"):
            video.dislike_count = max(0, video.dislike_count - 1)

        if action == "like":
            self.likes[key] = True
            video.like_count += 1
        elif action == "dislike":
            self.likes[key] = False
            video.dislike_count += 1
        elif action == "remove":
            self.likes.pop(key, None)

        return {"likes": video.like_count, "dislikes": video.dislike_count}

    # ── Subscriptions ─────────────────────────

    def subscribe(self, subscriber_id: str, channel_id: str) -> dict:
        """Subscribe to a channel. Fan-out on write for <1M subscriber channels."""
        if subscriber_id not in self.users or channel_id not in self.users:
            return {"error": "User not found"}

        channel = self.users[channel_id]
        subscriber = self.users[subscriber_id]
        channel.subscribers.add(subscriber_id)
        channel.subscriber_count = len(channel.subscribers)

        return {
            "subscribed": True,
            "channel": channel.channel_name,
            "subscriber_count": channel.subscriber_count
        }

    def get_subscription_feed(self, user_id: str, limit: int = 20) -> list:
        """Return latest videos from subscribed channels (fan-out on read)."""
        user = self.users.get(user_id)
        if not user:
            return []

        # Find channels this user subscribes to
        subscribed_channels = [
            uid for uid, u in self.users.items()
            if user_id in u.subscribers
        ]

        # Gather all ready videos from subscribed channels
        feed_videos = []
        for channel_id in subscribed_channels:
            for video in self.videos.values():
                if (video.user_id == channel_id
                        and video.status == VideoStatus.READY):
                    feed_videos.append(video)

        # Sort by publish time (newest first)
        feed_videos.sort(key=lambda v: v.created_at, reverse=True)
        return feed_videos[:limit]

    # ── Recommendations ───────────────────────

    def get_recommendations(self, user_id: str, limit: int = 10) -> list:
        """Return personalized video recommendations."""
        rec_ids = self.recommender.get_recommendations(user_id, limit)
        result = []
        for vid_id in rec_ids:
            video = self.videos.get(vid_id)
            if video and video.status == VideoStatus.READY:
                result.append({
                    "video_id": vid_id,
                    "title": video.title,
                    "channel": self.users[video.user_id].channel_name
                    if video.user_id in self.users else "Unknown",
                    "views": video.view_count,
                    "duration_secs": video.duration_secs,
                })
        return result

    # ── Search ────────────────────────────────

    def search_videos(self, query: str, limit: int = 10) -> list:
        """Full-text search using TF-IDF inverted index."""
        results = self.search_index.search(query, top_k=limit)
        enriched = []
        for video_id, score in results:
            video = self.videos.get(video_id)
            if video and video.status == VideoStatus.READY:
                enriched.append({
                    "video_id": video_id,
                    "title": video.title,
                    "score": round(score, 4),
                    "views": video.view_count,
                    "thumbnail": video.thumbnail_url,
                })
        return enriched


# ─────────────────────────────────────────────
# Demo / Simulation
# ─────────────────────────────────────────────

def run_demo():
    print("=" * 60)
    print("YOUTUBE SYSTEM DESIGN DEMO")
    print("=" * 60)

    yt = YouTubeSystem()

    # Create users
    print("\n--- Creating Users ---")
    alice = yt.create_user("alice", "Alice Codes")
    bob = yt.create_user("bob", "Bob Learns")
    carol = yt.create_user("carol", "Carol Builds")
    dave = yt.create_user("dave", "Dave Watches")
    print(f"Created users: {alice.username}, {bob.username}, {carol.username}, {dave.username}")

    # Upload videos
    print("\n--- Video Upload Pipeline ---")
    v1 = yt.upload_video(
        alice.user_id, "Python Tutorial for Beginners",
        "Learn Python from scratch. Variables, loops, functions.",
        ["python", "tutorial", "beginner", "programming"], duration_secs=1800
    )
    v2 = yt.upload_video(
        alice.user_id, "Advanced Python: Decorators and Metaclasses",
        "Deep dive into Python internals.",
        ["python", "advanced", "decorators"], duration_secs=2700
    )
    v3 = yt.upload_video(
        bob.user_id, "System Design Interview Guide",
        "How to ace system design interviews. HLD, LLD.",
        ["system design", "interview", "HLD"], duration_secs=3600
    )
    v4 = yt.upload_video(
        carol.user_id, "Building a REST API with FastAPI",
        "Create production-ready APIs in Python.",
        ["python", "fastapi", "api", "backend"], duration_secs=2400
    )

    # Process all videos
    print("\n--- Processing Videos (Transcoding Pipeline) ---")
    for video in [v1, v2, v3, v4]:
        yt.process_video(video.video_id)

    # Subscriptions
    print("\n--- Subscriptions ---")
    yt.subscribe(bob.user_id, alice.user_id)
    yt.subscribe(carol.user_id, alice.user_id)
    yt.subscribe(dave.user_id, alice.user_id)
    yt.subscribe(dave.user_id, bob.user_id)
    print(f"Alice's subscriber count: {yt.users[alice.user_id].subscriber_count}")
    print(f"Bob's subscriber count: {yt.users[bob.user_id].subscriber_count}")

    # View counts (simulating Redis INCR + batch flush)
    print("\n--- View Count Tracking (Redis INCR simulation) ---")
    for _ in range(150):
        yt.increment_view_count(v1.video_id, dave.user_id)
    for _ in range(80):
        yt.increment_view_count(v2.video_id, bob.user_id)
    for _ in range(200):
        yt.increment_view_count(v3.video_id, alice.user_id)

    print(f"Pending (Redis) views for v1: {yt.view_counter.get_pending(v1.video_id)}")
    print(f"Real-time view count for v1: {yt.view_counter.get_pending(v1.video_id) + v1.view_count}")
    yt.flush_view_counts()
    print(f"After flush - v1 views in DB: {v1.view_count}")
    print(f"After flush - v3 views in DB: {v3.view_count}")

    # Watch history for recommendations
    print("\n--- Building Watch History for Recommendations ---")
    for user_id in [bob.user_id, carol.user_id]:
        yt.recommender.record_watch(user_id, v1.video_id)
        yt.recommender.record_watch(user_id, v2.video_id)
    yt.recommender.record_watch(bob.user_id, v3.video_id)

    # Comments
    print("\n--- Comments ---")
    c1 = yt.add_comment(v1.video_id, bob.user_id, "Great tutorial, very helpful!")
    c2 = yt.add_comment(v1.video_id, carol.user_id, "I learned so much from this.")
    c3 = yt.add_comment(v1.video_id, dave.user_id, "Can you do a part 2?",
                         parent_id=c1.comment_id)
    top_level = yt.get_comments(v1.video_id, parent_id=None)
    print(f"Top-level comments on '{v1.title}':")
    for c in top_level:
        print(f"  @{yt.users[c.user_id].username}: {c.content}")

    # Likes
    print("\n--- Likes / Dislikes ---")
    yt.like_video(bob.user_id, v1.video_id, "like")
    yt.like_video(carol.user_id, v1.video_id, "like")
    yt.like_video(dave.user_id, v1.video_id, "like")
    yt.like_video(alice.user_id, v3.video_id, "dislike")
    result = yt.like_video(bob.user_id, v3.video_id, "like")
    print(f"v1 reactions: {v1.like_count} likes, {v1.dislike_count} dislikes")
    print(f"v3 reactions: {v3.like_count} likes, {v3.dislike_count} dislikes")

    # Search
    print("\n--- Search (TF-IDF Inverted Index) ---")
    for query in ["python tutorial", "system design interview", "advanced python"]:
        results = yt.search_videos(query, limit=3)
        print(f"Query: '{query}' -> {len(results)} results")
        for r in results:
            print(f"  [{r['score']:.4f}] {r['title']}")

    # Adaptive streaming
    print("\n--- Adaptive Bitrate Streaming ---")
    stream = yt.get_video_stream(v1.video_id, "auto")
    print(f"Video: {stream['title']}")
    print(f"Available qualities: {stream['available_qualities']}")
    stream_720 = yt.get_video_stream(v1.video_id, "720p")
    print(f"720p stream URL: {stream_720['manifest_url']}")
    print(f"720p bitrate: {stream_720['bitrate_kbps']} Kbps")

    # Recommendations
    print("\n--- Recommendations ---")
    recs = yt.get_recommendations(carol.user_id, limit=5)
    print(f"Recommendations for @{carol.username}:")
    for r in recs:
        print(f"  [{r['channel']}] {r['title']} ({r['views']} views)")

    # Subscription feed
    print("\n--- Subscription Feed ---")
    feed = yt.get_subscription_feed(dave.user_id, limit=5)
    print(f"Subscription feed for @{dave.username}:")
    for v in feed:
        channel = yt.users[v.user_id].channel_name
        print(f"  [{channel}] {v.title} - {v.view_count} views")

    # Scale estimates
    print("\n--- Back-of-Envelope Scale Estimates ---")
    stats = {
        "Upload rate": "500 hours video/minute -> 1TB raw/minute",
        "Storage growth": "288TB/day (after transcoding compression)",
        "CDN bandwidth": "~30 Tbps peak (10M concurrent * 3 Mbps)",
        "View QPS": "100,000 views/second (1B hours/day watched)",
        "Redis INCR": "Handles 1M+ ops/second per node",
        "DB write reduction": "60-second batch flush = 6000x fewer DB writes",
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
