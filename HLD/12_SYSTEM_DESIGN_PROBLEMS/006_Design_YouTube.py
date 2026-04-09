"""
SYSTEM DESIGN: YOUTUBE (Video Streaming)
==========================================

Problem Statement:
Design a video platform where users can upload, search, and stream videos.

Functional Requirements:
  - Upload video (up to 4GB, any format)
  - Stream video (adaptive bitrate)
  - Search videos by title/description
  - Recommendations feed
  - Like, comment, subscribe
  - View count

Non-Functional Requirements:
  - 2B MAU, 500 hours of video uploaded per minute
  - 1B hours watched daily → ~41M concurrent viewers
  - Video start latency < 2s (with CDN)
  - 99.99% availability

Video Upload Pipeline:
  1. Client → chunked HTTP upload → Upload Service → Raw Storage (S3)
  2. S3 PUT event → Kafka → Transcoding Service (FFmpeg workers)
  3. Transcode to: 360p, 480p, 720p, 1080p, 4K
  4. Package as HLS (.m3u8 + .ts segments) or MPEG-DASH
  5. Store segments to S3; update metadata DB
  6. Push to CDN (invalidate/preload)

HLS (HTTP Live Streaming):
  Master playlist (.m3u8): lists all quality variants.
  Per-quality playlist: lists 2-10s .ts segments.
  Client player: picks quality based on bandwidth → ABR (Adaptive Bitrate).
  Manifest:
    #EXTM3U
    #EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360
    /video/{id}/360p/index.m3u8
    #EXT-X-STREAM-INF:BANDWIDTH=4800000,RESOLUTION=1920x1080
    /video/{id}/1080p/index.m3u8

CDN Strategy:
  Pull CDN: origin is S3; CDN caches on first request.
  Push CDN: proactively push popular videos to edge nodes.
  Long-tail: most videos are rarely watched; pull CDN is fine.
  Hot videos (viral): push to all PoPs immediately.

Video Recommendations:
  Collaborative filtering: users who watched X also watched Y.
  Content-based: similar topics, channel.
  ML model: user_embedding × video_embedding → score.
  YouTube actually uses deep neural network (DNN) with two towers:
    Candidate generation → Ranking → Served recommendations.

Search:
  Elasticsearch: index title, description, tags, transcript.
  Query: BM25 + recency boost + view count boost.

View Count:
  Approximate at scale: Redis INCR per video.
  Periodic flush to DB. Not real-time exact.
  Anti-fraud: dedup by user_id; don't count bot traffic.

Thumbnail:
  Auto-generated: FFmpeg extract frame at 10% of duration.
  A/B tested: YouTube tests multiple thumbnails, picks highest CTR.
"""

from __future__ import annotations

import time
import uuid
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict


# ─────────────────────────────────────────────
# VIDEO RESOLUTION / QUALITY
# ─────────────────────────────────────────────

@dataclass
class VideoQuality:
    label:       str      # "360p", "1080p"
    width:       int
    height:      int
    bitrate_kbps: int
    codec:       str = "h264"

    def estimated_size_mb(self, duration_s: float) -> float:
        return self.bitrate_kbps * duration_s / 8 / 1024


VIDEO_QUALITIES = [
    VideoQuality("360p",  640,  360,   800),
    VideoQuality("480p",  854,  480,  1500),
    VideoQuality("720p",  1280, 720,  3000),
    VideoQuality("1080p", 1920, 1080, 6000),
    VideoQuality("4K",    3840, 2160, 20000),
]


# ─────────────────────────────────────────────
# VIDEO METADATA
# ─────────────────────────────────────────────

class VideoStatus(Enum):
    UPLOADING    = "uploading"
    PROCESSING   = "processing"
    READY        = "ready"
    FAILED       = "failed"


@dataclass
class Video:
    video_id:    str
    channel_id:  str
    title:       str
    description: str
    duration_s:  float
    status:      VideoStatus
    created_at:  float
    view_count:  int = 0
    like_count:  int = 0
    thumbnail_url: Optional[str] = None
    available_qualities: List[str] = field(default_factory=list)
    tags:        List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# HLS MANIFEST GENERATOR
# ─────────────────────────────────────────────

class HLSManifest:
    """Generates HLS playlist manifests."""

    SEGMENT_DURATION = 6   # seconds

    def master_playlist(self, video_id: str,
                        qualities: List[VideoQuality]) -> str:
        lines = ["#EXTM3U", "#EXT-X-VERSION:3"]
        for q in qualities:
            lines.append(f"#EXT-X-STREAM-INF:BANDWIDTH={q.bitrate_kbps*1000},"
                         f"RESOLUTION={q.width}x{q.height},CODECS=\"avc1.42E01E\"")
            lines.append(f"https://cdn.youtube.com/video/{video_id}/{q.label}/index.m3u8")
        return "\n".join(lines)

    def quality_playlist(self, video_id: str, quality: VideoQuality,
                         duration_s: float) -> str:
        n_segments = math.ceil(duration_s / self.SEGMENT_DURATION)
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{self.SEGMENT_DURATION}",
            "#EXT-X-MEDIA-SEQUENCE:0",
        ]
        for i in range(n_segments):
            seg_dur = min(self.SEGMENT_DURATION,
                          duration_s - i * self.SEGMENT_DURATION)
            lines.append(f"#EXTINF:{seg_dur:.3f},")
            lines.append(f"https://cdn.youtube.com/video/{video_id}"
                         f"/{quality.label}/seg{i:04d}.ts")
        lines.append("#EXT-X-ENDLIST")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# TRANSCODING PIPELINE
# ─────────────────────────────────────────────

@dataclass
class TranscodeJob:
    job_id:      str
    video_id:    str
    source_url:  str
    target_qualities: List[str]
    created_at:  float
    started_at:  Optional[float] = None
    completed_at: Optional[float] = None
    status:      str = "queued"   # queued, running, done, failed


class TranscodingService:
    """
    Simulates FFmpeg transcoding workers.
    In prod: auto-scaling EC2 spot instances.
    """

    def __init__(self):
        self._jobs:    Dict[str, TranscodeJob] = {}
        self._workers: int = 8    # parallel workers

    def submit(self, video_id: str, source_url: str) -> TranscodeJob:
        job = TranscodeJob(
            job_id    = uuid.uuid4().hex[:8],
            video_id  = video_id,
            source_url= source_url,
            target_qualities = [q.label for q in VIDEO_QUALITIES],
            created_at= time.time(),
        )
        self._jobs[job.job_id] = job
        return job

    def process(self, job: TranscodeJob, video_duration_s: float
                ) -> Dict[str, float]:
        """
        Simulate transcoding. Returns {quality: output_size_mb}.
        """
        job.status     = "running"
        job.started_at = time.time()
        sizes = {}
        for q in VIDEO_QUALITIES:
            if q.label in job.target_qualities:
                sizes[q.label] = q.estimated_size_mb(video_duration_s)
        job.status       = "done"
        job.completed_at = time.time()
        return sizes

    def queue_depth(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "queued")


# ─────────────────────────────────────────────
# VIEW COUNTER (Redis-like)
# ─────────────────────────────────────────────

class ViewCounter:
    """
    Redis INCR for view counts with deduplication.
    Flushed to DB every 30 seconds.
    """

    def __init__(self, dedup_window_s: float = 300):
        self._counts: Dict[str, int]       = defaultdict(int)
        self._seen:   Dict[str, set]       = defaultdict(set)  # video→{user_ids}
        self._dedup_window = dedup_window_s
        self._last_flush   = time.time()

    def increment(self, video_id: str, viewer_id: str) -> bool:
        """Returns True if counted (not duplicate)."""
        key = video_id
        if viewer_id in self._seen[key]:
            return False   # already counted in window
        self._seen[key].add(viewer_id)
        self._counts[key] += 1
        return True

    def count(self, video_id: str) -> int:
        return self._counts.get(video_id, 0)

    def flush(self) -> Dict[str, int]:
        """Returns counts to flush to DB; resets window."""
        flushed = dict(self._counts)
        # In prod: add to DB counters, clear only seen set (not counts)
        self._seen.clear()
        self._last_flush = time.time()
        return flushed


# ─────────────────────────────────────────────
# VIDEO SEARCH (Elasticsearch simulation)
# ─────────────────────────────────────────────

class VideoSearch:
    """Simple inverted index for video search."""

    def __init__(self):
        self._index: Dict[str, List[str]] = defaultdict(list)  # token → [video_ids]
        self._videos: Dict[str, Video] = {}

    def index(self, video: Video):
        self._videos[video.video_id] = video
        tokens = set()
        for text in [video.title, video.description] + video.tags:
            tokens.update(text.lower().split())
        for tok in tokens:
            self._index[tok].append(video.video_id)

    def search(self, query: str, limit: int = 10) -> List[Video]:
        tokens    = query.lower().split()
        vid_score: Dict[str, int] = defaultdict(int)
        for tok in tokens:
            for vid_id in self._index.get(tok, []):
                vid_score[vid_id] += 1

        # Boost by view count
        def rank(vid_id: str) -> float:
            v     = self._videos.get(vid_id)
            views = math.log1p(v.view_count) if v else 0
            return vid_score[vid_id] * 2 + views

        ranked = sorted(vid_score.keys(), key=rank, reverse=True)
        return [self._videos[v] for v in ranked[:limit] if v in self._videos]


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE (simplified)
# ─────────────────────────────────────────────

class RecommendationEngine:
    """
    Simple collaborative filtering: users who watched X also watched Y.
    In prod: two-tower DNN model (candidate generation + ranking).
    """

    def __init__(self):
        # video_id → {video_ids watched by same viewers}
        self._co_watch: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._user_history: Dict[str, List[str]] = defaultdict(list)

    def record_watch(self, user_id: str, video_id: str):
        history = self._user_history[user_id]
        # Update co-watch counts for all previously watched videos
        for prev_vid in history[-10:]:
            self._co_watch[prev_vid][video_id] += 1
            self._co_watch[video_id][prev_vid] += 1
        history.append(video_id)

    def recommend(self, user_id: str, n: int = 5) -> List[str]:
        watched = set(self._user_history.get(user_id, []))
        scores: Dict[str, int] = defaultdict(int)
        for vid_id in list(watched)[-5:]:
            for co_vid, cnt in self._co_watch.get(vid_id, {}).items():
                if co_vid not in watched:
                    scores[co_vid] += cnt
        return sorted(scores.keys(), key=lambda v: -scores[v])[:n]


# ─────────────────────────────────────────────
# YOUTUBE SERVICE
# ─────────────────────────────────────────────

class YouTubeService:
    def __init__(self):
        self._videos:     Dict[str, Video]  = {}
        self._transcode   = TranscodingService()
        self._hls         = HLSManifest()
        self._views       = ViewCounter()
        self._search      = VideoSearch()
        self._recs        = RecommendationEngine()

    def upload(self, channel_id: str, title: str, description: str,
               duration_s: float, tags: List[str] = None) -> Tuple[Video, str]:
        video_id = uuid.uuid4().hex[:12]
        raw_url  = f"s3://yt-raw/{video_id}/original.mp4"

        video = Video(
            video_id    = video_id,
            channel_id  = channel_id,
            title       = title,
            description = description,
            duration_s  = duration_s,
            status      = VideoStatus.PROCESSING,
            created_at  = time.time(),
            tags        = tags or [],
            thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        )
        self._videos[video_id] = video

        # Submit transcoding job
        job   = self._transcode.submit(video_id, raw_url)
        sizes = self._transcode.process(job, duration_s)

        video.available_qualities = [q for q in sizes]
        video.status = VideoStatus.READY

        # Index for search
        self._search.index(video)
        return video, job.job_id

    def watch(self, video_id: str, viewer_id: str
              ) -> Optional[Tuple[Video, str]]:
        """Returns (video, HLS master playlist)."""
        video = self._videos.get(video_id)
        if not video or video.status != VideoStatus.READY:
            return None
        counted = self._views.increment(video_id, viewer_id)
        if counted:
            video.view_count += 1
        self._recs.record_watch(viewer_id, video_id)
        available = [q for q in VIDEO_QUALITIES
                     if q.label in video.available_qualities]
        manifest = self._hls.master_playlist(video_id, available)
        return video, manifest

    def like(self, video_id: str):
        v = self._videos.get(video_id)
        if v:
            v.like_count += 1

    def search(self, query: str, limit: int = 10) -> List[Video]:
        return self._search.search(query, limit)

    def recommend(self, user_id: str) -> List[Video]:
        vid_ids = self._recs.recommend(user_id)
        return [self._videos[v] for v in vid_ids if v in self._videos]


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_youtube():
    print("=" * 65)
    print("SYSTEM DESIGN: YOUTUBE")
    print("=" * 65)

    svc = YouTubeService()
    random.seed(42)

    # ── Upload Videos ─────────────────────────
    print("\n[1] VIDEO UPLOAD AND TRANSCODING")
    print("─" * 55)

    videos_data = [
        ("chan-001", "Python Tutorial: Advanced Decorators", "Deep dive into decorators", 1800, ["python","tutorial","decorators"]),
        ("chan-001", "System Design Interview Guide",        "How to ace system design",   3600, ["system-design","interview","engineering"]),
        ("chan-002", "Cooking Pasta Carbonara",              "Authentic Italian recipe",   1200, ["cooking","pasta","italian","recipe"]),
        ("chan-003", "Gym Workout: Full Body",               "45 minute full body session",2700, ["gym","workout","fitness"]),
        ("chan-001", "Python vs Go: Which to Choose",        "Comparison of two languages",2100, ["python","go","programming"]),
    ]

    uploaded = []
    for channel, title, desc, dur, tags in videos_data:
        video, job_id = svc.upload(channel, title, desc, dur, tags)
        uploaded.append(video)
        sizes = {q: f"{v:.1f}MB" for q, v in
                 zip(video.available_qualities,
                     [q.estimated_size_mb(dur) for q in VIDEO_QUALITIES
                      if q.label in video.available_qualities])}
        print(f"  '{title[:40]}' ({dur//60}m)")
        print(f"    id={video.video_id}  qualities={video.available_qualities}")

    # ── HLS Manifest ──────────────────────────
    print("\n[2] HLS MANIFEST")
    print("─" * 55)

    vid = uploaded[0]
    available = [q for q in VIDEO_QUALITIES if q.label in vid.available_qualities]
    manifest  = svc._hls.master_playlist(vid.video_id, available)
    for line in manifest.split("\n")[:10]:
        print(f"  {line}")

    seg_manifest = svc._hls.quality_playlist(vid.video_id, available[0], vid.duration_s)
    seg_lines    = seg_manifest.split("\n")
    print(f"\n  360p playlist ({vid.duration_s}s):")
    for line in seg_lines[:8]:
        print(f"    {line}")
    print(f"    ... ({len(seg_lines)} total lines)")

    # ── Watch and Views ───────────────────────
    print("\n[3] VIDEO STREAMING AND VIEW COUNTING")
    print("─" * 55)

    users = [f"user_{i:04d}" for i in range(50)]
    for user in users[:30]:
        for v in random.sample(uploaded, k=random.randint(1, 3)):
            svc.watch(v.video_id, user)

    # Duplicate views (should not double count)
    for _ in range(20):
        svc.watch(uploaded[0].video_id, "user_0001")

    print(f"  Video views after 50 users watching:")
    for v in uploaded:
        print(f"    '{v.title[:40]}': {v.view_count} views")

    # ── Search ────────────────────────────────
    print("\n[4] VIDEO SEARCH")
    print("─" * 55)

    queries = ["python tutorial", "cooking pasta", "system design"]
    for q in queries:
        results = svc.search(q, limit=3)
        print(f"  Query: '{q}'")
        for r in results:
            print(f"    - '{r.title[:45]}' ({r.view_count} views)")

    # ── Recommendations ───────────────────────
    print("\n[5] RECOMMENDATIONS")
    print("─" * 55)

    # Simulate user with history
    for vid in uploaded[:3]:
        svc._recs.record_watch("heavy_user", vid.video_id)

    recs = svc.recommend("heavy_user")
    print(f"  Recommendations for user who watched first 3 videos:")
    for r in recs:
        print(f"    - '{r.title[:50]}'")

    # ── Transcoding Cost Estimation ───────────
    print("\n[6] TRANSCODING COST ANALYSIS")
    print("─" * 55)

    upload_rate = 500 * 60   # 500 hours/min × 60s = 30000 s of video per second
    print(f"  Upload rate: 500 hours/min")
    print(f"  = {upload_rate:,} seconds of source video per second")
    print()
    print(f"  {'Quality':<8} {'Bitrate':>10}  {'1h video':>10}  {'Cost/GB~'}")
    for q in VIDEO_QUALITIES:
        size_1h = q.estimated_size_mb(3600) / 1024
        print(f"  {q.label:<8} {q.bitrate_kbps:>8}kbps  {size_1h:>8.1f}GB  ~$0.023/GB")

    # ── Architecture ──────────────────────────
    print("\n[7] YOUTUBE ARCHITECTURE SUMMARY")
    print("─" * 55)

    arch = [
        ("Upload",          "Resumable chunked upload → S3 raw bucket"),
        ("Transcoding",     "S3 event → Kafka → EC2 Spot FFmpeg workers → S3"),
        ("HLS/DASH",        "Segments stored in S3; pulled by CDN on request"),
        ("CDN",             "Akamai/Fastly globally; long-tail pull; hot videos push"),
        ("ABR player",      "Shaka Player / HLS.js picks quality by bandwidth"),
        ("Metadata DB",     "Spanner / MySQL: video_id, channel, title, status"),
        ("View count",      "Redis INCR per video; dedup by user; flush to Spanner"),
        ("Search",          "Elasticsearch: title/desc/tags + BM25 + view boost"),
        ("Recommendations", "YouTube: two-tower DNN; candidate gen + ranking"),
        ("Comments",        "Spanner (ACID) with per-video comment thread"),
    ]
    for component, detail in arch:
        print(f"  {component:<18} {detail}")


if __name__ == "__main__":
    demonstrate_youtube()
