"""
Design Netflix — Python Simulation
======================================
Simulates core Netflix mechanics:
  - Video transcoding pipeline (multi-quality/codec job queue)
  - Adaptive bitrate selection (buffer-based ABR algorithm)
  - Recommendation engine (collaborative filtering + content-based)
  - Content CDN with edge node selection by user location
  - Watch history store (per-profile time-ordered records)
  - Continue Watching feature (incomplete watches)
  - A/B Testing framework (stable experiment assignment + metric tracking)
"""

import uuid
import time
import math
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from enum import Enum
from datetime import datetime


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

class VideoQuality(Enum):
    Q480P = "480p"
    Q720P = "720p"
    Q1080P = "1080p"
    Q4K = "4K"
    Q4K_HDR = "4K_HDR"


class Codec(Enum):
    H264 = "H.264"
    H265 = "H.265"
    AV1 = "AV1"


# Bitrates in Kbps per quality level
QUALITY_BITRATES = {
    VideoQuality.Q480P: 700,
    VideoQuality.Q720P: 3000,
    VideoQuality.Q1080P: 6000,
    VideoQuality.Q4K: 15000,
    VideoQuality.Q4K_HDR: 20000
}

SEGMENT_DURATION = 4   # seconds


@dataclass
class Show:
    show_id: str
    title: str
    show_type: str       # 'movie' or 'series'
    genres: list[str]
    cast: list[str]
    description: str
    duration_sec: int    # For movies
    is_original: bool = False
    rating: float = 0.0
    release_year: int = 2024


@dataclass
class TranscodeJob:
    job_id: str
    show_id: str
    quality: VideoQuality
    codec: Codec
    status: str = "queued"    # queued / processing / done / failed
    s3_key: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class WatchRecord:
    profile_id: str
    show_id: str
    position_sec: int
    duration_sec: int
    watched_at: float = field(default_factory=time.time)

    def completion_pct(self) -> float:
        if self.duration_sec == 0:
            return 0
        return self.position_sec / self.duration_sec

    def is_completed(self) -> bool:
        return self.completion_pct() >= 0.9


@dataclass
class EdgeNode:
    node_id: str
    name: str
    latitude: float
    longitude: float
    capacity_gbps: float
    cached_shows: set = field(default_factory=set)


@dataclass
class StreamSession:
    session_id: str
    profile_id: str
    show_id: str
    current_quality: VideoQuality
    buffer_seconds: float
    start_time: float = field(default_factory=time.time)
    segments_loaded: int = 0
    quality_switches: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Video Transcoder
# ---------------------------------------------------------------------------

class VideoTranscoder:
    """
    Job queue for encoding videos into multiple quality/codec combinations.
    In production: Distributed encoding farm with GPU workers (AWS Elemental).
    """

    # Default encoding targets per show
    DEFAULT_PROFILES = [
        (VideoQuality.Q480P, Codec.H264),
        (VideoQuality.Q720P, Codec.H264),
        (VideoQuality.Q1080P, Codec.H264),
        (VideoQuality.Q1080P, Codec.H265),
        (VideoQuality.Q4K, Codec.AV1),
        (VideoQuality.Q4K_HDR, Codec.AV1),
    ]

    def __init__(self):
        self._jobs: dict[str, TranscodeJob] = {}
        self._encoded_content: dict[str, dict] = {}    # show_id -> {quality -> s3_key}

    def submit_encoding(self, show_id: str, raw_s3_key: str) -> list[str]:
        """Submit encoding jobs for all quality/codec combinations."""
        job_ids = []
        for quality, codec in self.DEFAULT_PROFILES:
            job = TranscodeJob(
                job_id=str(uuid.uuid4()),
                show_id=show_id,
                quality=quality,
                codec=codec
            )
            self._jobs[job.job_id] = job
            job_ids.append(job.job_id)
        return job_ids

    def process_jobs(self, show_id: str):
        """Simulate processing all encoding jobs for a show."""
        show_jobs = [j for j in self._jobs.values()
                     if j.show_id == show_id and j.status == "queued"]

        if show_id not in self._encoded_content:
            self._encoded_content[show_id] = {}

        for job in show_jobs:
            job.status = "processing"
            s3_key = f"encoded/{show_id}/{job.quality.value}/{job.codec.value}/manifest.mpd"
            job.s3_key = s3_key
            job.status = "done"
            job.completed_at = time.time()
            self._encoded_content[show_id][job.quality] = s3_key

        return len(show_jobs)

    def get_available_qualities(self, show_id: str) -> list[VideoQuality]:
        return list(self._encoded_content.get(show_id, {}).keys())

    def get_manifest_url(self, show_id: str, quality: VideoQuality) -> Optional[str]:
        content = self._encoded_content.get(show_id, {})
        s3_key = content.get(quality)
        return f"https://s3.amazonaws.com/{s3_key}" if s3_key else None

    def encoding_report(self) -> dict:
        done = sum(1 for j in self._jobs.values() if j.status == "done")
        total = len(self._jobs)
        return {"total_jobs": total, "completed": done, "success_rate": f"{done/total:.0%}" if total else "0%"}


# ---------------------------------------------------------------------------
# Adaptive Bitrate Selector
# ---------------------------------------------------------------------------

class AdaptiveBitrateSelector:
    """
    Buffer-based ABR algorithm.
    Selects video quality tier based on current buffer level and bandwidth estimate.
    """

    LOW_WATERMARK = 5.0      # Buffer below this -> downgrade quality (seconds)
    HIGH_WATERMARK = 20.0    # Buffer above this -> upgrade quality (seconds)

    def __init__(self, available_qualities: list[VideoQuality]):
        # Sort by bitrate ascending
        self._qualities = sorted(
            available_qualities,
            key=lambda q: QUALITY_BITRATES.get(q, 0)
        )

    def select_quality(self, buffer_seconds: float,
                       bandwidth_kbps: float,
                       current_quality: VideoQuality) -> tuple[VideoQuality, str]:
        """
        Returns (selected_quality, reason).
        Buffer-based with bandwidth upper bound.
        """
        if not self._qualities:
            return current_quality, "no_qualities_available"

        # Find max quality supportable by bandwidth
        max_quality_by_bandwidth = self._qualities[0]
        for q in self._qualities:
            required_bitrate = QUALITY_BITRATES.get(q, 0)
            # Allow up to 80% of bandwidth (leave headroom for variance)
            if required_bitrate <= bandwidth_kbps * 0.8:
                max_quality_by_bandwidth = q

        # Buffer-based decision
        current_idx = self._qualities.index(current_quality) if current_quality in self._qualities else 0
        max_idx = self._qualities.index(max_quality_by_bandwidth)

        if buffer_seconds < self.LOW_WATERMARK:
            # Low buffer -> aggressive downgrade
            new_idx = max(0, current_idx - 1)
            reason = f"buffer_low ({buffer_seconds:.1f}s)"
        elif buffer_seconds > self.HIGH_WATERMARK and current_idx < max_idx:
            # High buffer and bandwidth supports upgrade
            new_idx = min(len(self._qualities) - 1, current_idx + 1)
            reason = f"buffer_high ({buffer_seconds:.1f}s)"
        else:
            new_idx = min(current_idx, max_idx)
            reason = "stable"

        return self._qualities[new_idx], reason

    def simulate_stream(self, show_id: str, profile_id: str,
                        bandwidth_profile: list[float],
                        start_quality: VideoQuality = VideoQuality.Q480P) -> dict:
        """
        Simulate a streaming session with varying bandwidth.
        bandwidth_profile: list of Kbps values per 4-second interval.
        """
        session = StreamSession(
            session_id=str(uuid.uuid4()),
            profile_id=profile_id,
            show_id=show_id,
            current_quality=start_quality,
            buffer_seconds=0.0
        )

        quality_distribution: dict[str, int] = defaultdict(int)
        stall_events = 0
        buffer = 0.0

        for interval, bandwidth in enumerate(bandwidth_profile):
            # Try to download a segment at current quality
            segment_bitrate = QUALITY_BITRATES.get(session.current_quality, 3000)
            segment_size_kb = segment_bitrate * SEGMENT_DURATION / 8

            download_time = segment_size_kb / bandwidth if bandwidth > 0 else 10
            net_buffer_change = SEGMENT_DURATION - download_time

            if net_buffer_change < 0 and buffer < abs(net_buffer_change):
                stall_events += 1
                buffer = 0
            else:
                buffer = max(0, buffer + net_buffer_change)

            # ABR quality selection for next segment
            new_quality, reason = self.select_quality(
                buffer, bandwidth, session.current_quality
            )
            if new_quality != session.current_quality:
                session.quality_switches.append({
                    "interval": interval,
                    "from": session.current_quality.value,
                    "to": new_quality.value,
                    "reason": reason
                })
                session.current_quality = new_quality

            quality_distribution[new_quality.value] += 1
            session.segments_loaded += 1

        session.buffer_seconds = buffer
        return {
            "session_id": session.session_id,
            "total_segments": session.segments_loaded,
            "quality_switches": len(session.quality_switches),
            "stall_events": stall_events,
            "final_buffer_sec": round(buffer, 1),
            "quality_distribution": dict(quality_distribution),
            "switch_log": session.quality_switches[:5]
        }


# ---------------------------------------------------------------------------
# Recommendation Engine
# ---------------------------------------------------------------------------

class RecommendationEngine:
    """Collaborative filtering + content-based filtering."""

    def __init__(self):
        # profile_id -> {show_id: completion_pct}
        self._watch_matrix: dict[str, dict[str, float]] = defaultdict(dict)
        # show_id -> Show metadata
        self._shows: dict[str, Show] = {}

    def register_show(self, show: Show):
        self._shows[show.show_id] = show

    def record_watch(self, profile_id: str, show_id: str, completion: float):
        self._watch_matrix[profile_id][show_id] = completion

    def get_recommendations(self, profile_id: str, n: int = 10) -> list[dict]:
        """Blend collaborative + content-based recommendations."""
        collab = self._collaborative_filter(profile_id, n * 2)
        content = self._content_based(profile_id, n * 2)

        # Merge and deduplicate
        seen = set(self._watch_matrix.get(profile_id, {}).keys())
        combined: dict[str, float] = {}
        for show_id, score in collab:
            if show_id not in seen:
                combined[show_id] = combined.get(show_id, 0) + score * 0.6
        for show_id, score in content:
            if show_id not in seen:
                combined[show_id] = combined.get(show_id, 0) + score * 0.4

        ranked = sorted(combined, key=combined.get, reverse=True)[:n]
        return [
            {
                "show_id": sid,
                "title": self._shows[sid].title if sid in self._shows else "Unknown",
                "score": round(combined[sid], 3)
            }
            for sid in ranked if sid in self._shows
        ]

    def _collaborative_filter(self, profile_id: str, n: int) -> list[tuple[str, float]]:
        user_watched = self._watch_matrix.get(profile_id, {})
        if not user_watched:
            return []

        # Find similar users
        similarities: dict[str, float] = {}
        for other_id, other_watched in self._watch_matrix.items():
            if other_id == profile_id:
                continue
            common = set(user_watched.keys()) & set(other_watched.keys())
            if len(common) < 1:
                continue
            # Cosine similarity based on completion rates
            dot = sum(user_watched[s] * other_watched[s] for s in common)
            norm_a = math.sqrt(sum(v**2 for v in user_watched.values()))
            norm_b = math.sqrt(sum(v**2 for v in other_watched.values()))
            if norm_a > 0 and norm_b > 0:
                similarities[other_id] = dot / (norm_a * norm_b)

        top_similar = sorted(similarities, key=similarities.get, reverse=True)[:5]

        # Score unseen items
        scores: dict[str, float] = defaultdict(float)
        for other_id in top_similar:
            sim = similarities[other_id]
            for show_id, completion in self._watch_matrix[other_id].items():
                if show_id not in user_watched:
                    scores[show_id] += sim * completion

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

    def _content_based(self, profile_id: str, n: int) -> list[tuple[str, float]]:
        user_watched = self._watch_matrix.get(profile_id, {})
        if not user_watched:
            return []

        # Build user genre preference vector
        genre_scores: dict[str, float] = defaultdict(float)
        for show_id, completion in user_watched.items():
            show = self._shows.get(show_id)
            if show:
                for genre in show.genres:
                    genre_scores[genre] += completion

        # Score all unseen shows by genre overlap
        candidate_scores: dict[str, float] = defaultdict(float)
        for show_id, show in self._shows.items():
            if show_id in user_watched:
                continue
            for genre in show.genres:
                candidate_scores[show_id] += genre_scores.get(genre, 0)

        return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]


# ---------------------------------------------------------------------------
# Content CDN
# ---------------------------------------------------------------------------

class ContentCDN:
    """Edge node selection based on user location + cache state."""

    def __init__(self, edge_nodes: list[EdgeNode]):
        self._nodes = {n.node_id: n for n in edge_nodes}
        self._cache_hits = 0
        self._cache_misses = 0

    def pre_populate(self, node_id: str, show_ids: list[str]):
        """Pre-populate edge node with popular content."""
        if node_id in self._nodes:
            self._nodes[node_id].cached_shows.update(show_ids)

    def get_stream_url(self, show_id: str, quality: VideoQuality,
                       user_lat: float, user_lon: float) -> dict:
        """Return best CDN URL for user location."""
        nearest_node = self._find_nearest_node(user_lat, user_lon, show_id)

        if nearest_node and show_id in nearest_node.cached_shows:
            self._cache_hits += 1
            url = (f"https://cdn-{nearest_node.node_id}.netflix.com/"
                   f"{show_id}/{quality.value}/manifest.mpd")
            return {
                "url": url,
                "edge_node": nearest_node.name,
                "cache_status": "HIT",
                "latency_ms": self._estimate_latency(user_lat, user_lon, nearest_node)
            }
        else:
            self._cache_misses += 1
            # Fall back to origin S3
            if nearest_node:
                nearest_node.cached_shows.add(show_id)   # Cache for future
            return {
                "url": f"https://s3.amazonaws.com/netflix-content/{show_id}/{quality.value}/manifest.mpd",
                "edge_node": "origin_s3",
                "cache_status": "MISS",
                "latency_ms": 80
            }

    def _find_nearest_node(self, lat: float, lon: float,
                            show_id: str) -> Optional[EdgeNode]:
        nodes_with_content = [n for n in self._nodes.values()
                               if show_id in n.cached_shows]
        if not nodes_with_content:
            # Return nearest node regardless of cache status
            all_nodes = list(self._nodes.values())
            return min(all_nodes,
                       key=lambda n: self._distance(lat, lon, n.latitude, n.longitude))
        return min(nodes_with_content,
                   key=lambda n: self._distance(lat, lon, n.latitude, n.longitude))

    def _distance(self, lat1, lon1, lat2, lon2) -> float:
        return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

    def _estimate_latency(self, lat1, lon1, node: EdgeNode) -> float:
        dist = self._distance(lat1, lon1, node.latitude, node.longitude)
        return round(dist * 10 + 2, 1)

    def cache_stats(self) -> dict:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {"hits": self._cache_hits, "misses": self._cache_misses,
                "hit_rate": f"{hit_rate:.1%}"}


# ---------------------------------------------------------------------------
# Watch History Store
# ---------------------------------------------------------------------------

class WatchHistoryStore:
    """Per-profile time-ordered watch records (Cassandra simulation)."""

    def __init__(self):
        # profile_id -> list of WatchRecord (sorted by watched_at DESC)
        self._records: dict[str, list[WatchRecord]] = defaultdict(list)

    def record_progress(self, profile_id: str, show_id: str,
                        position_sec: int, duration_sec: int):
        records = self._records[profile_id]
        # Update existing record if same show watched today
        for r in records:
            if r.show_id == show_id:
                r.position_sec = position_sec
                r.watched_at = time.time()
                return
        records.append(WatchRecord(
            profile_id=profile_id,
            show_id=show_id,
            position_sec=position_sec,
            duration_sec=duration_sec
        ))
        records.sort(key=lambda r: r.watched_at, reverse=True)

    def get_continue_watching(self, profile_id: str) -> list[dict]:
        """Return incomplete watches sorted by most recently watched."""
        records = self._records.get(profile_id, [])
        incomplete = [r for r in records if not r.is_completed()]
        return [
            {
                "show_id": r.show_id,
                "position_sec": r.position_sec,
                "duration_sec": r.duration_sec,
                "completion_pct": f"{r.completion_pct():.0%}",
                "watched_at": datetime.fromtimestamp(r.watched_at).strftime('%Y-%m-%d %H:%M')
            }
            for r in incomplete[:20]
        ]

    def get_watch_history(self, profile_id: str, limit: int = 20) -> list[dict]:
        records = self._records.get(profile_id, [])
        return [
            {
                "show_id": r.show_id,
                "completion_pct": f"{r.completion_pct():.0%}",
                "position_sec": r.position_sec
            }
            for r in records[:limit]
        ]


# ---------------------------------------------------------------------------
# A/B Testing Framework
# ---------------------------------------------------------------------------

class ABTestingFramework:
    """Stable experiment assignment + metric tracking."""

    def __init__(self):
        self._experiments: dict[str, dict] = {}
        self._metrics: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    def create_experiment(self, experiment_id: str, variants: list[str],
                          traffic_pct: float = 1.0):
        self._experiments[experiment_id] = {
            "variants": variants,
            "traffic_pct": traffic_pct,
            "created_at": time.time()
        }

    def get_variant(self, experiment_id: str, profile_id: str) -> Optional[str]:
        """Stable assignment via consistent hash."""
        if experiment_id not in self._experiments:
            return None

        exp = self._experiments[experiment_id]
        # Stable hash: same profile always gets same variant
        hash_val = int(hashlib.md5(f"{experiment_id}:{profile_id}".encode()).hexdigest(), 16)
        bucket = (hash_val % 10000) / 10000   # 0.0 to 1.0

        if bucket > exp["traffic_pct"]:
            return "control"   # Not in experiment traffic

        variant_idx = hash_val % len(exp["variants"])
        return exp["variants"][variant_idx]

    def record_metric(self, experiment_id: str, profile_id: str,
                      metric_name: str, value: float):
        variant = self.get_variant(experiment_id, profile_id)
        if variant:
            self._metrics[experiment_id][f"{variant}:{metric_name}"].append(value)

    def get_results(self, experiment_id: str) -> dict:
        results = {}
        exp = self._experiments.get(experiment_id, {})
        for variant in exp.get("variants", []):
            variant_results = {}
            for key, values in self._metrics[experiment_id].items():
                if key.startswith(f"{variant}:") and values:
                    metric = key.split(":", 1)[1]
                    variant_results[metric] = {
                        "mean": round(sum(values) / len(values), 3),
                        "n": len(values)
                    }
            results[variant] = variant_results
        return results


# ---------------------------------------------------------------------------
# Main Netflix System
# ---------------------------------------------------------------------------

class NetflixSystem:
    def __init__(self, edge_nodes: list[EdgeNode]):
        self.transcoder = VideoTranscoder()
        self.watch_history = WatchHistoryStore()
        self.cdn = ContentCDN(edge_nodes)
        self.recommendations = RecommendationEngine()
        self.ab_testing = ABTestingFramework()
        self._shows: dict[str, Show] = {}

    def upload_content(self, show: Show) -> dict:
        """Full content ingestion pipeline."""
        self._shows[show.show_id] = show
        self.recommendations.register_show(show)

        # Submit encoding jobs
        raw_key = f"raw/{show.show_id}/source.mp4"
        job_ids = self.transcoder.submit_encoding(show.show_id, raw_key)

        # Process encoding (async in production)
        jobs_done = self.transcoder.process_jobs(show.show_id)
        qualities = self.transcoder.get_available_qualities(show.show_id)

        return {
            "show_id": show.show_id,
            "jobs_submitted": len(job_ids),
            "jobs_completed": jobs_done,
            "available_qualities": [q.value for q in qualities]
        }

    def stream_video(self, profile_id: str, show_id: str,
                     user_lat: float, user_lon: float,
                     bandwidth_kbps: float = 6000) -> dict:
        """Get stream URL and resume position."""
        qualities = self.transcoder.get_available_qualities(show_id)
        if not qualities:
            return {"error": "content_not_available"}

        # ABR: select appropriate quality for bandwidth
        abr = AdaptiveBitrateSelector(qualities)
        start_quality = VideoQuality.Q480P   # Start low, ramp up
        selected, _ = abr.select_quality(
            buffer_seconds=10.0, bandwidth_kbps=bandwidth_kbps,
            current_quality=start_quality
        )

        cdn_result = self.cdn.get_stream_url(show_id, selected, user_lat, user_lon)

        # Get resume position
        history = self.watch_history.get_watch_history(profile_id, limit=50)
        resume_pos = next(
            (h["position_sec"] for h in history if h["show_id"] == show_id), 0
        )

        return {
            "show_id": show_id,
            "selected_quality": selected.value,
            "manifest_url": cdn_result["url"],
            "edge_node": cdn_result["edge_node"],
            "cache_status": cdn_result["cache_status"],
            "resume_position_sec": resume_pos,
            "bandwidth_kbps": bandwidth_kbps
        }

    def record_watch_progress(self, profile_id: str, show_id: str,
                               position_sec: int):
        show = self._shows.get(show_id)
        duration = show.duration_sec if show else 7200
        self.watch_history.record_progress(profile_id, show_id, position_sec, duration)
        self.recommendations.record_watch(profile_id, show_id, position_sec / duration)

    def get_recommendations(self, profile_id: str) -> list[dict]:
        return self.recommendations.get_recommendations(profile_id, n=10)

    def get_continue_watching(self, profile_id: str) -> list[dict]:
        return self.watch_history.get_continue_watching(profile_id)

    def search_content(self, query: str) -> list[dict]:
        tokens = query.lower().split()
        results = []
        for show in self._shows.values():
            text = f"{show.title} {show.description} {' '.join(show.genres)} {' '.join(show.cast)}"
            score = sum(1 for t in tokens if t in text.lower())
            if score > 0:
                results.append({"show_id": show.show_id, "title": show.title,
                                 "type": show.show_type, "score": score})
        return sorted(results, key=lambda x: x["score"], reverse=True)[:10]


# ---------------------------------------------------------------------------
# Demo / Simulation
# ---------------------------------------------------------------------------

def run_simulation():
    print("=" * 65)
    print("  Netflix Streaming Platform Simulation")
    print("=" * 65)

    # Setup CDN edge nodes
    edge_nodes = [
        EdgeNode("oca_nyc", "New York OCA", 40.71, -74.01, 100.0),
        EdgeNode("oca_la", "Los Angeles OCA", 34.05, -118.24, 100.0),
        EdgeNode("oca_chicago", "Chicago OCA", 41.88, -87.63, 80.0),
        EdgeNode("oca_london", "London OCA", 51.51, -0.13, 60.0),
    ]

    netflix = NetflixSystem(edge_nodes)

    # Upload content
    shows = [
        Show("s1", "Stranger Things S5", "series", ["Sci-Fi", "Horror", "Drama"],
             ["Millie Bobby Brown", "Finn Wolfhard"], "Kids fight interdimensional evil",
             duration_sec=3600, is_original=True, rating=9.0),
        Show("s2", "The Crown S6", "series", ["Drama", "History"],
             ["Imelda Staunton"], "Royal family drama", duration_sec=3000,
             is_original=True, rating=8.5),
        Show("s3", "Breaking Bad", "series", ["Drama", "Crime", "Thriller"],
             ["Bryan Cranston", "Aaron Paul"], "Chemistry teacher turns criminal",
             duration_sec=2700, rating=9.5),
        Show("s4", "Squid Game S2", "series", ["Thriller", "Drama", "Sci-Fi"],
             ["Lee Jung-jae"], "Deadly competition returns",
             duration_sec=3200, is_original=True, rating=8.8),
        Show("s5", "The Witcher S4", "series", ["Fantasy", "Drama", "Action"],
             ["Henry Cavill"], "Monster hunter adventures",
             duration_sec=3600, is_original=True, rating=8.2),
    ]

    print("\n[1] Content ingestion pipeline")
    for show in shows:
        result = netflix.upload_content(show)
        qualities_str = ", ".join(result["available_qualities"])
        print(f"    {show.title:<30} | {result['jobs_completed']} jobs | [{qualities_str}]")

    # Pre-populate CDN with popular content
    netflix.cdn.pre_populate("oca_nyc", ["s1", "s2", "s3"])
    netflix.cdn.pre_populate("oca_la", ["s1", "s3", "s4"])
    netflix.cdn.pre_populate("oca_chicago", ["s1", "s3"])
    print("\n    CDN pre-populated with popular shows at US edge nodes")

    # Streaming
    print("\n[2] User streaming requests")
    users = [
        ("alice", 40.73, -73.99, 8000),    # NYC high bandwidth
        ("bob", 34.06, -118.25, 3000),      # LA medium bandwidth
        ("carol", 41.88, -87.63, 1500),     # Chicago low bandwidth
        ("dave", 51.51, -0.13, 5000),       # London
    ]
    for user_id, lat, lon, bw in users:
        stream = netflix.stream_video(user_id, "s1", lat, lon, bw)
        print(f"    {user_id:<8} | {stream['selected_quality']:<6} | "
              f"CDN: {stream['cache_status']} ({stream['edge_node']}) | "
              f"BW: {bw} Kbps")

    # ABR simulation
    print("\n[3] Adaptive bitrate streaming simulation")
    qualities = netflix.transcoder.get_available_qualities("s1")
    abr = AdaptiveBitrateSelector(qualities)
    # Bandwidth profile: starts high, drops, recovers
    bandwidth_profile = [8000, 7000, 3000, 1500, 800, 1200, 3000, 6000, 8000, 7500]
    abr_result = abr.simulate_stream("s1", "alice", bandwidth_profile, VideoQuality.Q480P)
    print(f"    Segments loaded    : {abr_result['total_segments']}")
    print(f"    Quality switches   : {abr_result['quality_switches']}")
    print(f"    Stall events       : {abr_result['stall_events']}")
    print(f"    Final buffer (sec) : {abr_result['final_buffer_sec']}")
    print(f"    Quality distribution: {abr_result['quality_distribution']}")
    if abr_result['switch_log']:
        print("    Switch log:")
        for sw in abr_result['switch_log']:
            print(f"      Interval {sw['interval']}: {sw['from']} -> {sw['to']} [{sw['reason']}]")

    # Record watch progress
    print("\n[4] Watch progress and Continue Watching")
    netflix.record_watch_progress("alice", "s1", 1800)    # 50% through Stranger Things
    netflix.record_watch_progress("alice", "s3", 2400)    # 89% through Breaking Bad
    netflix.record_watch_progress("alice", "s4", 800)     # 25% through Squid Game
    netflix.record_watch_progress("bob", "s1", 3600)      # Completed Stranger Things
    netflix.record_watch_progress("bob", "s2", 1500)      # 50% through The Crown
    netflix.record_watch_progress("carol", "s3", 2700)    # Completed Breaking Bad

    continue_watching = netflix.get_continue_watching("alice")
    print("    Alice's Continue Watching:")
    for cw in continue_watching:
        show = netflix._shows.get(cw['show_id'])
        title = show.title if show else "Unknown"
        print(f"      {title:<30} | {cw['completion_pct']} | pos={cw['position_sec']}s")

    # Recommendations
    print("\n[5] Personalized recommendations")
    # Give more users watch history for better collaborative filtering
    netflix.record_watch_progress("bob", "s3", 2700)
    netflix.record_watch_progress("bob", "s4", 3200)
    netflix.record_watch_progress("carol", "s1", 3600)
    netflix.record_watch_progress("carol", "s5", 2000)

    recs = netflix.get_recommendations("alice")
    print("    Alice's recommendations:")
    for r in recs[:5]:
        print(f"      {r['title']:<35} (score: {r['score']})")

    # Search
    print("\n[6] Content search")
    for query in ["strange sci-fi", "crime drama", "fantasy monster"]:
        results = netflix.search_content(query)
        top = results[0]["title"] if results else "No results"
        print(f"    '{query}' -> {len(results)} results, top: {top}")

    # A/B Testing
    print("\n[7] A/B testing framework")
    netflix.ab_testing.create_experiment(
        "thumbnail_test", ["landscape", "closeup", "action_shot"], traffic_pct=1.0
    )
    netflix.ab_testing.create_experiment(
        "recommendation_algo", ["cf_only", "hybrid", "content_only"], traffic_pct=0.5
    )

    profiles = ["alice", "bob", "carol", "dave", "eve", "frank"]
    print("    Thumbnail experiment assignments:")
    for p in profiles:
        variant = netflix.ab_testing.get_variant("thumbnail_test", p)
        print(f"      {p:<8}: {variant}")
        # Record CTR metric
        ctr = {"landscape": 0.12, "closeup": 0.15, "action_shot": 0.18}.get(variant, 0.1)
        netflix.ab_testing.record_metric("thumbnail_test", p, "ctr", ctr)
        netflix.ab_testing.record_metric("thumbnail_test", p, "stream_start", ctr * 0.8)

    results = netflix.ab_testing.get_results("thumbnail_test")
    print("\n    Thumbnail A/B Test Results:")
    for variant, metrics in results.items():
        if metrics:
            ctr_data = metrics.get("ctr", {})
            print(f"      {variant:<15}: CTR={ctr_data.get('mean', 0):.3f} (n={ctr_data.get('n', 0)})")

    # Encoding report
    print("\n[8] Encoding pipeline stats")
    enc_report = netflix.transcoder.encoding_report()
    for k, v in enc_report.items():
        print(f"    {k:<20}: {v}")

    # CDN stats
    print("\n[9] CDN cache statistics")
    cdn_stats = netflix.cdn.cache_stats()
    for k, v in cdn_stats.items():
        print(f"    {k:<15}: {v}")

    print("\n" + "=" * 65)
    print("  Simulation Complete")
    print("=" * 65)


if __name__ == "__main__":
    run_simulation()
