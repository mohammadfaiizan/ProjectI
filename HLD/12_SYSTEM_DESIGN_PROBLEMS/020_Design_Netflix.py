"""
NETFLIX — Video Streaming Platform
=====================================

FUNCTIONAL REQUIREMENTS:
- Browse content catalog (movies, TV shows, documentaries)
- Stream video at adaptive quality (240p → 4K)
- Continue watching across devices
- Personalised recommendations
- Offline downloads
- Multi-user profiles per account

NON-FUNCTIONAL REQUIREMENTS:
- 250 M subscribers, 100 M concurrent streams at peak
- 15% of global internet traffic
- Stream start time < 2 s (p99)
- Rebuffering ratio < 0.5%
- Content catalog: 15,000+ titles

ARCHITECTURE:
  Client ──HTTPS──▶ Open Connect CDN (edge)
                          │
                   ┌──────▼───────┐
                   │ Control Plane │ (API, auth, content metadata)
                   │  (AWS)       │
                   └──────┬───────┘
         ┌─────────────────┼──────────────────┐
         ▼                 ▼                  ▼
    Content Svc     User Profile Svc    Recommendation Svc
    (DynamoDB)       (Cassandra)          (Spark ML)

KEY DESIGN DECISIONS:
1. CDN — Netflix's Open Connect Appliances (OCAs) deployed in ISP datacenters.
   OCAs cache popular content; 95%+ of traffic served from edge, not origin.
   Cache miss → pull from origin S3.  Weekly popularity score determines
   what gets pushed to which OCA cluster.

2. ADAPTIVE BITRATE (ABR) — content pre-encoded at multiple bitrates/resolutions
   using HEVC/H.264.  HLS or MPEG-DASH manifest lists available segments.
   Player monitors bandwidth → switches quality every 2-second segment.
   Buffer target: 30 seconds ahead.

3. ENCODING PIPELINE:
   Raw upload → scene detection → per-scene quality optimisation (VMAF score)
   → encode at 5-8 bitrate ladders × 3 codecs (H.264, HEVC, AV1)
   → upload to S3 + distribute to OCAs.
   Netflix's per-title encoding saves ~20% bandwidth vs fixed-bitrate ladder.

4. RECOMMENDATIONS — offline: collaborative filtering (SVD) on watch history.
   Online: contextual bandits for personalised row ordering.
   Two-stage: candidate generation (ANN on embeddings) → ranking (DNN).

5. WATCH HISTORY & PROGRESS — Cassandra: partition=user_id, clustering=updated_at DESC.
   Bookmarks (playback position) written every 10 seconds.

6. A/B TESTING — 40+ concurrent experiments (artwork, recommendation algorithms,
   encoding parameters).  Metric: engagement, stream starts, completion rate.

7. CHAOS ENGINEERING — Chaos Monkey, Chaos Gorilla (AZ failure), Latency Monkey.
   Every service must tolerate dependency failure gracefully.
"""

from __future__ import annotations
import time
import uuid
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
import random


# ---------------------------------------------------------------------------
# Content Catalog
# ---------------------------------------------------------------------------

class ContentType(Enum):
    MOVIE = "movie"
    SERIES = "series"
    DOCUMENTARY = "documentary"
    SPECIAL = "special"


class VideoQuality(Enum):
    SD_240 = ("240p", 250_000)       # 250 Kbps
    SD_360 = ("360p", 500_000)
    SD_480 = ("480p", 1_000_000)
    HD_720 = ("720p", 3_000_000)
    HD_1080 = ("1080p", 5_000_000)
    UHD_4K = ("4K", 16_000_000)

    def __init__(self, label: str, bitrate_bps: int):
        self.label = label
        self.bitrate_bps = bitrate_bps

    @property
    def bitrate_mbps(self) -> float:
        return self.bitrate_bps / 1_000_000


@dataclass
class Episode:
    episode_id: str
    series_id: str
    season: int
    episode_number: int
    title: str
    duration_seconds: int
    synopsis: str = ""


@dataclass
class Content:
    content_id: str
    title: str
    content_type: ContentType
    genres: List[str]
    year: int
    rating: str          # "PG", "TV-MA", etc.
    imdb_score: float
    duration_seconds: int   # 0 for series
    synopsis: str = ""
    cast: List[str] = field(default_factory=list)
    director: str = ""
    languages: List[str] = field(default_factory=list)
    available_qualities: List[VideoQuality] = field(default_factory=list)
    episodes: List[Episode] = field(default_factory=list)
    popularity_score: float = 0.0   # used for CDN pre-positioning
    created_at: float = field(default_factory=time.time)

    @property
    def duration_display(self) -> str:
        m = self.duration_seconds // 60
        return f"{m // 60}h {m % 60}m" if m >= 60 else f"{m}m"


# ---------------------------------------------------------------------------
# Encoding Pipeline
# ---------------------------------------------------------------------------

@dataclass
class EncodedVariant:
    content_id: str
    quality: VideoQuality
    codec: str           # "h264" | "hevc" | "av1"
    segment_urls: List[str]    # S3 URLs (simulated)
    vmaf_score: float    # Video Multi-Method Assessment Fusion quality score
    file_size_bytes: int


class EncodingPipeline:
    """Simulates Netflix's per-title encoding optimisation."""

    CODECS = ["h264", "hevc", "av1"]

    def encode(self, content_id: str, duration_s: int,
               qualities: List[VideoQuality] = None) -> List[EncodedVariant]:
        if qualities is None:
            qualities = [VideoQuality.SD_480, VideoQuality.HD_720,
                         VideoQuality.HD_1080, VideoQuality.UHD_4K]

        variants = []
        for quality in qualities:
            for codec in self.CODECS:
                # Per-title optimisation: higher complexity content needs more bits
                complexity_factor = random.uniform(0.8, 1.2)
                # AV1 is ~50% more efficient than H.264
                codec_efficiency = {"h264": 1.0, "hevc": 0.7, "av1": 0.5}[codec]
                effective_bitrate = int(quality.bitrate_bps * complexity_factor * codec_efficiency)

                # VMAF score: higher bitrate → better quality
                vmaf = min(99.0, 60 + 15 * math.log10(effective_bitrate / 100_000))

                file_size = effective_bitrate * duration_s // 8

                # Generate segment URLs (2-second segments)
                num_segments = math.ceil(duration_s / 2)
                segments = [f"s3://netflix-content/{content_id}/{codec}/{quality.label}/seg_{i:04d}.ts"
                            for i in range(num_segments)]

                variants.append(EncodedVariant(
                    content_id=content_id,
                    quality=quality,
                    codec=codec,
                    segment_urls=segments,
                    vmaf_score=round(vmaf, 1),
                    file_size_bytes=file_size,
                ))
        return variants

    def generate_manifest(self, variants: List[EncodedVariant], codec: str = "hevc") -> str:
        """Generate HLS master manifest."""
        filtered = [v for v in variants if v.codec == codec]
        filtered.sort(key=lambda v: v.quality.bitrate_bps)

        lines = ["#EXTM3U", "#EXT-X-VERSION:6"]
        for v in filtered:
            lines.append(f"#EXT-X-STREAM-INF:BANDWIDTH={v.quality.bitrate_bps},"
                         f"RESOLUTION={v.quality.label},CODECS=\"{codec}\"")
            lines.append(f"/{v.content_id}/{codec}/{v.quality.label}/playlist.m3u8")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CDN (Open Connect Appliances)
# ---------------------------------------------------------------------------

@dataclass
class OCANode:
    node_id: str
    isp: str
    region: str
    capacity_gb: int
    cached_content: Dict[str, float] = field(default_factory=dict)  # content_id → popularity

    def cache_hit(self, content_id: str) -> bool:
        return content_id in self.cached_content

    def evict_lru(self, new_content_id: str, popularity: float):
        """Evict lowest-popularity content to make room."""
        if len(self.cached_content) >= 100:  # capacity limit simulation
            min_key = min(self.cached_content, key=self.cached_content.get)
            del self.cached_content[min_key]
        self.cached_content[new_content_id] = popularity


class CDNService:
    def __init__(self):
        self._nodes: Dict[str, OCANode] = {}
        self._origin_hits = 0
        self._edge_hits = 0

    def add_node(self, node: OCANode):
        self._nodes[node.node_id] = node

    def preposition(self, content_id: str, popularity: float, regions: List[str]):
        """Push popular content to edge nodes in specified regions."""
        for node in self._nodes.values():
            if node.region in regions:
                node.evict_lru(content_id, popularity)

    def request(self, content_id: str, user_region: str,
                user_isp: str) -> Tuple[str, str]:
        """Returns (url, source) where source is 'edge' or 'origin'."""
        # Find best OCA: same ISP and region first, then same region
        best = None
        for node in self._nodes.values():
            if node.cache_hit(content_id):
                if node.isp == user_isp and node.region == user_region:
                    best = node
                    break
                elif node.region == user_region and not best:
                    best = node

        if best:
            self._edge_hits += 1
            return f"https://oca-{best.node_id}.netflix.net/{content_id}", "edge"
        else:
            self._origin_hits += 1
            return f"https://origin.netflix.com/{content_id}", "origin"

    @property
    def cache_hit_ratio(self) -> float:
        total = self._edge_hits + self._origin_hits
        return self._edge_hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Adaptive Bitrate Player (Client-side simulation)
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    content_id: str
    user_id: str
    current_quality: VideoQuality
    buffer_seconds: float = 0.0
    playback_position_s: float = 0.0
    stalls: int = 0
    quality_switches: int = 0


class ABRPlayer:
    """
    Adaptive Bitrate algorithm: buffer-based with bandwidth estimation.
    Buffer-Based ABR (BBA): choose quality based on buffer level.
    """

    BUFFER_TARGET = 30.0     # seconds
    BUFFER_MIN = 5.0         # stall if below

    def __init__(self, available_qualities: List[VideoQuality]):
        self._qualities = sorted(available_qualities, key=lambda q: q.bitrate_bps)

    def choose_quality(self, buffer_s: float, bandwidth_bps: float) -> VideoQuality:
        """Choose highest quality that won't drain the buffer."""
        best = self._qualities[0]  # lowest quality is always safe
        for q in self._qualities:
            # Will this quality drain the buffer?
            # Each 2-second segment needs q.bitrate_bps * 2 bits
            download_time = (q.bitrate_bps * 2) / bandwidth_bps
            if download_time <= 2.0:  # Can download faster than playback
                best = q
        return best

    def simulate_stream(self, duration_s: int, bandwidth_profile: List[float]) -> PlayerState:
        """
        Simulate a streaming session.
        bandwidth_profile: list of bandwidth measurements per 2s segment.
        """
        state = PlayerState(
            content_id="demo",
            user_id="user",
            current_quality=self._qualities[0],
        )

        for seg_idx in range(math.ceil(duration_s / 2)):
            # Get current bandwidth (cycle through profile)
            bw = bandwidth_profile[seg_idx % len(bandwidth_profile)]

            # Choose quality
            new_quality = self.choose_quality(state.buffer_seconds, bw)
            if new_quality != state.current_quality:
                state.quality_switches += 1
                state.current_quality = new_quality

            # Simulate download
            download_time = (new_quality.bitrate_bps * 2) / bw
            state.buffer_seconds = max(0, state.buffer_seconds + 2.0 - download_time)

            if state.buffer_seconds < self.BUFFER_MIN:
                state.stalls += 1
                state.buffer_seconds = 0

            state.buffer_seconds = min(state.buffer_seconds, self.BUFFER_TARGET)
            state.playback_position_s = min((seg_idx + 1) * 2, duration_s)

        return state


# ---------------------------------------------------------------------------
# User Profile & Watch History
# ---------------------------------------------------------------------------

@dataclass
class WatchRecord:
    content_id: str
    episode_id: Optional[str]
    position_s: float        # current playback position
    duration_s: int
    last_watched: float
    completed: bool = False

    @property
    def progress_pct(self) -> int:
        return int(100 * self.position_s / self.duration_s) if self.duration_s else 0


@dataclass
class UserProfile:
    profile_id: str
    account_id: str
    name: str
    avatar: str = ""
    kids_mode: bool = False
    language: str = "en"
    watch_history: Dict[str, WatchRecord] = field(default_factory=dict)


class WatchHistoryService:
    def __init__(self):
        self._profiles: Dict[str, UserProfile] = {}

    def create_profile(self, account_id: str, name: str, kids_mode: bool = False) -> UserProfile:
        p = UserProfile(
            profile_id=str(uuid.uuid4())[:8],
            account_id=account_id,
            name=name,
            kids_mode=kids_mode,
        )
        self._profiles[p.profile_id] = p
        return p

    def update_position(self, profile_id: str, content_id: str,
                        position_s: float, duration_s: int,
                        episode_id: Optional[str] = None) -> WatchRecord:
        profile = self._profiles[profile_id]
        key = episode_id or content_id
        completed = position_s >= duration_s * 0.9  # 90% = completed
        record = WatchRecord(
            content_id=content_id,
            episode_id=episode_id,
            position_s=position_s,
            duration_s=duration_s,
            last_watched=time.time(),
            completed=completed,
        )
        profile.watch_history[key] = record
        return record

    def continue_watching(self, profile_id: str, limit: int = 10) -> List[WatchRecord]:
        profile = self._profiles.get(profile_id)
        if not profile:
            return []
        incomplete = [r for r in profile.watch_history.values()
                      if not r.completed and r.progress_pct > 5]
        return sorted(incomplete, key=lambda r: r.last_watched, reverse=True)[:limit]

    def get_profile(self, profile_id: str) -> Optional[UserProfile]:
        return self._profiles.get(profile_id)


# ---------------------------------------------------------------------------
# Recommendation Service
# ---------------------------------------------------------------------------

class RecommendationService:
    """
    Two-stage recommendation: candidate generation → ranking.
    Simulated with genre-based collaborative filtering.
    """

    def __init__(self, catalog_items: List[Content]):
        self._catalog = {c.content_id: c for c in catalog_items}
        # In production: user embeddings from SVD on watch matrix
        self._user_genre_affinity: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def record_watch(self, profile_id: str, content: Content, completion_pct: float):
        """Update user's genre affinity based on what they watched."""
        weight = min(1.0, completion_pct / 100)
        for genre in content.genres:
            self._user_genre_affinity[profile_id][genre] += weight

    def recommend(self, profile_id: str, watch_history: Dict[str, WatchRecord],
                  limit: int = 10) -> List[Content]:
        affinities = self._user_genre_affinity.get(profile_id, {})
        watched_ids = {r.content_id for r in watch_history.values()}

        scored = []
        for content in self._catalog.values():
            if content.content_id in watched_ids:
                continue  # Skip already watched
            # Score = sum of affinity for each genre
            genre_score = sum(affinities.get(g, 0) for g in content.genres)
            # Boost by popularity and IMDB score
            score = genre_score * 0.6 + content.imdb_score * 0.3 + content.popularity_score * 0.1
            scored.append((content, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:limit]]


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demonstrate_1_content_catalog_encoding():
    print("\n=== 1. Content Catalog & Encoding Pipeline ===")
    pipeline = EncodingPipeline()

    movie = Content(
        content_id="tt1234567",
        title="The Grand Adventure",
        content_type=ContentType.MOVIE,
        genres=["Action", "Adventure"],
        year=2024,
        rating="PG-13",
        imdb_score=7.8,
        duration_seconds=7200,  # 2 hours
        popularity_score=85.0,
        available_qualities=[VideoQuality.SD_480, VideoQuality.HD_720,
                             VideoQuality.HD_1080, VideoQuality.UHD_4K],
    )

    variants = pipeline.encode(movie.content_id, movie.duration_seconds)
    manifest = pipeline.generate_manifest(variants, codec="hevc")

    total_size_gb = sum(v.file_size_bytes for v in variants) / 1e9
    print(f"Title: {movie.title} ({movie.duration_display})")
    print(f"Encoded variants: {len(variants)} ({len(pipeline.CODECS)} codecs × {len(variants)//len(pipeline.CODECS)} qualities)")
    print(f"Total storage: {total_size_gb:.1f} GB")

    # Show VMAF scores
    hevc_variants = [v for v in variants if v.codec == "hevc"]
    hevc_variants.sort(key=lambda v: v.quality.bitrate_bps)
    print(f"\nHEVC quality ladder:")
    for v in hevc_variants:
        print(f"  {v.quality.label}: {v.quality.bitrate_mbps:.1f} Mbps, "
              f"VMAF={v.vmaf_score}, size={v.file_size_bytes/1e9:.2f} GB")

    print(f"\nHLS Manifest (first 4 lines):")
    for line in manifest.split("\n")[:5]:
        print(f"  {line}")

    return movie


def demonstrate_2_cdn_and_edge():
    print("\n=== 2. CDN & Open Connect Edge Caching ===")
    cdn = CDNService()

    # Add OCA nodes
    cdn.add_node(OCANode("oca-us-west-1", "Comcast", "us-west", 100_000))
    cdn.add_node(OCANode("oca-us-west-2", "AT&T", "us-west", 100_000))
    cdn.add_node(OCANode("oca-us-east-1", "Verizon", "us-east", 100_000))

    # Pre-position popular content
    cdn.preposition("tt1234567", popularity=85.0, regions=["us-west", "us-east"])
    cdn.preposition("tt9999999", popularity=30.0, regions=["us-east"])

    # Requests
    test_cases = [
        ("tt1234567", "us-west", "Comcast"),   # Should hit edge (same ISP)
        ("tt1234567", "us-west", "AT&T"),      # Edge (same region)
        ("tt9999999", "us-west", "Comcast"),   # Not pre-positioned to us-west
        ("tt0000000", "us-east", "Verizon"),   # Not in CDN at all
    ]

    for content_id, region, isp in test_cases:
        url, source = cdn.request(content_id, region, isp)
        print(f"  [{source.upper():6}] {content_id} for {isp}/{region}: {url[:60]}...")

    print(f"\nOverall cache hit ratio: {cdn.cache_hit_ratio:.1%}")


def demonstrate_3_adaptive_bitrate():
    print("\n=== 3. Adaptive Bitrate Streaming ===")
    qualities = [VideoQuality.SD_480, VideoQuality.HD_720,
                 VideoQuality.HD_1080, VideoQuality.UHD_4K]
    player = ABRPlayer(qualities)

    # Bandwidth profile: good network → drops → recovers
    bandwidth_profile = [
        20_000_000,  # 20 Mbps (4K)
        20_000_000,
        5_000_000,   # 5 Mbps (1080p)
        1_000_000,   # 1 Mbps (poor — stall risk)
        500_000,     # Very poor
        15_000_000,  # Recovers
        15_000_000,
    ]

    state = player.simulate_stream(duration_s=90, bandwidth_profile=bandwidth_profile)
    print(f"Stream simulation (90s movie clip):")
    print(f"  Final quality: {state.current_quality.label}")
    print(f"  Quality switches: {state.quality_switches}")
    print(f"  Stall events: {state.stalls}")
    print(f"  Buffer level at end: {state.buffer_seconds:.1f}s")
    print(f"  Playback position: {state.playback_position_s:.0f}s")


def demonstrate_4_watch_history_and_continue():
    print("\n=== 4. Watch History & Continue Watching ===")
    svc = WatchHistoryService()

    profile = svc.create_profile("acc_001", "Alice")

    # Alice watches several shows
    svc.update_position(profile.profile_id, "tt1234567", 3600, 7200)  # 50% of movie
    svc.update_position(profile.profile_id, "tt1111111", 6840, 7200)  # 95% — completed
    svc.update_position(profile.profile_id, "tt2222222", 1200, 3600)  # 33% of episode

    continue_watching = svc.continue_watching(profile.profile_id)
    print(f"Profile: {profile.name}")
    print(f"Continue watching ({len(continue_watching)} items):")
    for r in continue_watching:
        print(f"  {r.content_id}: {r.progress_pct}% complete "
              f"(at {r.position_s:.0f}s / {r.duration_s}s)")


def demonstrate_5_recommendations():
    print("\n=== 5. Personalised Recommendations ===")
    catalog = [
        Content("c001", "Inception", ContentType.MOVIE, ["Sci-Fi", "Thriller"],
                2010, "PG-13", 8.8, 8880, popularity_score=95),
        Content("c002", "The Dark Knight", ContentType.MOVIE, ["Action", "Thriller"],
                2008, "PG-13", 9.0, 9120, popularity_score=98),
        Content("c003", "Interstellar", ContentType.MOVIE, ["Sci-Fi", "Drama"],
                2014, "PG", 8.6, 9720, popularity_score=90),
        Content("c004", "The Office", ContentType.SERIES, ["Comedy"],
                2005, "TV-14", 9.0, 0, popularity_score=85),
        Content("c005", "Planet Earth", ContentType.DOCUMENTARY, ["Nature"],
                2006, "G", 9.4, 0, popularity_score=70),
    ]

    rec_svc = RecommendationService(catalog)
    svc = WatchHistoryService()
    profile = svc.create_profile("acc_002", "Bob")

    # Bob watches Inception completely
    c001 = catalog[0]
    rec_svc.record_watch(profile.profile_id, c001, 100)
    svc.update_position(profile.profile_id, c001.content_id, c001.duration_seconds,
                        c001.duration_seconds)

    # Recommend based on preferences
    recommendations = rec_svc.recommend(
        profile.profile_id,
        svc.get_profile(profile.profile_id).watch_history
    )

    print(f"Bob watched: Inception (Sci-Fi, Thriller)")
    print(f"\nRecommendations:")
    for c in recommendations:
        print(f"  {c.title} ({', '.join(c.genres)}) — ★{c.imdb_score}")


if __name__ == "__main__":
    demonstrate_1_content_catalog_encoding()
    demonstrate_2_cdn_and_edge()
    demonstrate_3_adaptive_bitrate()
    demonstrate_4_watch_history_and_continue()
    demonstrate_5_recommendations()
