"""
BLOB STORAGE FOR MEDIA (CDN-BACKED)
======================================

Problem Statement:
Serving images, videos, and audio files at scale requires:
  - Storing large binary objects (blobs) durably.
  - Delivering with low latency globally (CDN).
  - Transforming on-demand (resize, transcode, compress).
  - Handling upload pipelines (client → origin → CDN).

Architecture:
  Upload:  Client → API Server → Blob Store (S3/GCS).
           Or: Client → Presigned URL → Blob Store directly.
  Serve:   Client → CDN Edge → (cache hit) → served immediately.
                             → (cache miss) → Origin → Blob Store.
  Transform: Lazy resize on first request (or pre-generate thumbnails).

Media Metadata:
  Content-Type, dimensions, duration, codec, bitrate.
  Stored in a database (not in the blob store itself).

Image Variants:
  Original → thumbnail (100px), small (400px), medium (800px), large (1600px).
  Generated on upload or on-demand (lazy). Stored as separate objects.
  URL pattern: /images/{id}/w{width}.jpg (imgproxy / Cloudflare Images style).

Video Processing Pipeline:
  Upload raw → transcode to HLS (multiple bitrates) → store segments.
  HLS: .m3u8 playlist + .ts segments. Adaptive bitrate (ABR).

CDN Integration:
  Push CDN: upload to CDN origin server, CDN pulls from there.
  Pull CDN: CDN fetches from origin on first miss, caches at edge.
  Cache-Control headers control CDN TTL.
  Invalidation: purge by URL or cache tag (surrogate keys).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import time
import uuid
import math


# ─────────────────────────────────────────────
# MEDIA TYPE + DIMENSIONS
# ─────────────────────────────────────────────

class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class ImageDimensions:
    width : int
    height: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0

    def resize_to_width(self, target_width: int) -> "ImageDimensions":
        scale  = target_width / self.width
        return ImageDimensions(target_width, int(self.height * scale))


@dataclass
class MediaMetadata:
    media_id      : str
    filename      : str
    media_type    : MediaType
    content_type  : str
    size_bytes    : int
    dimensions    : Optional[ImageDimensions] = None
    duration_s    : Optional[float]           = None   # video/audio
    codec         : Optional[str]             = None
    bitrate_kbps  : Optional[int]             = None
    created_at    : float = field(default_factory=time.time)


# ─────────────────────────────────────────────
# BLOB STORE (origin)
# ─────────────────────────────────────────────

class BlobStore:
    """Simulates origin object storage (S3-like)."""

    def __init__(self, bucket: str):
        self.bucket     = bucket
        self._blobs     : Dict[str, bytes] = {}
        self._metadata  : Dict[str, Dict]  = {}

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        etag = hashlib.md5(data).hexdigest()
        self._blobs[key]    = data
        self._metadata[key] = {"content_type": content_type, "etag": etag,
                                "size": len(data)}
        return etag

    def get(self, key: str) -> Optional[bytes]:
        return self._blobs.get(key)

    def exists(self, key: str) -> bool:
        return key in self._blobs

    def delete(self, key: str) -> bool:
        if key in self._blobs:
            del self._blobs[key]
            del self._metadata[key]
            return True
        return False

    def list_prefix(self, prefix: str) -> List[str]:
        return [k for k in self._blobs if k.startswith(prefix)]


# ─────────────────────────────────────────────
# IMAGE VARIANT GENERATOR
# ─────────────────────────────────────────────

IMAGE_VARIANTS = {
    "thumbnail" : 100,
    "small"     : 400,
    "medium"    : 800,
    "large"     : 1600,
}


class ImageVariantService:
    """
    Generates and caches image variants at different sizes.
    In production: uses ImageMagick, Sharp, or imgproxy.
    """

    def __init__(self, blob_store: BlobStore):
        self._store = blob_store
        self.transforms_done = 0

    def _simulate_resize(self, original: bytes, original_dims: ImageDimensions,
                         target_width: int) -> Tuple[bytes, ImageDimensions]:
        """Simulate resize: in reality calls Sharp/ImageMagick."""
        new_dims     = original_dims.resize_to_width(target_width)
        # Simulate smaller file (proportional to pixel count reduction)
        pixel_ratio  = (new_dims.width * new_dims.height) / \
                       (original_dims.width * original_dims.height)
        simulated_size = max(1000, int(len(original) * pixel_ratio * 0.7))  # JPEG compression
        resized_data = original[:simulated_size] + b"RESIZED"
        return resized_data, new_dims

    def ensure_variant(self, media_id: str, original_key: str,
                       variant_name: str, original_dims: ImageDimensions) -> str:
        """Generate variant if not already cached. Returns variant key."""
        target_width = IMAGE_VARIANTS[variant_name]
        variant_key  = f"variants/{media_id}/{variant_name}.jpg"

        if self._store.exists(variant_key):
            return variant_key   # already generated

        original = self._store.get(original_key)
        if not original:
            raise FileNotFoundError(f"Original {original_key} not found")

        if original_dims.width <= target_width:
            # No upscaling — use original
            self._store.put(variant_key, original, "image/jpeg")
        else:
            resized, _ = self._simulate_resize(original, original_dims, target_width)
            self._store.put(variant_key, resized, "image/jpeg")
            self.transforms_done += 1

        return variant_key

    def generate_all_variants(self, media_id: str, original_key: str,
                               original_dims: ImageDimensions) -> Dict[str, str]:
        return {
            name: self.ensure_variant(media_id, original_key, name, original_dims)
            for name in IMAGE_VARIANTS
        }


# ─────────────────────────────────────────────
# CDN LAYER
# ─────────────────────────────────────────────

@dataclass
class CacheEntry:
    key        : str
    data       : bytes
    content_type: str
    cached_at  : float
    ttl_s      : int
    hits       : int = 0

    def is_fresh(self) -> bool:
        return (time.time() - self.cached_at) < self.ttl_s


class CDNEdge:
    """
    Simulates a CDN edge node: serves from cache or fetches from origin.
    """

    def __init__(self, edge_id: str, origin: BlobStore, default_ttl_s: int = 86400):
        self.edge_id     = edge_id
        self._origin     = origin
        self._cache      : Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl_s
        self.cache_hits  = 0
        self.cache_misses= 0

    def serve(self, key: str, ttl_s: int = None) -> Tuple[Optional[bytes], bool]:
        """Returns (data, cache_hit)."""
        entry = self._cache.get(key)
        if entry and entry.is_fresh():
            entry.hits += 1
            self.cache_hits += 1
            return entry.data, True

        # Cache miss → fetch from origin
        data = self._origin.get(key)
        if data is None:
            self.cache_misses += 1
            return None, False

        self._cache[key] = CacheEntry(
            key=key, data=data, content_type="application/octet-stream",
            cached_at=time.time(), ttl_s=ttl_s or self.default_ttl
        )
        self.cache_misses += 1
        return data, False

    def invalidate(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def invalidate_prefix(self, prefix: str) -> int:
        to_remove = [k for k in self._cache if k.startswith(prefix)]
        for k in to_remove:
            del self._cache[k]
        return len(to_remove)

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def cache_stats(self) -> Dict:
        return {
            "cached_objects": len(self._cache),
            "cache_hits"    : self.cache_hits,
            "cache_misses"  : self.cache_misses,
            "hit_rate"      : self.hit_rate,
        }


# ─────────────────────────────────────────────
# HLS VIDEO PIPELINE (simplified)
# ─────────────────────────────────────────────

@dataclass
class HLSSegment:
    index     : int
    duration_s: float
    bitrate   : str      # "360p", "720p", "1080p"
    data      : bytes


class HLSTranscoder:
    """
    Simulates HLS (HTTP Live Streaming) transcoding pipeline.
    Real systems use FFmpeg + AWS Elemental MediaConvert.
    Produces: master.m3u8 (playlist of playlists) + per-bitrate .m3u8 + .ts segments.
    """

    BITRATES = {
        "360p"  : (640,  360,  800),   # (width, height, kbps)
        "720p"  : (1280, 720,  3000),
        "1080p" : (1920, 1080, 6000),
    }
    SEGMENT_DURATION = 6.0   # seconds

    def transcode(self, media_id: str, raw_data: bytes,
                  duration_s: float, store: BlobStore) -> str:
        """Transcode raw video to HLS. Returns master playlist key."""
        n_segments = math.ceil(duration_s / self.SEGMENT_DURATION)
        playlists  = {}

        for bitrate_name, (w, h, kbps) in self.BITRATES.items():
            segment_keys = []
            for i in range(n_segments):
                seg_duration = min(self.SEGMENT_DURATION,
                                   duration_s - i * self.SEGMENT_DURATION)
                # Simulate segment data (proportional to bitrate × duration)
                seg_size = int(kbps * 1024 / 8 * seg_duration)
                seg_data = raw_data[:seg_size] + f"SEG{i}_{bitrate_name}".encode()
                key = f"hls/{media_id}/{bitrate_name}/seg{i:04d}.ts"
                store.put(key, seg_data, "video/MP2T")
                segment_keys.append((key, seg_duration))

            # Per-bitrate playlist
            m3u8  = "#EXTM3U\n#EXT-X-VERSION:3\n"
            m3u8 += f"#EXT-X-TARGETDURATION:{int(self.SEGMENT_DURATION)}\n"
            for key, dur in segment_keys:
                m3u8 += f"#EXTINF:{dur:.1f},\n{key}\n"
            m3u8 += "#EXT-X-ENDLIST\n"
            playlist_key = f"hls/{media_id}/{bitrate_name}/playlist.m3u8"
            store.put(playlist_key, m3u8.encode(), "application/x-mpegURL")
            playlists[bitrate_name] = (playlist_key, kbps, w, h)

        # Master playlist
        master  = "#EXTM3U\n"
        for bitrate_name, (playlist_key, kbps, w, h) in playlists.items():
            master += f'#EXT-X-STREAM-INF:BANDWIDTH={kbps*1000},RESOLUTION={w}x{h}\n'
            master += f"{playlist_key}\n"
        master_key = f"hls/{media_id}/master.m3u8"
        store.put(master_key, master.encode(), "application/x-mpegURL")
        return master_key


# ─────────────────────────────────────────────
# MEDIA UPLOAD SERVICE
# ─────────────────────────────────────────────

class MediaUploadService:
    def __init__(self, blob_store: BlobStore, cdn: CDNEdge):
        self._store    = blob_store
        self._cdn      = cdn
        self._metadata : Dict[str, MediaMetadata] = {}
        self._variants  = ImageVariantService(blob_store)
        self._transcoder= HLSTranscoder()

    def upload_image(self, filename: str, data: bytes,
                     dimensions: ImageDimensions) -> MediaMetadata:
        media_id = str(uuid.uuid4())[:8]
        key      = f"originals/{media_id}/{filename}"
        self._store.put(key, data, "image/jpeg")

        meta = MediaMetadata(
            media_id=media_id, filename=filename, media_type=MediaType.IMAGE,
            content_type="image/jpeg", size_bytes=len(data), dimensions=dimensions,
        )
        self._metadata[media_id] = meta

        # Pre-generate all variants
        self._variants.generate_all_variants(media_id, key, dimensions)
        return meta

    def upload_video(self, filename: str, data: bytes,
                     duration_s: float) -> MediaMetadata:
        media_id = str(uuid.uuid4())[:8]
        key      = f"originals/{media_id}/{filename}"
        self._store.put(key, data, "video/mp4")

        meta = MediaMetadata(
            media_id=media_id, filename=filename, media_type=MediaType.VIDEO,
            content_type="video/mp4", size_bytes=len(data), duration_s=duration_s,
        )
        self._metadata[media_id] = meta

        # Transcode to HLS
        self._transcoder.transcode(media_id, data, duration_s, self._store)
        return meta

    def get_image_url(self, media_id: str, variant: str = "medium") -> str:
        meta = self._metadata.get(media_id)
        if not meta:
            return ""
        return f"https://cdn.example.com/variants/{media_id}/{variant}.jpg"

    def get_video_url(self, media_id: str) -> str:
        return f"https://cdn.example.com/hls/{media_id}/master.m3u8"


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_blob_storage():
    print("=" * 65)
    print("BLOB STORAGE FOR MEDIA (CDN-BACKED)")
    print("=" * 65)

    store   = BlobStore("media-origin")
    cdn     = CDNEdge("edge-us-east", store, default_ttl_s=86400)
    service = MediaUploadService(store, cdn)

    # ── Image Upload + Variants ───────────────────
    print("\n[1] IMAGE UPLOAD — MULTI-VARIANT GENERATION")
    print("─" * 55)

    img_data = b"\xff\xd8" + b"JPEG_PIXEL_DATA" * 500   # ~7.5KB simulated
    img_dims = ImageDimensions(2400, 1800)
    meta     = service.upload_image("photo.jpg", img_data, img_dims)

    print(f"  Uploaded: {meta.media_id} — {meta.size_bytes}B {meta.dimensions.width}x{meta.dimensions.height}")
    print(f"  Variants generated:")
    for name, width in IMAGE_VARIANTS.items():
        key  = f"variants/{meta.media_id}/{name}.jpg"
        blob = store.get(key)
        print(f"    {name:<12} ({width}px): {len(blob)}B stored")
    print(f"  Transforms done: {service._variants.transforms_done}")

    # ── CDN Serving ───────────────────────────────
    print("\n\n[2] CDN EDGE — CACHE HIT vs MISS")
    print("─" * 55)

    key = f"variants/{meta.media_id}/medium.jpg"
    data1, hit1 = cdn.serve(key)   # cold miss
    data2, hit2 = cdn.serve(key)   # warm hit
    data3, hit3 = cdn.serve(key)   # warm hit
    print(f"  Request 1 (cold): hit={hit1}, {len(data1)}B fetched from origin")
    print(f"  Request 2 (warm): hit={hit2}")
    print(f"  Request 3 (warm): hit={hit3}")
    stats = cdn.cache_stats()
    print(f"  CDN stats: hits={stats['cache_hits']} misses={stats['cache_misses']} "
          f"hit_rate={stats['hit_rate']:.0%}")

    # ── Cache Invalidation ────────────────────────
    print("\n\n[3] CACHE INVALIDATION")
    print("─" * 55)

    removed = cdn.invalidate_prefix(f"variants/{meta.media_id}/")
    print(f"  Invalidated {removed} cached variant(s) for media_id={meta.media_id}")
    _, hit4 = cdn.serve(key)
    print(f"  Re-request after invalidation: hit={hit4} (expected False)")

    # ── Video HLS Transcoding ─────────────────────
    print("\n\n[4] VIDEO HLS TRANSCODING PIPELINE")
    print("─" * 55)

    video_data = b"RAW_VIDEO_BYTES" * 1000
    vmeta      = service.upload_video("tour.mp4", video_data, duration_s=30.0)
    hls_keys   = store.list_prefix(f"hls/{vmeta.media_id}/")
    ts_keys    = [k for k in hls_keys if k.endswith(".ts")]
    m3u8_keys  = [k for k in hls_keys if k.endswith(".m3u8")]

    print(f"  Uploaded video: {vmeta.media_id} ({vmeta.duration_s}s)")
    print(f"  HLS output: {len(m3u8_keys)} playlists, {len(ts_keys)} segments")
    print(f"  Bitrates: {sorted(set(k.split('/')[3] for k in ts_keys))}")
    master = store.get(f"hls/{vmeta.media_id}/master.m3u8")
    print(f"  Master playlist preview:\n    {'    '.join(master.decode().splitlines()[:6])}")
    print(f"  Video URL: {service.get_video_url(vmeta.media_id)}")

    # ── URL Generation ────────────────────────────
    print("\n\n[5] IMAGE URL VARIANTS")
    print("─" * 55)
    for variant in IMAGE_VARIANTS:
        url = service.get_image_url(meta.media_id, variant)
        print(f"  {variant:<12}: {url}")

    # ── Architecture Summary ──────────────────────
    print("\n\n[6] MEDIA STORAGE DESIGN DECISIONS")
    print("─" * 55)
    decisions = [
        ("Blob store",          "S3/GCS as origin — durable, cheap, HTTP accessible"),
        ("CDN edge",            "Serve from edge — low latency globally, offload origin"),
        ("Presigned upload",    "Client uploads directly to blob store — bypasses server"),
        ("Variant generation",  "Pre-generate on upload — avoids on-demand latency spikes"),
        ("HLS for video",       "Adaptive bitrate streaming — adjusts to network speed"),
        ("Cache-Control TTL",   "Long TTL for immutable content (e.g. max-age=31536000)"),
        ("Surrogate-Key header","Tag-based CDN invalidation — purge by media_id"),
        ("Content-addressed key","key = hash(content) → deduplication + immutability"),
    ]
    for decision, reason in decisions:
        print(f"  {decision:<26} {reason}")


if __name__ == "__main__":
    demonstrate_blob_storage()
