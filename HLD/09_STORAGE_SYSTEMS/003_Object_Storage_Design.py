"""
OBJECT STORAGE DESIGN (S3 INTERNALS)
======================================

Problem Statement:
How does AWS S3 reliably store trillions of objects across thousands of servers,
handle millions of requests per second, and guarantee 11-nines durability?

S3 Architecture:
  Front-End Layer:  Load balancers → API handlers (PUT/GET/DELETE/LIST).
  Metadata Store:   Maps bucket+key → object location + metadata.
                    Often a distributed KV store (DynamoDB-like).
  Storage Layer:    Actual object bytes on disk servers (chunk servers).
  Index/Namespace:  Bucket namespace is global; key prefix routing for LIST.

Multipart Upload:
  Large objects split into parts (5MB–5GB each).
  Each part uploaded independently (parallelism, resumability).
  Server assembles on CompleteMultipartUpload.
  AbortMultipartUpload cleans up partial parts.
  Best for: objects > 100MB.

Presigned URLs:
  Time-limited URL signed with secret key.
  Allows untrusted clients to upload/download directly (bypasses backend).
  Signature = HMAC(secret, canonical_request + expiry).
  S3 validates signature and expiry on receipt.

Storage Classes:
  Standard          : Frequent access. 3+ AZ replication. Highest cost.
  Standard-IA       : Infrequent access. Cheaper storage, retrieval fee.
  One Zone-IA       : Single AZ. Cheaper, lower durability.
  Glacier Instant   : Millisecond retrieval. Cold but fast.
  Glacier Flexible  : Minutes–hours retrieval.
  Glacier Deep      : 12 hours retrieval. Cheapest.
  Intelligent-Tier  : Auto-moves between tiers based on access patterns.

Lifecycle Policies:
  Rules transition objects between storage classes by age.
  Rule: after 30 days → IA; after 90 days → Glacier; after 365 days → delete.

Consistency Model (post-2020):
  S3 now provides strong read-after-write consistency for all operations.
  Previously eventual consistency for LIST after PUT.
  Achieved via: metadata version tracking in consistent KV layer.

Durability:
  11-nines (99.999999999%) achieved by:
  - Cross-AZ replication (3+ copies).
  - Erasure coding in some layers.
  - Continuous integrity checks (background CRC verification).
  - Versioning (protect against accidental deletes).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple
from enum import Enum
import hashlib
import hmac
import time
import uuid
import math


# ─────────────────────────────────────────────
# STORAGE CLASS + LIFECYCLE
# ─────────────────────────────────────────────

class StorageClass(Enum):
    STANDARD          = "STANDARD"
    STANDARD_IA       = "STANDARD_IA"
    ONEZONE_IA        = "ONEZONE_IA"
    GLACIER_INSTANT   = "GLACIER_INSTANT"
    GLACIER_FLEXIBLE  = "GLACIER_FLEXIBLE"
    GLACIER_DEEP      = "GLACIER_DEEP"
    INTELLIGENT_TIERING = "INTELLIGENT_TIERING"


STORAGE_COST_PER_GB_MONTH = {
    StorageClass.STANDARD         : 0.023,
    StorageClass.STANDARD_IA      : 0.0125,
    StorageClass.ONEZONE_IA       : 0.010,
    StorageClass.GLACIER_INSTANT  : 0.004,
    StorageClass.GLACIER_FLEXIBLE : 0.0036,
    StorageClass.GLACIER_DEEP     : 0.00099,
    StorageClass.INTELLIGENT_TIERING: 0.023,
}

RETRIEVAL_LATENCY_MS = {
    StorageClass.STANDARD         : 1,
    StorageClass.STANDARD_IA      : 2,
    StorageClass.ONEZONE_IA       : 2,
    StorageClass.GLACIER_INSTANT  : 5,
    StorageClass.GLACIER_FLEXIBLE : 3_600_000,    # ~1 hour
    StorageClass.GLACIER_DEEP     : 43_200_000,   # ~12 hours
    StorageClass.INTELLIGENT_TIERING: 1,
}


@dataclass
class LifecycleRule:
    rule_id        : str
    prefix         : str                    # applies to keys with this prefix
    transition_days: Dict[int, StorageClass] # {days_old: target_class}
    expiry_days    : Optional[int] = None   # delete after N days


# ─────────────────────────────────────────────
# OBJECT METADATA
# ─────────────────────────────────────────────

@dataclass
class ObjectMetadata:
    bucket        : str
    key           : str
    size          : int
    etag          : str
    version_id    : str
    storage_class : StorageClass
    created_at    : float
    last_accessed : float
    user_metadata : Dict[str, str] = field(default_factory=dict)
    content_type  : str = "application/octet-stream"
    parts         : int = 1            # 1 = single-part upload


# ─────────────────────────────────────────────
# MULTIPART UPLOAD
# ─────────────────────────────────────────────

@dataclass
class UploadPart:
    part_number : int
    data        : bytes
    etag        : str = ""

    def __post_init__(self):
        self.etag = hashlib.md5(self.data).hexdigest()


class MultipartUpload:
    """
    Simulates S3 multipart upload: initiate → upload parts → complete.
    Min part size: 5MB (except last). Max parts: 10,000.
    """
    MIN_PART_SIZE = 5 * 1024 * 1024   # 5 MB

    def __init__(self, upload_id: str, bucket: str, key: str):
        self.upload_id  = upload_id
        self.bucket     = bucket
        self.key        = key
        self._parts     : Dict[int, UploadPart] = {}
        self.initiated_at = time.time()
        self.aborted    = False

    def upload_part(self, part_number: int, data: bytes) -> str:
        if self.aborted:
            raise RuntimeError("Upload was aborted")
        if part_number < 1 or part_number > 10_000:
            raise ValueError(f"Part number must be 1-10000, got {part_number}")
        part = UploadPart(part_number=part_number, data=data)
        self._parts[part_number] = part
        return part.etag

    def complete(self, part_etags: List[Tuple[int, str]]) -> Tuple[bytes, str]:
        """Validates etags and assembles final object."""
        if self.aborted:
            raise RuntimeError("Upload was aborted")
        # Validate
        for part_num, etag in part_etags:
            if part_num not in self._parts:
                raise ValueError(f"Part {part_num} not uploaded")
            if self._parts[part_num].etag != etag:
                raise ValueError(f"ETag mismatch for part {part_num}")
        # Assemble in order
        sorted_parts = sorted(part_etags, key=lambda x: x[0])
        assembled    = b"".join(self._parts[pn].data for pn, _ in sorted_parts)
        # Multi-part ETag: md5 of concatenated part md5s + "-N"
        part_md5s   = b"".join(bytes.fromhex(self._parts[pn].etag) for pn, _ in sorted_parts)
        final_etag  = f"{hashlib.md5(part_md5s).hexdigest()}-{len(sorted_parts)}"
        return assembled, final_etag

    def abort(self):
        self.aborted = True
        self._parts.clear()

    def list_parts(self) -> List[UploadPart]:
        return sorted(self._parts.values(), key=lambda p: p.part_number)


# ─────────────────────────────────────────────
# PRESIGNED URL
# ─────────────────────────────────────────────

class PresignedURLService:
    """
    Signs and validates time-limited object URLs.
    Real S3 uses SigV4; here we use HMAC-SHA256 for illustration.
    """

    def __init__(self, secret_key: str):
        self._secret = secret_key.encode()

    def generate(self, bucket: str, key: str, operation: str,
                 expires_in_s: int = 3600) -> str:
        expires_at = int(time.time()) + expires_in_s
        canonical  = f"{operation}:{bucket}/{key}:{expires_at}"
        sig        = hmac.new(self._secret, canonical.encode(), hashlib.sha256).hexdigest()
        return f"https://s3.example.com/{bucket}/{key}?X-Expires={expires_at}&X-Signature={sig}&X-Op={operation}"

    def validate(self, url: str) -> Tuple[bool, str]:
        """Returns (valid, reason)."""
        try:
            path  = url.split("?")[0]
            parts = path.split("/")
            bucket, key = parts[-2], parts[-1]
            params = dict(p.split("=") for p in url.split("?")[1].split("&"))
            expires_at = int(params["X-Expires"])
            sig        = params["X-Signature"]
            operation  = params["X-Op"]

            if time.time() > expires_at:
                return False, "URL expired"

            canonical    = f"{operation}:{bucket}/{key}:{expires_at}"
            expected_sig = hmac.new(self._secret, canonical.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected_sig):
                return False, "Invalid signature"

            return True, f"Valid {operation} URL for {bucket}/{key}"
        except Exception as e:
            return False, f"Malformed URL: {e}"


# ─────────────────────────────────────────────
# OBJECT STORAGE ENGINE
# ─────────────────────────────────────────────

class ObjectStorageEngine:
    """
    S3-like object storage with versioning, lifecycle, and storage classes.
    """

    def __init__(self):
        self._buckets       : Dict[str, Dict] = {}        # bucket config
        self._objects       : Dict[str, Dict[str, List[ObjectMetadata]]] = {}  # bucket→key→versions
        self._data_store    : Dict[str, bytes] = {}       # version_id → data
        self._lifecycle     : Dict[str, List[LifecycleRule]] = {}
        self._uploads       : Dict[str, MultipartUpload] = {}   # in-flight multipart
        self.puts = self.gets = self.deletes = 0

    def create_bucket(self, bucket: str, versioning: bool = False,
                      region: str = "us-east-1"):
        self._buckets[bucket] = {"versioning": versioning, "region": region}
        self._objects[bucket] = {}

    def put_object(self, bucket: str, key: str, data: bytes,
                   storage_class: StorageClass = StorageClass.STANDARD,
                   metadata: Dict = None) -> ObjectMetadata:
        if bucket not in self._buckets:
            raise ValueError(f"Bucket {bucket!r} does not exist")
        version_id = str(uuid.uuid4())[:8]
        etag       = hashlib.md5(data).hexdigest()
        obj = ObjectMetadata(
            bucket=bucket, key=key, size=len(data), etag=etag,
            version_id=version_id, storage_class=storage_class,
            created_at=time.time(), last_accessed=time.time(),
            user_metadata=metadata or {},
        )
        if key not in self._objects[bucket]:
            self._objects[bucket][key] = []
        self._objects[bucket][key].append(obj)
        self._data_store[version_id] = data
        self.puts += 1
        return obj

    def get_object(self, bucket: str, key: str,
                   version_id: str = None) -> Tuple[Optional[ObjectMetadata], Optional[bytes]]:
        versions = self._objects.get(bucket, {}).get(key, [])
        if not versions:
            return None, None
        if version_id:
            obj = next((v for v in versions if v.version_id == version_id), None)
        else:
            obj = versions[-1]
        if obj is None:
            return None, None
        obj.last_accessed = time.time()
        data = self._data_store.get(obj.version_id)
        self.gets += 1
        return obj, data

    def delete_object(self, bucket: str, key: str,
                      version_id: str = None) -> bool:
        versions = self._objects.get(bucket, {}).get(key)
        if not versions:
            return False
        if version_id:
            to_remove = [v for v in versions if v.version_id == version_id]
            if not to_remove:
                return False
            for v in to_remove:
                versions.remove(v)
                self._data_store.pop(v.version_id, None)
        else:
            # Delete all (or latest if versioning on)
            if self._buckets[bucket]["versioning"]:
                # add delete marker (simplified: just remove latest)
                removed = versions.pop()
                self._data_store.pop(removed.version_id, None)
            else:
                for v in versions:
                    self._data_store.pop(v.version_id, None)
                self._objects[bucket][key] = []
        self.deletes += 1
        return True

    def list_objects(self, bucket: str, prefix: str = "",
                     delimiter: str = "/") -> Dict:
        """Returns objects + common prefixes (S3 LIST behavior)."""
        all_keys = list(self._objects.get(bucket, {}).keys())
        matching = [k for k in all_keys if k.startswith(prefix)
                    and self._objects[bucket][k]]
        objects          = []
        common_prefixes  = set()
        for key in matching:
            suffix = key[len(prefix):]
            if delimiter and delimiter in suffix:
                # Folder-like prefix
                folder = prefix + suffix[:suffix.index(delimiter) + 1]
                common_prefixes.add(folder)
            else:
                obj = self._objects[bucket][key][-1]
                objects.append({"key": key, "size": obj.size,
                                "storage_class": obj.storage_class.value})
        return {"objects": objects, "common_prefixes": sorted(common_prefixes)}

    def set_lifecycle(self, bucket: str, rules: List[LifecycleRule]):
        self._lifecycle[bucket] = rules

    def apply_lifecycle(self, bucket: str, simulate_days: int):
        """Transition/expire objects based on lifecycle rules."""
        rules    = self._lifecycle.get(bucket, [])
        actions  = []
        for key, versions in self._objects.get(bucket, {}).items():
            if not versions:
                continue
            obj       = versions[-1]
            age_days  = (time.time() - obj.created_at) / 86400 + simulate_days
            for rule in rules:
                if not key.startswith(rule.prefix):
                    continue
                if rule.expiry_days and age_days >= rule.expiry_days:
                    actions.append(("DELETE", key, None))
                    break
                # Find applicable transition
                for days_threshold in sorted(rule.transition_days.keys(), reverse=True):
                    if age_days >= days_threshold:
                        target = rule.transition_days[days_threshold]
                        if obj.storage_class != target:
                            obj.storage_class = target
                            actions.append(("TRANSITION", key, target.value))
                        break
        return actions

    # ── Multipart Upload ──────────────────────────

    def create_multipart_upload(self, bucket: str, key: str) -> str:
        upload_id = str(uuid.uuid4())[:12]
        self._uploads[upload_id] = MultipartUpload(upload_id, bucket, key)
        return upload_id

    def upload_part(self, upload_id: str, part_number: int, data: bytes) -> str:
        upload = self._uploads.get(upload_id)
        if not upload:
            raise ValueError("Unknown upload_id")
        return upload.upload_part(part_number, data)

    def complete_multipart_upload(self, upload_id: str,
                                  part_etags: List[Tuple[int, str]]) -> ObjectMetadata:
        upload = self._uploads.pop(upload_id, None)
        if not upload:
            raise ValueError("Unknown upload_id")
        assembled, etag = upload.complete(part_etags)
        obj = self.put_object(upload.bucket, upload.key, assembled)
        obj.etag  = etag
        obj.parts = len(part_etags)
        return obj

    def abort_multipart_upload(self, upload_id: str):
        upload = self._uploads.pop(upload_id, None)
        if upload:
            upload.abort()


# ─────────────────────────────────────────────
# COST CALCULATOR
# ─────────────────────────────────────────────

def estimate_monthly_cost(objects: List[ObjectMetadata]) -> Dict[StorageClass, float]:
    cost_by_class: Dict[StorageClass, float] = {}
    for obj in objects:
        size_gb = obj.size / (1024 ** 3)
        cost    = size_gb * STORAGE_COST_PER_GB_MONTH[obj.storage_class]
        cost_by_class[obj.storage_class] = cost_by_class.get(obj.storage_class, 0) + cost
    return cost_by_class


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_object_storage():
    print("=" * 65)
    print("OBJECT STORAGE DESIGN (S3 INTERNALS)")
    print("=" * 65)

    engine = ObjectStorageEngine()
    engine.create_bucket("media", versioning=True)
    engine.create_bucket("backups")

    # ── Basic PUT / GET ───────────────────────────
    print("\n[1] PUT / GET / LIST")
    print("─" * 55)

    objs = [
        ("media", "images/cat.jpg",  b"\xff\xd8" + b"JPEG" * 200,  StorageClass.STANDARD),
        ("media", "videos/tour.mp4", b"ftyp" + b"\x00" * 1000,     StorageClass.STANDARD_IA),
        ("media", "images/dog.jpg",  b"\xff\xd8" + b"JPEG" * 150,  StorageClass.STANDARD),
        ("backups", "db/2024-01.sql.gz", b"GZIP" + b"\x00" * 500,  StorageClass.GLACIER_FLEXIBLE),
    ]
    stored = []
    for bucket, key, data, sc in objs:
        m = engine.put_object(bucket, key, data, sc)
        stored.append(m)
        print(f"  PUT s3://{bucket}/{key}: {m.size}B etag={m.etag[:8]} class={m.storage_class.value}")

    listing = engine.list_objects("media", prefix="images/")
    print(f"\n  LIST media/images/: {[o['key'] for o in listing['objects']]}")

    listing2 = engine.list_objects("media", prefix="", delimiter="/")
    print(f"  LIST media/ (delimiter='/'): prefixes={listing2['common_prefixes']}")

    # ── Versioning ────────────────────────────────
    print("\n\n[2] VERSIONING — MULTIPLE VERSIONS OF SAME KEY")
    print("─" * 55)

    v1 = engine.put_object("media", "images/cat.jpg", b"VERSION_1" * 50)
    v2 = engine.put_object("media", "images/cat.jpg", b"VERSION_2" * 60)
    versions = engine._objects["media"]["images/cat.jpg"]
    print(f"  Key 'images/cat.jpg' has {len(versions)} versions:")
    for v in versions:
        print(f"    v={v.version_id} size={v.size}B etag={v.etag[:8]}")

    meta, data = engine.get_object("media", "images/cat.jpg", version_id=v1.version_id)
    print(f"  Read specific version {v1.version_id}: data={data[:9]}")

    # ── Multipart Upload ──────────────────────────
    print("\n\n[3] MULTIPART UPLOAD — LARGE OBJECT ASSEMBLY")
    print("─" * 55)

    upload_id  = engine.create_multipart_upload("backups", "db/large-dump.sql")
    part_etags = []
    parts_data = [b"PART_" + bytes([i]) * (6 * 1024 * 1024) for i in range(3)]
    for i, part_data in enumerate(parts_data, start=1):
        etag = engine.upload_part(upload_id, i, part_data)
        part_etags.append((i, etag))
        print(f"  Uploaded part {i}: {len(part_data)//1024//1024}MB etag={etag[:8]}")

    final = engine.complete_multipart_upload(upload_id, part_etags)
    print(f"  Assembled object: {final.size}B etag={final.etag[:20]} parts={final.parts}")

    # ── Presigned URLs ────────────────────────────
    print("\n\n[4] PRESIGNED URLs — TEMPORARY DIRECT ACCESS")
    print("─" * 55)

    svc   = PresignedURLService(secret_key="my-secret-key-2024")
    url   = svc.generate("media", "images/cat.jpg", "GET", expires_in_s=300)
    valid, reason = svc.validate(url)
    print(f"  GET URL: ...{url[-60:]}")
    print(f"  Valid: {valid} — {reason}")

    expired_url = svc.generate("media", "images/cat.jpg", "GET", expires_in_s=-1)
    valid2, reason2 = svc.validate(expired_url)
    print(f"  Expired URL valid: {valid2} — {reason2}")

    # ── Lifecycle Policies ────────────────────────
    print("\n\n[5] LIFECYCLE POLICIES — AUTOMATIC TIERING")
    print("─" * 55)

    engine.set_lifecycle("backups", [
        LifecycleRule(
            rule_id="archive-policy",
            prefix="db/",
            transition_days={
                30:  StorageClass.STANDARD_IA,
                90:  StorageClass.GLACIER_FLEXIBLE,
                365: StorageClass.GLACIER_DEEP,
            },
            expiry_days=2555,   # 7 years
        )
    ])

    actions = engine.apply_lifecycle("backups", simulate_days=95)
    print(f"  Simulating 95 days old:")
    for action, key, target in actions:
        print(f"    {action} {key} → {target or 'deleted'}")

    # ── Storage Class Cost Comparison ────────────
    print("\n\n[6] STORAGE CLASS COST COMPARISON (1 TB/month)")
    print("─" * 55)

    size_gb = 1024
    print(f"  {'Storage Class':<24} {'$/GB/month':>12} {'1TB/month ($)':>14} {'Retrieval latency'}")
    print(f"  {'─'*68}")
    for sc in StorageClass:
        if sc == StorageClass.INTELLIGENT_TIERING:
            continue
        cost        = STORAGE_COST_PER_GB_MONTH[sc]
        monthly     = cost * size_gb
        latency     = RETRIEVAL_LATENCY_MS[sc]
        lat_str     = f"{latency}ms" if latency < 1000 else f"{latency//3600000}h"
        print(f"  {sc.value:<24} ${cost:>10.5f}  ${monthly:>12.2f}  {lat_str}")


if __name__ == "__main__":
    demonstrate_object_storage()
