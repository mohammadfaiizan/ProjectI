"""
BLOCK vs OBJECT vs FILE STORAGE
==================================

Problem Statement:
Storage is not one-size-fits-all. Different workloads need different storage abstractions.
Choosing wrong leads to poor performance, high cost, or operational pain.

Three Fundamental Storage Types:

  1. Block Storage:
     Raw blocks of data at fixed size (512B–4KB sectors).
     No concept of files or metadata — just addresses and data.
     Accessed via protocols: iSCSI, Fibre Channel, NVMe-oF.
     OS formats it with a filesystem (ext4, NTFS, XFS).
     Examples: AWS EBS, Google Persistent Disk, SAN.
     Use for: databases (PostgreSQL, MySQL), VMs (boot volumes), high-IOPS apps.
     Pros: lowest latency, random read/write, full OS control.
     Cons: single instance attachment (usually), no built-in sharing.

  2. File Storage (NAS / Network File System):
     Hierarchical directory tree. Files + metadata (owner, permissions, timestamps).
     Accessed via: NFS, SMB/CIFS, POSIX.
     Can be shared across multiple servers simultaneously.
     Examples: AWS EFS, Azure Files, GCP Filestore, NFS server.
     Use for: shared config, home directories, CMS content, ML training data.
     Pros: familiar filesystem semantics, multi-client access, POSIX compliant.
     Cons: higher latency than block, scalability limits at extreme sizes.

  3. Object Storage (Blob Storage):
     Flat namespace. Objects = data + metadata + unique key.
     No hierarchy (prefix simulation for directory-like UX).
     Accessed via HTTP/S REST API (GET/PUT/DELETE).
     Examples: AWS S3, Google Cloud Storage, Azure Blob, MinIO.
     Use for: images, videos, backups, logs, static assets, data lake.
     Pros: massive scale, HTTP access, cheap, versioning, lifecycle policies.
     Cons: no random access (whole object), eventual consistency (historically).
     No file-lock semantics. Higher latency than block.

Decision Matrix:
  Database (OLTP/OLAP)?         → Block storage (EBS, Persistent Disk)
  Shared file access (NFS)?     → File storage (EFS, Azure Files)
  Media/images/backups/blobs?   → Object storage (S3, GCS)
  VMs / boot volumes?           → Block storage
  Static website assets?        → Object storage (+ CDN)
  Logs archive, data lake?      → Object storage
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time
import hashlib
import uuid


# ─────────────────────────────────────────────
# STORAGE TYPES
# ─────────────────────────────────────────────

class StorageType(Enum):
    BLOCK  = "block"
    FILE   = "file"
    OBJECT = "object"


# ─────────────────────────────────────────────
# BLOCK STORAGE SIMULATOR
# ─────────────────────────────────────────────

class BlockStorage:
    """
    Simulates block storage: fixed-size blocks, random access by block number.
    Like EBS or a raw disk partition.
    """
    BLOCK_SIZE = 4096   # 4 KB blocks

    def __init__(self, size_blocks: int = 1000):
        self._blocks   : Dict[int, bytes] = {}
        self._capacity = size_blocks
        self.reads     = 0
        self.writes    = 0

    def write_block(self, block_num: int, data: bytes) -> bool:
        if block_num >= self._capacity:
            return False
        # Pad or truncate to block size
        data = (data + b'\x00' * self.BLOCK_SIZE)[:self.BLOCK_SIZE]
        self._blocks[block_num] = data
        self.writes += 1
        return True

    def read_block(self, block_num: int) -> Optional[bytes]:
        self.reads += 1
        return self._blocks.get(block_num, b'\x00' * self.BLOCK_SIZE)

    def write_range(self, start_block: int, data: bytes) -> int:
        """Write multi-block data. Returns number of blocks written."""
        n_blocks = (len(data) + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        for i in range(n_blocks):
            chunk = data[i * self.BLOCK_SIZE: (i + 1) * self.BLOCK_SIZE]
            self.write_block(start_block + i, chunk)
        return n_blocks

    @property
    def iops_ratio(self) -> float:
        total = self.reads + self.writes
        return self.writes / total if total else 0.0


# ─────────────────────────────────────────────
# FILE STORAGE SIMULATOR (NFS-like)
# ─────────────────────────────────────────────

@dataclass
class FileMetadata:
    path       : str
    size       : int
    owner      : str
    permissions: str   # e.g. "rw-r--r--"
    created_at : float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)


class FileStorage:
    """
    Simulates NFS/POSIX file storage: hierarchical namespace, metadata, sharing.
    Multiple clients can mount and read/write simultaneously.
    """

    def __init__(self):
        self._files    : Dict[str, bytes] = {}
        self._metadata : Dict[str, FileMetadata] = {}
        self._locks    : Dict[str, str] = {}   # path → client_id holding advisory lock
        self.operations = 0

    def write(self, path: str, data: bytes, owner: str = "root",
              permissions: str = "rw-r--r--") -> bool:
        self._files[path] = data
        self._metadata[path] = FileMetadata(
            path=path, size=len(data), owner=owner, permissions=permissions)
        self.operations += 1
        return True

    def read(self, path: str) -> Optional[bytes]:
        self.operations += 1
        return self._files.get(path)

    def list_dir(self, prefix: str) -> List[str]:
        """List files with given path prefix (directory simulation)."""
        return [p for p in self._files if p.startswith(prefix)]

    def get_metadata(self, path: str) -> Optional[FileMetadata]:
        return self._metadata.get(path)

    def lock(self, path: str, client_id: str) -> bool:
        """Advisory lock — multiple clients must cooperate."""
        if path in self._locks:
            return False
        self._locks[path] = client_id
        return True

    def unlock(self, path: str, client_id: str) -> bool:
        if self._locks.get(path) == client_id:
            del self._locks[path]
            return True
        return False

    def rename(self, src: str, dst: str) -> bool:
        if src not in self._files:
            return False
        self._files[dst]    = self._files.pop(src)
        self._metadata[dst] = self._metadata.pop(src)
        self._metadata[dst].path = dst
        return True


# ─────────────────────────────────────────────
# OBJECT STORAGE SIMULATOR (S3-like)
# ─────────────────────────────────────────────

@dataclass
class S3Object:
    bucket    : str
    key       : str
    data      : bytes
    metadata  : Dict[str, str] = field(default_factory=dict)
    etag      : str = ""
    version_id: str = ""
    size      : int = 0
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        self.etag       = hashlib.md5(self.data).hexdigest()
        self.size       = len(self.data)
        self.version_id = str(uuid.uuid4())[:8]


class ObjectStorage:
    """
    S3-like object storage: flat namespace, REST semantics, versioning.
    Key insight: entire object read/written atomically (no random access).
    """

    def __init__(self):
        self._buckets  : Dict[str, Dict[str, List[S3Object]]] = {}   # bucket→key→versions
        self.puts      = 0
        self.gets      = 0
        self.deletes   = 0

    def create_bucket(self, bucket: str):
        self._buckets[bucket] = {}

    def put(self, bucket: str, key: str, data: bytes,
            metadata: Dict = None) -> S3Object:
        obj = S3Object(bucket=bucket, key=key, data=data, metadata=metadata or {})
        if bucket not in self._buckets:
            self._buckets[bucket] = {}
        if key not in self._buckets[bucket]:
            self._buckets[bucket][key] = []
        self._buckets[bucket][key].append(obj)
        self.puts += 1
        return obj

    def get(self, bucket: str, key: str,
            version_id: str = None) -> Optional[S3Object]:
        self.gets += 1
        versions = self._buckets.get(bucket, {}).get(key, [])
        if not versions:
            return None
        if version_id:
            return next((v for v in versions if v.version_id == version_id), None)
        return versions[-1]   # latest version

    def delete(self, bucket: str, key: str) -> bool:
        versions = self._buckets.get(bucket, {}).get(key)
        if versions:
            del self._buckets[bucket][key]
            self.deletes += 1
            return True
        return False

    def list_objects(self, bucket: str, prefix: str = "") -> List[str]:
        return [k for k in self._buckets.get(bucket, {}) if k.startswith(prefix)]

    def get_versions(self, bucket: str, key: str) -> List[S3Object]:
        return self._buckets.get(bucket, {}).get(key, [])

    def total_size_bytes(self, bucket: str) -> int:
        return sum(v[-1].size for v in self._buckets.get(bucket, {}).values() if v)


# ─────────────────────────────────────────────
# STORAGE ADVISOR
# ─────────────────────────────────────────────

def recommend_storage(workload: Dict[str, Any]) -> Tuple[StorageType, str]:
    """Simple rule-based recommender."""
    if workload.get("random_access") and workload.get("low_latency"):
        return StorageType.BLOCK, "Block storage — low-latency random I/O"
    if workload.get("shared_access") and workload.get("posix_needed"):
        return StorageType.FILE, "File storage — POSIX + multi-client sharing"
    if workload.get("large_objects") or workload.get("http_access"):
        return StorageType.OBJECT, "Object storage — HTTP, scale, cheap"
    if workload.get("database"):
        return StorageType.BLOCK, "Block storage — databases need random I/O"
    return StorageType.OBJECT, "Object storage — default for blobs"


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_storage_types():
    print("=" * 65)
    print("BLOCK vs OBJECT vs FILE STORAGE")
    print("=" * 65)

    # ── Block Storage ─────────────────────────────
    print("\n[1] BLOCK STORAGE — RANDOM ACCESS BY BLOCK NUMBER")
    print("─" * 55)
    block = BlockStorage(size_blocks=100)
    n = block.write_range(0, b"Database page data: user records and indexes" * 50)
    r = block.read_block(2)
    print(f"  Wrote {n} blocks of 4KB each")
    print(f"  Read block 2: {r[:20]}...")
    print(f"  I/O ops: reads={block.reads} writes={block.writes}")
    print(f"  Suitable for: PostgreSQL, MySQL, VM boot volumes")

    # ── File Storage ──────────────────────────────
    print("\n\n[2] FILE STORAGE — POSIX HIERARCHICAL NAMESPACE")
    print("─" * 55)
    fs = FileStorage()
    fs.write("/ml-data/training/batch_1.csv", b"feature1,feature2,label\n1,2,0\n3,4,1",
             owner="ml-user")
    fs.write("/ml-data/training/batch_2.csv", b"feature1,feature2,label\n5,6,1\n7,8,0",
             owner="ml-user")
    fs.write("/ml-data/config/model.yaml", b"epochs: 10\nbatch_size: 32")

    files = fs.list_dir("/ml-data/training/")
    print(f"  Training files: {files}")
    meta = fs.get_metadata("/ml-data/training/batch_1.csv")
    print(f"  Metadata: size={meta.size}B owner={meta.owner} perms={meta.permissions}")

    # Multi-client lock
    locked_by_a = fs.lock("/ml-data/config/model.yaml", "worker-A")
    locked_by_b = fs.lock("/ml-data/config/model.yaml", "worker-B")
    print(f"  Worker-A lock: {locked_by_a}  Worker-B lock (conflicts): {locked_by_b}")
    fs.unlock("/ml-data/config/model.yaml", "worker-A")

    # ── Object Storage ────────────────────────────
    print("\n\n[3] OBJECT STORAGE — FLAT NAMESPACE, REST, VERSIONING")
    print("─" * 55)
    s3 = ObjectStorage()
    s3.create_bucket("user-uploads")
    s3.create_bucket("backups")

    # Upload objects
    obj1 = s3.put("user-uploads", "avatars/user-42.jpg", b"\xff\xd8" + b"JPEG_DATA" * 100,
                  {"Content-Type": "image/jpeg"})
    obj2 = s3.put("backups", "db/2024-01-15.sql.gz", b"GZIP" + b"\x00" * 500)
    print(f"  Uploaded avatar: etag={obj1.etag} size={obj1.size}B version={obj1.version_id}")

    # Versioning: update object
    obj1_v2 = s3.put("user-uploads", "avatars/user-42.jpg", b"\xff\xd8" + b"NEW_JPEG" * 80,
                     {"Content-Type": "image/jpeg"})
    versions = s3.get_versions("user-uploads", "avatars/user-42.jpg")
    print(f"  Avatar has {len(versions)} versions: v1={versions[0].version_id} v2={versions[1].version_id}")

    # List with prefix
    keys = s3.list_objects("user-uploads", prefix="avatars/")
    print(f"  Objects with prefix 'avatars/': {keys}")
    print(f"  Total bucket size: {s3.total_size_bytes('user-uploads')} bytes")

    # ── Storage Advisor ───────────────────────────
    print("\n\n[4] STORAGE TYPE RECOMMENDATION ENGINE")
    print("─" * 55)
    workloads = [
        {"database": True, "random_access": True, "low_latency": True},
        {"shared_access": True, "posix_needed": True},
        {"large_objects": True, "http_access": True},
        {"http_access": True},
        {"random_access": True, "low_latency": True},
    ]
    for wl in workloads:
        storage_type, reason = recommend_storage(wl)
        print(f"  Workload {str(wl)[:40]:<42} → {reason}")

    # ── Comparison Table ──────────────────────────
    print("\n\n[5] STORAGE TYPE COMPARISON")
    print("─" * 55)
    rows = [
        ("Access pattern",  "Random R/W by block",    "POSIX file/dir",        "GET/PUT by key"),
        ("Latency",         "Sub-ms",                 "1-10ms",                "10-100ms"),
        ("Sharing",         "Single instance",        "Multi-client (NFS)",    "HTTP — unlimited"),
        ("Scalability",     "Volume-bound",           "Petabytes (EFS scale)", "Exabytes (S3)"),
        ("Cost",            "Expensive/GB",           "Medium/GB",             "Cheapest/GB"),
        ("Best for",        "DB, VMs, high IOPS",     "Shared files, ML data", "Media, backups, logs"),
        ("AWS example",     "EBS",                    "EFS",                   "S3"),
    ]
    print(f"  {'Aspect':<18} {'Block':<26} {'File':<26} {'Object'}")
    print(f"  {'─'*82}")
    for aspect, block, file, obj in rows:
        print(f"  {aspect:<18} {block:<26} {file:<26} {obj}")


if __name__ == "__main__":
    demonstrate_storage_types()
