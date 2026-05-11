"""
Design Dropbox — Python Simulation
====================================
Simulates core Dropbox mechanics:
  - File chunking with SHA-256 content-addressable storage
  - Cross-user chunk deduplication
  - Delta sync (only changed chunks transferred)
  - File versioning (up to 30 versions)
  - Conflict resolution (last-write-wins + conflict copy)
  - Sharing (public links + per-user permissions)
  - Offline sync queue with replay
"""

import hashlib
import uuid
import time
import json
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from datetime import datetime


CHUNK_SIZE = 4 * 1024 * 1024   # 4 MB in bytes (simulated with smaller chunks)
MAX_VERSIONS = 30


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_hash: str          # SHA-256 hex digest
    s3_key: str              # Storage location key
    size_bytes: int


@dataclass
class FileVersion:
    version_id: str
    file_id: str
    version_num: int
    chunk_hashes: list       # Ordered list of chunk hashes
    size_bytes: int
    created_at: float
    created_by: str
    is_current: bool = False


@dataclass
class FileMetadata:
    file_id: str
    user_id: str
    name: str
    parent_folder: Optional[str]
    size_bytes: int
    is_deleted: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class Share:
    share_id: str
    resource_id: str
    resource_type: str       # 'file' or 'folder'
    owner_id: str
    shared_with: Optional[str]   # None = public
    permission: str          # 'view' or 'edit'
    public_token: Optional[str]
    expires_at: Optional[float]


@dataclass
class SyncEvent:
    event_id: str
    event_type: str          # 'upload', 'delete', 'rename'
    file_id: str
    version_id: str
    user_id: str
    timestamp: float


# ---------------------------------------------------------------------------
# File Chunker
# ---------------------------------------------------------------------------

class FileChunker:
    """Splits file content into fixed-size chunks and computes SHA-256 per chunk."""

    def __init__(self, chunk_size: int = 256):
        # Use 256 bytes as simulated chunk size for demo (real = 4MB)
        self.chunk_size = chunk_size

    def chunk_content(self, content: bytes) -> list[tuple[str, bytes]]:
        """Returns list of (chunk_hash, chunk_bytes) tuples."""
        chunks = []
        for start in range(0, len(content), self.chunk_size):
            chunk_bytes = content[start:start + self.chunk_size]
            chunk_hash = hashlib.sha256(chunk_bytes).hexdigest()
            chunks.append((chunk_hash, chunk_bytes))
        return chunks

    def compute_file_hash(self, content: bytes) -> str:
        """Compute overall file SHA-256 (for integrity checking)."""
        return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Deduplication Store
# ---------------------------------------------------------------------------

class DeduplicationStore:
    """
    Content-addressable storage: chunk_hash -> s3_key.
    Cross-user deduplication: same chunk stored exactly once.
    """

    def __init__(self):
        self._store: dict[str, Chunk] = {}          # hash -> Chunk
        self._ref_count: dict[str, int] = defaultdict(int)   # hash -> # references
        self.stats = {"hits": 0, "misses": 0, "bytes_saved": 0}

    def has_chunk(self, chunk_hash: str) -> bool:
        return chunk_hash in self._store

    def store_chunk(self, chunk_hash: str, chunk_bytes: bytes) -> str:
        """Store chunk if not already present. Returns s3_key."""
        if chunk_hash in self._store:
            self.stats["hits"] += 1
            self.stats["bytes_saved"] += len(chunk_bytes)
            self._ref_count[chunk_hash] += 1
            return self._store[chunk_hash].s3_key

        # New chunk — "upload" to object store
        s3_key = f"chunks/{chunk_hash[:4]}/{chunk_hash}"
        self._store[chunk_hash] = Chunk(
            chunk_hash=chunk_hash,
            s3_key=s3_key,
            size_bytes=len(chunk_bytes)
        )
        self._ref_count[chunk_hash] = 1
        self.stats["misses"] += 1
        return s3_key

    def release_chunk(self, chunk_hash: str):
        """Decrement reference count; GC chunk if count reaches 0."""
        if chunk_hash in self._ref_count:
            self._ref_count[chunk_hash] -= 1
            if self._ref_count[chunk_hash] <= 0:
                del self._store[chunk_hash]
                del self._ref_count[chunk_hash]

    def get_chunk_url(self, chunk_hash: str) -> Optional[str]:
        chunk = self._store.get(chunk_hash)
        return f"https://cdn.dropbox.com/{chunk.s3_key}" if chunk else None

    def dedup_report(self) -> dict:
        total = self.stats["hits"] + self.stats["misses"]
        ratio = self.stats["hits"] / total if total > 0 else 0
        return {
            "total_requests": total,
            "dedup_hits": self.stats["hits"],
            "dedup_ratio": f"{ratio:.1%}",
            "bytes_saved": self.stats["bytes_saved"]
        }


# ---------------------------------------------------------------------------
# Delta Sync Engine
# ---------------------------------------------------------------------------

class DeltaSync:
    """Computes the minimal set of chunks that need to be transferred."""

    @staticmethod
    def compute_delta(old_chunks: list[str], new_chunks: list[str]) -> dict:
        """
        Compare old and new chunk hash lists.
        Returns: chunks to upload, chunks to delete, and unchanged chunks.
        """
        old_set = set(old_chunks)
        new_set = set(new_chunks)

        to_upload = new_set - old_set       # Present in new, not in old
        to_delete = old_set - new_set       # Present in old, not in new
        unchanged = old_set & new_set       # Present in both

        bandwidth_saved = len(unchanged)
        total_new = len(new_chunks)
        savings_pct = (bandwidth_saved / total_new * 100) if total_new > 0 else 0

        return {
            "to_upload": list(to_upload),
            "to_delete": list(to_delete),
            "unchanged": list(unchanged),
            "chunks_saved": bandwidth_saved,
            "bandwidth_savings_pct": f"{savings_pct:.1f}%"
        }


# ---------------------------------------------------------------------------
# Metadata Store
# ---------------------------------------------------------------------------

class MetadataStore:
    """
    Stores file tree: files, versions, chunk mappings.
    In production: sharded PostgreSQL.
    """

    def __init__(self):
        self._files: dict[str, FileMetadata] = {}
        self._versions: dict[str, list[FileVersion]] = defaultdict(list)
        self._current_versions: dict[str, FileVersion] = {}
        self._shares: dict[str, Share] = {}
        self._events: list[SyncEvent] = []

    def create_file(self, user_id: str, name: str, parent_folder: Optional[str] = None) -> str:
        file_id = str(uuid.uuid4())
        self._files[file_id] = FileMetadata(
            file_id=file_id,
            user_id=user_id,
            name=name,
            parent_folder=parent_folder,
            size_bytes=0
        )
        return file_id

    def add_version(self, file_id: str, chunk_hashes: list[str],
                    size_bytes: int, created_by: str) -> FileVersion:
        versions = self._versions[file_id]
        version_num = len(versions) + 1

        # Mark previous current version as non-current
        if file_id in self._current_versions:
            self._current_versions[file_id].is_current = False

        new_version = FileVersion(
            version_id=str(uuid.uuid4()),
            file_id=file_id,
            version_num=version_num,
            chunk_hashes=chunk_hashes,
            size_bytes=size_bytes,
            created_at=time.time(),
            created_by=created_by,
            is_current=True
        )

        versions.append(new_version)
        self._current_versions[file_id] = new_version

        # Enforce 30-version limit
        if len(versions) > MAX_VERSIONS:
            versions.pop(0)

        # Update file metadata
        if file_id in self._files:
            self._files[file_id].size_bytes = size_bytes
            self._files[file_id].updated_at = time.time()

        # Record sync event
        event = SyncEvent(
            event_id=str(uuid.uuid4()),
            event_type="upload",
            file_id=file_id,
            version_id=new_version.version_id,
            user_id=created_by,
            timestamp=time.time()
        )
        self._events.append(event)

        return new_version

    def get_current_version(self, file_id: str) -> Optional[FileVersion]:
        return self._current_versions.get(file_id)

    def get_versions(self, file_id: str) -> list[FileVersion]:
        return self._versions.get(file_id, [])

    def get_file(self, file_id: str) -> Optional[FileMetadata]:
        return self._files.get(file_id)

    def get_changes_since(self, user_id: str, since_timestamp: float) -> list[SyncEvent]:
        user_file_ids = {fid for fid, f in self._files.items() if f.user_id == user_id}
        return [e for e in self._events
                if e.user_id == user_id or e.file_id in user_file_ids
                and e.timestamp > since_timestamp]

    def add_share(self, share: Share):
        self._shares[share.share_id] = share

    def get_share_by_token(self, token: str) -> Optional[Share]:
        return next((s for s in self._shares.values() if s.public_token == token), None)


# ---------------------------------------------------------------------------
# Conflict Resolver
# ---------------------------------------------------------------------------

class ConflictResolver:
    """
    Last-Write-Wins: the most recently synced version wins.
    The losing version is saved as a conflict copy.
    """

    def __init__(self, metadata_store: MetadataStore):
        self.metadata = metadata_store

    def resolve(self, file_id: str, incoming_chunks: list[str],
                incoming_size: int, uploader_id: str,
                client_base_version: int) -> dict:
        """
        Returns action: 'accepted' or 'conflict'.
        On conflict, creates a conflict copy file.
        """
        current = self.metadata.get_current_version(file_id)

        if current is None or current.version_num == client_base_version:
            # No conflict — linear history
            version = self.metadata.add_version(file_id, incoming_chunks, incoming_size, uploader_id)
            return {"action": "accepted", "version_id": version.version_id}

        # Conflict: server version advanced since client last synced
        file_meta = self.metadata.get_file(file_id)
        conflict_name = (
            f"{file_meta.name} (conflict copy {uploader_id} "
            f"{datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')})"
        )
        # Create conflict copy as new file
        conflict_file_id = self.metadata.create_file(
            file_meta.user_id, conflict_name, file_meta.parent_folder
        )
        conflict_version = self.metadata.add_version(
            conflict_file_id, incoming_chunks, incoming_size, uploader_id
        )

        return {
            "action": "conflict",
            "winner_version_id": current.version_id,
            "conflict_copy_file_id": conflict_file_id,
            "conflict_version_id": conflict_version.version_id
        }


# ---------------------------------------------------------------------------
# Offline Sync Queue
# ---------------------------------------------------------------------------

class OfflineSyncQueue:
    """Buffers file operations while offline; replays in order when reconnected."""

    def __init__(self):
        self._queue: list[dict] = []

    def enqueue(self, operation: dict):
        operation["queued_at"] = time.time()
        self._queue.append(operation)

    def replay(self, dropbox_system) -> list[dict]:
        results = []
        while self._queue:
            op = self._queue.pop(0)
            if op["type"] == "upload":
                result = dropbox_system.upload_file(
                    op["user_id"], op["filename"], op["content"], op["base_version"]
                )
                results.append({"op": op["filename"], "result": result})
        return results

    def pending_count(self) -> int:
        return len(self._queue)


# ---------------------------------------------------------------------------
# Main Dropbox System
# ---------------------------------------------------------------------------

class DropboxSystem:
    """Orchestrates all Dropbox components."""

    def __init__(self):
        self.chunker = FileChunker(chunk_size=64)    # 64 bytes for simulation
        self.dedup_store = DeduplicationStore()
        self.metadata = MetadataStore()
        self.delta_sync = DeltaSync()
        self.conflict_resolver = ConflictResolver(self.metadata)
        self.offline_queue = OfflineSyncQueue()

    def upload_file(self, user_id: str, filename: str, content: bytes,
                    base_version: int = 0, folder_id: Optional[str] = None) -> dict:
        """Full upload flow: chunk → dedup → delta → commit metadata."""

        # 1. Chunk the file
        chunks = self.chunker.chunk_content(content)
        new_chunk_hashes = [h for h, _ in chunks]

        # 2. Get current chunk list for delta computation
        file_id = self._find_file(user_id, filename, folder_id)
        old_chunk_hashes = []
        if file_id:
            current_ver = self.metadata.get_current_version(file_id)
            if current_ver:
                old_chunk_hashes = current_ver.chunk_hashes

        # 3. Compute delta
        delta = self.delta_sync.compute_delta(old_chunk_hashes, new_chunk_hashes)

        # 4. Upload only new chunks (dedup check per chunk)
        uploaded = 0
        deduped = 0
        for chunk_hash, chunk_bytes in chunks:
            if chunk_hash in delta["to_upload"]:
                self.dedup_store.store_chunk(chunk_hash, chunk_bytes)
                uploaded += 1
            elif self.dedup_store.has_chunk(chunk_hash):
                deduped += 1

        # 5. Create or update file record
        if not file_id:
            file_id = self.metadata.create_file(user_id, filename, folder_id)

        # 6. Resolve conflicts and commit version
        resolution = self.conflict_resolver.resolve(
            file_id, new_chunk_hashes, len(content), user_id, base_version
        )

        return {
            "file_id": file_id,
            "filename": filename,
            "total_chunks": len(chunks),
            "chunks_uploaded": uploaded,
            "chunks_deduped": deduped,
            "bandwidth_savings": delta["bandwidth_savings_pct"],
            "resolution": resolution
        }

    def download_file(self, user_id: str, file_id: str,
                      version_num: Optional[int] = None) -> dict:
        """Get ordered chunk URLs for a file (or specific version)."""
        file_meta = self.metadata.get_file(file_id)
        if not file_meta:
            return {"error": "file_not_found"}

        if version_num:
            versions = self.metadata.get_versions(file_id)
            version = next((v for v in versions if v.version_num == version_num), None)
        else:
            version = self.metadata.get_current_version(file_id)

        if not version:
            return {"error": "version_not_found"}

        chunk_urls = []
        for i, chunk_hash in enumerate(version.chunk_hashes):
            url = self.dedup_store.get_chunk_url(chunk_hash)
            chunk_urls.append({"index": i, "hash": chunk_hash[:12] + "...", "url": url})

        return {
            "file_id": file_id,
            "filename": file_meta.name,
            "version": version.version_num,
            "size_bytes": version.size_bytes,
            "chunk_count": len(chunk_urls),
            "chunk_urls": chunk_urls
        }

    def get_versions(self, file_id: str) -> list[dict]:
        versions = self.metadata.get_versions(file_id)
        return [
            {
                "version_num": v.version_num,
                "version_id": v.version_id,
                "size_bytes": v.size_bytes,
                "chunk_count": len(v.chunk_hashes),
                "created_at": datetime.fromtimestamp(v.created_at).strftime('%Y-%m-%d %H:%M:%S'),
                "is_current": v.is_current
            }
            for v in versions
        ]

    def share_file(self, owner_id: str, file_id: str,
                   shared_with: Optional[str] = None,
                   permission: str = "view") -> dict:
        share = Share(
            share_id=str(uuid.uuid4()),
            resource_id=file_id,
            resource_type="file",
            owner_id=owner_id,
            shared_with=shared_with,
            permission=permission,
            public_token=uuid.uuid4().hex[:12] if shared_with is None else None,
            expires_at=None
        )
        self.metadata.add_share(share)
        result = {
            "share_id": share.share_id,
            "permission": permission,
        }
        if share.public_token:
            result["public_url"] = f"https://www.dropbox.com/s/{share.public_token}"
        else:
            result["shared_with"] = shared_with
        return result

    def sync_changes(self, user_id: str, since_timestamp: float) -> dict:
        events = self.metadata.get_changes_since(user_id, since_timestamp)
        return {
            "changes": [
                {
                    "event_type": e.event_type,
                    "file_id": e.file_id,
                    "version_id": e.version_id,
                    "timestamp": e.timestamp
                }
                for e in events
            ],
            "count": len(events)
        }

    def _find_file(self, user_id: str, filename: str,
                   folder_id: Optional[str]) -> Optional[str]:
        for fid, f in self.metadata._files.items():
            if f.user_id == user_id and f.name == filename and not f.is_deleted:
                return fid
        return None


# ---------------------------------------------------------------------------
# Demo / Simulation
# ---------------------------------------------------------------------------

def run_simulation():
    print("=" * 65)
    print("  Dropbox System Simulation")
    print("=" * 65)

    db = DropboxSystem()

    # --- Scenario 1: Upload a new file ---
    print("\n[1] Alice uploads report.pdf (1st time)")
    content_v1 = b"Introduction to distributed systems. " * 10 + b"Chapter 1: Basics."
    result = db.upload_file("alice", "report.pdf", content_v1)
    print(f"    File ID       : {result['file_id'][:8]}...")
    print(f"    Total chunks  : {result['total_chunks']}")
    print(f"    Uploaded      : {result['chunks_uploaded']}")
    print(f"    Bandwidth save: {result['bandwidth_savings']}")
    file_id = result['file_id']

    # --- Scenario 2: Update file (delta sync) ---
    print("\n[2] Alice updates report.pdf (appends new chapter)")
    content_v2 = content_v1 + b" Chapter 2: Scalability and fault tolerance concepts explained."
    result2 = db.upload_file("alice", "report.pdf", content_v2, base_version=1)
    print(f"    Total chunks  : {result2['total_chunks']}")
    print(f"    Uploaded      : {result2['chunks_uploaded']} (delta!)")
    print(f"    Bandwidth save: {result2['bandwidth_savings']}")

    # --- Scenario 3: Cross-user deduplication ---
    print("\n[3] Bob uploads the same file as Alice (deduplication)")
    result3 = db.upload_file("bob", "report_copy.pdf", content_v1)
    print(f"    Chunks uploaded: {result3['chunks_uploaded']}")
    print(f"    Chunks deduped : {result3['chunks_deduped']}")
    dedup_report = db.dedup_store.dedup_report()
    print(f"    Global dedup ratio: {dedup_report['dedup_ratio']}")
    print(f"    Bytes saved: {dedup_report['bytes_saved']} bytes")

    # --- Scenario 4: Download file ---
    print("\n[4] Alice downloads report.pdf")
    download = db.download_file("alice", file_id)
    print(f"    Filename    : {download['filename']}")
    print(f"    Version     : {download['version']}")
    print(f"    Size bytes  : {download['size_bytes']}")
    print(f"    Chunks      : {download['chunk_count']}")
    if download['chunk_urls']:
        print(f"    First URL   : {download['chunk_urls'][0]['url']}")

    # --- Scenario 5: Version history ---
    print("\n[5] File version history")
    versions = db.get_versions(file_id)
    for v in versions:
        marker = " <-- current" if v["is_current"] else ""
        print(f"    v{v['version_num']} | {v['size_bytes']} bytes | "
              f"{v['chunk_count']} chunks | {v['created_at']}{marker}")

    # --- Scenario 6: Conflict resolution ---
    print("\n[6] Conflict: Carol edits report.pdf based on old version")
    content_carol = content_v1 + b" Carol's conflicting edit to chapter 1."
    result_carol = db.upload_file("alice", "report.pdf", content_carol, base_version=1)
    print(f"    Resolution action : {result_carol['resolution']['action']}")
    if result_carol['resolution']['action'] == 'conflict':
        conflict_id = result_carol['resolution'].get('conflict_copy_file_id', '')
        print(f"    Conflict copy ID  : {conflict_id[:8]}...")
        print("    Conflict copy saved alongside original")

    # --- Scenario 7: File sharing ---
    print("\n[7] Alice shares report.pdf publicly")
    share = db.share_file("alice", file_id, permission="view")
    print(f"    Share ID    : {share['share_id'][:8]}...")
    print(f"    Public URL  : {share['public_url']}")

    print("\n[8] Alice shares report.pdf with Bob (edit permission)")
    share2 = db.share_file("alice", file_id, shared_with="bob", permission="edit")
    print(f"    Shared with : {share2['shared_with']}")
    print(f"    Permission  : {share2['permission']}")

    # --- Scenario 8: Offline sync queue ---
    print("\n[9] Dave is offline — queuing changes")
    offline_content = b"Dave's offline notes. " * 5
    db.offline_queue.enqueue({
        "type": "upload",
        "user_id": "dave",
        "filename": "offline_notes.txt",
        "content": offline_content,
        "base_version": 0
    })
    db.offline_queue.enqueue({
        "type": "upload",
        "user_id": "dave",
        "filename": "offline_notes.txt",
        "content": offline_content + b" Updated paragraph.",
        "base_version": 1
    })
    print(f"    Queued operations: {db.offline_queue.pending_count()}")
    print("    Dave reconnects — replaying queue...")
    replay_results = db.offline_queue.replay(db)
    for r in replay_results:
        print(f"    Replayed '{r['op']}' -> chunks uploaded: {r['result']['chunks_uploaded']}")

    # --- Scenario 9: Sync changes ---
    print("\n[10] Alice syncs — checking for changes")
    changes = db.sync_changes("alice", since_timestamp=0)
    print(f"    Total change events: {changes['count']}")

    # --- Final dedup stats ---
    print("\n" + "=" * 65)
    print("  Final Deduplication Statistics")
    print("=" * 65)
    final_stats = db.dedup_store.dedup_report()
    for k, v in final_stats.items():
        print(f"  {k:<25}: {v}")
    print("=" * 65)


if __name__ == "__main__":
    run_simulation()
