"""
DROPBOX — File Storage and Sync Service
========================================

FUNCTIONAL REQUIREMENTS:
- Upload / download files up to 50 GB
- Sync changes across devices in real-time
- Share files/folders with other users
- Version history (30-day retention)
- Conflict detection and resolution

NON-FUNCTIONAL REQUIREMENTS:
- 500 M registered users, 100 M DAU
- Average file size 1 MB → 100 TB/day upload
- 99.99% durability (S3-backed, 11 nines target)
- Sync latency < 1 s for small files

ARCHITECTURE OVERVIEW:
┌────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Sync Client│────▶│  API Server  │────▶│  Metadata DB     │
│ (desktop)  │     │  (REST/gRPC) │     │  (PostgreSQL)    │
└────────────┘     └──────┬───────┘     └──────────────────┘
                          │                       │
                   ┌──────▼───────┐     ┌─────────▼────────┐
                   │  Block Store │     │  Notification Svc│
                   │  (S3-backed) │     │  (SSE / WebSocket│
                   └──────────────┘     └──────────────────┘

KEY DESIGN DECISIONS:
1. CHUNKING — split files into 4 MB blocks; each block content-addressed by SHA-256.
   Dedup: if block hash already in store, skip upload.  Delta sync: only upload
   changed blocks (great for large files with small edits).

2. METADATA SERVICE — stores file tree (path → {block_list, version, mtime}).
   Separate from block data → can scale independently.  PostgreSQL with JSONB
   for block lists; CockroachDB for global distribution.

3. CONFLICT RESOLUTION — last-write-wins based on client-reported mtime.
   On true concurrent edit: server keeps both versions and renames one to
   "filename (conflicted copy 2024-01-01).ext".

4. SYNC PROTOCOL:
   a. Client computes block hashes of changed file.
   b. Calls /diff endpoint → server returns list of missing block hashes.
   c. Client uploads only missing blocks.
   d. Client commits new file version with ordered block list.
   e. Server notifies other devices via long-poll / SSE.

5. BANDWIDTH OPTIMISATION — rsync-style rolling hash for identifying unchanged
   regions within a block boundary; compression (zstd) before upload.

BLOCK STORAGE:
- Content-addressed: key = SHA-256 of raw block data
- Immutable once written (no updates, only new blocks)
- Reference-counted for GC: delete block when refcount = 0
- Multi-region replication for durability

VERSION HISTORY:
- Every commit creates a new FileVersion record pointing to block list.
- Restore = commit old block list as new head version.
- 30-day retention via scheduled GC job.

STORAGE ESTIMATES:
- 100 M files/day × 1 MB avg = 100 TB/day
- After dedup + compression ≈ 50 TB/day net new data
- 5-year storage: 50 TB × 365 × 5 ≈ 91 PB
"""

from __future__ import annotations
import hashlib
import time
import uuid
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict


# ---------------------------------------------------------------------------
# Block Store — content-addressed, immutable
# ---------------------------------------------------------------------------

BLOCK_SIZE = 4 * 1024 * 1024  # 4 MB


@dataclass
class Block:
    hash: str        # SHA-256 hex
    data: bytes
    size: int
    ref_count: int = 0

    @staticmethod
    def from_data(data: bytes) -> "Block":
        h = hashlib.sha256(data).hexdigest()
        return Block(hash=h, data=data, size=len(data))


class BlockStore:
    """Content-addressed block storage (simulates S3 backend)."""

    def __init__(self):
        self._blocks: Dict[str, Block] = {}
        self._upload_bytes = 0

    def has(self, block_hash: str) -> bool:
        return block_hash in self._blocks

    def put(self, block: Block) -> str:
        if block.hash not in self._blocks:
            self._blocks[block.hash] = block
            self._upload_bytes += block.size
        self._blocks[block.hash].ref_count += 1
        return block.hash

    def get(self, block_hash: str) -> Optional[bytes]:
        b = self._blocks.get(block_hash)
        return b.data if b else None

    def release(self, block_hash: str) -> None:
        if block_hash in self._blocks:
            self._blocks[block_hash].ref_count -= 1
            if self._blocks[block_hash].ref_count <= 0:
                del self._blocks[block_hash]

    @property
    def total_blocks(self) -> int:
        return len(self._blocks)

    @property
    def total_bytes(self) -> int:
        return sum(b.size for b in self._blocks.values())


# ---------------------------------------------------------------------------
# Chunker — splits file data into blocks
# ---------------------------------------------------------------------------

class Chunker:
    """Fixed-size chunker; real Dropbox uses content-defined chunking (CDC)."""

    @staticmethod
    def split(data: bytes) -> List[Block]:
        blocks = []
        for i in range(0, len(data), BLOCK_SIZE):
            chunk = data[i: i + BLOCK_SIZE]
            blocks.append(Block.from_data(chunk))
        return blocks

    @staticmethod
    def reassemble(store: BlockStore, block_hashes: List[str]) -> bytes:
        parts = []
        for h in block_hashes:
            part = store.get(h)
            if part is None:
                raise ValueError(f"Block {h[:8]}... not found")
            parts.append(part)
        return b"".join(parts)


# ---------------------------------------------------------------------------
# Metadata Service
# ---------------------------------------------------------------------------

@dataclass
class FileVersion:
    version_id: str
    block_hashes: List[str]
    size: int
    mtime: float          # client mtime (epoch seconds)
    created_at: float     # server ingestion time
    uploader_device: str
    checksum: str         # SHA-256 of full file


@dataclass
class FileMeta:
    file_id: str
    owner_id: str
    path: str             # e.g. "/Photos/vacation.jpg"
    versions: List[FileVersion] = field(default_factory=list)
    is_deleted: bool = False

    @property
    def head(self) -> Optional[FileVersion]:
        return self.versions[-1] if self.versions else None


class MetadataService:
    """Stores file tree and version history."""

    def __init__(self):
        # owner_id → path → FileMeta
        self._files: Dict[str, Dict[str, FileMeta]] = defaultdict(dict)
        self._id_index: Dict[str, FileMeta] = {}

    def create_file(self, owner_id: str, path: str) -> FileMeta:
        meta = FileMeta(
            file_id=str(uuid.uuid4()),
            owner_id=owner_id,
            path=path,
        )
        self._files[owner_id][path] = meta
        self._id_index[meta.file_id] = meta
        return meta

    def get_file(self, owner_id: str, path: str) -> Optional[FileMeta]:
        return self._files[owner_id].get(path)

    def commit_version(
        self,
        owner_id: str,
        path: str,
        block_hashes: List[str],
        size: int,
        mtime: float,
        device_id: str,
        checksum: str,
    ) -> Tuple[FileVersion, bool]:
        """Returns (version, conflict_detected)."""
        meta = self.get_file(owner_id, path)
        if meta is None:
            meta = self.create_file(owner_id, path)

        conflict = False
        if meta.head and meta.head.mtime > mtime + 1:
            # Server version is newer — conflict
            conflict = True
            # Create conflict copy path
            base, _, ext = path.rpartition(".")
            conflict_path = f"{base} (conflicted copy {time.strftime('%Y-%m-%d')}).{ext}"
            conflict_meta = self.create_file(owner_id, conflict_path)
            meta = conflict_meta

        version = FileVersion(
            version_id=str(uuid.uuid4()),
            block_hashes=block_hashes,
            size=size,
            mtime=mtime,
            created_at=time.time(),
            uploader_device=device_id,
            checksum=checksum,
        )
        meta.versions.append(version)
        return version, conflict

    def list_files(self, owner_id: str) -> List[FileMeta]:
        return [f for f in self._files[owner_id].values() if not f.is_deleted]

    def delete_file(self, owner_id: str, path: str) -> bool:
        meta = self.get_file(owner_id, path)
        if meta:
            meta.is_deleted = True
            return True
        return False

    def restore_version(self, owner_id: str, path: str, version_id: str) -> Optional[FileVersion]:
        meta = self.get_file(owner_id, path)
        if not meta:
            return None
        for v in meta.versions:
            if v.version_id == version_id:
                # Re-commit old version as new head
                new_version = FileVersion(
                    version_id=str(uuid.uuid4()),
                    block_hashes=v.block_hashes,
                    size=v.size,
                    mtime=time.time(),
                    created_at=time.time(),
                    uploader_device="restore",
                    checksum=v.checksum,
                )
                meta.versions.append(new_version)
                return new_version
        return None


# ---------------------------------------------------------------------------
# Sync Engine — client-side delta computation
# ---------------------------------------------------------------------------

@dataclass
class SyncDiff:
    missing_hashes: List[str]    # blocks server doesn't have
    upload_bytes: int


class SyncEngine:
    """Orchestrates the upload protocol."""

    def __init__(self, block_store: BlockStore, metadata: MetadataService):
        self.block_store = block_store
        self.metadata = metadata

    def compute_diff(self, block_hashes: List[str]) -> SyncDiff:
        """Server returns which blocks it's missing."""
        missing = [h for h in block_hashes if not self.block_store.has(h)]
        upload_bytes = 0
        return SyncDiff(missing_hashes=missing, upload_bytes=upload_bytes)

    def upload_blocks(self, blocks: List[Block], needed: List[str]) -> int:
        """Upload only needed blocks. Returns bytes uploaded."""
        needed_set = set(needed)
        uploaded = 0
        for block in blocks:
            if block.hash in needed_set:
                self.block_store.put(block)
                uploaded += block.size
        return uploaded

    def commit(
        self,
        owner_id: str,
        path: str,
        block_hashes: List[str],
        size: int,
        mtime: float,
        device_id: str,
        checksum: str,
    ) -> Tuple[FileVersion, bool]:
        return self.metadata.commit_version(
            owner_id, path, block_hashes, size, mtime, device_id, checksum
        )

    def download(self, owner_id: str, path: str) -> Optional[bytes]:
        meta = self.metadata.get_file(owner_id, path)
        if not meta or not meta.head:
            return None
        return Chunker.reassemble(self.block_store, meta.head.block_hashes)


# ---------------------------------------------------------------------------
# Notification Service (SSE simulation)
# ---------------------------------------------------------------------------

@dataclass
class ChangeEvent:
    user_id: str
    path: str
    event_type: str   # "modified" | "deleted" | "created"
    version_id: str
    timestamp: float


class NotificationService:
    """Fans out change events to all devices of a user."""

    def __init__(self):
        # user_id → list of device callbacks (simulated)
        self._subscribers: Dict[str, List[str]] = defaultdict(list)
        self._event_log: List[ChangeEvent] = []

    def subscribe(self, user_id: str, device_id: str) -> None:
        if device_id not in self._subscribers[user_id]:
            self._subscribers[user_id].append(device_id)

    def notify(self, event: ChangeEvent) -> List[str]:
        self._event_log.append(event)
        notified = list(self._subscribers.get(event.user_id, []))
        return notified

    def pending_events(self, user_id: str, since: float) -> List[ChangeEvent]:
        return [e for e in self._event_log if e.user_id == user_id and e.timestamp > since]


# ---------------------------------------------------------------------------
# Sharing
# ---------------------------------------------------------------------------

@dataclass
class ShareLink:
    link_id: str
    owner_id: str
    path: str
    permission: str   # "view" | "edit"
    expires_at: Optional[float]
    created_at: float = field(default_factory=time.time)


class SharingService:
    def __init__(self):
        self._links: Dict[str, ShareLink] = {}
        self._collaborators: Dict[str, Dict[str, str]] = defaultdict(dict)  # path → user → perm

    def create_link(self, owner_id: str, path: str, permission: str = "view",
                    ttl_hours: float = 24 * 7) -> ShareLink:
        link = ShareLink(
            link_id=str(uuid.uuid4())[:8],
            owner_id=owner_id,
            path=path,
            permission=permission,
            expires_at=time.time() + ttl_hours * 3600,
        )
        self._links[link.link_id] = link
        return link

    def resolve_link(self, link_id: str) -> Optional[ShareLink]:
        link = self._links.get(link_id)
        if link and (link.expires_at is None or link.expires_at > time.time()):
            return link
        return None

    def add_collaborator(self, path: str, user_id: str, permission: str) -> None:
        self._collaborators[path][user_id] = permission

    def get_permission(self, path: str, user_id: str) -> Optional[str]:
        return self._collaborators[path].get(user_id)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demonstrate_1_chunking_and_dedup():
    print("\n=== 1. Chunking & Deduplication ===")
    store = BlockStore()

    # Simulate two files sharing a common block
    shared_block_data = b"X" * 4096   # 4 KB chunk
    file_a = b"A" * 2048 + shared_block_data
    file_b = b"B" * 2048 + shared_block_data  # second half identical to file_a

    blocks_a = Chunker.split(file_a)
    blocks_b = Chunker.split(file_b)

    print(f"File A blocks: {len(blocks_a)}, File B blocks: {len(blocks_b)}")

    for b in blocks_a:
        store.put(b)
    for b in blocks_b:
        store.put(b)

    # Count unique vs total
    hashes_a = {b.hash for b in blocks_a}
    hashes_b = {b.hash for b in blocks_b}
    shared = hashes_a & hashes_b
    print(f"Unique blocks in store: {store.total_blocks}")
    print(f"Shared blocks (deduped): {len(shared)}")
    print(f"Total logical blocks: {len(blocks_a) + len(blocks_b)}")
    dedup_ratio = 1 - store.total_blocks / (len(blocks_a) + len(blocks_b))
    print(f"Dedup ratio: {dedup_ratio:.1%}")


def demonstrate_2_delta_sync():
    print("\n=== 2. Delta Sync (Upload Only Changed Blocks) ===")
    block_store = BlockStore()
    metadata = MetadataService()
    engine = SyncEngine(block_store, metadata)

    owner = "user_alice"
    path = "/docs/report.pdf"
    device = "laptop_001"

    # --- Initial upload ---
    original_data = b"Section 1: Introduction\n" * 200 + b"Section 2: Data\n" * 200
    blocks_v1 = Chunker.split(original_data)
    checksum_v1 = hashlib.sha256(original_data).hexdigest()
    hashes_v1 = [b.hash for b in blocks_v1]

    diff = engine.compute_diff(hashes_v1)
    uploaded_v1 = engine.upload_blocks(blocks_v1, diff.missing_hashes)
    version1, conflict = engine.commit(owner, path, hashes_v1, len(original_data),
                                        time.time(), device, checksum_v1)
    print(f"Initial upload: {uploaded_v1} bytes uploaded, {len(blocks_v1)} blocks")
    print(f"Conflict: {conflict}")

    # --- Update: only section 2 changes ---
    updated_data = b"Section 1: Introduction\n" * 200 + b"Section 2: Updated!\n" * 200
    blocks_v2 = Chunker.split(updated_data)
    checksum_v2 = hashlib.sha256(updated_data).hexdigest()
    hashes_v2 = [b.hash for b in blocks_v2]

    diff2 = engine.compute_diff(hashes_v2)
    uploaded_v2 = engine.upload_blocks(blocks_v2, diff2.missing_hashes)
    version2, conflict2 = engine.commit(owner, path, hashes_v2, len(updated_data),
                                         time.time(), device, checksum_v2)

    unchanged_blocks = len(blocks_v2) - len(diff2.missing_hashes)
    print(f"\nAfter edit: {len(diff2.missing_hashes)} blocks changed, "
          f"{unchanged_blocks} blocks reused from server")
    print(f"Delta upload: {uploaded_v2} bytes (vs full {len(updated_data)} bytes)")
    savings = 1 - uploaded_v2 / max(len(updated_data), 1)
    print(f"Bandwidth saving: {savings:.1%}")


def demonstrate_3_conflict_resolution():
    print("\n=== 3. Conflict Resolution ===")
    metadata = MetadataService()
    block_store = BlockStore()
    engine = SyncEngine(block_store, metadata)

    owner = "user_bob"
    path = "/notes/meeting.txt"

    # Device A commits at time T
    t_server = time.time()
    data_a = b"Notes from device A"
    blocks_a = Chunker.split(data_a)
    for b in blocks_a:
        block_store.put(b)
    v1, _ = metadata.commit_version(
        owner, path, [b.hash for b in blocks_a], len(data_a),
        t_server, "device_A", hashlib.sha256(data_a).hexdigest()
    )
    print(f"Device A committed at mtime={t_server:.0f}")

    # Device B was offline; tries to commit with older mtime
    t_old = t_server - 3600  # 1 hour older
    data_b = b"Notes from device B (older)"
    blocks_b = Chunker.split(data_b)
    for b in blocks_b:
        block_store.put(b)
    v2, conflict = metadata.commit_version(
        owner, path, [b.hash for b in blocks_b], len(data_b),
        t_old, "device_B", hashlib.sha256(data_b).hexdigest()
    )
    print(f"Device B committed with older mtime={t_old:.0f}")
    print(f"Conflict detected: {conflict}")

    files = metadata.list_files(owner)
    print(f"Files in namespace after conflict: {len(files)}")
    for f in files:
        print(f"  {f.path}  (versions: {len(f.versions)})")


def demonstrate_4_version_history():
    print("\n=== 4. Version History & Restore ===")
    block_store = BlockStore()
    metadata = MetadataService()
    engine = SyncEngine(block_store, metadata)

    owner = "user_carol"
    path = "/code/main.py"
    device = "workstation"

    versions = []
    for i in range(1, 4):
        data = f"# Version {i}\nprint('hello v{i}')\n".encode()
        blocks = Chunker.split(data)
        for b in blocks:
            block_store.put(b)
        v, _ = metadata.commit_version(
            owner, path, [b.hash for b in blocks], len(data),
            time.time() + i, device, hashlib.sha256(data).hexdigest()
        )
        versions.append(v)
        print(f"  Committed version {i}: {v.version_id[:8]}...")

    meta = metadata.get_file(owner, path)
    print(f"\nTotal versions stored: {len(meta.versions)}")
    print(f"Current head: version_id={meta.head.version_id[:8]}...")

    # Restore v1
    restored = metadata.restore_version(owner, path, versions[0].version_id)
    print(f"\nRestored to v1: new version_id={restored.version_id[:8]}...")
    print(f"Total versions after restore: {len(meta.versions)}")


def demonstrate_5_sharing():
    print("\n=== 5. File Sharing ===")
    sharing = SharingService()

    owner = "user_dave"
    path = "/photos/vacation.zip"

    # Create public link
    link = sharing.create_link(owner, path, permission="view", ttl_hours=48)
    print(f"Share link created: id={link.link_id}, permission={link.permission}")

    resolved = sharing.resolve_link(link.link_id)
    print(f"Link resolved: {resolved is not None}, path={resolved.path if resolved else 'N/A'}")

    # Add collaborator with edit access
    sharing.add_collaborator(path, "user_eve", "edit")
    perm = sharing.get_permission(path, "user_eve")
    print(f"Collaborator 'user_eve' permission: {perm}")

    # Non-collaborator
    perm_other = sharing.get_permission(path, "user_frank")
    print(f"Non-collaborator 'user_frank' permission: {perm_other}")


def demonstrate_6_notification_sync():
    print("\n=== 6. Real-time Sync Notifications ===")
    notif = NotificationService()

    user = "user_grace"
    notif.subscribe(user, "laptop")
    notif.subscribe(user, "phone")
    notif.subscribe(user, "tablet")

    # Simulate file change from laptop — should notify phone and tablet
    event = ChangeEvent(
        user_id=user,
        path="/docs/budget.xlsx",
        event_type="modified",
        version_id=str(uuid.uuid4()),
        timestamp=time.time(),
    )
    notified = notif.notify(event)
    print(f"Change event sent. Devices notified: {notified}")

    # Simulate device coming online — fetch missed events
    events = notif.pending_events(user, since=time.time() - 60)
    print(f"Pending events in last 60s: {len(events)}")
    for e in events:
        print(f"  [{e.event_type}] {e.path}")


if __name__ == "__main__":
    demonstrate_1_chunking_and_dedup()
    demonstrate_2_delta_sync()
    demonstrate_3_conflict_resolution()
    demonstrate_4_version_history()
    demonstrate_5_sharing()
    demonstrate_6_notification_sync()
