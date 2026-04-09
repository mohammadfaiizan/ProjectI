"""
CHUNKING AND DEDUPLICATION
============================

Problem Statement:
Storing large files repeatedly wastes disk space. Backup systems,
version control, and cloud storage use content-aware chunking +
deduplication to store each unique chunk only once.

Key Concepts:

  Fixed-size Chunking:
    Split file into equal-size blocks (4KB, 64KB, 1MB).
    Simple. Poor deduplication on shifted content.
    When bytes inserted at offset 0, every block changes → no dedup match.

  Variable-size / Content-Defined Chunking (CDC):
    Rolling hash (Rabin fingerprint) over sliding window.
    Cut at chunk boundaries defined by hash modulo threshold.
    Insertions/deletions only affect local chunks → high dedup ratio.
    Average chunk size controlled by threshold (target: 4KB–8KB).

  Rabin Fingerprint:
    Rolling polynomial hash: fast to update with sliding window.
    hash(window[i+1]) = update(hash(window[i]), remove old byte, add new byte).
    Cut boundary when hash & mask == 0.

  Deduplication Store:
    Content-Addressed Storage (CAS): key = SHA-256(chunk_data).
    Reference count tracks how many files reference each chunk.
    Garbage collection removes unreferenced chunks.

  Deduplication Ratio:
    ratio = total_data_bytes / unique_data_bytes.
    Typical: 2x–20x for backup workloads. Up to 100x for VMs.

  Inline vs Post-process:
    Inline: hash during write, dedup before storing. Latency impact.
    Post-process: store first, dedup in background. Faster writes.

  Delta Compression:
    After dedup, compress similar-but-not-identical chunks using deltas.
    rsync/zstd delta: store diff between similar chunks.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple
import hashlib
import time
import random


# ─────────────────────────────────────────────
# FIXED-SIZE CHUNKER
# ─────────────────────────────────────────────

class FixedChunker:
    def __init__(self, chunk_size: int = 4096):
        self.chunk_size = chunk_size

    def chunks(self, data: bytes) -> Iterator[bytes]:
        for i in range(0, len(data), self.chunk_size):
            yield data[i: i + self.chunk_size]

    def chunk_count(self, data_len: int) -> int:
        return (data_len + self.chunk_size - 1) // self.chunk_size


# ─────────────────────────────────────────────
# CONTENT-DEFINED CHUNKER (Rabin-like CDC)
# ─────────────────────────────────────────────

class CDCChunker:
    """
    Content-Defined Chunking using a simplified rolling hash.
    Cuts at positions where rolling_hash & mask == trigger.
    Average chunk size ≈ (mask + 1) bytes.
    """

    WINDOW_SIZE   = 48
    MIN_CHUNK     = 512
    MAX_CHUNK     = 65536

    def __init__(self, avg_chunk_size: int = 4096):
        # mask controls cut frequency: avg_chunk ≈ mask+1
        self.mask    = avg_chunk_size - 1
        self.trigger = 0     # cut when hash & mask == trigger

    def _rolling_hash(self, window: bytearray) -> int:
        """Simplified polynomial rolling hash over window."""
        h = 0
        for b in window:
            h = (h * 31 + b) & 0xFFFFFFFF
        return h

    def chunks(self, data: bytes) -> Iterator[bytes]:
        if len(data) <= self.MIN_CHUNK:
            yield data
            return

        start  = 0
        window = bytearray(data[:self.WINDOW_SIZE])
        i      = self.WINDOW_SIZE

        while i < len(data):
            # Slide window
            window.pop(0)
            window.append(data[i])
            chunk_len = i - start

            if chunk_len >= self.MIN_CHUNK:
                rh = self._rolling_hash(window)
                if (rh & self.mask) == self.trigger or chunk_len >= self.MAX_CHUNK:
                    yield data[start:i]
                    start = i
            i += 1

        # Last chunk
        if start < len(data):
            yield data[start:]

    def chunk_sizes(self, data: bytes) -> List[int]:
        return [len(c) for c in self.chunks(data)]


# ─────────────────────────────────────────────
# CHUNK STORE (Content-Addressed Storage)
# ─────────────────────────────────────────────

@dataclass
class ChunkEntry:
    hash_id    : str
    data       : bytes
    ref_count  : int = 0
    size       : int = 0
    stored_at  : float = field(default_factory=time.time)

    def __post_init__(self):
        self.size = len(self.data)


class ChunkStore:
    """
    Content-addressed store: chunks keyed by SHA-256 hash.
    Reference counting enables safe garbage collection.
    """

    def __init__(self):
        self._store     : Dict[str, ChunkEntry] = {}
        self.writes_total   = 0
        self.dedup_hits     = 0

    def store_chunk(self, data: bytes) -> str:
        """Store chunk; returns hash. Increments ref count if duplicate."""
        hash_id = hashlib.sha256(data).hexdigest()[:16]
        if hash_id in self._store:
            self._store[hash_id].ref_count += 1
            self.dedup_hits += 1
        else:
            self._store[hash_id] = ChunkEntry(hash_id=hash_id, data=data, ref_count=1)
        self.writes_total += 1
        return hash_id

    def get_chunk(self, hash_id: str) -> Optional[bytes]:
        entry = self._store.get(hash_id)
        return entry.data if entry else None

    def release_chunk(self, hash_id: str):
        """Decrement reference count; remove if zero."""
        entry = self._store.get(hash_id)
        if entry:
            entry.ref_count -= 1
            if entry.ref_count <= 0:
                del self._store[hash_id]

    def gc(self) -> int:
        """Remove unreferenced chunks. Returns count removed."""
        orphans = [h for h, e in self._store.items() if e.ref_count <= 0]
        for h in orphans:
            del self._store[h]
        return len(orphans)

    def stats(self) -> Dict:
        unique_bytes  = sum(e.size for e in self._store.values())
        total_logical = self.writes_total  # one write per chunk reference attempt
        return {
            "unique_chunks"  : len(self._store),
            "unique_bytes"   : unique_bytes,
            "writes_total"   : self.writes_total,
            "dedup_hits"     : self.dedup_hits,
            "dedup_ratio"    : self.writes_total / max(len(self._store), 1),
        }


# ─────────────────────────────────────────────
# FILE BACKUP SYSTEM
# ─────────────────────────────────────────────

@dataclass
class BackupFile:
    filename  : str
    chunk_ids : List[str]
    size_bytes: int
    created_at: float = field(default_factory=time.time)


class BackupSystem:
    """
    Deduplication-based backup: chunked + content-addressed.
    Multiple backups of similar files share physical chunks.
    """

    def __init__(self, chunker=None):
        self._chunker  = chunker or CDCChunker(avg_chunk_size=1024)
        self._store    = ChunkStore()
        self._backups  : Dict[str, BackupFile] = {}   # backup_id → file

    def backup(self, filename: str, data: bytes) -> BackupFile:
        chunk_ids = []
        for chunk in self._chunker.chunks(data):
            cid = self._store.store_chunk(chunk)
            chunk_ids.append(cid)
        bf = BackupFile(filename=filename, chunk_ids=chunk_ids, size_bytes=len(data))
        backup_id = f"{filename}@{int(time.time()*1000)}"
        self._backups[backup_id] = bf
        return bf

    def restore(self, backup_file: BackupFile) -> bytes:
        parts = []
        for cid in backup_file.chunk_ids:
            chunk = self._store.get_chunk(cid)
            if chunk is None:
                raise RuntimeError(f"Missing chunk {cid}")
            parts.append(chunk)
        return b"".join(parts)

    def delete_backup(self, backup_id: str):
        bf = self._backups.pop(backup_id, None)
        if bf:
            for cid in bf.chunk_ids:
                self._store.release_chunk(cid)
            self._store.gc()

    def storage_stats(self) -> Dict:
        total_logical = sum(b.size_bytes for b in self._backups.values())
        cs            = self._store.stats()
        return {
            "backups"         : len(self._backups),
            "logical_bytes"   : total_logical,
            "physical_bytes"  : cs["unique_bytes"],
            "dedup_ratio"     : total_logical / max(cs["unique_bytes"], 1),
            "space_saved_pct" : (1 - cs["unique_bytes"] / max(total_logical, 1)) * 100,
            **cs,
        }


# ─────────────────────────────────────────────
# DEDUP RATIO ANALYSIS
# ─────────────────────────────────────────────

def analyze_dedup(original: bytes, modified_bytes: int) -> Dict:
    """
    Compare fixed vs CDC chunking dedup efficiency
    when N bytes are inserted at the start of a file.
    """
    modified = b"X" * modified_bytes + original[:-modified_bytes] if modified_bytes else original

    fixed   = FixedChunker(chunk_size=512)
    cdc     = CDCChunker(avg_chunk_size=512)

    def chunk_hashes(chunker, data: bytes) -> set:
        return {hashlib.sha256(c).hexdigest() for c in chunker.chunks(data)}

    orig_fixed = chunk_hashes(fixed, original)
    mod_fixed  = chunk_hashes(fixed, modified)
    dedup_fixed = len(orig_fixed & mod_fixed) / max(len(orig_fixed | mod_fixed), 1)

    orig_cdc  = chunk_hashes(cdc, original)
    mod_cdc   = chunk_hashes(cdc, modified)
    dedup_cdc = len(orig_cdc & mod_cdc) / max(len(orig_cdc | mod_cdc), 1)

    return {
        "modified_bytes"   : modified_bytes,
        "fixed_dedup_ratio": dedup_fixed,
        "cdc_dedup_ratio"  : dedup_cdc,
        "cdc_advantage"    : dedup_cdc - dedup_fixed,
    }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_chunking_dedup():
    print("=" * 65)
    print("CHUNKING AND DEDUPLICATION")
    print("=" * 65)

    random.seed(42)

    # ── Chunking Strategies ───────────────────────
    print("\n[1] FIXED vs CDC CHUNKING — CHUNK SIZE DISTRIBUTION")
    print("─" * 55)

    base_text  = b"The quick brown fox jumps over the lazy dog. " * 200   # ~9KB
    modified   = b"INSERTED TEXT. " + base_text   # 15 bytes inserted at start

    fixed = FixedChunker(chunk_size=512)
    cdc   = CDCChunker(avg_chunk_size=512)

    fixed_orig = list(fixed.chunks(base_text))
    cdc_orig   = list(cdc.chunks(base_text))

    print(f"  Original file ({len(base_text)}B):")
    print(f"    Fixed-size: {len(fixed_orig)} chunks of exactly 512B")
    cdc_sizes = [len(c) for c in cdc_orig]
    print(f"    CDC:        {len(cdc_orig)} chunks, sizes: "
          f"min={min(cdc_sizes)} avg={sum(cdc_sizes)//len(cdc_sizes)} max={max(cdc_sizes)}")

    # ── Dedup After Modification ──────────────────
    print("\n\n[2] DEDUP RATIO AFTER SMALL EDIT")
    print("─" * 55)
    print(f"  {'Bytes modified':>16} {'Fixed dedup':>14} {'CDC dedup':>12} {'CDC advantage':>14}")
    print(f"  {'─'*58}")
    for mod_bytes in [0, 1, 10, 50, 100]:
        r = analyze_dedup(base_text, mod_bytes)
        print(f"  {mod_bytes:>16}   {r['fixed_dedup_ratio']:>12.1%}   "
              f"{r['cdc_dedup_ratio']:>10.1%}   {r['cdc_advantage']:>+12.1%}")
    print(f"  → CDC maintains dedup ratio even when bytes inserted at offset 0")

    # ── Backup System ─────────────────────────────
    print("\n\n[3] BACKUP SYSTEM — MULTI-VERSION DEDUP")
    print("─" * 55)

    bsys = BackupSystem(CDCChunker(avg_chunk_size=512))

    # Simulate 3 daily backups with small daily changes
    doc_v1 = base_text
    doc_v2 = base_text[:4000] + b"NEW PARAGRAPH " + base_text[4000:]
    doc_v3 = doc_v2 + b"\nFINAL LINE APPENDED"

    bf1 = bsys.backup("report.docx", doc_v1)
    bf2 = bsys.backup("report.docx", doc_v2)
    bf3 = bsys.backup("report.docx", doc_v3)

    print(f"  Backed up 3 versions of report.docx:")
    print(f"    v1: {bf1.size_bytes}B → {len(bf1.chunk_ids)} chunks")
    print(f"    v2: {bf2.size_bytes}B → {len(bf2.chunk_ids)} chunks")
    print(f"    v3: {bf3.size_bytes}B → {len(bf3.chunk_ids)} chunks")

    s = bsys.storage_stats()
    print(f"\n  Storage stats:")
    print(f"    Logical bytes:  {s['logical_bytes']:,}B")
    print(f"    Physical bytes: {s['physical_bytes']:,}B")
    print(f"    Dedup ratio:    {s['dedup_ratio']:.1f}x")
    print(f"    Space saved:    {s['space_saved_pct']:.0f}%")
    print(f"    Unique chunks:  {s['unique_chunks']}")
    print(f"    Dedup hits:     {s['dedup_hits']}")

    # ── Restore ───────────────────────────────────
    print("\n\n[4] RESTORE — REASSEMBLE FROM CHUNKS")
    print("─" * 55)

    restored = bsys.restore(bf1)
    print(f"  Restored v1: {len(restored)}B == original: {restored == doc_v1}")
    restored3 = bsys.restore(bf3)
    print(f"  Restored v3: {len(restored3)}B == original: {restored3 == doc_v3}")

    # ── Garbage Collection ────────────────────────
    print("\n\n[5] GARBAGE COLLECTION — FREE ORPHANED CHUNKS")
    print("─" * 55)

    before = bsys._store.stats()["unique_chunks"]
    # Delete v1 backup — some of its unique chunks may become orphaned
    backup_ids = list(bsys._backups.keys())
    bsys.delete_backup(backup_ids[0])
    after = bsys._store.stats()["unique_chunks"]
    print(f"  Before delete v1: {before} unique chunks")
    print(f"  After GC:         {after} unique chunks")
    print(f"  Freed:            {before - after} chunks")

    # ── Design Summary ────────────────────────────
    print("\n\n[6] CHUNKING DESIGN DECISIONS")
    print("─" * 55)
    decisions = [
        ("CDC (variable size)", "Handles insertions/deletions; better dedup than fixed"),
        ("Content hash key",    "SHA-256 of chunk = unique ID; automatic dedup"),
        ("Reference counting",  "Safe GC: only delete when no backup references chunk"),
        ("Min/Max chunk size",  "Min prevents tiny chunks; max prevents huge chunks"),
        ("Inline dedup",        "Hash on write path; higher latency but saves I/O"),
        ("Rabin fingerprint",   "O(1) rolling update; efficient boundary detection"),
        ("4KB target size",     "Matches OS page size; aligns with filesystem blocks"),
        ("SHA-256 for security","Cryptographic hash prevents hash collision exploits"),
    ]
    for decision, reason in decisions:
        print(f"  {decision:<26} {reason}")


if __name__ == "__main__":
    demonstrate_chunking_dedup()
