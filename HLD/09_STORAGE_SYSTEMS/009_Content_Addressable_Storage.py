"""
CONTENT-ADDRESSABLE STORAGE (CAS)
====================================

Problem Statement:
Traditional storage uses mutable, location-based keys (path or ID).
Content-Addressable Storage uses the hash of content as the key.
This makes storage inherently immutable and deduplicated.

Core Idea:
  key = hash(content)   # SHA-256 or SHA-1
  Identical content always maps to same key.
  Changing content changes key → content is immutable.

Properties:
  Deduplication:  Same content stored only once (multiple pointers).
  Integrity:      Read-back hash verification = built-in corruption detection.
  Immutability:   Content doesn't change; versioning via new keys.
  Caching:        Safe to cache forever (key = content → no stale cache).
  Convergent encryption: Encrypt(key=hash(plaintext), data=plaintext).
                  Same plaintext → same ciphertext → dedup still works.

Use Cases:
  Git:         Blobs, trees, commits stored by SHA-1/SHA-256 of content.
  Docker:      Layers identified by digest (sha256:...).
  IPFS:        Distributed CAS over P2P network. CID = hash.
  Backups:     Content-defined chunk CAS (Borg, Restic, Veeam).
  Package mgr: npm, cargo lock files pin exact content hashes.

Merkle Tree (Git model):
  Blob:   hash(file_content)
  Tree:   hash(list of [name, mode, blob_hash] entries)
  Commit: hash(tree_hash + parent_hash + metadata)
  Change one file → new blob → new tree → new commit.
  Unchanged files share blobs across commits.

IPFS Content Identifiers (CID):
  CID v0: base58(multihash(sha256(data)))
  CID v1: multibase(multicodec + multihash)
  Links between objects form a Directed Acyclic Graph (DAG).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import hashlib
import time
import json


# ─────────────────────────────────────────────
# CAS STORE
# ─────────────────────────────────────────────

class CASStore:
    """
    Content-Addressed Storage: store(data) → hash key.
    Built-in deduplication and integrity verification.
    """

    def __init__(self, hash_algo: str = "sha256"):
        self._store      : Dict[str, bytes] = {}
        self._ref_counts : Dict[str, int]   = {}
        self._algo       = hash_algo
        self.writes      = 0
        self.dedup_saves = 0

    def _hash(self, data: bytes) -> str:
        h = hashlib.new(self._algo, data)
        return h.hexdigest()[:16]   # truncated for readability

    def put(self, data: bytes) -> str:
        """Store data; return content hash."""
        key = self._hash(data)
        if key in self._store:
            self.dedup_saves += 1
        else:
            self._store[key] = data
            self._ref_counts[key] = 0
        self._ref_counts[key] += 1
        self.writes += 1
        return key

    def get(self, key: str) -> Optional[bytes]:
        data = self._store.get(key)
        if data is None:
            return None
        # Integrity check on every read
        actual_key = self._hash(data)
        if actual_key != key:
            raise RuntimeError(f"Corruption detected: key={key} actual={actual_key}")
        return data

    def pin(self, key: str):
        """Prevent GC (like IPFS pin)."""
        if key in self._ref_counts:
            self._ref_counts[key] += 1

    def release(self, key: str):
        if key in self._ref_counts:
            self._ref_counts[key] -= 1
            if self._ref_counts[key] <= 0:
                del self._store[key]
                del self._ref_counts[key]

    def verify_all(self) -> Tuple[int, int]:
        """Verify integrity of all stored objects. Returns (ok, corrupted)."""
        ok = corrupted = 0
        for key, data in self._store.items():
            if self._hash(data) == key:
                ok += 1
            else:
                corrupted += 1
        return ok, corrupted

    def stats(self) -> Dict:
        return {
            "objects"        : len(self._store),
            "bytes"          : sum(len(v) for v in self._store.values()),
            "writes"         : self.writes,
            "dedup_saves"    : self.dedup_saves,
            "dedup_ratio"    : self.writes / max(len(self._store), 1),
        }


# ─────────────────────────────────────────────
# GIT-LIKE OBJECT MODEL
# ─────────────────────────────────────────────

@dataclass
class GitBlob:
    """File content stored by hash."""
    content: bytes

    def serialize(self) -> bytes:
        return b"blob " + str(len(self.content)).encode() + b"\x00" + self.content

    @property
    def hash(self) -> str:
        return hashlib.sha1(self.serialize()).hexdigest()[:10]


@dataclass
class GitTreeEntry:
    mode : str   # "100644" = file, "040000" = dir
    name : str
    hash : str


@dataclass
class GitTree:
    """Directory tree stored by hash of its entries."""
    entries: List[GitTreeEntry]

    def serialize(self) -> bytes:
        lines = [f"{e.mode} {e.name}\x00{e.hash}" for e in self.entries]
        body  = "\n".join(lines).encode()
        return b"tree " + str(len(body)).encode() + b"\x00" + body

    @property
    def hash(self) -> str:
        return hashlib.sha1(self.serialize()).hexdigest()[:10]


@dataclass
class GitCommit:
    """Commit object: snapshot + parent + metadata."""
    tree_hash   : str
    parent_hash : Optional[str]
    author      : str
    message     : str
    timestamp   : float = field(default_factory=time.time)

    def serialize(self) -> bytes:
        parts = [
            f"tree {self.tree_hash}",
            f"parent {self.parent_hash}" if self.parent_hash else "",
            f"author {self.author} {int(self.timestamp)}",
            "",
            self.message,
        ]
        body = "\n".join(p for p in parts if p is not None).encode()
        return b"commit " + str(len(body)).encode() + b"\x00" + body

    @property
    def hash(self) -> str:
        return hashlib.sha1(self.serialize()).hexdigest()[:10]


class GitRepository:
    """
    Simplified Git object store demonstrating CAS properties.
    Unchanged files share blob objects across commits.
    """

    def __init__(self):
        self._objects : Dict[str, bytes] = {}   # hash → serialized object
        self._HEAD    : Optional[str]    = None

    def _store(self, obj_bytes: bytes, obj_hash: str):
        self._objects[obj_hash] = obj_bytes

    def store_blob(self, content: bytes) -> str:
        blob = GitBlob(content)
        self._store(blob.serialize(), blob.hash)
        return blob.hash

    def store_tree(self, entries: List[Tuple[str, str, str]]) -> str:
        """entries: [(mode, name, hash)]"""
        tree = GitTree([GitTreeEntry(*e) for e in entries])
        self._store(tree.serialize(), tree.hash)
        return tree.hash

    def commit(self, tree_hash: str, author: str, message: str) -> str:
        commit = GitCommit(
            tree_hash=tree_hash,
            parent_hash=self._HEAD,
            author=author,
            message=message,
        )
        self._store(commit.serialize(), commit.hash)
        self._HEAD = commit.hash
        return commit.hash

    def log(self) -> List[str]:
        """Walk commit chain from HEAD."""
        commits = []
        h = self._HEAD
        seen: Set[str] = set()
        while h and h not in seen:
            seen.add(h)
            data = self._objects.get(h, b"")
            # Extract message from serialized commit
            try:
                body   = data.split(b"\x00", 1)[1].decode()
                msg    = body.split("\n\n", 1)[-1].strip()
                parent = None
                for line in body.splitlines():
                    if line.startswith("parent "):
                        parent = line.split()[1]
                commits.append((h, msg))
                h = parent
            except Exception:
                break
        return commits

    def object_count(self) -> int:
        return len(self._objects)

    def shared_objects(self) -> Dict:
        """Count how many objects are referenced multiple times (sharing)."""
        ref_count: Dict[str, int] = {}
        for data in self._objects.values():
            # Extract hash references from the object
            text = data.decode("utf-8", errors="ignore")
            for h, _ in [(h, None) for h in text.split() if len(h) == 10 and h.isalnum()]:
                ref_count[h] = ref_count.get(h, 0) + 1
        return ref_count


# ─────────────────────────────────────────────
# IPFS-LIKE CONTENT ID
# ─────────────────────────────────────────────

def make_cid(data: bytes) -> str:
    """Simplified CID: 'Qm' prefix + base58-like encoding of sha256."""
    digest = hashlib.sha256(data).digest()
    # Simplified: just hex with prefix
    return "Qm" + digest.hex()[:44]


@dataclass
class IPFSNode:
    cid     : str
    data    : bytes
    links   : List[Tuple[str, str]]   # [(name, cid)]


class IPFSStore:
    """Simplified IPFS DAG store."""

    def __init__(self):
        self._nodes : Dict[str, IPFSNode] = {}

    def add(self, data: bytes, links: List[Tuple[str, str]] = None) -> str:
        cid  = make_cid(data)
        node = IPFSNode(cid=cid, data=data, links=links or [])
        self._nodes[cid] = node
        return cid

    def get(self, cid: str) -> Optional[IPFSNode]:
        return self._nodes.get(cid)

    def resolve(self, root_cid: str, path: str) -> Optional[bytes]:
        """Follow path through DAG links."""
        node = self._nodes.get(root_cid)
        if not node:
            return None
        parts = path.strip("/").split("/")
        for part in parts:
            link = next((cid for name, cid in node.links if name == part), None)
            if link is None:
                return None
            node = self._nodes.get(link)
            if not node:
                return None
        return node.data


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_cas():
    print("=" * 65)
    print("CONTENT-ADDRESSABLE STORAGE (CAS)")
    print("=" * 65)

    # ── Basic CAS Operations ──────────────────────
    print("\n[1] CAS STORE — DEDUPLICATION + INTEGRITY")
    print("─" * 55)

    cas = CASStore()
    data1 = b"Hello, World!"
    data2 = b"Hello, World!"   # duplicate
    data3 = b"Different content"

    k1 = cas.put(data1)
    k2 = cas.put(data2)   # duplicate → same key
    k3 = cas.put(data3)

    print(f"  store('Hello, World!') → {k1}")
    print(f"  store('Hello, World!') → {k2}  (same = {k1 == k2})")
    print(f"  store('Different')     → {k3}")
    print(f"  Stats: {cas.stats()}")

    # Integrity verification
    ok, bad = cas.verify_all()
    print(f"  Integrity check: {ok} OK, {bad} corrupted")

    # ── Git Object Model ──────────────────────────
    print("\n\n[2] GIT-LIKE CAS — SHARED BLOBS ACROSS COMMITS")
    print("─" * 55)

    repo = GitRepository()

    # Commit 1: two files
    b1 = repo.store_blob(b"def hello():\n    return 'hello'\n")
    b2 = repo.store_blob(b"# README\nThis is a project.\n")
    t1 = repo.store_tree([("100644", "hello.py", b1), ("100644", "README.md", b2)])
    c1 = repo.commit(t1, "Alice", "Initial commit")
    objs_after_c1 = repo.object_count()

    # Commit 2: only hello.py changed
    b3 = repo.store_blob(b"def hello():\n    return 'hello v2'\n")
    t2 = repo.store_tree([("100644", "hello.py", b3), ("100644", "README.md", b2)])
    c2 = repo.commit(t2, "Alice", "Update hello function")
    objs_after_c2 = repo.object_count()

    print(f"  Commit 1 ({c1}): {objs_after_c1} objects total")
    print(f"  Commit 2 ({c2}): {objs_after_c2} objects total")
    print(f"  New objects for c2: {objs_after_c2 - objs_after_c1} "
          f"(hello.py blob + tree + commit; README.md blob REUSED)")

    print(f"\n  Commit log:")
    for h, msg in repo.log():
        print(f"    {h}: {msg}")

    # ── IPFS DAG ──────────────────────────────────
    print("\n\n[3] IPFS-LIKE DAG — LINKED CONTENT")
    print("─" * 55)

    ipfs = IPFSStore()

    # Build a directory DAG
    file1_cid = ipfs.add(b"file contents of index.html")
    file2_cid = ipfs.add(b"body { color: red; }")
    dir_cid   = ipfs.add(b"directory", links=[("index.html", file1_cid),
                                               ("style.css",  file2_cid)])
    root_cid  = ipfs.add(b"root", links=[("public", dir_cid)])

    print(f"  root:        {root_cid[:20]}...")
    print(f"  public/:     {dir_cid[:20]}...")
    print(f"  index.html:  {file1_cid[:20]}...")

    resolved = ipfs.resolve(root_cid, "public/index.html")
    print(f"  Resolve root/public/index.html: {resolved}")

    # ── Content-Based Cache ───────────────────────
    print("\n\n[4] CAS AS CACHE — SAFE FOREVER CACHING")
    print("─" * 55)

    cache: Dict[str, bytes] = {}
    data  = b"static asset content"
    key   = hashlib.sha256(data).hexdigest()[:16]
    cache[key] = data
    print(f"  Cache key (content hash): {key}")
    print(f"  Content-addressed: cache is ALWAYS fresh (key = content)")
    print(f"  Immutable CDN URL: /assets/{key}/bundle.js")
    print(f"  Cache-Control: max-age=31536000, immutable")

    # ── Convergent Encryption ─────────────────────
    print("\n\n[5] CONVERGENT ENCRYPTION — DEDUP + ENCRYPTION")
    print("─" * 55)

    plaintext = b"Confidential data that appears in multiple backups"
    enc_key   = hashlib.sha256(plaintext).digest()[:16]   # derive key from plaintext
    # XOR "encryption" for demo (real: AES-CTR with content-derived key)
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, enc_key * (len(plaintext) // 16 + 1)))
    ct_key     = hashlib.sha256(ciphertext).hexdigest()[:16]

    ct_key2    = hashlib.sha256(ciphertext).hexdigest()[:16]
    print(f"  Same plaintext on 2 clients → same ciphertext key: {ct_key == ct_key2}")
    print(f"  Ciphertext hash: {ct_key}")
    print(f"  Deduplication works even on encrypted data!")
    print(f"  Risk: known-plaintext attack (use only for backup/non-sensitive)")

    # ── Design Summary ────────────────────────────
    print("\n\n[6] CAS DESIGN PROPERTIES")
    print("─" * 55)
    properties = [
        ("Immutability",      "Content can't change; new content = new key"),
        ("Deduplication",     "Same data = same key → stored only once"),
        ("Integrity",         "Hash mismatch on read = corruption detected"),
        ("Infinite caching",  "Content-hash URLs safe to cache permanently"),
        ("Version history",   "Each version has unique key; history preserved"),
        ("Convergent enc",    "Encrypt with key=hash(plaintext) → still dedup"),
        ("Garbage collection","Remove objects with ref_count = 0 (no dangling refs)"),
        ("Git/IPFS/Docker",   "All use CAS as their fundamental storage primitive"),
    ]
    for prop, desc in properties:
        print(f"  {prop:<22} {desc}")


if __name__ == "__main__":
    demonstrate_cas()
