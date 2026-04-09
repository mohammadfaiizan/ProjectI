"""
DISTRIBUTED FILE SYSTEM DESIGN
================================

Problem Statement:
A single server's disk can't store petabytes of data or serve millions of concurrent reads.
Distributed File Systems (DFS) spread data across many commodity nodes, providing:
  - Fault tolerance via replication
  - Horizontal capacity scaling
  - High aggregate throughput via parallelism

Key DFS Systems:
  GFS (Google File System, 2003): Original Google DFS. Influenced HDFS.
  HDFS (Hadoop DFS): Open-source GFS clone. Core of Hadoop ecosystem.
  Ceph: Object/block/file unified storage. Used in OpenStack, Kubernetes.
  GlusterFS: Scale-out NAS. POSIX-compliant distributed filesystem.
  Azure Data Lake Storage: Cloud-native DFS for analytics.

GFS/HDFS Architecture:
  Master/NameNode: stores metadata (file→chunks mapping, chunk→server mapping).
                   Single point for namespace operations. Replicated for HA.
  ChunkServer/DataNode: stores actual data in fixed-size chunks (64MB in GFS/HDFS).
  Client: talks to master for metadata; reads/writes chunks directly to chunk servers.

Read Path:
  1. Client asks NameNode: "where are chunks for /data/file.csv?"
  2. NameNode returns: chunk IDs + DataNode locations.
  3. Client reads directly from nearest DataNode.

Write Path (append only in GFS):
  1. Client asks NameNode for chunk locations.
  2. NameNode returns primary + secondary replicas.
  3. Client sends data to primary DataNode.
  4. Primary chains write to replicas.
  5. Primary ACKs client after all replicas confirm.

Design Choices:
  Chunk size 64MB: fewer metadata entries, large sequential I/O efficient.
  Replication factor 3: handles 2 simultaneous failures.
  Append-only: avoids complex concurrent write coordination.
  Single master: simplifies consistency at cost of scalability.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import time
import hashlib
import uuid
import random


CHUNK_SIZE = 64 * 1024 * 1024   # 64 MB
REPLICATION_FACTOR = 3


# ─────────────────────────────────────────────
# CHUNK
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id  : str
    data      : bytes
    version   : int = 1
    checksum  : str = ""

    def __post_init__(self):
        self.checksum = hashlib.md5(self.data).hexdigest()

    @property
    def size(self) -> int:
        return len(self.data)


# ─────────────────────────────────────────────
# DATA NODE (chunk server)
# ─────────────────────────────────────────────

class DataNode:
    def __init__(self, node_id: str, capacity_gb: int = 100):
        self.node_id      = node_id
        self.capacity_gb  = capacity_gb
        self._chunks      : Dict[str, Chunk] = {}
        self.alive        = True
        self.reads        = 0
        self.writes       = 0

    def store(self, chunk: Chunk) -> bool:
        if not self.alive:
            return False
        self._chunks[chunk.chunk_id] = chunk
        self.writes += 1
        return True

    def read(self, chunk_id: str) -> Optional[Chunk]:
        if not self.alive:
            return None
        self.reads += 1
        return self._chunks.get(chunk_id)

    def has_chunk(self, chunk_id: str) -> bool:
        return chunk_id in self._chunks

    def used_bytes(self) -> int:
        return sum(c.size for c in self._chunks.values())

    def heartbeat(self) -> Dict:
        return {
            "node_id"    : self.node_id,
            "alive"      : self.alive,
            "chunks"     : list(self._chunks.keys()),
            "used_bytes" : self.used_bytes(),
        }


# ─────────────────────────────────────────────
# NAME NODE (metadata server)
# ─────────────────────────────────────────────

@dataclass
class FileEntry:
    path          : str
    chunks        : List[str]           # ordered list of chunk_ids
    size          : int
    replication   : int = REPLICATION_FACTOR
    created_at    : float = field(default_factory=time.time)


class NameNode:
    """
    Stores file system namespace: path → chunks, chunk → DataNodes.
    In HDFS: NameNode. In GFS: Master.
    """

    def __init__(self):
        self._files        : Dict[str, FileEntry] = {}
        self._chunk_to_nodes: Dict[str, List[str]] = {}   # chunk_id → [node_id]
        self._data_nodes   : Dict[str, DataNode]   = {}
        self._edit_log     : List[Dict] = []   # WAL for recovery

    def register_node(self, node: DataNode):
        self._data_nodes[node.node_id] = node

    def create_file(self, path: str, n_chunks: int, replication: int = 3) -> FileEntry:
        chunk_ids = [str(uuid.uuid4())[:12] for _ in range(n_chunks)]
        entry     = FileEntry(path=path, chunks=chunk_ids, size=n_chunks * CHUNK_SIZE,
                              replication=replication)
        self._files[path] = entry
        self._edit_log.append({"op": "create", "path": path})
        return entry

    def get_chunk_locations(self, chunk_id: str) -> List[str]:
        """Returns DataNode IDs that hold this chunk."""
        return self._chunk_to_nodes.get(chunk_id, [])

    def assign_chunks(self, chunk_ids: List[str], replication: int):
        """Assign chunk replicas to DataNodes (rack-aware in real HDFS)."""
        alive_nodes = [n for n in self._data_nodes.values() if n.alive]
        for chunk_id in chunk_ids:
            # Pick `replication` distinct nodes
            targets = random.sample(alive_nodes, min(replication, len(alive_nodes)))
            self._chunk_to_nodes[chunk_id] = [n.node_id for n in targets]

    def report_node_failure(self, node_id: str):
        """Remove node; under-replicated chunks need re-replication."""
        if node_id in self._data_nodes:
            self._data_nodes[node_id].alive = False
        under_replicated = []
        for chunk_id, nodes in self._chunk_to_nodes.items():
            if node_id in nodes:
                self._chunk_to_nodes[chunk_id] = [n for n in nodes if n != node_id]
                if len(self._chunk_to_nodes[chunk_id]) < REPLICATION_FACTOR:
                    under_replicated.append(chunk_id)
        return under_replicated

    def file_info(self, path: str) -> Optional[FileEntry]:
        return self._files.get(path)

    def list_dir(self, prefix: str) -> List[str]:
        return [p for p in self._files if p.startswith(prefix)]


# ─────────────────────────────────────────────
# DFS CLIENT
# ─────────────────────────────────────────────

class DFSClient:
    def __init__(self, namenode: NameNode):
        self.namenode  = namenode
        self.bytes_written = 0
        self.bytes_read    = 0
        self.chunk_reads   = 0

    def write(self, path: str, data: bytes, replication: int = 3) -> bool:
        """Split data into chunks, assign to DataNodes, write replicas."""
        n_chunks = max(1, (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE)
        entry    = self.namenode.create_file(path, n_chunks, replication)
        self.namenode.assign_chunks(entry.chunks, replication)

        for i, chunk_id in enumerate(entry.chunks):
            chunk_data = data[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE]
            chunk      = Chunk(chunk_id=chunk_id, data=chunk_data)
            node_ids   = self.namenode.get_chunk_locations(chunk_id)

            # Write to each replica node
            for node_id in node_ids:
                node = self.namenode._data_nodes.get(node_id)
                if node:
                    node.store(chunk)

        self.bytes_written += len(data)
        return True

    def read(self, path: str) -> Optional[bytes]:
        """Read all chunks and reconstruct file."""
        entry = self.namenode.file_info(path)
        if not entry:
            return None

        result = b""
        for chunk_id in entry.chunks:
            node_ids = self.namenode.get_chunk_locations(chunk_id)
            # Try replicas until one succeeds
            chunk_data = None
            for node_id in node_ids:
                node = self.namenode._data_nodes.get(node_id)
                chunk = node.read(chunk_id) if node else None
                if chunk:
                    chunk_data = chunk.data
                    self.chunk_reads += 1
                    break
            if chunk_data is None:
                return None   # data loss — all replicas unavailable
            result += chunk_data

        self.bytes_read += len(result)
        return result[:entry.size] if len(result) > entry.size else result


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_dfs():
    print("=" * 65)
    print("DISTRIBUTED FILE SYSTEM DESIGN")
    print("=" * 65)

    random.seed(42)

    # ── Setup cluster ─────────────────────────────
    nn     = NameNode()
    nodes  = [DataNode(f"dn-{i}", capacity_gb=200) for i in range(5)]
    for n in nodes:
        nn.register_node(n)
    client = DFSClient(nn)

    print(f"\n[1] CLUSTER SETUP")
    print(f"─" * 55)
    print(f"  NameNode + {len(nodes)} DataNodes, replication factor = {REPLICATION_FACTOR}")
    print(f"  Chunk size: {CHUNK_SIZE // (1024*1024)}MB")

    # ── Write files ───────────────────────────────
    print(f"\n\n[2] WRITE — SPLIT INTO CHUNKS + REPLICATE")
    print(f"─" * 55)

    files = [
        ("/data/logs/app.log",       b"LOG ENTRY " * 1000),
        ("/data/models/weights.bin", b"\x00\x01\x02\x03" * 500),
        ("/data/logs/error.log",     b"ERROR: " * 200),
    ]
    for path, data in files:
        client.write(path, data)
        entry = nn.file_info(path)
        print(f"  Wrote {path}: {len(data)}B → {len(entry.chunks)} chunk(s)")
        for cid in entry.chunks:
            locs = nn.get_chunk_locations(cid)
            print(f"    chunk {cid}: replicas on {locs}")

    # ── Read back ─────────────────────────────────
    print(f"\n\n[3] READ — FETCH CHUNKS FROM DATANODES")
    print(f"─" * 55)
    for path, original_data in files[:2]:
        read_back = client.read(path)
        matches   = read_back[:len(original_data)] == original_data
        print(f"  Read {path}: {len(read_back)}B  matches original: {matches}")
    print(f"  Total chunk reads: {client.chunk_reads}")

    # ── Node failure + re-replication ─────────────
    print(f"\n\n[4] NODE FAILURE — UNDER-REPLICATED CHUNKS")
    print(f"─" * 55)
    failed_node = "dn-1"
    under_rep   = nn.report_node_failure(failed_node)
    print(f"  Failed: {failed_node}")
    print(f"  Under-replicated chunks: {len(under_rep)}")

    # Read still works (other replicas available)
    path     = "/data/logs/app.log"
    original = b"LOG ENTRY " * 1000
    read_after_failure = client.read(path)
    print(f"  Read after failure: {'OK' if read_after_failure else 'FAILED'} "
          f"(other replicas served it)")

    # ── List directory ────────────────────────────
    print(f"\n\n[5] NAMESPACE OPERATIONS")
    print(f"─" * 55)
    log_files = nn.list_dir("/data/logs/")
    print(f"  ls /data/logs/: {log_files}")

    # ── Architecture Summary ──────────────────────
    print(f"\n\n[6] GFS/HDFS DESIGN DECISIONS")
    print(f"─" * 55)
    rows = [
        ("Chunk size (64MB)",  "Fewer metadata entries; optimized for large sequential reads"),
        ("Replication (×3)",   "Tolerates 2 simultaneous DataNode failures"),
        ("Append-only writes", "Avoids concurrent write conflicts; simpler consistency"),
        ("Single NameNode",    "Simple consistency; HA via standby NameNode"),
        ("Direct data read",   "Client reads from DataNode directly (bypasses NameNode)"),
        ("Heartbeat check",    "DataNodes send heartbeats; NameNode detects failures"),
    ]
    for decision, reason in rows:
        print(f"  {decision:<26} {reason}")


if __name__ == "__main__":
    demonstrate_dfs()
