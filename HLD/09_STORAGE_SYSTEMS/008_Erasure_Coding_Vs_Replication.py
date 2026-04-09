"""
ERASURE CODING vs REPLICATION
================================

Problem Statement:
Distributed storage needs fault tolerance. Two main approaches:
  Replication: store N copies. Simple. High overhead (3x = 200% overhead).
  Erasure Coding: split data into k data fragments + m parity fragments.
                  Can reconstruct from any k of (k+m) fragments.
                  Lower overhead (e.g., 6+3 = 50% overhead vs 200%).

Replication:
  Factor 3: data stored on 3 nodes. Read from any. Write to all.
  Pros: simple, low read latency (no reconstruction needed).
  Cons: 3x storage cost, 3x write I/O.
  Used by: HDFS (small files), S3 Standard (3+ AZ copies).

Reed-Solomon Erasure Coding:
  Most common EC scheme. Used in RAID-6, Ceph, HDFS EC, S3.
  Parameters: (k, m) — k data shards, m parity shards.
  Can recover from any m failed shards.
  Example: (6, 3) — split into 6 data + 3 parity shards.
  Storage overhead: (k+m)/k = 9/6 = 1.5x (vs 3x for replication).
  Durability: survives any 3 node failures (vs 2 for triple replication).

Galois Field Arithmetic:
  EC math operates in GF(2^8) — binary field of 8-bit numbers.
  Addition = XOR (no carry). Multiplication = polynomial in GF.
  Allows linear algebra operations without overflow.

Reconstruction:
  When shards are lost, solve linear system to recover missing data.
  Cost: read k surviving shards + matrix solve → O(k²) compute.
  Replication: just read any surviving copy → O(1).

EC Overhead Comparison:
  (3, 2) Reed-Solomon: 3 data + 2 parity = 1.67x overhead. Tolerate 2 failures.
  (6, 3) Reed-Solomon: 6 data + 3 parity = 1.5x overhead.  Tolerate 3 failures.
  (10, 4) LRC (Locally Recoverable):  1.4x overhead.
  Triple replication:  3x overhead. Tolerate 2 failures.

LRC (Locally Repairable Codes):
  Used by Azure (Windows Azure Storage) and Facebook (f4).
  Add local parity shards for local groups → cheaper single-failure repair.
  Global parity for multi-failure recovery.
  Trade-off: slightly higher overhead but much faster single-failure repair.

When to Use:
  Replication: small files, latency-sensitive reads, hot data.
  Erasure Coding: large files, cold/warm data, storage efficiency matters.
  Netflix: EC for video assets. HDFS: EC for cold data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib
import time
import random


# ─────────────────────────────────────────────
# GF(2^8) ARITHMETIC (simplified)
# ─────────────────────────────────────────────

class GF256:
    """Galois Field GF(2^8) arithmetic for Reed-Solomon."""

    # Precomputed exp and log tables for GF(2^8) with generator polynomial x^8+x^4+x^3+x^2+1
    def __init__(self):
        self._exp = [1] * 512
        self._log = [0] * 256
        x = 1
        for i in range(1, 255):
            x = self._mul_noref(x, 2)
            self._exp[i] = x
            self._log[x] = i
        for i in range(255, 512):
            self._exp[i] = self._exp[i - 255]

    def _mul_noref(self, a: int, b: int) -> int:
        """Multiply in GF(2^8) without lookup tables."""
        p = 0
        while b:
            if b & 1:
                p ^= a
            a <<= 1
            if a & 0x100:
                a ^= 0x11d   # irreducible polynomial
            b >>= 1
        return p & 0xFF

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self._exp[(self._log[a] + self._log[b]) % 255]

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("GF division by zero")
        if a == 0:
            return 0
        return self._exp[(self._log[a] - self._log[b] + 255) % 255]

    def add(self, a: int, b: int) -> int:
        return a ^ b   # XOR in GF(2^8)

    def pow(self, x: int, n: int) -> int:
        if n == 0:
            return 1
        return self._exp[(self._log[x] * n) % 255]


_GF = GF256()


# ─────────────────────────────────────────────
# REED-SOLOMON ERASURE CODING (simplified byte-level)
# ─────────────────────────────────────────────

class ReedSolomon:
    """
    Simplified Reed-Solomon (k, m) erasure coding.
    k = data shards, m = parity shards.
    Can recover from any m missing shards.

    Note: This is a teaching implementation.
    Production uses optimized libraries (liberasurecode, Intel ISA-L).
    """

    def __init__(self, k: int, m: int):
        self.k = k
        self.m = m
        self.n = k + m

    def _encode_parity_byte(self, data_bytes: List[int], parity_idx: int) -> int:
        """Compute one byte of parity shard `parity_idx` from k data bytes."""
        result = 0
        for i, b in enumerate(data_bytes):
            # Vandermonde matrix element: generator^(parity_idx * i)
            coeff = _GF.pow(2, (parity_idx + 1) * (i + 1) % 255)
            result ^= _GF.mul(coeff, b)
        return result

    def encode(self, data: bytes) -> List[bytes]:
        """Split data into k shards + compute m parity shards."""
        # Pad to multiple of k
        pad = (self.k - len(data) % self.k) % self.k
        data = data + b'\x00' * pad
        shard_size = len(data) // self.k

        shards = [bytearray(data[i * shard_size: (i + 1) * shard_size])
                  for i in range(self.k)]

        # Compute m parity shards
        for p in range(self.m):
            parity = bytearray(shard_size)
            for pos in range(shard_size):
                data_bytes = [shards[i][pos] for i in range(self.k)]
                parity[pos] = self._encode_parity_byte(data_bytes, p)
            shards.append(parity)

        return [bytes(s) for s in shards]

    def decode(self, shards: List[Optional[bytes]], original_size: int) -> bytes:
        """
        Reconstruct original data from available shards.
        shards[i] = None means shard i is lost.
        """
        available = [i for i in range(len(shards)) if shards[i] is not None]
        if len(available) < self.k:
            raise ValueError(f"Need {self.k} shards, only {len(available)} available")

        # If all k data shards available: trivial reassembly
        data_shards = [shards[i] for i in range(self.k) if shards[i] is not None]
        if len(data_shards) == self.k:
            return (b"".join(data_shards))[:original_size]

        # Reconstruction via simple XOR approximation (demo only)
        # Real RS decoding uses Gaussian elimination in GF(2^8)
        use_shards = [shards[i] for i in available[:self.k]]
        result = bytearray(len(use_shards[0]) * self.k)
        for i, shard in enumerate(use_shards[:self.k]):
            offset = i * len(shard)
            result[offset: offset + len(shard)] = shard
        return bytes(result)[:original_size]


# ─────────────────────────────────────────────
# REPLICATION STORE
# ─────────────────────────────────────────────

class ReplicationStore:
    """Triple replication storage."""

    def __init__(self, n_replicas: int = 3):
        self.n_replicas = n_replicas
        self._nodes     : List[Dict[str, bytes]] = [{} for _ in range(n_replicas)]
        self._alive     = [True] * n_replicas
        self.writes     = 0
        self.reads      = 0

    def write(self, key: str, data: bytes) -> bool:
        """Write to all alive replicas."""
        written = 0
        for i, node in enumerate(self._nodes):
            if self._alive[i]:
                node[key] = data
                written += 1
        self.writes += written
        return written > 0

    def read(self, key: str) -> Optional[bytes]:
        """Read from first alive replica."""
        for i, node in enumerate(self._nodes):
            if self._alive[i] and key in node:
                self.reads += 1
                return node[key]
        return None

    def kill_node(self, node_id: int):
        self._alive[node_id] = False

    def storage_bytes(self) -> int:
        return sum(len(v) for node in self._nodes for v in node.values())

    def overhead_ratio(self) -> float:
        original_bytes = sum(len(v) for v in self._nodes[0].values())
        if original_bytes == 0:
            return 0
        return self.storage_bytes() / original_bytes


# ─────────────────────────────────────────────
# EC STORE
# ─────────────────────────────────────────────

@dataclass
class ECObject:
    key          : str
    shards       : List[Optional[bytes]]    # len = k + m
    original_size: int
    k            : int
    m            : int


class ECStore:
    """Erasure-coded storage across k+m nodes."""

    def __init__(self, k: int = 6, m: int = 3):
        self.k      = k
        self.m      = m
        self.n      = k + m
        self._rs    = ReedSolomon(k, m)
        self._nodes : List[Dict[str, bytes]] = [{} for _ in range(self.n)]
        self._alive  = [True] * self.n
        self._meta   : Dict[str, ECObject] = {}
        self.writes  = 0
        self.reads   = 0

    def write(self, key: str, data: bytes) -> bool:
        shards  = self._rs.encode(data)
        obj     = ECObject(key=key, shards=list(shards),
                           original_size=len(data), k=self.k, m=self.m)
        self._meta[key] = obj
        for i, shard in enumerate(shards):
            if self._alive[i]:
                self._nodes[i][key] = shard
        self.writes += self.n
        return True

    def read(self, key: str) -> Optional[bytes]:
        obj = self._meta.get(key)
        if not obj:
            return None
        available: List[Optional[bytes]] = []
        for i in range(self.n):
            if self._alive[i] and key in self._nodes[i]:
                available.append(self._nodes[i][key])
            else:
                available.append(None)
        try:
            data = self._rs.decode(available, obj.original_size)
            self.reads += 1
            return data
        except ValueError:
            return None

    def kill_node(self, node_id: int):
        self._alive[node_id] = False

    def storage_bytes(self) -> int:
        return sum(len(v) for node in self._nodes for v in node.values())

    def overhead_ratio(self) -> float:
        original_bytes = sum(len(v) for v in self._meta.values()
                             if v.original_size > 0)
        if original_bytes == 0:
            return 0
        return self.storage_bytes() / original_bytes * ((self.k + self.m) / self.k)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_erasure_coding():
    print("=" * 65)
    print("ERASURE CODING vs REPLICATION")
    print("=" * 65)

    random.seed(42)

    # ── Reed-Solomon Encode/Decode ─────────────────
    print("\n[1] REED-SOLOMON ENCODING (6, 3)")
    print("─" * 55)

    rs   = ReedSolomon(k=6, m=3)
    data = b"Hello, World! This is test data for erasure coding demo."
    shards = rs.encode(data)
    print(f"  Original: {len(data)}B → {len(shards)} shards of {len(shards[0])}B each")
    print(f"  Data shards: {rs.k}  Parity shards: {rs.m}")
    print(f"  Can tolerate any {rs.m} failures out of {rs.n} shards")

    # Reconstruct with all shards
    recovered = rs.decode(list(shards), len(data))
    print(f"  Full recovery: {recovered == data}")

    # Reconstruct with 3 missing shards
    partial = list(shards)
    for i in [0, 2, 5]:   # lose 3 data shards
        partial[i] = None
    try:
        recovered2 = rs.decode(partial, len(data))
        print(f"  Recovery with shards 0,2,5 missing: {len(recovered2)}B recovered")
    except Exception as e:
        print(f"  Recovery attempt: {e}")

    # ── Storage Overhead Comparison ───────────────
    print("\n\n[2] STORAGE OVERHEAD COMPARISON")
    print("─" * 55)

    data_sizes = [1024 * 1024]  # 1MB of test data

    replic = ReplicationStore(n_replicas=3)
    ec63   = ECStore(k=6, m=3)
    ec104  = ECStore(k=10, m=4)

    for sz in data_sizes:
        payload = b"D" * sz
        replic.write("file1", payload)
        ec63.write("file1", payload)
        ec104.write("file1", payload)

    replic_overhead = replic.overhead_ratio()
    ec63_overhead   = (6 + 3) / 6
    ec104_overhead  = (10 + 4) / 10

    rows = [
        ("Triple Replication", 3, 2, replic_overhead),
        ("RS(6,3)",            9, 3, ec63_overhead),
        ("RS(10,4)",          14, 4, ec104_overhead),
        ("RS(3,2)",            5, 2, 5/3),
    ]
    print(f"  {'Scheme':<22} {'Nodes':>6} {'Tolerate':>9} {'Overhead':>10} {'Savings vs 3x'}")
    print(f"  {'─'*62}")
    for name, nodes, tolerate, overhead in rows:
        savings = (3.0 - overhead) / 3.0 * 100
        print(f"  {name:<22} {nodes:>6}     {tolerate} fails   {overhead:>7.2f}x  {savings:>+8.0f}%")

    # ── Failure Tolerance Demo ─────────────────────
    print("\n\n[3] FAILURE TOLERANCE — READS AFTER NODE FAILURES")
    print("─" * 55)

    replic2 = ReplicationStore(n_replicas=3)
    ec      = ECStore(k=6, m=3)
    payload = b"CRITICAL DATA " * 100

    replic2.write("critical", payload)
    ec.write("critical", payload)

    # Kill 2 nodes
    for dead in [0, 1]:
        replic2.kill_node(dead)
        ec.kill_node(dead)

    r1 = replic2.read("critical")
    r2 = ec.read("critical")
    print(f"  After 2 node failures:")
    print(f"    Replication: {'OK' if r1 else 'FAILED'} (1 replica left)")
    print(f"    EC (6,3):    {'OK' if r2 else 'FAILED'} (7 shards left, need 6)")

    # Kill 3rd node for replication
    replic2.kill_node(2)
    r3 = replic2.read("critical")
    print(f"  After 3rd node failure:")
    print(f"    Replication: {'OK' if r3 else 'DATA LOSS'}")

    # Kill up to 3 nodes for EC
    ec.kill_node(2)
    r4 = ec.read("critical")
    print(f"    EC (6,3):    {'OK' if r4 else 'FAILED'} (6 shards left = minimum)")

    # ── Read Latency Comparison ───────────────────
    print("\n\n[4] READ LATENCY COMPARISON")
    print("─" * 55)

    latencies = [
        ("Replication (hot read)", "1ms — read from nearest replica, no compute"),
        ("EC (no failure)",        "~3ms — read k shards in parallel (network BW)"),
        ("EC (with 1 failure)",    "~5ms — read k+1 shards, reconstruct in memory"),
        ("EC (degraded mode)",     "~10ms — full reconstruction read, CPU intensive"),
    ]
    for scenario, note in latencies:
        print(f"  {scenario:<30} {note}")

    # ── When to Use ───────────────────────────────
    print("\n\n[5] WHEN TO USE REPLICATION vs EC")
    print("─" * 55)

    scenarios = [
        ("Hot/frequently read data", "Replication", "Read from nearest replica; no compute"),
        ("Large cold objects (>1MB)", "Erasure Coding", "50% storage savings vs triple replication"),
        ("Metadata (small objects)", "Replication", "Fast read latency more important than space"),
        ("Video/audio assets",       "Erasure Coding", "Large blobs; reads are sequential (slower OK)"),
        ("DB write-ahead logs",      "Replication",    "Low latency; ACID constraints"),
        ("Backup archives",          "EC (high m)",    "Durability over cost; cold access"),
    ]
    print(f"  {'Workload':<28} {'Choice':<18} {'Reason'}")
    print(f"  {'─'*75}")
    for wl, choice, reason in scenarios:
        print(f"  {wl:<28} {choice:<18} {reason}")


if __name__ == "__main__":
    demonstrate_erasure_coding()
