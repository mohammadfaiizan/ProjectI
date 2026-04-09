"""
DISTRIBUTED LOCKING PATTERNS
================================

Problem Statement:
Multiple service instances need to ensure mutual exclusion:
- Only ONE instance runs the daily billing job.
- Only ONE process claims a task from a shared queue.
- Prevent double-charging when two requests race for the same resource.
Local mutexes don't work across processes/machines. Need distributed lock.

Requirements for a Good Distributed Lock:
  Safety:   At most one holder at a time (mutual exclusion).
  Liveness: Lock can always eventually be acquired (no deadlock).
  Fault tolerance: Lock released if holder crashes (via TTL/lease).

Approaches:

  1. Redis SETNX + Expire (Simple):
     SET key token NX EX seconds
     NX = only set if not exists. EX = expiry TTL.
     Release: check token matches, then DEL (atomic via Lua script).
     Problem: single Redis node SPOF. Expiry too short → client still working.

  2. Redlock (Redis Multi-Node):
     Acquire lock on majority (≥N/2+1) of N Redis nodes.
     If majority acquired within timeout → lock held.
     Release from all nodes.
     Controversial (Martin Kleppmann argued it has subtle issues).
     More robust than single-node but not perfect for strict correctness.

  3. ZooKeeper Locks (Ephemeral Nodes):
     Create ephemeral sequential znode under /locks/resource-xxx.
     List children, find your predecessor.
     Watch predecessor for deletion. When deleted → you hold lock.
     Strong: ZAB protocol ensures consistency.
     Automatic release on session expiry (node death).
     Used by: HBase, Kafka, distributed job schedulers.

  4. etcd-Based Lock:
     Compare-and-swap on a key with a lease.
     Grant lease (with TTL) → key expires on lease expiry.
     Used by: Kubernetes, etcd-based leader election.

  5. Database Optimistic Locking:
     SELECT ... FOR UPDATE (row-level lock in PostgreSQL, MySQL).
     Check version field; reject update if version mismatch.
     Short-duration locks only (hold for one DB transaction).

Fencing Tokens:
  Every lock acquisition returns a monotonically increasing token.
  Client passes token on all writes to resources.
  Resource rejects writes with token < max seen token.
  Prevents "zombie leader" from writing after losing the lock.
  Fundamental for correctness even with Redlock.

Lock Design Principles:
  - Always set a TTL (prevents permanent lock on crash).
  - Store a unique token; verify before release (no token theft).
  - Use fencing tokens for resources that accept writes.
  - Keep lock hold duration minimal (acquire → critical section → release).
  - Prefer optimistic locking for low-contention scenarios.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import uuid
import threading
import random
import hashlib


# ─────────────────────────────────────────────
# REDIS-STYLE SETNX LOCK
# ─────────────────────────────────────────────

class RedisSimulator:
    """Simulates Redis SET NX EX and Lua script for atomic release."""

    def __init__(self):
        self._store     : Dict[str, Tuple[str, float]] = {}   # key → (token, expiry)
        self._lock      = threading.Lock()
        self._fence_seq  = 0

    def set_nx_ex(self, key: str, token: str, ttl_s: float) -> bool:
        """Atomic SET IF NOT EXISTS with expiry. Returns True if set."""
        with self._lock:
            now = time.time()
            existing = self._store.get(key)
            if existing and existing[1] > now:
                return False   # key exists and not expired
            self._fence_seq += 1
            self._store[key] = (token, now + ttl_s)
            return True

    def release(self, key: str, token: str) -> bool:
        """Atomic check-and-delete: only delete if token matches."""
        with self._lock:
            existing = self._store.get(key)
            if not existing or existing[0] != token:
                return False   # not our lock
            del self._store[key]
            return True

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            now = time.time()
            existing = self._store.get(key)
            if not existing or existing[1] < now:
                return None
            return existing[0]

    def fence_seq(self) -> int:
        return self._fence_seq


class DistributedLock:
    """
    Redis SETNX-based distributed lock with fencing token.
    """

    def __init__(self, redis: RedisSimulator, resource: str,
                 ttl_s: float = 10.0, retry_interval_s: float = 0.05,
                 max_wait_s: float = 5.0):
        self.redis          = redis
        self.resource       = resource
        self.ttl_s          = ttl_s
        self.retry_interval = retry_interval_s
        self.max_wait_s     = max_wait_s
        self._token         : Optional[str]  = None
        self._fencing_token : Optional[int]  = None

    def acquire(self) -> bool:
        """Blocking acquire with timeout."""
        token    = str(uuid.uuid4())
        deadline = time.time() + self.max_wait_s
        while time.time() < deadline:
            if self.redis.set_nx_ex(self.resource, token, self.ttl_s):
                self._token         = token
                self._fencing_token = self.redis.fence_seq()
                return True
            time.sleep(self.retry_interval)
        return False

    def release(self) -> bool:
        if not self._token:
            return False
        ok           = self.redis.release(self.resource, self._token)
        self._token  = None
        return ok

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock on {self.resource}")
        return self

    def __exit__(self, *_):
        self.release()

    @property
    def fencing_token(self) -> Optional[int]:
        return self._fencing_token


# ─────────────────────────────────────────────
# REDLOCK (Multi-Node)
# ─────────────────────────────────────────────

class RedlockAlgorithm:
    """
    Redlock: acquire lock on majority of N Redis nodes.
    If majority within validity time → lock held.
    """

    def __init__(self, nodes: List[RedisSimulator], ttl_s: float = 10.0):
        self.nodes  = nodes
        self.N      = len(nodes)
        self.ttl_s  = ttl_s
        self.quorum = self.N // 2 + 1

    def acquire(self, resource: str) -> Optional[Tuple[str, float]]:
        """Returns (token, validity_time) or None."""
        token    = str(uuid.uuid4())
        t0       = time.time()
        acks     = 0

        for node in self.nodes:
            if node.set_nx_ex(resource, token, self.ttl_s):
                acks += 1

        elapsed = time.time() - t0
        validity_time = self.ttl_s - elapsed

        if acks >= self.quorum and validity_time > 0:
            return token, validity_time
        else:
            # Release everywhere (cleanup)
            for node in self.nodes:
                node.release(resource, token)
            return None

    def release(self, resource: str, token: str):
        for node in self.nodes:
            node.release(resource, token)


# ─────────────────────────────────────────────
# ZOOKEEPER-STYLE LOCK (Ephemeral Nodes)
# ─────────────────────────────────────────────

class ZooKeeperLock:
    """
    ZooKeeper-style lock using sequential ephemeral nodes.
    Client with lowest sequence number holds the lock.
    Others watch the next-lower node for deletion.
    """

    def __init__(self, zk: "MockZooKeeper", path: str, client_id: str):
        self.zk         = zk
        self.path       = path
        self.client_id  = client_id
        self._node_path : Optional[str] = None
        self._held      = False

    def acquire(self) -> bool:
        self._node_path = self.zk.create_ephemeral_sequential(
            f"{self.path}/lock-", self.client_id)
        my_seq = int(self._node_path.split("-")[-1])

        while True:
            children = self.zk.get_children(self.path)
            seqs     = sorted(int(c.split("-")[-1]) for c in children)

            if seqs[0] == my_seq:
                self._held = True
                return True

            # Watch the node just before ours
            prev_seq  = max(s for s in seqs if s < my_seq)
            prev_node = f"{self.path}/lock-{prev_seq:08d}"
            exists    = self.zk.watch_and_wait(prev_node, timeout_s=2.0)
            if exists:
                continue
            # prev node deleted → we might be next, re-check

    def release(self):
        if self._node_path:
            self.zk.delete(self._node_path)
            self._held = False


class MockZooKeeper:
    def __init__(self):
        self._nodes  : Dict[str, str] = {}
        self._seq    = 0
        self._watchers: Dict[str, threading.Event] = {}
        self._lock   = threading.Lock()

    def create_ephemeral_sequential(self, path: str, value: str) -> str:
        with self._lock:
            self._seq += 1
            full_path = f"{path}{self._seq:08d}"
            self._nodes[full_path] = value
        return full_path

    def get_children(self, path: str) -> List[str]:
        with self._lock:
            return [k.split("/")[-1] for k in self._nodes
                    if k.startswith(path + "/")]

    def delete(self, path: str):
        with self._lock:
            self._nodes.pop(path, None)
            event = self._watchers.pop(path, None)
        if event:
            event.set()

    def watch_and_wait(self, path: str, timeout_s: float = 2.0) -> bool:
        """Wait for path to be deleted. Returns True if still exists."""
        event = threading.Event()
        with self._lock:
            if path not in self._nodes:
                return False
            self._watchers[path] = event
        return not event.wait(timeout=timeout_s)


# ─────────────────────────────────────────────
# FENCED WRITE RESOURCE
# ─────────────────────────────────────────────

class FencedSharedResource:
    """Shared resource that enforces fencing token ordering."""

    def __init__(self):
        self._max_token = 0
        self._data      = {}
        self.writes_ok  = 0
        self.writes_rejected = 0

    def write(self, key: str, value: Any, fencing_token: int) -> bool:
        if fencing_token < self._max_token:
            self.writes_rejected += 1
            return False
        if fencing_token > self._max_token:
            self._max_token = fencing_token
        self._data[key] = value
        self.writes_ok += 1
        return True


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_distributed_locking():
    print("=" * 65)
    print("DISTRIBUTED LOCKING PATTERNS")
    print("=" * 65)

    # ── Redis SETNX Lock ──────────────────────────
    print("\n[1] REDIS SETNX DISTRIBUTED LOCK")
    print("─" * 55)

    redis    = RedisSimulator()
    resource = FencedSharedResource()

    lock1 = DistributedLock(redis, "billing-job", ttl_s=5.0, max_wait_s=0.1)
    lock2 = DistributedLock(redis, "billing-job", ttl_s=5.0, max_wait_s=0.1)

    # Instance 1 acquires
    ok1 = lock1.acquire()
    print(f"  Instance 1 acquired: {ok1} (token={lock1._token[:8] if lock1._token else None})")

    # Instance 2 tries to acquire (should fail — already held)
    ok2 = lock2.acquire()
    print(f"  Instance 2 acquired: {ok2} (lock already held)")

    # Instance 1 writes with fencing token
    if ok1:
        resource.write("billing-run", "2024-01", lock1.fencing_token)
        print(f"  Instance 1 writes with fencing_token={lock1.fencing_token}")
        lock1.release()

    # Now instance 2 can acquire
    ok2b = lock2.acquire()
    print(f"  Instance 2 acquired after release: {ok2b} (token={lock2._token[:8] if lock2._token else None})")
    lock2.release()

    # Stale lock attempt: old fencing token rejected
    stale_ok = resource.write("billing-run", "duplicate", fencing_token=0)
    print(f"  Old client writes with fencing_token=0: {stale_ok} (rejected)")
    print(f"  Resource: writes_ok={resource.writes_ok} rejected={resource.writes_rejected}")

    # ── Redlock ───────────────────────────────────
    print("\n\n[2] REDLOCK — MULTI-NODE MAJORITY QUORUM")
    print("─" * 55)

    redis_nodes = [RedisSimulator() for _ in range(5)]
    redlock     = RedlockAlgorithm(redis_nodes, ttl_s=5.0)

    result1 = redlock.acquire("cron-job")
    print(f"  Client 1 Redlock acquire (5 nodes, quorum=3): {result1 is not None}")
    if result1:
        token, validity = result1
        print(f"  Token={token[:8]} validity={validity:.2f}s")

        result2 = redlock.acquire("cron-job")
        print(f"  Client 2 acquire while held: {result2 is not None}")

        redlock.release("cron-job", token)
        result3 = redlock.acquire("cron-job")
        print(f"  Client 2 acquire after release: {result3 is not None}")
        if result3:
            redlock.release("cron-job", result3[0])

    # ── ZooKeeper Lock ────────────────────────────
    print("\n\n[3] ZOOKEEPER SEQUENTIAL EPHEMERAL LOCK")
    print("─" * 55)

    zk   = MockZooKeeper()
    lockA = ZooKeeperLock(zk, "/locks/resource", "client-A")
    lockB = ZooKeeperLock(zk, "/locks/resource", "client-B")

    lockA.acquire()
    print(f"  Client A acquired: node={lockA._node_path}")

    # B waits in background
    b_acquired = threading.Event()
    def b_acquire():
        lockB.acquire()
        b_acquired.set()

    bt = threading.Thread(target=b_acquire, daemon=True)
    bt.start()
    time.sleep(0.1)
    children = zk.get_children("/locks/resource")
    print(f"  Client B waiting. ZK children: {children}")

    # A releases → B should acquire
    lockA.release()
    b_acquired.wait(timeout=1.0)
    print(f"  A released. B acquired: {lockB._held}")
    lockB.release()

    # ── Pattern Comparison ────────────────────────
    print("\n\n[4] DISTRIBUTED LOCK PATTERN COMPARISON")
    print("─" * 55)
    patterns = [
        ("Redis SETNX",     "Simple, fast",      "Single node SPOF",          "<1ms"),
        ("Redlock",         "Multi-node robust", "Subtle edge cases",          "~5ms"),
        ("ZooKeeper",       "Strong consistency", "Ops complexity, slower",    "~5-20ms"),
        ("etcd lease",      "Strong + TTL",       "etcd dependency",           "~5-10ms"),
        ("DB SELECT FOR UPDATE","No extra infra", "DB contention, no TTL",     "~1-10ms"),
        ("Optimistic lock", "No lock held",       "Retries on conflict",       "<1ms"),
    ]
    print(f"  {'Pattern':<24} {'Pro':<26} {'Con':<28} {'Latency'}")
    print(f"  {'─'*85}")
    for pattern, pro, con, latency in patterns:
        print(f"  {pattern:<24} {pro:<26} {con:<28} {latency}")

    print("\n\n[5] DISTRIBUTED LOCK DESIGN CHECKLIST")
    print("─" * 55)
    checklist = [
        "Always set TTL — prevents permanent lock if holder crashes",
        "Store unique token (UUID) — prevents token theft on release",
        "Use atomic acquire (SETNX) + atomic release (Lua/compare-and-delete)",
        "Keep critical section short — minimize time holding the lock",
        "Use fencing tokens for downstream resources",
        "Monitor lock acquisition latency — high latency = contention",
        "Prefer optimistic locking for low-contention, high-read workloads",
        "For strong guarantees: ZooKeeper/etcd over Redis",
    ]
    for item in checklist:
        print(f"  • {item}")


if __name__ == "__main__":
    demonstrate_distributed_locking()
