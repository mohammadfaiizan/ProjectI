"""
REDIS ARCHITECTURE DEEP DIVE
================================

Problem Statement:
Redis is used for caching, sessions, pub/sub, rate limiting, leaderboards,
distributed locks, job queues, and real-time analytics. Understanding its
internals helps design systems that use it correctly and avoid common pitfalls.

Redis Internals:
  Single-threaded event loop (I/O multiplexing via epoll/kqueue).
  All commands are atomic by design — no per-command locks needed.
  In-memory data structure server.
  Persistence: RDB (snapshots) + AOF (append-only log).

Data Structures and Use Cases:
  STRING     : key → value (bytes). Cache, counters (INCR), locks (SETNX).
  HASH       : key → {field: value}. User objects, session data.
  LIST       : key → [v1, v2, ...]. Message queues (LPUSH/BRPOP), activity feeds.
  SET        : key → {v1, v2, v3} (unique). Tags, online users, union/intersection.
  SORTED SET : key → {v1:score, v2:score}. Leaderboards, rate limiting, priority queues.
  BITMAP     : bit array. User presence tracking, feature flags.
  HYPERLOGLOG: Probabilistic cardinality estimation. DAU counting.
  STREAM     : Append-only log. Event sourcing, message queue (Kafka-lite).
  GEO        : Geospatial index. Nearby search (Uber/Lyft style).

Key Patterns:
  Atomic counters  : INCR key → guaranteed atomic increment
  Distributed lock : SET lock:key uuid NX PX 30000 (SETNX + TTL atomically)
  Rate limiting    : INCR + EXPIRE (per-minute window counter)
  Pub/Sub          : PUBLISH channel msg / SUBSCRIBE channel
  Leaderboard      : ZADD scores user_id score / ZRANGE BY SCORE
  Job Queue        : LPUSH queue job / BRPOP queue 0 (blocking pop)

Persistence:
  RDB (snapshot): fork + dump to disk. Fast restart, data loss up to last snapshot.
  AOF (append-only): log every write command. Slowest but most durable.
  Combined: RDB for fast restart + AOF for durability (default recommendation).

Replication:
  Primary → Replica (async by default).
  WAIT command: block until N replicas acknowledge (synchronous for critical data).

High Availability:
  Redis Sentinel: monitors primaries, promotes replica on failure.
  Redis Cluster: shards data across nodes + replication per shard.

Memory Management:
  maxmemory + maxmemory-policy (LRU, LFU, volatile-ttl, allkeys-random).
  Memory fragmentation: USE jemalloc. Monitor memory_fragmentation_ratio.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
import time
import random
import threading
import heapq
from collections import defaultdict, deque


# ─────────────────────────────────────────────
# REDIS DATA STRUCTURES
# ─────────────────────────────────────────────

class RedisString:
    """STRING: get, set, incr, setnx, getset, expire."""

    def __init__(self):
        self._store   : Dict[str, Any]   = {}
        self._expires : Dict[str, float] = {}

    def set(self, key: str, value: Any, px: int = None, nx: bool = False) -> bool:
        """SET key value [NX] [PX ms]"""
        if nx and key in self._store and not self._is_expired(key):
            return False   # NX: only set if not exists
        self._store[key] = value
        if px:
            self._expires[key] = time.time() + px / 1000
        return True

    def get(self, key: str) -> Optional[Any]:
        if self._is_expired(key):
            return None
        return self._store.get(key)

    def incr(self, key: str) -> int:
        val = int(self.get(key) or 0) + 1
        self._store[key] = val
        return val

    def incrby(self, key: str, delta: int) -> int:
        val = int(self.get(key) or 0) + delta
        self._store[key] = val
        return val

    def expire(self, key: str, seconds: float):
        self._expires[key] = time.time() + seconds

    def ttl(self, key: str) -> float:
        if key not in self._expires:
            return -1.0
        remaining = self._expires[key] - time.time()
        return max(0.0, remaining)

    def _is_expired(self, key: str) -> bool:
        if key in self._expires and time.time() > self._expires[key]:
            del self._store[key]
            del self._expires[key]
            return True
        return False

    def delete(self, key: str):
        self._store.pop(key, None)
        self._expires.pop(key, None)


class RedisHash:
    """HASH: hset, hget, hmset, hgetall, hincrby."""

    def __init__(self):
        self._store : Dict[str, Dict[str, Any]] = defaultdict(dict)

    def hset(self, key: str, field: str, value: Any):
        self._store[key][field] = value

    def hget(self, key: str, field: str) -> Optional[Any]:
        return self._store.get(key, {}).get(field)

    def hmset(self, key: str, mapping: Dict[str, Any]):
        self._store[key].update(mapping)

    def hgetall(self, key: str) -> Dict[str, Any]:
        return dict(self._store.get(key, {}))

    def hincrby(self, key: str, field: str, delta: int) -> int:
        val = int(self._store[key].get(field, 0)) + delta
        self._store[key][field] = val
        return val

    def hdel(self, key: str, *fields):
        for f in fields:
            self._store[key].pop(f, None)


class RedisSortedSet:
    """SORTED SET: zadd, zrange, zrangebyscore, zrank, zincrby."""

    def __init__(self):
        self._data : Dict[str, Dict[str, float]] = defaultdict(dict)  # key → {member: score}

    def zadd(self, key: str, mapping: Dict[str, float]):
        self._data[key].update(mapping)

    def zincrby(self, key: str, member: str, delta: float) -> float:
        self._data[key][member] = self._data[key].get(member, 0.0) + delta
        return self._data[key][member]

    def zrangebyscore(self, key: str, min_score: float, max_score: float,
                       withscores: bool = False) -> List:
        items = [(m, s) for m, s in self._data.get(key, {}).items()
                 if min_score <= s <= max_score]
        items.sort(key=lambda x: x[1])
        if withscores:
            return items
        return [m for m, _ in items]

    def zrevrange(self, key: str, start: int, stop: int,
                  withscores: bool = False) -> List:
        items = sorted(self._data.get(key, {}).items(), key=lambda x: x[1], reverse=True)
        sliced = items[start:stop+1] if stop >= 0 else items[start:]
        if withscores:
            return sliced
        return [m for m, _ in sliced]

    def zrank(self, key: str, member: str) -> Optional[int]:
        items = sorted(self._data.get(key, {}).items(), key=lambda x: x[1])
        for i, (m, _) in enumerate(items):
            if m == member:
                return i
        return None

    def zcard(self, key: str) -> int:
        return len(self._data.get(key, {}))

    def zcount(self, key: str, min_score: float, max_score: float) -> int:
        return sum(1 for s in self._data.get(key, {}).values() if min_score <= s <= max_score)


class RedisList:
    """LIST: lpush, rpush, lpop, rpop, lrange, blpop."""

    def __init__(self):
        self._store : Dict[str, deque] = defaultdict(deque)
        self._event : Dict[str, threading.Event] = {}

    def lpush(self, key: str, *values) -> int:
        for v in values:
            self._store[key].appendleft(v)
        if key in self._event:
            self._event[key].set()
        return len(self._store[key])

    def rpush(self, key: str, *values) -> int:
        for v in values:
            self._store[key].append(v)
        return len(self._store[key])

    def lpop(self, key: str) -> Optional[Any]:
        if self._store[key]:
            return self._store[key].popleft()
        return None

    def rpop(self, key: str) -> Optional[Any]:
        if self._store[key]:
            return self._store[key].pop()
        return None

    def lrange(self, key: str, start: int, stop: int) -> List:
        items = list(self._store[key])
        return items[start:stop+1] if stop >= 0 else items[start:]

    def llen(self, key: str) -> int:
        return len(self._store[key])

    def brpop(self, key: str, timeout: float = 0.0) -> Optional[Tuple[str, Any]]:
        """Blocking pop — waits until item available."""
        deadline = time.time() + timeout if timeout else None
        while True:
            val = self.rpop(key)
            if val is not None:
                return (key, val)
            if deadline and time.time() > deadline:
                return None
            time.sleep(0.01)


class RedisSet:
    """SET: sadd, smembers, sismember, sunion, sinter."""

    def __init__(self):
        self._store : Dict[str, Set[Any]] = defaultdict(set)

    def sadd(self, key: str, *members) -> int:
        before = len(self._store[key])
        self._store[key].update(members)
        return len(self._store[key]) - before

    def smembers(self, key: str) -> Set:
        return set(self._store.get(key, set()))

    def sismember(self, key: str, member: Any) -> bool:
        return member in self._store.get(key, set())

    def scard(self, key: str) -> int:
        return len(self._store.get(key, set()))

    def sunion(self, *keys) -> Set:
        result = set()
        for key in keys:
            result |= self._store.get(key, set())
        return result

    def sinter(self, *keys) -> Set:
        if not keys:
            return set()
        result = set(self._store.get(keys[0], set()))
        for key in keys[1:]:
            result &= self._store.get(key, set())
        return result


# ─────────────────────────────────────────────
# REDIS PATTERNS
# ─────────────────────────────────────────────

class DistributedLock:
    """
    Redis distributed lock using SET NX PX.
    Lock key = unique token to prevent theft by different holder.
    """

    def __init__(self, redis_str: RedisString, key: str, ttl_ms: int = 30_000):
        self.redis   = redis_str
        self.key     = f"lock:{key}"
        self.ttl_ms  = ttl_ms
        self.token   = str(random.getrandbits(64))
        self.acquired= False

    def acquire(self) -> bool:
        self.acquired = self.redis.set(self.key, self.token, px=self.ttl_ms, nx=True)
        return self.acquired

    def release(self) -> bool:
        """Only release if we hold the lock (prevent stealing)."""
        current = self.redis.get(self.key)
        if current == self.token:
            self.redis.delete(self.key)
            self.acquired = False
            return True
        return False   # lock stolen or already expired


class RateLimiter:
    """
    Fixed-window rate limiter using Redis INCR + EXPIRE.
    Key: ratelimit:{user_id}:{window}
    """

    def __init__(self, redis_str: RedisString, limit: int, window_s: int = 60):
        self.redis    = redis_str
        self.limit    = limit
        self.window_s = window_s

    def is_allowed(self, user_id: str) -> Tuple[bool, int, int]:
        """Returns (allowed, current_count, remaining)."""
        window = int(time.time() / self.window_s)
        key    = f"ratelimit:{user_id}:{window}"
        count  = self.redis.incr(key)
        if count == 1:
            self.redis.expire(key, self.window_s)
        remaining = max(0, self.limit - count)
        return count <= self.limit, count, remaining


class Leaderboard:
    """Real-time leaderboard using Redis Sorted Set."""

    def __init__(self, zset: RedisSortedSet, board_name: str):
        self.zset  = zset
        self.key   = f"leaderboard:{board_name}"

    def add_score(self, user_id: str, points: float):
        self.zset.zincrby(self.key, user_id, points)

    def set_score(self, user_id: str, score: float):
        self.zset.zadd(self.key, {user_id: score})

    def top_n(self, n: int) -> List[Tuple[str, float]]:
        return self.zset.zrevrange(self.key, 0, n - 1, withscores=True)

    def rank(self, user_id: str) -> Optional[int]:
        r = self.zset.zrank(self.key, user_id)
        return r   # 0-indexed from lowest

    def total_players(self) -> int:
        return self.zset.zcard(self.key)


class JobQueue:
    """Simple FIFO job queue using Redis LIST (LPUSH / BRPOP pattern)."""

    def __init__(self, redis_list: RedisList, queue_name: str):
        self.rlist = redis_list
        self.key   = f"queue:{queue_name}"
        self.processed = 0

    def enqueue(self, job: Dict) -> int:
        import json
        return self.rlist.lpush(self.key, json.dumps(job))

    def dequeue(self, timeout_s: float = 1.0) -> Optional[Dict]:
        import json
        result = self.rlist.brpop(self.key, timeout=timeout_s)
        if result:
            self.processed += 1
            return json.loads(result[1])
        return None

    def queue_depth(self) -> int:
        return self.rlist.llen(self.key)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_redis_architecture():
    print("=" * 65)
    print("REDIS ARCHITECTURE DEEP DIVE")
    print("=" * 65)

    random.seed(42)

    # ── Data Structure: String ─────────────────
    print("\n[1] REDIS STRING — COUNTERS AND LOCKS")
    print("─" * 55)
    r_str = RedisString()

    # Counter
    for i in range(5):
        count = r_str.incr("page_views:homepage")
    print(f"  INCR page_views:homepage × 5 → {count}")

    # Rate limiting counter
    r_str.set("rate:user-1:window-1", 0)
    for i in range(3):
        c = r_str.incr("rate:user-1:window-1")
    r_str.expire("rate:user-1:window-1", 60)
    ttl = r_str.ttl("rate:user-1:window-1")
    print(f"  Rate limit counter: count={c}  TTL={ttl:.1f}s")

    # SETNX lock
    lock = DistributedLock(r_str, "order-process", ttl_ms=5000)
    acquired1 = lock.acquire()
    lock2     = DistributedLock(r_str, "order-process", ttl_ms=5000)
    acquired2 = lock2.acquire()
    print(f"  Lock acquire #1: {acquired1}  Lock acquire #2 (contended): {acquired2}")
    released  = lock.release()
    acquired3 = lock2.acquire()
    print(f"  After release #1: lock2 acquires: {acquired3}")

    # ── Sorted Set: Leaderboard ────────────────
    print("\n\n[2] SORTED SET — REAL-TIME LEADERBOARD")
    print("─" * 55)
    zset  = RedisSortedSet()
    board = Leaderboard(zset, "global")

    players = ["Alice", "Bob", "Carol", "Dave", "Eve"]
    for player in players:
        board.set_score(player, random.randint(100, 1000))
    board.add_score("Alice", 250)
    board.add_score("Bob", 150)

    print(f"  Top 5 players:")
    for rank, (player, score) in enumerate(board.top_n(5), 1):
        print(f"    #{rank} {player:<8} score={score:.0f}")

    alice_rank = board.rank("Alice")
    print(f"\n  Alice's rank (0-indexed from lowest): {alice_rank}")
    print(f"  Total players: {board.total_players()}")

    # ── Hash: Session ──────────────────────────
    print("\n\n[3] HASH — SESSION STORAGE")
    print("─" * 55)
    r_hash = RedisHash()
    session_data = {
        "user_id": "u-123", "email": "alice@ex.com",
        "roles": "user,admin", "cart_count": "3",
        "last_seen": str(time.time())
    }
    r_hash.hmset("session:abc123", session_data)
    print(f"  HMSET session:abc123 (5 fields)")
    user_id = r_hash.hget("session:abc123", "user_id")
    print(f"  HGET session:abc123 user_id → {user_id}")
    new_count = r_hash.hincrby("session:abc123", "cart_count", 2)
    print(f"  HINCRBY cart_count 2 → {new_count}")
    all_fields = r_hash.hgetall("session:abc123")
    print(f"  HGETALL → {len(all_fields)} fields")

    # ── List: Job Queue ────────────────────────
    print("\n\n[4] LIST — JOB QUEUE (LPUSH / BRPOP)")
    print("─" * 55)
    r_list = RedisList()
    queue  = JobQueue(r_list, "email")

    jobs = [{"to": f"user{i}@ex.com", "subject": f"Welcome {i}"} for i in range(5)]
    for job in jobs:
        depth = queue.enqueue(job)
    print(f"  Enqueued 5 email jobs. Queue depth: {depth}")

    # Worker processes jobs
    for _ in range(3):
        job = queue.dequeue(timeout_s=0.05)
        if job:
            print(f"  Processed: {job['subject']}")

    print(f"  Remaining in queue: {queue.queue_depth()}")
    print(f"  Total processed: {queue.processed}")

    # ── Rate Limiter ──────────────────────────
    print("\n\n[5] RATE LIMITER (INCR + EXPIRE)")
    print("─" * 55)
    r_rl     = RedisString()
    limiter  = RateLimiter(r_rl, limit=5, window_s=60)

    print(f"  Rate limit: 5 requests per 60s")
    for i in range(7):
        allowed, count, remaining = limiter.is_allowed("user-1")
        status = "✓" if allowed else "✗ RATE LIMITED"
        print(f"  Request #{i+1}: {status}  count={count}  remaining={remaining}")

    # ── Set: Online Users ─────────────────────
    print("\n\n[6] SET — ONLINE USER TRACKING")
    print("─" * 55)
    r_set = RedisSet()

    # Track online users per room
    r_set.sadd("online:room-1", "alice", "bob", "carol")
    r_set.sadd("online:room-2", "bob", "dave", "eve")

    room1 = r_set.smembers("online:room-1")
    print(f"  Room 1 online: {room1}")
    print(f"  Bob in room 1: {r_set.sismember('online:room-1', 'bob')}")

    # Users in both rooms
    both = r_set.sinter("online:room-1", "online:room-2")
    either = r_set.sunion("online:room-1", "online:room-2")
    print(f"  In both rooms: {both}")
    print(f"  In either room: {either}")

    # ── Architecture Summary ───────────────────
    print("\n\n[7] REDIS ARCHITECTURE SUMMARY")
    print("─" * 55)
    summary = [
        ("Single-threaded",      "Event loop processes all commands serially — no races"),
        ("In-memory",            "All data in RAM — ~100K-1M ops/sec on single node"),
        ("Persistence",          "RDB snapshots + AOF log — configure based on durability need"),
        ("Replication",          "Primary → Replica (async). WAIT for sync if needed"),
        ("HA (Sentinel)",        "Auto-failover in <30s. Minimum 3 sentinel nodes"),
        ("Cluster",              "Shards data across 3+ primaries. 16384 hash slots"),
        ("Pub/Sub",              "Decoupled message broadcast — at-most-once delivery"),
        ("Lua scripting",        "Atomic multi-command operations via EVAL"),
        ("Memory policy",        "allkeys-lru for pure cache; volatile-lru for mixed"),
        ("Key expiry",           "Lazy (on access) + active sweep (periodic background)"),
    ]
    for feature, note in summary:
        print(f"  {feature:<22} {note}")


if __name__ == "__main__":
    demonstrate_redis_architecture()
