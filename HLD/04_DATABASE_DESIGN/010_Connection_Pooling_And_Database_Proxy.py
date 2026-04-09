"""
CONNECTION POOLING AND DATABASE PROXY
========================================

Problem Statement:
Each database connection consumes ~5-15 MB of RAM on the DB server and takes
50-200ms to establish. A 1000-instance app naively creating connections would
create 1000 DB connections — overwhelming the DB. Connection pooling reuses
connections efficiently and is non-negotiable in production.

Why Connection Pooling:
  - DB servers have limited connections (PostgreSQL default: 100)
  - Connection setup is expensive (TCP handshake + auth + session setup)
  - Idle connections still consume DB memory
  - Without pooling: each request creates+destroys connection → slow + OOM on DB

Pool Configuration:
  min_size : connections kept alive even when idle (warm pool)
  max_size : maximum connections (never exceed DB limit)
  timeout  : how long to wait for a connection from pool
  max_idle_time: close idle connections after this time

Connection Proxy (PgBouncer, ProxySQL):
  - Sits between app and DB
  - Manages a pool of real DB connections
  - Thousands of app connections → tens of DB connections
  - Session mode: assign DB connection for duration of client session
  - Transaction mode: assign DB connection only for transaction duration
  - Statement mode: assign per statement (fastest, limited)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import time
import threading
import queue
import random
import uuid


class PoolMode(Enum):
    SESSION     = "session"      # 1:1 app connection → DB connection
    TRANSACTION = "transaction"  # share DB connections per transaction
    STATEMENT   = "statement"    # share per statement (PgBouncer mode)


class ConnectionState(Enum):
    IDLE     = "idle"
    IN_USE   = "in_use"
    CLOSED   = "closed"
    CREATING = "creating"


@dataclass
class DBConnection:
    conn_id    : str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    state      : ConnectionState = ConnectionState.IDLE
    created_at : float = field(default_factory=time.time)
    last_used  : float = field(default_factory=time.time)
    queries_run: int = 0
    total_wait_ms: float = 0.0

    def execute(self, query: str) -> Dict:
        if self.state != ConnectionState.IN_USE:
            raise RuntimeError(f"Connection {self.conn_id} not acquired")
        self.queries_run += 1
        self.last_used   = time.time()
        latency = random.uniform(1, 10)
        time.sleep(latency / 1000)
        return {"query": query[:30], "rows": random.randint(0, 100),
                "latency_ms": round(latency, 2)}

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


# ─────────────────────────────────────────────
# CONNECTION POOL
# ─────────────────────────────────────────────

class ConnectionPool:
    """
    Thread-safe connection pool.
    Manages a set of reusable DB connections.
    """

    def __init__(self, db_url: str, min_size: int = 2, max_size: int = 10,
                 checkout_timeout_s: float = 5.0, max_idle_s: float = 300.0):
        self.db_url             = db_url
        self.min_size           = min_size
        self.max_size           = max_size
        self.checkout_timeout   = checkout_timeout_s
        self.max_idle_s         = max_idle_s
        self._pool              : List[DBConnection] = []
        self._in_use            : Set[str] = set()
        self._lock              = threading.Lock()
        self._available         = threading.Semaphore(max_size)
        self.total_checkouts    = 0
        self.total_waits        = 0
        self.total_wait_ms      = 0.0
        self.pool_exhausted     = 0

        # Pre-warm min connections
        for _ in range(min_size):
            self._pool.append(self._create_connection())

    def _create_connection(self) -> DBConnection:
        conn = DBConnection()
        conn.state = ConnectionState.IDLE
        time.sleep(0.002)   # simulate ~2ms connection setup
        return conn

    def acquire(self) -> Optional[DBConnection]:
        """Get a connection from the pool (blocks until available or timeout)."""
        start = time.time()
        if not self._available.acquire(timeout=self.checkout_timeout):
            self.pool_exhausted += 1
            return None   # timeout — pool exhausted

        wait_ms = (time.time() - start) * 1000

        with self._lock:
            # Find idle connection
            for conn in self._pool:
                if conn.state == ConnectionState.IDLE:
                    conn.state = ConnectionState.IN_USE
                    self._in_use.add(conn.conn_id)
                    self.total_checkouts += 1
                    if wait_ms > 1.0:
                        self.total_waits  += 1
                        self.total_wait_ms += wait_ms
                    return conn

            # Create new connection (up to max_size)
            if len(self._pool) < self.max_size:
                conn = self._create_connection()
                conn.state = ConnectionState.IN_USE
                self._pool.append(conn)
                self._in_use.add(conn.conn_id)
                self.total_checkouts += 1
                return conn

        self._available.release()
        return None

    def release(self, conn: DBConnection):
        """Return connection to pool."""
        with self._lock:
            conn.state    = ConnectionState.IDLE
            conn.last_used = time.time()
            self._in_use.discard(conn.conn_id)
        self._available.release()

    def evict_idle(self):
        """Close connections idle longer than max_idle_s (keep min_size)."""
        with self._lock:
            to_evict = [
                c for c in self._pool
                if (c.state == ConnectionState.IDLE and
                    c.idle_seconds > self.max_idle_s and
                    len([p for p in self._pool if p.state != ConnectionState.CLOSED]) > self.min_size)
            ]
            for conn in to_evict:
                conn.state = ConnectionState.CLOSED
                self._pool.remove(conn)

    @property
    def pool_size(self) -> int:
        return len([c for c in self._pool if c.state != ConnectionState.CLOSED])

    @property
    def idle_count(self) -> int:
        return len([c for c in self._pool if c.state == ConnectionState.IDLE])

    @property
    def in_use_count(self) -> int:
        return len(self._in_use)

    def report(self):
        avg_wait = (self.total_wait_ms / self.total_waits
                    if self.total_waits else 0)
        print(f"\n  ConnectionPool [{self.db_url}]:")
        print(f"    Pool size (min/max) : {self.min_size}/{self.max_size}")
        print(f"    Current pool size   : {self.pool_size}")
        print(f"    Idle connections    : {self.idle_count}")
        print(f"    In-use connections  : {self.in_use_count}")
        print(f"    Total checkouts     : {self.total_checkouts}")
        print(f"    Wait occurrences    : {self.total_waits}")
        print(f"    Avg wait (when wait): {avg_wait:.1f}ms")
        print(f"    Pool exhausted      : {self.pool_exhausted}")


# ─────────────────────────────────────────────
# PGBOUNCER-LIKE PROXY
# ─────────────────────────────────────────────

class DBProxy:
    """
    PgBouncer-like connection multiplexer.
    Many app connections → few actual DB connections.
    """

    def __init__(self, db_pool: ConnectionPool, mode: PoolMode = PoolMode.TRANSACTION):
        self.db_pool       = db_pool
        self.mode          = mode
        self._app_sessions : Dict[str, Optional[str]] = {}   # session_id → conn_id
        self.total_queries = 0
        self.connections_saved = 0

    def connect(self, session_id: str):
        """App connects to proxy — does NOT immediately create DB connection."""
        self._app_sessions[session_id] = None
        if self.mode == PoolMode.SESSION:
            # Session mode: hold real connection for session lifetime
            conn = self.db_pool.acquire()
            if conn:
                self._app_sessions[session_id] = conn.conn_id

    def query(self, session_id: str, sql: str) -> Optional[Dict]:
        """Execute query — borrows real DB connection."""
        self.total_queries += 1

        if self.mode == PoolMode.SESSION:
            # Find pre-assigned connection
            conn_id = self._app_sessions.get(session_id)
            conn = next((c for c in self.db_pool._pool if c.conn_id == conn_id), None)
            if conn:
                return conn.execute(sql)

        else:  # TRANSACTION or STATEMENT mode
            # Borrow connection just for this query
            conn = self.db_pool.acquire()
            if not conn:
                return {"error": "no connection available", "latency_ms": 0}
            try:
                result = conn.execute(sql)
                self.connections_saved += 1
                return result
            finally:
                self.db_pool.release(conn)

        return None

    def disconnect(self, session_id: str):
        if self.mode == PoolMode.SESSION:
            conn_id = self._app_sessions.get(session_id)
            if conn_id:
                conn = next((c for c in self.db_pool._pool if c.conn_id == conn_id), None)
                if conn:
                    self.db_pool.release(conn)
        del self._app_sessions[session_id]

    def report(self):
        print(f"\n  DBProxy [{self.mode.value} mode]:")
        print(f"    App sessions active   : {len(self._app_sessions)}")
        print(f"    Real DB connections   : {self.db_pool.pool_size}")
        print(f"    Total queries         : {self.total_queries}")
        ratio = len(self._app_sessions) / max(1, self.db_pool.pool_size)
        print(f"    Multiplexing ratio    : {ratio:.1f}x app:DB connections")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_connection_pooling():
    print("=" * 65)
    print("CONNECTION POOLING AND DATABASE PROXY")
    print("=" * 65)

    # ── No Pool (naive) ───────────────────────
    print("\n[1] WITHOUT CONNECTION POOL (naive — one conn per request)")
    print("─" * 55)
    print("  100 requests × 2ms setup + 5ms query = 700ms total")
    print("  1000 app instances × 1 conn = 1000 DB connections → DB OOM")
    print("  PostgreSQL default max_connections=100 → 900 connections rejected")

    # ── Connection Pool ────────────────────────
    print("\n\n[2] CONNECTION POOL (min=2, max=10)")
    print("─" * 55)
    pool = ConnectionPool("postgres://prod-db:5432/app", min_size=2, max_size=10,
                           checkout_timeout_s=3.0, max_idle_s=60.0)

    print(f"  Pre-warmed {pool.pool_size} connections (min_size={pool.min_size})")

    # Simulate concurrent requests
    results = []
    errors  = []

    def simulate_request(req_id: int):
        conn = pool.acquire()
        if not conn:
            errors.append(req_id)
            return
        try:
            conn.execute(f"SELECT * FROM users WHERE id = {req_id}")
            conn.execute(f"SELECT * FROM orders WHERE user_id = {req_id}")
        finally:
            pool.release(conn)
            results.append(req_id)

    threads = [threading.Thread(target=simulate_request, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"  20 concurrent requests: {len(results)} succeeded, {len(errors)} failed")
    pool.report()

    # ── PgBouncer Proxy ───────────────────────
    print("\n\n[3] PGBOUNCER-LIKE PROXY (transaction mode)")
    print("─" * 55)
    db_pool = ConnectionPool("postgres://prod-db:5432/app",
                              min_size=3, max_size=5,   # only 5 real DB connections
                              checkout_timeout_s=2.0)
    proxy = DBProxy(db_pool, mode=PoolMode.TRANSACTION)

    # 20 "app connections" multiplexed through 5 DB connections
    session_ids = [f"sess-{i}" for i in range(20)]
    for sid in session_ids:
        proxy.connect(sid)

    print(f"  20 app sessions connected through proxy")
    print(f"  Real DB connections: {db_pool.pool_size} (not 20!)")

    # Run queries
    for i, sid in enumerate(session_ids[:10]):
        result = proxy.query(sid, f"SELECT * FROM products WHERE id = {i}")
        if result and "error" not in result:
            pass

    for sid in session_ids:
        proxy.disconnect(sid)

    proxy.report()
    db_pool.report()

    # ── Pool Sizing Guide ─────────────────────
    print("\n\n[4] CONNECTION POOL SIZING GUIDE")
    print("─" * 55)
    print("  Formula: pool_size = (core_count × 2) + effective_spindle_count")
    print("  For SSD (no spindle): pool_size = core_count × 2 + 1")
    print()
    for cores in [2, 4, 8, 16]:
        recommended = cores * 2 + 1
        print(f"  {cores}-core DB server: recommended max connections ≈ {recommended}")

    print()
    print("  Per-service pool sizing:")
    print("  max_pool = DB_max_connections / number_of_services")
    print("  E.g.: 100 max_conn / 5 services = 20 connections per service")

    # ── Comparison ────────────────────────────
    print("\n\n[5] POOL MODE COMPARISON (PgBouncer)")
    print("─" * 55)
    rows = [
        ("Session mode",     "1:1 app:DB conn",     "Full PostgreSQL feature support",  "No multiplexing benefit"),
        ("Transaction mode", "Many:few multiplexed","Works for most apps (90% use case)","No SET, no advisory locks"),
        ("Statement mode",   "Per-statement share", "Maximum multiplexing",             "Very limited: no txns, no cursors"),
    ]
    print(f"  {'Mode':<20} {'Connection model':<25} {'Pros':<30} {'Cons'}")
    print(f"  {'─'*90}")
    for mode, model, pros, cons in rows:
        print(f"  {mode:<20} {model:<25} {pros:<30} {cons}")


if __name__ == "__main__":
    demonstrate_connection_pooling()
