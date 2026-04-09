"""
SESSION PERSISTENCE AND STICKY SESSIONS
==========================================

Problem Statement:
Some applications store user session state in-process (local memory). When a
subsequent request from the same user lands on a different server, the session
is not found and the user must re-authenticate. Sticky sessions solve this at
the LB level, but create hot spots and complicate deployments.

Sticky Session Methods:
  1. Cookie-based    → LB sets a cookie (e.g., SERVERID=web-2); subsequent
                       requests routed to web-2
  2. IP-based        → hash client IP to select backend (L4 friendly)
  3. URL rewriting   → session ID embedded in URL (legacy, not recommended)

Problems With Sticky Sessions:
  - Hot spots: popular users all on one server
  - Failover: if sticky server dies, sessions are lost
  - Scaling: can't freely add/remove servers
  - Blue-green deploys: hard to drain sticky users

Better Solution: Externalize Session State
  → Store sessions in Redis/Memcached
  → Any server can handle any request
  → LB can use any distribution algorithm without stickiness
  → Session survives server restarts
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib
import time
import random
import uuid


class StickyMethod(Enum):
    NONE         = "none"
    COOKIE       = "cookie"
    IP_HASH      = "ip_hash"
    URL_PARAM    = "url_param"


@dataclass
class Session:
    session_id  : str
    user_id     : str
    data        : Dict = field(default_factory=dict)
    created_at  : float = field(default_factory=time.time)
    expires_in_s: int = 3600

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.expires_in_s


@dataclass
class AppServer:
    server_id      : str
    local_sessions : Dict[str, Session] = field(default_factory=dict)
    requests_served: int = 0
    session_misses : int = 0

    def handle(self, session_id: Optional[str], user_id: str,
               action: str) -> Tuple[int, str]:
        self.requests_served += 1
        if session_id:
            sess = self.local_sessions.get(session_id)
            if sess and not sess.is_expired:
                sess.data["last_action"] = action
                return 200, f"OK: session found on {self.server_id}"
            else:
                self.session_misses += 1
                return 401, f"Session not found on {self.server_id} — login required"
        # New session
        new_sid = str(uuid.uuid4())[:12]
        self.local_sessions[new_sid] = Session(new_sid, user_id, {"last_action": action})
        return 200, f"New session {new_sid} created on {self.server_id}"


# ─────────────────────────────────────────────
# COOKIE-BASED STICKY LB
# ─────────────────────────────────────────────

class CookieStickyLB:
    """
    LB inserts a cookie (e.g., Set-Cookie: SERVERID=web-2) on first response.
    Subsequent requests with that cookie go to the same backend.
    """

    COOKIE_NAME = "SERVERID"

    def __init__(self, servers: List[AppServer]):
        self.servers      = servers
        self._cookie_map  : Dict[str, str] = {}   # cookie_val → server_id
        self._rr_index    = 0
        self.requests     = 0
        self.sticky_hits  = 0
        self.sticky_misses= 0

    def _pick_round_robin(self) -> AppServer:
        s = self.servers[self._rr_index % len(self.servers)]
        self._rr_index += 1
        return s

    def route(self, user_id: str, session_id: Optional[str] = None,
              sticky_cookie: Optional[str] = None) -> Tuple[AppServer, str, Optional[str]]:
        """
        Returns (server, response, new_cookie_to_set).
        """
        self.requests += 1
        new_cookie = None

        if sticky_cookie and sticky_cookie in self._cookie_map:
            target_id = self._cookie_map[sticky_cookie]
            server = next((s for s in self.servers if s.server_id == target_id), None)
            if server:
                self.sticky_hits += 1
                status, msg = server.handle(session_id, user_id, "action")
                return server, msg, None

        # No cookie / no match → pick new server
        self.sticky_misses += 1
        server = self._pick_round_robin()
        cookie_val = f"{server.server_id}-{uuid.uuid4().hex[:6]}"
        self._cookie_map[cookie_val] = server.server_id
        new_cookie = f"{self.COOKIE_NAME}={cookie_val}; Path=/; HttpOnly"
        status, msg = server.handle(session_id, user_id, "action")
        return server, msg, new_cookie

    def report(self):
        print(f"\n  CookieStickyLB stats:")
        print(f"    Total requests : {self.requests}")
        print(f"    Sticky hits    : {self.sticky_hits}")
        print(f"    Sticky misses  : {self.sticky_misses}")
        for s in self.servers:
            print(f"    {s.server_id}: {s.requests_served} requests  "
                  f"session_misses={s.session_misses}")


# ─────────────────────────────────────────────
# IP HASH STICKY LB
# ─────────────────────────────────────────────

class IPHashLB:
    def __init__(self, servers: List[AppServer]):
        self.servers  = servers
        self.requests = 0

    def route(self, client_ip: str) -> AppServer:
        self.requests += 1
        h = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return self.servers[h % len(self.servers)]


# ─────────────────────────────────────────────
# EXTERNAL SESSION STORE (Redis-like)
# ─────────────────────────────────────────────

class ExternalSessionStore:
    """
    Shared Redis-like session store.
    ANY server can find any session → no sticky sessions needed.
    """

    def __init__(self):
        self._store  : Dict[str, Session] = {}
        self.reads   = 0
        self.writes  = 0
        self.misses  = 0

    def get(self, session_id: str) -> Optional[Session]:
        self.reads += 1
        sess = self._store.get(session_id)
        if sess and not sess.is_expired:
            return sess
        self.misses += 1
        return None

    def set(self, session: Session):
        self.writes += 1
        self._store[session.session_id] = session

    def delete(self, session_id: str):
        self._store.pop(session_id, None)

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._store.values() if not s.is_expired)


class StatelessServer:
    """
    Server with no local session state.
    Always reads/writes sessions from external store.
    """

    def __init__(self, server_id: str, store: ExternalSessionStore):
        self.server_id      = server_id
        self.store          = store
        self.requests_served = 0

    def handle(self, session_id: Optional[str], user_id: str,
               action: str) -> Tuple[int, str]:
        self.requests_served += 1
        if session_id:
            sess = self.store.get(session_id)
            if sess:
                sess.data["last_action"] = action
                self.store.set(sess)
                return 200, f"OK: session {session_id[:8]} found via Redis ({self.server_id})"
            return 401, f"Session expired/not found — login required"
        # Create new session
        new_sid = str(uuid.uuid4())[:12]
        sess    = Session(new_sid, user_id, {"last_action": action})
        self.store.set(sess)
        return 200, f"New session {new_sid} → stored in Redis ({self.server_id})"


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_sticky_sessions():
    print("=" * 65)
    print("SESSION PERSISTENCE AND STICKY SESSIONS")
    print("=" * 65)
    random.seed(42)

    # ── Problem: No Stickiness ────────────────
    print("\n[1] PROBLEM: STATEFUL SERVERS WITHOUT STICKINESS")
    print("─" * 55)
    servers_raw = [AppServer(f"web-{i}") for i in range(1, 4)]
    # Manually add a session to web-1
    sid = "session-abc123"
    servers_raw[0].local_sessions[sid] = Session(sid, "alice", {"cart": [1, 2, 3]})

    print("  Session 'session-abc123' exists on web-1")
    print("  LB routes requests round-robin:")
    for i, srv in enumerate(servers_raw * 2):
        status, msg = srv.handle(sid, "alice", "view_cart")
        icon = "✅" if status == 200 else "❌"
        print(f"  Request {i+1}: {srv.server_id} → {icon} {msg}")

    # ── Cookie-based Sticky ───────────────────
    print("\n\n[2] COOKIE-BASED STICKY SESSIONS")
    print("─" * 55)
    sticky_servers = [AppServer(f"app-{i}") for i in range(1, 4)]
    cookie_lb = CookieStickyLB(sticky_servers)

    # First request — no cookie → gets assigned
    srv, msg, cookie = cookie_lb.route("alice")
    print(f"  1st request (alice, no cookie) → {srv.server_id}")
    print(f"  LB sets: {cookie}")

    # Subsequent requests with cookie → same server
    cookie_val = cookie.split("=")[1].split(";")[0] if cookie else None
    print(f"\n  Subsequent requests (alice, with SERVERID cookie):")
    for i in range(4):
        srv2, msg2, new_c = cookie_lb.route("alice", sticky_cookie=cookie_val)
        print(f"  Request {i+2}: → {srv2.server_id}  {'(sticky hit)' if not new_c else '(assigned)'}")

    # Different user — different server
    srv3, msg3, cookie3 = cookie_lb.route("bob")
    print(f"\n  1st request (bob, no cookie) → {srv3.server_id}")
    cookie_lb.report()

    # ── IP Hash Sticky ────────────────────────
    print("\n\n[3] IP HASH STICKY (no cookie needed)")
    print("─" * 55)
    ip_servers = [AppServer(f"node-{i}") for i in range(1, 4)]
    ip_lb = IPHashLB(ip_servers)
    test_ips = ["10.0.1.1", "10.0.1.2", "10.0.1.1", "10.0.1.3", "10.0.1.1"]
    print("  Same IP always → same server:")
    for ip in test_ips:
        srv = ip_lb.route(ip)
        print(f"  {ip} → {srv.server_id}")

    # ── External Session Store ────────────────
    print("\n\n[4] BETTER: EXTERNAL SESSION STORE (Redis)")
    print("─" * 55)
    store = ExternalSessionStore()
    stateless = [StatelessServer(f"svc-{i}", store) for i in range(1, 4)]

    # Create session on svc-1
    status, msg = stateless[0].handle(None, "alice", "login")
    print(f"  Login: {msg}")
    # Extract session id from message
    sid_ext = msg.split("New session ")[1].split(" ")[0]

    # Subsequent requests hit different servers — all find session in Redis
    print(f"\n  Subsequent requests (session in Redis):")
    for i, srv in enumerate(stateless * 2):
        status, msg = srv.handle(sid_ext, "alice", f"action_{i}")
        icon = "✅" if status == 200 else "❌"
        print(f"  {srv.server_id}: {icon} {msg}")

    print(f"\n  Redis stats: reads={store.reads}  writes={store.writes}  "
          f"misses={store.misses}  active_sessions={store.active_count}")

    # ── Comparison ────────────────────────────
    print("\n\n[5] STICKY SESSIONS vs EXTERNAL STORE")
    print("─" * 55)
    rows = [
        ("Server failure",      "Sessions lost",             "Sessions survive (in Redis)"),
        ("Scaling out",         "Complex — re-balance",      "Simple — all servers equal"),
        ("Hot spots",           "Yes — popular users cluster","No — any server handles any req"),
        ("Deployments",         "Must drain sticky users",   "Rolling deploys trivial"),
        ("Latency",             "No extra hop",              "+1ms for Redis read"),
        ("Infrastructure",      "No extra deps",             "Requires Redis cluster"),
        ("Session sharing",     "No (only on one server)",   "Yes (microservices share)"),
        ("Recommendation",      "Legacy/simple apps only",   "All modern architectures"),
    ]
    print(f"  {'Aspect':<25} {'Sticky Sessions':<30} {'External Store (Redis)'}")
    print(f"  {'─'*80}")
    for aspect, sticky, ext in rows:
        print(f"  {aspect:<25} {sticky:<30} {ext}")


if __name__ == "__main__":
    demonstrate_sticky_sessions()
