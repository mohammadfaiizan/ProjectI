"""
STATELESS VS STATEFUL SERVICES
================================

Problem Statement:
One of the most important architectural choices is whether to store state
inside the service process (stateful) or push state to an external store
(stateless). Stateless services are dramatically easier to scale horizontally.

Key Concepts:
- Stateless Service  : No per-request state stored in the process.
                       Any server can handle any request. Scales freely.
- Stateful Service   : Stores session/state in-process or local disk.
                       Requires sticky sessions or external state store.
- External State     : State in Redis/DB, accessible by any server instance.
- JWT (Stateless Auth): Session data encoded in a signed token; no server storage.
- Sticky Sessions    : Load balancer routes same user to same server (fragile).
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import hashlib
import time
import json
import base64


class SessionStorageType(Enum):
    IN_PROCESS = "in-process (stateful)"
    REDIS      = "redis (external, stateless)"
    JWT        = "jwt (client-side, stateless)"
    DATABASE   = "database (external, stateless)"


@dataclass
class SessionData:
    session_id  : str
    user_id     : str
    cart        : List[str]
    preferences : Dict[str, str]
    created_at  : float
    server_id   : Optional[str] = None    # which server created it


# ─────────────────────────────────────────────
# STATEFUL SERVICE (in-process state)
# ─────────────────────────────────────────────

class StatefulService:
    """Stores sessions in-process memory. Breaks under load balancing."""

    def __init__(self, server_id: str):
        self.server_id = server_id
        self._sessions: Dict[str, SessionData] = {}

    def create_session(self, user_id: str) -> str:
        sid = hashlib.md5(f"{user_id}{time.time()}".encode()).hexdigest()[:16]
        self._sessions[sid] = SessionData(
            session_id=sid, user_id=user_id, cart=[],
            preferences={}, created_at=time.time(), server_id=self.server_id
        )
        return sid

    def get_session(self, session_id: str) -> Optional[SessionData]:
        return self._sessions.get(session_id)

    def add_to_cart(self, session_id: str, item: str) -> bool:
        session = self._sessions.get(session_id)
        if session:
            session.cart.append(item)
            return True
        return False

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions


# ─────────────────────────────────────────────
# EXTERNAL STATE MANAGER (Redis-like)
# ─────────────────────────────────────────────

class ExternalStateStore:
    """Simulates a shared Redis instance accessible by all servers."""

    def __init__(self):
        self._store: Dict[str, dict] = {}

    def set(self, key: str, value: dict, ttl_s: float = 3600):
        self._store[key] = {"value": value, "expires_at": time.time() + ttl_s}

    def get(self, key: str) -> Optional[dict]:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() > entry["expires_at"]:
            del self._store[key]
            return None
        return entry["value"]

    def delete(self, key: str):
        self._store.pop(key, None)

    def size(self) -> int:
        return len(self._store)


class StatelessService:
    """Stores sessions in external Redis. Any server can handle any request."""

    def __init__(self, server_id: str, store: ExternalStateStore):
        self.server_id = server_id
        self.store     = store

    def create_session(self, user_id: str) -> str:
        sid = hashlib.md5(f"{user_id}{time.time()}".encode()).hexdigest()[:16]
        self.store.set(f"sess:{sid}", {
            "session_id": sid, "user_id": user_id,
            "cart": [], "preferences": {}, "server_id": self.server_id
        })
        return sid

    def get_session(self, session_id: str) -> Optional[dict]:
        return self.store.get(f"sess:{session_id}")

    def add_to_cart(self, session_id: str, item: str) -> bool:
        data = self.store.get(f"sess:{session_id}")
        if data is None:
            return False
        data["cart"].append(item)
        self.store.set(f"sess:{session_id}", data)
        return True


# ─────────────────────────────────────────────
# JWT SESSION (truly stateless)
# ─────────────────────────────────────────────

class JWTSessionManager:
    """
    Encodes session data inside a signed token.
    No server-side storage needed — scales infinitely.
    Downside: cannot revoke before expiry without a token blacklist.
    """

    def __init__(self, secret: str = "super-secret"):
        self.secret = secret

    def _sign(self, payload: str) -> str:
        return hashlib.sha256(f"{payload}{self.secret}".encode()).hexdigest()[:16]

    def create_token(self, user_id: str, cart: List[str]) -> str:
        payload = {"user_id": user_id, "cart": cart, "iat": int(time.time())}
        payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
        signature   = self._sign(payload_b64)
        return f"{payload_b64}.{signature}"

    def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload_b64, signature = token.rsplit(".", 1)
            if self._sign(payload_b64) != signature:
                return None
            return json.loads(base64.b64decode(payload_b64.encode()).decode())
        except Exception:
            return None


# ─────────────────────────────────────────────
# STICKY LOAD BALANCER
# ─────────────────────────────────────────────

class StickyLoadBalancer:
    """Routes each user to the same server (by session cookie). Fragile!"""

    def __init__(self, servers: List[StatefulService]):
        self.servers   = servers
        self._affinity : Dict[str, str] = {}  # session_id → server_id
        self._rr_idx   = 0

    def route(self, session_id: str) -> StatefulService:
        if session_id in self._affinity:
            sid = self._affinity[session_id]
            return next(s for s in self.servers if s.server_id == sid)
        # New session: round-robin assign
        server = self.servers[self._rr_idx % len(self.servers)]
        self._rr_idx += 1
        self._affinity[session_id] = server.server_id
        return server


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_stateless_vs_stateful():
    print("=" * 65)
    print("STATELESS VS STATEFUL SERVICES")
    print("=" * 65)

    # ── Stateful: the problem ─────────────────
    print("\n[1] STATEFUL SERVICE — THE PROBLEM")
    print("─" * 50)
    srv1 = StatefulService("server-1")
    srv2 = StatefulService("server-2")
    srv3 = StatefulService("server-3")

    sid = srv1.create_session("user-alice")
    srv1.add_to_cart(sid, "Laptop")
    srv1.add_to_cart(sid, "Mouse")
    print(f"  Session created on server-1: {sid}")
    print(f"  Cart on server-1: {srv1.get_session(sid).cart}")

    print("\n  Load balancer routes next request to server-2:")
    found = srv2.has_session(sid)
    print(f"  server-2 has session: {found}  ← 404/session lost!")

    print("\n  Load balancer routes next request to server-3:")
    found = srv3.has_session(sid)
    print(f"  server-3 has session: {found}  ← same problem!")

    # ── Sticky Sessions workaround ────────────
    print("\n[2] STICKY SESSIONS (workaround)")
    print("─" * 50)
    lb = StickyLoadBalancer([srv1, srv2, srv3])
    sid2 = srv1.create_session("user-bob")
    assigned_server = lb.route(sid2)
    print(f"  user-bob session → always routed to {assigned_server.server_id}")
    srv1.add_to_cart(sid2, "Keyboard")
    result = assigned_server.get_session(sid2)
    print(f"  Cart: {result.cart}")
    print("  ⚠  If server-1 dies, user-bob loses their session!")

    # ── Stateless: External State ─────────────
    print("\n[3] STATELESS WITH EXTERNAL STATE (Redis)")
    print("─" * 50)
    redis_store = ExternalStateStore()
    svc_a = StatelessService("server-a", redis_store)
    svc_b = StatelessService("server-b", redis_store)
    svc_c = StatelessService("server-c", redis_store)

    sid3 = svc_a.create_session("user-carol")
    svc_a.add_to_cart(sid3, "Monitor")
    svc_a.add_to_cart(sid3, "Webcam")
    print(f"  Session created on server-a: {sid3}")

    # Route to different servers — all work!
    for svc in [svc_a, svc_b, svc_c]:
        data = svc.get_session(sid3)
        print(f"  {svc.server_id} reads session: cart={data['cart']}")

    # ── JWT ───────────────────────────────────
    print("\n[4] JWT — TRULY STATELESS (no server storage)")
    print("─" * 50)
    jwt = JWTSessionManager(secret="my-secret-key")
    token = jwt.create_token("user-dave", cart=["Phone", "Case"])
    print(f"  JWT token (truncated): {token[:60]}…")

    # Any server can verify the token
    for i in range(1, 4):
        payload = jwt.verify_token(token)
        print(f"  server-{i} verifies token: user={payload['user_id']}  cart={payload['cart']}")

    # Tamper test
    tampered = token[:-5] + "XXXXX"
    payload  = jwt.verify_token(tampered)
    print(f"\n  Tampered token result: {payload}  ← rejected")

    # ── Comparison Summary ────────────────────
    print("\n[5] COMPARISON SUMMARY")
    print("─" * 50)
    print(f"  {'Aspect':<25} {'Stateful':<25} {'Stateless+Redis':<25} {'JWT'}")
    print(f"  {'─'*90}")
    rows = [
        ("Horizontal scale",  "❌ Sticky sessions",  "✅ Any server",       "✅ Any server"),
        ("Server failure",    "❌ Session lost",      "✅ Survives",         "✅ Token is safe"),
        ("Latency",           "✅ In-memory fast",   "⚠  Network to Redis", "✅ No network call"),
        ("Revocation",        "✅ Delete from memory","✅ Delete from Redis","❌ Wait for expiry"),
        ("State size limit",  "✅ Unlimited",         "⚠  RAM cost",        "❌ Token size limit"),
        ("Complexity",        "✅ Simplest",          "⚠  Need Redis",      "⚠  Key management"),
    ]
    for aspect, a, b, c in rows:
        print(f"  {aspect:<25} {a:<25} {b:<25} {c}")


if __name__ == "__main__":
    demonstrate_stateless_vs_stateful()
