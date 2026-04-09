"""
SESSION CACHE DESIGN
======================

Problem Statement:
HTTP is stateless. Web apps need to remember who you are across requests:
authentication state, shopping cart, user preferences, CSRF tokens.
Session data must be fast (every request reads it), available (HA needed),
and consistent (logged-in user must see their own writes).

Session Storage Options:

  1. Sticky Session (in-memory on app server):
     Session stored in app process memory. Load balancer must route user
     to same server (sticky). Simple but fails when server restarts.
     ✗ No HA: server crash = all sessions lost
     ✗ Couples load balancer to app servers

  2. Database-backed sessions:
     Session table in PostgreSQL/MySQL. Every request = DB read.
     ✓ Persistent, HA
     ✗ 10ms latency per request × 1000 QPS = 10K DB reads/sec

  3. Distributed Cache (Redis) — the standard:
     Session stored in Redis with TTL. App servers are stateless.
     ✓ Sub-ms reads, HA with Redis Sentinel/Cluster
     ✓ TTL-based expiry (no cleanup needed)
     ✓ Works with any number of app instances

Session Data:
  session_id → {user_id, email, roles, csrf_token, cart_id, created_at, last_seen}
  Stored as Redis HASH or serialized JSON.

Session Security:
  Session ID: 128-bit cryptographically random (not sequential)
  Cookie: HttpOnly, Secure, SameSite=Strict, Path=/
  Rotation: new session_id on privilege escalation (login, sudo)
  Absolute timeout: max session age regardless of activity (e.g., 24h)
  Idle timeout: expire if inactive for N minutes (e.g., 30 min)

Distributed Session Considerations:
  Replication lag: Redis async replication may lose recent session write
  → Use Redis Sentinel (sync replication) or accept brief inconsistency
  Session fixation attack: regenerate session_id after authentication
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time
import uuid
import random
import hashlib
import hmac
import json
from collections import defaultdict


# ─────────────────────────────────────────────
# SESSION DATA MODEL
# ─────────────────────────────────────────────

@dataclass
class Session:
    session_id   : str
    user_id      : Optional[str]
    email        : Optional[str]
    roles        : List[str]
    csrf_token   : str
    data         : Dict[str, Any]   # arbitrary session data (cart, prefs)
    created_at   : float = field(default_factory=time.time)
    last_seen_at : float = field(default_factory=time.time)
    ip_address   : str = ""
    user_agent   : str = ""

    def touch(self):
        self.last_seen_at = time.time()

    def idle_seconds(self) -> float:
        return time.time() - self.last_seen_at

    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def to_dict(self) -> Dict:
        return {
            "session_id"   : self.session_id,
            "user_id"      : self.user_id,
            "email"        : self.email,
            "roles"        : self.roles,
            "csrf_token"   : self.csrf_token,
            "data"         : self.data,
            "created_at"   : self.created_at,
            "last_seen_at" : self.last_seen_at,
            "ip_address"   : self.ip_address,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Session":
        return cls(
            session_id   = d["session_id"],
            user_id      = d.get("user_id"),
            email        = d.get("email"),
            roles        = d.get("roles", []),
            csrf_token   = d.get("csrf_token", ""),
            data         = d.get("data", {}),
            created_at   = d.get("created_at", time.time()),
            last_seen_at = d.get("last_seen_at", time.time()),
            ip_address   = d.get("ip_address", ""),
        )


# ─────────────────────────────────────────────
# SESSION ID GENERATOR
# ─────────────────────────────────────────────

class SessionIDGenerator:
    """Generates cryptographically secure session IDs."""

    @staticmethod
    def generate() -> str:
        """128-bit (16 bytes) random session ID, URL-safe."""
        return uuid.uuid4().hex + uuid.uuid4().hex[:8]   # 40 hex chars

    @staticmethod
    def is_valid_format(session_id: str) -> bool:
        return len(session_id) == 40 and all(c in "0123456789abcdef" for c in session_id)


# ─────────────────────────────────────────────
# REDIS SESSION STORE
# ─────────────────────────────────────────────

class RedisSessionStore:
    """
    Redis-backed session store.
    Key: session:{session_id}
    Value: JSON-serialized session data
    TTL: sliding (extended on access) + absolute max
    """

    def __init__(self, idle_timeout_s: float = 1800.0,
                 absolute_timeout_s: float = 86400.0):
        self._store      : Dict[str, str]   = {}   # session_id → JSON
        self._expires    : Dict[str, float] = {}
        self._abs_expires: Dict[str, float] = {}
        self.idle_timeout = idle_timeout_s
        self.abs_timeout  = absolute_timeout_s
        self.reads        = 0
        self.writes       = 0
        self.deletes      = 0

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def save(self, session: Session):
        """Store session with sliding TTL (re-set on each write)."""
        key = self._key(session.session_id)
        self._store[key]       = json.dumps(session.to_dict())
        self._expires[key]     = time.time() + self.idle_timeout
        if key not in self._abs_expires:
            self._abs_expires[key] = time.time() + self.abs_timeout
        self.writes += 1

    def load(self, session_id: str) -> Optional[Session]:
        """Load and slide the idle TTL."""
        key   = self._key(session_id)
        now   = time.time()
        self.reads += 1

        # Check absolute expiry
        if now > self._abs_expires.get(key, 0):
            self._evict(key)
            return None

        # Check idle expiry
        if now > self._expires.get(key, 0):
            self._evict(key)
            return None

        raw = self._store.get(key)
        if not raw:
            return None

        session = Session.from_dict(json.loads(raw))
        # Slide idle TTL
        self._expires[key] = now + self.idle_timeout
        return session

    def delete(self, session_id: str):
        """Explicit logout."""
        key = self._key(session_id)
        self._evict(key)
        self.deletes += 1

    def _evict(self, key: str):
        self._store.pop(key, None)
        self._expires.pop(key, None)
        self._abs_expires.pop(key, None)

    def active_count(self) -> int:
        now = time.time()
        return sum(1 for k, exp in self._expires.items()
                   if exp > now and self._abs_expires.get(k, 0) > now)

    def evict_expired(self) -> int:
        """Background sweep to reclaim memory."""
        now     = time.time()
        expired = [k for k, exp in self._expires.items() if exp <= now]
        for key in expired:
            self._evict(key)
        return len(expired)


# ─────────────────────────────────────────────
# SESSION MANAGER (Application Layer)
# ─────────────────────────────────────────────

class SessionManager:
    """
    Manages session lifecycle: create, load, update, rotate, destroy.
    """

    def __init__(self, store: RedisSessionStore, secret_key: str = "secret-key-32-bytes"):
        self.store      = store
        self.secret     = secret_key.encode()
        self._id_gen    = SessionIDGenerator()

    def _sign(self, session_id: str) -> str:
        """HMAC-sign session ID for cookie integrity."""
        sig = hmac.new(self.secret, session_id.encode(), hashlib.sha256).hexdigest()[:16]
        return f"{session_id}.{sig}"

    def _verify(self, signed_id: str) -> Optional[str]:
        """Verify signed session ID; return session_id or None."""
        parts = signed_id.rsplit(".", 1)
        if len(parts) != 2:
            return None
        session_id, sig = parts
        expected = hmac.new(self.secret, session_id.encode(), hashlib.sha256).hexdigest()[:16]
        if not hmac.compare_digest(sig, expected):
            return None
        return session_id

    def create_anonymous(self, ip: str = "", ua: str = "") -> Tuple[Session, str]:
        """Create an anonymous session (pre-login)."""
        session = Session(
            session_id = self._id_gen.generate(),
            user_id    = None, email=None, roles=[],
            csrf_token = self._id_gen.generate()[:32],
            data       = {},
            ip_address = ip, user_agent=ua
        )
        self.store.save(session)
        return session, self._sign(session.session_id)

    def authenticate(self, signed_id: str, user_id: str, email: str,
                     roles: List[str]) -> Optional[Tuple[Session, str]]:
        """
        Upgrade anonymous session to authenticated.
        ROTATES session_id to prevent session fixation attack.
        """
        session_id = self._verify(signed_id)
        if not session_id:
            return None
        old_session = self.store.load(session_id)
        if not old_session:
            return None

        # Destroy old session
        self.store.delete(session_id)

        # Create new session with new ID (session fixation prevention)
        new_session = Session(
            session_id = self._id_gen.generate(),
            user_id    = user_id, email=email, roles=roles,
            csrf_token = self._id_gen.generate()[:32],
            data       = old_session.data,   # preserve cart etc.
            ip_address = old_session.ip_address,
        )
        self.store.save(new_session)
        return new_session, self._sign(new_session.session_id)

    def get_session(self, signed_id: str) -> Optional[Session]:
        """Load and validate session from cookie value."""
        session_id = self._verify(signed_id)
        if not session_id:
            return None
        return self.store.load(session_id)

    def update_data(self, session: Session, key: str, value: Any):
        """Update session data field."""
        session.data[key] = value
        session.touch()
        self.store.save(session)

    def logout(self, signed_id: str):
        session_id = self._verify(signed_id)
        if session_id:
            self.store.delete(session_id)


    from typing import Tuple  # fix for Tuple type hint inside class


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_session_cache():
    print("=" * 65)
    print("SESSION CACHE DESIGN")
    print("=" * 65)

    random.seed(42)

    store   = RedisSessionStore(idle_timeout_s=1.0, absolute_timeout_s=5.0)
    manager = SessionManager(store)

    # ── Session Lifecycle ─────────────────────
    print("\n[1] SESSION LIFECYCLE")
    print("─" * 55)

    # Anonymous session
    anon_session, cookie = manager.create_anonymous(ip="192.168.1.1", ua="Chrome/121")
    print(f"  1. CREATE anonymous session")
    print(f"     session_id: {anon_session.session_id[:12]}...")
    print(f"     cookie:     {cookie[:20]}...  (signed)")
    print(f"     csrf_token: {anon_session.csrf_token[:12]}...")

    # Add cart item
    manager.update_data(anon_session, "cart", ["product:42", "product:7"])
    print(f"\n  2. ADD TO CART → session data updated")

    # Load session (as would happen on next request)
    loaded = manager.get_session(cookie)
    print(f"  3. LOAD session from cookie: user_id={loaded.user_id}  cart={loaded.data.get('cart')}")

    # Authenticate (login) → new session ID
    result = manager.authenticate(cookie, user_id="user-123",
                                   email="alice@example.com", roles=["user"])
    auth_session, new_cookie = result
    print(f"\n  4. AUTHENTICATE → session ID ROTATED (fixation prevention)")
    print(f"     old sid: {anon_session.session_id[:12]}...")
    print(f"     new sid: {auth_session.session_id[:12]}...")
    print(f"     user_id: {auth_session.user_id}  roles: {auth_session.roles}")
    print(f"     cart preserved: {auth_session.data.get('cart')}")

    # Verify old cookie is dead
    old_loaded = manager.get_session(cookie)
    print(f"\n  5. OLD cookie after rotation: {old_loaded}  (destroyed — security)")

    # Use new cookie
    auth_loaded = manager.get_session(new_cookie)
    print(f"  6. NEW cookie still valid: user={auth_loaded.email}  roles={auth_loaded.roles}")

    # Logout
    manager.logout(new_cookie)
    after_logout = manager.get_session(new_cookie)
    print(f"\n  7. LOGOUT → {after_logout}  (session deleted)")

    # ── Session ID Security ────────────────────
    print("\n\n[2] SESSION SECURITY")
    print("─" * 55)
    gen = SessionIDGenerator()
    for i in range(3):
        sid = gen.generate()
        print(f"  Generated session_id: {sid}  len={len(sid)} bits={len(sid)*4}")

    print(f"\n  HMAC-signed cookie: {new_cookie[:30]}...")
    print(f"  Tampered cookie: {new_cookie[:-5]}XXXXX")
    tampered = manager.get_session(new_cookie[:-5] + "XXXXX")
    print(f"  Tampered cookie result: {tampered}  (rejected by HMAC verification)")

    # ── Idle + Absolute Timeout ────────────────
    print("\n\n[3] IDLE AND ABSOLUTE TIMEOUT")
    print("─" * 55)
    s2, c2 = manager.create_anonymous()
    print(f"  Created session, idle_timeout=1s, abs_timeout=5s")

    for t in range(3):
        time.sleep(0.3)
        loaded = manager.get_session(c2)
        status = "valid" if loaded else "expired"
        print(f"  t={0.3*(t+1):.1f}s: {status}")

    time.sleep(0.9)   # > 1s idle timeout
    loaded = manager.get_session(c2)
    print(f"  t=1.9s (idle expired): {loaded}")

    # ── Active Sessions + Eviction ─────────────
    print("\n\n[4] ACTIVE SESSION TRACKING")
    print("─" * 55)
    store2  = RedisSessionStore(idle_timeout_s=0.2)
    manager2= SessionManager(store2)
    cookies = []

    # Create 10 sessions
    for i in range(10):
        _, c = manager2.create_anonymous(ip=f"10.0.0.{i}")
        cookies.append(c)

    print(f"  Created 10 sessions: active={store2.active_count()}")

    # Let some expire
    time.sleep(0.25)
    evicted = store2.evict_expired()
    print(f"  After 250ms (idle TTL=200ms): evicted={evicted}  active={store2.active_count()}")

    # ── Comparison Table ──────────────────────
    print("\n\n[5] SESSION STORAGE OPTIONS COMPARISON")
    print("─" * 55)
    options = [
        ("In-process (sticky)",  "sub-ms",    "No — server-specific", "Simple",   "✗ server crash = data loss"),
        ("Database (Postgres)",  "10-30ms",   "Yes",                  "Standard", "✗ DB reads on every request"),
        ("Redis (distributed)",  "0.5-2ms",   "Yes (Sentinel)",       "Standard", "✓ fast, HA, scalable"),
        ("JWT (stateless)",      "0ms",        "No store needed",      "Stateless","✗ can't revoke without blocklist"),
        ("Cookie (client-side)", "0ms",        "No store needed",      "Encrypted","✗ size limit, data in browser"),
    ]
    print(f"  {'Option':<24} {'Latency':<12} {'HA':<20} {'Pattern':<12} {'Note'}")
    print(f"  {'─'*80}")
    for opt, lat, ha, pat, note in options:
        print(f"  {opt:<24} {lat:<12} {ha:<20} {pat:<12} {note}")

    # ── Best Practices ─────────────────────────
    print("\n\n[6] SESSION CACHE BEST PRACTICES")
    print("─" * 55)
    practices = [
        ("Rotate session ID",    "After login, sudo, privilege escalation (prevent fixation)"),
        ("HMAC-sign cookie",     "Prevents tampering — server validates signature"),
        ("HttpOnly cookie",      "JS cannot read — prevents XSS session theft"),
        ("Secure flag",          "HTTPS only transmission — prevents sniffing"),
        ("SameSite=Strict",      "Prevents CSRF — cookie not sent cross-origin"),
        ("Short idle timeout",   "Inactive sessions expire: 30min for banking, 2h web"),
        ("Absolute timeout",     "Even active sessions expire: 8-24h prevents forever sessions"),
        ("Redis Sentinel",       "HA for session store — automatic failover in <30s"),
        ("Session data size",    "Keep small: user_id + roles only. Not full profile objects"),
        ("Encrypt sensitive data","PII in session encrypted at rest (Redis encryption)"),
    ]
    for practice, note in practices:
        print(f"  ✓ {practice:<24} {note}")


if __name__ == "__main__":
    demonstrate_session_cache()
