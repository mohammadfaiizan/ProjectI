"""
JWT AND SESSION MANAGEMENT
============================

Problem Statement:
After a user logs in, how do we identify them on subsequent requests
without making them re-authenticate every time?
Two main approaches: stateful sessions vs stateless JWTs.

Stateful Sessions:
  Server stores session data in memory or database.
  Client gets an opaque session ID (cookie).
  Every request: DB/cache lookup for session.
  Pros: revocable instantly. Easy logout. Small cookie.
  Cons: scaling requires shared session store (Redis).
        Session store becomes a bottleneck/SPOF.

Stateless JWT:
  Server generates signed token with embedded claims.
  Client stores in localStorage or httpOnly cookie.
  Every request: verify signature + expiry (no DB lookup).
  Pros: no server state. Scales horizontally.
  Cons: can't revoke before expiry. Claims can become stale.
        Larger than session cookie (typical: 300-500 bytes).

JWT Structure:
  header.payload.signature (base64url encoded)
  Header: {"alg": "RS256", "typ": "JWT", "kid": "key-id"}
  Payload: {"sub": "user-id", "iat": 1704000000, "exp": 1704003600}
  Signature: RSASHA256(header.payload, private_key)

Common JWT Pitfalls:
  alg=none attack: strip signature, set alg=none → always valid.
  RS256 → HS256 confusion: server uses public key as HMAC secret.
  Missing exp validation: expired tokens accepted.
  Storing in localStorage: XSS can steal tokens.
  Long expiry: 30-day JWT can't be revoked if key rotates.

Session Security:
  httpOnly: JS can't read cookie → XSS-safe.
  Secure: HTTPS only.
  SameSite=Strict/Lax: CSRF protection.
  Rotate session ID on privilege escalation (login → admin).

Hybrid Approach:
  Short-lived JWT (15min) + refresh token in httpOnly cookie.
  JWT for stateless API calls; refresh token for long sessions.
  Revoke refresh token to effectively log out user.

Token Storage:
  Memory (React state): cleared on tab close. XSS risk still possible.
  httpOnly cookie: best for session tokens. No JS access.
  localStorage: XSS risk. OK for low-sensitivity read tokens.
  Secure memory (mobile keychain): OS-protected storage.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import hmac
import time
import uuid
import json
import base64
import secrets


# ─────────────────────────────────────────────
# SESSION STORE (stateful)
# ─────────────────────────────────────────────

@dataclass
class Session:
    session_id : str
    user_id    : str
    created_at : float
    last_seen  : float
    ip_address : str
    user_agent : str
    data       : Dict[str, Any] = field(default_factory=dict)
    invalidated: bool = False

    def is_expired(self, ttl_s: int = 3600) -> bool:
        return (time.time() - self.last_seen) > ttl_s or self.invalidated

    def touch(self):
        self.last_seen = time.time()


class SessionStore:
    """
    Server-side session store (like Redis-backed sessions).
    Supports sliding window expiry, multi-device, forced logout.
    """

    def __init__(self, ttl_s: int = 3600, max_sessions_per_user: int = 10):
        self._sessions   : Dict[str, Session] = {}
        self._user_index : Dict[str, Set[str]] = {}   # user_id → session_ids
        self.ttl_s       = ttl_s
        self.max_sessions = max_sessions_per_user

    def create(self, user_id: str, ip: str, user_agent: str) -> Session:
        session_id = secrets.token_urlsafe(32)
        session    = Session(
            session_id=session_id, user_id=user_id,
            created_at=time.time(), last_seen=time.time(),
            ip_address=ip, user_agent=user_agent,
        )
        self._sessions[session_id] = session
        if user_id not in self._user_index:
            self._user_index[user_id] = set()
        self._user_index[user_id].add(session_id)
        self._enforce_max_sessions(user_id)
        return session

    def get(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if not session or session.is_expired(self.ttl_s):
            return None
        session.touch()
        return session

    def invalidate(self, session_id: str):
        session = self._sessions.get(session_id)
        if session:
            session.invalidated = True
            self._user_index.get(session.user_id, set()).discard(session_id)

    def invalidate_all_for_user(self, user_id: str):
        """Force logout all devices."""
        for sid in list(self._user_index.get(user_id, set())):
            self.invalidate(sid)

    def rotate(self, old_session_id: str) -> Optional[Session]:
        """Generate new session ID, invalidate old (on privilege change)."""
        old = self._sessions.get(old_session_id)
        if not old or old.is_expired(self.ttl_s):
            return None
        new_session = self.create(old.user_id, old.ip_address, old.user_agent)
        new_session.data = dict(old.data)
        self.invalidate(old_session_id)
        return new_session

    def user_sessions(self, user_id: str) -> List[Session]:
        return [self._sessions[sid] for sid in self._user_index.get(user_id, set())
                if not self._sessions[sid].is_expired(self.ttl_s)]

    def gc(self) -> int:
        """Remove expired sessions. Returns count."""
        expired = [sid for sid, s in self._sessions.items()
                   if s.is_expired(self.ttl_s)]
        for sid in expired:
            uid = self._sessions[sid].user_id
            self._user_index.get(uid, set()).discard(sid)
            del self._sessions[sid]
        return len(expired)

    def _enforce_max_sessions(self, user_id: str):
        sessions = sorted(
            [self._sessions[sid] for sid in self._user_index.get(user_id, set())
             if not self._sessions[sid].is_expired(self.ttl_s)],
            key=lambda s: s.created_at
        )
        while len(sessions) > self.max_sessions:
            oldest = sessions.pop(0)
            self.invalidate(oldest.session_id)


# ─────────────────────────────────────────────
# JWT SERVICE (with key rotation support)
# ─────────────────────────────────────────────

@dataclass
class JWTKey:
    key_id    : str
    secret    : bytes
    created_at: float = field(default_factory=time.time)
    active    : bool = True


class JWTService:
    """
    JWT with key rotation (kid in header).
    Multiple keys supported for rotation without downtime.
    """

    def __init__(self):
        self._keys   : Dict[str, JWTKey] = {}
        self._active_kid: Optional[str] = None

    def add_key(self, key_id: str, secret: str, make_active: bool = True):
        self._keys[key_id] = JWTKey(key_id=key_id, secret=secret.encode())
        if make_active:
            self._active_kid = key_id

    def rotate_key(self, new_key_id: str, new_secret: str):
        """Add new signing key; old keys still verify existing tokens."""
        self.add_key(new_key_id, new_secret, make_active=True)
        # Old key remains for verification of existing tokens

    def _b64(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    def _unb64(self, s: str) -> bytes:
        return base64.urlsafe_b64decode(s + "==")

    def sign(self, payload: Dict, expires_in_s: int = 3600) -> str:
        if not self._active_kid:
            raise RuntimeError("No active signing key")
        key = self._keys[self._active_kid]
        payload.update({"iat": int(time.time()), "exp": int(time.time()) + expires_in_s})
        header = self._b64(json.dumps({"alg": "HS256", "typ": "JWT",
                                         "kid": self._active_kid}).encode())
        body   = self._b64(json.dumps(payload).encode())
        sig    = self._b64(hmac.new(key.secret, f"{header}.{body}".encode(),
                                     hashlib.sha256).digest())
        return f"{header}.{body}.{sig}"

    def verify(self, token: str) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            h_str, b_str, sig_str = token.split(".")
            header = json.loads(self._unb64(h_str))
            # alg=none attack prevention
            if header.get("alg") == "none":
                return None, "alg_none_not_allowed"

            kid  = header.get("kid")
            key  = self._keys.get(kid)
            if not key:
                return None, f"unknown_kid_{kid}"

            expected = self._b64(hmac.new(key.secret, f"{h_str}.{b_str}".encode(),
                                           hashlib.sha256).digest())
            if not hmac.compare_digest(sig_str, expected):
                return None, "invalid_signature"

            payload = json.loads(self._unb64(b_str))
            if payload.get("exp", 0) < time.time():
                return None, "token_expired"
            return payload, None
        except Exception as e:
            return None, f"parse_error:{e}"


# ─────────────────────────────────────────────
# TOKEN BLACKLIST (for revocation)
# ─────────────────────────────────────────────

class TokenBlacklist:
    """
    Stores revoked JWT JTI (JWT ID) values until they expire.
    Allows JWT revocation without full session store.
    """

    def __init__(self):
        self._revoked: Dict[str, float] = {}  # jti → exp_timestamp

    def revoke(self, jti: str, exp: float):
        self._revoked[jti] = exp

    def is_revoked(self, jti: str) -> bool:
        return jti in self._revoked

    def gc(self) -> int:
        now   = time.time()
        stale = [jti for jti, exp in self._revoked.items() if exp < now]
        for jti in stale:
            del self._revoked[jti]
        return len(stale)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_sessions_jwt():
    print("=" * 65)
    print("JWT AND SESSION MANAGEMENT")
    print("=" * 65)

    # ── Stateful Session Store ─────────────────────
    print("\n[1] STATEFUL SESSION STORE")
    print("─" * 55)

    store = SessionStore(ttl_s=3600, max_sessions_per_user=3)
    s1 = store.create("alice", "192.168.1.1", "Chrome/120")
    s2 = store.create("alice", "10.0.0.1",   "Firefox/121")
    print(f"  Created session 1: {s1.session_id[:16]}... ip={s1.ip_address}")
    print(f"  Created session 2: {s2.session_id[:16]}... ip={s2.ip_address}")

    fetched = store.get(s1.session_id)
    print(f"  Get session 1: found={fetched is not None}")

    # Session rotation on privilege change
    rotated = store.rotate(s1.session_id)
    print(f"  Rotated session:  new_id={rotated.session_id[:16]}...")
    print(f"  Old session valid: {store.get(s1.session_id) is not None}")

    # Force logout all devices
    print(f"  Active sessions before logout: {len(store.user_sessions('alice'))}")
    store.invalidate_all_for_user("alice")
    print(f"  Active sessions after logout:  {len(store.user_sessions('alice'))}")

    # ── JWT with Key Rotation ─────────────────────
    print("\n\n[2] JWT WITH KEY ROTATION")
    print("─" * 55)

    jwt_svc = JWTService()
    jwt_svc.add_key("key-2024-01", "secret-jan-2024")

    token_v1 = jwt_svc.sign({"sub": "user-42", "roles": ["editor"]},
                              expires_in_s=7200)
    print(f"  Token signed with key-2024-01: {token_v1[:40]}...")
    payload, err = jwt_svc.verify(token_v1)
    print(f"  Verify: sub={payload['sub']} roles={payload.get('roles')} err={err}")

    # Rotate key
    jwt_svc.rotate_key("key-2024-02", "secret-feb-2024-different")
    token_v2 = jwt_svc.sign({"sub": "user-42", "roles": ["editor"]})
    print(f"  After rotation — new token: {token_v2[:40]}...")

    # Old token still verifiable (key still in store)
    p_old, e_old = jwt_svc.verify(token_v1)
    p_new, e_new = jwt_svc.verify(token_v2)
    print(f"  Old token (key-2024-01) still valid: {e_old is None}")
    print(f"  New token (key-2024-02) valid: {e_new is None}")

    # alg=none attack prevention
    header_none = base64.urlsafe_b64encode(
        json.dumps({"alg": "none", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()
    body_fake   = base64.urlsafe_b64encode(
        json.dumps({"sub": "admin", "roles": ["admin"]}).encode()
    ).rstrip(b"=").decode()
    forged = f"{header_none}.{body_fake}."
    _, err_forged = jwt_svc.verify(forged)
    print(f"  alg=none forged token rejected: {err_forged}")

    # ── Token Blacklist ────────────────────────────
    print("\n\n[3] JWT REVOCATION VIA BLACKLIST")
    print("─" * 55)

    blacklist = TokenBlacklist()
    payload_with_jti = {"sub": "user-42", "jti": "unique-token-id-abc"}
    revocable_token  = jwt_svc.sign(payload_with_jti, expires_in_s=3600)

    before_revoke, _ = jwt_svc.verify(revocable_token)
    print(f"  Before revoke: valid={before_revoke is not None}")

    blacklist.revoke("unique-token-id-abc", time.time() + 3600)
    jti_revoked = blacklist.is_revoked("unique-token-id-abc")
    print(f"  After revoke: jti blacklisted={jti_revoked}")
    print(f"  (Full validation must check blacklist after signature verify)")

    # ── Comparison ────────────────────────────────
    print("\n\n[4] SESSION vs JWT COMPARISON")
    print("─" * 55)

    rows = [
        ("Server state",      "Required (Redis/DB)",      "Not needed"),
        ("Scalability",       "Shared session store",      "Stateless, scales well"),
        ("Revocation",        "Instant (delete session)",  "Requires blacklist or short TTL"),
        ("Claims freshness",  "Always current",            "Stale until expiry"),
        ("Token size",        "Small (session ID only)",   "~300-500 bytes"),
        ("DB lookup/request", "Yes",                       "No (just crypto verify)"),
        ("XSS safety",        "httpOnly cookie",           "httpOnly cookie (not localStorage)"),
        ("CSRF",              "SameSite + CSRF token",     "Bearer header (CSRF-safe)"),
        ("Best for",          "Traditional web apps",      "Microservices, APIs, SPAs"),
    ]
    print(f"  {'Property':<24} {'Session':<28} {'JWT'}")
    print(f"  {'─'*74}")
    for prop, session, jwt in rows:
        print(f"  {prop:<24} {session:<28} {jwt}")

    # ── Cookie Security Attributes ────────────────
    print("\n\n[5] SECURE COOKIE CONFIGURATION")
    print("─" * 55)

    example_cookie = {
        "name"    : "session_id",
        "value"   : "opaque-session-token",
        "HttpOnly": True,        # No JS access → XSS safe
        "Secure"  : True,        # HTTPS only
        "SameSite": "Strict",    # No cross-site sending → CSRF protection
        "Path"    : "/",
        "Max-Age" : 3600,
        "Domain"  : ".example.com",
    }
    for k, v in example_cookie.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    demonstrate_sessions_jwt()
