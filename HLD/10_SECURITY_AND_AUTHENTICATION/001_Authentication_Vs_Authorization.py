"""
AUTHENTICATION vs AUTHORIZATION
==================================

Problem Statement:
Security systems must answer two distinct questions:
  Authentication (AuthN): "Who are you?" — verifying identity.
  Authorization  (AuthZ): "What can you do?" — verifying permissions.
  Conflating these leads to security vulnerabilities.

Authentication Methods:
  Password-based: Store hashed+salted passwords. Verify on login.
  Token-based:    Issue JWT or opaque token after login.
  API Keys:       Long-lived secrets for service-to-service.
  mTLS:           Client presents certificate. Mutual TLS.
  OAuth2:         Delegated authorization (user grants app access).
  SAML/OIDC:      Federation — trust external identity provider.
  MFA:            Something you know + something you have.

Authorization Models:
  ACL (Access Control List): per-resource list of allowed users/roles.
  RBAC (Role-Based):  User → Roles → Permissions. Most common.
  ABAC (Attribute-Based): Policy evaluated against attributes.
                   Context-aware: time, IP, resource attributes.
  ReBAC (Relationship-Based): Google Zanzibar model.
                   "User can read doc if user is owner or is in group that has reader."

Zero Trust:
  Never trust, always verify. Authenticate every request.
  Even internal service calls require auth.
  Principle of least privilege: grant minimum necessary permissions.

Token Types:
  JWT (stateless): self-contained claims. Verify with secret/public key.
                   Pro: no DB lookup. Con: can't revoke before expiry.
  Opaque token (stateful): random string. Server stores session data.
                   Pro: revocable. Con: DB lookup on every request.
  Refresh token: long-lived. Exchange for short-lived access token.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import hashlib
import hmac
import time
import uuid
import json
import base64


# ─────────────────────────────────────────────
# PASSWORD HASHING
# ─────────────────────────────────────────────

class PasswordHasher:
    """
    Simulates bcrypt-like password hashing (uses PBKDF2 for demo).
    Real production: use bcrypt or argon2id.
    """
    ITERATIONS = 100_000
    SALT_BYTES = 16

    def hash(self, password: str) -> str:
        salt = uuid.uuid4().hex[:32]
        dk   = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(),
                                    self.ITERATIONS)
        return f"$pbkdf2${salt}${dk.hex()}"

    def verify(self, password: str, stored_hash: str) -> bool:
        try:
            _, algo, salt, dk_hex = stored_hash.split("$")
        except ValueError:
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(),
                                  self.ITERATIONS)
        return hmac.compare_digest(dk.hex(), dk_hex)


# ─────────────────────────────────────────────
# JWT (simplified — no cryptographic signing)
# ─────────────────────────────────────────────

class JWT:
    """
    Simplified JWT: header.payload.signature.
    Signs with HMAC-SHA256 using a shared secret.
    NOT production-ready (use PyJWT or python-jose).
    """

    def __init__(self, secret: str):
        self._secret = secret.encode()

    def _b64(self, data: str) -> str:
        return base64.urlsafe_b64encode(data.encode()).rstrip(b"=").decode()

    def _unb64(self, data: str) -> str:
        pad = 4 - len(data) % 4
        return base64.urlsafe_b64decode(data + "=" * pad).decode()

    def sign(self, payload: Dict, expires_in_s: int = 3600) -> str:
        payload["iat"] = int(time.time())
        payload["exp"] = int(time.time()) + expires_in_s
        header  = self._b64(json.dumps({"alg": "HS256", "typ": "JWT"}))
        body    = self._b64(json.dumps(payload))
        signing = f"{header}.{body}"
        sig     = hmac.new(self._secret, signing.encode(), hashlib.sha256).hexdigest()
        return f"{signing}.{sig}"

    def verify(self, token: str) -> Tuple_like:
        """Returns (payload, error). payload=None if invalid."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None, "malformed token"
            header, body, sig = parts
            expected_sig = hmac.new(self._secret, f"{header}.{body}".encode(),
                                     hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected_sig):
                return None, "invalid signature"
            payload = json.loads(self._unb64(body))
            if payload.get("exp", 0) < time.time():
                return None, "token expired"
            return payload, None
        except Exception as e:
            return None, str(e)


# Workaround: Tuple return type alias
Tuple_like = Any


# ─────────────────────────────────────────────
# RBAC (Role-Based Access Control)
# ─────────────────────────────────────────────

class Permission(Enum):
    READ    = "read"
    WRITE   = "write"
    DELETE  = "delete"
    ADMIN   = "admin"
    EXECUTE = "execute"


@dataclass
class Role:
    name        : str
    permissions : Set[Permission]


@dataclass
class User:
    user_id       : str
    username      : str
    roles         : Set[str]
    password_hash : str = ""
    active        : bool = True
    mfa_enabled   : bool = False


class RBACEngine:
    """
    Role-Based Access Control.
    Users → Roles → Permissions.
    """

    def __init__(self):
        self._roles : Dict[str, Role]  = {}
        self._users : Dict[str, User]  = {}

    def define_role(self, name: str, permissions: Set[Permission]):
        self._roles[name] = Role(name=name, permissions=permissions)

    def register_user(self, user: User):
        self._users[user.user_id] = user

    def assign_role(self, user_id: str, role_name: str):
        if user_id in self._users and role_name in self._roles:
            self._users[user_id].roles.add(role_name)

    def has_permission(self, user_id: str, permission: Permission,
                        resource: str = None) -> bool:
        user = self._users.get(user_id)
        if not user or not user.active:
            return False
        for role_name in user.roles:
            role = self._roles.get(role_name)
            if role and permission in role.permissions:
                return True
        return False

    def effective_permissions(self, user_id: str) -> Set[Permission]:
        user = self._users.get(user_id)
        if not user:
            return set()
        perms: Set[Permission] = set()
        for role_name in user.roles:
            role = self._roles.get(role_name)
            if role:
                perms.update(role.permissions)
        return perms


# ─────────────────────────────────────────────
# ABAC (Attribute-Based Access Control)
# ─────────────────────────────────────────────

@dataclass
class ABACContext:
    user_attrs    : Dict[str, Any]    # user.department, user.clearance_level
    resource_attrs: Dict[str, Any]    # resource.classification, resource.owner
    env_attrs     : Dict[str, Any]    # time_of_day, ip_address, device_type


@dataclass
class ABACPolicy:
    name      : str
    effect    : str   # "ALLOW" or "DENY"
    conditions: List[Dict]   # list of {attr, op, value}

    def evaluate(self, ctx: ABACContext) -> bool:
        all_attrs = {**ctx.user_attrs, **ctx.resource_attrs, **ctx.env_attrs}
        for cond in self.conditions:
            attr  = cond["attr"]
            op    = cond["op"]
            value = cond["value"]
            actual = all_attrs.get(attr)
            if op == "==" and actual != value:
                return False
            elif op == ">=" and (actual is None or actual < value):
                return False
            elif op == "in" and actual not in value:
                return False
            elif op == "not_in" and actual in value:
                return False
        return True


class ABACEngine:
    def __init__(self):
        self._policies: List[ABACPolicy] = []

    def add_policy(self, policy: ABACPolicy):
        self._policies.append(policy)

    def authorize(self, ctx: ABACContext, action: str) -> Tuple[bool, str]:
        """Returns (allowed, matching_policy_name)."""
        for policy in self._policies:
            if policy.evaluate(ctx):
                return policy.effect == "ALLOW", policy.name
        return False, "default_deny"


# ─────────────────────────────────────────────
# OAUTH2 TOKEN FLOW (simplified)
# ─────────────────────────────────────────────

class OAuth2Server:
    """Simplified OAuth2 Authorization Code + Token flow."""

    def __init__(self, jwt_service: JWT):
        self._jwt           = jwt_service
        self._auth_codes    : Dict[str, Dict] = {}    # code → {user, scopes, client_id, exp}
        self._refresh_tokens: Dict[str, Dict] = {}    # rt → {user, scopes}

    def issue_auth_code(self, user_id: str, client_id: str,
                         scopes: List[str]) -> str:
        code = uuid.uuid4().hex[:16]
        self._auth_codes[code] = {
            "user_id": user_id, "client_id": client_id,
            "scopes": scopes, "exp": time.time() + 60,
        }
        return code

    def exchange_code(self, code: str, client_id: str) -> Optional[Dict]:
        auth = self._auth_codes.pop(code, None)
        if not auth or auth["client_id"] != client_id:
            return None
        if auth["exp"] < time.time():
            return None
        access_token  = self._jwt.sign({
            "sub": auth["user_id"], "scope": " ".join(auth["scopes"]),
            "client_id": client_id,
        }, expires_in_s=3600)
        refresh_token = uuid.uuid4().hex
        self._refresh_tokens[refresh_token] = {
            "user_id": auth["user_id"], "scopes": auth["scopes"],
        }
        return {"access_token": access_token, "token_type": "Bearer",
                "expires_in": 3600, "refresh_token": refresh_token,
                "scope": " ".join(auth["scopes"])}

    def refresh(self, refresh_token: str) -> Optional[Dict]:
        info = self._refresh_tokens.get(refresh_token)
        if not info:
            return None
        new_access = self._jwt.sign({
            "sub": info["user_id"], "scope": " ".join(info["scopes"]),
        }, expires_in_s=3600)
        return {"access_token": new_access, "token_type": "Bearer",
                "expires_in": 3600}


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_auth():
    print("=" * 65)
    print("AUTHENTICATION vs AUTHORIZATION")
    print("=" * 65)

    hasher  = PasswordHasher()
    jwt_svc = JWT(secret="super-secret-key-2024")

    # ── Password Hashing ──────────────────────────
    print("\n[1] PASSWORD HASHING (PBKDF2)")
    print("─" * 55)

    stored = hasher.hash("mysecretpassword")
    print(f"  Stored hash: {stored[:40]}...")
    print(f"  Verify correct password: {hasher.verify('mysecretpassword', stored)}")
    print(f"  Verify wrong password:   {hasher.verify('wrongpassword', stored)}")

    # ── JWT ───────────────────────────────────────
    print("\n\n[2] JWT — STATELESS TOKEN")
    print("─" * 55)

    token = jwt_svc.sign({"sub": "user-42", "roles": ["admin", "editor"]})
    print(f"  Token: {token[:50]}...")
    payload, err = jwt_svc.verify(token)
    print(f"  Verified: {err is None}  payload={payload}")

    expired = jwt_svc.sign({"sub": "user-42"}, expires_in_s=-1)
    _, err2 = jwt_svc.verify(expired)
    print(f"  Expired token error: {err2}")

    # ── RBAC ──────────────────────────────────────
    print("\n\n[3] RBAC — ROLES AND PERMISSIONS")
    print("─" * 55)

    rbac = RBACEngine()
    rbac.define_role("viewer",  {Permission.READ})
    rbac.define_role("editor",  {Permission.READ, Permission.WRITE})
    rbac.define_role("admin",   {Permission.READ, Permission.WRITE,
                                  Permission.DELETE, Permission.ADMIN})

    alice = User("u1", "alice", {"editor"}, hasher.hash("alice123"))
    bob   = User("u2", "bob",   {"viewer"}, hasher.hash("bob456"))
    carol = User("u3", "carol", {"admin"},  hasher.hash("carol789"))

    for user in [alice, bob, carol]:
        rbac.register_user(user)

    for user in [alice, bob, carol]:
        perms = rbac.effective_permissions(user.user_id)
        print(f"  {user.username:<8}: roles={user.roles}  "
              f"effective={sorted(p.value for p in perms)}")

    print(f"\n  Alice can DELETE: {rbac.has_permission('u1', Permission.DELETE)}")
    print(f"  Carol can DELETE: {rbac.has_permission('u3', Permission.DELETE)}")
    print(f"  Bob   can WRITE:  {rbac.has_permission('u2', Permission.WRITE)}")

    # ── ABAC ──────────────────────────────────────
    print("\n\n[4] ABAC — ATTRIBUTE-BASED POLICIES")
    print("─" * 55)

    abac = ABACEngine()
    abac.add_policy(ABACPolicy(
        name="internal-read",
        effect="ALLOW",
        conditions=[
            {"attr": "user.department", "op": "in", "value": ["engineering", "product"]},
            {"attr": "resource.classification", "op": "==", "value": "internal"},
        ]
    ))
    abac.add_policy(ABACPolicy(
        name="business-hours-only",
        effect="DENY",
        conditions=[
            {"attr": "time_of_day", "op": "not_in", "value": range(9, 18)},
            {"attr": "resource.classification", "op": "==", "value": "confidential"},
        ]
    ))

    ctx1 = ABACContext(
        user_attrs={"user.department": "engineering", "user.clearance": 3},
        resource_attrs={"resource.classification": "internal", "resource.owner": "u1"},
        env_attrs={"time_of_day": 10, "ip_address": "10.0.0.1"},
    )
    ctx2 = ABACContext(
        user_attrs={"user.department": "sales"},
        resource_attrs={"resource.classification": "internal"},
        env_attrs={"time_of_day": 14},
    )

    for ctx, label in [(ctx1, "Engineer accessing internal doc at 10am"),
                        (ctx2, "Sales accessing internal doc")]:
        allowed, policy = abac.authorize(ctx, "read")
        print(f"  {label}: {allowed} ({policy})")

    # ── OAuth2 Flow ───────────────────────────────
    print("\n\n[5] OAUTH2 AUTHORIZATION CODE FLOW")
    print("─" * 55)

    oauth = OAuth2Server(jwt_svc)
    code  = oauth.issue_auth_code("user-42", "app-client-id",
                                    ["read:profile", "write:posts"])
    print(f"  Auth code issued: {code}")

    tokens = oauth.exchange_code(code, "app-client-id")
    print(f"  Tokens received: access={tokens['access_token'][:30]}...")
    print(f"  Scopes: {tokens['scope']}")
    print(f"  Expires in: {tokens['expires_in']}s")

    refreshed = oauth.refresh(tokens["refresh_token"])
    print(f"  Token refreshed: {'OK' if refreshed else 'FAILED'}")

    # ── Summary ───────────────────────────────────
    print("\n\n[6] AUTHN vs AUTHZ DESIGN GUIDE")
    print("─" * 55)
    rows = [
        ("AuthN: password",   "Hash with bcrypt/argon2id; minimum cost factor 12"),
        ("AuthN: JWT",        "Short expiry (15min); refresh tokens for UX"),
        ("AuthN: API keys",   "Hash stored key; prefix for identification"),
        ("AuthZ: RBAC",       "Simple, predictable; good for SaaS with fixed roles"),
        ("AuthZ: ABAC",       "Flexible; good for policy-heavy (financial, healthcare)"),
        ("AuthZ: ReBAC",      "Google Zanzibar; good for document/folder hierarchies"),
        ("Zero Trust",        "Authenticate every request, even internal service calls"),
        ("Least privilege",   "Grant minimum permissions; revoke when no longer needed"),
    ]
    for key, guidance in rows:
        print(f"  {key:<24} {guidance}")


if __name__ == "__main__":
    demonstrate_auth()
