"""
SECRET MANAGEMENT
==================

Problem Statement:
Applications need credentials: DB passwords, API keys, TLS certificates.
Hardcoding secrets in code or config files is a critical security risk.
Secret management centralizes, protects, and rotates secrets safely.

Problems to Solve:
  1. Secret sprawl: credentials scattered in code, env vars, config files.
  2. No rotation: leaked credentials remain valid indefinitely.
  3. No audit trail: who accessed what secret, when?
  4. No least privilege: all services share same credentials.
  5. No expiry: long-lived credentials increase breach window.

Solutions:
  HashiCorp Vault:  Most popular. Dynamic secrets, lease expiry, audit.
  AWS Secrets Manager: Managed. Automatic rotation. IAM integration.
  AWS SSM Parameter Store: Simpler, cheaper. Good for configs.
  GCP Secret Manager: Cloud-native. IAM-controlled.
  Azure Key Vault: Keys, secrets, certificates. RBAC.
  Kubernetes Secrets: Base64 (not encrypted!). Use with external-secrets.

Vault Dynamic Secrets:
  Vault generates short-lived credentials on demand.
  DB credentials: Vault creates a user in PostgreSQL, returns credentials.
  Credentials expire after TTL (e.g., 1 hour). Auto-revoked.
  No shared long-lived passwords.

Secret Lifecycle:
  1. Create:   Generate secret. Store encrypted in vault.
  2. Distribute: App authenticates to vault, reads secret.
  3. Rotate:   Generate new version. Notify dependent apps.
  4. Revoke:   Remove access. Invalidate old version.
  5. Audit:    Log all access. Alert on anomalies.

AppRole Authentication (Vault):
  Service authenticates with role_id + secret_id.
  role_id: identifies the service (like a username). Long-lived.
  secret_id: one-time or short-lived credential. Rotated.
  Returns Vault token for subsequent secret reads.

Secret Rotation:
  Automatic: Vault + Lambda rotate DB password, update Secrets Manager.
  Application: detect new secret version, reconnect to DB.
  Zero-downtime: overlap period (both old+new passwords valid).

12-Factor App: Secret Injection:
  Factor 3: Store config in environment.
  Never commit secrets to git. Use .env files (gitignored).
  In production: inject via: K8s Secrets, Vault sidecar, AWS Secrets.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import hashlib
import hmac
import secrets
import time
import uuid
import json


# ─────────────────────────────────────────────
# SECRET ENTRY
# ─────────────────────────────────────────────

class SecretType(Enum):
    STATIC    = "static"     # password, API key
    DYNAMIC   = "dynamic"    # DB credentials (time-limited)
    CERT      = "certificate"
    ENCRYPTION_KEY = "encryption_key"


@dataclass
class SecretVersion:
    version_id  : str
    value       : str
    created_at  : float
    expires_at  : Optional[float]
    created_by  : str
    active      : bool = True
    metadata    : Dict[str, str] = field(default_factory=dict)

    def is_expired(self) -> bool:
        if self.expires_at and time.time() > self.expires_at:
            return True
        return not self.active

    def ttl_remaining(self) -> Optional[float]:
        if not self.expires_at:
            return None
        return max(0, self.expires_at - time.time())


@dataclass
class SecretEntry:
    secret_id   : str
    name        : str
    secret_type : SecretType
    versions    : List[SecretVersion] = field(default_factory=list)
    rotation_period_s: Optional[float] = None
    last_rotated: Optional[float] = None

    def current_version(self) -> Optional[SecretVersion]:
        active = [v for v in self.versions if not v.is_expired()]
        return active[-1] if active else None

    def add_version(self, value: str, created_by: str,
                    ttl_s: Optional[float] = None) -> SecretVersion:
        # Deprecate old versions
        for v in self.versions:
            v.active = False
        version_id = f"v{len(self.versions) + 1}"
        expires_at = time.time() + ttl_s if ttl_s else None
        version    = SecretVersion(
            version_id=version_id, value=value,
            created_at=time.time(), expires_at=expires_at,
            created_by=created_by,
        )
        self.versions.append(version)
        self.last_rotated = time.time()
        return version


# ─────────────────────────────────────────────
# VAULT SIMULATOR
# ─────────────────────────────────────────────

@dataclass
class VaultPolicy:
    name        : str
    capabilities: Dict[str, Set[str]]   # path → {read, write, delete, list}


@dataclass
class VaultRole:
    role_id     : str
    policies    : List[str]
    token_ttl_s : int = 3600


@dataclass
class VaultToken:
    token       : str
    role_id     : str
    policies    : List[str]
    created_at  : float = field(default_factory=time.time)
    ttl_s       : int = 3600
    uses_left   : Optional[int] = None   # None = unlimited

    def is_valid(self) -> bool:
        if (time.time() - self.created_at) > self.ttl_s:
            return False
        if self.uses_left is not None and self.uses_left <= 0:
            return False
        return True


class VaultSimulator:
    """
    HashiCorp Vault simulator: AppRole auth, policies, dynamic secrets, audit.
    """

    def __init__(self):
        self._secrets   : Dict[str, SecretEntry] = {}
        self._roles     : Dict[str, VaultRole]   = {}
        self._policies  : Dict[str, VaultPolicy] = {}
        self._tokens    : Dict[str, VaultToken]  = {}
        self._role_ids  : Dict[str, str] = {}   # role_id → role_name
        self._secret_ids: Dict[str, str] = {}   # secret_id → role_name
        self._audit_log : List[Dict] = []
        self._db_users  : Dict[str, float] = {}  # dynamic DB users

    def create_policy(self, name: str, capabilities: Dict[str, Set[str]]):
        self._policies[name] = VaultPolicy(name=name, capabilities=capabilities)

    def create_role(self, role_name: str, policies: List[str],
                     token_ttl_s: int = 3600) -> str:
        role_id = secrets.token_hex(16)
        self._roles[role_name]    = VaultRole(role_id=role_id, policies=policies,
                                               token_ttl_s=token_ttl_s)
        self._role_ids[role_id] = role_name
        return role_id

    def generate_secret_id(self, role_name: str) -> str:
        """One-time secret_id for AppRole auth."""
        secret_id = secrets.token_hex(16)
        self._secret_ids[secret_id] = role_name
        return secret_id

    def approle_login(self, role_id: str, secret_id: str) -> Tuple[Optional[str], Optional[str]]:
        """AppRole authentication. Returns (vault_token, error)."""
        role_name = self._role_ids.get(role_id)
        if not role_name:
            return None, "invalid_role_id"
        if self._secret_ids.get(secret_id) != role_name:
            return None, "invalid_secret_id"
        # Consume secret_id (one-time use)
        del self._secret_ids[secret_id]

        role  = self._roles[role_name]
        token_str = secrets.token_urlsafe(32)
        token = VaultToken(token=token_str, role_id=role_id,
                            policies=role.policies, ttl_s=role.token_ttl_s)
        self._tokens[token_str] = token
        self._audit("approle_login", role_name, "token_issued")
        return token_str, None

    def write_secret(self, vault_token: str, path: str,
                      value: str, ttl_s: float = None) -> Tuple[bool, str]:
        token = self._validate_token(vault_token, path, "write")
        if not token:
            return False, "permission_denied"
        if path not in self._secrets:
            secret_id = uuid.uuid4().hex[:8]
            self._secrets[path] = SecretEntry(
                secret_id=secret_id, name=path, secret_type=SecretType.STATIC
            )
        version = self._secrets[path].add_version(value, token.role_id, ttl_s)
        self._audit("write", path, f"version={version.version_id}", vault_token)
        return True, version.version_id

    def read_secret(self, vault_token: str, path: str) -> Tuple[Optional[str], Optional[str]]:
        token = self._validate_token(vault_token, path, "read")
        if not token:
            return None, "permission_denied"
        entry = self._secrets.get(path)
        if not entry:
            return None, "not_found"
        version = entry.current_version()
        if not version:
            return None, "no_active_version"
        self._audit("read", path, f"version={version.version_id}", vault_token)
        return version.value, None

    def get_dynamic_db_credential(self, vault_token: str,
                                   db_role: str = "app-role") -> Tuple[Optional[Dict], Optional[str]]:
        """Generate short-lived DB credentials (dynamic secrets)."""
        token = self._validate_token(vault_token, f"database/creds/{db_role}", "read")
        if not token:
            return None, "permission_denied"
        # Simulate creating a DB user
        username = f"vault_app_{uuid.uuid4().hex[:8]}"
        password = secrets.token_urlsafe(24)
        expires_at = time.time() + 3600
        self._db_users[username] = expires_at
        creds = {"username": username, "password": password,
                  "lease_duration": 3600, "expires_at": expires_at}
        self._audit("dynamic_creds", f"database/{db_role}", f"user={username}", vault_token)
        return creds, None

    def revoke_lease(self, username: str):
        self._db_users.pop(username, None)

    def rotate_secret(self, vault_token: str, path: str,
                       new_value: str) -> Tuple[bool, str]:
        ok, version_id = self.write_secret(vault_token, path, new_value)
        if ok:
            self._audit("rotate", path, f"new_version={version_id}", vault_token)
        return ok, version_id

    def _validate_token(self, vault_token: str, path: str,
                         capability: str) -> Optional[VaultToken]:
        token = self._tokens.get(vault_token)
        if not token or not token.is_valid():
            return None
        for policy_name in token.policies:
            policy = self._policies.get(policy_name)
            if not policy:
                continue
            for pattern, caps in policy.capabilities.items():
                if path.startswith(pattern.rstrip("*")) and capability in caps:
                    return token
        return None

    def _audit(self, operation: str, path: str, detail: str = "",
                vault_token: str = "system"):
        self._audit_log.append({
            "ts"       : time.time(),
            "op"       : operation,
            "path"     : path,
            "detail"   : detail,
            "token"    : vault_token[:8] + "..." if vault_token != "system" else "system",
        })

    def audit_log(self, last_n: int = 10) -> List[Dict]:
        return self._audit_log[-last_n:]


# ─────────────────────────────────────────────
# SECRET SCANNING (detect leaked secrets in code)
# ─────────────────────────────────────────────

import re

SECRET_PATTERNS = [
    (r"(?i)(api[_-]key|apikey)\s*=\s*['\"][A-Za-z0-9/+]{20,}['\"]", "API Key"),
    (r"(?i)(password|passwd|pwd)\s*=\s*['\"][^'\"]{8,}['\"]", "Password"),
    (r"(?i)(secret|token)\s*=\s*['\"][A-Za-z0-9/+_-]{20,}['\"]", "Secret/Token"),
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key"),
    (r"-----BEGIN (RSA|EC) PRIVATE KEY-----", "Private Key"),
]


def scan_for_secrets(code: str) -> List[Dict]:
    findings = []
    for i, line in enumerate(code.splitlines(), 1):
        for pattern, secret_type in SECRET_PATTERNS:
            if re.search(pattern, line):
                findings.append({
                    "line"        : i,
                    "type"        : secret_type,
                    "snippet"     : line.strip()[:60],
                })
    return findings


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_secret_management():
    print("=" * 65)
    print("SECRET MANAGEMENT")
    print("=" * 65)

    vault = VaultSimulator()

    # Setup policies
    vault.create_policy("app-policy", {
        "secret/app/"       : {"read"},
        "database/creds/"   : {"read"},
    })
    vault.create_policy("ops-policy", {
        "secret/"           : {"read", "write", "delete", "list"},
        "database/creds/"   : {"read"},
    })

    # ── AppRole Authentication ─────────────────────
    print("\n[1] APPROLE AUTHENTICATION (service identity)")
    print("─" * 55)

    role_id   = vault.create_role("web-app", ["app-policy"], token_ttl_s=3600)
    secret_id = vault.generate_secret_id("web-app")
    print(f"  role_id:   {role_id[:16]}...")
    print(f"  secret_id: {secret_id[:16]}... (one-time use)")

    token, err = vault.approle_login(role_id, secret_id)
    print(f"  Login: token={token[:16]}... err={err}")

    # Second use should fail
    _, err2 = vault.approle_login(role_id, secret_id)
    print(f"  Replay secret_id: err={err2} (one-time use enforced)")

    # Ops role
    ops_role_id   = vault.create_role("ops", ["ops-policy"])
    ops_secret_id = vault.generate_secret_id("ops")
    ops_token, _  = vault.approle_login(ops_role_id, ops_secret_id)

    # ── Write and Read Secrets ─────────────────────
    print("\n\n[2] WRITE AND READ SECRETS")
    print("─" * 55)

    ok, vid = vault.write_secret(ops_token, "secret/app/db-password",
                                   "superSecretDbPass123!")
    print(f"  Write secret: ok={ok} version={vid}")

    value, err = vault.read_secret(token, "secret/app/db-password")
    print(f"  Read  secret: value={value} err={err}")

    # Test permission denied
    _, err3 = vault.write_secret(token, "secret/app/db-password", "hacked")
    print(f"  Write with app-token (read-only): err={err3}")

    # ── Dynamic DB Credentials ────────────────────
    print("\n\n[3] DYNAMIC DB CREDENTIALS (ephemeral)")
    print("─" * 55)

    creds, err = vault.get_dynamic_db_credential(token, "app-role")
    print(f"  Dynamic DB creds:")
    print(f"    username:  {creds['username']}")
    print(f"    password:  {creds['password'][:16]}...")
    print(f"    expires:   {creds['lease_duration']}s")
    print(f"  Note: Vault creates real DB user, revokes on lease expiry")

    # ── Secret Rotation ───────────────────────────
    print("\n\n[4] SECRET ROTATION")
    print("─" * 55)

    ok2, vid2 = vault.rotate_secret(ops_token, "secret/app/db-password",
                                      "newRotatedPassword456!")
    new_val, _ = vault.read_secret(token, "secret/app/db-password")
    print(f"  Rotated to version {vid2}: {new_val}")

    entry = vault._secrets["secret/app/db-password"]
    print(f"  All versions: {[(v.version_id, v.active) for v in entry.versions]}")

    # ── Audit Log ─────────────────────────────────
    print("\n\n[5] AUDIT LOG")
    print("─" * 55)

    for entry_log in vault.audit_log(8):
        print(f"  {entry_log['op']:<16} {entry_log['path']:<35} {entry_log['detail']}")

    # ── Secret Scanning ───────────────────────────
    print("\n\n[6] SECRET SCANNING (detect leaked credentials)")
    print("─" * 55)

    bad_code = '''
    api_key = "sk_live_abc123def456xyz789012"
    password = "hardcoded_pass_123!"
    DB_PASSWORD = "admin1234"
    token = "eyJhbGciOiJIUzI1NiJ9.super_secret_token_12345"
    AKIAIOSFODNN7EXAMPLE
    '''
    findings = scan_for_secrets(bad_code)
    print(f"  Found {len(findings)} potential secrets:")
    for f in findings:
        print(f"    Line {f['line']:2}: [{f['type']}] {f['snippet']}")

    # ── Best Practices ────────────────────────────
    print("\n\n[7] SECRET MANAGEMENT BEST PRACTICES")
    print("─" * 55)

    practices = [
        ("Never hardcode secrets",    "Use env vars or secret manager; scan git history"),
        ("Rotate regularly",          "Auto-rotate with Vault/Secrets Manager + Lambda"),
        ("Dynamic secrets",           "Short-lived credentials reduce breach window"),
        ("Least privilege",           "Per-service vault role with minimal paths"),
        ("Audit all reads",           "Alert on unusual access patterns (off-hours, bulk)"),
        ("Seal old versions",         "Deprecate but retain for decryption key rotation"),
        ("Immutable infrastructure",  "Inject secrets at runtime, never bake into images"),
        ("Secret scanning in CI",     "Block commits with detected secrets (truffleHog, gitleaks)"),
    ]
    for practice, guidance in practices:
        print(f"  {practice:<28} {guidance}")


if __name__ == "__main__":
    demonstrate_secret_management()
