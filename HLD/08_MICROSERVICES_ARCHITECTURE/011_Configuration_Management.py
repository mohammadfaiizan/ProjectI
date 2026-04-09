"""
CONFIGURATION MANAGEMENT IN MICROSERVICES
============================================

Problem Statement:
Microservices have hundreds of configuration values: DB URLs, timeouts,
feature flags, external API keys, environment-specific settings.
Hardcoding or file-based configs cause: restart-required changes, drift
between environments, no audit trail, no rollback, and insecure secrets handling.

Centralized Configuration Server:
  A single source of truth for all service configurations.
  Examples: Spring Cloud Config, AWS AppConfig, HashiCorp Consul KV,
            GCP Secret Manager, Azure App Configuration.

  Key features:
    - Versioned: every config change is tracked.
    - Audited: who changed what and when.
    - Environment-scoped: dev/staging/prod have different values.
    - Hot reload: services pick up changes without restarting.
    - Encrypted: secrets stored encrypted, decrypted at fetch time.

Feature Flags (Feature Toggles):
  A specific type of config: boolean/percentage controls that gate features.
  Types:
    Release toggle:   Enable a feature only in prod when ready. Ship dark.
    Experiment toggle: A/B test; % of users see new experience.
    Ops toggle:       Kill switch for a performance-sensitive feature.
    Permission toggle: Enable for specific users/groups (early access).

  Flag evaluation inputs:
    - User ID (consistent hashing for % rollout)
    - Environment (prod/staging/dev)
    - % traffic (gradual rollout: 0 → 5 → 20 → 100)

  Key benefit: decouple deployment from feature release.
  Ship code merged to main but feature hidden behind a flag.
  Enable it without a deployment when ready.

Hot Reload:
  Services poll or subscribe to config changes.
  On change: update in-memory config without restarting.
  Use cases: adjust rate limits, timeout values, feature flags live.
  Risk: config change bugs affect production instantly — use gradual rollout.

Config Hierarchy (priority order, high to low):
  1. Environment variable overrides (ops emergency override)
  2. Config server (environment-specific)
  3. Service default (compiled into service)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
import time
import threading
import uuid
import hashlib


# ─────────────────────────────────────────────
# CONFIG ENTRY
# ─────────────────────────────────────────────

@dataclass
class ConfigEntry:
    key         : str
    value       : Any
    environment : str           # dev / staging / prod / *
    version     : int = 1
    updated_by  : str = "system"
    updated_at  : float = field(default_factory=time.time)
    encrypted   : bool = False


# ─────────────────────────────────────────────
# CONFIG SERVER
# ─────────────────────────────────────────────

class ConfigServer:
    """
    Centralized configuration store.
    Supports: set, get (with env fallback), versioning, change watchers.
    """

    def __init__(self):
        self._store     : Dict[str, Dict[str, ConfigEntry]] = {}  # key → {env → entry}
        self._version   : int = 0
        self._watchers  : Dict[str, List[Callable]] = {}   # key → [callbacks]
        self._audit_log : List[Dict] = []
        self._lock      = threading.Lock()

    def set(self, key: str, value: Any, environment: str = "*",
            updated_by: str = "system", encrypted: bool = False):
        with self._lock:
            self._version += 1
            if key not in self._store:
                self._store[key] = {}

            version = (self._store[key].get(environment) or
                       self._store[key].get("*"))
            new_version = (version.version + 1) if version else 1

            entry = ConfigEntry(
                key=key, value=value, environment=environment,
                version=new_version, updated_by=updated_by,
                encrypted=encrypted
            )
            self._store[key][environment] = entry
            self._audit_log.append({
                "action": "set", "key": key, "env": environment,
                "version": new_version, "by": updated_by,
                "ts": time.time(),
            })

        # Notify watchers outside lock
        for cb in self._watchers.get(key, []):
            threading.Thread(target=cb, args=(key, value, environment),
                             daemon=True).start()

    def get(self, key: str, environment: str = "prod",
            default: Any = None) -> Any:
        with self._lock:
            env_map = self._store.get(key, {})
            # Priority: exact env > wildcard '*' > default
            entry = env_map.get(environment) or env_map.get("*")
            if entry is None:
                return default
            return entry.value

    def get_entry(self, key: str, environment: str = "prod") -> Optional[ConfigEntry]:
        with self._lock:
            env_map = self._store.get(key, {})
            return env_map.get(environment) or env_map.get("*")

    def watch(self, key: str, callback: Callable):
        """Register a callback fired when config key changes."""
        with self._lock:
            self._watchers.setdefault(key, []).append(callback)

    def list_keys(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())

    def audit_log(self, key: Optional[str] = None) -> List[Dict]:
        with self._lock:
            if key:
                return [e for e in self._audit_log if e["key"] == key]
            return list(self._audit_log)


# ─────────────────────────────────────────────
# FEATURE FLAG ENGINE
# ─────────────────────────────────────────────

@dataclass
class FeatureFlag:
    name         : str
    enabled      : bool = False       # global on/off
    rollout_pct  : float = 0.0        # 0-100% traffic
    allowed_users: Set[str] = field(default_factory=set)
    allowed_envs : Set[str] = field(default_factory=lambda: {"dev","staging","prod"})
    description  : str = ""


class FeatureFlagEngine:
    """
    Evaluates feature flags per (user, environment).
    Uses consistent hashing so the same user always gets the same result
    for a given rollout percentage.
    """

    def __init__(self, config_server: ConfigServer):
        self._server   = config_server
        self._flags    : Dict[str, FeatureFlag] = {}
        self._eval_log : List[Dict] = []

    def define_flag(self, flag: FeatureFlag):
        self._flags[flag.name] = flag
        # Store in config server so it can be hot-reloaded
        self._server.set(f"feature.{flag.name}.enabled",
                         flag.enabled, updated_by="feature-engine")
        self._server.set(f"feature.{flag.name}.rollout_pct",
                         flag.rollout_pct, updated_by="feature-engine")

    def update_rollout(self, flag_name: str, pct: float, actor: str = "ops"):
        flag = self._flags.get(flag_name)
        if flag:
            flag.rollout_pct = max(0.0, min(100.0, pct))
            self._server.set(f"feature.{flag_name}.rollout_pct",
                             flag.rollout_pct, updated_by=actor)

    def enable(self, flag_name: str, actor: str = "ops"):
        flag = self._flags.get(flag_name)
        if flag:
            flag.enabled = True
            self._server.set(f"feature.{flag_name}.enabled",
                             True, updated_by=actor)

    def disable(self, flag_name: str, actor: str = "ops"):
        flag = self._flags.get(flag_name)
        if flag:
            flag.enabled = False
            self._server.set(f"feature.{flag_name}.enabled",
                             False, updated_by=actor)

    def is_enabled(self, flag_name: str, user_id: str,
                   environment: str = "prod") -> bool:
        flag = self._flags.get(flag_name)
        if flag is None:
            return False
        if not flag.enabled:
            return False
        if environment not in flag.allowed_envs:
            return False
        # Specific user allowlist
        if user_id in flag.allowed_users:
            self._log(flag_name, user_id, environment, True, "allowlist")
            return True
        # Percentage rollout — consistent hashing
        if flag.rollout_pct <= 0:
            self._log(flag_name, user_id, environment, False, "zero_pct")
            return False
        if flag.rollout_pct >= 100:
            self._log(flag_name, user_id, environment, True, "full_rollout")
            return True

        hash_input = f"{flag_name}:{user_id}".encode()
        hash_val   = int(hashlib.md5(hash_input).hexdigest(), 16) % 100
        result     = hash_val < flag.rollout_pct
        self._log(flag_name, user_id, environment, result, f"pct_{flag.rollout_pct}")
        return result

    def _log(self, flag: str, user: str, env: str, result: bool, reason: str):
        self._eval_log.append({
            "flag": flag, "user": user, "env": env,
            "result": result, "reason": reason
        })

    def rollout_stats(self, flag_name: str) -> Dict:
        relevant = [e for e in self._eval_log if e["flag"] == flag_name]
        total    = len(relevant)
        enabled  = sum(1 for e in relevant if e["result"])
        return {
            "flag"         : flag_name,
            "evaluations"  : total,
            "enabled_count": enabled,
            "disabled_count": total - enabled,
            "actual_pct"   : round(enabled / max(total, 1) * 100, 1),
        }


# ─────────────────────────────────────────────
# SERVICE CONFIG CONSUMER (hot reload)
# ─────────────────────────────────────────────

class ServiceConfig:
    """
    A service's view of its configuration.
    Watches the config server for changes; hot-reloads without restart.
    """

    def __init__(self, service_name: str, server: ConfigServer,
                 environment: str = "prod"):
        self.service_name = service_name
        self._server      = server
        self._env         = environment
        self._local_cache : Dict[str, Any] = {}
        self._reload_count = 0

        server.watch("*", self._on_any_change)

    def get(self, key: str, default: Any = None) -> Any:
        cached = self._local_cache.get(key)
        if cached is not None:
            return cached
        return self._server.get(key, self._env, default)

    def _on_any_change(self, key: str, value: Any, env: str):
        if env in (self._env, "*"):
            self._local_cache[key] = value
            self._reload_count += 1


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_configuration_management():
    print("=" * 65)
    print("CONFIGURATION MANAGEMENT IN MICROSERVICES")
    print("=" * 65)

    server  = ConfigServer()
    flags   = FeatureFlagEngine(server)

    # ── 1. Basic config set/get ───────────────────
    print("\n[1] CENTRALIZED CONFIG SERVER — SET AND GET")
    print("─" * 55)

    server.set("db.pool.size",          10,  environment="*",       updated_by="infra")
    server.set("db.pool.size",          20,  environment="prod",    updated_by="infra")
    server.set("api.timeout_ms",        500, environment="*",       updated_by="infra")
    server.set("api.timeout_ms",        2000,environment="dev",     updated_by="dev-team")
    server.set("payment.api_key",       "sk_prod_***", environment="prod",
               updated_by="security-team", encrypted=True)

    configs = [
        ("db.pool.size",   "dev"),
        ("db.pool.size",   "prod"),
        ("api.timeout_ms", "dev"),
        ("api.timeout_ms", "prod"),
        ("api.timeout_ms", "staging"),
    ]
    print(f"  {'Key':<22} {'Env':<10} {'Value'}")
    print(f"  {'─'*45}")
    for key, env in configs:
        val = server.get(key, env)
        print(f"  {key:<22} {env:<10} {val}")

    print(f"\n  'staging' falls back to '*' wildcard (no staging-specific override)")

    # ── 2. Feature flags — percentage rollout ────
    print("\n\n[2] FEATURE FLAGS — PERCENTAGE ROLLOUT")
    print("─" * 55)

    flags.define_flag(FeatureFlag(
        name="new_checkout_flow",
        enabled=True,
        rollout_pct=20.0,
        description="Redesigned checkout — rolling out to 20% of users"
    ))

    flags.define_flag(FeatureFlag(
        name="recommendation_v2",
        enabled=True,
        rollout_pct=50.0,
        description="New ML recommendation model"
    ))

    flags.define_flag(FeatureFlag(
        name="dark_mode",
        enabled=False,
        rollout_pct=0.0,
        description="Dark mode — disabled globally"
    ))

    # Evaluate across many users
    user_ids = [f"user-{i:04d}" for i in range(100)]
    checkout_enabled = sum(1 for u in user_ids
                          if flags.is_enabled("new_checkout_flow", u))
    recs_enabled     = sum(1 for u in user_ids
                          if flags.is_enabled("recommendation_v2", u))
    dark_enabled     = sum(1 for u in user_ids
                          if flags.is_enabled("dark_mode", u))

    print(f"  {'Flag':<26} {'Config %':<12} {'Actual % (100 users)'}")
    print(f"  {'─'*55}")
    print(f"  {'new_checkout_flow':<26} {'20%':<12} {checkout_enabled}%")
    print(f"  {'recommendation_v2':<26} {'50%':<12} {recs_enabled}%")
    print(f"  {'dark_mode':<26} {'0% (disabled)':<12} {dark_enabled}%")
    print(f"\n  Consistent hashing: same user always gets same result for same %")

    # ── 3. Specific user in allowlist ─────────────
    print("\n\n[3] FEATURE FLAG — USER ALLOWLIST (EARLY ACCESS)")
    print("─" * 55)
    flags.define_flag(FeatureFlag(
        name="admin_dashboard_v2",
        enabled=True,
        rollout_pct=0.0,
        allowed_users={"alice", "bob", "charlie"},
        description="New admin dashboard — only beta testers"
    ))
    test_users = ["alice", "bob", "dave", "eve"]
    for user in test_users:
        result = flags.is_enabled("admin_dashboard_v2", user)
        access = "ENABLED (allowlist)" if result else "DISABLED"
        print(f"  {user:<10} → {access}")

    # ── 4. Progressive rollout ────────────────────
    print("\n\n[4] PROGRESSIVE ROLLOUT — 0% → 10% → 50% → 100%")
    print("─" * 55)
    flags.define_flag(FeatureFlag(
        name="streaming_api",
        enabled=True,
        rollout_pct=0.0,
    ))
    users_50 = [f"u-{i}" for i in range(50)]
    for pct in [0, 10, 50, 100]:
        flags.update_rollout("streaming_api", float(pct), actor="release-eng")
        enabled_count = sum(1 for u in users_50
                           if flags.is_enabled("streaming_api", u))
        print(f"  Rollout={pct:>3}%  →  {enabled_count}/50 users enabled "
              f"({enabled_count*2:.0f}% actual)")

    # ── 5. Hot reload ─────────────────────────────
    print("\n\n[5] HOT RELOAD — CONFIG CHANGE WITHOUT RESTART")
    print("─" * 55)
    svc_config = ServiceConfig("order-service", server, environment="prod")

    print(f"  Initial timeout: {server.get('api.timeout_ms', 'prod')}ms")
    reloads_before = svc_config._reload_count

    server.set("api.timeout_ms", 1500, environment="prod", updated_by="ops")
    time.sleep(0.05)  # let watcher fire

    print(f"  After hot reload: {server.get('api.timeout_ms', 'prod')}ms")
    print(f"  Service reloads triggered: {svc_config._reload_count - reloads_before}")
    print(f"  → Service picked up new timeout without restart.")

    # ── 6. Audit log ──────────────────────────────
    print("\n\n[6] AUDIT LOG — WHO CHANGED WHAT")
    print("─" * 55)
    log = server.audit_log("api.timeout_ms")
    for entry in log:
        print(f"  key={entry['key']:<20} env={entry['env']:<10} "
              f"v{entry['version']} by={entry['by']}")

    # ── 7. Config hierarchy ───────────────────────
    print("\n\n[7] CONFIG RESOLUTION HIERARCHY")
    print("─" * 55)
    hierarchy = [
        ("1 (highest)", "Environment variable",  "Emergency override; set by ops"),
        ("2",           "Config server: env-specific", "prod/staging/dev overrides"),
        ("3",           "Config server: wildcard *",   "Shared defaults across envs"),
        ("4 (lowest)",  "Service default (code)",      "Fallback if config server down"),
    ]
    print(f"  {'Priority':<14} {'Source':<28} {'Purpose'}")
    print(f"  {'─'*62}")
    for prio, source, purpose in hierarchy:
        print(f"  {prio:<14} {source:<28} {purpose}")

    # ── 8. Flag rollout stats ─────────────────────
    print("\n\n[8] FEATURE FLAG EVALUATION STATISTICS")
    print("─" * 55)
    for flag_name in ["new_checkout_flow", "recommendation_v2", "streaming_api"]:
        stats = flags.rollout_stats(flag_name)
        print(f"  {flag_name:<28} evals={stats['evaluations']:<6} "
              f"enabled={stats['enabled_count']:<5} "
              f"actual_pct={stats['actual_pct']}%")


if __name__ == "__main__":
    demonstrate_configuration_management()
