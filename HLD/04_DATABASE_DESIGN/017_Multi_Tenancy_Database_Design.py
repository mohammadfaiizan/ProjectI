"""
MULTI-TENANCY DATABASE DESIGN
================================

Problem Statement:
A SaaS product serves multiple customers (tenants). How you store their data
affects isolation, performance, cost, and operational complexity.

Three Main Models:

  Model 1 — Separate Database per Tenant:
    Each tenant has their own database instance.
    ✓ Maximum isolation (data, performance, compliance)
    ✓ Easy per-tenant backup, migration, deletion (GDPR right to erasure)
    ✗ High operational overhead (1000 tenants = 1000 DBs)
    ✗ Poor resource utilization (small tenants waste dedicated DB)
    Use: high-compliance enterprise customers (HIPAA, FedRAMP)

  Model 2 — Shared Database, Separate Schema:
    One DB instance, one schema per tenant (PostgreSQL schemas/search_path).
    ✓ Better isolation than shared table
    ✓ Easier per-tenant migrations
    ✗ PostgreSQL supports ~1000 schemas per DB before slowdown
    ✗ Connection pool complexity (must route by tenant schema)

  Model 3 — Shared Database, Shared Schema:
    All tenants share tables; every row has tenant_id column.
    ✓ Best resource utilization, cheapest
    ✓ Easy to add new tenants (no provisioning)
    ✗ Risk of data leak if tenant_id filter forgotten
    ✗ Noisy neighbor — one tenant's heavy query affects others
    ✗ Harder to do per-tenant compliance (GDPR deletion requires WHERE)
    Use: SMB SaaS, thousands of small tenants

  Hybrid:
    Small tenants on shared schema; large enterprise on dedicated DB.
    Tiered pricing matches infrastructure cost.

Row-Level Security (RLS) — PostgreSQL:
  CREATE POLICY tenant_isolation ON orders
    USING (tenant_id = current_setting('app.tenant_id')::uuid);
  Forces tenant_id filter at DB level — can't forget it in application code.

Query Routing:
  Middleware must set tenant context (schema, tenant_id) before every query.
  Connection pools must be tenant-aware (separate pool per schema, or reset
  search_path per connection checkout).

Tenant Isolation Risks:
  - Missing tenant_id WHERE clause → cross-tenant data leak
  - Shared cache (Redis) with non-namespaced keys → data leak
  - Shared file storage with predictable paths → data leak
  - Logging PII without tenant scoping
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import time
import uuid
import hashlib
from collections import defaultdict


class TenancyModel(Enum):
    SEPARATE_DB    = "separate_db"
    SHARED_SCHEMA  = "shared_schema"    # one schema per tenant
    SHARED_TABLE   = "shared_table"     # shared tables with tenant_id


class TenantTier(Enum):
    FREE       = "free"
    STARTER    = "starter"
    BUSINESS   = "business"
    ENTERPRISE = "enterprise"


@dataclass
class Tenant:
    tenant_id  : str
    name       : str
    tier       : TenantTier
    schema_name: str = ""
    db_url     : str = ""
    row_count  : int = 0
    created_at : float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.schema_name:
            self.schema_name = f"tenant_{self.tenant_id.replace('-', '_')[:8]}"


@dataclass
class QueryResult:
    rows       : List[Dict[str, Any]]
    tenant_id  : str
    query_ms   : float
    row_count  : int = 0

    def __post_init__(self):
        self.row_count = len(self.rows)


# ─────────────────────────────────────────────
# MODEL 1: SEPARATE DATABASE
# ─────────────────────────────────────────────

class SeparateDatabaseModel:
    """
    Each tenant gets a dedicated database.
    High isolation, high operational overhead.
    """

    def __init__(self):
        self._dbs       : Dict[str, Dict] = {}   # tenant_id → {db_url, data}
        self.provisioned = 0

    def provision_tenant(self, tenant: Tenant) -> str:
        db_url = f"postgres://saas-db-{tenant.tenant_id[:6]}.rds.amazonaws.com/app"
        tenant.db_url = db_url
        self._dbs[tenant.tenant_id] = {
            "db_url"  : db_url,
            "tables"  : {"orders": [], "users": [], "products": []},
            "created" : time.time(),
        }
        self.provisioned += 1
        return db_url

    def insert(self, tenant_id: str, table: str, row: Dict) -> bool:
        db = self._dbs.get(tenant_id)
        if not db:
            return False
        db["tables"].setdefault(table, []).append(row)
        return True

    def query(self, tenant_id: str, table: str, where: Dict = None) -> QueryResult:
        start = time.perf_counter()
        db    = self._dbs.get(tenant_id)
        if not db:
            return QueryResult([], tenant_id, 0.0)
        rows = db["tables"].get(table, [])
        if where:
            rows = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
        elapsed = (time.perf_counter() - start) * 1000 + 2.0
        return QueryResult(list(rows), tenant_id, round(elapsed, 2))

    def delete_tenant(self, tenant_id: str):
        """GDPR right to erasure: drop entire database."""
        self._dbs.pop(tenant_id, None)


# ─────────────────────────────────────────────
# MODEL 2: SHARED DB, SEPARATE SCHEMA
# ─────────────────────────────────────────────

class SharedDatabaseSeparateSchema:
    """
    One PostgreSQL instance; each tenant has its own schema (namespace).
    Queries run in tenant schema via search_path setting.
    """

    def __init__(self):
        self._schemas   : Dict[str, Dict] = {}   # schema_name → {tables}
        self._tenant_map: Dict[str, str]  = {}   # tenant_id → schema_name
        self.schema_count = 0

    def provision_tenant(self, tenant: Tenant):
        schema = tenant.schema_name
        self._schemas[schema] = {
            "orders": [], "users": [], "products": []
        }
        self._tenant_map[tenant.tenant_id] = schema
        self.schema_count += 1
        return f"SET search_path = {schema}"

    def insert(self, tenant_id: str, table: str, row: Dict) -> bool:
        schema = self._tenant_map.get(tenant_id)
        if not schema or schema not in self._schemas:
            return False
        self._schemas[schema].setdefault(table, []).append(row)
        return True

    def query(self, tenant_id: str, table: str, where: Dict = None) -> QueryResult:
        start  = time.perf_counter()
        schema = self._tenant_map.get(tenant_id)
        rows   = self._schemas.get(schema, {}).get(table, [])
        if where:
            rows = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
        elapsed = (time.perf_counter() - start) * 1000 + 1.5
        return QueryResult(list(rows), tenant_id, round(elapsed, 2))

    def run_migration(self, schema: str, ddl: str):
        """Per-tenant migration: only affects one schema."""
        if schema in self._schemas:
            return f"Applied to schema '{schema}': {ddl[:50]}"
        return f"Schema '{schema}' not found"


# ─────────────────────────────────────────────
# MODEL 3: SHARED TABLE WITH TENANT_ID
# ─────────────────────────────────────────────

class RowLevelSecurityPolicy:
    """
    Simulates PostgreSQL Row-Level Security.
    Enforces tenant_id filter at the DB level.
    """

    def __init__(self, table: str, policy_col: str = "tenant_id"):
        self.table      = table
        self.policy_col = policy_col
        self._enabled   = False
        self.violations_blocked = 0

    def enable(self):
        self._enabled = True

    def check(self, row: Dict, current_tenant: str) -> bool:
        if not self._enabled:
            return True
        if row.get(self.policy_col) != current_tenant:
            self.violations_blocked += 1
            return False
        return True


class SharedTableModel:
    """
    Shared tables with tenant_id on every row.
    Uses RLS to enforce isolation at DB level.
    """

    def __init__(self):
        self._tables    : Dict[str, List[Dict]] = defaultdict(list)
        self._rls       : Dict[str, RowLevelSecurityPolicy] = {}
        self._current_tenant : Optional[str] = None
        self.total_rows = 0
        self.rls_blocks = 0

    def create_table_with_rls(self, table: str):
        self._tables[table] = []
        policy = RowLevelSecurityPolicy(table)
        policy.enable()
        self._rls[table] = policy

    def set_tenant_context(self, tenant_id: str):
        """SET app.tenant_id = '...' — called per connection checkout."""
        self._current_tenant = tenant_id

    def insert(self, table: str, row: Dict) -> bool:
        if self._current_tenant is None:
            raise RuntimeError("No tenant context set — missing set_tenant_context()")
        row_with_tenant = {"tenant_id": self._current_tenant, **row}
        self._tables[table].append(row_with_tenant)
        self.total_rows += 1
        return True

    def query(self, table: str, where: Dict = None) -> QueryResult:
        """Automatically scopes to current tenant via RLS."""
        start = time.perf_counter()
        if self._current_tenant is None:
            raise RuntimeError("No tenant context set")

        rls    = self._rls.get(table)
        rows   = self._tables.get(table, [])
        # RLS filters by current tenant
        rows   = [r for r in rows if rls.check(r, self._current_tenant)] if rls else rows
        # Additional WHERE filters
        if where:
            rows = [r for r in rows if all(r.get(k) == v for k, v in where.items())]

        elapsed = (time.perf_counter() - start) * 1000 + 0.5
        return QueryResult(list(rows), self._current_tenant, round(elapsed, 2))

    def unsafe_query_without_tenant(self, table: str) -> int:
        """Simulates a bug: query without tenant context."""
        # This returns ALL rows from all tenants — a data leak!
        return len(self._tables.get(table, []))

    def gdpr_delete_tenant(self, tenant_id: str) -> int:
        """GDPR deletion: DELETE FROM all tables WHERE tenant_id = ?"""
        deleted = 0
        for table in self._tables:
            before = len(self._tables[table])
            self._tables[table] = [r for r in self._tables[table]
                                    if r.get("tenant_id") != tenant_id]
            deleted += before - len(self._tables[table])
        return deleted


# ─────────────────────────────────────────────
# TENANT ROUTER (Middleware)
# ─────────────────────────────────────────────

class TenantRouter:
    """
    Middleware that extracts tenant from request and routes to correct DB/schema.
    Tenant can be identified by: subdomain, JWT claim, API key prefix, header.
    """

    def __init__(self):
        self._tenants    : Dict[str, Tenant] = {}
        self._api_keys   : Dict[str, str]    = {}   # api_key → tenant_id
        self._subdomains : Dict[str, str]    = {}   # subdomain → tenant_id
        self.requests_routed = 0

    def register_tenant(self, tenant: Tenant, api_key: str, subdomain: str):
        self._tenants[tenant.tenant_id]    = tenant
        self._api_keys[api_key]            = tenant.tenant_id
        self._subdomains[subdomain]        = tenant.tenant_id

    def resolve_from_subdomain(self, subdomain: str) -> Optional[Tenant]:
        self.requests_routed += 1
        tenant_id = self._subdomains.get(subdomain)
        return self._tenants.get(tenant_id)

    def resolve_from_api_key(self, api_key: str) -> Optional[Tenant]:
        self.requests_routed += 1
        tenant_id = self._api_keys.get(api_key)
        return self._tenants.get(tenant_id)

    def resolve_from_jwt_claim(self, claims: Dict) -> Optional[Tenant]:
        self.requests_routed += 1
        tenant_id = claims.get("tenant_id") or claims.get("org_id")
        return self._tenants.get(tenant_id)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_multi_tenancy():
    print("=" * 65)
    print("MULTI-TENANCY DATABASE DESIGN")
    print("=" * 65)

    # ── Model 1: Separate DB ──────────────────
    print("\n[1] MODEL 1 — SEPARATE DATABASE PER TENANT")
    print("─" * 55)

    sep_model = SeparateDatabaseModel()
    tenants_enterprise = [
        Tenant("ent-001", "AcmeCorp",    TenantTier.ENTERPRISE),
        Tenant("ent-002", "GlobalBank",  TenantTier.ENTERPRISE),
        Tenant("ent-003", "HealthFirst", TenantTier.ENTERPRISE),
    ]
    for t in tenants_enterprise:
        db_url = sep_model.provision_tenant(t)
        print(f"  Provisioned: {t.name} → {db_url}")

    # Insert and query — completely isolated
    sep_model.insert("ent-001", "orders", {"order_id": "o1", "amount": 5000})
    sep_model.insert("ent-002", "orders", {"order_id": "o2", "amount": 12000})

    result_1 = sep_model.query("ent-001", "orders")
    result_2 = sep_model.query("ent-002", "orders")
    print(f"\n  AcmeCorp   orders: {result_1.row_count} rows")
    print(f"  GlobalBank orders: {result_2.row_count} rows")
    print(f"  (No cross-tenant access possible — separate databases)")

    # GDPR deletion
    sep_model.delete_tenant("ent-003")
    print(f"\n  HealthFirst GDPR deletion: dropped database entirely")
    print(f"  Remaining DBs provisioned: {sep_model.provisioned} (3 total, 1 dropped)")

    # ── Model 2: Separate Schema ───────────────
    print("\n\n[2] MODEL 2 — SHARED DB, SEPARATE SCHEMA")
    print("─" * 55)

    schema_model = SharedDatabaseSeparateSchema()
    tenants_business = [
        Tenant("biz-001", "StartupA", TenantTier.BUSINESS),
        Tenant("biz-002", "StartupB", TenantTier.BUSINESS),
    ]
    for t in tenants_business:
        cmd = schema_model.provision_tenant(t)
        print(f"  Provisioned: {t.name} → schema={t.schema_name}  ({cmd})")

    schema_model.insert("biz-001", "users", {"name": "Alice", "email": "alice@a.com"})
    schema_model.insert("biz-001", "users", {"name": "Bob",   "email": "bob@a.com"})
    schema_model.insert("biz-002", "users", {"name": "Carol", "email": "carol@b.com"})

    r1 = schema_model.query("biz-001", "users")
    r2 = schema_model.query("biz-002", "users")
    print(f"\n  StartupA users: {r1.row_count}  StartupB users: {r2.row_count}")
    print(f"  Total schemas in DB: {schema_model.schema_count}")

    # Per-tenant migration
    result = schema_model.run_migration(tenants_business[0].schema_name,
                                         "ALTER TABLE orders ADD COLUMN notes TEXT")
    print(f"\n  Per-tenant migration: {result}")
    print(f"  (Migration only affects StartupA's schema — others untouched)")

    # ── Model 3: Shared Table ──────────────────
    print("\n\n[3] MODEL 3 — SHARED TABLE WITH tenant_id + RLS")
    print("─" * 55)

    shared = SharedTableModel()
    shared.create_table_with_rls("orders")
    shared.create_table_with_rls("users")

    # Tenant A inserts data
    shared.set_tenant_context("tenant-A")
    shared.insert("orders", {"order_id": "oa1", "product": "Widget A", "amount": 99})
    shared.insert("orders", {"order_id": "oa2", "product": "Widget B", "amount": 149})
    shared.insert("users",  {"name": "Alice"})

    # Tenant B inserts data
    shared.set_tenant_context("tenant-B")
    shared.insert("orders", {"order_id": "ob1", "product": "Gadget X", "amount": 299})
    shared.insert("users",  {"name": "Bob"})

    # Tenant A queries — only sees own rows (RLS enforced)
    shared.set_tenant_context("tenant-A")
    result_a = shared.query("orders")
    print(f"  Tenant A queries orders: sees {result_a.row_count} rows (own only)")
    for r in result_a.rows:
        print(f"    {r}")

    # Tenant B queries — only sees own rows
    shared.set_tenant_context("tenant-B")
    result_b = shared.query("orders")
    print(f"\n  Tenant B queries orders: sees {result_b.row_count} rows (own only)")

    # Show total rows in table (both tenants combined)
    total = shared.unsafe_query_without_tenant("orders")
    print(f"\n  Total rows in orders table (all tenants): {total}")
    print(f"  → RLS ensures tenants only see their {result_a.row_count} or {result_b.row_count} rows")

    # GDPR deletion by tenant_id
    deleted = shared.gdpr_delete_tenant("tenant-A")
    print(f"\n  Tenant A GDPR deletion: removed {deleted} rows across all tables")

    # ── Tenant Router ─────────────────────────
    print("\n\n[4] TENANT ROUTING MIDDLEWARE")
    print("─" * 55)
    router  = TenantRouter()
    tenant1 = Tenant("t-aaa", "AcmeCorp",  TenantTier.ENTERPRISE)
    tenant2 = Tenant("t-bbb", "StartupX",  TenantTier.STARTER)

    router.register_tenant(tenant1, api_key="key_acme_12345", subdomain="acme")
    router.register_tenant(tenant2, api_key="key_startx_789", subdomain="startupx")

    # Resolve by subdomain (request to acme.saas.com)
    t = router.resolve_from_subdomain("acme")
    print(f"  subdomain=acme       → tenant: {t.name} ({t.tier.value})")

    # Resolve by API key
    t = router.resolve_from_api_key("key_startx_789")
    print(f"  api_key=key_startx   → tenant: {t.name} ({t.tier.value})")

    # Resolve from JWT
    t = router.resolve_from_jwt_claim({"tenant_id": "t-aaa", "user": "admin"})
    print(f"  JWT claim tenant_id  → tenant: {t.name} ({t.tier.value})")

    print(f"\n  Total requests routed: {router.requests_routed}")

    # ── Model Comparison ───────────────────────
    print("\n\n[5] TENANCY MODEL COMPARISON")
    print("─" * 55)
    comparison = [
        ("Isolation",          "Maximum",        "Schema-level",    "Row-level (RLS)"),
        ("Operational cost",   "Very high",      "Medium",          "Low"),
        ("Tenant provisioning","Slow (new DB)",  "Fast (new schema)","Instant (no DDL)"),
        ("GDPR deletion",      "Drop database",  "Drop schema",     "DELETE WHERE tenant_id"),
        ("Noisy neighbor",     "None",           "Partial",         "Risk — shared resources"),
        ("Scale",              "100s of tenants","1000s",           "100,000s"),
        ("Per-tenant backup",  "Easy",           "pg_dump schema",  "Complex (filter export)"),
        ("Best for",           "Enterprise/HIPAA","Mid-market","SMB/high volume"),
    ]
    print(f"  {'Aspect':<22} {'Separate DB':<18} {'Sep Schema':<18} {'Shared Table'}")
    print(f"  {'─'*75}")
    for row in comparison:
        aspect, sep_db, sep_schema, shared_tbl = row
        print(f"  {aspect:<22} {sep_db:<18} {sep_schema:<18} {shared_tbl}")

    # ── Security Checklist ─────────────────────
    print("\n\n[6] MULTI-TENANCY SECURITY CHECKLIST")
    print("─" * 55)
    checks = [
        ("DB queries",    "Every query has tenant_id filter or uses RLS policy"),
        ("Cache keys",    "Redis keys namespaced: tenant:{id}:user:{uid}"),
        ("File storage",  "S3 paths: /{tenant_id}/... (never predictable global paths)"),
        ("Logging",       "Logs tagged with tenant_id for audit trail"),
        ("API responses", "Always validate resource belongs to request tenant"),
        ("Webhooks",      "Webhook URL validated against tenant whitelist"),
        ("Rate limits",   "Per-tenant limits (not per-IP — shared IPs for orgs)"),
        ("Encryption",    "Per-tenant encryption keys (KMS) for sensitive data"),
    ]
    for area, check in checks:
        print(f"  ✓ {area:<16} {check}")


if __name__ == "__main__":
    demonstrate_multi_tenancy()
