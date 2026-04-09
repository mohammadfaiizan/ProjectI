"""
SQL SCHEMA DESIGN BEST PRACTICES
===================================

Problem Statement:
Poor schema design leads to data integrity issues, slow queries, painful
migrations, and accumulating technical debt. Good schema design is critical
for long-term system health.

Normalization Forms:
  1NF: Each column holds atomic values; no repeating groups
  2NF: 1NF + every non-key column depends on the WHOLE primary key
  3NF: 2NF + no transitive dependencies (A→B→C, remove C to separate table)
  BCNF: Stricter 3NF — every determinant is a candidate key

When to Denormalize:
  - Read-heavy workloads where joins are expensive
  - Pre-computed aggregates (total_orders on users table)
  - CQRS read model (intentionally denormalized)

Key Design Principles:
  1. Use surrogate keys (UUID/bigserial) for PKs
  2. Use natural keys for UNIQUE constraints, not PKs
  3. Always add created_at, updated_at
  4. Use soft deletes (deleted_at) for audit trail
  5. Choose correct types (TIMESTAMPTZ, DECIMAL not FLOAT for money)
  6. Foreign keys + NOT NULL constraints = data integrity
  7. Index foreign keys (not done automatically in some DBs)
  8. Keep rows small — wide tables hurt performance

Common Mistakes:
  ❌ Using VARCHAR for enum values → use ENUM or lookup table
  ❌ Storing JSON blobs for structured data → proper columns
  ❌ Missing indexes on foreign keys
  ❌ FLOAT for money → use DECIMAL(19, 4)
  ❌ VARCHAR(255) everywhere → choose correct size
  ❌ No soft deletes → can't audit who deleted what
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time


class NormalizationForm(Enum):
    UNF  = "Unnormalized Form"
    NF1  = "First Normal Form"
    NF2  = "Second Normal Form"
    NF3  = "Third Normal Form"
    BCNF = "Boyce-Codd Normal Form"


@dataclass
class Column:
    name        : str
    data_type   : str
    nullable    : bool = True
    default     : Optional[str] = None
    constraint  : Optional[str] = None

    def ddl(self) -> str:
        parts = [f"{self.name} {self.data_type}"]
        if not self.nullable:
            parts.append("NOT NULL")
        if self.default is not None:
            parts.append(f"DEFAULT {self.default}")
        if self.constraint:
            parts.append(self.constraint)
        return " ".join(parts)


@dataclass
class TableDefinition:
    name        : str
    columns     : List[Column]
    primary_key : List[str]
    indexes     : List[str] = field(default_factory=list)
    foreign_keys: List[str] = field(default_factory=list)
    checks      : List[str] = field(default_factory=list)
    comment     : str = ""

    def ddl(self) -> str:
        lines = [f"CREATE TABLE {self.name} ("]
        for col in self.columns:
            lines.append(f"    {col.ddl()},")
        lines.append(f"    PRIMARY KEY ({', '.join(self.primary_key)})")
        for fk in self.foreign_keys:
            lines.append(f"    , FOREIGN KEY {fk}")
        for chk in self.checks:
            lines.append(f"    , CHECK ({chk})")
        lines.append(");")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# NORMALIZATION DEMONSTRATOR
# ─────────────────────────────────────────────

class NormalizationDemo:
    """Shows step-by-step normalization of an orders table."""

    @staticmethod
    def show_unnormalized():
        print("  Unnormalized (UNF) — PROBLEM: repeating groups, no atomicity")
        rows = [
            {"order_id": 1, "customer": "Alice",  "customer_city": "NYC",
             "items": "Laptop×1,Mouse×2", "total": 1059.97},
            {"order_id": 2, "customer": "Bob",    "customer_city": "LA",
             "items": "Keyboard×1",         "total": 79.99},
        ]
        print(f"  {'order_id':<10} {'customer':<10} {'city':<8} {'items':<25} {'total'}")
        print(f"  {'─'*65}")
        for r in rows:
            print(f"  {r['order_id']:<10} {r['customer']:<10} {r['customer_city']:<8} "
                  f"{r['items']:<25} {r['total']}")
        print("  Problems: items not atomic, can't query individual items")

    @staticmethod
    def show_1nf():
        print("\n  1NF — Atomic values, no repeating groups")
        rows = [
            (1, "Alice", "NYC", "Laptop",   1, 999.99),
            (1, "Alice", "NYC", "Mouse",    2, 29.99),
            (2, "Bob",   "LA",  "Keyboard", 1, 79.99),
        ]
        print(f"  {'order_id':<10} {'customer':<10} {'city':<6} {'product':<12} {'qty':<5} {'price'}")
        print(f"  {'─'*55}")
        for r in rows:
            print(f"  {r[0]:<10} {r[1]:<10} {r[2]:<6} {r[3]:<12} {r[4]:<5} {r[5]}")
        print("  Problem: customer/city repeated; non-key cols depend on partial key")

    @staticmethod
    def show_2nf():
        print("\n  2NF — Every non-key col depends on WHOLE key")
        print("  Split: orders table + order_items table")
        orders = [(1, "Alice", "NYC"), (2, "Bob", "LA")]
        items  = [(1, "Laptop", 1, 999.99), (1, "Mouse", 2, 29.99), (2, "Keyboard", 1, 79.99)]
        print("\n  orders:")
        print(f"  {'order_id':<10} {'customer':<12} {'city'}")
        for r in orders:
            print(f"  {r[0]:<10} {r[1]:<12} {r[2]}")
        print("\n  order_items:")
        print(f"  {'order_id':<10} {'product':<12} {'qty':<5} {'price'}")
        for r in items:
            print(f"  {r[0]:<10} {r[1]:<12} {r[2]:<5} {r[3]}")
        print("  Problem: customer_city depends on customer, not on order_id")

    @staticmethod
    def show_3nf():
        print("\n  3NF — No transitive dependencies")
        print("  Split: customers table (removes customer→city transitive dep)")
        customers = [(1, "Alice", "NYC"), (2, "Bob", "LA")]
        orders    = [(1, 1), (2, 2)]   # order_id, customer_id
        items     = [(1, "Laptop", 1, 999.99), (1, "Mouse", 2, 29.99), (2, "Keyboard", 1, 79.99)]
        print("\n  customers:")
        print(f"  {'customer_id':<12} {'name':<10} {'city'}")
        for r in customers:
            print(f"  {r[0]:<12} {r[1]:<10} {r[2]}")
        print("\n  orders:")
        print(f"  {'order_id':<10} {'customer_id'}")
        for r in orders:
            print(f"  {r[0]:<10} {r[1]}")
        print("\n  ✅ All non-key columns depend ONLY on the primary key")


# ─────────────────────────────────────────────
# PRODUCTION SCHEMA PATTERNS
# ─────────────────────────────────────────────

class ProductionSchemaPatterns:
    @staticmethod
    def user_table() -> str:
        return """
  -- ✅ Good user table design
  CREATE TABLE users (
      id           BIGSERIAL PRIMARY KEY,        -- surrogate key
      uuid         UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,  -- exposed in APIs
      email        VARCHAR(255) NOT NULL UNIQUE,
      name         VARCHAR(100) NOT NULL,
      password_hash VARCHAR(60) NOT NULL,         -- bcrypt hash
      status       VARCHAR(20) NOT NULL DEFAULT 'active'
                       CHECK (status IN ('active', 'inactive', 'suspended')),
      timezone     VARCHAR(50) NOT NULL DEFAULT 'UTC',
      created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      deleted_at   TIMESTAMPTZ,                  -- soft delete
      version      INTEGER NOT NULL DEFAULT 1    -- optimistic locking
  );

  CREATE INDEX idx_users_email        ON users(email);
  CREATE INDEX idx_users_uuid         ON users(uuid);
  CREATE INDEX idx_users_status_created ON users(status, created_at)
      WHERE deleted_at IS NULL;                  -- partial index: active only
  CREATE INDEX idx_users_deleted      ON users(deleted_at)
      WHERE deleted_at IS NOT NULL;"""

    @staticmethod
    def orders_table() -> str:
        return """
  -- ✅ Good orders table design
  CREATE TABLE orders (
      id            BIGSERIAL PRIMARY KEY,
      uuid          UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
      user_id       BIGINT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
      status        VARCHAR(20) NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','confirmed','shipped','delivered','cancelled')),
      total_usd     DECIMAL(19, 4) NOT NULL,     -- NEVER use FLOAT for money
      currency      CHAR(3) NOT NULL DEFAULT 'USD',
      shipping_addr JSONB NOT NULL,              -- flexible address struct
      notes         TEXT,                        -- nullable, unbounded
      created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      shipped_at    TIMESTAMPTZ,
      cancelled_at  TIMESTAMPTZ,
      cancelled_reason VARCHAR(500)
  );

  CREATE INDEX idx_orders_user_id    ON orders(user_id);          -- FK always indexed
  CREATE INDEX idx_orders_status     ON orders(status, created_at DESC);
  CREATE INDEX idx_orders_user_status ON orders(user_id, status)
      WHERE status NOT IN ('delivered', 'cancelled');             -- partial: active orders"""

    @staticmethod
    def common_mistakes() -> List[Tuple[str, str, str]]:
        return [
            ("FLOAT for money",         "FLOAT is imprecise",
             "DECIMAL(19, 4)"),
            ("VARCHAR(255) everywhere", "Wastes index space",
             "VARCHAR(100) for name, TEXT for content"),
            ("Missing FK index",        "JOIN scans without index",
             "Always CREATE INDEX idx_table_fk ON table(fk_col)"),
            ("TIMESTAMP not TIMESTAMPTZ","Loses timezone info",
             "TIMESTAMPTZ (stores UTC, displays in session timezone)"),
            ("Hard delete rows",        "Can't audit deletes",
             "Soft delete: deleted_at TIMESTAMPTZ"),
            ("NULL in CHECK constraint","NULL passes all CHECKs",
             "NOT NULL + CHECK combo"),
            ("Storing JSON for struct cols","Can't index/query inside",
             "Proper columns OR JSONB with GIN index"),
            ("ID in API responses",     "Exposes DB sequence, enables enumeration",
             "Expose UUID instead of bigserial ID"),
        ]


# ─────────────────────────────────────────────
# MIGRATION BEST PRACTICES
# ─────────────────────────────────────────────

class MigrationBestPractices:
    @staticmethod
    def safe_migration_steps() -> str:
        return """
  Safe schema migration steps (zero-downtime):

  Add column (safe):
    ALTER TABLE orders ADD COLUMN notes TEXT;          -- instant, nullable
    -- Deploy new code that writes notes
    ALTER TABLE orders ALTER COLUMN notes SET DEFAULT ''; -- optional

  Drop column (multi-step):
    1. Deploy code that ignores the column
    2. ALTER TABLE orders DROP COLUMN old_col;         -- safe after deploy

  Add NOT NULL column (multi-step):
    1. ALTER TABLE orders ADD COLUMN priority INT;     -- nullable first
    2. UPDATE orders SET priority = 0 WHERE priority IS NULL;  -- backfill
    3. ALTER TABLE orders ALTER COLUMN priority SET NOT NULL;

  Add index (non-blocking in PostgreSQL):
    CREATE INDEX CONCURRENTLY idx_orders_priority ON orders(priority);

  Rename column (multi-step):
    1. Add new_col with same type, copy data via trigger
    2. Deploy code that reads new_col (falls back to old_col)
    3. Drop old_col after verifying
  """


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_schema_design():
    print("=" * 65)
    print("SQL SCHEMA DESIGN BEST PRACTICES")
    print("=" * 65)

    # ── Normalization ──────────────────────────
    print("\n[1] NORMALIZATION — STEP BY STEP")
    print("─" * 55)
    demo = NormalizationDemo()
    demo.show_unnormalized()
    demo.show_1nf()
    demo.show_2nf()
    demo.show_3nf()

    # ── Production Schemas ────────────────────
    print("\n\n[2] PRODUCTION-QUALITY USER TABLE")
    print("─" * 55)
    print(ProductionSchemaPatterns.user_table())

    print("\n\n[3] PRODUCTION-QUALITY ORDERS TABLE")
    print("─" * 55)
    print(ProductionSchemaPatterns.orders_table())

    # ── Common Mistakes ───────────────────────
    print("\n\n[4] COMMON SCHEMA MISTAKES AND FIXES")
    print("─" * 55)
    print(f"  {'Mistake':<35} {'Problem':<32} {'Fix'}")
    print(f"  {'─'*85}")
    for mistake, problem, fix in ProductionSchemaPatterns.common_mistakes():
        print(f"  {mistake:<35} {problem:<32} {fix}")

    # ── Migration ─────────────────────────────
    print("\n\n[5] ZERO-DOWNTIME MIGRATION PATTERNS")
    print("─" * 55)
    print(MigrationBestPractices.safe_migration_steps())

    # ── Data Types Guide ──────────────────────
    print("\n\n[6] COLUMN TYPE GUIDE")
    print("─" * 55)
    types = [
        ("Primary key",      "BIGSERIAL",           "Auto-incrementing 64-bit integer"),
        ("Exposed ID",       "UUID",                "Never expose BIGSERIAL externally"),
        ("Short text",       "VARCHAR(N)",          "Set N based on actual max length"),
        ("Long text",        "TEXT",                "Unbounded; no VARCHAR(MAX)"),
        ("Money",            "DECIMAL(19, 4)",      "Never FLOAT — precision errors"),
        ("Datetime",         "TIMESTAMPTZ",         "Always include timezone"),
        ("Date only",        "DATE",                "No time component"),
        ("Duration",         "INTERVAL",            "Better than storing seconds"),
        ("Boolean",          "BOOLEAN",             "Not INT(1)"),
        ("JSON (flexible)",  "JSONB",               "Binary JSON — indexable"),
        ("Enum",             "VARCHAR + CHECK",     "Or lookup table for mutability"),
        ("Large binary",     "Object Storage (S3)", "Don't store blobs in DB"),
    ]
    print(f"  {'Use Case':<20} {'Type':<22} {'Why'}")
    print(f"  {'─'*70}")
    for use_case, dtype, why in types:
        print(f"  {use_case:<20} {dtype:<22} {why}")


if __name__ == "__main__":
    demonstrate_schema_design()
