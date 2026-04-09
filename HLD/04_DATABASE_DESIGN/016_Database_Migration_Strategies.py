"""
DATABASE MIGRATION STRATEGIES
================================

Problem Statement:
Deployed applications must evolve their schemas over time without downtime.
A naive ALTER TABLE on a 100M-row table can lock the table for hours,
causing a complete outage. Zero-downtime migrations require careful planning.

Migration Tools:
  Flyway   : SQL-file based, version-ordered (V1__init.sql, V2__add_col.sql)
  Liquibase: XML/YAML changesets with checksum verification
  Alembic  : Python (SQLAlchemy-based), revision chain
  Django   : makemigrations / migrate, ORM-generated DDL

Zero-Downtime Migration Patterns:

  Expand-Contract (Blue-Green Column):
    Phase 1 (Expand): Add new column (nullable), keep old column.
                      App writes to both, reads from old.
    Phase 2 (Migrate): Backfill data in batches (don't lock table).
    Phase 3 (Contract): App reads from new column.
    Phase 4 (Cleanup): Drop old column.

  Rename Column (Safe Steps):
    1. Add new column
    2. Dual-write to both
    3. Backfill old → new
    4. Switch reads to new
    5. Drop old column
    ❌ Never: ALTER TABLE RENAME COLUMN (breaks running queries)

  Adding Index (Safe):
    PostgreSQL: CREATE INDEX CONCURRENTLY (doesn't lock writes)
    MySQL: Online DDL (ALGORITHM=INPLACE, LOCK=NONE)

  Online Schema Change (OSC):
    gh-ost (GitHub): shadow table + changelog triggers + binlog replay
    pt-online-schema-change (Percona): trigger-based copy

  Blue-Green Database:
    Maintain two DB instances (blue=current, green=new schema).
    Run migrations on green, dual-write, cut over, keep blue for rollback.

Common Mistakes:
  - Adding NOT NULL column without DEFAULT (locks table in Postgres < 11)
  - Dropping column still used by running app version
  - Renaming column/table directly
  - Running migration in a single long transaction (holds locks)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import time
import uuid
import hashlib
from collections import defaultdict


class MigrationStatus(Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    SUCCESS  = "success"
    FAILED   = "failed"
    SKIPPED  = "skipped"


class RiskLevel(Enum):
    LOW    = "low"      # online, no lock
    MEDIUM = "medium"   # brief lock or replication lag
    HIGH   = "high"     # full table lock, downtime risk


@dataclass
class Migration:
    version     : str
    description : str
    sql_up      : str
    sql_down    : str
    risk        : RiskLevel = RiskLevel.LOW
    status      : MigrationStatus = MigrationStatus.PENDING
    checksum    : str = ""
    applied_at  : Optional[float] = None
    duration_ms : float = 0.0

    def __post_init__(self):
        self.checksum = hashlib.md5(self.sql_up.encode()).hexdigest()[:8]

    def __str__(self):
        return f"V{self.version}__{self.description}"


@dataclass
class SchemaVersion:
    """Tracks applied migration versions (like flyway_schema_history table)."""
    version     : str
    description : str
    checksum    : str
    applied_at  : float
    duration_ms : float
    success     : bool


# ─────────────────────────────────────────────
# SIMULATED DATABASE
# ─────────────────────────────────────────────

class SimulatedDB:
    """Simplified DB that tracks schema state and table sizes."""

    def __init__(self, name: str):
        self.name        = name
        self._tables     : Dict[str, Dict] = {}   # table_name → {columns, row_count, indexes}
        self._schema_history : List[SchemaVersion] = []
        self.ddl_log     : List[str] = []
        self.lock_events : List[str] = []

    def execute_ddl(self, sql: str, locks_table: bool = False, duration_ms: float = 10.0):
        self.ddl_log.append(sql.strip()[:80])
        if locks_table:
            self.lock_events.append(f"LOCK: {sql.strip()[:50]} ({duration_ms:.0f}ms)")
        time.sleep(duration_ms / 1000)

    def add_table(self, name: str, columns: List[str], row_count: int = 0):
        self._tables[name] = {
            "columns": list(columns),
            "indexes": [],
            "row_count": row_count,
        }

    def add_column(self, table: str, column: str, nullable: bool = True,
                   default: str = None, row_count_scale: int = 1):
        t = self._tables.get(table)
        if t:
            t["columns"].append(column)
            # PostgreSQL 11+: ADD COLUMN with DEFAULT is instant (catalog update only)
            # Older: rewrites entire table
            locks = not nullable and default is None
            duration = t["row_count"] * 0.001 if locks else 5.0
            self.execute_ddl(
                f"ALTER TABLE {table} ADD COLUMN {column}"
                + (" DEFAULT '" + default + "'" if default else "")
                + ("" if nullable else " NOT NULL"),
                locks_table=locks,
                duration_ms=duration
            )

    def add_index(self, table: str, column: str, concurrent: bool = True):
        t = self._tables.get(table)
        if t:
            idx_name = f"idx_{table}_{column}"
            t["indexes"].append(idx_name)
            sql = f"CREATE INDEX {'CONCURRENTLY ' if concurrent else ''}{idx_name} ON {table}({column})"
            # CONCURRENTLY: no table lock but takes longer
            duration = t["row_count"] * 0.002 if not concurrent else t["row_count"] * 0.003
            self.execute_ddl(sql, locks_table=not concurrent, duration_ms=max(5.0, duration))

    def drop_column(self, table: str, column: str):
        t = self._tables.get(table)
        if t and column in t["columns"]:
            t["columns"].remove(column)
            self.execute_ddl(f"ALTER TABLE {table} DROP COLUMN {column}", duration_ms=5.0)

    def table_info(self, table: str) -> Dict:
        return self._tables.get(table, {})

    def record_migration(self, m: Migration):
        self._schema_history.append(SchemaVersion(
            version=m.version, description=m.description,
            checksum=m.checksum, applied_at=m.applied_at or time.time(),
            duration_ms=m.duration_ms, success=(m.status == MigrationStatus.SUCCESS)
        ))


# ─────────────────────────────────────────────
# MIGRATION RUNNER (Flyway-like)
# ─────────────────────────────────────────────

class MigrationRunner:
    """
    Flyway/Liquibase-style migration runner.
    Applies pending migrations in version order.
    Verifies checksums of already-applied migrations.
    """

    def __init__(self, db: SimulatedDB):
        self.db        = db
        self._applied  : Dict[str, str] = {}   # version → checksum
        self.migrations: List[Migration] = []

    def register(self, migration: Migration):
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)

    def run(self) -> Tuple[int, int]:
        applied_count = 0
        skipped_count = 0

        for m in self.migrations:
            if m.version in self._applied:
                # Verify checksum integrity
                if self._applied[m.version] != m.checksum:
                    raise RuntimeError(
                        f"Checksum mismatch for V{m.version}: "
                        f"expected {self._applied[m.version]}, got {m.checksum}"
                    )
                m.status = MigrationStatus.SKIPPED
                skipped_count += 1
                continue

            # Apply migration
            m.status = MigrationStatus.RUNNING
            start = time.perf_counter()
            try:
                self.db.execute_ddl(m.sql_up, duration_ms=5.0)
                m.status      = MigrationStatus.SUCCESS
                m.applied_at  = time.time()
                m.duration_ms = (time.perf_counter() - start) * 1000
                self._applied[m.version] = m.checksum
                self.db.record_migration(m)
                applied_count += 1
                print(f"    Applied V{m.version}: {m.description} "
                      f"({m.duration_ms:.1f}ms) [{m.risk.value} risk]")
            except Exception as e:
                m.status = MigrationStatus.FAILED
                print(f"    FAILED V{m.version}: {e}")
                break

        return applied_count, skipped_count

    def rollback(self, to_version: str):
        """Roll back migrations newer than to_version."""
        to_rollback = [
            m for m in reversed(self.migrations)
            if m.version > to_version and m.status == MigrationStatus.SUCCESS
        ]
        for m in to_rollback:
            print(f"    Rolling back V{m.version}: {m.description}")
            self.db.execute_ddl(m.sql_down, duration_ms=5.0)
            m.status = MigrationStatus.PENDING
            self._applied.pop(m.version, None)

    def status_report(self):
        print(f"\n  Migration Status for '{self.db.name}':")
        for m in self.migrations:
            icon = {"pending": "○", "running": "→", "success": "✓",
                    "failed": "✗", "skipped": "="}.get(m.status.value, "?")
            print(f"    {icon} V{m.version} {m.description:<40} "
                  f"[{m.status.value}] checksum={m.checksum}")


# ─────────────────────────────────────────────
# EXPAND-CONTRACT PATTERN
# ─────────────────────────────────────────────

class ExpandContractMigration:
    """
    Demonstrates safe column rename using expand-contract pattern.
    Renames `full_name` → `display_name` without downtime.
    """

    def __init__(self, db: SimulatedDB, table: str, row_count: int):
        self.db        = db
        self.table     = table
        self.row_count = row_count
        self.phase     = 0
        self._log      : List[str] = []

    def _log_phase(self, phase: str, details: str):
        self._log.append(f"  Phase {self.phase}: {phase}")
        self._log.append(f"    {details}")

    def phase1_expand(self):
        """Add new column (nullable). App writes to both. Reads from old."""
        self.phase = 1
        self.db.add_column(self.table, "display_name", nullable=True)
        self._log_phase("EXPAND",
            "Added 'display_name' (nullable). App v2: writes to both full_name + display_name.")

    def phase2_backfill(self, batch_size: int = 10_000):
        """Backfill data in small batches to avoid long locks."""
        self.phase = 2
        batches     = max(1, self.row_count // batch_size)
        total_time  = batches * 2   # 2ms per batch (simulated)
        self._log_phase("BACKFILL",
            f"UPDATE {self.table} SET display_name=full_name WHERE display_name IS NULL "
            f"LIMIT {batch_size}  ({batches} batches, ~{total_time}ms total, no lock)")

    def phase3_switch_reads(self):
        """App reads from new column. Old still written for rollback safety."""
        self.phase = 3
        self._log_phase("SWITCH READS",
            "App v3: reads from 'display_name'. Still dual-writing for rollback.")

    def phase4_contract(self):
        """Drop old column after all app instances are on v3."""
        self.phase = 4
        self.db.drop_column(self.table, "full_name")
        self._log_phase("CONTRACT",
            "Dropped 'full_name'. Migration complete. App v4 reads/writes only display_name.")

    def run_all_phases(self):
        self.phase1_expand()
        self.phase2_backfill()
        self.phase3_switch_reads()
        self.phase4_contract()
        for line in self._log:
            print(line)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_db_migration():
    print("=" * 65)
    print("DATABASE MIGRATION STRATEGIES")
    print("=" * 65)

    # ── Flyway-style Migration Runner ─────────
    print("\n[1] FLYWAY-STYLE VERSIONED MIGRATIONS")
    print("─" * 55)

    db     = SimulatedDB("app_db")
    runner = MigrationRunner(db)

    # Define migrations
    migrations = [
        Migration("001", "create_users_table",
                  sql_up="CREATE TABLE users (id BIGSERIAL PRIMARY KEY, email TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW())",
                  sql_down="DROP TABLE users",
                  risk=RiskLevel.LOW),
        Migration("002", "add_name_column",
                  sql_up="ALTER TABLE users ADD COLUMN full_name TEXT",
                  sql_down="ALTER TABLE users DROP COLUMN full_name",
                  risk=RiskLevel.LOW),
        Migration("003", "create_orders_table",
                  sql_up="CREATE TABLE orders (id BIGSERIAL PRIMARY KEY, user_id BIGINT REFERENCES users(id), total_cents INT NOT NULL, status TEXT DEFAULT 'pending')",
                  sql_down="DROP TABLE orders",
                  risk=RiskLevel.LOW),
        Migration("004", "add_index_orders_user_id",
                  sql_up="CREATE INDEX CONCURRENTLY idx_orders_user_id ON orders(user_id)",
                  sql_down="DROP INDEX idx_orders_user_id",
                  risk=RiskLevel.LOW),
        Migration("005", "add_orders_created_at",
                  sql_up="ALTER TABLE orders ADD COLUMN created_at TIMESTAMPTZ DEFAULT NOW()",
                  sql_down="ALTER TABLE orders DROP COLUMN created_at",
                  risk=RiskLevel.LOW),
    ]
    for m in migrations:
        runner.register(m)

    applied, skipped = runner.run()
    print(f"\n  Applied: {applied}  Skipped: {skipped}")
    runner.status_report()

    # Re-run to show idempotency
    print(f"\n  Re-running migrations (already applied):")
    applied2, skipped2 = runner.run()
    print(f"  Applied: {applied2}  Skipped: {skipped2} (idempotent — no re-apply)")

    # ── Expand-Contract Pattern ───────────────
    print("\n\n[2] EXPAND-CONTRACT (ZERO-DOWNTIME COLUMN RENAME)")
    print("─" * 55)
    print("  Goal: rename 'full_name' → 'display_name' on 5M-row users table")
    print()

    db2 = SimulatedDB("production_db")
    db2.add_table("users", ["id", "email", "full_name"], row_count=5_000_000)

    expand_contract = ExpandContractMigration(db2, "users", row_count=5_000_000)
    expand_contract.run_all_phases()

    cols = db2.table_info("users")["columns"]
    print(f"\n  Final columns: {cols}")

    # ── Risky vs Safe Patterns ─────────────────
    print("\n\n[3] RISKY vs SAFE MIGRATION PATTERNS")
    print("─" * 55)
    patterns = [
        ("ADD COLUMN with DEFAULT",
         "ALTER TABLE t ADD COLUMN x INT DEFAULT 0 NOT NULL",
         "ALTER TABLE t ADD COLUMN x INT DEFAULT 0 NOT NULL",
         "PG 11+: instant (catalog). PG <11: full table rewrite",
         RiskLevel.LOW),
        ("ADD COLUMN nullable",
         "ALTER TABLE t ADD COLUMN x TEXT",
         "ALTER TABLE t ADD COLUMN x TEXT",
         "Always instant, no lock",
         RiskLevel.LOW),
        ("ADD INDEX blocking",
         "CREATE INDEX idx ON t(col)",
         "CREATE INDEX CONCURRENTLY idx ON t(col)",
         "CONCURRENT: no write lock, takes 2-3x longer",
         RiskLevel.LOW),
        ("DROP COLUMN",
         "ALTER TABLE t DROP COLUMN old_col",
         "Expand-contract: dual-write → backfill → switch reads → drop",
         "Direct drop safe if no running app uses it",
         RiskLevel.MEDIUM),
        ("RENAME TABLE",
         "ALTER TABLE t RENAME TO t_new",
         "Create new table + trigger copy + atomic swap",
         "Direct rename breaks running queries + cached plans",
         RiskLevel.HIGH),
        ("ADD NOT NULL col no default",
         "ALTER TABLE t ADD COLUMN x INT NOT NULL",
         "ADD nullable → backfill → set NOT NULL constraint",
         "Without default: full table rewrite on Postgres",
         RiskLevel.HIGH),
    ]
    print(f"  {'Operation':<30} {'Risk':<8} Note")
    print(f"  {'─'*70}")
    for op, risky, safe, note, risk in patterns:
        print(f"  {op:<30} {risk.value:<8} {note}")
        if risky != safe:
            print(f"    ❌ Risky : {risky[:70]}")
            print(f"    ✓ Safe  : {safe[:70]}")

    # ── Online Schema Change ───────────────────
    print("\n\n[4] ONLINE SCHEMA CHANGE (gh-ost / pt-osc)")
    print("─" * 55)
    print("  Problem: ALTER TABLE on 100M rows = full table lock for hours")
    print()
    steps = [
        ("1", "Create shadow table with new schema",       "No lock"),
        ("2", "Apply changelog via binlog replication",    "No lock — continuous sync"),
        ("3", "Backfill rows in small batches",            "Throttled — no replication lag"),
        ("4", "Wait for shadow table to catch up",         "Monitor lag < 100ms"),
        ("5", "Atomic table swap (RENAME)",                "Brief lock ~1-2ms"),
        ("6", "Drop old table",                            "Background cleanup"),
    ]
    for num, step, note in steps:
        print(f"  Step {num}: {step:<45} [{note}]")

    print(f"\n  gh-ost advantages over pt-osc:")
    print(f"    • Reads binlog directly (no triggers)")
    print(f"    • Pauseable/throttleable without dropping triggers")
    print(f"    • Cut-over is atomic and safe to retry")

    # ── Blue-Green Database ────────────────────
    print("\n\n[5] BLUE-GREEN DATABASE MIGRATION")
    print("─" * 55)
    phases = [
        ("1. Setup",         "Provision green DB with new schema (empty)"),
        ("2. Backfill",      "Copy existing data from blue → green"),
        ("3. Dual-write",    "App writes to both blue + green simultaneously"),
        ("4. Verify",        "Run validation queries — compare blue vs green"),
        ("5. Cut-over",      "Switch reads to green (update connection string)"),
        ("6. Monitor",       "Watch error rates, latency — keep blue hot for rollback"),
        ("7. Decommission",  "After 48h stable — stop writes to blue, archive"),
    ]
    for phase, desc in phases:
        print(f"  {phase:<18} {desc}")

    print(f"\n  Rollback: switch connection string back to blue (< 1 min)")
    print(f"  Cost: ~2x storage during migration window")

    # ── Migration Best Practices ───────────────
    print("\n\n[6] MIGRATION BEST PRACTICES")
    print("─" * 55)
    practices = [
        "Never modify an already-applied migration — create a new one",
        "Test migrations on production-sized data clone before applying",
        "Run long backfills in batches with LIMIT + sleep (avoid I/O storm)",
        "Use CREATE INDEX CONCURRENTLY — never blocking CREATE INDEX",
        "Deploy app before schema change when adding columns",
        "Deploy schema change before removing columns (old app still writes)",
        "Always have a sql_down (rollback) ready and tested",
        "Lock schema_migrations table to prevent concurrent runner instances",
        "Set statement_timeout on migration sessions to auto-abort runaway DDL",
    ]
    for i, p in enumerate(practices, 1):
        print(f"  {i}. {p}")


if __name__ == "__main__":
    demonstrate_db_migration()
