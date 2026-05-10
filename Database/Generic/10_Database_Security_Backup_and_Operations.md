# Database Security, Backup, and Operations

---

## Easy (Q1–Q7)

---

**Q1. What is the principle of least privilege as applied to database users?**

Each application, service, or person should have only the minimum database permissions required for their function.

**PostgreSQL example:**

```sql
-- Application read-write user: only DML on specific tables
CREATE ROLE app_user LOGIN PASSWORD 'strong_password';
GRANT CONNECT ON DATABASE myapp TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE orders, customers, products TO app_user;
-- NOT GRANTED: DROP TABLE, CREATE TABLE, TRUNCATE, pg_dump access

-- Read-only user for analytics/reporting
CREATE ROLE analytics_user LOGIN PASSWORD 'strong_password';
GRANT CONNECT ON DATABASE myapp TO analytics_user;
GRANT USAGE ON SCHEMA public TO analytics_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analytics_user;

-- Migration user: needed during deploys only
CREATE ROLE migrations_user LOGIN PASSWORD 'strong_password';
GRANT ALL ON SCHEMA public TO migrations_user;  -- can CREATE/ALTER/DROP tables
-- Revoke after migration completes, or use short-lived credentials
```

**Default privilege pitfall:**
```sql
-- Prevent future tables from being world-accessible
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    REVOKE ALL ON TABLES FROM PUBLIC;

-- Explicitly grant to specific roles
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_user;
```

**Why it matters:** If the application DB user can only do DML, a SQL injection vulnerability cannot drop tables, read sensitive tables outside the app's scope, or exfiltrate data to other schemas.

---

**Q2. What is Row-Level Security (RLS) in PostgreSQL and when is it useful?**

RLS allows you to define policies that restrict which rows a database user can see or modify, enforced at the database level regardless of which query the application sends.

```sql
-- Enable RLS on a table
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Policy: users can only see their own documents
CREATE POLICY user_documents ON documents
    USING (owner_id = current_setting('app.current_user_id')::bigint);

-- Policy: admins see everything
CREATE POLICY admin_access ON documents
    USING (current_setting('app.role') = 'admin');

-- Application sets the context variable at connection start
SET app.current_user_id = '42';
SET app.role = 'user';

-- Now: SELECT * FROM documents → only returns user 42's documents
```

**Use cases:**
- **Multi-tenant SaaS:** each tenant only sees their own data; eliminates an entire class of tenant data leakage bugs
- **Row-level data classification:** rows with `sensitivity = 'restricted'` hidden from non-privileged users
- **Audit + compliance:** PII fields visible only to roles with explicit access

**Trade-off:** RLS policies add a predicate to every query — can defeat index usage if the policy filter isn't indexed. Always index the RLS filter column.

---

**Q3. What is SQL injection and how do you prevent it at the database layer?**

SQL injection occurs when user-supplied input is concatenated into a SQL string, allowing an attacker to modify the query's logic.

**Vulnerable pattern:**
```python
# NEVER do this
query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
# Attacker input: username = "' OR '1'='1" -- 
# Result: bypasses authentication
```

**Prevention at the database layer:**

**1. Parameterized queries (prepared statements) — primary defense:**
```python
# Python psycopg2
cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s",
               (username, hashed_password))

# Never passes raw values into SQL string — DB parses query and data separately
```

**2. Stored procedures (limit direct table access):**
```sql
CREATE FUNCTION authenticate_user(p_username TEXT, p_password TEXT)
RETURNS BOOLEAN LANGUAGE plpgsql AS $$
BEGIN
    -- Application only calls the function, never writes raw SQL
    RETURN EXISTS (
        SELECT 1 FROM users
        WHERE username = p_username AND password_hash = crypt(p_password, password_hash)
    );
END;
$$;
```

**3. Input validation:** Validate types, lengths, and formats before sending to DB — defense in depth.

**4. Minimal permissions:** Even if injection succeeds, a read-only user cannot DROP TABLE or access other schemas.

---

**Q4. What is a database backup strategy and what are the three types of backups?**

**1. Full backup:** Complete copy of the entire database.
```bash
# PostgreSQL: logical dump
pg_dump -Fc mydb > mydb_$(date +%Y%m%d).dump

# Physical backup (faster for large DBs)
pg_basebackup -D /backup/base -Ft -z -P -Xs stream
```

**2. Incremental backup:** Only changes since the last full backup.
- PostgreSQL: achieved via WAL archiving (WAL files = incremental changes)
- Tools: `pgBackRest`, `Barman`, `WAL-G`

**3. Differential backup:** All changes since last full backup (differs from incremental: not since last backup of any type).
- Less common in PostgreSQL; more common in SQL Server

**Point-in-Time Recovery (PITR):**
```
Full base backup + WAL archives → restore to any moment in time

Example: Database corruption at 2:47 PM
→ Restore base backup from 2 AM
→ Replay WAL from 2 AM to 2:46 PM
→ Stop before corruption event
```

**3-2-1 backup rule:**
- 3 copies of data
- 2 different storage media types
- 1 copy offsite (different region/cloud)

---

**Q5. What is RPO vs RTO and how do they drive backup strategy?**

**RPO (Recovery Point Objective):** Maximum acceptable data loss measured in time.
- RPO = 1 hour → can lose up to 1 hour of data
- RPO = 0 → zero data loss (requires synchronous replication)

**RTO (Recovery Time Objective):** Maximum acceptable time to recover and resume service.
- RTO = 4 hours → system must be back online within 4 hours of failure
- RTO = 1 minute → requires hot standby (warm failover)

**Strategies by RPO/RTO:**

| RPO | RTO | Strategy |
|-----|-----|----------|
| 0 (zero loss) | < 1 min | Synchronous streaming replication + Patroni auto-failover |
| < 5 minutes | < 5 min | Asynchronous replica with auto-failover (small lag) |
| < 1 hour | < 1 hour | Hourly WAL archiving + warm standby |
| 24 hours | < 8 hours | Daily full backup + WAL archiving, restore from backup |
| 24 hours | < 24 hours | Daily full backup only, manual restore |

**Financial systems** typically require RPO=0, RTO < 1 minute.
**Internal tools** may accept RPO=24h, RTO=8h.

---

**Q6. What is database encryption at rest and in transit?**

**Encryption at rest:** Data files on disk are encrypted; physical theft of drives doesn't expose data.

```bash
# PostgreSQL: rely on OS/cloud-level encryption
# AWS: RDS Automatic Encryption (AES-256 at the volume level, transparent)
# Self-managed: use LUKS (Linux) or AWS EBS encryption

# PostgreSQL also supports pgcrypto for column-level encryption
CREATE EXTENSION pgcrypto;

-- Encrypt sensitive column values
INSERT INTO users (name, ssn_encrypted)
VALUES ('Alice', pgp_sym_encrypt('123-45-6789', 'encryption_key'));

-- Decrypt
SELECT pgp_sym_decrypt(ssn_encrypted, 'encryption_key') AS ssn FROM users;
```

**Encryption in transit:** TLS for all connections between applications and database.

```ini
# postgresql.conf
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file  = 'server.key'
ssl_ca_file   = 'root.crt'

# pg_hba.conf: require SSL for app connections
hostssl  mydb  app_user  0.0.0.0/0  scram-sha-256
```

**Application connection string:**
```
postgresql://user:pass@host:5432/db?sslmode=verify-full&sslrootcert=root.crt
```

`sslmode=verify-full` — verifies certificate chain AND hostname (prevents MITM).

---

**Q7. What is database auditing and how do you implement it?**

Database auditing records who did what to which data and when — essential for compliance (SOC 2, HIPAA, GDPR, PCI-DSS).

**Approach 1: Application-level audit log table**
```sql
CREATE TABLE audit_log (
    id          BIGSERIAL PRIMARY KEY,
    table_name  TEXT NOT NULL,
    record_id   BIGINT NOT NULL,
    operation   TEXT NOT NULL,  -- 'INSERT', 'UPDATE', 'DELETE'
    old_data    JSONB,
    new_data    JSONB,
    changed_by  TEXT NOT NULL,  -- application user, not DB user
    changed_at  TIMESTAMPTZ DEFAULT NOW(),
    ip_address  INET
);

-- Trigger function (PostgreSQL)
CREATE OR REPLACE FUNCTION audit_trigger() RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (table_name, record_id, operation, old_data, new_data, changed_by)
    VALUES (
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        TG_OP,
        CASE WHEN TG_OP != 'INSERT' THEN row_to_json(OLD) END,
        CASE WHEN TG_OP != 'DELETE' THEN row_to_json(NEW) END,
        current_setting('app.current_user', true)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_users
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit_trigger();
```

**Approach 2: pgaudit extension** — logs all SQL to PostgreSQL log files, configurable by object type, role, and statement type. Good for compliance requirements demanding immutable audit trail outside the DB.

---

## Medium (Q8–Q15)

---

**Q8. How do you implement zero-downtime database migrations?**

Migrations that take locks or change column definitions can cause outages. The safe pattern:

**Adding a column (safe):**
```sql
-- 1. Add nullable column (no table rewrite, instant)
ALTER TABLE users ADD COLUMN phone_number TEXT;

-- 2. Backfill in batches (avoid long lock)
UPDATE users SET phone_number = '' WHERE id BETWEEN 1 AND 100000;
UPDATE users SET phone_number = '' WHERE id BETWEEN 100001 AND 200000;
-- ... repeat

-- 3. Add NOT NULL constraint without validating existing rows
ALTER TABLE users ADD CONSTRAINT users_phone_not_null
    CHECK (phone_number IS NOT NULL) NOT VALID;

-- 4. Validate in background (doesn't lock writes)
ALTER TABLE users VALIDATE CONSTRAINT users_phone_not_null;

-- 5. (PostgreSQL 12+) Convert to actual NOT NULL
ALTER TABLE users ALTER COLUMN phone_number SET NOT NULL;
```

**Removing a column (safe sequence):**
```sql
-- 1. Stop reading/writing the column in application (deploy)
-- 2. Drop NOT NULL constraint first (if exists)
ALTER TABLE users ALTER COLUMN old_column DROP NOT NULL;
-- 3. Mark as unused with default (optional, for safety period)
-- 4. In next release: DROP COLUMN
ALTER TABLE users DROP COLUMN old_column;
```

**Adding an index (non-blocking):**
```sql
-- Standard: blocks writes while building
CREATE INDEX idx_users_email ON users(email);  -- DANGEROUS on large tables

-- Non-blocking: takes only brief ShareUpdateExclusiveLock
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
-- Note: cannot run inside a transaction; ~2-3× slower but never blocks
```

**Renaming a column (requires dual-read period):**
```sql
-- Cannot rename column atomically in use
-- Phase 1: Add new column, dual-write old + new, read from old
ALTER TABLE orders ADD COLUMN customer_id BIGINT;
-- Phase 2: Backfill new column from old
-- Phase 3: Read from new column, still write both
-- Phase 4: Stop writing old column
-- Phase 5: DROP old column
```

---

**Q9. How do you implement GDPR-compliant data deletion (right to be forgotten) in a database with complex relationships?**

GDPR Article 17 requires that personal data be erasable on request. Hard deletes in a relational DB with foreign keys and audit logs are complex.

**Strategy 1: Hard delete with cascade**
```sql
-- FK constraint with cascade (simple but loses history)
ALTER TABLE orders ADD CONSTRAINT fk_orders_customer
    FOREIGN KEY (customer_id) REFERENCES customers(id)
    ON DELETE SET NULL;  -- or ON DELETE CASCADE

-- Deletion function
CREATE OR REPLACE FUNCTION gdpr_delete_user(p_user_id BIGINT) RETURNS VOID AS $$
BEGIN
    -- Anonymize (not delete) orders for financial audit purposes
    UPDATE orders
    SET customer_id = NULL,
        customer_email = 'deleted@anonymous.com',
        customer_name = 'Deleted User'
    WHERE customer_id = p_user_id;
    
    -- Delete personal data
    DELETE FROM user_sessions WHERE user_id = p_user_id;
    DELETE FROM user_preferences WHERE user_id = p_user_id;
    DELETE FROM personal_data WHERE user_id = p_user_id;
    
    -- Anonymize the user record (don't delete — need for referential integrity)
    UPDATE users
    SET email = 'deleted_' || p_user_id || '@anonymous.invalid',
        name = 'Deleted User',
        phone = NULL,
        address = NULL,
        deleted_at = NOW(),
        is_gdpr_deleted = TRUE
    WHERE id = p_user_id;
END;
$$ LANGUAGE plpgsql;
```

**Strategy 2: Soft delete with anonymization (recommended)**
```sql
-- Partial unique index: email unique for non-deleted users only
CREATE UNIQUE INDEX idx_users_email_active
    ON users(email) WHERE (is_gdpr_deleted = FALSE);

-- After GDPR deletion: original email freed for re-registration
-- But business data (orders, payments) retained with anonymous reference
```

**Challenges:**
- **Backups:** Personal data deleted from live DB still exists in backups. Policy: backups older than 30 days are destroyed; or use encryption with key rotation (delete the key = effectively delete the data).
- **Audit logs:** Audit log entries contain personal data. Store personal data in audit logs as anonymized IDs, or age out audit logs after retention period.
- **Search indexes:** Elasticsearch, Redshift, ClickHouse must have data deleted or re-indexed after GDPR deletion.
- **Caches:** Redis TTL should be short; explicit cache invalidation on GDPR deletion.

---

**Q10. How do you handle database credential rotation without downtime?**

**Challenge:** Rotating DB passwords requires all application instances to use the new password simultaneously — if any instance has the old password cached, it gets connection errors.

**Method 1: Multiple valid credentials simultaneously**
```sql
-- PostgreSQL: create new user with same privileges
CREATE ROLE app_user_v2 LOGIN PASSWORD 'new_strong_password';
GRANT ALL ON ALL TABLES IN SCHEMA public TO app_user_v2;
-- Deploy application to use app_user_v2 (rolling deploy)
-- After all instances deployed: drop old user
DROP ROLE app_user_v1;
```

**Method 2: AWS Secrets Manager with automatic rotation**
```
1. Store DB credentials in AWS Secrets Manager
2. Lambda rotation function runs every 30 days:
   a. Creates new password in DB
   b. Tests new password
   c. Updates Secrets Manager with new password
   d. Deactivates old password
3. Applications fetch credentials from Secrets Manager at startup
   (or via SDK that auto-refreshes credentials)
```

**Method 3: PgBouncer credential delegation**
```ini
# PgBouncer auth_user: uses a single DB user for auth lookups
# When app credentials rotate, only PgBouncer needs to reload
# Applications authenticate to PgBouncer with their own credentials

[pgbouncer]
auth_type = scram-sha-256
auth_user = pgbouncer_auth
auth_query = SELECT p_user, p_password FROM pgbouncer.get_auth($1)

# Rotate app credentials in the pgbouncer.users table
# PgBouncer reload: pgbouncer> RELOAD; (zero downtime, keeps connections alive)
```

**Method 4: Vault dynamic secrets**
```
HashiCorp Vault generates short-lived database credentials (TTL: 1 hour)
Application fetches credential from Vault; Vault creates a PostgreSQL role
Credential expires → Vault creates new credential on next request
No rotation needed — credentials are ephemeral
```

---

**Q11. What is database connection security and what should you configure for production?**

**PostgreSQL pg_hba.conf (host-based authentication):**
```
# Type   DB      User          Address          Method
local    all     postgres                       peer         # local UNIX: system user = DB user
host     mydb    app_user      10.0.0.0/8       scram-sha-256 # internal network: password
hostssl  mydb    app_user      0.0.0.0/0        scram-sha-256 # internet: TLS required
host     mydb    analytics     10.0.1.0/24      scram-sha-256 # analytics subnet only
host     all     all           0.0.0.0/0        reject       # deny everything else
```

**Key principles:**
1. Never expose PostgreSQL port (5432) to the internet — use a VPN, bastion host, or SSH tunnel
2. Use `scram-sha-256` authentication (not `md5` which is weak, not `trust` which is no auth)
3. Restrict source IPs per user — analytics user can only connect from analytics subnet
4. Require SSL for all non-local connections (`hostssl` instead of `host`)
5. Rotate superuser password and disable remote superuser login

**Network security:**
```bash
# PostgreSQL should only listen on private interface
listen_addresses = '10.0.0.5'  # not '0.0.0.0' (all interfaces)

# Firewall: only allow DB port from app servers
iptables -A INPUT -p tcp --dport 5432 -s 10.0.0.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 5432 -j DROP
```

**Secrets management:**
- Never hardcode DB passwords in application code or environment variables in containers
- Use AWS Secrets Manager, HashiCorp Vault, or Kubernetes Secrets (encrypted at rest)
- Rotate credentials on a schedule or on staff turnover

---

**Q12. How do you design a database disaster recovery plan?**

**DR plan components:**

**1. Define objectives**
```
RPO: maximum data loss tolerable (e.g., 5 minutes)
RTO: maximum downtime tolerable (e.g., 30 minutes)
```

**2. Backup infrastructure**
```
Primary:   AWS us-east-1 PostgreSQL (Patroni cluster)
DR:        AWS us-west-2 PostgreSQL (warm standby)
Backups:   S3 in us-east-1 + cross-region replication to us-west-2
WAL:       Archived to S3 every 1 minute (WAL-G or pgBackRest)
```

**3. Failover procedures (runbook)**
```
Scenario A: Primary server fails, standby healthy
  1. Patroni automatically promotes standby (< 30s)
  2. DNS failover via Route 53 health check (TTL 30s)
  3. PgBouncer reconnects to new primary
  4. Alert team; assess cause of failure
  5. Rebuild failed primary as new standby

Scenario B: Entire region fails
  1. Declare DR event
  2. Restore base backup from S3 (us-west-2 copy)
  3. Apply WAL archives up to last available WAL file
  4. Start PostgreSQL in recovery mode
  5. Update application config to point to DR region
  6. Estimated RTO: ~30 minutes + WAL replay time

Scenario C: Data corruption (soft disaster)
  1. Identify timestamp of corruption from logs
  2. PITR: restore to 1 minute before corruption
  3. Export affected tables from PITR instance
  4. Import corrected data into production
```

**4. Regular DR testing**
```bash
# Monthly: test backup restoration in a separate environment
# Quarterly: full failover drill to DR region
# Annually: tabletop disaster exercise with all stakeholders

# Automated daily backup validation
pgbackrest --stanza=main restore --delta --target-time="$(date -d '1 hour ago' +%Y-%m-%dT%H:%M:%S)"
# Check row counts match expected values
```

**5. Monitoring for DR triggers**
```yaml
Alert: PrimaryUnreachable (60s) → page on-call
Alert: ReplicationLag > 5min → page on-call
Alert: BackupAgeHours > 25 → page on-call (missed daily backup)
Alert: WALArchiveStalled > 10min → page on-call
```

---

**Q13. How do you handle multi-tenant data isolation in a database?**

Three architectural options with different isolation/efficiency trade-offs:

**Option 1: Shared database, shared tables (RLS isolation)**
```sql
-- All tenants in same table; RLS enforces isolation
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON orders
    USING (tenant_id = current_setting('app.tenant_id')::bigint);

-- Application sets tenant context per connection
SET app.tenant_id = '42';

-- Pros: Most resource-efficient, easy to deploy
-- Cons: Tenant isolation breach if RLS policy has a bug; one tenant's slow query affects others
```

**Option 2: Shared database, separate schemas**
```sql
-- One schema per tenant
CREATE SCHEMA tenant_42;
CREATE TABLE tenant_42.orders (...);
CREATE TABLE tenant_42.products (...);

-- Pros: Strong isolation, easy to backup/restore single tenant
-- Cons: Schema management complexity at scale (10K+ tenants = 10K schemas); 
--       PostgreSQL catalog bloat with many schemas
-- Best for: < 1000 tenants, compliance requirements for data separation
```

**Option 3: Separate databases per tenant**
```
# One entire database per tenant
PostgreSQL: tenant_42_db, tenant_99_db, ...

# Pros: Complete isolation, independent backup/restore, different versions possible
# Cons: High operational overhead, connection pooling complexity, resource waste for small tenants
# Best for: Large enterprise tenants with strict compliance (SOC 2 Type II, HIPAA)
```

**Hybrid approach (common in practice):**
```
Free/small tenants → shared table with RLS (Option 1)
Mid-size tenants → separate schema (Option 2)
Enterprise tenants → separate database (Option 3)

Automated provisioning moves tenants between tiers as they grow
```

---

**Q14. What are common database security vulnerabilities and how do you mitigate them?**

**1. Excessive privileges**
- Mitigation: Least privilege (separate read-only, read-write, migration roles)

**2. Unencrypted connections**
- Mitigation: Enforce SSL in pg_hba.conf; use `sslmode=verify-full` in clients

**3. Weak authentication**
- Mitigation: `scram-sha-256`, strong passwords, rotate credentials, MFA for admin access

**4. Exposed database port**
- Mitigation: DB in private subnet, VPN/bastion for admin access only, firewall rules

**5. SQL injection**
- Mitigation: Parameterized queries always; stored procedures; WAF in front of API

**6. Sensitive data in logs**
- Mitigation: `log_min_duration_statement` logs queries — sanitize before shipping to log aggregator; never log full query text in production if it contains PII
```ini
# PostgreSQL: don't log query parameters (avoids PII/secret leakage)
log_parameter_max_length = 0  # don't log bind parameters
log_parameter_max_length_on_error = 0
```

**7. Unencrypted backups**
- Mitigation: Encrypt backups with pgBackRest (`--cipher-type=aes-256-cbc`) or AWS S3 SSE

**8. Superuser misuse**
- Mitigation: No application connects as `postgres` superuser; only used for admin tasks via bastion

**9. Stale/orphaned accounts**
- Mitigation: Audit `pg_roles` quarterly; remove unused roles; use ephemeral Vault credentials

**10. pg_hba.conf misconfiguration**
- Common mistake: `host all all 0.0.0.0/0 trust` (no authentication for any host)
- Mitigation: Review pg_hba.conf in code review; use `pg_hba.conf.d/` for modular config

---

**Q15. How do you implement database change management (schema migrations) in a CI/CD pipeline?**

**Tools:** Flyway, Liquibase, Alembic (Python), golang-migrate, Sqitch.

**Migration file structure (Flyway example):**
```
migrations/
  V1__create_users_table.sql
  V2__add_email_index.sql
  V3__create_orders_table.sql
  V4__add_customer_id_to_orders.sql
```

**CI/CD pipeline integration:**
```yaml
# .github/workflows/deploy.yml
jobs:
  database-migrate:
    runs-on: ubuntu-latest
    steps:
      - name: Run migrations
        run: |
          flyway -url=jdbc:postgresql://$DB_HOST/mydb \
                 -user=$DB_USER \
                 -password=$DB_PASS \
                 -locations=filesystem:./migrations \
                 migrate
      - name: Validate migration
        run: flyway validate  # checks applied migrations match files (detects tampering)
```

**Migration best practices:**
1. **Never modify applied migrations** — always add new migration files
2. **Test rollback** — every migration should have a rollback script
3. **Use `CONCURRENTLY`** for index creation — won't block deployments
4. **Separate migration from deploy** — run migrations before deploying new application code (not after)
5. **Idempotent migrations** — safe to run twice (`CREATE TABLE IF NOT EXISTS`)

**Blue-green deployment compatibility:**
```sql
-- Phase 1 migration: backward-compatible (both old and new app versions work)
ALTER TABLE orders ADD COLUMN new_status TEXT;  -- nullable, ignored by old app

-- Deploy new application version

-- Phase 2 migration: clean up old column once old app is gone
ALTER TABLE orders DROP COLUMN old_status;
```

---

## Hard (Q16–Q20)

---

**Q16. Design a database backup and recovery system with RPO=5min and RTO=15min for a financial application.**

**Requirements:**
- PostgreSQL primary + 2 replicas
- 500GB database, 100K transactions/day
- Maximum 5 minutes of data loss
- Maximum 15 minutes to recover and resume service

**Architecture:**

```
┌──────────────────────────────────────────────────────────────┐
│  Primary (us-east-1a)                                        │
│  PostgreSQL + Patroni agent                                  │
│  WAL archiving → S3 (every 60 seconds via WAL-G)            │
│  Streaming replication → Standby 1 + Standby 2              │
├──────────────────────────────────────────────────────────────┤
│  Standby 1 (us-east-1b) — synchronous                       │
│  synchronous_standby_names = 'standby1'                      │
│  Patroni leader election via etcd cluster                    │
│  RTO contribution: < 30 seconds (auto-promote)              │
├──────────────────────────────────────────────────────────────┤
│  Standby 2 (us-east-1c) — asynchronous                      │
│  For read queries + second failover target                   │
├──────────────────────────────────────────────────────────────┤
│  WAL-G + S3 (us-east-1)                                      │
│  Full backup: daily (pg_basebackup via WAL-G)               │
│  WAL archive: every 60 seconds → RPO = 60 seconds (< 5min)  │
│  Cross-region copy: S3 replication to us-west-2             │
└──────────────────────────────────────────────────────────────┘
```

**PostgreSQL configuration for synchronous replication:**
```ini
# postgresql.conf (primary)
synchronous_standby_names = 'FIRST 1 (standby1, standby2)'
# FIRST 1: transaction committed only when at least 1 of the named standbys confirms
# RPO = 0 for synchronous replica; WAL archive covers the async replica
wal_level = replica
archive_mode = on
archive_command = 'wal-g wal-push %p'
archive_timeout = 60  # force WAL switch every 60 seconds
```

**WAL-G backup configuration:**
```bash
# WAL-G environment
WALG_S3_PREFIX=s3://company-db-backups/prod
AWS_REGION=us-east-1

# Daily full backup (cron at 2 AM)
wal-g backup-push /var/lib/postgresql/data

# WAL archiving (continuous, from archive_command)
wal-g wal-push $WAL_FILE

# Retention policy
wal-g delete retain FULL 7  # keep 7 full backups
```

**Failover runbook (RTO breakdown):**
```
T+0:00   Primary unreachable — Patroni detects via etcd heartbeat
T+0:10   Patroni declares primary failed (10-second timeout)
T+0:20   Patroni promotes Standby1 to primary (10 seconds)
T+0:30   Patroni updates etcd with new leader
T+0:45   PgBouncer reconnects to new primary via DNS/service endpoint
T+1:00   Applications recover from connection errors (PgBouncer retries)
T+1:00   PAGE on-call engineer (auto-page on Patroni failover event)
→ Total RTO: ~60 seconds for automatic failover

Regional failure RTO (worst case):
T+0:00   Region failure detected
T+5:00   On-call declares DR event
T+8:00   Restore base backup in us-west-2 from S3 cross-region copy
T+12:00  WAL replay to latest available WAL file
T+14:00  Database online, update DNS, redirect traffic
T+15:00  Applications reconnecting — within 15-minute RTO ✓
```

**Testing (mandatory for financial systems):**
```bash
# Monthly: automated backup restoration test
wal-g backup-fetch /tmp/test_restore LATEST
# Start PostgreSQL on test restore, verify row counts
# Alert if test fails

# Quarterly: full DR drill
# Simulate regional failure; time the recovery; measure actual RTO
```

---

**Q17. You discover your database has been compromised — walk through the incident response procedure.**

**Hour 0: Detection and containment**

```bash
# 1. Verify breach evidence
# Check pg_stat_activity for suspicious connections
SELECT client_addr, usename, application_name, query, state
FROM pg_stat_activity
ORDER BY query_start;

# Check pg_log for suspicious auth or queries
grep -i "drop\|truncate\|pg_dump\|COPY TO" /var/log/postgresql/postgresql.log

# Check for new roles (potential backdoor users)
SELECT rolname, rolsuper, rolcreatedb, rolcreaterole, rolcanlogin
FROM pg_roles
WHERE rolname NOT IN ('postgres', 'app_user', 'analytics_user');  -- compare to known list

# 2. IMMEDIATE: Block external connections
-- Update pg_hba.conf to block all non-admin connections
-- Reload: SELECT pg_reload_conf();

# 3. Take forensic snapshot (before any changes)
pg_dump -Fc mydb > forensic_snapshot_$(date +%Y%m%d_%H%M%S).dump
cp /var/log/postgresql/postgresql.log forensic_postgresql.log
```

**Hour 1: Assessment**
```sql
-- 1. Check for data exfiltration evidence (large COPY TO operations)
grep "COPY" /var/log/postgresql/postgresql.log | grep "TO STDOUT\|TO '/"

-- 2. Check for unauthorized schema changes
SELECT schemaname, tablename, tableowner
FROM pg_tables
WHERE tableowner NOT IN ('postgres', 'migration_user');

-- 3. Check for new functions (potential backdoors)
SELECT proname, prosrc FROM pg_proc
WHERE proowner NOT IN (SELECT oid FROM pg_roles WHERE rolname IN ('postgres'))
  AND proname NOT IN (SELECT proname FROM known_functions_baseline);

-- 4. Identify timeline: when did the breach occur?
-- Check WAL files / pg_log timestamps for anomalous activity

-- 5. Assess blast radius: what data was accessed?
-- Review audit_log table (if enabled) for the attacker's user/IP
```

**Hour 2-4: Remediation**
```bash
# 1. Rotate all credentials immediately
# New passwords for all DB users
# Revoke and re-issue API keys
# Rotate application secrets (Vault/Secrets Manager)

# 2. Remove unauthorized access
-- DROP ROLE suspicious_user;
-- DROP FUNCTION if_backdoor_found;

# 3. Patch the entry point
# SQL injection? → Fix parameterized queries, deploy patch
# Leaked credentials? → Enforce secrets manager, no hardcoded passwords
# Exposed port? → Update firewall rules

# 4. Restore from clean backup if data was corrupted/deleted
wal-g backup-fetch /var/lib/postgresql/data LATEST
# PITR to before the breach if corruption was introduced

# 5. Enable enhanced logging temporarily
log_statement = 'all'  # log every SQL statement (not permanent, high volume)
log_connections = on
log_disconnections = on
```

**Hour 4+: Post-incident**
1. Formal incident report: timeline, root cause, blast radius
2. Customer/regulator notification (GDPR: 72-hour breach notification requirement)
3. Forensic analysis: what data was accessed/exfiltrated
4. Control improvements: add pgaudit, tighten RLS, improve IDS/alerting

---

**Q18. How do you migrate a 2TB PostgreSQL database to a new server with minimal downtime?**

**Goal:** Migrate from old server (on-prem) to new server (AWS RDS), target < 5 minutes downtime.

**Phase 1: Set up logical replication (days before cutover)**
```sql
-- On old server: enable logical replication
wal_level = logical
max_replication_slots = 5
max_wal_senders = 5

-- Create publication
CREATE PUBLICATION migration_pub FOR ALL TABLES;

-- On new server: create subscription
CREATE SUBSCRIPTION migration_sub
    CONNECTION 'host=old-server dbname=mydb user=replication password=xxx'
    PUBLICATION migration_pub;

-- Initial sync begins automatically (copies 2TB base data)
-- This takes several hours for 2TB
```

**Monitor initial sync progress:**
```sql
-- On new server
SELECT subname, received_lsn, latest_end_lsn,
       pg_size_pretty(pg_wal_lsn_diff(latest_end_lsn, received_lsn)) AS lag
FROM pg_stat_subscription;

-- Wait until lag is consistently < 100MB (near-realtime replication)
```

**Phase 2: Pre-cutover checklist**
```sql
-- 1. Verify row counts match
SELECT count(*) FROM important_table;  -- run on both servers

-- 2. Verify sequences are ahead on new server (prevent PK conflicts)
-- Logical replication does NOT replicate sequences
SELECT sequence_name, last_value FROM information_schema.sequences;
-- Manually advance sequences on new server: 
-- SELECT setval('orders_id_seq', (SELECT MAX(id) FROM orders) + 10000);

-- 3. Create all non-PK indexes (logical replication doesn't require them during sync)
-- Run CREATE INDEX CONCURRENTLY on new server during sync phase

-- 4. Test application connectivity to new server
```

**Phase 3: Cutover (< 5 minutes)**
```bash
# T-5min: Alert users of brief maintenance
# T-0: 
#   1. Put old application in maintenance mode (stop writes)
#   2. Wait for replication lag to reach 0
SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), received_lsn) AS lag_bytes
FROM pg_stat_replication;  -- wait until = 0

#   3. Verify row counts match
#   4. Update application connection string to new server
#   5. Re-enable application
# T+3min: Application running on new server

# Post-cutover:
#   Drop subscription on new server (no longer needed)
DROP SUBSCRIPTION migration_sub;
#   Monitor for 24 hours before decommissioning old server
```

---

**Q19. How do you implement database observability for compliance and data governance?**

**Compliance requirements (HIPAA, PCI-DSS, SOC 2 Type II):**
- Who accessed what data, when, from where
- All privileged operations logged
- Unauthorized access attempts detected and alerted
- Data retention and deletion policies enforced
- Encryption verified

**Implementation: pgaudit for comprehensive logging**
```ini
# postgresql.conf
shared_preload_libraries = 'pgaudit'

# Audit all DDL and DML on sensitive tables
pgaudit.log = 'ddl, write'                    # log DDL + INSERT/UPDATE/DELETE
pgaudit.log_catalog = on                      # log pg_catalog queries
pgaudit.log_client = on                       # include client info
pgaudit.log_level = log
pgaudit.log_parameter = on                    # log query parameters (careful with PII)
pgaudit.log_relation = on                     # log table/view name per statement
```

**Object-level audit (table-specific):**
```sql
-- Audit all access to PII tables
CREATE EXTENSION pgaudit;

-- Per-table audit
ALTER TABLE patients SET (pgaudit.log = 'select, write');
ALTER TABLE payment_methods SET (pgaudit.log = 'all');
```

**Log shipping to SIEM:**
```yaml
# Vector.dev: ship PostgreSQL logs to Elasticsearch
[sources.pg_logs]
type = "file"
includes = ["/var/log/postgresql/postgresql-*.log"]

[transforms.parse_pg]
type = "remap"
source = '''
.event_type = "database_audit"
.db_host = get_hostname!()
'''

[sinks.elasticsearch]
type = "elasticsearch"
endpoint = "https://elk.company.internal:9200"
index = "database-audit-%Y.%m.%d"
```

**Automated compliance checks:**
```sql
-- Daily check: no new superusers
SELECT rolname FROM pg_roles
WHERE rolsuper = true
  AND rolname NOT IN ('postgres');  -- alert if any results

-- Daily check: all tables with PII have RLS enabled
SELECT tablename FROM pg_tables
WHERE tablename IN ('users', 'patients', 'payment_methods')
  AND NOT EXISTS (
    SELECT 1 FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relname = tablename AND c.relrowsecurity = true
  );

-- Daily check: SSL required for all users
SELECT usename FROM pg_shadow
WHERE passwd IS NOT NULL;  -- ensure no trust auth users

-- Weekly: unused roles (potential orphaned access)
SELECT r.rolname
FROM pg_roles r
LEFT JOIN pg_stat_activity a ON a.usename = r.rolname
WHERE a.usename IS NULL
  AND r.rolcanlogin = true
  AND r.rolname NOT IN ('postgres', 'replication');
```

---

**Q20. Walk through the complete operational runbook for a PostgreSQL primary failure — from detection to full recovery.**

**Pre-requisites (already in place):**
- Patroni cluster: primary + 2 standbys, etcd for consensus
- PgBouncer in front of PostgreSQL
- HAProxy for load balancing
- Prometheus + Alertmanager + PagerDuty

---

**T+0:00 — Primary fails (hardware fault, OOM kill, etc.)**

**T+0:10 — Patroni detects failure**
```
Patroni agents on each standby fail to reach primary via etcd
etcd leader election: standby1 wins (most recent LSN)
Patroni promotes standby1: pg_ctl promote
Standby1 becomes new primary
```

**T+0:20 — PagerDuty page fires**
```
Alert: PostgreSQL primary unreachable
Alert: Patroni failover detected
→ On-call engineer acknowledges
```

**T+0:30 — HAProxy health check**
```
HAProxy detects primary changed via Patroni REST API
/primary endpoint: returns 200 on new primary, 503 on old
HAProxy routes new connections to standby1 (new primary)
```

**T+0:45 — PgBouncer reconnects**
```
PgBouncer loses connections to old primary (TCP error)
PgBouncer reconnects to new primary (HAProxy routes correctly)
Client connections: brief error burst (5-10 seconds), then auto-reconnect
```

**T+1:00 — Services recovered (automatic)**
- New primary: standby1 (us-east-1b)
- Remaining replica: standby2 (us-east-1c) replicates from new primary
- Applications: reconnected, normal operation resumed

---

**On-call engineer actions (T+1:00 onwards):**

**Step 1: Verify current cluster state**
```bash
# Check Patroni cluster status
patronictl -c /etc/patroni.yml list

# Output:
# + Cluster: myapp (123456789) -------+----+-----------+
# | Member    | Host          | Role    | State   | TL |
# +-----------+---------------+---------+---------+----+
# | primary   | 10.0.0.1:5432 | Replica | stopped | 1  |  ← failed
# | standby1  | 10.0.0.2:5432 | Leader  | running | 2  |  ← promoted
# | standby2  | 10.0.0.3:5432 | Replica | running | 2  |  ← replicating

# Verify applications are healthy
psql -h haproxy-endpoint -U app_user -c "SELECT 1";

# Check replication lag on remaining replica
psql -h standby1 -c "SELECT * FROM pg_stat_replication;"
```

**Step 2: Investigate failed primary**
```bash
# Attempt to SSH to failed server
ssh 10.0.0.1

# Check system logs
journalctl -u postgresql --since "1 hour ago"
dmesg | tail -50  # kernel OOM? hardware error?

# Check PostgreSQL logs
tail -200 /var/log/postgresql/postgresql.log

# Common causes:
# OOM kill: "Out of memory: Kill process"
# Disk full: "No space left on device"  
# Hardware: kernel panic / drive failure
# Network: "FATAL: network error" in logs
```

**Step 3: Rebuild failed node as new standby**
```bash
# If server is recoverable:

# 1. Fix the root cause (expand disk, add memory, etc.)

# 2. Clear old data directory (it's now stale)
sudo systemctl stop postgresql
sudo rm -rf /var/lib/postgresql/data

# 3. Rejoin cluster as replica via Patroni
sudo patronictl -c /etc/patroni.yml reinit myapp primary

# Patroni will:
# a. Run pg_basebackup from new primary
# b. Start in standby mode
# c. Stream WAL from new primary

# 4. Verify node rejoined
patronictl -c /etc/patroni.yml list
```

**Step 4: Restore cluster balance (optional)**
```bash
# If original primary server was preferred leader, failback:
# 1. Wait for rebuilt node to fully catch up (lag = 0)
# 2. Switchover (controlled, zero data loss)
patronictl -c /etc/patroni.yml switchover myapp --master standby1 --candidate primary

# Cluster returns to original topology
```

**Step 5: Post-incident actions**
```
1. Update incident ticket with timeline and root cause
2. Check for any missed transactions (compare row counts if in doubt)
3. Review backup freshness: did backup run during outage?
4. Schedule RCA review meeting
5. Update runbook if any step was unclear
6. Add monitoring for root cause (OOM → add memory alert; disk → add disk alert)
```

---

**Quick Reference**

| Backup tool | Type | Features |
|-------------|------|---------|
| `pg_dump` | Logical | Single DB, schema/data selective, slow for large DBs |
| `pg_basebackup` | Physical | Full cluster copy, fast, used for replication setup |
| `WAL-G` | Physical + WAL | Continuous WAL archiving, PITR, S3/GCS/Azure, compression |
| `pgBackRest` | Physical + WAL | Parallel backup, delta backup, encryption, retention policies |
| `Barman` | Physical + WAL | Enterprise-focused, SSH-based, catalog management |

| Security control | Purpose |
|-----------------|---------|
| Row-Level Security | Multi-tenant isolation, PII restriction |
| pg_hba.conf | Network-level access control per user/IP |
| `scram-sha-256` | Strong password authentication |
| `pgaudit` | Compliance-grade SQL audit logging |
| Parameterized queries | SQL injection prevention |
| Column encryption (pgcrypto) | Encrypt PII at rest at column level |
| SSL (`hostssl`) | Encryption in transit |
| Least privilege roles | Limit blast radius of compromised credentials |

| Compliance requirement | Implementation |
|-----------------------|---------------|
| GDPR right to erasure | Anonymize user records, partial unique indexes |
| PCI-DSS: audit all access | pgaudit + SIEM integration |
| SOC 2: change management | Flyway/Liquibase + CI/CD pipeline, reviewed migrations |
| HIPAA: access logs | pgaudit on PHI tables, log retention >= 6 years |
| Encryption at rest | RDS encryption or LUKS + pgcrypto for columns |
| Credential rotation | Vault dynamic secrets or Secrets Manager auto-rotation |
