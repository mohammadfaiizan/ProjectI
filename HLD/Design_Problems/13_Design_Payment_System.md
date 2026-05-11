# System Design: Payment System

---

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a payment processing system (like Stripe, PayPal, or a bank's internal payment engine) that handles 10 million transactions per day with strict ACID guarantees, idempotency, fraud detection, and reconciliation against external PSP (Payment Service Provider) records.

### Clarifying Questions

**Scale:**
- How many transactions per day? *10 million (~116 TPS average, 500 TPS peak)*
- What is the average transaction value? *~$50 USD*
- How many users? *50 million registered users*
- What currencies? *Start with USD, EUR, GBP — 30+ currencies eventually*

**Features:**
- One-time payments only or recurring too? *Both*
- Split payments (e.g., marketplace)? *Yes — split to seller + platform fee*
- Refunds: full and partial? *Yes*
- What payment methods? *Cards, bank transfers (ACH/SEPA), digital wallets*

**Reliability:**
- Can payments be lost? *Never — durability is paramount*
- Acceptable duplicate charge rate? *Zero — idempotency required*
- RPO/RTO? *RPO=0 (no data loss), RTO=30 minutes*

**Compliance:**
- PCI DSS compliance required? *Yes — no raw card data stored*
- Multi-region? *Yes — at least 2 regions for DR*

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
1. Initiate payment (card, wallet, bank transfer)
2. Process payment via PSP (Stripe, Braintree)
3. Handle PSP webhooks (payment success, failure, refund)
4. Refund payments (full and partial)
5. Query payment status and history
6. Double-entry bookkeeping ledger (all money flows recorded)
7. Daily reconciliation against PSP records
8. Fraud detection (velocity checks, ML scoring)
9. Idempotent payment creation (safe retries)
10. Split payments for marketplace use case

### Non-Functional Requirements
| Property | Target |
|---|---|
| Durability | Zero payment loss (ACID, 3-way replication) |
| Consistency | Strong (no double charges, balanced ledger) |
| Availability | 99.99% |
| Latency | < 2s p95 for card payments (PSP-bound) |
| Throughput | 500 TPS sustained, 2K TPS burst |
| Idempotency | Exact-once semantics for payment creation |
| Security | PCI DSS Level 1 compliance |

---

## 3. Capacity Estimation

### Traffic
- 10M transactions/day = 116 TPS average, 500 TPS peak
- Webhook events from PSP: ~3× transactions = 30M events/day
- Reconciliation: 10M records compared daily (nightly batch)

### Storage
- Payment record: ~1 KB each × 10M/day × 365 = **3.65 TB/year**
- Ledger entries: ~2 entries per payment × 10M = 20M rows/day = **7.3 TB/year**
- Idempotency keys: 30-day retention × 10M/day = **300M active keys**
- Idempotency key storage: 300M × 200 bytes = **60 GB** (fits in Redis)

### Database
- Active payments (90 days): 900M records → **900 GB**
- Ledger entries (7-year retention for compliance): **~50 TB** → archived to cold storage
- PostgreSQL primary + 2 synchronous replicas + 1 async replica for reporting

---

## 4. High-Level Architecture

```
                ┌────────────────────────────────────────────────┐
                │                  Clients                        │
                │   (Mobile App / Web / Merchant API)            │
                └──────────────────┬─────────────────────────────┘
                                   │
                ┌──────────────────▼─────────────────────────────┐
                │              API Gateway                        │
                │  (Auth, Rate Limiting, TLS, Idempotency check) │
                └──────┬──────────────────────┬──────────────────┘
                       │                      │
          ┌────────────▼───────┐  ┌───────────▼──────────────────┐
          │  Payment Service   │  │   Webhook Handler            │
          │  (initiate, status,│  │   (PSP → internal events)    │
          │   refund)          │  │   Signature verification     │
          └────────┬───────────┘  └───────────┬──────────────────┘
                   │                          │
          ┌────────▼──────────────────────────▼──────────────────┐
          │                   Kafka Message Bus                   │
          │  Topics: payment.initiated, payment.completed,        │
          │          payment.failed, payment.refunded,            │
          │          reconciliation.mismatch                      │
          └───┬────────────┬──────────────┬──────────────────────┘
              │            │              │
    ┌─────────▼──┐  ┌──────▼────┐  ┌─────▼──────────────────────┐
    │ Ledger     │  │ Fraud     │  │ Reconciliation Service      │
    │ Service    │  │ Detection │  │ (nightly batch)             │
    │ (double-   │  │ Service   │  │ PSP records vs internal DB  │
    │  entry)    │  │           │  └────────────────────────────┘
    └─────────┬──┘  └───────────┘
              │
    ┌─────────▼────────────────────────────────────────────────┐
    │                    Data Layer                             │
    │  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐  │
    │  │  PostgreSQL  │  │   Redis     │  │  Vault (HashiCorp│  │
    │  │  (payments,  │  │  (idempotency│  │   or AWS KMS)  │  │
    │  │   ledger,    │  │   cache,    │  │  Card tokenization│ │
    │  │   idempotency│  │   rate limit│  │  PCI DSS scope  │  │
    │  │   keys)      │  │   keys)     │  └─────────────────┘  │
    │  └──────────────┘  └─────────────┘                       │
    └──────────────────────────────────────────────────────────┘

                    ┌─────────────────────────┐
                    │   External PSPs          │
                    │   Stripe / Braintree     │
                    │   (webhook delivery)     │
                    └─────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Payment Flow (Initiate → Confirm → Settle)

```
Client → API GW → Payment Service → Check idempotency key (Redis)
                                   → Create payment record (status=PENDING)
                                   → Call PSP API (Stripe charge)
                                   → On success: update status=PROCESSING
                                   → Receive webhook from PSP
                                   → Update status=COMPLETED
                                   → Write ledger entries (debit/credit)
                                   → Publish payment.completed to Kafka
                                   → Notification Service sends receipt
```

### 5.2 Idempotency Keys

**Problem:** Network timeout on payment — client retries → double charge.

**Solution:**
1. Client generates a UUID idempotency key and includes it in the request header (`Idempotency-Key: <uuid>`)
2. API Gateway checks Redis: if key exists and status=COMPLETED → return cached response
3. If key exists and status=PROCESSING → wait or return 202 Accepted
4. If key doesn't exist → proceed with payment, store key with response when done
5. Keys expire after 24 hours (or configurable retention)

```
Redis key: idempotency:{key}
Value: { payment_id, status, response_hash, created_at }
TTL: 86400 seconds (24 hours)
```

### 5.3 Payment State Machine

```
              initiate()
   PENDING ──────────────────────► PROCESSING
      │                                 │
      │ timeout/error                   │ psp_webhook(success)
      ▼                                 ▼
   FAILED ◄──────────── ── ──── ── COMPLETED
                                        │
                                 refund_request()
                                        │
                                        ▼
                                   REFUND_INITIATED
                                        │
                                 psp_refund_webhook()
                                        │
                                        ▼
                                    REFUNDED
```

### 5.4 Double-Entry Bookkeeping (Ledger)

Every money movement generates two entries (debit + credit) that always sum to zero:

```
Payment of $100 from Alice to Merchant:
  DEBIT  alice.wallet          $100   (money leaves Alice)
  CREDIT merchant.receivable   $100   (money arrives at Merchant)

Platform fee ($2):
  DEBIT  merchant.receivable   $2     (fee deducted from merchant)
  CREDIT platform.revenue      $2     (platform earns)

Refund of $100:
  DEBIT  merchant.receivable   $100   (reverses the original credit)
  CREDIT alice.wallet          $100   (money returns to Alice)
```

**Invariant:** `SUM(all debit amounts) == SUM(all credit amounts)` always.
This is verified by a nightly balance check job.

### 5.5 PSP Webhook Handling

PSPs send HTTP POST callbacks when payment status changes:
1. **Signature verification:** Compute HMAC-SHA256 of webhook body using shared secret; reject if signature doesn't match
2. **At-least-once delivery:** PSPs retry webhooks for hours if we return non-2xx; ensure idempotent handling
3. **Event deduplication:** Store processed `event_id` in DB; skip if already processed
4. **Async processing:** Webhook handler ACKs immediately (returns 200), pushes to Kafka, processes asynchronously

### 5.6 Currency Handling

**Critical rule: NEVER store money as float.**
- Float arithmetic: `0.1 + 0.2 = 0.30000000000000004`
- **Solution:** Store all amounts as integers in minor units (cents, pence, paisa)
  - $10.99 USD → stored as `1099` (integer)
  - €5.00 EUR → stored as `500`
- Currency code stored separately (ISO 4217: USD, EUR, GBP)
- Conversion only at display layer; never in arithmetic

### 5.7 Fraud Detection

**Velocity Checks (rule-based, real-time):**
- Max N transactions per user per hour (e.g., 10)
- Max spend per user per day (e.g., $5,000)
- Max transactions per card per hour (e.g., 5)
- Block if user's IP is on known fraud list

**Behavioral Signals:**
- New device + high value transaction → require 3DS challenge
- Transaction from unusual geography (user always in NYC, now Bucharest)
- Pattern: many small charges rapidly (card testing)

**Decision:** Approve / Review (hold for manual review) / Decline

---

## 6. Database Design

```sql
-- Payments (one row per payment attempt)
CREATE TABLE payments (
    payment_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL,
    merchant_id     UUID,
    amount          BIGINT NOT NULL CHECK (amount > 0),  -- minor units
    currency        CHAR(3) NOT NULL DEFAULT 'USD',
    status          VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    payment_method  VARCHAR(30),  -- CARD, WALLET, ACH
    psp             VARCHAR(20),  -- STRIPE, BRAINTREE
    psp_reference   VARCHAR(255) UNIQUE,  -- PSP charge ID
    idempotency_key VARCHAR(255) UNIQUE,
    metadata        JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    INDEX idx_user (user_id),
    INDEX idx_status (status),
    INDEX idx_psp_ref (psp_reference),
    INDEX idx_idempotency (idempotency_key),
    INDEX idx_created (created_at)
);

-- Ledger entries (immutable double-entry bookkeeping)
CREATE TABLE ledger_entries (
    entry_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    payment_id      UUID NOT NULL REFERENCES payments(payment_id),
    account_id      VARCHAR(100) NOT NULL,  -- alice.wallet, merchant.receivable
    entry_type      VARCHAR(10) NOT NULL,   -- DEBIT, CREDIT
    amount          BIGINT NOT NULL CHECK (amount > 0),
    currency        CHAR(3) NOT NULL,
    description     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    INDEX idx_payment (payment_id),
    INDEX idx_account (account_id),
    INDEX idx_created (created_at)
) PARTITION BY RANGE (created_at);  -- monthly partitions for compliance archival

-- Idempotency keys (prevent duplicate payments)
CREATE TABLE idempotency_keys (
    idem_key        VARCHAR(255) PRIMARY KEY,
    payment_id      UUID REFERENCES payments(payment_id),
    user_id         UUID NOT NULL,
    status          VARCHAR(20),
    response_code   INT,
    response_body   JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    expires_at      TIMESTAMPTZ NOT NULL  -- TTL for cleanup
);

-- Webhook events (processed PSP events — for deduplication)
CREATE TABLE webhook_events (
    event_id        VARCHAR(255) PRIMARY KEY,  -- PSP event ID
    psp             VARCHAR(20) NOT NULL,
    event_type      VARCHAR(50),
    payload         JSONB,
    processed_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Refunds
CREATE TABLE refunds (
    refund_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    payment_id      UUID NOT NULL REFERENCES payments(payment_id),
    amount          BIGINT NOT NULL,
    reason          TEXT,
    status          VARCHAR(20) DEFAULT 'PENDING',
    psp_refund_ref  VARCHAR(255),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

**Key Constraints:**
- `amount` stored as integer (minor units) — enforced at DB level
- `idempotency_key` has UNIQUE constraint — DB-level duplicate prevention
- `psp_reference` UNIQUE — prevents duplicate PSP charges from being recorded twice
- Ledger entries are INSERT-ONLY (immutable audit trail)

---

## 7. API Design

```
POST /v1/payments
Headers: Idempotency-Key: <uuid>
Body: {
  amount: 9999,          // $99.99 in cents
  currency: "USD",
  payment_method_id: "pm_xxx",  // tokenized card
  merchant_id: "merch_yyy",
  metadata: { order_id: "ord_123" }
}
Response 201: { payment_id, status: "PROCESSING", created_at }

GET /v1/payments/{payment_id}
Response: { payment_id, status, amount, currency, created_at, updated_at }

GET /v1/payments?user_id=xxx&status=COMPLETED&from=2025-01-01&limit=50

POST /v1/payments/{payment_id}/refund
Body: { amount: 5000, reason: "Customer request" }  // partial: $50.00
Response: { refund_id, status: "PENDING", amount }

GET /v1/payments/{payment_id}/refund/{refund_id}

GET /v1/ledger/{account_id}?from=2025-01-01&to=2025-01-31
Response: { entries: [...], balance: { debit_total, credit_total } }

POST /v1/webhooks/stripe
Headers: Stripe-Signature: t=...,v1=...
Body: { Stripe event JSON }
Response: 200 OK (always — async processing)
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Database Write Throughput (500 TPS)
- Each payment = 1 payments row + 2 ledger entries + 1 idempotency key = **4 writes/transaction**
- 500 TPS × 4 = 2,000 DB writes/second
- **Solution:** PostgreSQL with connection pooling (PgBouncer), batch inserts for ledger entries, synchronous replicas for durability, async replica for reports

### Bottleneck 2: PSP Latency (Stripe API ~200ms-1s)
- Payment confirmation is PSP-bound; can't reduce network latency to Stripe
- **Solution:** Set client-facing timeout at 30s; PSP calls are in Payment Service thread pool; don't block the HTTP thread; use async I/O

### Bottleneck 3: Idempotency Key Lookup (All requests)
- Every payment request must check idempotency key first
- **Solution:** Redis cache for hot keys (90% of retries are within 5 minutes); DB as source of truth for older keys

### Bottleneck 4: Webhook Volume (30M/day = 350/second)
- PSPs can send bursts of webhooks during incidents
- **Solution:** Webhook handler puts events on Kafka immediately (< 1ms); Kafka buffers bursts; downstream consumers process at their own pace

### Bottleneck 5: Reconciliation Scale (10M records/night)
- Nightly batch must compare 10M internal records vs PSP CSV export
- **Solution:** Sort-merge join on `psp_reference`; parallelized by merchant/date partition; completes in < 2 hours; mismatch events published to Kafka for manual review

---

## 9. Trade-offs & Design Decisions

### Decision 1: ACID vs Performance
- **Chosen:** Full ACID via PostgreSQL with synchronous replication
- **Why:** Money movement must be consistent — partial writes are catastrophic (debit without credit, double charge)
- **Trade-off:** Lower throughput vs NoSQL; mitigated by connection pooling, partitioning

### Decision 2: Synchronous vs Async Payment Confirmation
- **Chosen:** Synchronous PSP call during payment initiation, async webhook for final status
- **Why:** Client needs an immediate "payment received" response; final settlement confirmation can arrive asynchronously
- **Risk:** PSP times out → payment status unknown → idempotency key retry resolves this

### Decision 3: Storing Idempotency Keys in Redis vs PostgreSQL
- **Chosen:** Both (Redis as fast cache, PostgreSQL as durable store)
- **Why:** Redis gives sub-millisecond lookup for recent retries; PostgreSQL ensures durability if Redis restarts

### Decision 4: Ledger as Event Log vs Mutable Balance Table
- **Chosen:** Immutable ledger entries + derived balance (never store a mutable "balance" field)
- **Why:** Immutable audit trail is required for compliance; balance can always be recomputed from ledger; prevents "balance inconsistency" class of bugs

### Decision 5: Fraud Detection — Synchronous vs Async
- **Chosen:** Synchronous for rule-based velocity checks (< 10ms), async ML scoring runs post-authorization
- **Why:** Simple rules must block before charging; ML scoring can flag for review after the fact without blocking the payment

---

## 10. Key Interview Talking Points

1. **Idempotency is the #1 payment engineering problem.** Any retry, network blip, or duplicate request must never cause a double charge. Idempotency keys (client-generated UUID + server-side deduplication) are the standard solution.

2. **Double-Entry Bookkeeping:** Every credit has a matching debit. The sum of all ledger entries must always be zero. This invariant, enforced by the application and verified by nightly reconciliation, makes it impossible to lose money silently.

3. **Currency as Integers:** `float` types cannot represent decimal currency exactly due to IEEE 754 binary representation. Store $1.99 as `199` (cents). Convert only at the presentation layer.

4. **Webhook Idempotency:** PSPs retry webhooks on failure. The webhook handler must be idempotent — process each `event_id` exactly once. Store processed event IDs in a deduplication table.

5. **Two-Phase Payment:** Authorize (card check) → Capture (charge). Many payment systems separate these. Authorize happens at checkout; capture at fulfillment. Uncaptured authorizations are released after 7 days.

6. **Reconciliation:** Even with perfect code, PSPs can have bugs (duplicate charges, missed settlements). Daily reconciliation compares your ledger vs PSP reports. Mismatches trigger alerts for human review and automatic dispute filing.

7. **PCI DSS Compliance:** Never store raw PAN (card numbers). Use PSP tokenization — you store a token (e.g., `tok_visa_4242`); the PSP stores the card number in their PCI-compliant vault. This scopes PCI compliance to the PSP, not your system.

8. **Split Payments:** Marketplace model: $100 order → $95 to seller + $5 platform fee. Handled by a single atomic transaction that creates multiple ledger entries. PSP split payment APIs (Stripe Connect) handle the actual fund routing.

9. **Failure Scenarios to Discuss:**
   - PSP times out → status PENDING, idempotency key set → client retry returns cached response
   - DB down after PSP charge → WAL recovery restores the payment record → PSP reconciliation catches it
   - Webhook lost → reconciliation detects missing settlement → raises alert

10. **Key Metrics:** TPS, p99 payment latency, double-charge rate (must be 0), reconciliation mismatch rate (target < 0.001%), fraud false positive rate (blocks legitimate payments), webhook processing lag.
