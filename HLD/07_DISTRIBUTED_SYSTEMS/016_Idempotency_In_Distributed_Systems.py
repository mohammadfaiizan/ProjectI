"""
IDEMPOTENCY IN DISTRIBUTED SYSTEMS
=====================================

Problem Statement:
In distributed systems, "exactly-once" delivery is hard.
At-least-once delivery is easier but creates duplicates.
If operations are idempotent, duplicates don't matter: processing twice = same result as once.

Idempotency Definition:
  f(f(x)) = f(x) — applying the operation multiple times has the same effect as once.

Examples:
  Idempotent:     SET balance = 100 (same result every time)
  NOT idempotent: INCREMENT balance (each call changes state)
  Idempotent:     DELETE order WHERE id='123' (second delete is a no-op)
  NOT idempotent: INSERT INTO orders (same insert twice = duplicate row)

Idempotency Key Pattern:
  Client generates a unique idempotency_key per logical operation.
  Server stores: (idempotency_key → result) with TTL.
  On receipt: if key seen before → return stored result. Don't re-execute.
  If key not seen: execute + store result.
  Used by: Stripe API, PayPal, all major payment APIs.

Idempotency Key Requirements:
  - Unique per logical operation (not per HTTP request).
  - Client generates it (UUID or UUID based on request content).
  - TTL: 24 hours (stripe) to 7 days.
  - Atomic check-and-store (DB transaction or Redis SETNX).

Implementation Patterns:
  1. Database deduplication: unique constraint on idempotency_key.
     INSERT ... ON CONFLICT DO NOTHING → no duplicate processing.

  2. Redis check-and-set: SETNX on key → process if success, skip if fail.
     Fast but Redis is ephemeral — use DB for durability.

  3. Message dedup window: MQ broker deduplicates within N minutes.
     AWS SQS FIFO queues: dedup_id → 5-minute dedup window.

  4. Natural idempotency (best): design operations to be inherently idempotent.
     Use SET instead of INCREMENT. Use conditional UPDATE (WHERE version=N).
     Use INSERT ON CONFLICT (upsert). Use PUT instead of POST.

Conditional Writes:
  UPDATE accounts SET balance=100 WHERE balance=120
  If balance changed → update does nothing. Caller can retry if needed.
  No idempotency key needed. No extra state.

HTTP Idempotency:
  GET:    Idempotent (no state change).
  PUT:    Idempotent (set to value).
  DELETE: Idempotent (delete is no-op if already gone).
  POST:   NOT idempotent by default. Add Idempotency-Key header to make it so.
  PATCH:  NOT idempotent by default (increment operations).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict
import time
import uuid
import threading


# ─────────────────────────────────────────────
# IDEMPOTENCY KEY STORE
# ─────────────────────────────────────────────

@dataclass
class IdempotencyRecord:
    key        : str
    result     : Any
    status     : str     # "processing" | "completed" | "failed"
    created_at : float = field(default_factory=time.time)
    expires_at : float = 0.0


class IdempotencyStore:
    """
    Stores idempotency keys with results.
    Supports atomic check-and-set to prevent concurrent duplicate processing.
    """

    def __init__(self, ttl_s: float = 3600.0):
        self._records : Dict[str, IdempotencyRecord] = {}
        self._lock    = threading.Lock()
        self.ttl_s    = ttl_s
        self.hits     = 0
        self.misses   = 0

    def check_or_start(self, key: str) -> Tuple[bool, Optional[Any]]:
        """
        Atomically:
        - If key exists and completed → return (True, result) [duplicate]
        - If key exists and processing → return (True, None) [in-progress]
        - If key not seen → mark as processing, return (False, None) [new]
        """
        with self._lock:
            now     = time.time()
            record  = self._records.get(key)

            if record and record.expires_at > now:
                self.hits += 1
                if record.status == "completed":
                    return True, record.result
                return True, None   # still processing

            # New key — mark as processing
            self._records[key] = IdempotencyRecord(
                key        = key,
                result     = None,
                status     = "processing",
                expires_at = now + self.ttl_s,
            )
            self.misses += 1
            return False, None

    def complete(self, key: str, result: Any):
        with self._lock:
            record = self._records.get(key)
            if record:
                record.result = result
                record.status = "completed"

    def fail(self, key: str, error: str):
        with self._lock:
            record = self._records.get(key)
            if record:
                record.result = error
                record.status = "failed"
                # Remove so next attempt can retry (or keep depending on strategy)
                del self._records[key]

    def cleanup(self):
        with self._lock:
            now = time.time()
            expired = [k for k, r in self._records.items() if r.expires_at < now]
            for k in expired:
                del self._records[k]
            return len(expired)


# ─────────────────────────────────────────────
# IDEMPOTENT PAYMENT SERVICE
# ─────────────────────────────────────────────

class PaymentService:
    """
    Stripe-style idempotent payment endpoint.
    Caller provides idempotency_key (UUID). Duplicate calls return same result.
    """

    def __init__(self):
        self.idem_store   = IdempotencyStore(ttl_s=86400)   # 24h
        self._charges     : Dict[str, float] = {}
        self.executions   = 0    # how many times actually charged

    def charge(self, customer_id: str, amount: float,
               idempotency_key: str) -> Dict[str, Any]:
        """Returns charge result. Idempotent via idempotency_key."""
        is_dup, cached_result = self.idem_store.check_or_start(idempotency_key)

        if is_dup and cached_result is not None:
            return {**cached_result, "from_cache": True}

        if is_dup and cached_result is None:
            # Still processing (concurrent duplicate) — wait briefly
            time.sleep(0.01)
            _, cached_result2 = self.idem_store.check_or_start(idempotency_key)
            if cached_result2:
                return {**cached_result2, "from_cache": True}

        # Execute the charge
        self.executions += 1
        charge_id = str(uuid.uuid4())[:8]
        self._charges[charge_id] = amount
        result = {
            "charge_id"  : charge_id,
            "customer_id": customer_id,
            "amount"     : amount,
            "status"     : "succeeded",
            "from_cache" : False,
        }
        self.idem_store.complete(idempotency_key, result)
        return result


# ─────────────────────────────────────────────
# NATURAL IDEMPOTENCY: CONDITIONAL UPDATE
# ─────────────────────────────────────────────

class InventoryService:
    """
    Uses conditional updates for natural idempotency.
    UPDATE inventory SET qty=qty-N WHERE sku=? AND qty >= N AND version=?
    Retry-safe without an idempotency key.
    """

    def __init__(self):
        self._inventory: Dict[str, Dict] = {}

    def add_item(self, sku: str, initial_qty: int):
        self._inventory[sku] = {"qty": initial_qty, "version": 0}

    def reserve(self, sku: str, qty: int, expected_version: int) -> Tuple[bool, str]:
        """
        Idempotent reservation using optimistic locking.
        Returns (success, reason).
        """
        item = self._inventory.get(sku)
        if not item:
            return False, "sku_not_found"
        if item["version"] != expected_version:
            return False, "version_conflict"   # concurrent update
        if item["qty"] < qty:
            return False, "insufficient_stock"
        item["qty"]     -= qty
        item["version"] += 1
        return True, "reserved"

    def get(self, sku: str) -> Optional[Dict]:
        return self._inventory.get(sku)


# ─────────────────────────────────────────────
# MESSAGE DEDUPLICATION (SQS FIFO-style)
# ─────────────────────────────────────────────

class SQSFifoQueue:
    """
    SQS FIFO-style deduplication window.
    Messages with same MessageDeduplicationId within 5 minutes are dropped.
    """

    def __init__(self, dedup_window_s: float = 5 * 60):
        self._dedup_window = dedup_window_s
        self._seen         : Dict[str, float] = {}   # dedup_id → expiry
        self._queue        : List[Dict] = []
        self.accepted      = 0
        self.dropped       = 0

    def send(self, message: Any, dedup_id: str) -> bool:
        now = time.time()
        if dedup_id in self._seen and self._seen[dedup_id] > now:
            self.dropped += 1
            return False   # duplicate
        self._seen[dedup_id] = now + self._dedup_window
        self._queue.append({"message": message, "dedup_id": dedup_id})
        self.accepted += 1
        return True

    def receive(self) -> Optional[Dict]:
        return self._queue.pop(0) if self._queue else None


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_idempotency():
    print("=" * 65)
    print("IDEMPOTENCY IN DISTRIBUTED SYSTEMS")
    print("=" * 65)

    # ── Idempotent Payment ────────────────────────
    print("\n[1] IDEMPOTENT PAYMENT — STRIPE-STYLE")
    print("─" * 55)

    svc = PaymentService()

    # First charge
    idem_key = str(uuid.uuid4())
    result1  = svc.charge("cust-A", 99.0, idem_key)
    print(f"  First charge: charge_id={result1['charge_id']} "
          f"from_cache={result1['from_cache']}")

    # Retry (network failure, client retries)
    result2 = svc.charge("cust-A", 99.0, idem_key)
    print(f"  Retry with same key: charge_id={result2['charge_id']} "
          f"from_cache={result2['from_cache']}")

    # New charge with different key
    result3 = svc.charge("cust-A", 99.0, str(uuid.uuid4()))
    print(f"  New key: charge_id={result3['charge_id']} "
          f"from_cache={result3['from_cache']}")

    print(f"\n  Actual executions: {svc.executions} "
          f"(3 calls but only 2 actual charges)")
    print(f"  Cache hits: {svc.idem_store.hits}  Misses: {svc.idem_store.misses}")

    # ── Conditional Update ────────────────────────
    print("\n\n[2] NATURAL IDEMPOTENCY — CONDITIONAL UPDATE")
    print("─" * 55)

    inv = InventoryService()
    inv.add_item("SKU-A", 10)
    print(f"  Initial stock: SKU-A={inv.get('SKU-A')['qty']}")

    # First reservation
    ok1, reason1 = inv.reserve("SKU-A", qty=3, expected_version=0)
    print(f"  Reserve 3 (v=0): {ok1} ({reason1})")
    print(f"  Stock after: {inv.get('SKU-A')}")

    # Retry same reservation (network glitch) — now version=1, conflict
    ok2, reason2 = inv.reserve("SKU-A", qty=3, expected_version=0)
    print(f"  Retry reserve 3 (v=0): {ok2} ({reason2})")

    # Correct retry with updated version
    ok3, reason3 = inv.reserve("SKU-A", qty=2, expected_version=1)
    print(f"  Reserve 2 (v=1): {ok3} ({reason3})")

    # ── SQS FIFO Deduplication ────────────────────
    print("\n\n[3] MESSAGE QUEUE DEDUPLICATION (SQS FIFO)")
    print("─" * 55)

    q = SQSFifoQueue(dedup_window_s=5.0)

    sends = [
        ("Process order ORD-001", "dedup-ORD-001"),
        ("Process order ORD-002", "dedup-ORD-002"),
        ("Process order ORD-001", "dedup-ORD-001"),   # duplicate
        ("Process order ORD-001", "dedup-ORD-001"),   # duplicate
        ("Process order ORD-003", "dedup-ORD-003"),
    ]

    for msg, dedup_id in sends:
        accepted = q.send(msg, dedup_id)
        print(f"  send('{dedup_id}'): {'✓ accepted' if accepted else '✗ duplicate dropped'}")

    print(f"\n  Queue depth: {len(q._queue)} messages "
          f"(accepted={q.accepted} dropped={q.dropped})")

    # ── HTTP Idempotency Methods ───────────────────
    print("\n\n[4] HTTP METHOD IDEMPOTENCY")
    print("─" * 55)
    methods = [
        ("GET",    "Yes",   "Read-only. No state change"),
        ("PUT",    "Yes",   "Sets resource to value. Same value every time"),
        ("DELETE", "Yes",   "Deleting already-deleted resource = no-op"),
        ("POST",   "No",    "Creates new resource each time (by default)"),
        ("PATCH",  "No",    "Often increments — not idempotent"),
        ("HEAD",   "Yes",   "Like GET, no body, no state change"),
    ]
    print(f"  {'Method':<10} {'Idempotent?':<14} {'Reason'}")
    print(f"  {'─'*55}")
    for method, is_idem, reason in methods:
        print(f"  {method:<10} {is_idem:<14} {reason}")

    # ── Idempotency Design Patterns ───────────────
    print("\n\n[5] IDEMPOTENCY DESIGN PATTERNS")
    print("─" * 55)
    patterns = [
        ("Idempotency key",   "UUID per operation; server stores key+result 24h"),
        ("Natural: SET",       "Use SET x=100 instead of INCREMENT x by 1"),
        ("Conditional UPDATE", "WHERE version=N → reject stale retries"),
        ("Upsert",             "INSERT ON CONFLICT DO UPDATE — always idempotent"),
        ("Event dedup",        "Store processed event_ids in Redis (TTL-based)"),
        ("Dedup window",       "SQS FIFO: same dedup_id within 5 min dropped"),
        ("Content-based key",  "Hash(operation_params) → deterministic dedup key"),
    ]
    for pattern, description in patterns:
        print(f"  {pattern:<26} {description}")


if __name__ == "__main__":
    demonstrate_idempotency()
