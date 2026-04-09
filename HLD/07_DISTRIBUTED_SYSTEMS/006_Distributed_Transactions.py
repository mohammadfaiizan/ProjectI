"""
DISTRIBUTED TRANSACTIONS
==========================

Problem Statement:
A transaction spanning multiple services/databases must be atomic:
either ALL operations commit, or NONE do.
Single-database ACID transactions don't extend across service boundaries.

Approaches (in order of complexity/reliability):

  1. Two-Phase Commit (2PC):
     Phase 1 (Prepare): Coordinator asks all participants "can you commit?"
       Each participant locks resources and responds "YES" or "NO".
     Phase 2 (Commit/Abort): If all YES → coordinator sends COMMIT to all.
       If any NO → coordinator sends ABORT to all.
     Problem: blocking protocol. If coordinator crashes after prepare,
       participants hold locks forever (blocked). Single coordinator = SPOF.
     Used by: XA transactions (JDBC), some distributed databases.

  2. Three-Phase Commit (3PC):
     Adds a pre-commit phase to allow recovery without blocking.
     Still vulnerable to network partition. Complex. Rarely used in practice.

  3. Saga Pattern:
     Break transaction into local transactions + compensating transactions.
     If Ti fails: run C(i-1), C(i-2), ..., C(1) to undo.
     No locks held across services. Eventually consistent (not ACID).
     See 011_Saga_Pattern_For_Distributed_Txns.py for full implementation.

  4. Transactional Outbox:
     Write event to same DB transaction as state change.
     Outbox table polled → messages published.
     Achieves at-least-once event delivery without 2PC.

  5. Try-Confirm-Cancel (TCC):
     Phase 1 (Try): Reserve resources in each service.
     Phase 2a (Confirm): All reservations committed.
     Phase 2b (Cancel): If any Try failed → cancel all reservations.
     Resources locked briefly. Requires compensating logic.
     Used by: Alibaba's Seata framework.

  6. Best-Effort Delivery + Idempotency:
     Give up on atomicity. Accept eventual consistency.
     Each operation is idempotent. Retry until success.
     Only suitable for some workloads (not financial).

CAP and Transactions:
  Strict ACID across distributed nodes requires CP (consensus, coordination).
  Most modern systems accept BASE (eventual) over full ACID.
  Where ACID needed: use single-shard transactions or XA within same DB.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import time
import uuid
import threading
import random


# ─────────────────────────────────────────────
# 2PC STATES
# ─────────────────────────────────────────────

class TxnState(Enum):
    INIT       = "init"
    PREPARING  = "preparing"
    PREPARED   = "prepared"
    COMMITTING = "committing"
    COMMITTED  = "committed"
    ABORTING   = "aborting"
    ABORTED    = "aborted"


# ─────────────────────────────────────────────
# 2PC PARTICIPANT
# ─────────────────────────────────────────────

class TwoPhaseParticipant:
    """
    Simulates a 2PC participant (e.g., a microservice or database shard).
    """

    def __init__(self, participant_id: str, failure_rate: float = 0.0):
        self.participant_id = participant_id
        self.failure_rate   = failure_rate
        self._prepared_txns : Dict[str, Any]     = {}   # txn_id → prepared data
        self._committed     : Dict[str, Any]     = {}
        self._lock          = threading.Lock()
        self.prepare_calls  = 0
        self.commit_calls   = 0
        self.abort_calls    = 0

    def prepare(self, txn_id: str, operation: Dict) -> bool:
        """
        Phase 1: Validate + acquire locks + write to WAL.
        Returns True if ready to commit.
        """
        self.prepare_calls += 1
        if random.random() < self.failure_rate:
            return False   # simulate failure

        with self._lock:
            if txn_id in self._prepared_txns:
                return True   # idempotent
            # Validate
            if operation.get("amount", 0) > 10000:
                return False   # business rule violation
            self._prepared_txns[txn_id] = operation
        return True

    def commit(self, txn_id: str) -> bool:
        """Phase 2a: Apply and release locks."""
        self.commit_calls += 1
        with self._lock:
            if txn_id not in self._prepared_txns:
                return False
            operation = self._prepared_txns.pop(txn_id)
            self._committed[txn_id] = operation
        return True

    def abort(self, txn_id: str) -> bool:
        """Phase 2b: Roll back and release locks."""
        self.abort_calls += 1
        with self._lock:
            self._prepared_txns.pop(txn_id, None)
        return True

    def is_committed(self, txn_id: str) -> bool:
        return txn_id in self._committed


# ─────────────────────────────────────────────
# 2PC COORDINATOR
# ─────────────────────────────────────────────

class TwoPCCoordinator:
    """
    Two-Phase Commit Coordinator.
    Manages the distributed transaction protocol.
    Persists state to WAL to allow recovery on crash.
    """

    def __init__(self, coordinator_id: str):
        self.coordinator_id = coordinator_id
        self._wal           : Dict[str, TxnState] = {}   # txn_id → state
        self.committed      = 0
        self.aborted        = 0

    def execute(self, txn_id: str, participants: List[TwoPhaseParticipant],
                operations: List[Dict]) -> Tuple[bool, str]:
        """
        Returns (success, reason).
        """
        # Record start
        self._wal[txn_id] = TxnState.PREPARING
        print(f"    [Coordinator] BEGIN txn={txn_id}")

        # Phase 1: Prepare all
        votes = {}
        for i, participant in enumerate(participants):
            operation = operations[i] if i < len(operations) else {}
            vote = participant.prepare(txn_id, operation)
            votes[participant.participant_id] = vote
            print(f"    [Coordinator] PREPARE {participant.participant_id}: vote={vote}")

        all_yes = all(votes.values())
        self._wal[txn_id] = TxnState.PREPARED if all_yes else TxnState.ABORTING

        # Phase 2: Commit or Abort
        if all_yes:
            self._wal[txn_id] = TxnState.COMMITTING
            for participant in participants:
                participant.commit(txn_id)
                print(f"    [Coordinator] COMMIT {participant.participant_id}")
            self._wal[txn_id] = TxnState.COMMITTED
            self.committed += 1
            print(f"    [Coordinator] COMMITTED txn={txn_id}")
            return True, "committed"
        else:
            failed = [pid for pid, vote in votes.items() if not vote]
            for participant in participants:
                participant.abort(txn_id)
                print(f"    [Coordinator] ABORT {participant.participant_id}")
            self._wal[txn_id] = TxnState.ABORTED
            self.aborted += 1
            print(f"    [Coordinator] ABORTED txn={txn_id} (failed: {failed})")
            return False, f"participant(s) voted NO: {failed}"


# ─────────────────────────────────────────────
# TRANSACTIONAL OUTBOX PATTERN
# ─────────────────────────────────────────────

@dataclass
class OutboxEntry:
    entry_id    : str  = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type  : str  = ""
    payload     : Any  = None
    published   : bool = False
    created_at  : float = field(default_factory=time.time)


class TransactionalOutbox:
    """
    Write events to the same 'transaction' as state changes.
    A background poller publishes events from the outbox.
    Guarantees at-least-once event delivery.
    """

    def __init__(self):
        self._orders  : Dict[str, Dict] = {}
        self._outbox  : List[OutboxEntry] = []
        self._lock    = threading.Lock()
        self.published_count = 0

    def create_order(self, order_id: str, amount: float):
        """Atomically create order + write to outbox (same 'transaction')."""
        with self._lock:
            self._orders[order_id] = {"amount": amount, "status": "pending"}
            self._outbox.append(OutboxEntry(
                event_type = "order.created",
                payload    = {"order_id": order_id, "amount": amount},
            ))

    def poll_and_publish(self, publisher: Callable[[OutboxEntry], bool]):
        """Relay: poll outbox, publish unpublished entries."""
        with self._lock:
            unpublished = [e for e in self._outbox if not e.published]

        for entry in unpublished:
            if publisher(entry):
                with self._lock:
                    entry.published = True
                    self.published_count += 1

    @property
    def pending_entries(self) -> int:
        return sum(1 for e in self._outbox if not e.published)


# ─────────────────────────────────────────────
# TCC (Try-Confirm-Cancel) Pattern
# ─────────────────────────────────────────────

class TCCResource:
    """
    TCC participant: Try reserves; Confirm commits; Cancel releases.
    """

    def __init__(self, resource_id: str, initial_balance: float):
        self.resource_id = resource_id
        self._balance    = initial_balance
        self._reserved   : Dict[str, float] = {}   # txn_id → reserved amount
        self._lock       = threading.Lock()

    def try_reserve(self, txn_id: str, amount: float) -> bool:
        """Phase 1: Reserve amount tentatively."""
        with self._lock:
            available = self._balance - sum(self._reserved.values())
            if available < amount:
                return False
            self._reserved[txn_id] = amount
            return True

    def confirm(self, txn_id: str) -> bool:
        """Phase 2a: Commit reservation."""
        with self._lock:
            amount = self._reserved.pop(txn_id, None)
            if amount is None:
                return False
            self._balance -= amount
            return True

    def cancel(self, txn_id: str) -> bool:
        """Phase 2b: Release reservation."""
        with self._lock:
            self._reserved.pop(txn_id, None)
            return True

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def available(self) -> float:
        return self._balance - sum(self._reserved.values())


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_distributed_transactions():
    print("=" * 65)
    print("DISTRIBUTED TRANSACTIONS")
    print("=" * 65)

    random.seed(0)

    # ── 2PC Success ───────────────────────────────
    print("\n[1] TWO-PHASE COMMIT — SUCCESS")
    print("─" * 55)

    coord     = TwoPCCoordinator("coordinator")
    inventory = TwoPhaseParticipant("inventory-svc")
    payment   = TwoPhaseParticipant("payment-svc")
    shipping  = TwoPhaseParticipant("shipping-svc")

    ok, reason = coord.execute(
        txn_id       = "TXN-001",
        participants = [inventory, payment, shipping],
        operations   = [
            {"action": "reserve_stock", "sku": "A1", "qty": 2},
            {"action": "charge",        "amount": 99.0},
            {"action": "create_label",  "address": "123 Main St"},
        ],
    )
    print(f"  Result: success={ok} reason={reason}")

    # ── 2PC Abort (one participant fails) ─────────
    print("\n\n[2] TWO-PHASE COMMIT — PARTICIPANT FAILS → ABORT")
    print("─" * 55)

    coord2    = TwoPCCoordinator("coordinator")
    inventory2 = TwoPhaseParticipant("inventory-svc")
    payment2   = TwoPhaseParticipant("payment-svc", failure_rate=1.0)  # always fails

    ok2, reason2 = coord2.execute(
        txn_id       = "TXN-002",
        participants = [inventory2, payment2],
        operations   = [
            {"action": "reserve_stock", "amount": 50.0},
            {"action": "charge",        "amount": 150.0},
        ],
    )
    print(f"  Result: success={ok2} reason={reason2}")
    print(f"  inventory prepared then aborted: "
          f"prepare={inventory2.prepare_calls} abort={inventory2.abort_calls}")

    # ── Transactional Outbox ──────────────────────
    print("\n\n[3] TRANSACTIONAL OUTBOX — AT-LEAST-ONCE EVENTS")
    print("─" * 55)

    outbox     = TransactionalOutbox()
    published  = []

    def mock_publisher(entry: OutboxEntry) -> bool:
        published.append(entry.event_type)
        return True

    for i in range(3):
        outbox.create_order(f"ORD-{i:03d}", amount=(i + 1) * 100.0)

    print(f"  Orders created: 3  Outbox pending: {outbox.pending_entries}")
    outbox.poll_and_publish(mock_publisher)
    print(f"  After poller: published={outbox.published_count} "
          f"pending={outbox.pending_entries}")
    print(f"  Events published: {published}")

    # ── TCC Pattern ───────────────────────────────
    print("\n\n[4] TRY-CONFIRM-CANCEL (TCC) PATTERN")
    print("─" * 55)

    account_a = TCCResource("account-A", initial_balance=500.0)
    account_b = TCCResource("account-B", initial_balance=100.0)

    txn_id = "TXN-003"
    amount = 200.0

    print(f"  Transfer ${amount} from A to B")
    print(f"  Before: A=${account_a.balance} B=${account_b.balance}")

    # Try phase
    try_a = account_a.try_reserve(txn_id, amount)
    try_b = account_b.try_reserve(txn_id, 0)   # B is destination (no reservation needed)
    print(f"  Try: A={try_a} B={try_b}")

    if try_a and try_b:
        account_a.confirm(txn_id)
        account_b._balance += amount   # credit destination
        print(f"  Confirm: transfer complete")
    else:
        account_a.cancel(txn_id)
        account_b.cancel(txn_id)

    print(f"  After: A=${account_a.balance} B=${account_b.balance}")

    # TCC failure: insufficient funds
    txn_id2 = "TXN-004"
    try_a2   = account_a.try_reserve(txn_id2, 1000.0)   # more than available
    print(f"\n  Transfer $1000 from A (balance=${account_a.balance}):")
    print(f"  Try: A={try_a2} (cancelled)")

    # ── Pattern Comparison ────────────────────────
    print("\n\n[5] DISTRIBUTED TRANSACTION PATTERN COMPARISON")
    print("─" * 55)
    patterns = [
        ("2PC",           "Blocking, coordinator SPOF", "JDBC XA, some DBs", "Low"),
        ("3PC",           "Non-blocking, complex",       "Rare, academic",   "Very Low"),
        ("Saga",          "No locks, eventual",          "Microservices",    "High"),
        ("Outbox",        "At-least-once events",        "Event-driven",     "High"),
        ("TCC",           "Short locks, reversible",     "Seata (Alibaba)",  "Medium"),
        ("Best-effort",   "Idempotent + retry",          "Simple services",  "Very High"),
    ]
    print(f"  {'Pattern':<15} {'Trade-off':<30} {'Used by':<20} {'Availability'}")
    print(f"  {'─'*78}")
    for pattern, tradeoff, used_by, avail in patterns:
        print(f"  {pattern:<15} {tradeoff:<30} {used_by:<20} {avail}")


if __name__ == "__main__":
    demonstrate_distributed_transactions()
