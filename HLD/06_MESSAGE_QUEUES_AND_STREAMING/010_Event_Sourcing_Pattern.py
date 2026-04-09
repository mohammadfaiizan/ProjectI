"""
EVENT SOURCING PATTERN
========================

Problem Statement:
Traditional systems store current state: "Account balance = $500."
If something goes wrong, you can't ask: "What happened to get here?"
Event sourcing stores the sequence of events that produced the state:
  AccountOpened(+$1000) → MoneyWithdrawn(-$200) → MoneyDeposited(+$100) → MoneyWithdrawn(-$400)
Current state = fold (reduce) over the event log.

Key Properties:
  1. Append-only log: events are never updated or deleted. Immutable history.
  2. State is derived: apply all events to an empty aggregate → current state.
  3. Full audit trail: every change is a first-class record with who/when/why.
  4. Time-travel: replay events up to any point in time → state at that moment.
  5. Event as truth: the log IS the database. Projections are derived, rebuildable.

Snapshotting:
  Replaying all events from the beginning becomes slow for long-lived aggregates.
  Solution: periodically snapshot current state. On load: start from latest snapshot,
  replay only events after the snapshot version.
  Snapshot is an optimization — events remain authoritative.

Upcasting (Schema Evolution):
  Old events have old schemas. New code must still be able to read them.
  Upcast: transform event v1 → v2 before applying. Stored event unchanged.

Projections:
  Any read model is a projection: apply a subset of events, build a view.
  Multiple projections from the same event stream. Each rebuilds independently.

Trade-offs:
  ✓ Full audit log for free (compliance, debugging, analytics).
  ✓ Temporal queries: "What was the balance on March 1st?"
  ✓ Event-driven naturally: events are the integration points.
  ✓ Projections are disposable — schema mismatch? Rebuild.
  ✗ Query current state requires replay (or projection + snapshot).
  ✗ Schema evolution needs upcasting strategy.
  ✗ Storage grows forever (append-only). Compaction needed long-term.
  ✗ Eventual consistency between write store and projections.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict
import time
import uuid
import copy


# ─────────────────────────────────────────────
# DOMAIN EVENTS
# ─────────────────────────────────────────────

@dataclass
class Event:
    event_id     : str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type   : str   = ""
    aggregate_id : str   = ""
    version      : int   = 0          # aggregate version after this event
    payload      : Any   = None
    metadata     : Dict  = field(default_factory=dict)   # user_id, correlation_id, etc.
    timestamp    : float = field(default_factory=time.time)


# ─────────────────────────────────────────────
# EVENT STORE
# ─────────────────────────────────────────────

class OptimisticConcurrencyError(Exception):
    pass


class EventStore:
    """
    Append-only store for domain events.
    Supports optimistic concurrency: reject append if expected_version mismatch.
    """

    def __init__(self):
        self._streams : Dict[str, List[Event]] = defaultdict(list)
        self._all     : List[Event] = []

    def append(self, aggregate_id: str, events: List[Event],
               expected_version: int):
        """
        expected_version: the version the caller expects the stream to be at.
        Raises OptimisticConcurrencyError if stream has advanced (concurrent write).
        """
        stream = self._streams[aggregate_id]
        current_version = len(stream)
        if current_version != expected_version:
            raise OptimisticConcurrencyError(
                f"Expected version {expected_version}, got {current_version} "
                f"for aggregate {aggregate_id}")
        for i, evt in enumerate(events):
            evt.version = current_version + i + 1
            stream.append(evt)
            self._all.append(evt)

    def load(self, aggregate_id: str, from_version: int = 0) -> List[Event]:
        """Load events for an aggregate, optionally starting from a version."""
        return [e for e in self._streams[aggregate_id] if e.version > from_version]

    def load_all(self) -> List[Event]:
        return list(self._all)

    def current_version(self, aggregate_id: str) -> int:
        return len(self._streams.get(aggregate_id, []))


# ─────────────────────────────────────────────
# SNAPSHOT STORE
# ─────────────────────────────────────────────

@dataclass
class Snapshot:
    aggregate_id: str
    version     : int
    state       : Dict   # serialized aggregate state
    taken_at    : float = field(default_factory=time.time)


class SnapshotStore:
    def __init__(self, snapshot_every: int = 5):
        self._snapshots     : Dict[str, Snapshot] = {}
        self.snapshot_every = snapshot_every

    def save(self, snapshot: Snapshot):
        self._snapshots[snapshot.aggregate_id] = snapshot

    def latest(self, aggregate_id: str) -> Optional[Snapshot]:
        return self._snapshots.get(aggregate_id)

    def should_snapshot(self, version: int) -> bool:
        return version % self.snapshot_every == 0


# ─────────────────────────────────────────────
# BANK ACCOUNT AGGREGATE (Event Sourced)
# ─────────────────────────────────────────────

class InsufficientFundsError(Exception):
    pass

class AccountClosedError(Exception):
    pass


class BankAccountAggregate:
    """
    Event-sourced bank account. State is fully derived from events.
    No mutable DB row — only the event log.
    """

    def __init__(self, account_id: str):
        self.account_id = account_id
        self.balance    : float = 0.0
        self.owner      : str   = ""
        self.is_open    : bool  = False
        self.version    : int   = 0
        self._uncommitted: List[Event] = []

    # ── Apply (reconstitutes state) ────────────────
    def apply(self, event: Event):
        t = event.event_type
        p = event.payload
        if t == "AccountOpened":
            self.owner   = p["owner"]
            self.balance = p["initial_deposit"]
            self.is_open = True
        elif t == "MoneyDeposited":
            self.balance += p["amount"]
        elif t == "MoneyWithdrawn":
            self.balance -= p["amount"]
        elif t == "AccountClosed":
            self.is_open = False
        elif t == "InterestApplied":
            self.balance += self.balance * p["rate"]
        self.version = event.version

    def _raise(self, event_type: str, payload: Any, metadata: Dict = None):
        evt = Event(event_type=event_type, aggregate_id=self.account_id,
                    payload=payload, metadata=metadata or {})
        self._uncommitted.append(evt)
        # Apply locally so business rules can read updated state
        evt.version = self.version + len(self._uncommitted)
        self.apply(evt)

    # ── Command Handlers (business logic + validation) ──
    def open(self, owner: str, initial_deposit: float, metadata: Dict = None):
        if self.is_open:
            raise ValueError("Account already open")
        if initial_deposit < 0:
            raise ValueError("Initial deposit must be non-negative")
        self._raise("AccountOpened", {"owner": owner, "initial_deposit": initial_deposit},
                    metadata)

    def deposit(self, amount: float, metadata: Dict = None):
        if not self.is_open:
            raise AccountClosedError("Account is closed")
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._raise("MoneyDeposited", {"amount": amount}, metadata)

    def withdraw(self, amount: float, metadata: Dict = None):
        if not self.is_open:
            raise AccountClosedError("Account is closed")
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise InsufficientFundsError(
                f"Insufficient funds: balance={self.balance:.2f}, requested={amount:.2f}")
        self._raise("MoneyWithdrawn", {"amount": amount}, metadata)

    def apply_interest(self, rate: float, metadata: Dict = None):
        if not self.is_open:
            raise AccountClosedError("Account is closed")
        self._raise("InterestApplied", {"rate": rate}, metadata)

    def close(self, metadata: Dict = None):
        if not self.is_open:
            raise AccountClosedError("Account already closed")
        self._raise("AccountClosed", {"final_balance": self.balance}, metadata)

    def pop_uncommitted(self) -> List[Event]:
        evts = self._uncommitted[:]
        self._uncommitted.clear()
        return evts

    def to_snapshot_state(self) -> Dict:
        return {"balance": self.balance, "owner": self.owner, "is_open": self.is_open}

    @classmethod
    def from_snapshot(cls, account_id: str, snapshot: Snapshot) -> "BankAccountAggregate":
        agg          = cls(account_id)
        agg.balance  = snapshot.state["balance"]
        agg.owner    = snapshot.state["owner"]
        agg.is_open  = snapshot.state["is_open"]
        agg.version  = snapshot.version
        return agg


# ─────────────────────────────────────────────
# ACCOUNT REPOSITORY (loads + saves aggregates)
# ─────────────────────────────────────────────

class AccountRepository:
    def __init__(self, event_store: EventStore, snapshot_store: SnapshotStore):
        self.event_store    = event_store
        self.snapshot_store = snapshot_store

    def load(self, account_id: str) -> BankAccountAggregate:
        snapshot = self.snapshot_store.latest(account_id)
        if snapshot:
            agg    = BankAccountAggregate.from_snapshot(account_id, snapshot)
            events = self.event_store.load(account_id, from_version=snapshot.version)
        else:
            agg    = BankAccountAggregate(account_id)
            events = self.event_store.load(account_id)

        for evt in events:
            agg.apply(evt)
        return agg

    def save(self, agg: BankAccountAggregate):
        uncommitted = agg.pop_uncommitted()
        if not uncommitted:
            return
        expected_version = agg.version - len(uncommitted)
        self.event_store.append(agg.account_id, uncommitted, expected_version)

        # Snapshot if threshold reached
        if self.snapshot_store.should_snapshot(agg.version):
            self.snapshot_store.save(Snapshot(
                aggregate_id = agg.account_id,
                version      = agg.version,
                state        = agg.to_snapshot_state(),
            ))


# ─────────────────────────────────────────────
# PROJECTION: Transaction History
# ─────────────────────────────────────────────

class TransactionHistoryProjection:
    """Read model: full ledger of all transactions per account."""

    def __init__(self):
        self._ledger : Dict[str, List[Dict]] = defaultdict(list)
        self.events_applied = 0

    def apply(self, event: Event):
        p = event.payload
        if event.event_type == "AccountOpened":
            self._ledger[event.aggregate_id].append({
                "type": "open", "amount": p["initial_deposit"],
                "balance_after": p["initial_deposit"], "version": event.version,
            })
        elif event.event_type == "MoneyDeposited":
            self._update_ledger(event, "deposit", +p["amount"])
        elif event.event_type == "MoneyWithdrawn":
            self._update_ledger(event, "withdraw", -p["amount"])
        elif event.event_type == "InterestApplied":
            prev = self._last_balance(event.aggregate_id)
            interest = prev * p["rate"]
            self._update_ledger(event, "interest", +interest)
        self.events_applied += 1

    def _last_balance(self, account_id: str) -> float:
        ledger = self._ledger[account_id]
        return ledger[-1]["balance_after"] if ledger else 0.0

    def _update_ledger(self, event: Event, txn_type: str, delta: float):
        prev = self._last_balance(event.aggregate_id)
        self._ledger[event.aggregate_id].append({
            "type": txn_type, "amount": abs(delta),
            "balance_after": prev + delta, "version": event.version,
        })

    def get_history(self, account_id: str) -> List[Dict]:
        return self._ledger.get(account_id, [])


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_event_sourcing():
    print("=" * 65)
    print("EVENT SOURCING PATTERN")
    print("=" * 65)

    event_store    = EventStore()
    snapshot_store = SnapshotStore(snapshot_every=5)
    repo           = AccountRepository(event_store, snapshot_store)
    projection     = TransactionHistoryProjection()

    # ── Lifecycle of an account ───────────────────
    print("\n[1] ACCOUNT LIFECYCLE — COMMAND HANDLING")
    print("─" * 55)

    # Open account
    acc = repo.load("ACC-001")
    acc.open("Alice", initial_deposit=1000.0, metadata={"user": "alice"})
    repo.save(acc)

    # Deposits and withdrawals
    operations = [
        ("deposit",  200.0),
        ("withdraw", 150.0),
        ("deposit",  500.0),
        ("withdraw", 75.0),
        ("interest", 0.02),   # 2% interest
        ("withdraw", 300.0),
    ]
    for op, amount in operations:
        acc = repo.load("ACC-001")
        if op == "deposit"  : acc.deposit(amount)
        elif op == "withdraw": acc.withdraw(amount)
        elif op == "interest": acc.apply_interest(amount)
        repo.save(acc)

    acc = repo.load("ACC-001")
    print(f"  Final balance : ${acc.balance:.2f}")
    print(f"  Version       : {acc.version}")
    print(f"  Snapshot taken: {snapshot_store.latest('ACC-001') is not None}")

    # ── Event Log ────────────────────────────────
    print("\n\n[2] EVENT LOG — IMMUTABLE HISTORY")
    print("─" * 55)
    events = event_store.load("ACC-001")
    for evt in events:
        p = evt.payload
        detail = ""
        if "amount" in p:      detail = f"${p['amount']:.2f}"
        elif "rate" in p:      detail = f"{p['rate']*100:.0f}%"
        elif "initial_deposit" in p: detail = f"${p['initial_deposit']:.2f}"
        print(f"  v{evt.version:<2} {evt.event_type:<20} {detail}")

    # ── Time-Travel Query ─────────────────────────
    print("\n\n[3] TIME-TRAVEL — STATE AT HISTORICAL VERSION")
    print("─" * 55)
    for target_version in [1, 3, 5, len(events)]:
        replay_agg = BankAccountAggregate("ACC-001")
        for evt in event_store.load("ACC-001"):
            if evt.version <= target_version:
                replay_agg.apply(evt)
        print(f"  Balance at v{target_version}: ${replay_agg.balance:.2f}")

    # ── Projection ────────────────────────────────
    print("\n\n[4] TRANSACTION HISTORY PROJECTION")
    print("─" * 55)
    for evt in event_store.load("ACC-001"):
        projection.apply(evt)
    history = projection.get_history("ACC-001")
    print(f"  {'Txn':<10} {'Amount':>10} {'Balance After':>14} {'Version':>8}")
    print(f"  {'─'*46}")
    for row in history:
        print(f"  {row['type']:<10} ${row['amount']:>9.2f} ${row['balance_after']:>13.2f} "
              f"{'v'+str(row['version']):>8}")

    # ── Optimistic Concurrency ────────────────────
    print("\n\n[5] OPTIMISTIC CONCURRENCY — CONFLICT DETECTION")
    print("─" * 55)
    acc_v1  = repo.load("ACC-001")
    acc_v1b = repo.load("ACC-001")   # same version

    acc_v1.deposit(100.0)
    repo.save(acc_v1)   # first write succeeds
    print(f"  First writer deposited $100 — version now {acc_v1.version}")

    acc_v1b.deposit(50.0)
    try:
        repo.save(acc_v1b)   # second writer has stale version → conflict
        print(f"  Second writer succeeded (unexpected)")
    except OptimisticConcurrencyError as e:
        print(f"  Second writer REJECTED (optimistic lock): {e}")

    # ── Summary ───────────────────────────────────
    print("\n\n[6] EVENT SOURCING TRADE-OFFS")
    print("─" * 55)
    rows = [
        ("✓", "Audit log",          "Every change recorded — free compliance log"),
        ("✓", "Time-travel",        "Replay to any past version"),
        ("✓", "Projections",        "Rebuild any view from the same event stream"),
        ("✓", "Debugging",          "Replay failure scenario exactly"),
        ("✗", "Query current state","Need projection or replay (no direct SELECT)"),
        ("✗", "Schema evolution",   "Old events must be upcastable to new schema"),
        ("✗", "Storage growth",     "Append-only; compaction needed long-term"),
        ("✗", "Complexity",         "Snapshots, upcasting, projection sync add work"),
    ]
    for symbol, label, note in rows:
        print(f"  {symbol} {label:<20} {note}")


if __name__ == "__main__":
    demonstrate_event_sourcing()
