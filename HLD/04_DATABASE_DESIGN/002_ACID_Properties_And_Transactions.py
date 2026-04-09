"""
ACID PROPERTIES AND TRANSACTIONS
===================================

Problem Statement:
Database transactions must behave correctly even when concurrent users
modify the same data or the system crashes mid-operation. ACID properties
define the guarantees a database transaction system must provide.

ACID Properties:
  A — Atomicity    : All operations in a transaction succeed or all fail
                     (no partial writes). "All or nothing."
  C — Consistency  : Data always moves from one valid state to another.
                     Constraints, cascades, triggers always hold.
  I — Isolation    : Concurrent transactions don't interfere with each other.
                     Each transaction sees a consistent snapshot.
  D — Durability   : Once committed, data survives crashes.
                     Written to WAL (Write-Ahead Log) before returning.

Isolation Levels (ANSI SQL, weakest to strongest):
  READ UNCOMMITTED : Can read uncommitted (dirty) data — fastest, dangerous
  READ COMMITTED   : Only see committed rows — default in PostgreSQL
  REPEATABLE READ  : Rows don't change during transaction — MySQL default
  SERIALIZABLE     : Transactions appear to run one at a time — safest, slowest

Concurrency Problems:
  Dirty Read     : Read uncommitted data from another transaction
  Non-repeatable: Same row read twice returns different values
  Phantom Read   : New rows appear in a repeated range query
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import time
import uuid
import threading


class IsolationLevel(Enum):
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED   = "read_committed"
    REPEATABLE_READ  = "repeatable_read"
    SERIALIZABLE     = "serializable"


class TxnState(Enum):
    ACTIVE    = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass
class WALEntry:
    """Write-Ahead Log entry — durability mechanism."""
    txn_id     : str
    operation  : str   # INSERT/UPDATE/DELETE
    table      : str
    key        : str
    old_value  : Any
    new_value  : Any
    timestamp  : float = field(default_factory=time.time)
    committed  : bool  = False


# ─────────────────────────────────────────────
# WRITE-AHEAD LOG
# ─────────────────────────────────────────────

class WriteAheadLog:
    """
    WAL ensures durability.
    Changes are logged before being applied to the actual data.
    On crash recovery, replay WAL to restore committed transactions.
    """

    def __init__(self):
        self._entries : List[WALEntry] = []
        self._lock    = threading.Lock()

    def append(self, entry: WALEntry):
        with self._lock:
            self._entries.append(entry)

    def mark_committed(self, txn_id: str):
        with self._lock:
            for e in self._entries:
                if e.txn_id == txn_id:
                    e.committed = True

    def replay(self) -> List[WALEntry]:
        """Return committed entries for crash recovery."""
        return [e for e in self._entries if e.committed]

    def show(self, last_n: int = 10):
        print(f"\n  WAL entries (last {last_n}):")
        for e in self._entries[-last_n:]:
            status = "✅ committed" if e.committed else "⏳ uncommitted"
            print(f"    [{e.txn_id[:8]}] {e.operation} {e.table}.{e.key}  "
                  f"{e.old_value} → {e.new_value}  {status}")


# ─────────────────────────────────────────────
# TRANSACTION
# ─────────────────────────────────────────────

class Transaction:
    def __init__(self, txn_id: str, isolation: IsolationLevel,
                 db: "TransactionalDB"):
        self.txn_id       = txn_id
        self.isolation    = isolation
        self._db          = db
        self.state        = TxnState.ACTIVE
        self._writeSet    : Dict[str, Any] = {}   # key → value (pending writes)
        self._readSet     : Dict[str, Any] = {}   # key → value read (for conflict detection)
        self.start_time   = time.time()

    def read(self, table: str, key: str) -> Optional[Any]:
        if self.state != TxnState.ACTIVE:
            raise RuntimeError(f"Transaction {self.txn_id} is not active")
        full_key = f"{table}:{key}"
        # Check own write buffer first
        if full_key in self._writeSet:
            return self._writeSet[full_key]
        val = self._db._read_with_isolation(full_key, self.txn_id, self.isolation)
        self._readSet[full_key] = val
        return val

    def write(self, table: str, key: str, value: Any):
        if self.state != TxnState.ACTIVE:
            raise RuntimeError(f"Transaction {self.txn_id} is not active")
        full_key = f"{table}:{key}"
        self._writeSet[full_key] = value

    def commit(self):
        if self.state != TxnState.ACTIVE:
            raise RuntimeError("Cannot commit: not active")
        self._db._commit(self)
        self.state = TxnState.COMMITTED

    def rollback(self):
        if self.state != TxnState.ACTIVE:
            raise RuntimeError("Cannot rollback: not active")
        self._db._rollback(self)
        self.state = TxnState.ROLLED_BACK


# ─────────────────────────────────────────────
# TRANSACTIONAL DATABASE
# ─────────────────────────────────────────────

class TransactionalDB:
    """
    Simplified ACID database simulation.
    Demonstrates commit/rollback, WAL, and isolation.
    """

    def __init__(self, isolation: IsolationLevel = IsolationLevel.READ_COMMITTED):
        self.default_isolation = isolation
        self._data             : Dict[str, Any] = {}
        self._committed_snapshots: Dict[str, Dict] = {}   # txn_id → snapshot at commit
        self.wal               = WriteAheadLog()
        self._active_txns      : Dict[str, Transaction] = {}
        self._committed_data   : Dict[str, Any] = {}   # last committed values
        self._lock             = threading.Lock()
        self.committed_txns    = 0
        self.rolled_back_txns  = 0

    def begin(self, isolation: IsolationLevel = None) -> Transaction:
        txn_id = str(uuid.uuid4())[:8]
        txn    = Transaction(txn_id, isolation or self.default_isolation, self)
        with self._lock:
            self._active_txns[txn_id] = txn
        return txn

    def _read_with_isolation(self, full_key: str, txn_id: str,
                              isolation: IsolationLevel) -> Optional[Any]:
        with self._lock:
            if isolation == IsolationLevel.READ_UNCOMMITTED:
                # Can see other active txns' writes
                for tid, txn in self._active_txns.items():
                    if tid != txn_id and full_key in txn._writeSet:
                        return txn._writeSet[full_key]   # dirty read!
            # READ_COMMITTED and above: only see committed data
            return self._committed_data.get(full_key, self._data.get(full_key))

    def _commit(self, txn: Transaction):
        with self._lock:
            # Log to WAL first (durability)
            for key, new_val in txn._writeSet.items():
                old_val = self._committed_data.get(key)
                entry   = WALEntry(txn.txn_id, "WRITE", key.split(":")[0],
                                    key.split(":")[1], old_val, new_val)
                self.wal.append(entry)

            # Apply writes to committed store (atomically)
            for key, val in txn._writeSet.items():
                self._committed_data[key] = val
                self._data[key]           = val

            # Mark WAL entries as committed
            self.wal.mark_committed(txn.txn_id)

            del self._active_txns[txn.txn_id]
            self.committed_txns += 1

    def _rollback(self, txn: Transaction):
        with self._lock:
            # Discard all writes — nothing applied
            del self._active_txns[txn.txn_id]
            self.rolled_back_txns += 1

    def set_initial(self, table: str, key: str, value: Any):
        self._data[f"{table}:{key}"]           = value
        self._committed_data[f"{table}:{key}"] = value


# ─────────────────────────────────────────────
# ISOLATION LEVEL PROBLEMS
# ─────────────────────────────────────────────

class IsolationProblemDemo:
    """Shows classic isolation anomalies."""

    @staticmethod
    def dirty_read_demo(db: TransactionalDB):
        """T1 reads T2's uncommitted write."""
        print("\n  Dirty Read (READ UNCOMMITTED):")
        # Setup
        db.set_initial("accounts", "alice", 1000)

        t1 = db.begin(IsolationLevel.READ_UNCOMMITTED)
        t2 = db.begin(IsolationLevel.READ_UNCOMMITTED)

        print(f"    T2 writes alice=1500 (not yet committed)")
        t2.write("accounts", "alice", 1500)

        val = t1.read("accounts", "alice")
        print(f"    T1 reads alice: {val}  (dirty read — sees T2's uncommitted write)")

        print(f"    T2 ROLLBACK")
        t2.rollback()
        val2 = t1.read("accounts", "alice")
        print(f"    T1 reads alice again: {val2}  (data was never real)")
        t1.rollback()

    @staticmethod
    def atomicity_demo(db: TransactionalDB):
        """Transfer money — must be atomic."""
        print("\n  Atomicity — Bank Transfer:")
        db.set_initial("accounts", "alice_bal", 1000)
        db.set_initial("accounts", "bob_bal",   500)

        print(f"    Before: alice=1000, bob=500")
        txn = db.begin(IsolationLevel.SERIALIZABLE)
        alice_bal = txn.read("accounts", "alice_bal")
        bob_bal   = txn.read("accounts", "bob_bal")

        amount = 200
        if alice_bal >= amount:
            txn.write("accounts", "alice_bal", alice_bal - amount)
            txn.write("accounts", "bob_bal",   bob_bal + amount)
            txn.commit()
            print(f"    Transferred ${amount}. COMMIT.")
        else:
            txn.rollback()
            print(f"    Insufficient funds. ROLLBACK.")

        alice_new = db._committed_data.get("accounts:alice_bal")
        bob_new   = db._committed_data.get("accounts:bob_bal")
        print(f"    After : alice={alice_new}, bob={bob_new}")
        print(f"    Sum stays: {alice_new + bob_new} (was 1500)")

    @staticmethod
    def rollback_demo(db: TransactionalDB):
        """Show rollback undoes all changes."""
        print("\n  Rollback — Partial Update Undone:")
        db.set_initial("inventory", "item_99", 10)
        print(f"    Initial stock: item_99 = 10")

        txn = db.begin()
        txn.write("inventory", "item_99", 5)
        print(f"    Wrote item_99=5 (in progress, not committed)")

        # Read from another txn — READ_COMMITTED sees original
        txn2 = db.begin(IsolationLevel.READ_COMMITTED)
        val  = txn2.read("inventory", "item_99")
        print(f"    Other txn reads item_99: {val}  (still sees 10 — READ COMMITTED)")
        txn2.rollback()

        txn.rollback()
        print(f"    Rollback! item_99 = {db._committed_data.get('inventory:item_99')}")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_acid():
    print("=" * 65)
    print("ACID PROPERTIES AND TRANSACTIONS")
    print("=" * 65)

    db = TransactionalDB(IsolationLevel.READ_COMMITTED)
    demo = IsolationProblemDemo()

    # ── Atomicity ─────────────────────────────
    print("\n[1] ATOMICITY — BANK TRANSFER")
    print("─" * 55)
    demo.atomicity_demo(db)

    # ── Dirty Read ────────────────────────────
    print("\n\n[2] ISOLATION — DIRTY READ ANOMALY")
    print("─" * 55)
    demo.dirty_read_demo(db)

    # ── Rollback ──────────────────────────────
    print("\n\n[3] ROLLBACK — PARTIAL WRITES UNDONE")
    print("─" * 55)
    demo.rollback_demo(db)

    # ── WAL ───────────────────────────────────
    print("\n\n[4] WRITE-AHEAD LOG (DURABILITY)")
    print("─" * 55)
    db.wal.show()

    # ── Isolation Levels ──────────────────────
    print("\n\n[5] ISOLATION LEVELS COMPARISON")
    print("─" * 55)
    rows = [
        ("READ UNCOMMITTED", "❌ Possible", "❌ Possible", "❌ Possible", "Highest",  "Almost never"),
        ("READ COMMITTED",   "✅ Prevented","❌ Possible", "❌ Possible", "High",     "PostgreSQL default"),
        ("REPEATABLE READ",  "✅ Prevented","✅ Prevented","❌ Possible", "Medium",   "MySQL default"),
        ("SERIALIZABLE",     "✅ Prevented","✅ Prevented","✅ Prevented","Lowest",   "Financial systems"),
    ]
    print(f"  {'Level':<22} {'Dirty Rd':<15} {'Non-Repeat':<15} {'Phantom':<12} {'Throughput':<12} {'Use case'}")
    print(f"  {'─'*90}")
    for row in rows:
        print(f"  {row[0]:<22} {row[1]:<15} {row[2]:<15} {row[3]:<12} {row[4]:<12} {row[5]}")

    print(f"\n  DB stats: committed={db.committed_txns}  rolled_back={db.rolled_back_txns}")

    # ── ACID in Distributed Systems ───────────
    print("\n\n[6] ACID IN DISTRIBUTED SYSTEMS")
    print("─" * 55)
    print("  Single DB ACID: straightforward — one lock manager, one WAL")
    print("  Distributed ACID: much harder — 2PC (two-phase commit) required")
    print()
    twophase = [
        ("Phase 1 Prepare:", "Coordinator asks all nodes: can you commit?"),
        ("All vote Yes:",     "Each node writes to WAL, locks resources"),
        ("Phase 2 Commit:",   "Coordinator sends COMMIT to all nodes"),
        ("Any vote No:",      "Coordinator sends ROLLBACK to all nodes"),
        ("Failure risk:",     "If coordinator crashes after Phase 1 → blocked"),
        ("Alternatives:",     "Saga pattern, eventual consistency, CRDT"),
    ]
    for step, detail in twophase:
        print(f"  {step:<22} {detail}")


if __name__ == "__main__":
    demonstrate_acid()
