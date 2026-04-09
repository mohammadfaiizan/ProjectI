"""
WRITE-AHEAD LOG (WAL) DESIGN
================================

Problem Statement:
How does a database survive a crash mid-write without data corruption?
The Write-Ahead Log (WAL) ensures durability:
  "Log the change before applying it."
  On crash: replay the log to recover to a consistent state.

WAL Fundamentals:
  1. Before modifying any data page, write a log record (append-only).
  2. Log record: {LSN, transaction_id, operation, before_image, after_image}.
  3. fsync() the log before acknowledging the write to the client.
  4. Data pages can be written lazily (in background by checkpoint).
  5. On crash recovery: replay log from last checkpoint → redo committed,
     undo uncommitted transactions.

LSN (Log Sequence Number):
  Monotonically increasing identifier for each log record.
  Each page header stores: page_LSN = max LSN of applied change.
  Recovery: replay all log records where LSN > page_LSN.

ARIES Recovery Algorithm (standard in RDBMS):
  1. Analysis pass: find dirty pages and uncommitted txns at crash.
  2. Redo pass: replay all changes from oldest dirty page LSN.
  3. Undo pass: rollback uncommitted transactions (using CLR records).

CLR (Compensation Log Records):
  Written during undo to record rollback actions.
  Prevents re-undoing on repeated crashes.

Checkpoint:
  Flush all dirty pages to disk. Write checkpoint record to log.
  Recovery only needs to replay from last checkpoint → bounded recovery time.
  Fuzzy checkpoint: continue accepting writes during checkpoint (PostgreSQL WAL).

WAL Applications Beyond DB:
  Kafka: partition log is a WAL (append-only, offset-based).
  etcd: Raft log is a WAL.
  ZooKeeper: transaction log is a WAL.
  Redis AOF: append-only file = WAL for Redis.
  SQLite WAL mode: readers don't block writers.

PostgreSQL WAL:
  WAL segments: 16MB files in pg_wal/.
  WAL writer: background process flushing WAL buffer.
  Archive: WAL segments shipped to standby for replication.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time
import os
import json
import hashlib


# ─────────────────────────────────────────────
# LOG RECORD TYPES
# ─────────────────────────────────────────────

class LogRecordType(Enum):
    INSERT      = "INSERT"
    UPDATE      = "UPDATE"
    DELETE      = "DELETE"
    BEGIN       = "BEGIN"
    COMMIT      = "COMMIT"
    ABORT       = "ABORT"
    CHECKPOINT  = "CHECKPOINT"
    CLR         = "CLR"     # Compensation Log Record (undo record)


@dataclass
class LogRecord:
    lsn           : int
    txn_id        : str
    record_type   : LogRecordType
    table         : str
    key           : str
    before_image  : Optional[Any]    # value before change (for undo)
    after_image   : Optional[Any]    # value after change (for redo)
    prev_lsn      : Optional[int]    # previous LSN for this transaction (undo chain)
    timestamp     : float = field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        d = {
            "lsn": self.lsn, "txn_id": self.txn_id,
            "type": self.record_type.value,
            "table": self.table, "key": self.key,
            "before": self.before_image, "after": self.after_image,
            "prev_lsn": self.prev_lsn, "ts": self.timestamp,
        }
        return json.dumps(d).encode()

    @property
    def checksum(self) -> str:
        return hashlib.crc32(self.to_bytes()).to_bytes(4, "big").hex()


# ─────────────────────────────────────────────
# WAL (WRITE-AHEAD LOG)
# ─────────────────────────────────────────────

class WriteAheadLog:
    """
    Append-only WAL. Each write must be flushed before data page update.
    Supports: append, flush (fsync simulation), read_from_lsn.
    """

    SEGMENT_SIZE = 16 * 1024 * 1024   # 16 MB (PostgreSQL WAL segment)

    def __init__(self):
        self._records   : List[LogRecord] = []
        self._lsn       = 0
        self._flushed_to: int = -1    # last flushed LSN
        self.fsyncs     = 0
        self.appends    = 0

    def append(self, txn_id: str, record_type: LogRecordType,
               table: str = "", key: str = "",
               before: Any = None, after: Any = None,
               prev_lsn: Optional[int] = None) -> LogRecord:
        """Write log record. Returns record with assigned LSN."""
        self._lsn += 1
        record = LogRecord(
            lsn=self._lsn, txn_id=txn_id, record_type=record_type,
            table=table, key=key, before_image=before, after_image=after,
            prev_lsn=prev_lsn,
        )
        self._records.append(record)
        self.appends += 1
        return record

    def flush(self, up_to_lsn: int = None):
        """Simulate fsync: mark records as durable."""
        target = up_to_lsn if up_to_lsn else self._lsn
        self._flushed_to = max(self._flushed_to, target)
        self.fsyncs += 1

    def is_durable(self, lsn: int) -> bool:
        return lsn <= self._flushed_to

    def read_from(self, lsn: int) -> List[LogRecord]:
        """Read all records with LSN >= given value."""
        return [r for r in self._records if r.lsn >= lsn]

    def records_for_txn(self, txn_id: str) -> List[LogRecord]:
        return [r for r in self._records if r.txn_id == txn_id]

    def current_lsn(self) -> int:
        return self._lsn

    def stats(self) -> Dict:
        return {
            "current_lsn": self._lsn,
            "flushed_to" : self._flushed_to,
            "records"    : len(self._records),
            "fsyncs"     : self.fsyncs,
        }


# ─────────────────────────────────────────────
# DATA PAGE (simulated)
# ─────────────────────────────────────────────

@dataclass
class DataPage:
    table    : str
    page_lsn : int   # LSN of last change applied to this page
    data     : Dict[str, Any] = field(default_factory=dict)
    dirty    : bool = False


# ─────────────────────────────────────────────
# TRANSACTION + WAL-BACKED DATABASE
# ─────────────────────────────────────────────

class WALDatabase:
    """
    Simple key-value database backed by a WAL.
    All writes go to WAL first; pages updated in memory (simulating buffer pool).
    """

    def __init__(self):
        self._wal           = WriteAheadLog()
        self._pages         : Dict[str, DataPage] = {}
        self._txn_prev_lsn  : Dict[str, Optional[int]] = {}   # txn → last LSN
        self._active_txns   : Dict[str, bool] = {}
        self._checkpoint_lsn: int = 0

    def _get_page(self, table: str) -> DataPage:
        if table not in self._pages:
            self._pages[table] = DataPage(table=table, page_lsn=0)
        return self._pages[table]

    def begin(self, txn_id: str):
        rec = self._wal.append(txn_id, LogRecordType.BEGIN)
        self._wal.flush(rec.lsn)
        self._txn_prev_lsn[txn_id]  = rec.lsn
        self._active_txns[txn_id]   = True

    def write(self, txn_id: str, table: str, key: str, value: Any):
        """WAL-protected write: log before apply."""
        if not self._active_txns.get(txn_id):
            raise RuntimeError(f"Transaction {txn_id} not active")
        page   = self._get_page(table)
        before = page.data.get(key)
        # 1. Write to log FIRST
        rec = self._wal.append(
            txn_id, LogRecordType.UPDATE if before is not None else LogRecordType.INSERT,
            table=table, key=key, before=before, after=value,
            prev_lsn=self._txn_prev_lsn.get(txn_id),
        )
        self._txn_prev_lsn[txn_id] = rec.lsn
        # 2. Flush WAL (durability guarantee)
        self._wal.flush(rec.lsn)
        # 3. Apply to page in memory
        page.data[key]  = value
        page.page_lsn   = rec.lsn
        page.dirty      = True

    def delete(self, txn_id: str, table: str, key: str):
        page   = self._get_page(table)
        before = page.data.get(key)
        if before is None:
            return
        rec = self._wal.append(txn_id, LogRecordType.DELETE,
                                table=table, key=key, before=before, after=None,
                                prev_lsn=self._txn_prev_lsn.get(txn_id))
        self._txn_prev_lsn[txn_id] = rec.lsn
        self._wal.flush(rec.lsn)
        del page.data[key]
        page.page_lsn = rec.lsn
        page.dirty    = True

    def commit(self, txn_id: str):
        rec = self._wal.append(txn_id, LogRecordType.COMMIT,
                                prev_lsn=self._txn_prev_lsn.get(txn_id))
        self._wal.flush(rec.lsn)
        self._active_txns.pop(txn_id, None)
        self._txn_prev_lsn.pop(txn_id, None)

    def abort(self, txn_id: str):
        """Undo uncommitted transaction using before-images from WAL."""
        rec = self._wal.append(txn_id, LogRecordType.ABORT,
                                prev_lsn=self._txn_prev_lsn.get(txn_id))
        self._wal.flush(rec.lsn)
        # Undo all changes in reverse order
        txn_records = self._wal.records_for_txn(txn_id)
        for r in reversed(txn_records):
            if r.record_type in (LogRecordType.INSERT, LogRecordType.UPDATE,
                                 LogRecordType.DELETE):
                page = self._get_page(r.table)
                if r.record_type == LogRecordType.INSERT:
                    page.data.pop(r.key, None)
                else:
                    page.data[r.key] = r.before_image
                page.page_lsn = rec.lsn
                # Write CLR
                self._wal.append(txn_id, LogRecordType.CLR, table=r.table,
                                  key=r.key, before=r.after_image,
                                  after=r.before_image, prev_lsn=r.lsn)
        self._active_txns.pop(txn_id, None)
        self._txn_prev_lsn.pop(txn_id, None)

    def read(self, table: str, key: str) -> Optional[Any]:
        return self._get_page(table).data.get(key)

    def checkpoint(self):
        """Flush dirty pages + write checkpoint LSN to log."""
        for page in self._pages.values():
            if page.dirty:
                page.dirty = False
        rec = self._wal.append("system", LogRecordType.CHECKPOINT)
        self._wal.flush(rec.lsn)
        self._checkpoint_lsn = rec.lsn

    def crash_recovery(self, crash_lsn: int = 0):
        """ARIES-style: replay log from last checkpoint."""
        start_lsn   = max(self._checkpoint_lsn, crash_lsn)
        replay_recs = self._wal.read_from(start_lsn)
        committed   : set = set()
        aborted     : set = set()
        for r in replay_recs:
            if r.record_type == LogRecordType.COMMIT: committed.add(r.txn_id)
            if r.record_type == LogRecordType.ABORT:  aborted.add(r.txn_id)

        redo_count = undo_count = 0
        for r in replay_recs:
            if r.record_type in (LogRecordType.INSERT, LogRecordType.UPDATE,
                                 LogRecordType.DELETE):
                page = self._get_page(r.table)
                if r.txn_id in committed and r.lsn > page.page_lsn:
                    page.data[r.key] = r.after_image
                    page.page_lsn = r.lsn
                    redo_count += 1
                elif r.txn_id not in committed and r.txn_id not in aborted:
                    # Uncommitted at crash → undo
                    if r.before_image is not None:
                        page.data[r.key] = r.before_image
                    else:
                        page.data.pop(r.key, None)
                    undo_count += 1
        return {"redo": redo_count, "undo": undo_count}

    def wal_stats(self) -> Dict:
        return self._wal.stats()


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_wal():
    print("=" * 65)
    print("WRITE-AHEAD LOG (WAL) DESIGN")
    print("=" * 65)

    db = WALDatabase()

    # ── Normal Transactions ───────────────────────
    print("\n[1] WAL-BACKED TRANSACTIONS")
    print("─" * 55)

    db.begin("txn-1")
    db.write("txn-1", "users", "u1", {"name": "Alice", "age": 30})
    db.write("txn-1", "users", "u2", {"name": "Bob",   "age": 25})
    db.commit("txn-1")

    db.begin("txn-2")
    db.write("txn-2", "orders", "o1", {"user": "u1", "amount": 99.0})
    db.commit("txn-2")

    print(f"  After 2 committed txns:")
    print(f"    users.u1 = {db.read('users', 'u1')}")
    print(f"    orders.o1 = {db.read('orders', 'o1')}")

    s = db.wal_stats()
    print(f"  WAL: {s['records']} records, {s['fsyncs']} fsyncs, LSN={s['current_lsn']}")

    # ── WAL Records ───────────────────────────────
    print("\n\n[2] WAL RECORD STRUCTURE")
    print("─" * 55)

    for rec in db._wal.records_for_txn("txn-1"):
        print(f"  LSN={rec.lsn:3} txn={rec.txn_id} type={rec.record_type.value:<8} "
              f"table={rec.table:<8} key={rec.key}")

    # ── Transaction Abort (Undo) ──────────────────
    print("\n\n[3] TRANSACTION ABORT — ROLLBACK VIA WAL")
    print("─" * 55)

    db.begin("txn-3")
    db.write("txn-3", "users", "u3", {"name": "Carol", "age": 22})
    db.write("txn-3", "users", "u1", {"name": "Alice MODIFIED", "age": 31})

    print(f"  Before abort: users.u1 = {db.read('users', 'u1')}")
    print(f"                users.u3 = {db.read('users', 'u3')}")

    db.abort("txn-3")
    print(f"  After abort:  users.u1 = {db.read('users', 'u1')}  (restored)")
    print(f"                users.u3 = {db.read('users', 'u3')}  (removed)")

    # ── Checkpoint ────────────────────────────────
    print("\n\n[4] CHECKPOINT — BOUNDING RECOVERY TIME")
    print("─" * 55)

    lsn_before_ckpt = db.wal_stats()["current_lsn"]
    db.checkpoint()
    lsn_after_ckpt  = db.wal_stats()["current_lsn"]
    print(f"  Checkpoint written at LSN {lsn_after_ckpt}")
    print(f"  Recovery will replay from LSN {db._checkpoint_lsn} (not from 0)")
    print(f"  Records to replay after crash: "
          f"{len(db._wal.read_from(db._checkpoint_lsn))}")

    # ── Crash Recovery ────────────────────────────
    print("\n\n[5] CRASH RECOVERY — REDO + UNDO")
    print("─" * 55)

    # Simulate uncommitted txn at crash time
    db.begin("txn-crash")
    db.write("txn-crash", "users", "u99", {"name": "Zombie"})
    # Simulate crash: txn-crash never committed

    stats = db.crash_recovery()
    print(f"  Recovery complete: redo={stats['redo']} operations, "
          f"undo={stats['undo']} operations")
    print(f"  Uncommitted txn-crash record removed: "
          f"{db.read('users', 'u99') is None}")

    # ── WAL Design Decisions ──────────────────────
    print("\n\n[6] WAL DESIGN DECISIONS")
    print("─" * 55)

    decisions = [
        ("Log before write",    "Durability: crash can't lose committed data"),
        ("fsync on commit",     "Data survives power failure; most expensive step"),
        ("Before + after image","Enables both redo (forward) and undo (rollback)"),
        ("LSN per page",        "Recovery: skip pages already up-to-date"),
        ("Checkpoint",          "Bound recovery time: replay from last checkpoint"),
        ("CLR records",         "Idempotent undo: safe to re-run recovery multiple times"),
        ("Append-only log",     "Sequential writes: 100x faster than random writes"),
        ("WAL shipping",        "Stream WAL to standby for replication (PostgreSQL)"),
    ]
    for decision, reason in decisions:
        print(f"  {decision:<26} {reason}")


if __name__ == "__main__":
    demonstrate_wal()
