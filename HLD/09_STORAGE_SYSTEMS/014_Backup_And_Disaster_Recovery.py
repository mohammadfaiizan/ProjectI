"""
BACKUP AND DISASTER RECOVERY
================================

Problem Statement:
Data loss from hardware failure, ransomware, human error, or datacenter
outage can be catastrophic. Backup and DR strategies protect against loss.

Key Metrics:
  RPO (Recovery Point Objective): Maximum acceptable data loss.
               "We can afford to lose at most 1 hour of transactions."
  RTO (Recovery Time Objective): Maximum acceptable downtime.
               "System must be operational again within 4 hours."
  MTTR (Mean Time To Recovery): Average time to recover from failure.

Backup Types:
  Full backup:        Copy of all data. Simplest to restore. High storage/time.
  Incremental backup: Only changes since last backup (any type). Fast. Complex restore.
  Differential backup:Changes since last FULL backup. Moderate size. Simpler restore than incr.
  Continuous:         Change Data Capture (CDC) or WAL shipping. Near-zero RPO.

Backup 3-2-1 Rule:
  3 copies of data.
  2 different media types (SSD + tape, or local + cloud).
  1 copy offsite.
  Modern: 3-2-1-1-0 (+ 1 air-gapped copy + 0 errors verified).

Disaster Recovery Strategies (in order of RTO/cost):
  Backup & Restore:  Cheapest. Highest RTO (hours/days). RPO = last backup.
  Pilot Light:       Core services running (DB replication). Servers off.
                     RTO: minutes-hours. Cost: minimal idle infrastructure.
  Warm Standby:      Reduced-capacity copy always running.
                     RTO: minutes. Scale up on failover.
  Multi-Site Active-Active: Full capacity in multiple regions.
                     RTO: seconds. Highest cost.

Point-in-Time Recovery (PITR):
  Full backup + WAL/binlog streaming.
  Can restore to any second in the past.
  PostgreSQL: base backup + WAL archive.
  MySQL: binary log + mysqldump.

Testing Backups:
  "A backup that hasn't been restored is not a backup."
  Regular restore tests. Automated verification.
  Chaos: intentionally destroy prod data, restore from backup, measure RTO.

Cloud DR:
  AWS: S3 versioning + replication + S3 Glacier (RPO: seconds, RTO: hours).
  RDS Multi-AZ: synchronous standby (RTO: 2 min, RPO: 0).
  Aurora Global: cross-region replication (RTO: <1 min, RPO: <1 sec).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import time
import random


# ─────────────────────────────────────────────
# BACKUP TYPES
# ─────────────────────────────────────────────

class BackupType(Enum):
    FULL         = "FULL"
    INCREMENTAL  = "INCREMENTAL"
    DIFFERENTIAL = "DIFFERENTIAL"
    CONTINUOUS   = "CONTINUOUS"    # WAL/binlog stream


@dataclass
class BackupSnapshot:
    backup_id   : str
    backup_type : BackupType
    timestamp   : float
    base_backup  : Optional[str]   # ID of full backup this builds on
    data        : Dict[str, Any]   # {key: value} snapshot
    changed_keys: set              # keys changed since last backup
    size_bytes  : int
    duration_s  : float
    checksum    : str

    @classmethod
    def create(cls, backup_id: str, backup_type: BackupType,
               data: Dict, changed_keys: set, base_id: str = None,
               duration_s: float = 0.0) -> "BackupSnapshot":
        checksum = hashlib.md5(str(sorted(data.items())).encode()).hexdigest()[:8]
        return cls(
            backup_id=backup_id, backup_type=backup_type,
            timestamp=time.time(), base_backup=base_id,
            data=dict(data), changed_keys=set(changed_keys),
            size_bytes=len(str(data).encode()), duration_s=duration_s,
            checksum=checksum,
        )


# ─────────────────────────────────────────────
# BACKUP MANAGER
# ─────────────────────────────────────────────

class BackupManager:
    """
    Manages full, incremental, differential, and continuous backups.
    """

    def __init__(self):
        self._backups       : List[BackupSnapshot]  = []
        self._wal           : List[Tuple[float, str, Any]] = []  # (ts, key, val) WAL entries
        self._last_full_id  : Optional[str] = None
        self._backup_counter = 0

    def _next_id(self, prefix: str) -> str:
        self._backup_counter += 1
        return f"{prefix}-{self._backup_counter:04d}"

    def full_backup(self, data: Dict) -> BackupSnapshot:
        t0  = time.time()
        bid = self._next_id("FULL")
        snap = BackupSnapshot.create(bid, BackupType.FULL, data,
                                      set(data.keys()), duration_s=time.time()-t0)
        self._backups.append(snap)
        self._last_full_id = bid
        return snap

    def incremental_backup(self, data: Dict, prev_snapshot_id: str) -> BackupSnapshot:
        """Only backs up keys changed since the previous snapshot (of any type)."""
        prev = self._get_backup(prev_snapshot_id)
        if not prev:
            raise ValueError(f"Previous snapshot {prev_snapshot_id} not found")
        changed = {k: v for k, v in data.items()
                   if k not in prev.data or prev.data[k] != v}
        bid  = self._next_id("INCR")
        snap = BackupSnapshot.create(bid, BackupType.INCREMENTAL,
                                      changed, set(changed.keys()),
                                      base_id=prev_snapshot_id)
        self._backups.append(snap)
        return snap

    def differential_backup(self, data: Dict) -> BackupSnapshot:
        """Changes since last FULL backup."""
        if not self._last_full_id:
            return self.full_backup(data)
        full_snap = self._get_backup(self._last_full_id)
        changed   = {k: v for k, v in data.items()
                     if k not in full_snap.data or full_snap.data[k] != v}
        bid  = self._next_id("DIFF")
        snap = BackupSnapshot.create(bid, BackupType.DIFFERENTIAL,
                                      changed, set(changed.keys()),
                                      base_id=self._last_full_id)
        self._backups.append(snap)
        return snap

    def record_wal(self, key: str, value: Any):
        """Record change to WAL for PITR."""
        self._wal.append((time.time(), key, value))

    def restore_full(self, backup_id: str) -> Dict:
        """Restore from a full backup."""
        snap = self._get_backup(backup_id)
        if not snap or snap.backup_type != BackupType.FULL:
            raise ValueError("Not a full backup")
        return dict(snap.data)

    def restore_incremental_chain(self, final_backup_id: str) -> Dict:
        """Restore by replaying incremental chain from full backup."""
        chain = self._build_restore_chain(final_backup_id)
        result = {}
        for snap in chain:
            result.update(snap.data)
        return result

    def restore_to_point_in_time(self, base_backup_id: str,
                                   target_ts: float) -> Dict:
        """PITR: full backup + replay WAL entries up to target timestamp."""
        base = self._get_backup(base_backup_id)
        if not base:
            raise ValueError("Base backup not found")
        result = dict(base.data)
        # Replay WAL entries after backup and before target_ts
        for ts, key, value in self._wal:
            if base.timestamp <= ts <= target_ts:
                result[key] = value
        return result

    def _build_restore_chain(self, backup_id: str) -> List[BackupSnapshot]:
        chain = []
        bid   = backup_id
        while bid:
            snap = self._get_backup(bid)
            if not snap:
                break
            chain.append(snap)
            if snap.backup_type == BackupType.FULL:
                break
            bid = snap.base_backup
        return list(reversed(chain))

    def _get_backup(self, backup_id: str) -> Optional[BackupSnapshot]:
        return next((b for b in self._backups if b.backup_id == backup_id), None)

    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity via checksum."""
        snap = self._get_backup(backup_id)
        if not snap:
            return False
        expected = hashlib.md5(str(sorted(snap.data.items())).encode()).hexdigest()[:8]
        return expected == snap.checksum

    def backup_catalog(self) -> List[Dict]:
        return [
            {"id": b.backup_id, "type": b.backup_type.value,
             "size_bytes": b.size_bytes, "keys": len(b.data),
             "base": b.base_backup, "checksum": b.checksum}
            for b in self._backups
        ]


# ─────────────────────────────────────────────
# DR STRATEGIES
# ─────────────────────────────────────────────

class DRTier(Enum):
    BACKUP_RESTORE  = "Backup & Restore"
    PILOT_LIGHT     = "Pilot Light"
    WARM_STANDBY    = "Warm Standby"
    ACTIVE_ACTIVE   = "Multi-Site Active-Active"


@dataclass
class DRStrategy:
    tier            : DRTier
    rto_minutes     : float   # Recovery Time Objective
    rpo_seconds     : float   # Recovery Point Objective
    cost_multiplier : float   # relative cost vs single-site
    description     : str


DR_STRATEGIES = [
    DRStrategy(DRTier.BACKUP_RESTORE,  240, 3600,  1.1,
               "Restore from S3 backup. Cheapest. Long RTO."),
    DRStrategy(DRTier.PILOT_LIGHT,     30,  300,   1.3,
               "DB replicated. Servers off. Start on failover."),
    DRStrategy(DRTier.WARM_STANDBY,    5,   60,    1.8,
               "Reduced capacity always running. Scale up on failover."),
    DRStrategy(DRTier.ACTIVE_ACTIVE,   0.1, 1,     3.0,
               "Full capacity in 2+ regions. Traffic routed instantly."),
]


# ─────────────────────────────────────────────
# RPO/RTO CALCULATOR
# ─────────────────────────────────────────────

def calculate_rpo_rto(strategy: DRStrategy, incident_time: float,
                      data_rate_per_hour: int) -> Dict:
    """Estimate data loss and downtime cost."""
    rpo_hours  = strategy.rpo_seconds / 3600
    data_lost  = int(data_rate_per_hour * rpo_hours)
    return {
        "strategy"     : strategy.tier.value,
        "rpo_seconds"  : strategy.rpo_seconds,
        "rto_minutes"  : strategy.rto_minutes,
        "data_records_lost": data_lost,
        "downtime_cost_usd": strategy.rto_minutes * (100_000 / 60),  # $100k/hour
        "cost_multiplier"  : strategy.cost_multiplier,
    }


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_backup_dr():
    print("=" * 65)
    print("BACKUP AND DISASTER RECOVERY")
    print("=" * 65)

    bm = BackupManager()
    random.seed(42)

    # ── Initial data ──────────────────────────────
    database = {f"user:{i}": {"name": f"User{i}", "balance": i * 10}
                for i in range(20)}

    # ── Full Backup ───────────────────────────────
    print("\n[1] BACKUP TYPES — FULL / INCREMENTAL / DIFFERENTIAL")
    print("─" * 55)

    full1 = bm.full_backup(database)
    print(f"  Full backup: {full1.backup_id} — {full1.size_bytes}B, "
          f"{len(full1.data)} keys, checksum={full1.checksum}")

    # Apply changes
    database["user:5"]["balance"] += 100
    database["user:12"]["balance"] += 50
    database["user:21"] = {"name": "User21", "balance": 210}
    bm.record_wal("user:5",  database["user:5"])
    bm.record_wal("user:12", database["user:12"])
    bm.record_wal("user:21", database["user:21"])

    # Incremental 1
    incr1 = bm.incremental_backup(database, full1.backup_id)
    print(f"  Incremental: {incr1.backup_id} — {incr1.size_bytes}B, "
          f"{len(incr1.data)} changed keys (full={len(full1.data)} keys)")

    # More changes
    database["user:3"]["balance"] += 200
    bm.record_wal("user:3", database["user:3"])

    # Differential (changes since last full)
    diff1 = bm.differential_backup(database)
    print(f"  Differential: {diff1.backup_id} — {diff1.size_bytes}B, "
          f"{len(diff1.data)} keys changed since full")

    # Another incremental
    incr2 = bm.incremental_backup(database, incr1.backup_id)
    print(f"  Incremental2: {incr2.backup_id} — {incr2.size_bytes}B, "
          f"{len(incr2.data)} changed keys")

    # ── Restore ───────────────────────────────────
    print("\n\n[2] RESTORE — FULL / INCREMENTAL CHAIN")
    print("─" * 55)

    # Restore full
    restored_full = bm.restore_full(full1.backup_id)
    print(f"  Restore from full ({full1.backup_id}): {len(restored_full)} keys")
    print(f"    user:5.balance = {restored_full['user:5']['balance']} "
          f"(original: 50, current: {database['user:5']['balance']})")

    # Restore via incremental chain: full → incr1 → incr2
    restored_chain = bm.restore_incremental_chain(incr2.backup_id)
    print(f"  Restore via chain (full→incr1→incr2): {len(restored_chain)} keys")
    print(f"    user:5.balance = {restored_chain.get('user:5', {}).get('balance')}")
    print(f"    user:21 exists: {'user:21' in restored_chain}")

    # ── PITR ──────────────────────────────────────
    print("\n\n[3] POINT-IN-TIME RECOVERY (PITR)")
    print("─" * 55)

    # Target: state after incr1 but before user:3 change
    target_ts = time.time() + 0.001  # just now (all WAL entries included)
    pitr_data = bm.restore_to_point_in_time(full1.backup_id, target_ts)
    print(f"  PITR to current state: {len(pitr_data)} keys")
    print(f"    user:3.balance = {pitr_data.get('user:3', {}).get('balance')} "
          f"(expected: {database['user:3']['balance']})")

    # ── Backup Verification ───────────────────────
    print("\n\n[4] BACKUP VERIFICATION")
    print("─" * 55)

    for snap in [full1, incr1, diff1, incr2]:
        valid = bm.verify_backup(snap.backup_id)
        print(f"  {snap.backup_id}: checksum={snap.checksum} → {'VALID' if valid else 'CORRUPT'}")

    # ── DR Strategy Comparison ────────────────────
    print("\n\n[5] DR STRATEGY COMPARISON")
    print("─" * 55)

    print(f"  {'Strategy':<28} {'RTO':>8} {'RPO':>10} {'Records lost':>14} {'Cost mult'}")
    print(f"  {'─'*74}")
    for strategy in DR_STRATEGIES:
        calc = calculate_rpo_rto(strategy, time.time(), 10_000)
        print(f"  {calc['strategy']:<28} {calc['rto_minutes']:>5.0f}min "
              f"{calc['rpo_seconds']:>8.0f}s  {calc['data_records_lost']:>12,}  "
              f"{calc['cost_multiplier']:>6.1f}x")

    # ── 3-2-1 Rule ────────────────────────────────
    print("\n\n[6] 3-2-1 BACKUP RULE")
    print("─" * 55)

    backup_321 = [
        ("Copy 1", "Primary disk",          "Production NVMe SSD",     "Local"),
        ("Copy 2", "Secondary local",       "On-prem NAS / HDD",       "Local"),
        ("Copy 3", "Offsite / cloud",       "AWS S3 + Glacier",        "Offsite"),
        ("Air gap","Isolated network",      "Tape / offline storage",  "Air-gapped"),
    ]
    for label, medium, example, location in backup_321:
        print(f"  {label:<10} {medium:<22} {example:<28} {location}")

    print("\n  Modern 3-2-1-1-0 rule:")
    print("  + 1 copy air-gapped (ransomware protection)")
    print("  + 0 backup errors (automated restore verification)")

    # ── Design Summary ────────────────────────────
    print("\n\n[7] BACKUP & DR DESIGN DECISIONS")
    print("─" * 55)

    decisions = [
        ("Define RPO first",      "Determines backup frequency (hourly WAL vs daily full)"),
        ("Define RTO first",      "Determines DR strategy (warm standby vs active-active)"),
        ("Test restores regularly","Untested backup = no backup; automate restore verification"),
        ("WAL streaming for PITR","Near-zero RPO; pg_basebackup + pg_wal archive"),
        ("Incremental for size",  "Daily full + hourly incremental → 24x smaller backups"),
        ("Offsite copy mandatory","Local disk failure + fire/flood = lose all local copies"),
        ("Encrypt backups at rest","Breached backup = breached data; AES-256 + KMS"),
        ("Immutable backup copies","S3 Object Lock / WORM prevents ransomware deletion"),
    ]
    for decision, reason in decisions:
        print(f"  {decision:<26} {reason}")


if __name__ == "__main__":
    demonstrate_backup_dr()
