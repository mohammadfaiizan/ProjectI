"""
Payment System - Core Implementation
Demonstrates: payment state machine, double-entry ledger, idempotency store,
reconciliation job, fraud detection with velocity checks.
Standard library only. All amounts in minor units (cents).
"""

import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class PaymentStatus(Enum):
    PENDING           = "PENDING"
    PROCESSING        = "PROCESSING"
    COMPLETED         = "COMPLETED"
    FAILED            = "FAILED"
    REFUND_INITIATED  = "REFUND_INITIATED"
    REFUNDED          = "REFUNDED"


class EntryType(Enum):
    DEBIT  = "DEBIT"
    CREDIT = "CREDIT"


VALID_TRANSITIONS = {
    PaymentStatus.PENDING:          {PaymentStatus.PROCESSING, PaymentStatus.FAILED},
    PaymentStatus.PROCESSING:       {PaymentStatus.COMPLETED, PaymentStatus.FAILED},
    PaymentStatus.COMPLETED:        {PaymentStatus.REFUND_INITIATED},
    PaymentStatus.FAILED:           set(),
    PaymentStatus.REFUND_INITIATED: {PaymentStatus.REFUNDED},
    PaymentStatus.REFUNDED:         set(),
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Payment:
    payment_id:      str
    user_id:         str
    merchant_id:     str
    amount:          int        # minor units (cents)
    currency:        str        # ISO 4217
    status:          PaymentStatus = PaymentStatus.PENDING
    psp_reference:   Optional[str] = None
    idempotency_key: Optional[str] = None
    metadata:        Dict[str, Any] = field(default_factory=dict)
    created_at:      float = field(default_factory=time.time)
    updated_at:      float = field(default_factory=time.time)


@dataclass
class LedgerEntry:
    entry_id:    str
    payment_id:  str
    account_id:  str       # e.g. "user:alice:wallet"
    entry_type:  EntryType
    amount:      int       # always positive, minor units
    currency:    str
    description: str
    created_at:  float = field(default_factory=time.time)


@dataclass
class IdempotencyRecord:
    idem_key:    str
    payment_id:  str
    status:      PaymentStatus
    response:    Dict[str, Any]
    created_at:  float = field(default_factory=time.time)
    expires_at:  float = 0.0


@dataclass
class Refund:
    refund_id:      str
    payment_id:     str
    amount:         int
    reason:         str
    status:         str = "PENDING"
    psp_refund_ref: Optional[str] = None
    created_at:     float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Payment State Machine
# ---------------------------------------------------------------------------

class PaymentStateMachine:
    """Enforces valid payment status transitions."""

    @staticmethod
    def transition(payment: Payment, new_status: PaymentStatus) -> bool:
        allowed = VALID_TRANSITIONS.get(payment.status, set())
        if new_status not in allowed:
            print(f"  [FSM] Invalid transition: {payment.status.value} -> "
                  f"{new_status.value} for payment {payment.payment_id[:8]}")
            return False
        payment.status = new_status
        payment.updated_at = time.time()
        return True


# ---------------------------------------------------------------------------
# Ledger System (Double-Entry Bookkeeping)
# ---------------------------------------------------------------------------

class LedgerSystem:
    """
    Immutable double-entry ledger.
    Every transaction creates a DEBIT entry and a matching CREDIT entry.
    Invariant: SUM(DEBIT amounts) == SUM(CREDIT amounts) at all times.
    """

    def __init__(self):
        self._entries: List[LedgerEntry] = []

    def record_transfer(self, payment_id: str, from_account: str, to_account: str,
                        amount: int, currency: str, description: str) -> Tuple[str, str]:
        """
        Record a money transfer as two ledger entries.
        Returns (debit_entry_id, credit_entry_id).
        """
        if amount <= 0:
            raise ValueError(f"Amount must be positive, got {amount}")

        debit_id = str(uuid.uuid4())
        credit_id = str(uuid.uuid4())

        self._entries.append(LedgerEntry(
            entry_id=debit_id,
            payment_id=payment_id,
            account_id=from_account,
            entry_type=EntryType.DEBIT,
            amount=amount,
            currency=currency,
            description=description,
        ))
        self._entries.append(LedgerEntry(
            entry_id=credit_id,
            payment_id=payment_id,
            account_id=to_account,
            entry_type=EntryType.CREDIT,
            amount=amount,
            currency=currency,
            description=description,
        ))
        return debit_id, credit_id

    def get_account_balance(self, account_id: str, currency: str = "USD") -> int:
        """
        Balance = SUM(CREDIT) - SUM(DEBIT) for the account.
        Positive = money in account; Negative = account owes money.
        """
        credits = sum(e.amount for e in self._entries
                      if e.account_id == account_id
                      and e.entry_type == EntryType.CREDIT
                      and e.currency == currency)
        debits  = sum(e.amount for e in self._entries
                      if e.account_id == account_id
                      and e.entry_type == EntryType.DEBIT
                      and e.currency == currency)
        return credits - debits

    def verify_balance_invariant(self) -> Tuple[bool, int, int]:
        """
        Checks that total debits == total credits across all accounts.
        Returns (balanced, total_debits, total_credits).
        """
        total_debits  = sum(e.amount for e in self._entries if e.entry_type == EntryType.DEBIT)
        total_credits = sum(e.amount for e in self._entries if e.entry_type == EntryType.CREDIT)
        return total_debits == total_credits, total_debits, total_credits

    def get_payment_entries(self, payment_id: str) -> List[LedgerEntry]:
        return [e for e in self._entries if e.payment_id == payment_id]

    def entry_count(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Idempotency Store
# ---------------------------------------------------------------------------

class IdempotencyStore:
    """
    Prevents double-charges on retry.
    In production: Redis cache with PostgreSQL fallback.
    """

    TTL_SECONDS = 86400  # 24 hours

    def __init__(self):
        self._store: Dict[str, IdempotencyRecord] = {}

    def get(self, idem_key: str) -> Optional[IdempotencyRecord]:
        record = self._store.get(idem_key)
        if record and time.time() > record.expires_at:
            del self._store[idem_key]
            return None
        return record

    def set(self, idem_key: str, payment_id: str, status: PaymentStatus,
            response: Dict[str, Any]):
        self._store[idem_key] = IdempotencyRecord(
            idem_key=idem_key,
            payment_id=payment_id,
            status=status,
            response=response,
            expires_at=time.time() + self.TTL_SECONDS,
        )

    def update_status(self, idem_key: str, new_status: PaymentStatus,
                      response: Dict[str, Any]):
        record = self._store.get(idem_key)
        if record:
            record.status = new_status
            record.response = response


# ---------------------------------------------------------------------------
# Fraud Detector
# ---------------------------------------------------------------------------

class FraudDetector:
    """
    Rule-based velocity checks for real-time fraud prevention.
    In production: combined with ML scoring (XGBoost / deep learning).
    """

    MAX_TRANSACTIONS_PER_HOUR = 10
    MAX_SPEND_PER_HOUR = 50000  # $500.00 in cents
    MAX_PER_CARD_PER_HOUR = 5

    def __init__(self):
        # user_id -> [(timestamp, amount), ...]
        self._user_history: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        # card_token -> [timestamp, ...]
        self._card_history: Dict[str, List[float]] = defaultdict(list)
        self._blocked_users: set = set()

    def check(self, user_id: str, amount: int, card_token: Optional[str] = None,
              ) -> Tuple[bool, str]:
        """Returns (allow, reason). allow=False means reject."""
        if user_id in self._blocked_users:
            return False, "User is blocked"

        now = time.time()
        one_hour_ago = now - 3600

        # Velocity: transaction count per user per hour
        user_txns = [(t, a) for t, a in self._user_history[user_id] if t > one_hour_ago]
        if len(user_txns) >= self.MAX_TRANSACTIONS_PER_HOUR:
            return False, (f"Velocity: {len(user_txns)} transactions in last hour "
                           f"(max {self.MAX_TRANSACTIONS_PER_HOUR})")

        # Velocity: spend per user per hour
        hourly_spend = sum(a for _, a in user_txns)
        if hourly_spend + amount > self.MAX_SPEND_PER_HOUR:
            return False, (f"Velocity: ${(hourly_spend + amount)/100:.2f} hourly spend "
                           f"exceeds ${self.MAX_SPEND_PER_HOUR/100:.2f} limit")

        # Card-level velocity check
        if card_token:
            card_txns = [t for t in self._card_history[card_token] if t > one_hour_ago]
            if len(card_txns) >= self.MAX_PER_CARD_PER_HOUR:
                return False, (f"Card velocity: {len(card_txns)} charges in last hour "
                               f"(max {self.MAX_PER_CARD_PER_HOUR})")

        return True, "APPROVED"

    def record_transaction(self, user_id: str, amount: int,
                           card_token: Optional[str] = None):
        now = time.time()
        self._user_history[user_id].append((now, amount))
        if card_token:
            self._card_history[card_token].append(now)

    def block_user(self, user_id: str):
        self._blocked_users.add(user_id)

    def get_user_stats(self, user_id: str) -> Dict:
        now = time.time()
        one_hour_ago = now - 3600
        recent = [(t, a) for t, a in self._user_history[user_id] if t > one_hour_ago]
        return {
            "transactions_last_hour": len(recent),
            "spend_last_hour_cents": sum(a for _, a in recent),
            "total_transactions": len(self._user_history[user_id]),
        }


# ---------------------------------------------------------------------------
# Reconciliation Job
# ---------------------------------------------------------------------------

@dataclass
class PSPRecord:
    """Simulates a row from PSP's settlement report."""
    psp_reference: str
    amount: int
    currency: str
    status: str
    created_at: float


class ReconciliationJob:
    """
    Nightly job: compare internal payment records vs PSP settlement report.
    Finds: missing in PSP, missing in internal, amount mismatches.
    """

    def run(self, internal_payments: List[Payment],
            psp_records: List[PSPRecord]) -> Dict[str, List]:
        internal_map = {
            p.psp_reference: p for p in internal_payments if p.psp_reference
        }
        psp_map = {r.psp_reference: r for r in psp_records}

        all_refs = set(internal_map) | set(psp_map)
        results = {
            "matched": [],
            "missing_in_psp": [],        # internal says COMPLETED, PSP has no record
            "missing_in_internal": [],   # PSP settled, we have no record
            "amount_mismatch": [],       # both have record but amounts differ
            "status_mismatch": [],       # status differs
        }

        for ref in all_refs:
            internal = internal_map.get(ref)
            psp = psp_map.get(ref)

            if internal and not psp:
                if internal.status == PaymentStatus.COMPLETED:
                    results["missing_in_psp"].append({
                        "psp_reference": ref,
                        "internal_amount": internal.amount,
                        "internal_status": internal.status.value,
                    })
            elif psp and not internal:
                results["missing_in_internal"].append({
                    "psp_reference": ref,
                    "psp_amount": psp.amount,
                    "psp_status": psp.status,
                })
            elif internal and psp:
                if internal.amount != psp.amount:
                    results["amount_mismatch"].append({
                        "psp_reference": ref,
                        "internal_amount": internal.amount,
                        "psp_amount": psp.amount,
                        "delta": psp.amount - internal.amount,
                    })
                elif internal.status == PaymentStatus.COMPLETED and psp.status != "SETTLED":
                    results["status_mismatch"].append({
                        "psp_reference": ref,
                        "internal_status": internal.status.value,
                        "psp_status": psp.status,
                    })
                else:
                    results["matched"].append(ref)

        return results


# ---------------------------------------------------------------------------
# Main Payment System
# ---------------------------------------------------------------------------

class PaymentSystem:

    PLATFORM_FEE_BPS = 200  # 2% platform fee (200 basis points)

    def __init__(self):
        self._payments: Dict[str, Payment] = {}
        self._refunds:  Dict[str, Refund]  = {}
        self.ledger      = LedgerSystem()
        self.idempotency = IdempotencyStore()
        self.fraud       = FraudDetector()
        self._fsm        = PaymentStateMachine()

    def initiate_payment(
        self, user_id: str, merchant_id: str, amount: int, currency: str,
        payment_method: str, idempotency_key: str, metadata: Optional[Dict] = None,
    ) -> Tuple[Optional[Dict], str]:
        """
        Create a payment with idempotency guarantee.
        Returns (response_dict, message).
        """
        # 1. Idempotency check — deduplicate retries
        existing = self.idempotency.get(idempotency_key)
        if existing:
            print(f"  [IDEMPOTENCY] Duplicate request for key {idempotency_key[:12]}... "
                  f"returning cached response (status={existing.status.value})")
            return existing.response, "Duplicate request — returning cached response"

        # 2. Fraud check
        allowed, fraud_reason = self.fraud.check(user_id, amount, payment_method)
        if not allowed:
            response = {"error": "payment_declined", "reason": fraud_reason}
            self.idempotency.set(idempotency_key, "", PaymentStatus.FAILED, response)
            return None, f"FRAUD DECLINED: {fraud_reason}"

        # 3. Create payment record
        payment_id = str(uuid.uuid4())
        payment = Payment(
            payment_id=payment_id,
            user_id=user_id,
            merchant_id=merchant_id,
            amount=amount,
            currency=currency,
            idempotency_key=idempotency_key,
            metadata=metadata or {},
        )
        self._payments[payment_id] = payment

        # 4. Store idempotency key immediately (before PSP call)
        self.idempotency.set(
            idempotency_key, payment_id, PaymentStatus.PENDING,
            {"payment_id": payment_id, "status": "PENDING"}
        )

        return {"payment_id": payment_id, "status": "PENDING"}, "Payment initiated"

    def process_payment(self, payment_id: str, psp_token: str
                        ) -> Tuple[bool, str]:
        """
        Phase 2: Call PSP, update status, write ledger entries.
        """
        payment = self._payments.get(payment_id)
        if not payment:
            return False, "Payment not found"

        # Transition PENDING -> PROCESSING
        if not self._fsm.transition(payment, PaymentStatus.PROCESSING):
            return False, "Invalid state for processing"

        # Simulate PSP call (token "fail_*" triggers failure)
        psp_success = not psp_token.startswith("fail_")
        psp_ref = f"psp_ch_{hashlib.md5(payment_id.encode()).hexdigest()[:12]}" \
                  if psp_success else None

        if psp_success:
            payment.psp_reference = psp_ref
            self._fsm.transition(payment, PaymentStatus.COMPLETED)

            # Write double-entry ledger
            self._write_payment_ledger(payment)

            # Record transaction for fraud velocity tracking
            self.fraud.record_transaction(payment.user_id, payment.amount, psp_token)

            # Update idempotency record
            response = {
                "payment_id": payment_id,
                "status": "COMPLETED",
                "psp_reference": psp_ref,
                "amount": payment.amount,
                "currency": payment.currency,
            }
            if payment.idempotency_key:
                self.idempotency.update_status(
                    payment.idempotency_key, PaymentStatus.COMPLETED, response
                )

            print(f"  [PAYMENT] {payment_id[:8]} COMPLETED: "
                  f"${payment.amount/100:.2f} {payment.currency} "
                  f"from {payment.user_id} to {payment.merchant_id}")
            return True, f"Payment completed (PSP ref: {psp_ref})"

        else:
            self._fsm.transition(payment, PaymentStatus.FAILED)
            if payment.idempotency_key:
                self.idempotency.update_status(
                    payment.idempotency_key, PaymentStatus.FAILED,
                    {"payment_id": payment_id, "status": "FAILED"}
                )
            return False, "PSP declined the payment"

    def _write_payment_ledger(self, payment: Payment):
        """Double-entry: user wallet -> merchant account, minus platform fee."""
        fee = int(payment.amount * self.PLATFORM_FEE_BPS / 10000)
        net = payment.amount - fee

        user_account     = f"user:{payment.user_id}:wallet"
        merchant_account = f"merchant:{payment.merchant_id}:receivable"
        platform_account = "platform:revenue"

        # Full amount flows from user to merchant
        self.ledger.record_transfer(
            payment.payment_id, user_account, merchant_account,
            payment.amount, payment.currency,
            f"Payment {payment.payment_id[:8]}"
        )
        # Platform fee flows back from merchant to platform
        if fee > 0:
            self.ledger.record_transfer(
                payment.payment_id, merchant_account, platform_account,
                fee, payment.currency,
                f"Platform fee {self.PLATFORM_FEE_BPS}bps"
            )

    def refund_payment(self, payment_id: str, amount: Optional[int] = None,
                       reason: str = "Customer request") -> Tuple[Optional[str], str]:
        """Partial or full refund. Returns (refund_id, message)."""
        payment = self._payments.get(payment_id)
        if not payment:
            return None, "Payment not found"
        if payment.status != PaymentStatus.COMPLETED:
            return None, f"Cannot refund payment in status {payment.status.value}"

        refund_amount = amount or payment.amount
        if refund_amount > payment.amount:
            return None, "Refund amount exceeds payment amount"

        # Transition COMPLETED -> REFUND_INITIATED
        self._fsm.transition(payment, PaymentStatus.REFUND_INITIATED)

        refund_id = str(uuid.uuid4())
        refund = Refund(
            refund_id=refund_id,
            payment_id=payment_id,
            amount=refund_amount,
            reason=reason,
            psp_refund_ref=f"psp_re_{refund_id[:12]}",
        )
        self._refunds[refund_id] = refund

        # Simulate PSP refund success
        refund.status = "COMPLETED"
        self._fsm.transition(payment, PaymentStatus.REFUNDED)

        # Reverse ledger entries
        merchant_account = f"merchant:{payment.merchant_id}:receivable"
        user_account     = f"user:{payment.user_id}:wallet"
        self.ledger.record_transfer(
            payment_id, merchant_account, user_account,
            refund_amount, payment.currency,
            f"Refund for payment {payment_id[:8]}: {reason}"
        )

        print(f"  [REFUND] ${refund_amount/100:.2f} refunded for payment {payment_id[:8]}")
        return refund_id, f"Refund of ${refund_amount/100:.2f} initiated"

    def get_payment_status(self, payment_id: str) -> Optional[Dict]:
        p = self._payments.get(payment_id)
        if not p:
            return None
        return {
            "payment_id": p.payment_id,
            "status": p.status.value,
            "amount": p.amount,
            "currency": p.currency,
            "psp_reference": p.psp_reference,
        }

    def get_user_payments(self, user_id: str) -> List[Dict]:
        return [
            {"payment_id": p.payment_id, "amount": p.amount,
             "status": p.status.value, "created_at": p.created_at}
            for p in self._payments.values() if p.user_id == user_id
        ]


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demo_full_payment_flow():
    print("=== Full Payment Flow ===")
    system = PaymentSystem()

    # Normal payment
    idem_key = str(uuid.uuid4())
    resp, msg = system.initiate_payment(
        "user_alice", "merch_bookstore", 2999, "USD",
        "tok_visa_4242", idem_key, {"order_id": "ord_001"}
    )
    print(f"Initiate: {msg} | {resp}")

    ok, msg2 = system.process_payment(resp["payment_id"], "tok_visa_4242")
    print(f"Process:  {msg2}")

    # Idempotency: retry with same key
    resp2, msg3 = system.initiate_payment(
        "user_alice", "merch_bookstore", 2999, "USD",
        "tok_visa_4242", idem_key
    )
    print(f"Retry (same idempotency key): {msg3}")

    # Check ledger
    balanced, total_d, total_c = system.ledger.verify_balance_invariant()
    print(f"\nLedger balanced: {balanced} (debits={total_d}, credits={total_c})")
    alice_balance = system.ledger.get_account_balance("user:user_alice:wallet")
    merch_balance = system.ledger.get_account_balance("merchant:merch_bookstore:receivable")
    platform_rev  = system.ledger.get_account_balance("platform:revenue")
    print(f"Alice wallet:    ${alice_balance/100:.2f} (expected: -$29.99)")
    print(f"Merchant net:    ${merch_balance/100:.2f} (expected: $29.39 after 2% fee)")
    print(f"Platform fee:    ${platform_rev/100:.2f} (expected: $0.60)")

    # Refund
    payment_id = resp["payment_id"]
    refund_id, rmsg = system.refund_payment(payment_id, reason="Wrong item")
    print(f"\nRefund: {rmsg}")

    # Verify ledger still balanced after refund
    balanced, total_d, total_c = system.ledger.verify_balance_invariant()
    print(f"Ledger balanced after refund: {balanced}")
    alice_after = system.ledger.get_account_balance("user:user_alice:wallet")
    print(f"Alice wallet after refund: ${alice_after/100:.2f} (expected: $0.00)")


def demo_fraud_detection():
    print("\n=== Fraud Detection Demo ===")
    system = PaymentSystem()
    # Lower threshold for demo (normally 10/hr)
    system.fraud.MAX_TRANSACTIONS_PER_HOUR = 3

    # Simulate normal usage (3 legitimate transactions)
    for i in range(3):
        ikey = str(uuid.uuid4())
        resp, msg = system.initiate_payment(
            "user_bob", "merch_shop", 1000, "USD", "tok_card_001", ikey
        )
        if resp:
            system.process_payment(resp["payment_id"], "tok_card_001")
    print(f"Normal 3 transactions: OK (threshold={system.fraud.MAX_TRANSACTIONS_PER_HOUR}/hr)")
    print(f"  Stats: {system.fraud.get_user_stats('user_bob')}")

    # Simulate velocity attack (many small charges — card testing pattern)
    print("\nSimulating velocity attack (5 more rapid transactions, should all be blocked)...")
    failures = 0
    for i in range(5):
        ikey = str(uuid.uuid4())
        resp, msg = system.initiate_payment(
            "user_bob", "merch_shop", 100, "USD", f"tok_card_{i:03d}", ikey
        )
        if resp is None:
            failures += 1
            if failures == 1:
                print(f"  First block: {msg}")
    print(f"  Blocked attempts: {failures} out of 5 (expected: 5)")


def demo_reconciliation():
    print("\n=== Reconciliation Demo ===")
    system = PaymentSystem()
    job = ReconciliationJob()

    # Create payments
    payments_created = []
    for i in range(5):
        ikey = str(uuid.uuid4())
        resp, _ = system.initiate_payment(
            f"user_{i}", "merch_main", 1000 * (i + 1), "USD", "tok_card", ikey
        )
        ok, _ = system.process_payment(resp["payment_id"], "tok_card")
        if ok:
            payments_created.append(resp["payment_id"])

    # Build internal payments list
    internal = list(system._payments.values())

    # Simulate PSP records — with intentional discrepancies
    psp_records = []
    for p in internal:
        if p.psp_reference:
            # Introduce a mismatch for one payment
            amount = p.amount if p.payment_id != payments_created[2] else p.amount + 50
            psp_records.append(PSPRecord(
                psp_reference=p.psp_reference,
                amount=amount,
                currency=p.currency,
                status="SETTLED",
                created_at=p.created_at,
            ))

    # Add a ghost PSP record (PSP has it, we don't)
    psp_records.append(PSPRecord(
        psp_reference="psp_ch_ghost_12345",
        amount=9999,
        currency="USD",
        status="SETTLED",
        created_at=time.time(),
    ))

    results = job.run(internal, psp_records)
    print(f"  Matched:              {len(results['matched'])}")
    print(f"  Amount mismatches:    {len(results['amount_mismatch'])}")
    if results['amount_mismatch']:
        for m in results['amount_mismatch']:
            print(f"    {m['psp_reference']}: internal=${m['internal_amount']/100:.2f}, "
                  f"PSP=${m['psp_amount']/100:.2f}, delta=${m['delta']/100:.2f}")
    print(f"  Missing in PSP:       {len(results['missing_in_psp'])}")
    print(f"  Missing in internal:  {len(results['missing_in_internal'])}")
    if results['missing_in_internal']:
        for m in results['missing_in_internal']:
            print(f"    Ghost PSP ref: {m['psp_reference']}, "
                  f"amount=${m['psp_amount']/100:.2f}")


def demo_state_machine():
    print("\n=== Payment State Machine Demo ===")
    fsm = PaymentStateMachine()
    p = Payment(str(uuid.uuid4()), "u1", "m1", 500, "USD")
    print(f"Initial: {p.status.value}")
    fsm.transition(p, PaymentStatus.PROCESSING)
    print(f"After PROCESSING: {p.status.value}")
    # Invalid: can't go from PROCESSING back to PENDING
    result = fsm.transition(p, PaymentStatus.PENDING)
    print(f"Invalid PENDING transition blocked: {not result}")
    fsm.transition(p, PaymentStatus.COMPLETED)
    print(f"After COMPLETED: {p.status.value}")
    fsm.transition(p, PaymentStatus.REFUND_INITIATED)
    fsm.transition(p, PaymentStatus.REFUNDED)
    print(f"After REFUNDED: {p.status.value}")
    # Terminal state — no transitions allowed
    result = fsm.transition(p, PaymentStatus.FAILED)
    print(f"Terminal state transition blocked: {not result}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_full_payment_flow()
    demo_fraud_detection()
    demo_reconciliation()
    demo_state_machine()
