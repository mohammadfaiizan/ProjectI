"""
SYSTEM DESIGN: PAYMENT PROCESSING SYSTEM
==========================================

Problem Statement:
Design a payment system that processes credit card transactions,
supports refunds, and maintains financial consistency.

Functional Requirements:
  - Charge a payment method (card, wallet, bank)
  - Authorize then capture (two-step payment)
  - Refund (full or partial)
  - Payment history per user/merchant
  - Webhooks for payment events (paid, failed, refunded)

Non-Functional Requirements:
  - Process 10K transactions/sec
  - Exactly-once processing (no double charges)
  - ACID transactions (atomicity critical for money movement)
  - PCI-DSS compliance (card data handling)
  - 99.999% availability ($1M/minute downtime cost)

Key Concepts:

  Idempotency Keys:
    Client sends idempotency_key with every request.
    If server crashes after charging but before responding:
    client retries with same key → server returns cached result.
    DB stores: idempotency_key → (response, timestamp).
    Prevents double charges on retry.

  Two-Phase Payment:
    Authorize: Reserve funds on card. Card network holds funds.
    Capture:   Collect the authorized amount. Typically < 7 days.
    Good for: "charge when item ships"; pre-auth hotel stays.

  Ledger (Double-Entry Accounting):
    Every transaction = debit one account + credit another.
    Assets = Liabilities + Equity (always balanced).
    Immutable: never update or delete. Append-only.
    Audit trail: sum all entries → current balance.

  PCI-DSS Compliance:
    Store only: last4, card_type, expiry (never full PAN).
    Tokenize card: real card → provider token (Stripe/Braintree).
    Server never sees full card number (handled by Stripe.js/SDK).

  Payment Flow:
    Client → Payment API → Idempotency check →
    → Card tokenization (Stripe) → Authorize → DB record →
    → Capture → Ledger entry → Webhook → Client

  Retry Logic:
    Network failures: retry with same idempotency_key.
    Provider errors: 402 = card declined (don't retry).
                     503 = provider unavailable (retry with backoff).

  Currency Handling:
    Store amounts in SMALLEST UNIT (cents, not dollars).
    Avoid floating point for money. Use integer arithmetic.
    1.99 USD = 199 cents in DB.
"""

from __future__ import annotations

import time
import uuid
import random
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP


# ─────────────────────────────────────────────
# MONEY (integer cents to avoid float issues)
# ─────────────────────────────────────────────

class Money:
    """Immutable money value in smallest currency unit (cents)."""

    def __init__(self, amount_cents: int, currency: str = "USD"):
        self._cents   = amount_cents
        self._currency = currency

    @classmethod
    def from_float(cls, dollars: float, currency: str = "USD") -> "Money":
        cents = int(Decimal(str(dollars)).quantize(Decimal("0.01"),
                    rounding=ROUND_HALF_UP) * 100)
        return cls(cents, currency)

    @property
    def cents(self) -> int:
        return self._cents

    @property
    def dollars(self) -> Decimal:
        return Decimal(self._cents) / 100

    def __add__(self, other: "Money") -> "Money":
        if self._currency != other._currency:
            raise ValueError("Currency mismatch")
        return Money(self._cents + other._cents, self._currency)

    def __sub__(self, other: "Money") -> "Money":
        if self._currency != other._currency:
            raise ValueError("Currency mismatch")
        return Money(self._cents - other._cents, self._currency)

    def __repr__(self) -> str:
        return f"{self._currency} {self.dollars:.2f}"

    def __eq__(self, other) -> bool:
        return isinstance(other, Money) and self._cents == other._cents

    def __gt__(self, other: "Money") -> bool:
        return self._cents > other._cents


# ─────────────────────────────────────────────
# PAYMENT STATUS
# ─────────────────────────────────────────────

class PaymentStatus(Enum):
    PENDING     = "pending"
    AUTHORIZED  = "authorized"
    CAPTURED    = "captured"
    FAILED      = "failed"
    REFUNDED    = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"
    CANCELLED   = "cancelled"


# ─────────────────────────────────────────────
# PAYMENT RECORD
# ─────────────────────────────────────────────

@dataclass
class PaymentMethod:
    method_id:    str
    type:         str   # card, wallet, bank_transfer
    last4:        str   # last 4 digits
    brand:        str   # visa, mastercard, amex
    token:        str   # provider token (never store real card)
    exp_month:    int
    exp_year:     int


@dataclass
class Payment:
    payment_id:       str
    user_id:          str
    merchant_id:      str
    amount:           Money
    status:           PaymentStatus
    method:           PaymentMethod
    idempotency_key:  str
    created_at:       float
    provider_ref:     Optional[str] = None   # Stripe charge ID
    auth_code:        Optional[str] = None
    captured_at:      Optional[float] = None
    refunded_amount:  int = 0   # in cents
    metadata:         Dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────
# DOUBLE-ENTRY LEDGER
# ─────────────────────────────────────────────

@dataclass
class LedgerEntry:
    entry_id:    str
    account_id:  str
    amount_cents: int    # positive = credit, negative = debit
    currency:    str
    payment_id:  str
    description: str
    created_at:  float
    is_reversal: bool = False


class Ledger:
    """
    Append-only double-entry accounting ledger.
    Every transaction: debit one account + credit another.
    """

    def __init__(self):
        self._entries: List[LedgerEntry] = []
        self._lock = threading.Lock()

    def record(self, payment_id: str, debit_account: str,
               credit_account: str, amount_cents: int,
               currency: str, description: str) -> Tuple[str, str]:
        """
        Record a transfer: debit from debit_account → credit to credit_account.
        Returns (debit_entry_id, credit_entry_id).
        """
        ts = time.time()
        with self._lock:
            debit_id  = uuid.uuid4().hex[:12]
            credit_id = uuid.uuid4().hex[:12]
            self._entries.append(LedgerEntry(
                debit_id, debit_account, -amount_cents, currency,
                payment_id, description, ts
            ))
            self._entries.append(LedgerEntry(
                credit_id, credit_account, amount_cents, currency,
                payment_id, description, ts
            ))
        return debit_id, credit_id

    def balance(self, account_id: str) -> int:
        """Sum all entries for an account (in cents)."""
        return sum(e.amount_cents for e in self._entries
                   if e.account_id == account_id)

    def history(self, account_id: str, limit: int = 50) -> List[LedgerEntry]:
        entries = [e for e in self._entries if e.account_id == account_id]
        return sorted(entries, key=lambda e: -e.created_at)[:limit]

    def is_balanced(self) -> bool:
        """Total of all entries across all accounts should be 0."""
        return sum(e.amount_cents for e in self._entries) == 0


# ─────────────────────────────────────────────
# IDEMPOTENCY STORE
# ─────────────────────────────────────────────

class IdempotencyStore:
    """
    Caches responses by idempotency_key.
    Prevents double-charging on retries.
    """

    def __init__(self, ttl_s: float = 86400 * 7):
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._ttl   = ttl_s

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if not entry:
            return None
        response, ts = entry
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        return response

    def set(self, key: str, response: Any):
        self._store[key] = (response, time.time())

    def contains(self, key: str) -> bool:
        return self.get(key) is not None


# ─────────────────────────────────────────────
# PAYMENT PROVIDER SIMULATOR (Stripe-like)
# ─────────────────────────────────────────────

class ProviderError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        self.code      = code
        self.message   = message
        self.retryable = retryable
        super().__init__(message)


class PaymentProviderSimulator:
    """
    Simulates Stripe/Braintree payment processing.
    """

    def __init__(self, decline_rate: float = 0.05, error_rate: float = 0.02):
        self._decline_rate = decline_rate
        self._error_rate   = error_rate

    def authorize(self, token: str, amount_cents: int,
                  currency: str) -> Tuple[str, str]:
        """Returns (provider_ref, auth_code). Raises ProviderError on failure."""
        r = random.random()
        if r < self._error_rate:
            raise ProviderError("provider_unavailable", "Stripe unavailable",
                                retryable=True)
        if r < self._error_rate + self._decline_rate:
            raise ProviderError("card_declined", "Insufficient funds",
                                retryable=False)
        provider_ref = f"ch_{uuid.uuid4().hex[:20]}"
        auth_code    = f"AUTH{random.randint(100000, 999999)}"
        return provider_ref, auth_code

    def capture(self, provider_ref: str) -> bool:
        return True   # simplified

    def refund(self, provider_ref: str, amount_cents: int) -> str:
        return f"re_{uuid.uuid4().hex[:20]}"


# ─────────────────────────────────────────────
# PAYMENT SERVICE
# ─────────────────────────────────────────────

class PaymentService:
    def __init__(self):
        self._payments:  Dict[str, Payment]  = {}
        self._provider   = PaymentProviderSimulator()
        self._idempotency = IdempotencyStore()
        self._ledger     = Ledger()
        self._lock       = threading.Lock()

    def charge(self, user_id: str, merchant_id: str,
               amount: Money, method: PaymentMethod,
               idempotency_key: str,
               metadata: Optional[Dict] = None) -> Payment:
        """
        Charge a payment method.
        Idempotent: same key returns same result.
        """
        # Check idempotency
        cached = self._idempotency.get(idempotency_key)
        if cached:
            return cached

        payment_id = f"pay_{uuid.uuid4().hex[:16]}"
        payment = Payment(
            payment_id      = payment_id,
            user_id         = user_id,
            merchant_id     = merchant_id,
            amount          = amount,
            status          = PaymentStatus.PENDING,
            method          = method,
            idempotency_key = idempotency_key,
            created_at      = time.time(),
            metadata        = metadata or {},
        )
        self._payments[payment_id] = payment

        try:
            # Authorize + Capture in one step (synchronous charge)
            provider_ref, auth_code = self._provider.authorize(
                method.token, amount.cents, amount._currency)
            self._provider.capture(provider_ref)

            payment.status      = PaymentStatus.CAPTURED
            payment.provider_ref = provider_ref
            payment.auth_code   = auth_code
            payment.captured_at = time.time()

            # Record in ledger
            self._ledger.record(
                payment_id   = payment_id,
                debit_account  = f"user:{user_id}",
                credit_account = f"merchant:{merchant_id}",
                amount_cents = amount.cents,
                currency     = amount._currency,
                description  = f"Payment {payment_id}",
            )

        except ProviderError as e:
            payment.status = PaymentStatus.FAILED
            payment.metadata["error_code"]    = e.code
            payment.metadata["error_message"] = e.message

        self._idempotency.set(idempotency_key, payment)
        return payment

    def authorize(self, user_id: str, merchant_id: str,
                  amount: Money, method: PaymentMethod,
                  idempotency_key: str) -> Payment:
        """Two-phase: authorize only (capture separately)."""
        cached = self._idempotency.get(idempotency_key)
        if cached:
            return cached

        payment_id = f"pay_{uuid.uuid4().hex[:16]}"
        payment = Payment(
            payment_id      = payment_id,
            user_id         = user_id,
            merchant_id     = merchant_id,
            amount          = amount,
            status          = PaymentStatus.PENDING,
            method          = method,
            idempotency_key = idempotency_key,
            created_at      = time.time(),
        )
        self._payments[payment_id] = payment

        try:
            provider_ref, auth_code = self._provider.authorize(
                method.token, amount.cents, amount._currency)
            payment.status      = PaymentStatus.AUTHORIZED
            payment.provider_ref = provider_ref
            payment.auth_code   = auth_code
        except ProviderError as e:
            payment.status = PaymentStatus.FAILED
            payment.metadata["error_code"] = e.code

        self._idempotency.set(idempotency_key, payment)
        return payment

    def capture(self, payment_id: str) -> Payment:
        payment = self._payments.get(payment_id)
        if not payment or payment.status != PaymentStatus.AUTHORIZED:
            raise ValueError("Payment not found or not authorized")

        self._provider.capture(payment.provider_ref)
        payment.status      = PaymentStatus.CAPTURED
        payment.captured_at = time.time()
        self._ledger.record(
            payment_id, f"user:{payment.user_id}",
            f"merchant:{payment.merchant_id}",
            payment.amount.cents, payment.amount._currency,
            f"Capture {payment_id}",
        )
        return payment

    def refund(self, payment_id: str,
               amount: Optional[Money] = None) -> Payment:
        payment = self._payments.get(payment_id)
        if not payment or payment.status not in (PaymentStatus.CAPTURED,
                                                  PaymentStatus.PARTIALLY_REFUNDED):
            raise ValueError("Cannot refund: payment not captured")

        refund_amount = amount or payment.amount
        if refund_amount.cents > (payment.amount.cents - payment.refunded_amount):
            raise ValueError("Refund exceeds captured amount")

        ref_ref = self._provider.refund(payment.provider_ref, refund_amount.cents)
        payment.refunded_amount += refund_amount.cents

        if payment.refunded_amount >= payment.amount.cents:
            payment.status = PaymentStatus.REFUNDED
        else:
            payment.status = PaymentStatus.PARTIALLY_REFUNDED

        # Reverse ledger entry
        self._ledger.record(
            payment_id, f"merchant:{payment.merchant_id}",
            f"user:{payment.user_id}",
            refund_amount.cents, refund_amount._currency,
            f"Refund for {payment_id}",
        )
        return payment

    def history(self, user_id: str, limit: int = 20) -> List[Payment]:
        payments = [p for p in self._payments.values() if p.user_id == user_id]
        return sorted(payments, key=lambda p: -p.created_at)[:limit]


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_payments():
    print("=" * 65)
    print("SYSTEM DESIGN: PAYMENT PROCESSING SYSTEM")
    print("=" * 65)

    random.seed(42)
    svc = PaymentService()

    # ── Money Arithmetic ──────────────────────
    print("\n[1] MONEY (integer cents)")
    print("─" * 55)

    price  = Money.from_float(19.99)
    tax    = Money.from_float(1.60)
    total  = price + tax
    print(f"  Price:  {price}  ({price.cents} cents)")
    print(f"  Tax:    {tax}   ({tax.cents} cents)")
    print(f"  Total:  {total}  ({total.cents} cents)")
    print(f"  (Stored as integers; no floating point errors)")

    # ── Payment Method ────────────────────────
    card = PaymentMethod(
        method_id = "pm_abc123",
        type      = "card",
        last4     = "4242",
        brand     = "visa",
        token     = "tok_stripe_visa_4242",   # Stripe token
        exp_month = 12, exp_year = 2027,
    )

    # ── Direct Charge ─────────────────────────
    print("\n[2] DIRECT CHARGE")
    print("─" * 55)

    payment = svc.charge(
        user_id         = "user_001",
        merchant_id     = "merch_shop",
        amount          = Money.from_float(29.99),
        method          = card,
        idempotency_key = "idem_001_20240115",
    )
    print(f"  Payment ID: {payment.payment_id}")
    print(f"  Status:     {payment.status.value}")
    print(f"  Amount:     {payment.amount}")
    if payment.status == PaymentStatus.CAPTURED:
        print(f"  Provider:   {payment.provider_ref}")

    # ── Idempotency (retry) ───────────────────
    print("\n[3] IDEMPOTENCY (retry protection)")
    print("─" * 55)

    # Retry with same idempotency key
    p2 = svc.charge(
        "user_001", "merch_shop",
        Money.from_float(29.99), card,
        idempotency_key="idem_001_20240115",   # same key!
    )
    print(f"  Same idempotency_key → same payment_id: {p2.payment_id == payment.payment_id}")
    print(f"  (No double charge; returned cached result)")

    # ── Two-Phase: Authorize + Capture ────────
    print("\n[4] TWO-PHASE: AUTHORIZE + CAPTURE")
    print("─" * 55)

    auth_payment = svc.authorize(
        "user_002", "merch_hotel",
        Money.from_float(200.00), card,
        idempotency_key="idem_hotel_001",
    )
    print(f"  Authorize: {auth_payment.status.value}  amount={auth_payment.amount}")

    if auth_payment.status == PaymentStatus.AUTHORIZED:
        captured = svc.capture(auth_payment.payment_id)
        print(f"  Capture:   {captured.status.value}  captured_at set")

    # ── Refund ────────────────────────────────
    print("\n[5] REFUNDS")
    print("─" * 55)

    if payment.status == PaymentStatus.CAPTURED:
        # Partial refund
        partial = svc.refund(payment.payment_id, Money.from_float(10.00))
        print(f"  Partial refund $10.00: status={partial.status.value}")
        print(f"  Refunded so far: ${partial.refunded_amount/100:.2f} of ${payment.amount.dollars:.2f}")

        # Full refund
        full_refund = svc.refund(payment.payment_id)
        print(f"  Full refund: status={full_refund.status.value}")

    # ── Ledger ────────────────────────────────
    print("\n[6] DOUBLE-ENTRY LEDGER")
    print("─" * 55)

    balanced = svc._ledger.is_balanced()
    print(f"  Ledger balanced: {balanced} (sum of all entries = 0)")
    print(f"  Total entries:   {len(svc._ledger._entries)}")

    # Show user balance
    user_bal    = svc._ledger.balance("user:user_001")
    merchant_bal= svc._ledger.balance("merchant:merch_shop")
    print(f"\n  user_001 balance:    ${user_bal / 100:.2f} (negative = paid out)")
    print(f"  merch_shop balance:  ${merchant_bal / 100:.2f} (positive = received)")

    print("\n  Recent ledger entries for user_001:")
    for e in svc._ledger.history("user:user_001")[:4]:
        sign = "+" if e.amount_cents > 0 else ""
        print(f"    {sign}${e.amount_cents/100:.2f}  {e.description[:45]}")

    # ── Payment History ───────────────────────
    print("\n[7] PAYMENT HISTORY")
    print("─" * 55)

    history = svc.history("user_001")
    for p in history:
        print(f"  {p.payment_id[:12]}  {p.amount}  {p.status.value}")

    # ── Architecture ──────────────────────────
    print("\n[8] PAYMENT SYSTEM ARCHITECTURE")
    print("─" * 55)

    arch = [
        ("Idempotency",   "Redis: idempotency_key → payment_id (7d TTL)"),
        ("DB",            "Postgres ACID; payments + ledger in same transaction"),
        ("Provider",      "Stripe/Braintree; card tokens never stored server-side"),
        ("PCI-DSS",       "Stripe.js in browser; server sees only tokens"),
        ("Ledger",        "Append-only double-entry; never update/delete rows"),
        ("Currency",      "Store in cents (int); use Decimal for display"),
        ("Webhooks",      "Payment events → Kafka → Webhook service → merchant"),
        ("Retry",         "Exponential backoff; same idempotency_key on retry"),
        ("Fraud",         "ML model: device fingerprint + velocity checks"),
        ("Reconciliation","Daily batch: DB totals vs provider totals must match"),
    ]
    for component, detail in arch:
        print(f"  {component:<18} {detail}")


if __name__ == "__main__":
    demonstrate_payments()
