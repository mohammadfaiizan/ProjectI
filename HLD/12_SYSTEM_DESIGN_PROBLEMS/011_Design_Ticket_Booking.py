"""
SYSTEM DESIGN: TICKET BOOKING SYSTEM (like Ticketmaster)
==========================================================

Problem Statement:
Design a high-traffic ticket booking system for concerts/events.
Must handle massive traffic spikes when tickets go on sale, prevent
overselling, and provide a fair purchasing experience.

Functional Requirements:
  - Browse events and check availability
  - Select seats (or general admission quantity)
  - Reserve seats temporarily (hold for 10 min)
  - Complete purchase (payment + confirmation)
  - Cancel / refund

Non-Functional Requirements:
  - 10M concurrent users during popular on-sale events
  - < 1s to show seat availability
  - No overselling (exactness required — NOT eventual consistent)
  - Fair access during high demand (virtual queue)

Key Challenges:
  1. OVERSELLING: Two users book the same seat simultaneously.
     Solution: Pessimistic locking or conditional writes.
     DB-level: SELECT FOR UPDATE on seat row.
     Redis: SET key NX (set if not exists) → atomic reservation.

  2. TRAFFIC SPIKE: 10M users hitting "sale starts" simultaneously.
     Solution: Virtual waiting room queue.
     Users enter virtual queue → served in order when capacity allows.

  3. INVENTORY ACCURACY: "5 seats left" must be accurate enough.
     Solution: Redis counter for available count. DB for exact inventory.

  4. SEAT HOLD EXPIRY: Reserved seat not paid for in 10 min → release.
     Solution: Redis key with 10-min TTL → on expiry, increment available count.

  5. PAYMENT FAILURE: Payment fails after seat reserved → release seat.
     Solution: 2-phase: hold seat → charge → confirm. If charge fails → release.

Data Model:
  events:    event_id, name, venue, start_time, total_seats
  seats:     seat_id, event_id, row, number, section, status (available/held/sold)
  holds:     hold_id, seat_id, user_id, expires_at
  orders:    order_id, user_id, event_id, seat_ids, status, total_price
  payments:  payment_id, order_id, amount, status, provider_ref

Seat Assignment Modes:
  1. Map-based: user selects specific seat → hold that exact seat
  2. Best-available: system assigns best available seat
"""

from __future__ import annotations

import time
import uuid
import threading
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import deque


# ─────────────────────────────────────────────
# SEAT STATUS
# ─────────────────────────────────────────────

class SeatStatus(Enum):
    AVAILABLE = "available"
    HELD      = "held"
    SOLD      = "sold"


class OrderStatus(Enum):
    PENDING   = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    REFUNDED  = "refunded"


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class Seat:
    seat_id:   str
    event_id:  str
    section:   str
    row:       str
    number:    int
    price:     float
    status:    SeatStatus = SeatStatus.AVAILABLE
    held_by:   Optional[str] = None    # hold_id
    held_until: Optional[float] = None


@dataclass
class Event:
    event_id:    str
    name:        str
    venue:       str
    start_time:  float
    total_seats: int


@dataclass
class SeatHold:
    hold_id:   str
    user_id:   str
    seat_ids:  List[str]
    event_id:  str
    expires_at: float
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


@dataclass
class Order:
    order_id:    str
    user_id:     str
    event_id:    str
    seat_ids:    List[str]
    total_price: float
    status:      OrderStatus
    created_at:  float
    hold_id:     Optional[str] = None


# ─────────────────────────────────────────────
# INVENTORY MANAGER (atomic seat operations)
# ─────────────────────────────────────────────

class InventoryManager:
    """
    Thread-safe seat inventory.
    In production: Redis SET seat:{seat_id} NX (atomic hold).
    Or: SELECT FOR UPDATE on seat row in Postgres.
    """

    HOLD_TTL_S = 600   # 10 minutes

    def __init__(self):
        self._seats:   Dict[str, Seat]     = {}
        self._holds:   Dict[str, SeatHold] = {}
        self._lock     = threading.Lock()

    def add_seats(self, seats: List[Seat]):
        with self._lock:
            for s in seats:
                self._seats[s.seat_id] = s

    def available_seats(self, event_id: str) -> List[Seat]:
        self._expire_holds()
        return [s for s in self._seats.values()
                if s.event_id == event_id and s.status == SeatStatus.AVAILABLE]

    def _expire_holds(self):
        now    = time.time()
        expired = [hid for hid, h in self._holds.items() if h.is_expired()]
        for hid in expired:
            hold = self._holds.pop(hid)
            for sid in hold.seat_ids:
                seat = self._seats.get(sid)
                if seat and seat.status == SeatStatus.HELD and seat.held_by == hid:
                    seat.status    = SeatStatus.AVAILABLE
                    seat.held_by   = None
                    seat.held_until= None

    def hold_seats(self, user_id: str, event_id: str,
                   seat_ids: List[str]) -> Optional[SeatHold]:
        """
        Atomically hold seats.
        Returns SeatHold if successful; None if any seat unavailable.
        """
        with self._lock:
            self._expire_holds()
            # Validate all seats are available
            for sid in seat_ids:
                seat = self._seats.get(sid)
                if not seat or seat.event_id != event_id:
                    return None
                if seat.status != SeatStatus.AVAILABLE:
                    return None

            # Hold all seats
            hold_id    = uuid.uuid4().hex[:12]
            expires_at = time.time() + self.HOLD_TTL_S
            hold       = SeatHold(hold_id, user_id, seat_ids, event_id, expires_at)

            for sid in seat_ids:
                seat           = self._seats[sid]
                seat.status    = SeatStatus.HELD
                seat.held_by   = hold_id
                seat.held_until= expires_at

            self._holds[hold_id] = hold
            return hold

    def confirm_hold(self, hold_id: str) -> bool:
        """Convert hold → sold. Called after payment success."""
        with self._lock:
            hold = self._holds.get(hold_id)
            if not hold or hold.is_expired():
                return False
            for sid in hold.seat_ids:
                seat = self._seats.get(sid)
                if seat and seat.held_by == hold_id:
                    seat.status    = SeatStatus.SOLD
                    seat.held_by   = None
                    seat.held_until= None
            del self._holds[hold_id]
            return True

    def release_hold(self, hold_id: str):
        """Release hold (payment failed or user cancelled)."""
        with self._lock:
            hold = self._holds.pop(hold_id, None)
            if not hold:
                return
            for sid in hold.seat_ids:
                seat = self._seats.get(sid)
                if seat and seat.held_by == hold_id:
                    seat.status    = SeatStatus.AVAILABLE
                    seat.held_by   = None
                    seat.held_until= None

    def seat_counts(self, event_id: str) -> Dict[str, int]:
        counts = {"available": 0, "held": 0, "sold": 0}
        for s in self._seats.values():
            if s.event_id == event_id:
                counts[s.status.value] += 1
        return counts


# ─────────────────────────────────────────────
# VIRTUAL WAITING ROOM
# ─────────────────────────────────────────────

@dataclass
class QueuePosition:
    user_id:     str
    position:    int
    entered_at:  float
    token:       str


class VirtualQueue:
    """
    Manages fair access during high-demand on-sale events.
    Users get a position in the queue; admitted in order.
    """

    def __init__(self, throughput_per_sec: int = 100):
        self._queue:      deque = deque()
        self._position_map: Dict[str, QueuePosition] = {}
        self._admitted:   Set[str] = set()
        self._throughput  = throughput_per_sec
        self._last_admit  = time.time()
        self._total_admitted = 0

    def enter(self, user_id: str) -> QueuePosition:
        if user_id in self._position_map:
            return self._position_map[user_id]
        pos = QueuePosition(
            user_id    = user_id,
            position   = len(self._queue) + 1,
            entered_at = time.time(),
            token      = uuid.uuid4().hex[:16],
        )
        self._queue.append(user_id)
        self._position_map[user_id] = pos
        return pos

    def admit(self, n: Optional[int] = None) -> List[str]:
        """Admit up to n (or throughput-based) users."""
        now     = time.time()
        elapsed = now - self._last_admit
        to_admit = n or int(elapsed * self._throughput)
        admitted = []
        for _ in range(to_admit):
            if not self._queue:
                break
            uid = self._queue.popleft()
            self._admitted.add(uid)
            admitted.append(uid)
        if admitted:
            self._last_admit = now
            self._total_admitted += len(admitted)
        return admitted

    def is_admitted(self, user_id: str) -> bool:
        return user_id in self._admitted

    def queue_depth(self) -> int:
        return len(self._queue)

    def estimated_wait_s(self, user_id: str) -> Optional[float]:
        pos = self._position_map.get(user_id)
        if not pos or user_id in self._admitted:
            return 0.0
        # Count users ahead
        ahead = sum(1 for uid in self._queue
                    if self._position_map.get(uid, pos).entered_at < pos.entered_at)
        return ahead / max(self._throughput, 1)


# ─────────────────────────────────────────────
# PAYMENT SIMULATOR
# ─────────────────────────────────────────────

class PaymentSimulator:
    def __init__(self, fail_rate: float = 0.05):
        self._fail_rate = fail_rate

    def charge(self, user_id: str, amount: float,
               payment_method: str) -> Tuple[bool, str]:
        """Returns (success, transaction_id_or_error)."""
        if random.random() < self._fail_rate:
            return False, "card_declined"
        txn_id = f"txn_{uuid.uuid4().hex[:12]}"
        return True, txn_id


# ─────────────────────────────────────────────
# TICKET BOOKING SERVICE
# ─────────────────────────────────────────────

class TicketBookingService:
    def __init__(self):
        self._events:    Dict[str, Event]  = {}
        self._inventory  = InventoryManager()
        self._orders:    Dict[str, Order]  = {}
        self._payment    = PaymentSimulator()
        self._queues:    Dict[str, VirtualQueue] = {}

    def create_event(self, name: str, venue: str, start_time: float,
                     sections: Dict[str, Tuple[int, float]]
                     ) -> Event:
        """
        sections: {section_name: (seat_count, price_per_seat)}
        """
        event = Event(uuid.uuid4().hex[:10], name, venue, start_time,
                      sum(v[0] for v in sections.values()))
        self._events[event.event_id] = event

        seats = []
        for section, (count, price) in sections.items():
            for i in range(count):
                row_num = chr(ord("A") + i // 10)
                seat    = Seat(
                    seat_id  = f"{event.event_id}_{section}_{i:04d}",
                    event_id = event.event_id,
                    section  = section,
                    row      = row_num,
                    number   = i % 10 + 1,
                    price    = price,
                )
                seats.append(seat)

        self._inventory.add_seats(seats)
        self._queues[event.event_id] = VirtualQueue(throughput_per_sec=50)
        return event

    def join_queue(self, user_id: str, event_id: str) -> QueuePosition:
        q = self._queues.get(event_id, VirtualQueue())
        return q.enter(user_id)

    def select_and_hold(self, user_id: str, event_id: str,
                         quantity: int = 1) -> Optional[SeatHold]:
        """Best-available seat selection + hold."""
        q = self._queues.get(event_id)
        if q and not q.is_admitted(user_id):
            return None   # not yet admitted from queue

        available = self._inventory.available_seats(event_id)
        if len(available) < quantity:
            return None

        # Best available: take lowest-priced first (or by section)
        best = sorted(available, key=lambda s: (s.section, s.row, s.number))[:quantity]
        seat_ids = [s.seat_id for s in best]
        return self._inventory.hold_seats(user_id, event_id, seat_ids)

    def purchase(self, user_id: str, hold_id: str,
                 payment_method: str) -> Optional[Order]:
        """Complete purchase: charge card → confirm hold → create order."""
        hold = self._inventory._holds.get(hold_id)
        if not hold or hold.user_id != user_id:
            return None
        if hold.is_expired():
            return None

        # Calculate total
        total = sum(
            self._inventory._seats[sid].price
            for sid in hold.seat_ids
            if sid in self._inventory._seats
        )

        # Charge payment
        success, txn_or_err = self._payment.charge(user_id, total, payment_method)
        if not success:
            self._inventory.release_hold(hold_id)
            return None

        # Confirm hold → sold
        if not self._inventory.confirm_hold(hold_id):
            return None

        order = Order(
            order_id    = uuid.uuid4().hex[:10],
            user_id     = user_id,
            event_id    = hold.event_id,
            seat_ids    = hold.seat_ids,
            total_price = total,
            status      = OrderStatus.CONFIRMED,
            created_at  = time.time(),
            hold_id     = hold_id,
        )
        self._orders[order.order_id] = order
        return order


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_ticketing():
    print("=" * 65)
    print("SYSTEM DESIGN: TICKET BOOKING SYSTEM")
    print("=" * 65)

    random.seed(42)
    svc = TicketBookingService()

    # ── Create Event ──────────────────────────
    print("\n[1] EVENT CREATION")
    print("─" * 55)

    event = svc.create_event(
        "Rock Concert 2025",
        "Madison Square Garden",
        time.time() + 86400 * 30,
        {
            "Floor":   (50, 150.0),
            "Pit":     (100, 200.0),
            "Section A": (200, 100.0),
            "Section B": (300, 75.0),
        }
    )
    print(f"  Event: {event.name}")
    print(f"  Venue: {event.venue}")
    print(f"  Total seats: {event.total_seats}")
    counts = svc._inventory.seat_counts(event.event_id)
    for status, count in counts.items():
        print(f"    {status}: {count}")

    # ── Virtual Queue ─────────────────────────
    print("\n[2] VIRTUAL WAITING ROOM")
    print("─" * 55)

    # 200 users enter the queue simultaneously
    users = [f"user_{i:04d}" for i in range(200)]
    for uid in users:
        svc.join_queue(uid, event.event_id)

    q = svc._queues[event.event_id]
    print(f"  {len(users)} users in queue")
    for uid in users[:3]:
        pos = q._position_map[uid]
        wait = q.estimated_wait_s(uid)
        print(f"    {uid}: position={pos.position}  est_wait={wait:.1f}s")

    # Admit first 100 users
    admitted = q.admit(n=100)
    print(f"\n  Admitted {len(admitted)} users")
    print(f"  Queue remaining: {q.queue_depth()}")

    # ── Hold Seats (race condition test) ──────
    print("\n[3] CONCURRENT SEAT HOLDING (race condition prevention)")
    print("─" * 55)

    results = []
    hold_ids = []
    def try_hold(uid: str):
        hold = svc.select_and_hold(uid, event.event_id, quantity=2)
        results.append((uid, hold.hold_id if hold else None))
        if hold:
            hold_ids.append(hold.hold_id)

    # 10 admitted users try to book same seats simultaneously
    threads = [threading.Thread(target=try_hold, args=(users[i],))
               for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    held = sum(1 for _, hid in results if hid)
    print(f"  10 users tried to hold seats simultaneously")
    print(f"  Successful holds: {held}/10 (no double-booking)")
    counts = svc._inventory.seat_counts(event.event_id)
    print(f"  Seat status: {counts}")

    # ── Purchase Flow ─────────────────────────
    print("\n[4] PURCHASE FLOW")
    print("─" * 55)

    success_orders = 0
    fail_orders    = 0
    for uid, hid in results:
        if hid:
            order = svc.purchase(uid, hid, "visa_4242")
            if order:
                success_orders += 1
                if success_orders == 1:
                    print(f"  Order confirmed: {order.order_id}")
                    print(f"    Seats: {order.seat_ids[:2]}")
                    print(f"    Total: ${order.total_price:.2f}")
            else:
                fail_orders += 1

    print(f"\n  Confirmed orders: {success_orders}")
    print(f"  Failed (payment/expired): {fail_orders}")

    # ── Hold Expiry ───────────────────────────
    print("\n[5] HOLD EXPIRY SIMULATION")
    print("─" * 55)

    # Create a short-TTL hold
    svc._inventory.HOLD_TTL_S = 0.1  # 100ms for demo
    if users[50] in q._admitted:
        hold2 = svc.select_and_hold(users[50], event.event_id, quantity=1)
    else:
        q.admit(n=50)
        hold2 = svc.select_and_hold(users[50], event.event_id, quantity=1)

    if hold2:
        print(f"  Hold created: {hold2.hold_id}  expires_in=100ms")
        time.sleep(0.15)
        # Trigger expiry check
        svc._inventory._expire_holds()
        counts_after = svc._inventory.seat_counts(event.event_id)
        print(f"  After 150ms: hold expired → seat released")
        print(f"  Available seats increased: {counts_after['available']}")

    svc._inventory.HOLD_TTL_S = 600  # restore

    # ── Architecture ──────────────────────────
    print("\n[6] TICKET BOOKING ARCHITECTURE")
    print("─" * 55)

    arch = [
        ("Inventory store",   "Redis (hot): seat status with NX for atomic hold"),
        ("DB (source of truth)","Postgres: seats + orders + holds + payments"),
        ("Hold expiry",       "Redis TTL + keyspace notifications → release seat"),
        ("Payment",           "Stripe charge → webhook confirm → finalize order"),
        ("Virtual queue",     "Redis sorted set; admit N/sec based on capacity"),
        ("Concurrency",       "SELECT FOR UPDATE on seat row OR Redis SET NX"),
        ("Traffic spike",     "Queue-it / virtual waiting room; CDN static pages"),
        ("Fairness",          "FIFO queue; randomly shuffle users entering same second"),
        ("Mobile tickets",    "QR code generated on confirmation; scanned at venue"),
        ("Scalability",       "Read replicas for event browsing; master for booking"),
    ]
    for component, detail in arch:
        print(f"  {component:<22} {detail}")


if __name__ == "__main__":
    demonstrate_ticketing()
