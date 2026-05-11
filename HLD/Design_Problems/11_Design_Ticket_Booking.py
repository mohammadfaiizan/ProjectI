"""
Ticket Booking System - Core Implementation
Demonstrates: seat state machine, optimistic/pessimistic locking,
hold expiry via min-heap, concurrent booking with race condition prevention.
Standard library only.
"""

import heapq
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class SeatStatus(Enum):
    AVAILABLE = "AVAILABLE"
    HELD = "HELD"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"


class BookingStatus(Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class Seat:
    seat_id: str
    event_id: str
    section: str
    row: str
    number: int
    price: int  # minor units (cents)
    status: SeatStatus = SeatStatus.AVAILABLE
    version: int = 0  # optimistic locking version
    held_by: Optional[str] = None
    held_until: Optional[float] = None  # unix timestamp
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass
class Hold:
    hold_id: str
    user_id: str
    event_id: str
    seat_ids: List[str]
    expires_at: float  # unix timestamp
    status: str = "ACTIVE"  # ACTIVE, CONFIRMED, EXPIRED, CANCELLED


@dataclass
class Booking:
    booking_id: str
    user_id: str
    event_id: str
    seat_ids: List[str]
    total_amount: int
    status: BookingStatus = BookingStatus.PENDING


# ---------------------------------------------------------------------------
# Priority Queue for Hold Expiry (min-heap by expiry timestamp)
# ---------------------------------------------------------------------------

class HoldExpiryQueue:
    """
    Min-heap ordered by expiry time. Background thread processes expired holds.
    Entry: (expiry_timestamp, hold_id)
    """

    def __init__(self):
        self._heap: List[Tuple[float, str]] = []
        self._lock = threading.Lock()

    def push(self, hold_id: str, expiry: float):
        with self._lock:
            heapq.heappush(self._heap, (expiry, hold_id))

    def pop_expired(self, now: Optional[float] = None) -> List[str]:
        """Return hold_ids whose expiry <= now."""
        now = now or time.time()
        expired = []
        with self._lock:
            while self._heap and self._heap[0][0] <= now:
                _, hold_id = heapq.heappop(self._heap)
                expired.append(hold_id)
        return expired

    def peek_next_expiry(self) -> Optional[float]:
        with self._lock:
            return self._heap[0][0] if self._heap else None


# ---------------------------------------------------------------------------
# Distributed Lock Simulation (Redis SET NX PX in production)
# ---------------------------------------------------------------------------

class DistributedLockManager:
    """
    Simulates Redis distributed locks (SET key value NX PX milliseconds).
    In production this wraps redis-py with Redlock algorithm.
    """

    def __init__(self):
        self._locks: Dict[str, Tuple[str, float]] = {}  # key -> (owner_token, expiry)
        self._lock = threading.Lock()

    def acquire(self, key: str, owner: str, ttl_ms: int = 10000) -> bool:
        """Returns True if lock acquired, False if already held."""
        expiry = time.time() + ttl_ms / 1000
        with self._lock:
            existing = self._locks.get(key)
            if existing:
                _, exp = existing
                if time.time() < exp:
                    return False  # lock still held
            self._locks[key] = (owner, expiry)
            return True

    def release(self, key: str, owner: str) -> bool:
        """Only release if we are the owner (compare-and-delete)."""
        with self._lock:
            existing = self._locks.get(key)
            if existing and existing[0] == owner:
                del self._locks[key]
                return True
            return False

    def lock_key(self, event_id: str, seat_id: str) -> str:
        return f"seat_lock:{event_id}:{seat_id}"


# ---------------------------------------------------------------------------
# Main Ticket Booking System
# ---------------------------------------------------------------------------

class TicketBookingSystem:

    HOLD_DURATION_SECONDS = 600  # 10 minutes

    def __init__(self):
        # event_id -> {seat_id -> Seat}
        self._seats: Dict[str, Dict[str, Seat]] = defaultdict(dict)
        # hold_id -> Hold
        self._holds: Dict[str, Hold] = {}
        # booking_id -> Booking
        self._bookings: Dict[str, Booking] = {}
        # user_id -> [booking_id]
        self._user_bookings: Dict[str, List[str]] = defaultdict(list)

        self._lock_manager = DistributedLockManager()
        self._expiry_queue = HoldExpiryQueue()

        # Global lock for bookings dict (seats have per-seat locks)
        self._bookings_lock = threading.Lock()

        # Start background expiry worker
        self._running = True
        self._expiry_thread = threading.Thread(
            target=self._expiry_worker, daemon=True, name="hold-expiry-worker"
        )
        self._expiry_thread.start()

    # ------------------------------------------------------------------
    # Seed Data
    # ------------------------------------------------------------------

    def create_event(self, event_id: str, sections: Dict[str, Dict]):
        """
        sections = {
            "A": {"rows": 10, "seats_per_row": 20, "price": 10000, "category": "PREMIUM"},
            "B": {"rows": 20, "seats_per_row": 30, "price": 5000, "category": "STANDARD"},
        }
        """
        for section, cfg in sections.items():
            for r in range(1, cfg["rows"] + 1):
                for s in range(1, cfg["seats_per_row"] + 1):
                    seat = Seat(
                        seat_id=f"{event_id}_{section}{r}_{s}",
                        event_id=event_id,
                        section=section,
                        row=str(r),
                        number=s,
                        price=cfg["price"],
                    )
                    self._seats[event_id][seat.seat_id] = seat
        print(f"[EVENT] Created event {event_id} with "
              f"{len(self._seats[event_id])} seats")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_available_seats(self, event_id: str,
                            section: Optional[str] = None) -> List[Seat]:
        """Returns list of AVAILABLE seats (would hit read replica + Redis cache)."""
        self._release_expired_holds_sync(event_id)  # ensure fresh state
        seats = self._seats.get(event_id, {}).values()
        return [
            s for s in seats
            if s.status == SeatStatus.AVAILABLE
            and (section is None or s.section == section)
        ]

    def get_seat_status(self, event_id: str, seat_id: str) -> Optional[SeatStatus]:
        seat = self._seats.get(event_id, {}).get(seat_id)
        return seat.status if seat else None

    # ------------------------------------------------------------------
    # Hold (Pessimistic Locking + Distributed Lock)
    # ------------------------------------------------------------------

    def hold_seats(self, user_id: str, event_id: str,
                   seat_ids: List[str]) -> Tuple[Optional[str], str]:
        """
        Returns (hold_id, message). hold_id is None on failure.
        Uses distributed lock (Redis SET NX) + per-seat pessimistic lock.
        """
        acquired_locks = []
        lock_token = str(uuid.uuid4())

        try:
            # Step 1: Acquire distributed locks for all requested seats
            for seat_id in seat_ids:
                key = self._lock_manager.lock_key(event_id, seat_id)
                if not self._lock_manager.acquire(key, lock_token, ttl_ms=5000):
                    return None, f"Seat {seat_id} is being modified by another user"
                acquired_locks.append(key)

            # Step 2: Validate all seats are AVAILABLE (pessimistic: read under lock)
            seats_to_hold = []
            for seat_id in seat_ids:
                seat = self._seats.get(event_id, {}).get(seat_id)
                if not seat:
                    return None, f"Seat {seat_id} not found"
                with seat.lock:
                    if seat.status != SeatStatus.AVAILABLE:
                        return None, f"Seat {seat_id} is {seat.status.value}"
                    seats_to_hold.append(seat)

            # Step 3: Atomically transition all seats to HELD
            expiry = time.time() + self.HOLD_DURATION_SECONDS
            hold_id = str(uuid.uuid4())

            for seat in seats_to_hold:
                with seat.lock:
                    seat.status = SeatStatus.HELD
                    seat.held_by = user_id
                    seat.held_until = expiry
                    seat.version += 1  # bump version

            # Step 4: Record hold
            hold = Hold(
                hold_id=hold_id,
                user_id=user_id,
                event_id=event_id,
                seat_ids=seat_ids,
                expires_at=expiry,
            )
            self._holds[hold_id] = hold

            # Step 5: Push to expiry queue
            self._expiry_queue.push(hold_id, expiry)

            return hold_id, "Hold created successfully"

        finally:
            # Always release distributed locks
            for key in acquired_locks:
                self._lock_manager.release(key, lock_token)

    # ------------------------------------------------------------------
    # Confirm Booking (2-phase: hold -> payment -> confirm)
    # ------------------------------------------------------------------

    def confirm_booking(self, hold_id: str,
                        payment_token: str) -> Tuple[Optional[str], str]:
        """
        Phase 1 (already done): seats are HELD.
        Phase 2: process payment, then transition HELD -> CONFIRMED.
        Returns (booking_id, message).
        """
        hold = self._holds.get(hold_id)
        if not hold:
            return None, "Hold not found"
        if hold.status != "ACTIVE":
            return None, f"Hold is {hold.status}"
        if time.time() > hold.expires_at:
            self._expire_hold(hold_id)
            return None, "Hold has expired"

        # Simulate payment processing (PSP call)
        payment_ok, payment_msg = self._process_payment(
            hold.user_id, payment_token,
            sum(self._seats[hold.event_id][sid].price for sid in hold.seat_ids)
        )
        if not payment_ok:
            # Payment failed — release hold
            self._expire_hold(hold_id)
            return None, f"Payment failed: {payment_msg}"

        # Transition HELD -> CONFIRMED
        total = 0
        for seat_id in hold.seat_ids:
            seat = self._seats[hold.event_id][seat_id]
            with seat.lock:
                if seat.status != SeatStatus.HELD or seat.held_by != hold.user_id:
                    # Rare race: another process altered the seat
                    return None, f"Seat {seat_id} state mismatch — please retry"
                seat.status = SeatStatus.CONFIRMED
                seat.held_by = None
                seat.held_until = None
                seat.version += 1
                total += seat.price

        # Mark hold as used
        hold.status = "CONFIRMED"

        # Create booking record
        booking_id = str(uuid.uuid4())
        booking = Booking(
            booking_id=booking_id,
            user_id=hold.user_id,
            event_id=hold.event_id,
            seat_ids=hold.seat_ids[:],
            total_amount=total,
            status=BookingStatus.CONFIRMED,
        )
        with self._bookings_lock:
            self._bookings[booking_id] = booking
            self._user_bookings[hold.user_id].append(booking_id)

        print(f"[BOOKING] Confirmed booking {booking_id} for user {hold.user_id}, "
              f"seats={hold.seat_ids}, total=${total/100:.2f}")
        return booking_id, "Booking confirmed"

    # ------------------------------------------------------------------
    # Cancel Booking
    # ------------------------------------------------------------------

    def cancel_booking(self, booking_id: str) -> Tuple[bool, str]:
        """Transitions CONFIRMED -> CANCELLED and releases seats."""
        with self._bookings_lock:
            booking = self._bookings.get(booking_id)
            if not booking:
                return False, "Booking not found"
            if booking.status != BookingStatus.CONFIRMED:
                return False, f"Cannot cancel booking in status {booking.status.value}"
            booking.status = BookingStatus.CANCELLED

        for seat_id in booking.seat_ids:
            seat = self._seats[booking.event_id][seat_id]
            with seat.lock:
                seat.status = SeatStatus.AVAILABLE
                seat.held_by = None
                seat.held_until = None
                seat.version += 1

        print(f"[CANCEL] Booking {booking_id} cancelled; "
              f"seats {booking.seat_ids} returned to AVAILABLE")
        return True, "Booking cancelled; refund initiated"

    def release_hold(self, hold_id: str, user_id: str) -> Tuple[bool, str]:
        """Manually release a hold before expiry."""
        hold = self._holds.get(hold_id)
        if not hold:
            return False, "Hold not found"
        if hold.user_id != user_id:
            return False, "Unauthorized"
        if hold.status != "ACTIVE":
            return False, f"Hold already {hold.status}"
        self._expire_hold(hold_id)
        return True, "Hold released"

    # ------------------------------------------------------------------
    # Internal: Hold Expiry
    # ------------------------------------------------------------------

    def _expire_hold(self, hold_id: str):
        hold = self._holds.get(hold_id)
        if not hold or hold.status not in ("ACTIVE",):
            return
        hold.status = "EXPIRED"
        for seat_id in hold.seat_ids:
            seat = self._seats.get(hold.event_id, {}).get(seat_id)
            if seat:
                with seat.lock:
                    if seat.status == SeatStatus.HELD and seat.held_by == hold.user_id:
                        seat.status = SeatStatus.AVAILABLE
                        seat.held_by = None
                        seat.held_until = None
                        seat.version += 1
        print(f"[EXPIRY] Hold {hold_id} expired; seats {hold.seat_ids} released")

    def _release_expired_holds_sync(self, event_id: str):
        """Synchronous scan for a specific event — used before availability reads."""
        now = time.time()
        for hold in list(self._holds.values()):
            if hold.event_id == event_id and hold.status == "ACTIVE" and now > hold.expires_at:
                self._expire_hold(hold.hold_id)

    def _expiry_worker(self):
        """Background thread: drains expired holds from priority queue."""
        while self._running:
            expired = self._expiry_queue.pop_expired()
            for hold_id in expired:
                self._expire_hold(hold_id)
            # Sleep until next expiry or 5 seconds, whichever is sooner
            next_exp = self._expiry_queue.peek_next_expiry()
            wait = min(5.0, max(0.1, (next_exp - time.time()) if next_exp else 5.0))
            time.sleep(wait)

    def _process_payment(self, user_id: str, token: str,
                         amount: int) -> Tuple[bool, str]:
        """Simulates PSP call. Token 'FAIL' triggers failure for testing."""
        if token == "FAIL":
            return False, "Card declined"
        time.sleep(0.01)  # simulate ~10ms PSP latency
        return True, "OK"

    def shutdown(self):
        self._running = False

    # ------------------------------------------------------------------
    # Optimistic Locking Demo
    # ------------------------------------------------------------------

    def update_seat_price_optimistic(self, event_id: str, seat_id: str,
                                     new_price: int, expected_version: int) -> bool:
        """
        CAS-style update: only update if version matches expected_version.
        Simulates optimistic locking (used for low-contention metadata updates).
        """
        seat = self._seats.get(event_id, {}).get(seat_id)
        if not seat:
            return False
        with seat.lock:
            if seat.version != expected_version:
                print(f"[OPTIMISTIC] Version mismatch on {seat_id}: "
                      f"expected {expected_version}, got {seat.version} — retry")
                return False
            seat.price = new_price
            seat.version += 1
            return True

    def stats(self, event_id: str) -> Dict:
        seats = self._seats.get(event_id, {}).values()
        by_status = defaultdict(int)
        for s in seats:
            by_status[s.status.value] += 1
        return dict(by_status)


# ---------------------------------------------------------------------------
# Concurrent Booking Simulation
# ---------------------------------------------------------------------------

def simulate_concurrent_booking():
    """
    Simulates 10 threads all trying to book the same 2 seats simultaneously.
    Expected: exactly 1 succeeds, 9 get 'seat already held' error.
    """
    system = TicketBookingSystem()
    system.create_event("EVT001", {
        "A": {"rows": 2, "seats_per_row": 5, "price": 5000, "category": "PREMIUM"},
    })

    target_seats = ["EVT001_A1_1", "EVT001_A1_2"]
    results = []
    results_lock = threading.Lock()

    def try_book(user_num: int):
        user_id = f"user_{user_num:03d}"
        hold_id, msg = system.hold_seats(user_id, "EVT001", target_seats)
        with results_lock:
            results.append((user_id, hold_id, msg))

    threads = [threading.Thread(target=try_book, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = [(uid, hid) for uid, hid, _ in results if hid is not None]
    failures = [(uid, msg) for uid, hid, msg in results if hid is None]

    print("\n=== Concurrent Booking Simulation ===")
    print(f"Total attempts: {len(results)}")
    print(f"Successful holds: {len(successes)} (expected: 1)")
    print(f"Failed attempts: {len(failures)}")
    for uid, hid in successes:
        print(f"  SUCCESS: {uid} got hold {hid}")
    for uid, msg in failures[:3]:  # show first 3
        print(f"  FAILED:  {uid} — {msg}")

    system.shutdown()
    return system, successes


def simulate_hold_expiry():
    """Demonstrates the hold expiry lifecycle with a short TTL."""
    system = TicketBookingSystem()
    system.HOLD_DURATION_SECONDS = 2  # 2 second hold for demo
    system.create_event("EVT002", {
        "B": {"rows": 1, "seats_per_row": 3, "price": 2000, "category": "STANDARD"},
    })

    print("\n=== Hold Expiry Simulation ===")
    hold_id, msg = system.hold_seats("user_001", "EVT002", ["EVT002_B1_1"])
    print(f"Hold created: {hold_id} | {msg}")
    print(f"Available seats before expiry: {len(system.get_available_seats('EVT002'))}")

    print("Waiting 3 seconds for hold to expire...")
    time.sleep(3)

    available = system.get_available_seats("EVT002")
    print(f"Available seats after expiry: {len(available)} (expected: 3)")
    system.shutdown()


def full_booking_flow():
    """End-to-end: hold -> confirm -> cancel."""
    system = TicketBookingSystem()
    system.create_event("EVT003", {
        "VIP": {"rows": 1, "seats_per_row": 5, "price": 15000, "category": "VIP"},
    })

    print("\n=== Full Booking Flow ===")
    print(f"Stats before: {system.stats('EVT003')}")

    # 1. Hold seats
    hold_id, msg = system.hold_seats("alice", "EVT003",
                                     ["EVT003_VIP1_1", "EVT003_VIP1_2"])
    print(f"Hold: {msg} | hold_id={hold_id}")
    print(f"Stats after hold: {system.stats('EVT003')}")

    # 2. Another user tries same seats (should fail)
    hold2, msg2 = system.hold_seats("bob", "EVT003",
                                    ["EVT003_VIP1_1"])
    print(f"Bob tries same seat: {msg2}")

    # 3. Confirm booking
    booking_id, msg3 = system.confirm_booking(hold_id, "tok_visa_4242")
    print(f"Confirm: {msg3} | booking_id={booking_id}")
    print(f"Stats after confirm: {system.stats('EVT003')}")

    # 4. Cancel booking
    ok, msg4 = system.cancel_booking(booking_id)
    print(f"Cancel: {msg4}")
    print(f"Stats after cancel: {system.stats('EVT003')}")

    # 5. Payment failure test
    hold3, _ = system.hold_seats("charlie", "EVT003",
                                 ["EVT003_VIP1_1"])
    b3, msg5 = system.confirm_booking(hold3, "FAIL")
    print(f"Payment failure test: {msg5}")
    print(f"Stats after failed payment: {system.stats('EVT003')}")

    # 6. Optimistic locking demo
    seat = system._seats["EVT003"]["EVT003_VIP1_3"]
    current_ver = seat.version
    ok1 = system.update_seat_price_optimistic("EVT003", "EVT003_VIP1_3", 20000, current_ver)
    ok2 = system.update_seat_price_optimistic("EVT003", "EVT003_VIP1_3", 25000, current_ver)
    print(f"Optimistic update 1 (correct version): {ok1}")
    print(f"Optimistic update 2 (stale version): {ok2} (expected False)")

    system.shutdown()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    full_booking_flow()
    simulate_concurrent_booking()

    # Minimal hold expiry demo (avoid long sleep in full run)
    system = TicketBookingSystem()
    system.HOLD_DURATION_SECONDS = 1
    system.create_event("EVT002", {
        "B": {"rows": 1, "seats_per_row": 3, "price": 2000, "category": "STANDARD"}
    })
    print("\n=== Hold Expiry Simulation ===")
    hold_id, msg = system.hold_seats("user_001", "EVT002", ["EVT002_B1_1"])
    print(f"Hold created: {msg}")
    print(f"Available before expiry: {len(system.get_available_seats('EVT002'))}/3")
    time.sleep(2)
    print(f"Available after expiry:  {len(system.get_available_seats('EVT002'))}/3")
    system.shutdown()
