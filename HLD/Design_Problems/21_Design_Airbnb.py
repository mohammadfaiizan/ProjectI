"""
Airbnb System Design - Python Implementation
Demonstrates: GeoSearch, AvailabilityCalendar, BookingManager,
              DynamicPricingEngine, ReviewSystem (mutual reveal), Messaging
No external dependencies - standard library only.
"""

import math
import uuid
import hashlib
from datetime import date, timedelta
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class BookingStatus(Enum):
    CONFIRMED  = "confirmed"
    CANCELLED  = "cancelled"
    COMPLETED  = "completed"
    CHECKED_IN = "checked_in"

class ReviewStatus(Enum):
    PENDING           = "pending"
    GUEST_SUBMITTED   = "guest_submitted"
    HOST_SUBMITTED    = "host_submitted"
    BOTH_SUBMITTED    = "both_submitted"
    REVEALED          = "revealed"

@dataclass
class Listing:
    id: str
    host_id: str
    title: str
    lat: float
    lon: float
    base_price: float
    max_guests: int
    amenities: list = field(default_factory=list)
    avg_rating: float = 0.0
    review_count: int = 0
    instant_book: bool = True

@dataclass
class Booking:
    id: str
    listing_id: str
    guest_id: str
    check_in: date
    check_out: date
    guests: int
    total_price: float
    status: BookingStatus = BookingStatus.CONFIRMED

@dataclass
class Review:
    id: str
    booking_id: str
    reviewer_id: str
    reviewee_id: str
    listing_id: str
    rating: int
    text: str
    review_type: str        # 'guest_to_host' or 'host_to_guest'
    revealed: bool = False

@dataclass
class Message:
    id: str
    thread_id: str
    sender_id: str
    recipient_id: str
    body: str
    booking_id: Optional[str] = None


# ---------------------------------------------------------------------------
# 1. GeoSearch — bounding box + haversine distance filtering
# ---------------------------------------------------------------------------

class GeoSearch:
    """
    Simple in-memory geo search using a bounding-box pre-filter
    followed by exact haversine distance computation.
    In production: Elasticsearch geo_point + geo_distance query.
    """
    EARTH_RADIUS_KM = 6371.0

    def __init__(self):
        self._listings: dict[str, Listing] = {}

    def add_listing(self, listing: Listing) -> None:
        self._listings[listing.id] = listing

    def remove_listing(self, listing_id: str) -> None:
        self._listings.pop(listing_id, None)

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Returns distance in kilometres between two lat/lon points."""
        r = GeoSearch.EARTH_RADIUS_KM
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lam = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    def search(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        price_min: float = 0,
        price_max: float = float("inf"),
        min_guests: int = 1,
        required_amenities: list = None,
    ) -> list[tuple[Listing, float]]:
        """
        Returns list of (Listing, distance_km) sorted by distance.
        Bounding box pre-filter avoids full haversine scan.
        """
        required_amenities = required_amenities or []
        # Approximate degree delta for bounding box
        lat_delta = radius_km / self.EARTH_RADIUS_KM * (180 / math.pi)
        lon_delta = lat_delta / max(math.cos(math.radians(center_lat)), 1e-9)

        results = []
        for listing in self._listings.values():
            # Bounding box pre-filter (cheap)
            if not (center_lat - lat_delta <= listing.lat <= center_lat + lat_delta):
                continue
            if not (center_lon - lon_delta <= listing.lon <= center_lon + lon_delta):
                continue
            # Price and guest filter
            if not (price_min <= listing.base_price <= price_max):
                continue
            if listing.max_guests < min_guests:
                continue
            # Amenity filter
            if required_amenities:
                if not all(a in listing.amenities for a in required_amenities):
                    continue
            # Exact distance check
            dist = self.haversine(center_lat, center_lon, listing.lat, listing.lon)
            if dist <= radius_km:
                results.append((listing, round(dist, 2)))

        results.sort(key=lambda x: x[1])
        return results


# ---------------------------------------------------------------------------
# 2. AvailabilityCalendar — date-range blocking
# ---------------------------------------------------------------------------

class AvailabilityCalendar:
    """
    Tracks availability per listing using a set of blocked dates.
    In production: availability table in PostgreSQL with row-level locking.
    """

    def __init__(self):
        # listing_id -> set of blocked dates
        self._blocked: dict[str, set[date]] = defaultdict(set)

    def _date_range(self, check_in: date, check_out: date) -> list[date]:
        """Returns all dates in [check_in, check_out) — check_out is departure day."""
        days = (check_out - check_in).days
        return [check_in + timedelta(days=i) for i in range(days)]

    def is_available(self, listing_id: str, check_in: date, check_out: date) -> bool:
        blocked = self._blocked[listing_id]
        return not any(d in blocked for d in self._date_range(check_in, check_out))

    def block_dates(self, listing_id: str, check_in: date, check_out: date) -> None:
        """Mark dates as booked — called inside booking transaction."""
        for d in self._date_range(check_in, check_out):
            self._blocked[listing_id].add(d)

    def unblock_dates(self, listing_id: str, check_in: date, check_out: date) -> None:
        """Restore dates on cancellation."""
        for d in self._date_range(check_in, check_out):
            self._blocked[listing_id].discard(d)

    def get_blocked_dates(self, listing_id: str) -> list[date]:
        return sorted(self._blocked[listing_id])


# ---------------------------------------------------------------------------
# 3. DynamicPricingEngine
# ---------------------------------------------------------------------------

class DynamicPricingEngine:
    """
    Computes final price = base_price × seasonal_factor × demand_multiplier
                         × length_of_stay_discount × (1 + service_fee_rate)
    """
    SERVICE_FEE_RATE   = 0.14   # 14% guest service fee
    CLEANING_FEE_BASE  = 50.0

    SEASONAL_FACTORS = {
        # month -> factor
        1: 0.85, 2: 0.85, 3: 1.00, 4: 1.10,
        5: 1.15, 6: 1.30, 7: 1.50, 8: 1.50,
        9: 1.10, 10: 1.00, 11: 0.90, 12: 1.20,
    }

    def __init__(self):
        # listing_id -> recent booking count (demand signal)
        self._recent_bookings: dict[str, int] = defaultdict(int)
        self._baseline_bookings: float = 5.0  # average bookings per week area-wide

    def record_booking(self, listing_id: str) -> None:
        self._recent_bookings[listing_id] += 1

    def _demand_multiplier(self, listing_id: str) -> float:
        ratio = self._recent_bookings[listing_id] / self._baseline_bookings
        # Clamp between 0.8 and 2.0
        return max(0.8, min(2.0, 0.8 + ratio * 0.4))

    def _length_discount(self, nights: int) -> float:
        if nights >= 28:
            return 0.75
        if nights >= 7:
            return 0.90
        return 1.0

    def calculate_price(
        self, listing: Listing, check_in: date, check_out: date
    ) -> dict:
        nights = (check_out - check_in).days
        seasonal = self.SEASONAL_FACTORS[check_in.month]
        demand   = self._demand_multiplier(listing.id)
        length   = self._length_discount(nights)

        nightly_price   = listing.base_price * seasonal * demand * length
        subtotal        = nightly_price * nights
        cleaning_fee    = self.CLEANING_FEE_BASE
        service_fee     = round(subtotal * self.SERVICE_FEE_RATE, 2)
        total           = round(subtotal + cleaning_fee + service_fee, 2)

        return {
            "nightly_price":  round(nightly_price, 2),
            "nights":         nights,
            "subtotal":       round(subtotal, 2),
            "cleaning_fee":   cleaning_fee,
            "service_fee":    service_fee,
            "total":          total,
            "seasonal_factor": seasonal,
            "demand_multiplier": round(demand, 2),
        }


# ---------------------------------------------------------------------------
# 4. BookingManager — conflict prevention + state management
# ---------------------------------------------------------------------------

class BookingManager:
    """
    Manages bookings with atomic conflict detection.
    In production: SELECT FOR UPDATE inside a PostgreSQL transaction.
    """

    def __init__(self, calendar: AvailabilityCalendar, pricing: DynamicPricingEngine):
        self._bookings: dict[str, Booking] = {}
        self._calendar  = calendar
        self._pricing   = pricing

    def create_booking(
        self,
        listing: Listing,
        guest_id: str,
        check_in: date,
        check_out: date,
        guests: int,
    ) -> Booking:
        if check_out <= check_in:
            raise ValueError("check_out must be after check_in")
        if guests > listing.max_guests:
            raise ValueError(f"Listing supports max {listing.max_guests} guests")

        # Atomic availability check + block (simulates SELECT FOR UPDATE + UPDATE)
        if not self._calendar.is_available(listing.id, check_in, check_out):
            raise RuntimeError("Dates not available — conflict detected")

        price_breakdown = self._pricing.calculate_price(listing, check_in, check_out)
        booking = Booking(
            id          = str(uuid.uuid4())[:8],
            listing_id  = listing.id,
            guest_id    = guest_id,
            check_in    = check_in,
            check_out   = check_out,
            guests      = guests,
            total_price = price_breakdown["total"],
        )
        # Block dates AFTER booking object created (simulates DB transaction)
        self._calendar.block_dates(listing.id, check_in, check_out)
        self._bookings[booking.id] = booking
        self._pricing.record_booking(listing.id)
        return booking

    def cancel_booking(self, booking_id: str, requestor_id: str) -> Booking:
        booking = self._bookings.get(booking_id)
        if not booking:
            raise KeyError(f"Booking {booking_id} not found")
        if booking.status != BookingStatus.CONFIRMED:
            raise ValueError(f"Cannot cancel booking in status {booking.status}")
        booking.status = BookingStatus.CANCELLED
        self._calendar.unblock_dates(booking.listing_id, booking.check_in, booking.check_out)
        return booking

    def get_booking(self, booking_id: str) -> Optional[Booking]:
        return self._bookings.get(booking_id)

    def get_user_bookings(self, guest_id: str) -> list[Booking]:
        return [b for b in self._bookings.values() if b.guest_id == guest_id]


# ---------------------------------------------------------------------------
# 5. ReviewSystem — mutual reveal pattern
# ---------------------------------------------------------------------------

class ReviewSystem:
    """
    Both parties submit reviews independently.
    Reviews are hidden until BOTH submit OR 14-day deadline passes.
    """

    def __init__(self):
        self._reviews: dict[str, Review]          = {}
        # booking_id -> {"guest_review": id, "host_review": id}
        self._booking_reviews: dict[str, dict]    = defaultdict(dict)
        self._listing_ratings: dict[str, list]    = defaultdict(list)

    def submit_review(
        self,
        booking: Booking,
        reviewer_id: str,
        reviewee_id: str,
        listing_id: str,
        rating: int,
        text: str,
        reviewer_is_guest: bool,
    ) -> Review:
        if not (1 <= rating <= 5):
            raise ValueError("Rating must be between 1 and 5")

        review_type = "guest_to_host" if reviewer_is_guest else "host_to_guest"
        slot_key    = "guest_review" if reviewer_is_guest else "host_review"

        booking_slots = self._booking_reviews[booking.id]
        if slot_key in booking_slots:
            raise ValueError(f"{review_type} already submitted for this booking")

        review = Review(
            id          = str(uuid.uuid4())[:8],
            booking_id  = booking.id,
            reviewer_id = reviewer_id,
            reviewee_id = reviewee_id,
            listing_id  = listing_id,
            rating      = rating,
            text        = text,
            review_type = review_type,
            revealed    = False,
        )
        self._reviews[review.id] = review
        booking_slots[slot_key] = review.id

        # Mutual reveal: if both have now submitted, reveal both
        if "guest_review" in booking_slots and "host_review" in booking_slots:
            self._reveal_booking_reviews(booking.id)

        return review

    def _reveal_booking_reviews(self, booking_id: str) -> None:
        slots = self._booking_reviews[booking_id]
        for review_id in slots.values():
            review = self._reviews[review_id]
            review.revealed = True
            self._listing_ratings[review.listing_id].append(review.rating)

    def get_listing_reviews(self, listing_id: str) -> list[Review]:
        return [
            r for r in self._reviews.values()
            if r.listing_id == listing_id and r.revealed and r.review_type == "guest_to_host"
        ]

    def get_listing_avg_rating(self, listing_id: str) -> float:
        ratings = self._listing_ratings[listing_id]
        return round(sum(ratings) / len(ratings), 2) if ratings else 0.0


# ---------------------------------------------------------------------------
# 6. MessagingSystem
# ---------------------------------------------------------------------------

class MessagingSystem:
    def __init__(self):
        self._messages: list[Message] = []
        # (user_a, user_b) -> thread_id
        self._threads:  dict[tuple, str] = {}

    def _get_or_create_thread(self, user_a: str, user_b: str) -> str:
        key = tuple(sorted([user_a, user_b]))
        if key not in self._threads:
            self._threads[key] = hashlib.md5(f"{key}".encode()).hexdigest()[:8]
        return self._threads[key]

    def send_message(
        self, sender_id: str, recipient_id: str, body: str, booking_id: str = None
    ) -> Message:
        thread_id = self._get_or_create_thread(sender_id, recipient_id)
        msg = Message(
            id           = str(uuid.uuid4())[:8],
            thread_id    = thread_id,
            sender_id    = sender_id,
            recipient_id = recipient_id,
            body         = body,
            booking_id   = booking_id,
        )
        self._messages.append(msg)
        return msg

    def get_thread(self, user_a: str, user_b: str) -> list[Message]:
        thread_id = self._get_or_create_thread(user_a, user_b)
        return [m for m in self._messages if m.thread_id == thread_id]


# ---------------------------------------------------------------------------
# 7. AirbnbSystem — Facade tying everything together
# ---------------------------------------------------------------------------

class AirbnbSystem:
    def __init__(self):
        self._geo_search  = GeoSearch()
        self._calendar    = AvailabilityCalendar()
        self._pricing     = DynamicPricingEngine()
        self._booking_mgr = BookingManager(self._calendar, self._pricing)
        self._reviews     = ReviewSystem()
        self._messaging   = MessagingSystem()
        self._listings: dict[str, Listing] = {}

    # -- Listing Management --------------------------------------------------

    def add_listing(self, listing: Listing) -> None:
        self._listings[listing.id] = listing
        self._geo_search.add_listing(listing)

    # -- Search --------------------------------------------------------------

    def search_listings(
        self,
        lat: float, lon: float,
        check_in: date, check_out: date,
        guests: int = 1,
        radius_km: float = 20,
        price_min: float = 0,
        price_max: float = float("inf"),
        amenities: list = None,
    ) -> list[dict]:
        geo_results = self._geo_search.search(
            lat, lon, radius_km, price_min, price_max, guests, amenities or []
        )
        output = []
        for listing, dist_km in geo_results:
            if self._calendar.is_available(listing.id, check_in, check_out):
                price_info = self._pricing.calculate_price(listing, check_in, check_out)
                output.append({
                    "listing_id":  listing.id,
                    "title":       listing.title,
                    "distance_km": dist_km,
                    "nightly":     price_info["nightly_price"],
                    "total":       price_info["total"],
                    "rating":      listing.avg_rating,
                    "amenities":   listing.amenities,
                })
        return output

    # -- Availability --------------------------------------------------------

    def check_availability(
        self, listing_id: str, check_in: date, check_out: date
    ) -> bool:
        return self._calendar.is_available(listing_id, check_in, check_out)

    # -- Booking -------------------------------------------------------------

    def create_booking(
        self,
        listing_id: str,
        guest_id: str,
        check_in: date,
        check_out: date,
        guests: int = 1,
    ) -> Booking:
        listing = self._listings[listing_id]
        return self._booking_mgr.create_booking(listing, guest_id, check_in, check_out, guests)

    def cancel_booking(self, booking_id: str, requestor_id: str) -> Booking:
        return self._booking_mgr.cancel_booking(booking_id, requestor_id)

    # -- Reviews -------------------------------------------------------------

    def submit_review(
        self,
        booking_id: str,
        reviewer_id: str,
        reviewee_id: str,
        listing_id: str,
        rating: int,
        text: str,
        reviewer_is_guest: bool,
    ) -> Review:
        booking = self._booking_mgr.get_booking(booking_id)
        if not booking:
            raise KeyError(f"Booking {booking_id} not found")
        return self._reviews.submit_review(
            booking, reviewer_id, reviewee_id, listing_id, rating, text, reviewer_is_guest
        )

    # -- Messaging -----------------------------------------------------------

    def send_message(
        self, sender_id: str, recipient_id: str, body: str, booking_id: str = None
    ) -> Message:
        return self._messaging.send_message(sender_id, recipient_id, body, booking_id)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    system = AirbnbSystem()

    # Create listings
    l1 = Listing("L1", "host_1", "Cozy Downtown Flat",
                 lat=37.7749, lon=-122.4194, base_price=120.0,
                 max_guests=4, amenities=["wifi", "kitchen", "parking"])
    l2 = Listing("L2", "host_2", "Ocean View Cottage",
                 lat=37.7800, lon=-122.4100, base_price=200.0,
                 max_guests=2, amenities=["wifi", "pool"])
    l3 = Listing("L3", "host_3", "Far Away Farm",
                 lat=37.9000, lon=-122.5000, base_price=80.0,
                 max_guests=6, amenities=["kitchen"])

    for l in [l1, l2, l3]:
        system.add_listing(l)

    # Search
    print("=== Search Results ===")
    results = system.search_listings(
        lat=37.7749, lon=-122.4194,
        check_in=date(2025, 7, 10), check_out=date(2025, 7, 15),
        guests=2, radius_km=15
    )
    for r in results:
        print(f"  {r['title']}: ${r['nightly']}/night | total ${r['total']} | {r['distance_km']}km")

    # Create a booking
    print("\n=== Booking ===")
    b1 = system.create_booking("L1", "guest_1", date(2025, 7, 10), date(2025, 7, 15), guests=2)
    print(f"  Booking {b1.id}: {b1.check_in} to {b1.check_out} | ${b1.total_price} | {b1.status.value}")

    # Try to double-book same dates (should fail)
    print("\n=== Double Booking Attempt ===")
    try:
        system.create_booking("L1", "guest_2", date(2025, 7, 12), date(2025, 7, 14), guests=1)
    except RuntimeError as e:
        print(f"  Caught: {e}")

    # Submit mutual reviews
    print("\n=== Reviews (Mutual Reveal) ===")
    r1 = system.submit_review(b1.id, "guest_1", "host_1", "L1", 5, "Amazing stay!", reviewer_is_guest=True)
    print(f"  Guest review submitted — revealed: {r1.revealed}")
    r2 = system.submit_review(b1.id, "host_1", "guest_1", "L1", 4, "Great guest.", reviewer_is_guest=False)
    print(f"  Host review submitted  — revealed: {r2.revealed}")
    print(f"  Both revealed now: guest={r1.revealed}, host={r2.revealed}")
    print(f"  Listing avg rating: {system._reviews.get_listing_avg_rating('L1')}")

    # Messaging
    print("\n=== Messaging ===")
    system.send_message("guest_1", "host_1", "Hi! Is early check-in possible?", booking_id=b1.id)
    system.send_message("host_1", "guest_1", "Yes, 11am works fine!")
    thread = system._messaging.get_thread("guest_1", "host_1")
    for msg in thread:
        print(f"  [{msg.sender_id}]: {msg.body}")

    # Cancel booking
    print("\n=== Cancellation ===")
    b2 = system.create_booking("L2", "guest_3", date(2025, 8, 1), date(2025, 8, 5), guests=2)
    print(f"  Created booking {b2.id}")
    cancelled = system.cancel_booking(b2.id, "guest_3")
    print(f"  Cancelled: {cancelled.status.value}")
    # Dates should be free again
    avail = system.check_availability("L2", date(2025, 8, 1), date(2025, 8, 5))
    print(f"  L2 available again after cancel: {avail}")
