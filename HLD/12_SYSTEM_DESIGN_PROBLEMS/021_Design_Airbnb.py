"""
AIRBNB — Home Rental Marketplace
==================================

FUNCTIONAL REQUIREMENTS:
- Hosts list properties with photos, amenities, pricing rules
- Guests search by location, dates, guests, filters (price, type, amenities)
- Real-time availability calendar (no double bookings)
- Booking: instant book or request-to-book with host approval
- Reviews: bidirectional (host reviews guest, guest reviews property)
- Messaging between host and guest

NON-FUNCTIONAL REQUIREMENTS:
- 150 M guests, 4 M hosts, 7 M listings
- 500 K bookings/day
- Search: < 200 ms p99
- Calendar availability: strongly consistent (no overbooking)
- Global availability (EU, Asia-Pacific, Americas)

ARCHITECTURE:
  ┌──────────┐     ┌─────────────┐     ┌──────────────────┐
  │ Client   │────▶│  API GW     │────▶│  Search Service  │──▶ Elasticsearch
  └──────────┘     └─────────────┘     └──────────────────┘
                         │             ┌──────────────────┐
                         ├────────────▶│  Listing Service │──▶ PostgreSQL
                         │             └──────────────────┘
                         │             ┌──────────────────┐
                         ├────────────▶│  Calendar Service│──▶ DynamoDB (strong)
                         │             └──────────────────┘
                         │             ┌──────────────────┐
                         └────────────▶│  Booking Service │──▶ PostgreSQL
                                       └──────────────────┘

KEY DESIGN DECISIONS:
1. AVAILABILITY CALENDAR — each date stored as a separate record in DynamoDB.
   Partition key: listing_id.  Sort key: date.
   Booking = conditional write: update N dates from OPEN → BLOCKED atomically.
   DynamoDB transactions (up to 25 items) prevent double booking.

2. SEARCH — Elasticsearch with geo_point field for location.
   Query: geo_distance filter + date availability filter + facets.
   Availability filter: Elasticsearch cannot directly query DynamoDB, so
   availability index is maintained (async update on every booking/cancellation).

3. PRICING ENGINE — base price + seasonal adjustments + length-of-stay discounts
   + dynamic demand-based pricing.  Hosts set base price + rules.

4. REVIEW SYSTEM — dual review: host→guest and guest→listing.
   Reviews only visible after BOTH parties submit OR 14-day window expires.
   Prevents strategic reviewing (gaming the system).

5. DISTRIBUTED SEARCH — searches return listing summaries from Elasticsearch;
   full listing details fetched from PostgreSQL.

6. INSTANT BOOK vs REQUEST-TO-BOOK:
   - Instant: calendar locked immediately on payment.
   - Request: host has 24h to accept; calendar held (soft lock) during this time.
"""

from __future__ import annotations
import time
import uuid
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import defaultdict
from datetime import date, timedelta
import threading


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class PropertyType(Enum):
    ENTIRE_HOME = "entire_home"
    PRIVATE_ROOM = "private_room"
    SHARED_ROOM = "shared_room"
    HOTEL_ROOM = "hotel_room"


class BookingStatus(Enum):
    PENDING = "pending"          # Request sent, awaiting host approval
    CONFIRMED = "confirmed"      # Approved and payment captured
    CANCELLED_GUEST = "cancelled_guest"
    CANCELLED_HOST = "cancelled_host"
    COMPLETED = "completed"


@dataclass
class Location:
    lat: float
    lng: float
    city: str
    country: str
    neighbourhood: str = ""

    def distance_km(self, other: "Location") -> float:
        R = 6371
        dlat = math.radians(other.lat - self.lat)
        dlng = math.radians(other.lng - self.lng)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(self.lat)) *
             math.cos(math.radians(other.lat)) *
             math.sin(dlng / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class PricingRule:
    base_price_cents: int
    weekend_premium_pct: int = 20    # Friday/Saturday = +20%
    weekly_discount_pct: int = 10    # 7+ nights = -10%
    monthly_discount_pct: int = 25   # 28+ nights = -25%
    cleaning_fee_cents: int = 5000
    min_nights: int = 1
    max_nights: int = 365


@dataclass
class Listing:
    listing_id: str
    host_id: str
    title: str
    description: str
    location: Location
    property_type: PropertyType
    amenities: Set[str]
    max_guests: int
    bedrooms: int
    bathrooms: float
    pricing: PricingRule
    instant_book: bool = True
    rating: float = 0.0
    review_count: int = 0
    photos: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Availability Calendar — DynamoDB simulation with strong consistency
# ---------------------------------------------------------------------------

class DateStatus(Enum):
    OPEN = "open"
    BLOCKED = "blocked"       # Booked by guest or blocked by host
    PENDING = "pending"       # Soft-locked for pending request


@dataclass
class CalendarEntry:
    listing_id: str
    date: str           # ISO format "YYYY-MM-DD"
    status: DateStatus
    booking_id: Optional[str] = None
    version: int = 0    # For optimistic locking


class AvailabilityCalendar:
    """
    DynamoDB-backed availability calendar.
    Atomic multi-date locking via DynamoDB transactions.
    """

    def __init__(self):
        self._calendar: Dict[str, Dict[str, CalendarEntry]] = defaultdict(dict)
        self._lock = threading.Lock()

    def _date_range(self, check_in: str, check_out: str) -> List[str]:
        """Generate list of dates from check_in (inclusive) to check_out (exclusive)."""
        start = date.fromisoformat(check_in)
        end = date.fromisoformat(check_out)
        dates = []
        current = start
        while current < end:
            dates.append(current.isoformat())
            current += timedelta(days=1)
        return dates

    def is_available(self, listing_id: str, check_in: str, check_out: str) -> bool:
        dates = self._date_range(check_in, check_out)
        cal = self._calendar.get(listing_id, {})
        for d in dates:
            entry = cal.get(d)
            if entry and entry.status != DateStatus.OPEN:
                return False
        return True

    def soft_lock(self, listing_id: str, check_in: str,
                  check_out: str, booking_id: str) -> bool:
        """Soft lock for pending request (24-hour hold)."""
        with self._lock:
            if not self.is_available(listing_id, check_in, check_out):
                return False
            dates = self._date_range(check_in, check_out)
            for d in dates:
                self._calendar[listing_id][d] = CalendarEntry(
                    listing_id, d, DateStatus.PENDING, booking_id
                )
            return True

    def confirm_lock(self, listing_id: str, check_in: str,
                     check_out: str, booking_id: str) -> bool:
        """Upgrade soft lock to hard lock (CONFIRMED)."""
        with self._lock:
            dates = self._date_range(check_in, check_out)
            cal = self._calendar.get(listing_id, {})
            # Verify all dates are still pending for this booking
            for d in dates:
                entry = cal.get(d)
                if not entry or entry.status != DateStatus.PENDING or \
                        entry.booking_id != booking_id:
                    return False
            for d in dates:
                self._calendar[listing_id][d].status = DateStatus.BLOCKED
                self._calendar[listing_id][d].version += 1
            return True

    def instant_lock(self, listing_id: str, check_in: str,
                     check_out: str, booking_id: str) -> bool:
        """Atomic lock for instant booking — no pending phase."""
        with self._lock:
            if not self.is_available(listing_id, check_in, check_out):
                return False
            dates = self._date_range(check_in, check_out)
            for d in dates:
                self._calendar[listing_id][d] = CalendarEntry(
                    listing_id, d, DateStatus.BLOCKED, booking_id
                )
            return True

    def release(self, listing_id: str, check_in: str, check_out: str) -> None:
        """Release calendar on cancellation."""
        with self._lock:
            dates = self._date_range(check_in, check_out)
            cal = self._calendar.get(listing_id, {})
            for d in dates:
                cal.pop(d, None)

    def get_availability(self, listing_id: str, month: str) -> Dict[str, str]:
        """Return status of all dates in a month (YYYY-MM)."""
        cal = self._calendar.get(listing_id, {})
        return {
            d: entry.status.value
            for d, entry in cal.items()
            if d.startswith(month)
        }


# ---------------------------------------------------------------------------
# Pricing Engine
# ---------------------------------------------------------------------------

class PricingEngine:
    def calculate(self, listing: Listing, check_in: str, check_out: str,
                  guests: int) -> Dict[str, int]:
        """Returns itemised price breakdown in cents."""
        start = date.fromisoformat(check_in)
        end = date.fromisoformat(check_out)
        nights = (end - start).days

        if nights < listing.pricing.min_nights:
            return {}

        nightly_total = 0
        current = start
        while current < end:
            price = listing.pricing.base_price_cents
            if current.weekday() in (4, 5):  # Friday, Saturday
                price = int(price * (1 + listing.pricing.weekend_premium_pct / 100))
            nightly_total += price
            current += timedelta(days=1)

        # Length-of-stay discount
        discount_pct = 0
        if nights >= 28:
            discount_pct = listing.pricing.monthly_discount_pct
        elif nights >= 7:
            discount_pct = listing.pricing.weekly_discount_pct

        discount = int(nightly_total * discount_pct / 100)
        subtotal = nightly_total - discount
        cleaning = listing.pricing.cleaning_fee_cents
        service_fee = int(subtotal * 0.14)  # 14% Airbnb service fee
        total = subtotal + cleaning + service_fee

        return {
            "nights": nights,
            "nightly_avg_cents": nightly_total // nights,
            "nightly_total_cents": nightly_total,
            "discount_pct": discount_pct,
            "discount_cents": discount,
            "cleaning_fee_cents": cleaning,
            "service_fee_cents": service_fee,
            "total_cents": total,
        }


# ---------------------------------------------------------------------------
# Search Service
# ---------------------------------------------------------------------------

@dataclass
class SearchQuery:
    location: Location
    check_in: str
    check_out: str
    guests: int
    radius_km: float = 50.0
    min_price_cents: int = 0
    max_price_cents: int = 100_000
    amenities: Set[str] = field(default_factory=set)
    property_types: List[PropertyType] = field(default_factory=list)
    min_bedrooms: int = 0
    instant_book_only: bool = False


@dataclass
class SearchResult:
    listing: Listing
    distance_km: float
    price_breakdown: Dict[str, int]
    availability: bool


class SearchService:
    def __init__(self, calendar: AvailabilityCalendar, pricing: PricingEngine):
        self._listings: Dict[str, Listing] = {}
        self._calendar = calendar
        self._pricing = pricing

    def index_listing(self, listing: Listing):
        self._listings[listing.listing_id] = listing

    def search(self, query: SearchQuery, limit: int = 20) -> List[SearchResult]:
        results = []
        for listing in self._listings.values():
            # Geo filter
            dist = listing.location.distance_km(query.location)
            if dist > query.radius_km:
                continue
            # Guest capacity
            if listing.max_guests < query.guests:
                continue
            # Amenities filter
            if query.amenities and not query.amenities.issubset(listing.amenities):
                continue
            # Property type filter
            if query.property_types and listing.property_type not in query.property_types:
                continue
            # Bedrooms
            if listing.bedrooms < query.min_bedrooms:
                continue
            # Instant book filter
            if query.instant_book_only and not listing.instant_book:
                continue
            # Availability
            available = self._calendar.is_available(
                listing.listing_id, query.check_in, query.check_out
            )
            # Price filter
            breakdown = self._pricing.calculate(listing, query.check_in, query.check_out, query.guests)
            if not breakdown:
                continue
            nightly = breakdown.get("nightly_avg_cents", 0)
            if not (query.min_price_cents <= nightly <= query.max_price_cents):
                continue

            results.append(SearchResult(listing, dist, breakdown, available))

        # Sort: available first, then by combined score (distance + rating)
        results.sort(key=lambda r: (
            not r.availability,
            r.distance_km * 0.3 - r.listing.rating * 10
        ))
        return results[:limit]


# ---------------------------------------------------------------------------
# Booking Service
# ---------------------------------------------------------------------------

@dataclass
class Booking:
    booking_id: str
    listing_id: str
    guest_id: str
    host_id: str
    check_in: str
    check_out: str
    guests: int
    total_cents: int
    status: BookingStatus = BookingStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    message_to_host: str = ""


class BookingService:
    def __init__(self, calendar: AvailabilityCalendar, listings: Dict[str, Listing]):
        self._bookings: Dict[str, Booking] = {}
        self._user_bookings: Dict[str, List[str]] = defaultdict(list)
        self._calendar = calendar
        self._listings = listings

    def create_booking(self, listing_id: str, guest_id: str,
                       check_in: str, check_out: str, guests: int,
                       total_cents: int, message: str = "") -> Optional[Booking]:
        listing = self._listings.get(listing_id)
        if not listing:
            return None

        booking = Booking(
            booking_id=str(uuid.uuid4())[:12],
            listing_id=listing_id,
            guest_id=guest_id,
            host_id=listing.host_id,
            check_in=check_in,
            check_out=check_out,
            guests=guests,
            total_cents=total_cents,
            message_to_host=message,
        )

        if listing.instant_book:
            locked = self._calendar.instant_lock(
                listing_id, check_in, check_out, booking.booking_id
            )
            if not locked:
                return None
            booking.status = BookingStatus.CONFIRMED
        else:
            locked = self._calendar.soft_lock(
                listing_id, check_in, check_out, booking.booking_id
            )
            if not locked:
                return None
            booking.status = BookingStatus.PENDING

        self._bookings[booking.booking_id] = booking
        self._user_bookings[guest_id].append(booking.booking_id)
        return booking

    def approve_booking(self, booking_id: str, host_id: str) -> bool:
        booking = self._bookings.get(booking_id)
        if not booking or booking.host_id != host_id:
            return False
        if booking.status != BookingStatus.PENDING:
            return False

        confirmed = self._calendar.confirm_lock(
            booking.listing_id, booking.check_in,
            booking.check_out, booking.booking_id
        )
        if not confirmed:
            return False

        booking.status = BookingStatus.CONFIRMED
        booking.updated_at = time.time()
        return True

    def cancel(self, booking_id: str, actor_id: str) -> bool:
        booking = self._bookings.get(booking_id)
        if not booking:
            return False
        if actor_id not in (booking.guest_id, booking.host_id):
            return False
        if booking.status in (BookingStatus.COMPLETED,
                               BookingStatus.CANCELLED_GUEST,
                               BookingStatus.CANCELLED_HOST):
            return False

        self._calendar.release(booking.listing_id, booking.check_in, booking.check_out)
        if actor_id == booking.guest_id:
            booking.status = BookingStatus.CANCELLED_GUEST
        else:
            booking.status = BookingStatus.CANCELLED_HOST
        booking.updated_at = time.time()
        return True

    def get_booking(self, booking_id: str) -> Optional[Booking]:
        return self._bookings.get(booking_id)

    def guest_bookings(self, guest_id: str) -> List[Booking]:
        return [self._bookings[bid] for bid in self._user_bookings.get(guest_id, [])
                if bid in self._bookings]


# ---------------------------------------------------------------------------
# Review System
# ---------------------------------------------------------------------------

@dataclass
class Review:
    review_id: str
    booking_id: str
    reviewer_id: str
    reviewee_id: str   # guest_id for host review, listing_id for guest review
    rating: int        # 1-5
    comment: str
    is_public: bool = False   # Revealed only after both parties review
    created_at: float = field(default_factory=time.time)


class ReviewService:
    REVIEW_WINDOW_DAYS = 14

    def __init__(self):
        self._reviews: Dict[str, List[Review]] = defaultdict(list)  # booking_id → reviews
        self._listing_ratings: Dict[str, List[int]] = defaultdict(list)

    def submit_review(self, booking_id: str, reviewer_id: str,
                      reviewee_id: str, rating: int, comment: str) -> Review:
        r = Review(
            review_id=str(uuid.uuid4())[:8],
            booking_id=booking_id,
            reviewer_id=reviewer_id,
            reviewee_id=reviewee_id,
            rating=max(1, min(5, rating)),
            comment=comment,
        )
        self._reviews[booking_id].append(r)
        # Make public when both sides have reviewed (or window expires)
        if len(self._reviews[booking_id]) >= 2:
            for rev in self._reviews[booking_id]:
                rev.is_public = True
        return r

    def get_listing_reviews(self, listing_id: str) -> List[Review]:
        result = []
        for reviews in self._reviews.values():
            for r in reviews:
                if r.reviewee_id == listing_id and r.is_public:
                    result.append(r)
        return sorted(result, key=lambda r: r.created_at, reverse=True)

    def listing_avg_rating(self, listing_id: str) -> Optional[float]:
        reviews = self.get_listing_reviews(listing_id)
        if not reviews:
            return None
        return round(sum(r.rating for r in reviews) / len(reviews), 2)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def build_test_data():
    calendar = AvailabilityCalendar()
    pricing = PricingEngine()
    search_svc = SearchService(calendar, pricing)

    listings = [
        Listing(
            listing_id="lst_001",
            host_id="host_alice",
            title="Cozy Downtown Loft",
            description="Modern loft in city center",
            location=Location(37.7749, -122.4194, "San Francisco", "US", "SOMA"),
            property_type=PropertyType.ENTIRE_HOME,
            amenities={"wifi", "kitchen", "washer", "parking"},
            max_guests=4,
            bedrooms=1,
            bathrooms=1.0,
            pricing=PricingRule(base_price_cents=12000, cleaning_fee_cents=6000),
            instant_book=True,
            rating=4.8,
            review_count=42,
        ),
        Listing(
            listing_id="lst_002",
            host_id="host_bob",
            title="Sunny Mission Room",
            description="Private room in shared house",
            location=Location(37.7599, -122.4148, "San Francisco", "US", "Mission"),
            property_type=PropertyType.PRIVATE_ROOM,
            amenities={"wifi", "kitchen"},
            max_guests=2,
            bedrooms=1,
            bathrooms=1.0,
            pricing=PricingRule(base_price_cents=6500, cleaning_fee_cents=2500),
            instant_book=False,
            rating=4.5,
            review_count=18,
        ),
        Listing(
            listing_id="lst_003",
            host_id="host_carol",
            title="Oakland Family House",
            description="Spacious 3BR house with garden",
            location=Location(37.8044, -122.2712, "Oakland", "US", "Rockridge"),
            property_type=PropertyType.ENTIRE_HOME,
            amenities={"wifi", "kitchen", "washer", "parking", "pool"},
            max_guests=8,
            bedrooms=3,
            bathrooms=2.0,
            pricing=PricingRule(base_price_cents=22000, cleaning_fee_cents=10000,
                                 weekly_discount_pct=15),
            instant_book=True,
            rating=4.9,
            review_count=95,
        ),
    ]
    for lst in listings:
        search_svc.index_listing(lst)

    booking_svc = BookingService(calendar, {l.listing_id: l for l in listings})
    return calendar, pricing, search_svc, booking_svc, listings


def demonstrate_1_search():
    print("\n=== 1. Property Search ===")
    calendar, pricing, search_svc, booking_svc, listings = build_test_data()

    query = SearchQuery(
        location=Location(37.7749, -122.4194, "San Francisco", "US"),
        check_in="2026-06-01",
        check_out="2026-06-05",
        guests=2,
        radius_km=50,
        amenities={"wifi"},
    )
    results = search_svc.search(query)
    print(f"Search near SF for 2 guests, Jun 1-5:")
    for r in results:
        total = r.price_breakdown.get("total_cents", 0)
        print(f"  {r.listing.title} — ${total/100:.0f} total, "
              f"★{r.listing.rating}, {r.distance_km:.1f} km, "
              f"available={r.availability}, instant={r.listing.instant_book}")


def demonstrate_2_pricing():
    print("\n=== 2. Pricing Breakdown ===")
    _, pricing, _, _, listings = build_test_data()

    lst = listings[0]  # Downtown Loft
    breakdown = pricing.calculate(lst, "2026-06-01", "2026-06-08", guests=2)  # 7 nights
    print(f"Listing: {lst.title}")
    print(f"Check-in: Jun 1, Check-out: Jun 8 ({breakdown['nights']} nights)")
    print(f"  Nightly avg: ${breakdown['nightly_avg_cents']/100:.2f}")
    print(f"  Nightly subtotal: ${breakdown['nightly_total_cents']/100:.2f}")
    print(f"  Weekly discount ({breakdown['discount_pct']}%): "
          f"-${breakdown['discount_cents']/100:.2f}")
    print(f"  Cleaning fee: ${breakdown['cleaning_fee_cents']/100:.2f}")
    print(f"  Service fee: ${breakdown['service_fee_cents']/100:.2f}")
    print(f"  TOTAL: ${breakdown['total_cents']/100:.2f}")


def demonstrate_3_booking_instant():
    print("\n=== 3. Instant Booking ===")
    calendar, pricing, search_svc, booking_svc, listings = build_test_data()

    lst = listings[0]  # instant_book=True
    breakdown = pricing.calculate(lst, "2026-06-01", "2026-06-04", 2)
    booking = booking_svc.create_booking(
        lst.listing_id, "guest_dave", "2026-06-01", "2026-06-04",
        2, breakdown["total_cents"], "Looking forward to the stay!"
    )
    print(f"Instant booking: {booking.booking_id}")
    print(f"  Status: {booking.status.value}")
    print(f"  Total: ${booking.total_cents/100:.2f}")

    # Try double booking same dates
    booking2 = booking_svc.create_booking(
        lst.listing_id, "guest_eve", "2026-06-02", "2026-06-05",
        1, 10000
    )
    print(f"Overlapping booking attempt: {'Rejected (double-booking prevented)' if not booking2 else 'Accepted (BUG!)'}")


def demonstrate_4_request_to_book():
    print("\n=== 4. Request-to-Book Flow ===")
    calendar, pricing, search_svc, booking_svc, listings = build_test_data()

    lst = listings[1]  # instant_book=False
    breakdown = pricing.calculate(lst, "2026-07-10", "2026-07-13", 1)
    booking = booking_svc.create_booking(
        lst.listing_id, "guest_frank", "2026-07-10", "2026-07-13",
        1, breakdown["total_cents"], "I'm a quiet professional traveler."
    )
    print(f"Request submitted: {booking.booking_id}, status={booking.status.value}")

    # Host approves
    approved = booking_svc.approve_booking(booking.booking_id, lst.host_id)
    print(f"Host approved: {approved}, new status={booking.status.value}")

    # Verify calendar blocked
    avail = calendar.is_available(lst.listing_id, "2026-07-11", "2026-07-12")
    print(f"Jul 11 available after confirmation: {avail}")


def demonstrate_5_reviews():
    print("\n=== 5. Review System (Blind Review) ===")
    review_svc = ReviewService()

    # After a stay
    booking_id = "bk_001"
    listing_id = "lst_001"
    host_id = "host_alice"
    guest_id = "guest_dave"

    # Guest reviews listing
    r1 = review_svc.submit_review(booking_id, guest_id, listing_id,
                                   5, "Fantastic location, spotless!")
    print(f"Guest review submitted. Public: {r1.is_public}")

    # Host reviews guest
    r2 = review_svc.submit_review(booking_id, host_id, guest_id,
                                   5, "Perfect guest, would host again!")
    print(f"Host review submitted. Both public: {r1.is_public}, {r2.is_public}")

    avg = review_svc.listing_avg_rating(listing_id)
    print(f"Listing avg rating: ★{avg}")


if __name__ == "__main__":
    demonstrate_1_search()
    demonstrate_2_pricing()
    demonstrate_3_booking_instant()
    demonstrate_4_request_to_book()
    demonstrate_5_reviews()
