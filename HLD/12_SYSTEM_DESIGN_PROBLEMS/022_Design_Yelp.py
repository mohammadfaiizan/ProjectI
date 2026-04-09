"""
YELP — Local Business Search Platform
=======================================

FUNCTIONAL REQUIREMENTS:
- Search businesses by name, category, location
- Filter: distance, rating, price range, open now, amenities
- Business profiles: hours, photos, menus, attributes
- Reviews: text, rating (1-5), photos, reactions
- Check-ins, bookmarks, tips
- Business owner responses to reviews

NON-FUNCTIONAL REQUIREMENTS:
- 178 M reviews, 5.7 M businesses
- 35 M searches/day
- Search < 100 ms p99
- Geo queries: find businesses within radius
- Review trust and spam filtering

ARCHITECTURE:
  Client ──▶ API GW ──▶ Search Svc (Elasticsearch + PostGIS)
                    ──▶ Business Svc (PostgreSQL)
                    ──▶ Review Svc (Cassandra)
                    ──▶ Photo Svc (S3 + CDN)
                    ──▶ Spam Filter (ML classifier)

KEY DESIGN DECISIONS:
1. GEO SEARCH — PostgreSQL with PostGIS extension OR Elasticsearch geo_distance.
   Bounding box first (cheap), then exact haversine filter.
   Geohash indexing for O(1) neighbourhood lookup.

2. REVIEW STORAGE — Cassandra: partition=business_id, cluster=created_at DESC.
   Allows efficient "get latest reviews for business" query.
   Separate table for user's reviews (partition=user_id).

3. SEARCH RANKING — composite score:
   score = text_relevance × 0.4 + rating × 0.3 + distance_decay × 0.2 + review_count_log × 0.1
   Personalisation: boost businesses the user's friends have visited.

4. REVIEW TRUST SCORE — Yelp's "not recommended" filter:
   ML classifier based on: account age, review count, social connections,
   IP diversity, review frequency. Reviews below threshold hidden (but accessible).

5. BUSINESS HOURS — stored as JSON; "open now" filter computed server-side
   by comparing current time with opening hours for today's day_of_week.

6. PHOTOS — uploaded to S3; CDN-served.
   Business profile photo: selected by owner or highest-voted by users.
   Reviews photos: linked to review entity.
"""

from __future__ import annotations
import time
import uuid
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import defaultdict
import threading
from datetime import datetime


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class PriceRange(Enum):
    CHEAP = 1          # $
    MODERATE = 2       # $$
    EXPENSIVE = 3      # $$$
    VERY_EXPENSIVE = 4  # $$$$

    @property
    def label(self) -> str:
        return "$" * self.value


@dataclass
class Location:
    lat: float
    lng: float
    address: str
    city: str
    state: str
    zip_code: str
    country: str = "US"

    def distance_km(self, other: "Location") -> float:
        R = 6371
        dlat = math.radians(other.lat - self.lat)
        dlng = math.radians(other.lng - self.lng)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(self.lat)) *
             math.cos(math.radians(other.lat)) *
             math.sin(dlng / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def geohash(self, precision: int = 6) -> str:
        """Simplified geohash for indexing."""
        base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
        lat_range = (-90.0, 90.0)
        lng_range = (-180.0, 180.0)
        bits, result = 0, 0
        hash_chars = []
        is_lng = True

        while len(hash_chars) < precision:
            if is_lng:
                mid = (lng_range[0] + lng_range[1]) / 2
                if self.lng >= mid:
                    result = (result << 1) | 1
                    lng_range = (mid, lng_range[1])
                else:
                    result = result << 1
                    lng_range = (lng_range[0], mid)
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if self.lat >= mid:
                    result = (result << 1) | 1
                    lat_range = (mid, lat_range[1])
                else:
                    result = result << 1
                    lat_range = (lat_range[0], mid)
            is_lng = not is_lng
            bits += 1
            if bits == 5:
                hash_chars.append(base32[result & 0x1f])
                bits = 0
                result = 0

        return "".join(hash_chars)


@dataclass
class BusinessHours:
    """Hours per day_of_week (0=Mon, 6=Sun)."""
    hours: Dict[int, Tuple[str, str]] = field(default_factory=dict)  # day → (open, close)

    def is_open_now(self, dt: Optional[datetime] = None) -> bool:
        if dt is None:
            dt = datetime.now()
        day = dt.weekday()
        if day not in self.hours:
            return False
        open_time, close_time = self.hours[day]
        current = dt.strftime("%H:%M")
        return open_time <= current <= close_time

    @staticmethod
    def typical_restaurant() -> "BusinessHours":
        hours = {}
        for day in range(7):
            hours[day] = ("11:00", "22:00")
        return BusinessHours(hours)


@dataclass
class Business:
    business_id: str
    owner_id: str
    name: str
    categories: List[str]
    location: Location
    price_range: PriceRange
    hours: BusinessHours
    phone: str = ""
    website: str = ""
    description: str = ""
    attributes: Set[str] = field(default_factory=set)  # wifi, delivery, takeout, etc.
    photos: List[str] = field(default_factory=list)
    rating: float = 0.0
    review_count: int = 0
    checkin_count: int = 0
    is_claimed: bool = False
    is_closed_permanently: bool = False
    created_at: float = field(default_factory=time.time)

    @property
    def price_label(self) -> str:
        return self.price_range.label


# ---------------------------------------------------------------------------
# Review System
# ---------------------------------------------------------------------------

@dataclass
class Review:
    review_id: str
    business_id: str
    user_id: str
    rating: int           # 1-5
    text: str
    photos: List[str] = field(default_factory=list)
    useful: int = 0
    funny: int = 0
    cool: int = 0
    trust_score: float = 1.0   # 0-1; below threshold → "not recommended"
    is_recommended: bool = True
    owner_response: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class ReviewTrustFilter:
    """
    Simplified version of Yelp's recommendation engine.
    Real system: gradient boosted trees on 50+ features.
    """

    TRUST_THRESHOLD = 0.4

    def compute_trust_score(self, user_id: str, user_review_count: int,
                             account_age_days: int,
                             is_friend_of_business_owner: bool) -> float:
        score = 0.0
        # Account age (max 0.3)
        score += min(0.3, account_age_days / 365 * 0.3)
        # Review history (max 0.4)
        score += min(0.4, math.log1p(user_review_count) / 10 * 0.4)
        # Social connections (max 0.3)
        if not is_friend_of_business_owner:
            score += 0.3
        return round(min(1.0, score), 3)

    def is_recommended(self, trust_score: float) -> bool:
        return trust_score >= self.TRUST_THRESHOLD


class ReviewService:
    def __init__(self, trust_filter: ReviewTrustFilter):
        self._reviews: Dict[str, List[Review]] = defaultdict(list)  # business_id → reviews
        self._user_reviews: Dict[str, List[str]] = defaultdict(list)
        self._review_by_id: Dict[str, Review] = {}
        self._trust = trust_filter
        self._business_ratings: Dict[str, List[int]] = defaultdict(list)

    def submit_review(self, business_id: str, user_id: str, rating: int,
                      text: str, user_review_count: int = 5,
                      account_age_days: int = 180,
                      is_friend: bool = False) -> Review:
        trust = self._trust.compute_trust_score(
            user_id, user_review_count, account_age_days, is_friend
        )
        review = Review(
            review_id=str(uuid.uuid4())[:8],
            business_id=business_id,
            user_id=user_id,
            rating=max(1, min(5, rating)),
            text=text,
            trust_score=trust,
            is_recommended=self._trust.is_recommended(trust),
        )
        self._reviews[business_id].append(review)
        self._user_reviews[user_id].append(review.review_id)
        self._review_by_id[review.review_id] = review

        # Update business rating (only recommended reviews count)
        if review.is_recommended:
            self._business_ratings[business_id].append(review.rating)

        return review

    def add_owner_response(self, review_id: str, owner_id: str, response: str) -> bool:
        review = self._review_by_id.get(review_id)
        if review:
            review.owner_response = response
            return True
        return False

    def react(self, review_id: str, reaction: str) -> bool:
        review = self._review_by_id.get(review_id)
        if not review or reaction not in ("useful", "funny", "cool"):
            return False
        setattr(review, reaction, getattr(review, reaction) + 1)
        return True

    def get_reviews(self, business_id: str, recommended_only: bool = True,
                    sort_by: str = "recent", limit: int = 20) -> List[Review]:
        reviews = [r for r in self._reviews.get(business_id, [])
                   if not recommended_only or r.is_recommended]
        if sort_by == "rating_high":
            reviews.sort(key=lambda r: r.rating, reverse=True)
        elif sort_by == "useful":
            reviews.sort(key=lambda r: r.useful, reverse=True)
        else:  # recent
            reviews.sort(key=lambda r: r.created_at, reverse=True)
        return reviews[:limit]

    def business_avg_rating(self, business_id: str) -> Optional[float]:
        ratings = self._business_ratings.get(business_id, [])
        return round(sum(ratings) / len(ratings), 1) if ratings else None

    def rating_distribution(self, business_id: str) -> Dict[int, int]:
        dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in self._reviews.get(business_id, []):
            if r.is_recommended:
                dist[r.rating] += 1
        return dist


# ---------------------------------------------------------------------------
# Search Service
# ---------------------------------------------------------------------------

@dataclass
class SearchQuery:
    location: Location
    query: str = ""
    category: str = ""
    radius_km: float = 10.0
    price_ranges: List[PriceRange] = field(default_factory=list)
    min_rating: float = 0.0
    open_now: bool = False
    attributes: Set[str] = field(default_factory=set)
    sort_by: str = "best_match"  # best_match | distance | rating | review_count


@dataclass
class SearchResult:
    business: Business
    distance_km: float
    score: float


class SearchService:
    def __init__(self, review_svc: ReviewService):
        self._businesses: Dict[str, Business] = {}
        self._category_index: Dict[str, Set[str]] = defaultdict(set)
        self._geohash_index: Dict[str, Set[str]] = defaultdict(set)
        self._review_svc = review_svc
        # term → business_ids (text index)
        self._text_index: Dict[str, Set[str]] = defaultdict(set)

    def index_business(self, business: Business) -> None:
        self._businesses[business.business_id] = business
        for cat in business.categories:
            self._category_index[cat.lower()].add(business.business_id)
        gh = business.location.geohash(6)
        self._geohash_index[gh].add(business.business_id)
        # Text index
        tokens = business.name.lower().split() + [c.lower() for c in business.categories]
        for token in tokens:
            self._text_index[token].add(business.business_id)

    def search(self, query: SearchQuery, limit: int = 20) -> List[SearchResult]:
        # Candidate set from text/category query
        if query.query:
            tokens = query.query.lower().split()
            candidates = self._text_index.get(tokens[0], set()).copy()
            for t in tokens[1:]:
                candidates |= self._text_index.get(t, set())
        elif query.category:
            candidates = self._category_index.get(query.category.lower(), set()).copy()
        else:
            candidates = set(self._businesses.keys())

        results = []
        for bid in candidates:
            biz = self._businesses.get(bid)
            if not biz or biz.is_closed_permanently:
                continue

            # Geo filter
            dist = biz.location.distance_km(query.location)
            if dist > query.radius_km:
                continue

            # Price filter
            if query.price_ranges and biz.price_range not in query.price_ranges:
                continue

            # Rating filter
            avg = self._review_svc.business_avg_rating(biz.business_id) or biz.rating
            if avg < query.min_rating:
                continue

            # Open now
            if query.open_now and not biz.hours.is_open_now():
                continue

            # Attribute filter
            if query.attributes and not query.attributes.issubset(biz.attributes):
                continue

            # Scoring
            text_score = 1.0 if query.query else 0.5
            distance_decay = math.exp(-dist / query.radius_km)
            review_count_log = math.log1p(biz.review_count) / 10
            score = (text_score * 0.4 + avg / 5 * 0.3 +
                     distance_decay * 0.2 + review_count_log * 0.1)

            results.append(SearchResult(biz, dist, score))

        if query.sort_by == "distance":
            results.sort(key=lambda r: r.distance_km)
        elif query.sort_by == "rating":
            results.sort(key=lambda r: -(self._review_svc.business_avg_rating(r.business.business_id) or 0))
        elif query.sort_by == "review_count":
            results.sort(key=lambda r: -r.business.review_count)
        else:
            results.sort(key=lambda r: -r.score)

        return results[:limit]


# ---------------------------------------------------------------------------
# Check-in Service
# ---------------------------------------------------------------------------

class CheckinService:
    def __init__(self):
        self._checkins: Dict[str, List[Dict]] = defaultdict(list)  # business_id → checkins

    def checkin(self, business_id: str, user_id: str) -> Dict:
        entry = {
            "checkin_id": str(uuid.uuid4())[:8],
            "user_id": user_id,
            "business_id": business_id,
            "ts": time.time(),
        }
        self._checkins[business_id].append(entry)
        return entry

    def count(self, business_id: str) -> int:
        return len(self._checkins.get(business_id, []))

    def user_checkins(self, user_id: str) -> List[Dict]:
        result = []
        for checkins in self._checkins.values():
            result.extend(c for c in checkins if c["user_id"] == user_id)
        return sorted(result, key=lambda c: c["ts"], reverse=True)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def build_test_data():
    trust = ReviewTrustFilter()
    review_svc = ReviewService(trust)
    search_svc = SearchService(review_svc)
    checkin_svc = CheckinService()

    sf_center = Location(37.7749, -122.4194, "1 Market St", "San Francisco", "CA", "94105")

    businesses = [
        Business(
            business_id="biz_001", owner_id="owner_a",
            name="Golden Gate Bistro", categories=["Restaurants", "French"],
            location=Location(37.7745, -122.4180, "22 Main St", "San Francisco", "CA", "94105"),
            price_range=PriceRange.EXPENSIVE,
            hours=BusinessHours.typical_restaurant(),
            attributes={"wifi", "delivery", "outdoor_seating"},
            rating=4.5, review_count=320, is_claimed=True,
        ),
        Business(
            business_id="biz_002", owner_id="owner_b",
            name="Bay Area Coffee Co", categories=["Coffee", "Cafes"],
            location=Location(37.7760, -122.4175, "88 Union St", "San Francisco", "CA", "94133"),
            price_range=PriceRange.CHEAP,
            hours=BusinessHours.typical_restaurant(),
            attributes={"wifi", "takeout"},
            rating=4.2, review_count=150,
        ),
        Business(
            business_id="biz_003", owner_id="owner_c",
            name="Mission Tacos", categories=["Restaurants", "Mexican"],
            location=Location(37.7599, -122.4148, "451 Valencia St", "San Francisco", "CA", "94103"),
            price_range=PriceRange.MODERATE,
            hours=BusinessHours.typical_restaurant(),
            attributes={"takeout", "delivery"},
            rating=4.7, review_count=890,
        ),
    ]

    for biz in businesses:
        search_svc.index_business(biz)

    return sf_center, businesses, review_svc, search_svc, checkin_svc


def demonstrate_1_search():
    print("\n=== 1. Business Search ===")
    sf, businesses, review_svc, search_svc, _ = build_test_data()

    # Simple keyword search
    results = search_svc.search(SearchQuery(sf, query="coffee", radius_km=5))
    print(f"Search 'coffee' near SF:")
    for r in results:
        print(f"  {r.business.name} ({r.business.price_label}) — "
              f"{r.distance_km:.2f} km, ★{r.business.rating}")

    # Category search with price filter
    results2 = search_svc.search(SearchQuery(
        sf, category="Restaurants",
        price_ranges=[PriceRange.CHEAP, PriceRange.MODERATE],
        min_rating=4.0, radius_km=10
    ))
    print(f"\nRestaurants ($-$$, ★4+) near SF:")
    for r in results2:
        print(f"  {r.business.name} — {r.distance_km:.2f} km, score={r.score:.3f}")


def demonstrate_2_reviews():
    print("\n=== 2. Reviews & Trust Scoring ===")
    _, businesses, review_svc, _, _ = build_test_data()
    biz = businesses[0]  # Golden Gate Bistro

    # Established reviewer
    r1 = review_svc.submit_review(biz.business_id, "user_alice", 5,
                                   "Absolutely wonderful! The duck confit was divine.",
                                   user_review_count=87, account_age_days=900)
    # New suspicious reviewer
    r2 = review_svc.submit_review(biz.business_id, "user_spammer", 1,
                                   "Worst place ever!",
                                   user_review_count=1, account_age_days=2,
                                   is_friend=True)
    # Average reviewer
    r3 = review_svc.submit_review(biz.business_id, "user_bob", 4,
                                   "Great atmosphere, a bit pricey.",
                                   user_review_count=15, account_age_days=400)

    print(f"Reviews submitted for '{biz.name}':")
    for r in [r1, r2, r3]:
        status = "RECOMMENDED" if r.is_recommended else "NOT RECOMMENDED"
        print(f"  [{r.user_id}] ★{r.rating} trust={r.trust_score:.3f} → {status}")

    avg = review_svc.business_avg_rating(biz.business_id)
    dist = review_svc.rating_distribution(biz.business_id)
    print(f"\nAverage (recommended only): ★{avg}")
    print(f"Rating distribution: {dist}")


def demonstrate_3_owner_response():
    print("\n=== 3. Owner Response to Review ===")
    _, businesses, review_svc, _, _ = build_test_data()
    biz = businesses[0]

    r = review_svc.submit_review(biz.business_id, "user_carol", 3,
                                   "Good food but service was slow.",
                                   user_review_count=20, account_age_days=600)
    # React to review
    review_svc.react(r.review_id, "useful")
    review_svc.react(r.review_id, "useful")

    # Owner responds
    responded = review_svc.add_owner_response(
        r.review_id, biz.owner_id,
        "Thank you for your feedback! We're working on improving our service times."
    )
    print(f"Review: '{r.text}'")
    print(f"  Useful votes: {r.useful}")
    print(f"  Owner responded: {responded}")
    print(f"  Response: '{r.owner_response}'")


def demonstrate_4_geohash_search():
    print("\n=== 4. Geohash-based Proximity ===")
    sf, businesses, _, _, _ = build_test_data()

    # Show geohash precision
    for biz in businesses:
        gh = biz.location.geohash(6)
        dist = biz.location.distance_km(sf)
        print(f"  {biz.name}: geohash={gh}, dist={dist:.2f} km")

    # Nearby businesses share geohash prefix
    geo1 = businesses[0].location.geohash(4)
    geo2 = businesses[1].location.geohash(4)
    print(f"\n4-char geohash: biz_001={geo1}, biz_002={geo2}")
    print(f"Same neighbourhood prefix: {geo1 == geo2}")


def demonstrate_5_checkins():
    print("\n=== 5. Check-ins ===")
    _, businesses, _, _, checkin_svc = build_test_data()
    biz = businesses[2]  # Mission Tacos

    for user in ["alice", "bob", "carol", "alice"]:  # alice checks in twice
        checkin_svc.checkin(biz.business_id, f"user_{user}")

    total = checkin_svc.count(biz.business_id)
    print(f"Total check-ins at '{biz.name}': {total}")

    alice_visits = checkin_svc.user_checkins("user_alice")
    print(f"Alice's total check-in count: {len(alice_visits)}")


if __name__ == "__main__":
    demonstrate_1_search()
    demonstrate_2_reviews()
    demonstrate_3_owner_response()
    demonstrate_4_geohash_search()
    demonstrate_5_checkins()
