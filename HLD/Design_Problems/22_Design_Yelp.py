"""
Yelp System Design - Python Implementation
Demonstrates: Geohash encoding, QuadTree spatial index, BusinessIndex,
              RatingAggregator (Bayesian), YelpSystem facade.
No external dependencies - standard library only.
"""

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Business:
    id: str
    name: str
    lat: float
    lon: float
    categories: list
    price_range: int        # 1-4
    address: str = ""
    phone: str = ""
    hours: dict = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)
    avg_rating: float = 0.0
    review_count: int = 0
    bayesian_rating: float = 0.0

@dataclass
class Review:
    id: str
    business_id: str
    user_id: str
    rating: int
    text: str
    useful: int = 0
    funny: int = 0
    cool: int = 0

@dataclass
class SearchResult:
    business: Business
    distance_km: float
    score: float


# ---------------------------------------------------------------------------
# 1. Geohash — encode/decode lat/lon, get neighbors
# ---------------------------------------------------------------------------

class Geohash:
    """
    Geohash encoding: subdivides earth into grid cells.
    Precision 6 → cells ~1.2km × 0.6km.
    Adjacent cells share common prefix (mostly).
    """
    BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    DECODE_MAP = {c: i for i, c in enumerate(BASE32)}

    @classmethod
    def encode(cls, lat: float, lon: float, precision: int = 6) -> str:
        min_lat, max_lat = -90.0, 90.0
        min_lon, max_lon = -180.0, 180.0
        result, bits, bit = [], 0, 0
        use_lon = True   # alternate lon/lat each bit

        while len(result) < precision:
            if use_lon:
                mid = (min_lon + max_lon) / 2
                if lon >= mid:
                    bit = (bit << 1) | 1
                    min_lon = mid
                else:
                    bit = bit << 1
                    max_lon = mid
            else:
                mid = (min_lat + max_lat) / 2
                if lat >= mid:
                    bit = (bit << 1) | 1
                    min_lat = mid
                else:
                    bit = bit << 1
                    max_lat = mid
            use_lon = not use_lon
            bits += 1
            if bits == 5:
                result.append(cls.BASE32[bit])
                bits, bit = 0, 0
        return "".join(result)

    @classmethod
    def decode_bounds(cls, geohash: str) -> tuple:
        """Returns (min_lat, max_lat, min_lon, max_lon) bounding box."""
        min_lat, max_lat = -90.0, 90.0
        min_lon, max_lon = -180.0, 180.0
        use_lon = True
        for char in geohash:
            val = cls.DECODE_MAP[char]
            for i in range(4, -1, -1):
                bit = (val >> i) & 1
                if use_lon:
                    mid = (min_lon + max_lon) / 2
                    if bit:  min_lon = mid
                    else:    max_lon = mid
                else:
                    mid = (min_lat + max_lat) / 2
                    if bit:  min_lat = mid
                    else:    max_lat = mid
                use_lon = not use_lon
        return min_lat, max_lat, min_lon, max_lon

    @classmethod
    def decode(cls, geohash: str) -> tuple:
        """Returns (lat, lon) center of geohash cell."""
        min_lat, max_lat, min_lon, max_lon = cls.decode_bounds(geohash)
        return (min_lat + max_lat) / 2, (min_lon + max_lon) / 2

    @classmethod
    def neighbors(cls, geohash: str) -> list[str]:
        """
        Returns 8 neighboring geohash cells.
        Simplified version: decode center, re-encode with offsets.
        """
        min_lat, max_lat, min_lon, max_lon = cls.decode_bounds(geohash)
        lat_step = max_lat - min_lat
        lon_step = max_lon - min_lon
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        precision = len(geohash)

        result = []
        for dlat in [-1, 0, 1]:
            for dlon in [-1, 0, 1]:
                if dlat == 0 and dlon == 0:
                    continue
                nlat = center_lat + dlat * lat_step
                nlon = center_lon + dlon * lon_step
                # Clamp to valid range
                nlat = max(-89.9, min(89.9, nlat))
                nlon = ((nlon + 180) % 360) - 180
                result.append(cls.encode(nlat, nlon, precision))
        return result


# ---------------------------------------------------------------------------
# 2. QuadTree — spatial partitioning
# ---------------------------------------------------------------------------

class QuadTreeNode:
    MAX_ITEMS = 10  # max businesses per leaf before split

    def __init__(self, min_lat, max_lat, min_lon, max_lon, depth=0):
        self.min_lat, self.max_lat = min_lat, max_lat
        self.min_lon, self.max_lon = min_lon, max_lon
        self.depth = depth
        self.items: list[Business] = []
        self.children: list['QuadTreeNode'] = []  # NW, NE, SW, SE

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def _contains(self, lat: float, lon: float) -> bool:
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon

    def _split(self):
        mid_lat = (self.min_lat + self.max_lat) / 2
        mid_lon = (self.min_lon + self.max_lon) / 2
        self.children = [
            QuadTreeNode(mid_lat, self.max_lat, self.min_lon, mid_lon, self.depth+1),  # NW
            QuadTreeNode(mid_lat, self.max_lat, mid_lon, self.max_lon, self.depth+1),  # NE
            QuadTreeNode(self.min_lat, mid_lat, self.min_lon, mid_lon, self.depth+1),  # SW
            QuadTreeNode(self.min_lat, mid_lat, mid_lon, self.max_lon, self.depth+1),  # SE
        ]
        for item in self.items:
            for child in self.children:
                if child._contains(item.lat, item.lon):
                    child.insert(item)
                    break
        self.items = []

    def insert(self, business: Business) -> None:
        if not self._contains(business.lat, business.lon):
            return
        if self.is_leaf:
            self.items.append(business)
            if len(self.items) > self.MAX_ITEMS and self.depth < 12:
                self._split()
        else:
            for child in self.children:
                if child._contains(business.lat, business.lon):
                    child.insert(business)
                    return

    def query_radius(
        self, center_lat: float, center_lon: float, radius_km: float
    ) -> list[Business]:
        """Returns all businesses within radius of center point."""
        # Bounding box overlap check (skip entire node if no overlap)
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * max(math.cos(math.radians(center_lat)), 0.01))
        if (center_lat - lat_delta > self.max_lat or
                center_lat + lat_delta < self.min_lat or
                center_lon - lon_delta > self.max_lon or
                center_lon + lon_delta < self.min_lon):
            return []
        if self.is_leaf:
            return self.items[:]
        results = []
        for child in self.children:
            results.extend(child.query_radius(center_lat, center_lon, radius_km))
        return results


class QuadTree:
    def __init__(self):
        self.root = QuadTreeNode(-90, 90, -180, 180)

    def insert(self, business: Business) -> None:
        self.root.insert(business)

    def query(self, lat: float, lon: float, radius_km: float) -> list[Business]:
        candidates = self.root.query_radius(lat, lon, radius_km)
        # Exact haversine filter
        result = []
        for b in candidates:
            if haversine(lat, lon, b.lat, b.lon) <= radius_km:
                result.append(b)
        return result


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# 3. BusinessIndex — inverted index for text search + geo filter
# ---------------------------------------------------------------------------

class BusinessIndex:
    """
    Simplified inverted index on business name + categories.
    In production: Elasticsearch with BM25 ranking + geo_distance filter.
    """

    def __init__(self):
        self._businesses: dict[str, Business] = {}
        self._inverted: dict[str, set[str]] = defaultdict(set)   # token -> {business_id}

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().replace(",", " ").split()

    def add(self, business: Business) -> None:
        self._businesses[business.id] = business
        tokens = self._tokenize(business.name)
        for cat in business.categories:
            tokens.extend(self._tokenize(cat))
        for token in tokens:
            self._inverted[token].add(business.id)

    def search(self, query: str) -> list[Business]:
        """Returns businesses matching all query tokens (AND semantics)."""
        tokens = self._tokenize(query)
        if not tokens:
            return list(self._businesses.values())
        matching_ids = None
        for token in tokens:
            ids = self._inverted.get(token, set())
            matching_ids = ids if matching_ids is None else matching_ids & ids
        if not matching_ids:
            return []
        return [self._businesses[bid] for bid in matching_ids]

    def get(self, business_id: str) -> Optional[Business]:
        return self._businesses.get(business_id)


# ---------------------------------------------------------------------------
# 4. RatingAggregator — Bayesian average
# ---------------------------------------------------------------------------

class RatingAggregator:
    """
    Bayesian average: (v*R + m*C) / (v+m)
    v = review count, R = business avg, m = min votes threshold, C = global avg
    """
    GLOBAL_AVG = 3.5   # C
    MIN_VOTES  = 10    # m

    def __init__(self):
        # business_id -> [list of ratings]
        self._ratings: dict[str, list[int]] = defaultdict(list)

    def add_rating(self, business_id: str, rating: int) -> None:
        self._ratings[business_id].append(rating)

    def raw_average(self, business_id: str) -> float:
        ratings = self._ratings[business_id]
        if not ratings:
            return 0.0
        return round(sum(ratings) / len(ratings), 2)

    def bayesian_average(self, business_id: str) -> float:
        """Shrinks toward global mean for low-count businesses."""
        ratings = self._ratings[business_id]
        v = len(ratings)
        if v == 0:
            return self.GLOBAL_AVG
        R = sum(ratings) / v
        score = (v * R + self.MIN_VOTES * self.GLOBAL_AVG) / (v + self.MIN_VOTES)
        return round(score, 2)

    def review_count(self, business_id: str) -> int:
        return len(self._ratings[business_id])


# ---------------------------------------------------------------------------
# 5. YelpSystem — Facade
# ---------------------------------------------------------------------------

class YelpSystem:
    SEARCH_WEIGHTS = {"distance": 0.35, "rating": 0.30, "popularity": 0.20, "text": 0.15}

    def __init__(self):
        self._index     = BusinessIndex()
        self._quadtree  = QuadTree()
        self._ratings   = RatingAggregator()
        self._reviews: dict[str, Review]          = {}
        self._biz_reviews: dict[str, list[str]]   = defaultdict(list)
        self._checkins: dict[str, list[str]]       = defaultdict(list)  # biz_id -> [user_ids]

    # -- Business Management ------------------------------------------------

    def add_business(self, business: Business) -> None:
        self._index.add(business)
        self._quadtree.insert(business)

    def get_business_details(self, business_id: str) -> Optional[Business]:
        biz = self._index.get(business_id)
        if biz:
            biz.avg_rating     = self._ratings.raw_average(business_id)
            biz.review_count   = self._ratings.review_count(business_id)
            biz.bayesian_rating = self._ratings.bayesian_average(business_id)
        return biz

    # -- Search -------------------------------------------------------------

    def _rank_score(
        self, business: Business, dist_km: float, text_relevance: float = 0.5
    ) -> float:
        w = self.SEARCH_WEIGHTS
        dist_score   = 1.0 / (1.0 + dist_km)
        rating_score = business.bayesian_rating / 5.0
        pop_score    = min(math.log(business.review_count + 1) / 10.0, 1.0)
        return (w["distance"] * dist_score + w["rating"] * rating_score +
                w["popularity"] * pop_score + w["text"] * text_relevance)

    def search_nearby(
        self,
        lat: float,
        lon: float,
        query: str = "",
        radius_km: float = 5.0,
        categories: list = None,
        price_range: list = None,
        sort: str = "relevance",
        limit: int = 20,
    ) -> list[SearchResult]:
        # Step 1: geo candidates from QuadTree
        geo_candidates = self._quadtree.query(lat, lon, radius_km)

        # Step 2: text filter from inverted index
        if query:
            text_ids = {b.id for b in self._index.search(query)}
        else:
            text_ids = None

        results = []
        for biz in geo_candidates:
            # Text relevance filter
            text_rel = 1.0 if (text_ids is None or biz.id in text_ids) else 0.0
            if query and text_ids is not None and biz.id not in text_ids:
                continue
            # Category filter
            if categories:
                if not any(c.lower() in [x.lower() for x in biz.categories] for c in categories):
                    continue
            # Price range filter
            if price_range and biz.price_range not in price_range:
                continue

            # Refresh ratings
            biz.avg_rating      = self._ratings.raw_average(biz.id)
            biz.review_count    = self._ratings.review_count(biz.id)
            biz.bayesian_rating = self._ratings.bayesian_average(biz.id)

            dist = haversine(lat, lon, biz.lat, biz.lon)
            score = self._rank_score(biz, dist, text_rel)
            results.append(SearchResult(biz, round(dist, 2), round(score, 4)))

        # Sort
        if sort == "distance":
            results.sort(key=lambda r: r.distance_km)
        elif sort == "rating":
            results.sort(key=lambda r: r.business.bayesian_rating, reverse=True)
        else:  # relevance (default)
            results.sort(key=lambda r: r.score, reverse=True)

        return results[:limit]

    # -- Reviews ------------------------------------------------------------

    def write_review(
        self,
        business_id: str,
        user_id: str,
        rating: int,
        text: str,
    ) -> Review:
        if not (1 <= rating <= 5):
            raise ValueError("Rating must be 1-5")
        # One review per user per business check
        for rid in self._biz_reviews[business_id]:
            existing = self._reviews[rid]
            if existing.user_id == user_id:
                raise ValueError("User already reviewed this business")

        review = Review(
            id          = str(uuid.uuid4())[:8],
            business_id = business_id,
            user_id     = user_id,
            rating      = rating,
            text        = text,
        )
        self._reviews[review.id] = review
        self._biz_reviews[business_id].append(review.id)
        self._ratings.add_rating(business_id, rating)
        return review

    def get_reviews(self, business_id: str, limit: int = 20) -> list[Review]:
        review_ids = self._biz_reviews[business_id][-limit:]
        return [self._reviews[rid] for rid in review_ids]

    # -- Trending -----------------------------------------------------------

    def get_trending_nearby(
        self, lat: float, lon: float, radius_km: float = 5.0, top_n: int = 5
    ) -> list[Business]:
        """Returns top N businesses by check-in count within radius."""
        candidates = self._quadtree.query(lat, lon, radius_km)
        scored = []
        for biz in candidates:
            checkin_count = len(self._checkins[biz.id])
            scored.append((biz, checkin_count))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [biz for biz, _ in scored[:top_n]]

    def check_in(self, business_id: str, user_id: str) -> None:
        self._checkins[business_id].append(user_id)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    system = YelpSystem()

    # Add businesses
    businesses = [
        Business("B1", "Tony's Pizza",       37.7749, -122.4194, ["restaurant", "pizza", "italian"], 2),
        Business("B2", "Golden Gate Ramen",  37.7755, -122.4180, ["restaurant", "ramen", "japanese"], 2),
        Business("B3", "Bay Brew Coffee",    37.7760, -122.4200, ["coffee", "cafe"],                  1),
        Business("B4", "Sakura Sushi",       37.7800, -122.4100, ["restaurant", "sushi", "japanese"], 3),
        Business("B5", "Far Away Diner",     37.9500, -122.5000, ["restaurant", "american"],          1),
    ]
    for b in businesses:
        system.add_business(b)

    # Add reviews
    system.write_review("B1", "u1", 5, "Best pizza in SF!")
    system.write_review("B1", "u2", 4, "Good but pricey.")
    system.write_review("B1", "u3", 5, "Incredible.")
    system.write_review("B2", "u1", 4, "Solid ramen spot.")
    system.write_review("B3", "u2", 3, "Average coffee.")

    # Geohash demo
    print("=== Geohash Demo ===")
    gh = Geohash.encode(37.7749, -122.4194, precision=6)
    print(f"  SF geohash (precision 6): {gh}")
    lat, lon = Geohash.decode(gh)
    print(f"  Decoded center: ({lat:.4f}, {lon:.4f})")
    print(f"  Neighbors: {Geohash.neighbors(gh)[:3]} ... (8 total)")

    # Nearby search
    print("\n=== Nearby Search (radius 5km) ===")
    results = system.search_nearby(37.7749, -122.4194, radius_km=5.0, sort="relevance")
    for r in results:
        b = r.business
        print(f"  {b.name}: {r.distance_km}km | rating={b.bayesian_rating} "
              f"({b.review_count} reviews) | score={r.score}")

    # Category search
    print("\n=== Japanese Restaurants ===")
    results = system.search_nearby(37.7749, -122.4194, radius_km=10.0, categories=["japanese"])
    for r in results:
        print(f"  {r.business.name}: {r.distance_km}km")

    # Text search
    print("\n=== Text Search: 'pizza' ===")
    results = system.search_nearby(37.7749, -122.4194, query="pizza", radius_km=5.0)
    for r in results:
        print(f"  {r.business.name}: {r.distance_km}km")

    # Bayesian rating comparison
    print("\n=== Bayesian vs Raw Ratings ===")
    agg = system._ratings
    for bid in ["B1", "B2", "B3"]:
        b = system.get_business_details(bid)
        print(f"  {b.name}: raw={b.avg_rating} | bayesian={b.bayesian_rating} "
              f"| count={b.review_count}")

    # Trending
    print("\n=== Trending (check-ins) ===")
    for _ in range(10): system.check_in("B2", "u_random")
    for _ in range(5):  system.check_in("B1", "u_random")
    trending = system.get_trending_nearby(37.7749, -122.4194, radius_km=5.0)
    for b in trending:
        count = len(system._checkins[b.id])
        print(f"  {b.name}: {count} check-ins")
