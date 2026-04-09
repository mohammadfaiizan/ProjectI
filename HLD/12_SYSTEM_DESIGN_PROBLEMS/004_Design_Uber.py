"""
SYSTEM DESIGN: UBER (Ride Sharing)
=====================================

Problem Statement:
Design a ride-sharing platform where riders can request rides,
drivers can accept, and the system matches them efficiently.

Functional Requirements:
  - Rider: request a ride (pickup + destination)
  - Driver: broadcast location, accept/decline rides
  - Matching: find nearest available driver
  - Trip tracking: live location during trip
  - Fare calculation: based on time + distance
  - Payment: charge at end of trip

Non-Functional Requirements:
  - 14M trips/day → ~160 trips/sec
  - 1M active drivers → frequent location updates (every 4s)
  - Location updates: 1M drivers × 15/min = 250K updates/sec
  - Match latency: < 1s (rider waits < 2s for driver assignment)
  - Location precision: < 50m

Location Update Pipeline:
  Driver app → WebSocket gateway → Kafka (location events) →
  Location worker → Redis GEO (current position) + Cassandra (history)

Geospatial Indexing:
  QuadTree: 2D spatial index. Subdivide cell when > N drivers.
  S2 Geometry (Google): hierarchical grid cells at multiple resolutions.
  Geohash: 5-char geohash ≈ 4.9km × 4.9km cell. 6-char ≈ 1.2km × 1.2km.
  Redis GEOADD / GEORADIUS: built-in geo commands for proximity queries.

Matching Algorithm:
  1. Rider requests ride from (lat, lng).
  2. Query Redis GEO for drivers within R km.
  3. Filter: available drivers only.
  4. Score by estimated_arrival_time (ETA) + driver_rating.
  5. Offer to nearest driver. If decline/timeout (10s) → next.
  6. Once accepted → lock driver, start trip.

Dynamic Pricing (Surge):
  surge_multiplier = f(supply, demand)
  supply  = available_drivers_in_zone
  demand  = unfulfilled_requests_in_zone
  surge   = max(1.0, demand / supply × base_surge)
  Displayed to rider before confirmation.

ETA Calculation:
  Mapbox / Google Maps API for routing.
  Or: Uber's internal routing engine (H3-based hexagonal grid).
  ETA includes pickup + trip duration.

Trip State Machine:
  REQUESTED → MATCHED → DRIVER_EN_ROUTE → ARRIVED →
  IN_PROGRESS → COMPLETED → PAID
  (or CANCELLED at most states)

Fault Tolerance:
  If driver app disconnects mid-trip: reconnect with trip_id → resume.
  If matching worker dies: Kafka consumer group re-assigns partitions.
  If payment fails: retry with exponential backoff; async notification.
"""

from __future__ import annotations

import math
import time
import uuid
import heapq
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict


# ─────────────────────────────────────────────
# GEOSPATIAL UTILITIES
# ─────────────────────────────────────────────

@dataclass
class Location:
    lat: float
    lng: float

    def distance_km(self, other: "Location") -> float:
        """Haversine formula."""
        R    = 6371.0
        dlat = math.radians(other.lat - self.lat)
        dlng = math.radians(other.lng - self.lng)
        a    = (math.sin(dlat / 2) ** 2 +
                math.cos(math.radians(self.lat)) *
                math.cos(math.radians(other.lat)) *
                math.sin(dlng / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def geohash(self, precision: int = 6) -> str:
        """Simplified geohash (just for demo; not RFC-compliant)."""
        lat_n = int((self.lat + 90) / 180 * (32 ** precision))
        lng_n = int((self.lng + 180) / 360 * (32 ** precision))
        return f"{lat_n:0{precision}x}{lng_n:0{precision}x}"[:precision]


# ─────────────────────────────────────────────
# TRIP STATE
# ─────────────────────────────────────────────

class TripState(Enum):
    REQUESTED       = "requested"
    MATCHED         = "matched"
    DRIVER_EN_ROUTE = "driver_en_route"
    ARRIVED         = "arrived"
    IN_PROGRESS     = "in_progress"
    COMPLETED       = "completed"
    CANCELLED       = "cancelled"
    PAID            = "paid"


# ─────────────────────────────────────────────
# DRIVER
# ─────────────────────────────────────────────

class DriverStatus(Enum):
    AVAILABLE  = "available"
    BUSY       = "busy"
    OFFLINE    = "offline"


@dataclass
class Driver:
    driver_id:  str
    name:       str
    rating:     float
    location:   Location
    status:     DriverStatus = DriverStatus.AVAILABLE
    current_trip_id: Optional[str] = None
    last_update:     float = field(default_factory=time.time)

    def eta_to(self, destination: Location) -> float:
        """Estimated arrival time in minutes (simplified)."""
        dist_km = self.location.distance_km(destination)
        speed   = 30.0   # avg 30 km/h in city
        return dist_km / speed * 60


# ─────────────────────────────────────────────
# RIDER
# ─────────────────────────────────────────────

@dataclass
class Rider:
    rider_id:    str
    name:        str
    rating:      float
    location:    Location


# ─────────────────────────────────────────────
# TRIP
# ─────────────────────────────────────────────

@dataclass
class Trip:
    trip_id:        str
    rider_id:       str
    driver_id:      Optional[str]
    pickup:         Location
    destination:    Location
    state:          TripState
    created_at:     float
    surge:          float = 1.0
    fare_usd:       Optional[float] = None
    started_at:     Optional[float] = None
    completed_at:   Optional[float] = None

    def calculate_fare(self, dist_km: float, duration_min: float) -> float:
        base_fare   = 2.50
        per_km      = 1.20
        per_min     = 0.25
        fare        = (base_fare + per_km * dist_km + per_min * duration_min) * self.surge
        return round(max(5.0, fare), 2)   # minimum fare $5


# ─────────────────────────────────────────────
# LOCATION SERVICE (Redis GEO simulation)
# ─────────────────────────────────────────────

class LocationService:
    """
    Simulates Redis GEOADD / GEORADIUS.
    Stores driver locations; supports proximity search.
    """

    def __init__(self):
        self._positions: Dict[str, Location] = {}
        self._update_count = 0

    def update(self, driver_id: str, location: Location):
        self._positions[driver_id] = location
        self._update_count += 1

    def find_nearby(self, center: Location, radius_km: float,
                    status_filter: Optional[set] = None,
                    drivers: Optional[Dict[str, Driver]] = None,
                    limit: int = 10) -> List[Tuple[str, float]]:
        """Returns [(driver_id, distance_km)] sorted by distance."""
        results = []
        for did, loc in self._positions.items():
            dist = center.distance_km(loc)
            if dist <= radius_km:
                if drivers and status_filter:
                    driver = drivers.get(did)
                    if not driver or driver.status not in status_filter:
                        continue
                results.append((did, dist))
        results.sort(key=lambda x: x[1])
        return results[:limit]


# ─────────────────────────────────────────────
# SURGE PRICING ENGINE
# ─────────────────────────────────────────────

class SurgeEngine:
    def __init__(self):
        # zone → (available_drivers, pending_requests)
        self._zones: Dict[str, Tuple[int, int]] = {}

    def update_zone(self, zone: str, drivers: int, requests: int):
        self._zones[zone] = (drivers, requests)

    def surge_multiplier(self, zone: str) -> float:
        drivers, requests = self._zones.get(zone, (10, 5))
        if drivers <= 0:
            return 3.0
        ratio = requests / drivers
        if ratio <= 0.5:   return 1.0
        if ratio <= 1.0:   return 1.2
        if ratio <= 2.0:   return 1.5
        if ratio <= 3.0:   return 2.0
        return min(3.0, ratio * 0.8)


# ─────────────────────────────────────────────
# MATCHING ENGINE
# ─────────────────────────────────────────────

class MatchingEngine:
    """
    Finds best available driver for a trip request.
    Scores by: ETA (lower = better) + driver rating.
    """

    SEARCH_RADIUS_KM = 5.0
    OFFER_TIMEOUT_S  = 10.0

    def __init__(self, location_svc: LocationService,
                 drivers: Dict[str, Driver]):
        self._loc     = location_svc
        self._drivers = drivers

    def find_best_driver(self, pickup: Location) -> Optional[Driver]:
        candidates = self._loc.find_nearby(
            pickup, self.SEARCH_RADIUS_KM,
            status_filter={DriverStatus.AVAILABLE},
            drivers=self._drivers,
            limit=5,
        )
        if not candidates:
            return None

        # Score: lower ETA + higher rating = better match
        def score(driver_id: str, dist_km: float) -> float:
            driver = self._drivers[driver_id]
            eta    = driver.eta_to(pickup)
            return -eta + driver.rating * 0.5   # maximize this

        best_id = max(candidates, key=lambda x: score(x[0], x[1]))[0]
        return self._drivers[best_id]


# ─────────────────────────────────────────────
# RIDE SHARING SERVICE
# ─────────────────────────────────────────────

class UberService:
    def __init__(self):
        self._drivers:  Dict[str, Driver] = {}
        self._riders:   Dict[str, Rider]  = {}
        self._trips:    Dict[str, Trip]   = {}
        self._loc       = LocationService()
        self._surge     = SurgeEngine()

    def register_driver(self, name: str, lat: float, lng: float,
                        rating: float = 4.8) -> Driver:
        driver = Driver(
            driver_id = uuid.uuid4().hex[:8],
            name      = name,
            rating    = rating,
            location  = Location(lat, lng),
        )
        self._drivers[driver.driver_id] = driver
        self._loc.update(driver.driver_id, driver.location)
        return driver

    def register_rider(self, name: str, lat: float, lng: float) -> Rider:
        rider = Rider(uuid.uuid4().hex[:8], name, 4.9, Location(lat, lng))
        self._riders[rider.rider_id] = rider
        return rider

    def update_driver_location(self, driver_id: str, lat: float, lng: float):
        driver = self._drivers.get(driver_id)
        if driver:
            driver.location = Location(lat, lng)
            driver.last_update = time.time()
            self._loc.update(driver_id, driver.location)

    def request_ride(self, rider_id: str,
                     dest_lat: float, dest_lng: float) -> Optional[Trip]:
        rider = self._riders.get(rider_id)
        if not rider:
            return None

        zone    = rider.location.geohash(4)
        surge   = self._surge.surge_multiplier(zone)

        trip = Trip(
            trip_id     = uuid.uuid4().hex[:10],
            rider_id    = rider_id,
            driver_id   = None,
            pickup      = rider.location,
            destination = Location(dest_lat, dest_lng),
            state       = TripState.REQUESTED,
            created_at  = time.time(),
            surge       = surge,
        )
        self._trips[trip.trip_id] = trip

        # Find driver
        engine = MatchingEngine(self._loc, self._drivers)
        driver = engine.find_best_driver(rider.location)
        if driver:
            driver.status   = DriverStatus.BUSY
            driver.current_trip_id = trip.trip_id
            trip.driver_id  = driver.driver_id
            trip.state      = TripState.MATCHED
        return trip

    def start_trip(self, trip_id: str) -> Trip:
        trip       = self._trips[trip_id]
        trip.state = TripState.IN_PROGRESS
        trip.started_at = time.time()
        return trip

    def complete_trip(self, trip_id: str) -> Trip:
        trip             = self._trips[trip_id]
        trip.state       = TripState.COMPLETED
        trip.completed_at= time.time()

        dist_km  = trip.pickup.distance_km(trip.destination)
        dur_min  = (trip.completed_at - (trip.started_at or trip.created_at)) * 60
        trip.fare_usd = trip.calculate_fare(dist_km, max(5.0, dur_min))
        trip.state = TripState.PAID

        # Free up driver
        driver = self._drivers.get(trip.driver_id)
        if driver:
            driver.status   = DriverStatus.AVAILABLE
            driver.current_trip_id = None
        return trip

    def nearby_drivers(self, lat: float, lng: float,
                       radius_km: float = 3.0) -> List[Tuple[str, float]]:
        return self._loc.find_nearby(Location(lat, lng), radius_km,
                                     drivers=self._drivers)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_uber():
    print("=" * 65)
    print("SYSTEM DESIGN: UBER")
    print("=" * 65)

    random.seed(42)
    svc = UberService()

    # ── Register Drivers ──────────────────────
    print("\n[1] DRIVERS")
    print("─" * 55)

    drivers = []
    coords  = [(37.7749, -122.4194), (37.7751, -122.4188),
               (37.7760, -122.4200), (37.7730, -122.4170),
               (37.7740, -122.4210)]
    names   = ["Carlos", "Priya", "Mike", "Sara", "Ahmed"]
    for (lat, lng), name in zip(coords, names):
        d = svc.register_driver(name, lat + random.uniform(-0.005, 0.005),
                                lng + random.uniform(-0.005, 0.005),
                                rating=round(random.uniform(4.5, 5.0), 1))
        drivers.append(d)
        print(f"  {name:<10} lat={d.location.lat:.4f}  lng={d.location.lng:.4f}  "
              f"rating={d.rating}")

    # ── Register Rider ────────────────────────
    print("\n[2] RIDER REQUESTS RIDE")
    print("─" * 55)

    rider = svc.register_rider("Alice", 37.7749, -122.4194)
    print(f"  Rider: {rider.name}")
    print(f"  Pickup: {rider.location.lat:.4f}, {rider.location.lng:.4f}")
    print(f"  Destination: 37.7800, -122.4100")

    # ── Surge Pricing ─────────────────────────
    zone = rider.location.geohash(4)
    svc._surge.update_zone(zone, drivers=3, requests=7)
    surge = svc._surge.surge_multiplier(zone)
    print(f"\n  Surge zone: {zone}  drivers=3  requests=7  surge={surge:.1f}×")

    # ── Match ─────────────────────────────────
    print("\n[3] MATCHING")
    print("─" * 55)

    trip = svc.request_ride(rider.rider_id, 37.7800, -122.4100)
    if trip:
        driver = svc._drivers[trip.driver_id]
        eta    = driver.eta_to(trip.pickup)
        dist   = driver.location.distance_km(trip.pickup)
        print(f"  Trip ID:   {trip.trip_id}")
        print(f"  State:     {trip.state.value}")
        print(f"  Driver:    {driver.name} (rating {driver.rating})")
        print(f"  Distance:  {dist:.2f}km away")
        print(f"  ETA:       {eta:.1f} min")
        print(f"  Surge:     {trip.surge:.1f}×")

    # ── Trip Lifecycle ────────────────────────
    print("\n[4] TRIP LIFECYCLE")
    print("─" * 55)

    states = [TripState.DRIVER_EN_ROUTE, TripState.ARRIVED, TripState.IN_PROGRESS]
    for state in states:
        trip.state = state
        print(f"  → {state.value}")
        time.sleep(0.01)

    trip.started_at = time.time() - 600   # 10 min ago (simulated)
    completed = svc.complete_trip(trip.trip_id)
    dist_km   = trip.pickup.distance_km(trip.destination)

    print(f"  → {completed.state.value}")
    print(f"\n  Distance:     {dist_km:.2f}km")
    print(f"  Fare (surge {completed.surge:.1f}×): ${completed.fare_usd:.2f}")

    # ── Nearby Drivers ────────────────────────
    print("\n[5] NEARBY DRIVERS")
    print("─" * 55)

    nearby = svc.nearby_drivers(37.7749, -122.4194, radius_km=2.0)
    print(f"  Drivers within 2km of rider:")
    for did, dist in nearby:
        d = svc._drivers[did]
        print(f"    {d.name:<10} {dist:.2f}km  {d.status.value}")

    # ── Geohash ───────────────────────────────
    print("\n[6] GEOHASH ZONES")
    print("─" * 55)

    test_locs = [
        (37.7749, -122.4194, "San Francisco center"),
        (37.7751, -122.4188, "1 block away"),
        (37.8000, -122.2700, "Oakland"),
    ]
    for lat, lng, label in test_locs:
        gh5 = Location(lat, lng).geohash(5)
        gh6 = Location(lat, lng).geohash(6)
        print(f"  {label:<25} geohash5={gh5}  geohash6={gh6}")

    # ── Architecture ──────────────────────────
    print("\n[7] UBER ARCHITECTURE SUMMARY")
    print("─" * 55)

    arch = [
        ("Location updates",  "WebSocket → Kafka → Redis GEO (250K events/sec)"),
        ("Geospatial index",  "Redis GEORADIUS / S2 library / H3 hexagons"),
        ("Matching",          "Kafka consumer → MatchingEngine → offer to driver"),
        ("Trip state",        "Cassandra with trip_id PK + state transitions"),
        ("Surge pricing",     "Demand/supply ratio per geohash zone; updated every 30s"),
        ("ETA",               "Uber's internal router (H3 grid) or Mapbox API"),
        ("Payment",           "Stripe/Braintree; async retry on failure"),
        ("Live tracking",     "WebSocket → server-sent events to rider app"),
        ("Driver matching",   "Offer timeout 10s → next driver in queue"),
        ("Fault tolerance",   "Kafka ensures at-least-once delivery; idempotent consumers"),
    ]
    for component, detail in arch:
        print(f"  {component:<20} {detail}")


if __name__ == "__main__":
    demonstrate_uber()
