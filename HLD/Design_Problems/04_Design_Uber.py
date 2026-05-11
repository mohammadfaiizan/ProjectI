"""
Uber Ride-Sharing System - Working Python Implementation
Demonstrates: geohash-based spatial indexing, driver-rider matching,
              ride lifecycle state machine, surge pricing, ETA estimation,
              driver location tracking, distributed lock (simulated).
No external dependencies — standard library only.
"""

import math
import time
import collections
import random
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from enum import Enum


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
GEOHASH_PRECISION = 5          # ~5km x 5km cells (precision 5)
DRIVER_LOCATION_TTL = 30       # seconds: driver removed if no update
MATCH_RADIUS_KM = 5.0
MATCH_RADIUS_EXPAND_KM = 10.0
MAX_DRIVERS_TO_TRY = 3
SURGE_ZONE_SIZE = 0.1          # ~11km per degree, 0.1 degree ≈ 1.1km zone width
BASE_FARE_PER_KM = 1.5
BASE_FARE_PER_MIN = 0.25
BASE_FARE_FIXED = 2.0


# ---------------------------------------------------------------------------
# Geohash Implementation
# ---------------------------------------------------------------------------
GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

def encode_geohash(lat: float, lng: float, precision: int = GEOHASH_PRECISION) -> str:
    """
    Encode lat/lng to a geohash string.
    Key property: nearby locations share a common prefix.
    Precision 5 = cells ~5km x 5km.
    """
    lat_range = [-90.0, 90.0]
    lng_range = [-180.0, 180.0]
    bits = [16, 8, 4, 2, 1]
    bit_idx = 0
    geohash = []
    even = True  # alternate between lng (even) and lat (odd)
    val = 0

    while len(geohash) < precision:
        if even:
            mid = (lng_range[0] + lng_range[1]) / 2
            if lng >= mid:
                val |= bits[bit_idx]
                lng_range[0] = mid
            else:
                lng_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                val |= bits[bit_idx]
                lat_range[0] = mid
            else:
                lat_range[1] = mid
        even = not even
        if bit_idx < 4:
            bit_idx += 1
        else:
            geohash.append(GEOHASH_BASE32[val])
            val = 0
            bit_idx = 0

    return "".join(geohash)


def get_neighbor_cells(geohash: str) -> List[str]:
    """
    Return the given cell plus its 8 neighbors (for radius search).
    Simplified: generate ±0.05 degree variations at precision 5.
    In production: proper geohash neighbor algorithm.
    """
    # Decode to approximate center
    lat, lng = decode_geohash_center(geohash)
    offsets = [
        (-0.05, -0.05), (-0.05, 0), (-0.05, 0.05),
        (0,     -0.05), (0,     0), (0,     0.05),
        (0.05,  -0.05), (0.05,  0), (0.05,  0.05),
    ]
    cells = set()
    for dlat, dlng in offsets:
        cells.add(encode_geohash(lat + dlat, lng + dlng, len(geohash)))
    return list(cells)


def decode_geohash_center(geohash: str) -> Tuple[float, float]:
    """Approximate center of a geohash cell."""
    lat_range = [-90.0, 90.0]
    lng_range = [-180.0, 180.0]
    even = True
    for char in geohash:
        cd = GEOHASH_BASE32.index(char)
        for mask in [16, 8, 4, 2, 1]:
            if even:
                mid = (lng_range[0] + lng_range[1]) / 2
                if cd & mask:
                    lng_range[0] = mid
                else:
                    lng_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if cd & mask:
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid
            even = not even
    return (lat_range[0] + lat_range[1]) / 2, (lng_range[0] + lng_range[1]) / 2


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate great-circle distance between two GPS coordinates.
    Uses the Haversine formula.
    """
    R = EARTH_RADIUS_KM
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Ride State Machine
# ---------------------------------------------------------------------------
class RideStatus(Enum):
    REQUESTED      = "requested"
    SEARCHING      = "searching"
    ACCEPTED       = "accepted"
    DRIVER_ARRIVING = "driver_arriving"
    IN_PROGRESS    = "in_progress"
    COMPLETED      = "completed"
    CANCELLED      = "cancelled"
    FAILED         = "failed"


class DriverStatus(Enum):
    OFFLINE    = "offline"
    AVAILABLE  = "available"
    ON_TRIP    = "on_trip"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
class Driver:
    def __init__(self, driver_id: int, name: str, vehicle_type: str = "uberx"):
        self.driver_id = driver_id
        self.name = name
        self.vehicle_type = vehicle_type
        self.rating = 4.8
        self.status = DriverStatus.OFFLINE
        self.lat: Optional[float] = None
        self.lng: Optional[float] = None
        self.current_trip_id: Optional[str] = None
        self.last_location_update: float = 0.0
        self.total_trips = 0

    def update_location(self, lat: float, lng: float) -> None:
        self.lat = lat
        self.lng = lng
        self.last_location_update = time.time()

    def __repr__(self):
        return f"Driver({self.driver_id}, {self.name}, {self.status.value})"


class Rider:
    def __init__(self, rider_id: int, name: str):
        self.rider_id = rider_id
        self.name = name
        self.rating = 4.9
        self.total_rides = 0

    def __repr__(self):
        return f"Rider({self.rider_id}, {self.name})"


class Trip:
    _counter = 1000

    def __init__(self, rider_id: int, pickup_lat: float, pickup_lng: float,
                 dest_lat: float, dest_lng: float, ride_type: str = "uberx"):
        Trip._counter += 1
        self.trip_id = f"trip_{Trip._counter}"
        self.rider_id = rider_id
        self.driver_id: Optional[int] = None
        self.status = RideStatus.REQUESTED
        self.pickup_lat = pickup_lat
        self.pickup_lng = pickup_lng
        self.dest_lat = dest_lat
        self.dest_lng = dest_lng
        self.ride_type = ride_type
        self.surge_multiplier = 1.0
        self.base_fare: Optional[float] = None
        self.total_fare: Optional[float] = None
        self.distance_km: Optional[float] = None
        self.duration_min: Optional[int] = None
        self.created_at = datetime.utcnow()
        self.accepted_at: Optional[datetime] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.version = 0           # optimistic locking

    def transition(self, new_status: RideStatus) -> bool:
        """
        Validate and apply state transition.
        Returns True if successful.
        """
        valid_transitions = {
            RideStatus.REQUESTED:       [RideStatus.SEARCHING, RideStatus.CANCELLED],
            RideStatus.SEARCHING:       [RideStatus.ACCEPTED, RideStatus.FAILED, RideStatus.CANCELLED],
            RideStatus.ACCEPTED:        [RideStatus.DRIVER_ARRIVING, RideStatus.CANCELLED],
            RideStatus.DRIVER_ARRIVING: [RideStatus.IN_PROGRESS, RideStatus.CANCELLED],
            RideStatus.IN_PROGRESS:     [RideStatus.COMPLETED],
            RideStatus.COMPLETED:       [],
            RideStatus.CANCELLED:       [],
            RideStatus.FAILED:          [],
        }
        if new_status not in valid_transitions.get(self.status, []):
            return False
        self.status = new_status
        self.version += 1
        return True

    def __repr__(self):
        return f"Trip({self.trip_id}, {self.status.value}, rider={self.rider_id})"


# ---------------------------------------------------------------------------
# Surge Pricing Engine
# ---------------------------------------------------------------------------
class SurgePricingEngine:
    """
    Calculates surge multiplier per geographic zone.
    Zone = rounded lat/lng grid (simplified; production uses H3 hexagons).
    """

    def __init__(self, zone_size: float = SURGE_ZONE_SIZE):
        self.zone_size = zone_size
        # { zone_id: { "demand": count, "supply": count, "last_updated": ts } }
        self._zones: Dict[str, dict] = collections.defaultdict(
            lambda: {"demand": 0, "supply": 0, "last_updated": time.time()}
        )

    def _zone_id(self, lat: float, lng: float) -> str:
        """Round lat/lng to zone grid."""
        z_lat = round(lat / self.zone_size) * self.zone_size
        z_lng = round(lng / self.zone_size) * self.zone_size
        return f"zone_{z_lat:.2f}_{z_lng:.2f}"

    def record_ride_request(self, lat: float, lng: float) -> None:
        zone = self._zone_id(lat, lng)
        self._zones[zone]["demand"] += 1

    def record_available_driver(self, lat: float, lng: float) -> None:
        zone = self._zone_id(lat, lng)
        self._zones[zone]["supply"] += 1

    def get_surge_multiplier(self, lat: float, lng: float) -> float:
        """
        Surge multiplier = 1.0 + 0.5 * max(0, (demand - supply) / max(supply, 1))
        Capped at 4.9x.
        """
        zone = self._zone_id(lat, lng)
        data = self._zones[zone]
        demand = data["demand"]
        supply = max(data["supply"], 1)   # avoid division by zero
        excess_ratio = max(0.0, (demand - supply) / supply)
        multiplier = 1.0 + 0.5 * excess_ratio
        return round(min(multiplier, 4.9), 1)

    def get_zone_info(self, lat: float, lng: float) -> dict:
        zone = self._zone_id(lat, lng)
        data = self._zones[zone]
        return {
            "zone_id": zone,
            "demand": data["demand"],
            "supply": data["supply"],
            "surge_multiplier": self.get_surge_multiplier(lat, lng),
        }


# ---------------------------------------------------------------------------
# Uber Core System
# ---------------------------------------------------------------------------
class UberSystem:
    """
    Core Uber ride-sharing system.
    Features: location tracking, geohash spatial index, matching,
              ride state machine, surge pricing, fare calculation, ETA.
    """

    def __init__(self):
        self._drivers: Dict[int, Driver] = {}
        self._riders: Dict[int, Rider] = {}
        self._trips: Dict[str, Trip] = {}
        self._surge_engine = SurgePricingEngine()

        # Spatial index: geohash -> set of available driver_ids
        self._geo_index: Dict[str, set] = collections.defaultdict(set)

        # Distributed lock simulation: trip_id -> driver_id (accepted)
        self._trip_locks: Dict[str, int] = {}

        # Driver's current geohash (for efficient removal on location update)
        self._driver_geohash: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def add_driver(self, driver_id: int, name: str) -> Driver:
        d = Driver(driver_id, name)
        self._drivers[driver_id] = d
        return d

    def add_rider(self, rider_id: int, name: str) -> Rider:
        r = Rider(rider_id, name)
        self._riders[rider_id] = r
        return r

    # ------------------------------------------------------------------
    # Driver Location Update
    # ------------------------------------------------------------------

    def update_driver_location(self, driver_id: int, lat: float, lng: float) -> dict:
        """
        Driver sends GPS update every 4 seconds.
        Updates Redis GEO (simulated here as geohash index).
        """
        driver = self._drivers.get(driver_id)
        if not driver:
            return {"error": "driver not found"}

        driver.update_location(lat, lng)

        # Update geo-spatial index
        new_hash = encode_geohash(lat, lng)

        # Remove from old geohash cell
        old_hash = self._driver_geohash.get(driver_id)
        if old_hash and old_hash != new_hash:
            self._geo_index[old_hash].discard(driver_id)

        # Add to new geohash cell (only if available)
        if driver.status == DriverStatus.AVAILABLE:
            self._geo_index[new_hash].add(driver_id)
            self._surge_engine.record_available_driver(lat, lng)

        self._driver_geohash[driver_id] = new_hash

        return {"received": True, "geohash": new_hash}

    def set_driver_available(self, driver_id: int, lat: float, lng: float) -> None:
        driver = self._drivers[driver_id]
        driver.status = DriverStatus.AVAILABLE
        self.update_driver_location(driver_id, lat, lng)

    # ------------------------------------------------------------------
    # Find Nearby Drivers
    # ------------------------------------------------------------------

    def find_nearby_drivers(
        self, lat: float, lng: float, radius_km: float = MATCH_RADIUS_KM, limit: int = 10
    ) -> List[Tuple[float, Driver]]:
        """
        Find available drivers within radius_km of (lat, lng).
        Uses geohash spatial index for O(1) cell lookup.
        Returns list of (distance_km, Driver) sorted by distance.
        """
        cell = encode_geohash(lat, lng)
        neighbor_cells = get_neighbor_cells(cell)

        candidates = []
        seen = set()
        for c in neighbor_cells:
            for driver_id in self._geo_index.get(c, set()):
                if driver_id in seen:
                    continue
                seen.add(driver_id)
                driver = self._drivers[driver_id]
                if driver.status != DriverStatus.AVAILABLE or driver.lat is None:
                    continue
                dist = haversine_km(lat, lng, driver.lat, driver.lng)
                if dist <= radius_km:
                    candidates.append((dist, driver))

        candidates.sort(key=lambda x: x[0])
        return candidates[:limit]

    # ------------------------------------------------------------------
    # Request & Match Ride
    # ------------------------------------------------------------------

    def request_ride(self, rider_id: int, pickup_lat: float, pickup_lng: float,
                     dest_lat: float, dest_lng: float) -> dict:
        """
        Rider requests a ride. System creates a trip and starts matching.
        """
        if rider_id not in self._riders:
            return {"error": "rider not found"}

        # Get surge multiplier for pickup location
        surge = self._surge_engine.get_surge_multiplier(pickup_lat, pickup_lng)
        self._surge_engine.record_ride_request(pickup_lat, pickup_lng)

        trip = Trip(rider_id, pickup_lat, pickup_lng, dest_lat, dest_lng)
        trip.surge_multiplier = surge
        trip.transition(RideStatus.SEARCHING)
        self._trips[trip.trip_id] = trip

        # Estimate fare
        dist = haversine_km(pickup_lat, pickup_lng, dest_lat, dest_lng)
        base = BASE_FARE_FIXED + dist * BASE_FARE_PER_KM
        trip.distance_km = round(dist, 2)

        # Pickup ETA estimate (simplified)
        nearby = self.find_nearby_drivers(pickup_lat, pickup_lng, radius_km=5.0, limit=1)
        eta_min = int(nearby[0][0] / 30 * 60) if nearby else 5  # assume 30 km/h avg

        return {
            "trip_id": trip.trip_id,
            "status": trip.status.value,
            "surge_multiplier": surge,
            "estimated_fare_min": round(base * surge, 2),
            "estimated_fare_max": round(base * surge * 1.2, 2),
            "estimated_pickup_eta_min": eta_min,
            "distance_km": dist,
        }

    def match_driver(self, trip_id: str) -> dict:
        """
        Attempt to match a driver to a trip.
        Sends request to top-3 nearest drivers; first to accept wins.
        Uses atomic lock to prevent double-accept.
        """
        trip = self._trips.get(trip_id)
        if not trip:
            return {"error": "trip not found"}

        nearby = self.find_nearby_drivers(
            trip.pickup_lat, trip.pickup_lng,
            radius_km=MATCH_RADIUS_KM,
            limit=MAX_DRIVERS_TO_TRY,
        )

        if not nearby:
            # Expand search radius
            nearby = self.find_nearby_drivers(
                trip.pickup_lat, trip.pickup_lng,
                radius_km=MATCH_RADIUS_EXPAND_KM,
                limit=MAX_DRIVERS_TO_TRY,
            )

        if not nearby:
            trip.transition(RideStatus.FAILED)
            return {"error": "no drivers available", "trip_id": trip_id}

        # Simulate: first driver in list accepts (in real system, they all get notified)
        for dist, driver in nearby:
            result = self.driver_accept_ride(driver.driver_id, trip_id)
            if result.get("accepted"):
                return result

        trip.transition(RideStatus.FAILED)
        return {"error": "all drivers rejected", "trip_id": trip_id}

    def driver_accept_ride(self, driver_id: int, trip_id: str) -> dict:
        """
        Driver accepts a ride request.
        Atomic: only the first driver to call this succeeds (distributed lock).
        """
        trip = self._trips.get(trip_id)
        driver = self._drivers.get(driver_id)

        if not trip or not driver:
            return {"error": "not found"}
        if trip.status != RideStatus.SEARCHING:
            return {"error": "trip not in searchable state"}

        # Distributed lock: SET ride:lock:{trip_id} driver_id NX (atomic in Redis)
        if trip_id in self._trip_locks:
            return {"accepted": False, "error": "ride already accepted by another driver"}

        # Acquire lock
        self._trip_locks[trip_id] = driver_id

        # Update trip
        trip.driver_id = driver_id
        trip.transition(RideStatus.ACCEPTED)
        trip.transition(RideStatus.DRIVER_ARRIVING)
        trip.accepted_at = datetime.utcnow()

        # Update driver state
        driver.status = DriverStatus.ON_TRIP
        driver.current_trip_id = trip_id

        # Remove driver from geo index
        old_hash = self._driver_geohash.get(driver_id)
        if old_hash:
            self._geo_index[old_hash].discard(driver_id)

        rider = self._riders[trip.rider_id]
        return {
            "accepted": True,
            "trip_id": trip_id,
            "rider": {"name": rider.name, "rating": rider.rating},
            "pickup": {"lat": trip.pickup_lat, "lng": trip.pickup_lng},
        }

    # ------------------------------------------------------------------
    # Ride Lifecycle
    # ------------------------------------------------------------------

    def start_ride(self, driver_id: int, trip_id: str) -> dict:
        """Driver picks up rider — ride begins."""
        trip = self._trips.get(trip_id)
        if not trip or trip.driver_id != driver_id:
            return {"error": "unauthorized or not found"}
        if not trip.transition(RideStatus.IN_PROGRESS):
            return {"error": f"invalid transition from {trip.status.value}"}
        trip.started_at = datetime.utcnow()
        return {"status": "in_progress", "trip_id": trip_id}

    def complete_ride(self, driver_id: int, trip_id: str) -> dict:
        """
        Driver completes the ride.
        Calculate fare: base + distance * rate * surge + time * rate.
        """
        trip = self._trips.get(trip_id)
        driver = self._drivers.get(driver_id)
        if not trip or trip.driver_id != driver_id:
            return {"error": "unauthorized or not found"}
        if not trip.transition(RideStatus.COMPLETED):
            return {"error": f"cannot complete from state {trip.status.value}"}

        trip.completed_at = datetime.utcnow()

        # Calculate fare
        dist = trip.distance_km or haversine_km(
            trip.pickup_lat, trip.pickup_lng, trip.dest_lat, trip.dest_lng
        )
        duration_s = (trip.completed_at - (trip.started_at or trip.accepted_at)).total_seconds()
        duration_min = max(1, int(duration_s / 60))
        trip.duration_min = duration_min

        base = BASE_FARE_FIXED + dist * BASE_FARE_PER_KM + duration_min * BASE_FARE_PER_MIN
        total = round(base * trip.surge_multiplier, 2)

        trip.base_fare = round(base, 2)
        trip.total_fare = total

        # Free up driver
        driver.status = DriverStatus.AVAILABLE
        driver.current_trip_id = None
        driver.total_trips += 1
        del self._trip_locks[trip_id]

        return {
            "trip_id": trip_id,
            "status": "completed",
            "distance_km": round(dist, 2),
            "duration_min": duration_min,
            "base_fare": trip.base_fare,
            "surge_multiplier": trip.surge_multiplier,
            "total_fare": total,
        }

    def cancel_ride(self, trip_id: str, cancelled_by: str = "rider") -> dict:
        trip = self._trips.get(trip_id)
        if not trip:
            return {"error": "trip not found"}
        if not trip.transition(RideStatus.CANCELLED):
            return {"error": "cannot cancel from current state"}

        # Free driver if assigned
        if trip.driver_id:
            driver = self._drivers.get(trip.driver_id)
            if driver:
                driver.status = DriverStatus.AVAILABLE
                driver.current_trip_id = None
        self._trip_locks.pop(trip_id, None)

        return {"trip_id": trip_id, "status": "cancelled", "cancelled_by": cancelled_by}

    # ------------------------------------------------------------------
    # Surge Info
    # ------------------------------------------------------------------

    def get_surge_info(self, lat: float, lng: float) -> dict:
        return self._surge_engine.get_zone_info(lat, lng)

    # ------------------------------------------------------------------
    # ETA Estimate (simplified Haversine-based)
    # ------------------------------------------------------------------

    def estimate_eta(self, origin_lat: float, origin_lng: float,
                     dest_lat: float, dest_lng: float,
                     speed_kmh: float = 30.0) -> int:
        """
        Estimate travel time in minutes.
        Production: Google Maps Directions API with real traffic data.
        """
        dist = haversine_km(origin_lat, origin_lng, dest_lat, dest_lng)
        hours = dist / speed_kmh
        return max(1, int(hours * 60))


# ---------------------------------------------------------------------------
# Demo / Simulation
# ---------------------------------------------------------------------------
def run_demo():
    print("=" * 65)
    print("UBER SYSTEM DEMO")
    print("=" * 65)

    uber = UberSystem()

    # Manhattan approximate coordinates
    MIDTOWN_LAT, MIDTOWN_LNG = 40.7549, -73.9840
    BROOKLYN_LAT, BROOKLYN_LNG = 40.6782, -73.9442

    # Create users
    rider1 = uber.add_rider(101, "Alice")
    rider2 = uber.add_rider(102, "Bob")
    d1 = uber.add_driver(201, "Carlos")
    d2 = uber.add_driver(202, "Diana")
    d3 = uber.add_driver(203, "Eve")

    print(f"\n[1] Created {len(uber._riders)} riders and {len(uber._drivers)} drivers")

    # Drivers come online near midtown
    print("\n[2] Drivers come online and send location updates")
    uber.set_driver_available(201, 40.7580, -73.9855)   # 0.3 km from midtown
    uber.set_driver_available(202, 40.7600, -73.9800)   # 0.6 km from midtown
    uber.set_driver_available(203, 40.7500, -73.9900)   # 0.7 km from midtown

    # Find nearby drivers
    print("\n[3] Finding drivers near midtown (5km radius)")
    nearby = uber.find_nearby_drivers(MIDTOWN_LAT, MIDTOWN_LNG, radius_km=5.0)
    print(f"    Found {len(nearby)} drivers:")
    for dist, drv in nearby:
        print(f"    - {drv.name}: {dist:.2f} km away")

    # Simulate surge (high demand)
    print("\n[4] Surge pricing simulation")
    # Record many ride requests to inflate demand
    for _ in range(15):
        uber._surge_engine.record_ride_request(MIDTOWN_LAT, MIDTOWN_LNG)
    for _ in range(3):
        uber._surge_engine.record_available_driver(MIDTOWN_LAT, MIDTOWN_LNG)
    zone_info = uber.get_surge_info(MIDTOWN_LAT, MIDTOWN_LNG)
    print(f"    Zone demand: {zone_info['demand']}, supply: {zone_info['supply']}")
    print(f"    Surge multiplier: {zone_info['surge_multiplier']}x")

    # Request ride
    print("\n[5] Alice requests a ride (Midtown -> Brooklyn)")
    ride = uber.request_ride(101, MIDTOWN_LAT, MIDTOWN_LNG, BROOKLYN_LAT, BROOKLYN_LNG)
    print(f"    Trip ID       : {ride['trip_id']}")
    print(f"    Surge         : {ride['surge_multiplier']}x")
    print(f"    Est. fare     : ${ride['estimated_fare_min']} - ${ride['estimated_fare_max']}")
    print(f"    Distance      : {ride['distance_km']:.1f} km")
    print(f"    Pickup ETA    : ~{ride['estimated_pickup_eta_min']} min")

    trip_id = ride["trip_id"]

    # Match driver
    print("\n[6] Matching driver to trip")
    match = uber.match_driver(trip_id)
    if "error" not in match:
        matched_driver_id = uber._trips[trip_id].driver_id
        matched_driver = uber._drivers[matched_driver_id]
        print(f"    Matched driver: {matched_driver.name} (id={matched_driver_id})")
        print(f"    Driver status : {matched_driver.status.value}")
        print(f"    Trip status   : {uber._trips[trip_id].status.value}")
    else:
        print(f"    Match failed: {match['error']}")

    # Test distributed lock (second driver tries to accept same ride)
    print("\n[7] Distributed lock — prevent double-accept")
    lock_test = uber.driver_accept_ride(202, trip_id)
    print(f"    Second driver accept result: {lock_test.get('error', 'accepted (WRONG!)')}")

    # Start ride
    print("\n[8] Ride lifecycle")
    start = uber.start_ride(matched_driver_id, trip_id)
    print(f"    Start ride: {start['status']}")

    # Simulate location updates during ride
    print("\n[9] Driver location updates during ride")
    route_points = [
        (40.7520, -73.9820),
        (40.7450, -73.9700),
        (40.7200, -73.9550),
        (40.6900, -73.9480),
    ]
    for lat, lng in route_points:
        uber.update_driver_location(matched_driver_id, lat, lng)
        eta = uber.estimate_eta(lat, lng, BROOKLYN_LAT, BROOKLYN_LNG)
        print(f"    Driver at ({lat:.3f}, {lng:.3f}), ETA to dest: {eta} min")

    # Complete ride
    print("\n[10] Complete ride")
    completion = uber.complete_ride(matched_driver_id, trip_id)
    print(f"    Trip completed!")
    print(f"    Distance   : {completion['distance_km']} km")
    print(f"    Duration   : {completion['duration_min']} min")
    print(f"    Base fare  : ${completion['base_fare']}")
    print(f"    Surge      : {completion['surge_multiplier']}x")
    print(f"    Total fare : ${completion['total_fare']}")

    # Driver back to available
    print(f"\n    Driver status after completion: {uber._drivers[matched_driver_id].status.value}")

    # Bob requests ride — tests fresh availability
    print("\n[11] Bob requests a ride (with 2 available drivers)")
    ride2 = uber.request_ride(102, 40.7600, -73.9800, 40.7282, -73.7949)
    print(f"    Trip ID: {ride2['trip_id']}, status={uber._trips[ride2['trip_id']].status.value}")
    match2 = uber.match_driver(ride2["trip_id"])
    print(f"    Match result: accepted={match2.get('accepted', False)}, error={match2.get('error', 'none')}")

    # Cancellation test
    print("\n[12] Cancellation test")
    ride3 = uber.request_ride(101, 40.75, -73.98, 40.70, -73.94)
    cancel = uber.cancel_ride(ride3["trip_id"], cancelled_by="rider")
    print(f"    Cancel result: {cancel['status']}")

    # Geohash demo
    print("\n[13] Geohash encoding")
    test_points = [
        (40.7549, -73.9840, "Midtown Manhattan"),
        (40.6501, -73.9496, "Flatbush Brooklyn"),
        (37.7749, -122.4194, "San Francisco"),
    ]
    for lat, lng, name in test_points:
        gh = encode_geohash(lat, lng, precision=5)
        decoded_lat, decoded_lng = decode_geohash_center(gh)
        print(f"    {name:25} -> geohash={gh}  (decoded: {decoded_lat:.3f}, {decoded_lng:.3f})")

    print("\n" + "=" * 65)
    print("DEMO COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
