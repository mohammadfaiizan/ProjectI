"""
RIDE SHARING SYSTEM - Complete System Design
===========================================

Problem Statement:
Design a comprehensive ride sharing system that handles:
- User registration and profile management (riders and drivers)
- Ride booking and matching algorithms
- Real-time location tracking and navigation
- Dynamic pricing and fare calculation
- Payment processing and billing
- Driver and vehicle management
- Ride rating and feedback system
- Trip history and analytics
- Safety features and emergency protocols
- Multi-city operations and expansion
- Integration with maps and navigation services

Requirements:
- Support rider and driver registration with verification
- Implement efficient ride matching based on location and preferences
- Handle real-time location updates and tracking
- Provide dynamic pricing based on demand and supply
- Support multiple payment methods and billing
- Manage driver profiles, vehicle information, and documents
- Implement rating and feedback system for quality control
- Provide comprehensive trip analytics and reporting
- Handle safety features including emergency contacts
- Support multiple vehicle types and service levels
- Scale to handle high-volume concurrent requests

Design Patterns Used:
- Strategy: Pricing and matching algorithms
- State: Trip state management
- Observer: Real-time location and status updates
- Factory: Trip and payment creation
- Command: Trip operations with history
- Template Method: Trip workflow
- Proxy: External service integration (maps, payments)
- Decorator: Additional services and features
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import time
import math
import random
import json
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class UserType(Enum):
    RIDER = "rider"
    DRIVER = "driver"
    ADMIN = "admin"


class VehicleType(Enum):
    ECONOMY = "economy"
    COMFORT = "comfort"
    PREMIUM = "premium"
    LUXURY = "luxury"
    SUV = "suv"
    BIKE = "bike"


class TripStatus(Enum):
    REQUESTED = "requested"
    SEARCHING = "searching"
    MATCHED = "matched"
    ACCEPTED = "accepted"
    DRIVER_ARRIVING = "driver_arriving"
    DRIVER_ARRIVED = "driver_arrived"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PAYMENT_PENDING = "payment_pending"


class PaymentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class DriverStatus(Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    BUSY = "busy"
    BREAK = "break"


@dataclass
class Location:
    """Geographic location."""
    latitude: float
    longitude: float
    address: str = ""
    city: str = ""
    
    def distance_to(self, other: 'Location') -> float:
        """Calculate distance to another location in kilometers."""
        # Haversine formula
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        delta_lat = math.radians(other.latitude - self.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def __str__(self) -> str:
        return f"{self.address or f'({self.latitude:.6f}, {self.longitude:.6f})'}"


@dataclass
class User:
    """Base user information."""
    user_id: str
    email: str
    phone: str
    first_name: str
    last_name: str
    user_type: UserType
    
    created_at: datetime = field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    
    # Profile
    profile_image_url: str = ""
    rating: float = 5.0
    total_ratings: int = 0
    
    # Verification
    is_verified: bool = False
    verification_documents: List[str] = field(default_factory=list)
    
    # Settings
    preferred_language: str = "en"
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.user_id:
            self.user_id = str(uuid.uuid4())
        
        if not self.notification_preferences:
            self.notification_preferences = {
                'trip_updates': True,
                'promotions': True,
                'safety_alerts': True
            }
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass
class Vehicle:
    """Vehicle information."""
    vehicle_id: str
    driver_id: str
    make: str
    model: str
    year: int
    color: str
    license_plate: str
    vehicle_type: VehicleType
    
    # Capacity
    max_passengers: int = 4
    
    # Documents
    registration_number: str = ""
    insurance_policy: str = ""
    inspection_date: Optional[datetime] = None
    
    # Status
    is_active: bool = True
    
    def __post_init__(self):
        if not self.vehicle_id:
            self.vehicle_id = str(uuid.uuid4())


@dataclass
class Driver(User):
    """Driver-specific information."""
    license_number: str = ""
    license_expiry: Optional[datetime] = None
    
    # Status
    status: DriverStatus = DriverStatus.OFFLINE
    current_location: Optional[Location] = None
    
    # Vehicle
    vehicle: Optional[Vehicle] = None
    
    # Statistics
    total_trips: int = 0
    total_earnings: float = 0.0
    acceptance_rate: float = 1.0
    cancellation_rate: float = 0.0
    
    # Preferences
    max_pickup_distance: float = 10.0  # kilometers
    preferred_areas: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        self.user_type = UserType.DRIVER
    
    def update_location(self, location: Location) -> None:
        """Update driver's current location."""
        self.current_location = location
        self.last_active = datetime.now()


@dataclass
class Rider(User):
    """Rider-specific information."""
    # Preferences
    preferred_vehicle_types: List[VehicleType] = field(default_factory=list)
    saved_locations: Dict[str, Location] = field(default_factory=dict)
    
    # Statistics
    total_trips: int = 0
    
    # Payment
    default_payment_method: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.user_type = UserType.RIDER


@dataclass
class TripRequest:
    """Trip request information."""
    request_id: str
    rider_id: str
    pickup_location: Location
    destination: Location
    vehicle_type: VehicleType
    
    created_at: datetime = field(default_factory=datetime.now)
    
    # Preferences
    max_wait_time: int = 10  # minutes
    notes: str = ""
    
    # Estimated values
    estimated_distance: float = 0.0
    estimated_duration: int = 0  # minutes
    estimated_fare: float = 0.0
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        
        if self.estimated_distance == 0.0:
            self.estimated_distance = self.pickup_location.distance_to(self.destination)
        
        if self.estimated_duration == 0:
            # Rough estimate: 30 km/h average speed in city
            self.estimated_duration = int(self.estimated_distance * 2)


@dataclass
class Trip:
    """Trip information."""
    trip_id: str
    rider_id: str
    driver_id: Optional[str] = None
    request: Optional[TripRequest] = None
    
    status: TripStatus = TripStatus.REQUESTED
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    matched_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Route
    actual_pickup_location: Optional[Location] = None
    actual_destination: Optional[Location] = None
    route_points: List[Location] = field(default_factory=list)
    
    # Metrics
    actual_distance: float = 0.0
    actual_duration: int = 0  # minutes
    
    # Financial
    base_fare: float = 0.0
    surge_multiplier: float = 1.0
    total_fare: float = 0.0
    driver_earnings: float = 0.0
    platform_fee: float = 0.0
    
    # Ratings
    rider_rating: Optional[int] = None
    driver_rating: Optional[int] = None
    rider_feedback: str = ""
    driver_feedback: str = ""
    
    def __post_init__(self):
        if not self.trip_id:
            self.trip_id = str(uuid.uuid4())


@dataclass
class Payment:
    """Payment information."""
    payment_id: str
    trip_id: str
    user_id: str
    amount: float
    currency: str = "USD"
    
    payment_method: str = ""
    status: PaymentStatus = PaymentStatus.PENDING
    
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    
    # Transaction details
    transaction_id: Optional[str] = None
    gateway_response: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.payment_id:
            self.payment_id = str(uuid.uuid4())


# ============================================================================
# PRICING STRATEGIES
# ============================================================================

class PricingStrategy(ABC):
    """Abstract pricing strategy."""
    
    @abstractmethod
    def calculate_fare(self, trip_request: TripRequest, surge_multiplier: float = 1.0) -> float:
        """Calculate fare for trip request."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class StandardPricing(PricingStrategy):
    """Standard distance and time-based pricing."""
    
    def __init__(self):
        # Base fares by vehicle type
        self.base_fares = {
            VehicleType.ECONOMY: 2.50,
            VehicleType.COMFORT: 3.00,
            VehicleType.PREMIUM: 4.00,
            VehicleType.LUXURY: 6.00,
            VehicleType.SUV: 4.50,
            VehicleType.BIKE: 1.50
        }
        
        # Per-kilometer rates
        self.per_km_rates = {
            VehicleType.ECONOMY: 1.20,
            VehicleType.COMFORT: 1.50,
            VehicleType.PREMIUM: 2.00,
            VehicleType.LUXURY: 3.00,
            VehicleType.SUV: 1.80,
            VehicleType.BIKE: 0.80
        }
        
        # Per-minute rates (for waiting time)
        self.per_minute_rates = {
            VehicleType.ECONOMY: 0.25,
            VehicleType.COMFORT: 0.30,
            VehicleType.PREMIUM: 0.40,
            VehicleType.LUXURY: 0.50,
            VehicleType.SUV: 0.35,
            VehicleType.BIKE: 0.15
        }
    
    def calculate_fare(self, trip_request: TripRequest, surge_multiplier: float = 1.0) -> float:
        """Calculate standard fare."""
        vehicle_type = trip_request.vehicle_type
        
        base_fare = self.base_fares.get(vehicle_type, 2.50)
        distance_fare = trip_request.estimated_distance * self.per_km_rates.get(vehicle_type, 1.20)
        time_fare = trip_request.estimated_duration * self.per_minute_rates.get(vehicle_type, 0.25)
        
        subtotal = base_fare + distance_fare + time_fare
        total_fare = subtotal * surge_multiplier
        
        return round(total_fare, 2)
    
    def get_strategy_name(self) -> str:
        return "Standard Pricing"


class DynamicPricing(PricingStrategy):
    """Dynamic pricing based on demand and supply."""
    
    def __init__(self):
        self.standard_pricing = StandardPricing()
        self.demand_multipliers = {
            "high": 1.5,
            "very_high": 2.0,
            "extreme": 2.5
        }
    
    def calculate_fare(self, trip_request: TripRequest, surge_multiplier: float = 1.0) -> float:
        """Calculate dynamic fare with demand-based pricing."""
        base_fare = self.standard_pricing.calculate_fare(trip_request, 1.0)
        
        # Apply surge multiplier
        dynamic_fare = base_fare * surge_multiplier
        
        # Apply additional factors
        time_of_day_multiplier = self._get_time_of_day_multiplier()
        weather_multiplier = self._get_weather_multiplier()
        
        final_fare = dynamic_fare * time_of_day_multiplier * weather_multiplier
        
        return round(final_fare, 2)
    
    def _get_time_of_day_multiplier(self) -> float:
        """Get multiplier based on time of day."""
        hour = datetime.now().hour
        
        # Peak hours: 7-9 AM and 5-7 PM
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            return 1.2
        # Late night: 11 PM - 5 AM
        elif hour >= 23 or hour <= 5:
            return 1.3
        else:
            return 1.0
    
    def _get_weather_multiplier(self) -> float:
        """Get multiplier based on weather conditions."""
        # Simplified weather simulation
        weather_conditions = ["sunny", "rainy", "stormy"]
        current_weather = random.choice(weather_conditions)
        
        if current_weather == "rainy":
            return 1.1
        elif current_weather == "stormy":
            return 1.2
        else:
            return 1.0
    
    def get_strategy_name(self) -> str:
        return "Dynamic Pricing"


# ============================================================================
# MATCHING ALGORITHMS
# ============================================================================

class MatchingAlgorithm(ABC):
    """Abstract driver matching algorithm."""
    
    @abstractmethod
    def find_driver(self, trip_request: TripRequest, available_drivers: List[Driver]) -> Optional[Driver]:
        """Find best driver for trip request."""
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get algorithm name."""
        pass


class NearestDriverMatching(MatchingAlgorithm):
    """Match with nearest available driver."""
    
    def find_driver(self, trip_request: TripRequest, available_drivers: List[Driver]) -> Optional[Driver]:
        """Find nearest driver."""
        if not available_drivers:
            return None
        
        nearest_driver = None
        min_distance = float('inf')
        
        for driver in available_drivers:
            if not driver.current_location:
                continue
            
            # Check if driver accepts this vehicle type
            if driver.vehicle and driver.vehicle.vehicle_type != trip_request.vehicle_type:
                continue
            
            distance = driver.current_location.distance_to(trip_request.pickup_location)
            
            # Check if within driver's preferred pickup distance
            if distance <= driver.max_pickup_distance and distance < min_distance:
                min_distance = distance
                nearest_driver = driver
        
        return nearest_driver
    
    def get_algorithm_name(self) -> str:
        return "Nearest Driver"


class SmartMatching(MatchingAlgorithm):
    """Smart matching considering multiple factors."""
    
    def find_driver(self, trip_request: TripRequest, available_drivers: List[Driver]) -> Optional[Driver]:
        """Find best driver using multiple criteria."""
        if not available_drivers:
            return None
        
        scored_drivers = []
        
        for driver in available_drivers:
            if not driver.current_location or not driver.vehicle:
                continue
            
            # Check vehicle type compatibility
            if driver.vehicle.vehicle_type != trip_request.vehicle_type:
                continue
            
            distance = driver.current_location.distance_to(trip_request.pickup_location)
            
            # Check if within pickup distance
            if distance > driver.max_pickup_distance:
                continue
            
            score = self._calculate_driver_score(driver, trip_request, distance)
            scored_drivers.append((driver, score))
        
        if not scored_drivers:
            return None
        
        # Sort by score (higher is better) and return best driver
        scored_drivers.sort(key=lambda x: x[1], reverse=True)
        return scored_drivers[0][0]
    
    def _calculate_driver_score(self, driver: Driver, trip_request: TripRequest, distance: float) -> float:
        """Calculate driver matching score."""
        score = 100.0  # Base score
        
        # Distance factor (closer is better)
        distance_score = max(0, 50 - distance * 5)  # Penalize distance
        
        # Rating factor
        rating_score = driver.rating * 10
        
        # Acceptance rate factor
        acceptance_score = driver.acceptance_rate * 20
        
        # Low cancellation rate bonus
        cancellation_penalty = driver.cancellation_rate * 30
        
        # Experience factor
        experience_score = min(driver.total_trips * 0.1, 10)
        
        total_score = (distance_score + rating_score + acceptance_score + 
                      experience_score - cancellation_penalty)
        
        return max(0, total_score)
    
    def get_algorithm_name(self) -> str:
        return "Smart Matching"


# ============================================================================
# TRIP STATE MANAGEMENT
# ============================================================================

class TripStateMachine:
    """Manage trip state transitions."""
    
    def __init__(self, trip: Trip):
        self.trip = trip
        self.valid_transitions = {
            TripStatus.REQUESTED: [TripStatus.SEARCHING, TripStatus.CANCELLED],
            TripStatus.SEARCHING: [TripStatus.MATCHED, TripStatus.CANCELLED],
            TripStatus.MATCHED: [TripStatus.ACCEPTED, TripStatus.CANCELLED],
            TripStatus.ACCEPTED: [TripStatus.DRIVER_ARRIVING, TripStatus.CANCELLED],
            TripStatus.DRIVER_ARRIVING: [TripStatus.DRIVER_ARRIVED, TripStatus.CANCELLED],
            TripStatus.DRIVER_ARRIVED: [TripStatus.IN_PROGRESS, TripStatus.CANCELLED],
            TripStatus.IN_PROGRESS: [TripStatus.COMPLETED, TripStatus.CANCELLED],
            TripStatus.COMPLETED: [TripStatus.PAYMENT_PENDING],
            TripStatus.PAYMENT_PENDING: [TripStatus.COMPLETED],
            TripStatus.CANCELLED: []  # Terminal state
        }
    
    def can_transition_to(self, new_status: TripStatus) -> bool:
        """Check if transition to new status is valid."""
        return new_status in self.valid_transitions.get(self.trip.status, [])
    
    def transition_to(self, new_status: TripStatus) -> bool:
        """Transition to new status if valid."""
        if not self.can_transition_to(new_status):
            return False
        
        old_status = self.trip.status
        self.trip.status = new_status
        
        # Update timestamps
        now = datetime.now()
        
        if new_status == TripStatus.MATCHED:
            self.trip.matched_at = now
        elif new_status == TripStatus.IN_PROGRESS:
            self.trip.started_at = now
        elif new_status == TripStatus.COMPLETED:
            self.trip.completed_at = now
        
        print(f"Trip {self.trip.trip_id[:8]} transitioned: {old_status.value} → {new_status.value}")
        return True


# ============================================================================
# MAIN RIDE SHARING SYSTEM
# ============================================================================

class RideSharingSystem:
    """Main ride sharing system."""
    
    def __init__(self, city_name: str = "Metro City"):
        self.city_name = city_name
        
        # Data storage
        self.users: Dict[str, User] = {}
        self.drivers: Dict[str, Driver] = {}
        self.riders: Dict[str, Rider] = {}
        self.trips: Dict[str, Trip] = {}
        self.payments: Dict[str, Payment] = {}
        
        # Active trip requests
        self.active_requests: Dict[str, TripRequest] = {}
        
        # System components
        self.pricing_strategy = DynamicPricing()
        self.matching_algorithm = SmartMatching()
        
        # Real-time tracking
        self.driver_locations: Dict[str, Location] = {}
        self.trip_routes: Dict[str, List[Location]] = {}
        
        # Analytics
        self.analytics = {
            'total_trips': 0,
            'completed_trips': 0,
            'cancelled_trips': 0,
            'total_revenue': 0.0,
            'active_drivers': 0,
            'average_wait_time': 0.0,
            'average_trip_duration': 0.0
        }
        
        # Threading
        self._lock = threading.RLock()
        
        # Background services
        self._start_background_services()
        
        print(f"🚗 Ride Sharing System initialized for {city_name}")
    
    def register_rider(self, email: str, phone: str, first_name: str, last_name: str) -> Rider:
        """Register a new rider."""
        rider = Rider(
            user_id=str(uuid.uuid4()),
            email=email,
            phone=phone,
            first_name=first_name,
            last_name=last_name
        )
        
        with self._lock:
            self.users[rider.user_id] = rider
            self.riders[rider.user_id] = rider
        
        return rider
    
    def register_driver(self, email: str, phone: str, first_name: str, last_name: str,
                       license_number: str, vehicle: Vehicle) -> Driver:
        """Register a new driver."""
        driver = Driver(
            user_id=str(uuid.uuid4()),
            email=email,
            phone=phone,
            first_name=first_name,
            last_name=last_name,
            license_number=license_number,
            vehicle=vehicle
        )
        
        # Set vehicle's driver_id
        vehicle.driver_id = driver.user_id
        
        with self._lock:
            self.users[driver.user_id] = driver
            self.drivers[driver.user_id] = driver
        
        return driver
    
    def set_driver_status(self, driver_id: str, status: DriverStatus) -> bool:
        """Set driver online/offline status."""
        with self._lock:
            if driver_id not in self.drivers:
                return False
            
            driver = self.drivers[driver_id]
            old_status = driver.status
            driver.status = status
            
            if status == DriverStatus.ONLINE:
                self.analytics['active_drivers'] += 1
            elif old_status == DriverStatus.ONLINE:
                self.analytics['active_drivers'] = max(0, self.analytics['active_drivers'] - 1)
            
            print(f"Driver {driver.full_name} status: {old_status.value} → {status.value}")
            return True
    
    def update_driver_location(self, driver_id: str, location: Location) -> bool:
        """Update driver's current location."""
        with self._lock:
            if driver_id not in self.drivers:
                return False
            
            driver = self.drivers[driver_id]
            driver.update_location(location)
            self.driver_locations[driver_id] = location
            
            return True
    
    def request_trip(self, rider_id: str, pickup_location: Location, destination: Location,
                    vehicle_type: VehicleType = VehicleType.ECONOMY) -> TripRequest:
        """Request a trip."""
        if rider_id not in self.riders:
            raise ValueError("Rider not found")
        
        trip_request = TripRequest(
            request_id=str(uuid.uuid4()),
            rider_id=rider_id,
            pickup_location=pickup_location,
            destination=destination,
            vehicle_type=vehicle_type
        )
        
        # Calculate estimated fare
        surge_multiplier = self._calculate_surge_multiplier(pickup_location)
        trip_request.estimated_fare = self.pricing_strategy.calculate_fare(trip_request, surge_multiplier)
        
        with self._lock:
            self.active_requests[trip_request.request_id] = trip_request
        
        # Start trip matching process
        self._start_trip_matching(trip_request)
        
        return trip_request
    
    def _start_trip_matching(self, trip_request: TripRequest) -> None:
        """Start the trip matching process."""
        # Find available drivers
        available_drivers = self._get_available_drivers(trip_request.vehicle_type)
        
        if not available_drivers:
            print(f"No available drivers for request {trip_request.request_id[:8]}")
            return
        
        # Find best driver
        selected_driver = self.matching_algorithm.find_driver(trip_request, available_drivers)
        
        if selected_driver:
            # Create trip
            trip = Trip(
                trip_id=str(uuid.uuid4()),
                rider_id=trip_request.rider_id,
                driver_id=selected_driver.user_id,
                request=trip_request
            )
            
            # Calculate fare
            surge_multiplier = self._calculate_surge_multiplier(trip_request.pickup_location)
            trip.base_fare = self.pricing_strategy.calculate_fare(trip_request, 1.0)
            trip.surge_multiplier = surge_multiplier
            trip.total_fare = trip.base_fare * surge_multiplier
            trip.platform_fee = trip.total_fare * 0.25  # 25% platform fee
            trip.driver_earnings = trip.total_fare - trip.platform_fee
            
            # Update trip status
            state_machine = TripStateMachine(trip)
            state_machine.transition_to(TripStatus.MATCHED)
            
            # Update driver status
            selected_driver.status = DriverStatus.BUSY
            
            with self._lock:
                self.trips[trip.trip_id] = trip
                self.analytics['total_trips'] += 1
                
                # Remove from active requests
                self.active_requests.pop(trip_request.request_id, None)
            
            print(f"Trip matched: {trip.trip_id[:8]} - {selected_driver.full_name}")
            
            # Simulate trip progression
            self._simulate_trip_progression(trip)
        
        else:
            print(f"No suitable driver found for request {trip_request.request_id[:8]}")
    
    def accept_trip(self, driver_id: str, trip_id: str) -> bool:
        """Driver accepts a trip."""
        with self._lock:
            if trip_id not in self.trips:
                return False
            
            trip = self.trips[trip_id]
            
            if trip.driver_id != driver_id:
                return False
            
            state_machine = TripStateMachine(trip)
            return state_machine.transition_to(TripStatus.ACCEPTED)
    
    def start_trip(self, driver_id: str, trip_id: str) -> bool:
        """Start the trip (driver picks up rider)."""
        with self._lock:
            if trip_id not in self.trips:
                return False
            
            trip = self.trips[trip_id]
            
            if trip.driver_id != driver_id:
                return False
            
            state_machine = TripStateMachine(trip)
            success = state_machine.transition_to(TripStatus.IN_PROGRESS)
            
            if success:
                trip.actual_pickup_location = trip.request.pickup_location
                # Initialize route tracking
                self.trip_routes[trip_id] = [trip.actual_pickup_location]
            
            return success
    
    def complete_trip(self, driver_id: str, trip_id: str, final_location: Location = None) -> bool:
        """Complete the trip."""
        with self._lock:
            if trip_id not in self.trips:
                return False
            
            trip = self.trips[trip_id]
            
            if trip.driver_id != driver_id:
                return False
            
            # Calculate actual metrics
            if final_location:
                trip.actual_destination = final_location
            else:
                trip.actual_destination = trip.request.destination
            
            # Calculate actual distance and duration
            if trip.started_at:
                trip.actual_duration = int((datetime.now() - trip.started_at).total_seconds() / 60)
            
            if trip.actual_pickup_location and trip.actual_destination:
                trip.actual_distance = trip.actual_pickup_location.distance_to(trip.actual_destination)
            
            state_machine = TripStateMachine(trip)
            success = state_machine.transition_to(TripStatus.COMPLETED)
            
            if success:
                # Update driver status
                driver = self.drivers[driver_id]
                driver.status = DriverStatus.ONLINE
                driver.total_trips += 1
                driver.total_earnings += trip.driver_earnings
                
                # Update rider stats
                rider = self.riders[trip.rider_id]
                rider.total_trips += 1
                
                # Update analytics
                self.analytics['completed_trips'] += 1
                self.analytics['total_revenue'] += trip.total_fare
                
                # Create payment
                self._create_payment(trip)
                
                print(f"Trip completed: {trip.trip_id[:8]} - ${trip.total_fare:.2f}")
            
            return success
    
    def cancel_trip(self, user_id: str, trip_id: str, reason: str = "") -> bool:
        """Cancel a trip."""
        with self._lock:
            if trip_id not in self.trips:
                return False
            
            trip = self.trips[trip_id]
            
            # Check if user has permission to cancel
            if user_id not in [trip.rider_id, trip.driver_id]:
                return False
            
            state_machine = TripStateMachine(trip)
            success = state_machine.transition_to(TripStatus.CANCELLED)
            
            if success:
                # Update driver status if assigned
                if trip.driver_id and trip.driver_id in self.drivers:
                    self.drivers[trip.driver_id].status = DriverStatus.ONLINE
                
                # Update analytics
                self.analytics['cancelled_trips'] += 1
                
                # Update cancellation rate for the user who cancelled
                if user_id == trip.driver_id:
                    driver = self.drivers[user_id]
                    total_trips = driver.total_trips + 1  # Include this cancelled trip
                    cancelled_trips = self.analytics['cancelled_trips']  # Simplified
                    driver.cancellation_rate = cancelled_trips / total_trips
                
                print(f"Trip cancelled: {trip.trip_id[:8]} - Reason: {reason}")
            
            return success
    
    def rate_trip(self, user_id: str, trip_id: str, rating: int, feedback: str = "") -> bool:
        """Rate a completed trip."""
        if rating < 1 or rating > 5:
            return False
        
        with self._lock:
            if trip_id not in self.trips:
                return False
            
            trip = self.trips[trip_id]
            
            if trip.status != TripStatus.COMPLETED:
                return False
            
            if user_id == trip.rider_id:
                trip.driver_rating = rating
                trip.driver_feedback = feedback
                
                # Update driver rating
                if trip.driver_id in self.drivers:
                    driver = self.drivers[trip.driver_id]
                    total_rating_points = driver.rating * driver.total_ratings + rating
                    driver.total_ratings += 1
                    driver.rating = total_rating_points / driver.total_ratings
                
            elif user_id == trip.driver_id:
                trip.rider_rating = rating
                trip.rider_feedback = feedback
                
                # Update rider rating
                if trip.rider_id in self.riders:
                    rider = self.riders[trip.rider_id]
                    total_rating_points = rider.rating * rider.total_ratings + rating
                    rider.total_ratings += 1
                    rider.rating = total_rating_points / rider.total_ratings
            
            else:
                return False
            
            return True
    
    def get_trip_history(self, user_id: str, limit: int = 20) -> List[Trip]:
        """Get trip history for a user."""
        user_trips = []
        
        for trip in self.trips.values():
            if trip.rider_id == user_id or trip.driver_id == user_id:
                user_trips.append(trip)
        
        # Sort by creation time, newest first
        user_trips.sort(key=lambda t: t.created_at, reverse=True)
        
        return user_trips[:limit]
    
    def get_nearby_drivers(self, location: Location, radius: float = 5.0,
                          vehicle_type: VehicleType = None) -> List[Driver]:
        """Get nearby available drivers."""
        nearby_drivers = []
        
        for driver in self.drivers.values():
            if (driver.status != DriverStatus.ONLINE or 
                not driver.current_location):
                continue
            
            if vehicle_type and driver.vehicle.vehicle_type != vehicle_type:
                continue
            
            distance = driver.current_location.distance_to(location)
            if distance <= radius:
                nearby_drivers.append(driver)
        
        # Sort by distance
        nearby_drivers.sort(key=lambda d: d.current_location.distance_to(location))
        
        return nearby_drivers
    
    def get_surge_pricing(self, location: Location) -> float:
        """Get current surge multiplier for location."""
        return self._calculate_surge_multiplier(location)
    
    def get_fare_estimate(self, pickup: Location, destination: Location,
                         vehicle_type: VehicleType = VehicleType.ECONOMY) -> Dict[str, Any]:
        """Get fare estimate for a trip."""
        trip_request = TripRequest(
            request_id="estimate",
            rider_id="estimate",
            pickup_location=pickup,
            destination=destination,
            vehicle_type=vehicle_type
        )
        
        surge_multiplier = self._calculate_surge_multiplier(pickup)
        base_fare = self.pricing_strategy.calculate_fare(trip_request, 1.0)
        surge_fare = base_fare * surge_multiplier
        
        return {
            'base_fare': base_fare,
            'surge_multiplier': surge_multiplier,
            'total_fare': surge_fare,
            'estimated_distance': trip_request.estimated_distance,
            'estimated_duration': trip_request.estimated_duration,
            'currency': 'USD'
        }
    
    def _get_available_drivers(self, vehicle_type: VehicleType) -> List[Driver]:
        """Get available drivers for specific vehicle type."""
        available_drivers = []
        
        for driver in self.drivers.values():
            if (driver.status == DriverStatus.ONLINE and
                driver.vehicle and
                driver.vehicle.vehicle_type == vehicle_type and
                driver.current_location):
                available_drivers.append(driver)
        
        return available_drivers
    
    def _calculate_surge_multiplier(self, location: Location) -> float:
        """Calculate surge pricing multiplier based on demand/supply."""
        # Simplified surge calculation
        # In reality, this would consider:
        # - Number of trip requests in area
        # - Number of available drivers in area
        # - Historical patterns
        # - Special events
        
        # Get nearby requests and drivers
        nearby_requests = sum(1 for req in self.active_requests.values()
                            if req.pickup_location.distance_to(location) <= 5.0)
        
        nearby_drivers = len(self.get_nearby_drivers(location, radius=5.0))
        
        if nearby_drivers == 0:
            return 2.0  # High surge when no drivers
        
        demand_ratio = nearby_requests / nearby_drivers
        
        if demand_ratio >= 3:
            return 2.5
        elif demand_ratio >= 2:
            return 2.0
        elif demand_ratio >= 1.5:
            return 1.8
        elif demand_ratio >= 1:
            return 1.5
        else:
            return 1.0
    
    def _create_payment(self, trip: Trip) -> Payment:
        """Create payment for completed trip."""
        payment = Payment(
            payment_id=str(uuid.uuid4()),
            trip_id=trip.trip_id,
            user_id=trip.rider_id,
            amount=trip.total_fare,
            payment_method="default"
        )
        
        # Simulate payment processing
        payment.status = PaymentStatus.COMPLETED
        payment.processed_at = datetime.now()
        
        with self._lock:
            self.payments[payment.payment_id] = payment
        
        return payment
    
    def _simulate_trip_progression(self, trip: Trip) -> None:
        """Simulate automatic trip progression for demo."""
        def progress_trip():
            import time
            
            # Simulate driver arriving
            time.sleep(2)
            state_machine = TripStateMachine(trip)
            state_machine.transition_to(TripStatus.DRIVER_ARRIVING)
            
            # Simulate driver arrived
            time.sleep(1)
            state_machine.transition_to(TripStatus.DRIVER_ARRIVED)
            
            # Auto-start trip
            time.sleep(1)
            self.start_trip(trip.driver_id, trip.trip_id)
            
            # Simulate trip in progress
            time.sleep(3)
            
            # Auto-complete trip
            self.complete_trip(trip.driver_id, trip.trip_id)
        
        # Run in background thread
        threading.Thread(target=progress_trip, daemon=True).start()
    
    def _start_background_services(self) -> None:
        """Start background services."""
        def cleanup_service():
            while True:
                try:
                    # Clean up old completed trips, update analytics, etc.
                    time.sleep(60)  # Run every minute
                    self._update_analytics()
                except Exception as e:
                    print(f"Background service error: {e}")
        
        threading.Thread(target=cleanup_service, daemon=True).start()
    
    def _update_analytics(self) -> None:
        """Update system analytics."""
        with self._lock:
            if self.analytics['completed_trips'] > 0:
                # Calculate average wait time (simplified)
                total_wait_time = 0
                completed_trips = [t for t in self.trips.values() 
                                 if t.status == TripStatus.COMPLETED and t.matched_at]
                
                for trip in completed_trips:
                    if trip.matched_at and trip.started_at:
                        wait_time = (trip.started_at - trip.matched_at).total_seconds() / 60
                        total_wait_time += wait_time
                
                if completed_trips:
                    self.analytics['average_wait_time'] = total_wait_time / len(completed_trips)
                
                # Calculate average trip duration
                total_duration = sum(trip.actual_duration or 0 for trip in completed_trips)
                if completed_trips:
                    self.analytics['average_trip_duration'] = total_duration / len(completed_trips)
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics."""
        with self._lock:
            return {
                **self.analytics,
                'total_users': len(self.users),
                'total_drivers': len(self.drivers),
                'total_riders': len(self.riders),
                'online_drivers': len([d for d in self.drivers.values() 
                                     if d.status == DriverStatus.ONLINE]),
                'active_trips': len([t for t in self.trips.values() 
                                   if t.status in [TripStatus.IN_PROGRESS, TripStatus.DRIVER_ARRIVING]]),
                'pending_requests': len(self.active_requests),
                'completion_rate': (self.analytics['completed_trips'] / 
                                  max(1, self.analytics['total_trips'])) * 100
            }


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_ride_sharing_system():
    """Demonstrate the ride sharing system."""
    print("=== RIDE SHARING SYSTEM DEMONSTRATION ===\n")
    
    # Initialize system
    print("1. SYSTEM INITIALIZATION:")
    
    system = RideSharingSystem("Demo City")
    print("   ✓ Ride sharing system initialized")
    print()
    
    # Register riders
    print("2. RIDER REGISTRATION:")
    
    riders = []
    rider_data = [
        ("alice@example.com", "+1234567890", "Alice", "Johnson"),
        ("bob@example.com", "+1234567891", "Bob", "Smith"),
        ("charlie@example.com", "+1234567892", "Charlie", "Brown")
    ]
    
    for email, phone, first_name, last_name in rider_data:
        rider = system.register_rider(email, phone, first_name, last_name)
        riders.append(rider)
        print(f"   ✓ Registered rider: {rider.full_name}")
    
    print()
    
    # Register drivers with vehicles
    print("3. DRIVER REGISTRATION:")
    
    drivers = []
    driver_data = [
        ("driver1@example.com", "+1234567893", "David", "Wilson", "DL123456", 
         Vehicle("", "", "Toyota", "Camry", 2020, "Silver", "ABC123", VehicleType.ECONOMY)),
        ("driver2@example.com", "+1234567894", "Emma", "Davis", "DL123457",
         Vehicle("", "", "BMW", "X3", 2021, "Black", "DEF456", VehicleType.PREMIUM)),
        ("driver3@example.com", "+1234567895", "Frank", "Miller", "DL123458",
         Vehicle("", "", "Honda", "Civic", 2019, "Blue", "GHI789", VehicleType.ECONOMY)),
        ("driver4@example.com", "+1234567896", "Grace", "Taylor", "DL123459",
         Vehicle("", "", "Mercedes", "S-Class", 2022, "White", "JKL012", VehicleType.LUXURY))
    ]
    
    for email, phone, first_name, last_name, license_num, vehicle in driver_data:
        driver = system.register_driver(email, phone, first_name, last_name, license_num, vehicle)
        drivers.append(driver)
        print(f"   ✓ Registered driver: {driver.full_name} ({vehicle.make} {vehicle.model})")
    
    print()
    
    # Set drivers online and update locations
    print("4. DRIVER STATUS AND LOCATIONS:")
    
    # Simulate city locations
    city_locations = [
        Location(40.7128, -74.0060, "Downtown", "Demo City"),
        Location(40.7589, -73.9851, "Midtown", "Demo City"),
        Location(40.6782, -73.9442, "Brooklyn", "Demo City"),
        Location(40.7831, -73.9712, "Upper West Side", "Demo City")
    ]
    
    for i, driver in enumerate(drivers):
        # Set driver online
        system.set_driver_status(driver.user_id, DriverStatus.ONLINE)
        
        # Update location
        location = city_locations[i % len(city_locations)]
        system.update_driver_location(driver.user_id, location)
        
        print(f"   ✓ {driver.full_name} online at {location}")
    
    print()
    
    # Test fare estimation
    print("5. FARE ESTIMATION:")
    
    pickup = Location(40.7128, -74.0060, "Downtown")
    destination = Location(40.7589, -73.9851, "Midtown")
    
    vehicle_types = [VehicleType.ECONOMY, VehicleType.PREMIUM, VehicleType.LUXURY]
    
    for vehicle_type in vehicle_types:
        estimate = system.get_fare_estimate(pickup, destination, vehicle_type)
        print(f"   {vehicle_type.value.title()}: ${estimate['total_fare']:.2f} "
              f"({estimate['estimated_distance']:.1f} km, "
              f"{estimate['estimated_duration']} min, "
              f"surge: {estimate['surge_multiplier']:.1f}x)")
    
    print()
    
    # Test trip requests and matching
    print("6. TRIP REQUESTS AND MATCHING:")
    
    # Alice requests an economy ride
    alice = riders[0]
    alice_pickup = Location(40.7128, -74.0060, "Alice's Location")
    alice_destination = Location(40.7589, -73.9851, "Alice's Destination")
    
    print(f"   {alice.full_name} requests Economy ride:")
    print(f"     From: {alice_pickup}")
    print(f"     To: {alice_destination}")
    
    trip_request = system.request_trip(
        alice.user_id, 
        alice_pickup, 
        alice_destination, 
        VehicleType.ECONOMY
    )
    
    print(f"   ✓ Trip request created: {trip_request.request_id[:8]}")
    print(f"   ✓ Estimated fare: ${trip_request.estimated_fare:.2f}")
    
    # Wait for trip matching
    import time
    time.sleep(1)
    
    # Bob requests a premium ride
    bob = riders[1]
    bob_pickup = Location(40.6782, -73.9442, "Bob's Location")
    bob_destination = Location(40.7831, -73.9712, "Bob's Destination")
    
    print(f"\n   {bob.full_name} requests Premium ride:")
    trip_request2 = system.request_trip(
        bob.user_id,
        bob_pickup,
        bob_destination,
        VehicleType.PREMIUM
    )
    
    print(f"   ✓ Trip request created: {trip_request2.request_id[:8]}")
    
    print()
    
    # Test nearby drivers
    print("7. NEARBY DRIVERS:")
    
    test_location = Location(40.7128, -74.0060, "Test Location")
    nearby_drivers = system.get_nearby_drivers(test_location, radius=10.0)
    
    print(f"   Drivers within 10km of {test_location}:")
    for driver in nearby_drivers:
        distance = driver.current_location.distance_to(test_location)
        print(f"     {driver.full_name}: {distance:.1f}km away "
              f"({driver.vehicle.vehicle_type.value}, rating: {driver.rating:.1f})")
    
    print()
    
    # Test surge pricing
    print("8. SURGE PRICING TEST:")
    
    surge_locations = [
        Location(40.7128, -74.0060, "High Demand Area"),
        Location(40.7589, -73.9851, "Normal Area"),
        Location(40.6782, -73.9442, "Low Demand Area")
    ]
    
    for location in surge_locations:
        surge = system.get_surge_pricing(location)
        print(f"   {location.address}: {surge:.1f}x surge")
    
    print()
    
    # Test trip progression (wait for auto-completion)
    print("9. TRIP PROGRESSION:")
    
    # Wait for trips to auto-complete
    time.sleep(8)
    
    # Check trip statuses
    for trip_id, trip in system.trips.items():
        driver_name = system.drivers[trip.driver_id].full_name if trip.driver_id else "Unknown"
        rider_name = system.riders[trip.rider_id].full_name
        
        print(f"   Trip {trip_id[:8]}: {rider_name} + {driver_name}")
        print(f"     Status: {trip.status.value}")
        print(f"     Fare: ${trip.total_fare:.2f}")
        
        if trip.status == TripStatus.COMPLETED:
            print(f"     Duration: {trip.actual_duration} minutes")
            print(f"     Distance: {trip.actual_distance:.1f} km")
    
    print()
    
    # Test ratings
    print("10. RATING SYSTEM:")
    
    for trip_id, trip in system.trips.items():
        if trip.status == TripStatus.COMPLETED:
            # Rider rates driver
            rider_rating = random.randint(4, 5)
            system.rate_trip(trip.rider_id, trip_id, rider_rating, "Great ride!")
            
            # Driver rates rider
            driver_rating = random.randint(4, 5)
            system.rate_trip(trip.driver_id, trip_id, driver_rating, "Polite passenger!")
            
            rider_name = system.riders[trip.rider_id].full_name
            driver_name = system.drivers[trip.driver_id].full_name
            
            print(f"   Trip {trip_id[:8]}:")
            print(f"     {rider_name} rated driver: {rider_rating}⭐")
            print(f"     {driver_name} rated rider: {driver_rating}⭐")
    
    print()
    
    # Test trip history
    print("11. TRIP HISTORY:")
    
    for rider in riders[:2]:  # Show history for first 2 riders
        history = system.get_trip_history(rider.user_id, limit=5)
        print(f"   {rider.full_name}'s trip history ({len(history)} trips):")
        
        for trip in history:
            driver_name = system.drivers[trip.driver_id].full_name if trip.driver_id else "Unknown"
            status_emoji = "✅" if trip.status == TripStatus.COMPLETED else "❌"
            
            print(f"     {status_emoji} {trip.created_at.strftime('%Y-%m-%d %H:%M')} - "
                  f"${trip.total_fare:.2f} with {driver_name}")
    
    print()
    
    # Test trip cancellation
    print("12. TRIP CANCELLATION TEST:")
    
    # Charlie requests a ride and then cancels
    charlie = riders[2]
    charlie_pickup = Location(40.7831, -73.9712, "Charlie's Location")
    charlie_destination = Location(40.7128, -74.0060, "Charlie's Destination")
    
    cancel_request = system.request_trip(
        charlie.user_id,
        charlie_pickup,
        charlie_destination,
        VehicleType.ECONOMY
    )
    
    print(f"   {charlie.full_name} requests a ride...")
    
    # Wait for matching
    time.sleep(1)
    
    # Find the trip and cancel it
    charlie_trip = None
    for trip in system.trips.values():
        if trip.rider_id == charlie.user_id and trip.status != TripStatus.CANCELLED:
            charlie_trip = trip
            break
    
    if charlie_trip:
        success = system.cancel_trip(charlie.user_id, charlie_trip.trip_id, "Changed plans")
        print(f"   {'✓' if success else '✗'} Trip cancelled by rider")
    
    print()
    
    # Test different matching algorithms
    print("13. MATCHING ALGORITHM COMPARISON:")
    
    # Test with nearest driver matching
    system.matching_algorithm = NearestDriverMatching()
    print(f"   Switched to: {system.matching_algorithm.get_algorithm_name()}")
    
    # Create a test request
    test_pickup = Location(40.7500, -73.9800, "Test Pickup")
    available_drivers = system._get_available_drivers(VehicleType.ECONOMY)
    
    if available_drivers:
        test_request = TripRequest("test", "test_rider", test_pickup, 
                                 Location(40.7600, -73.9700, "Test Dest"), VehicleType.ECONOMY)
        
        nearest_driver = system.matching_algorithm.find_driver(test_request, available_drivers)
        if nearest_driver:
            distance = nearest_driver.current_location.distance_to(test_pickup)
            print(f"   Nearest driver: {nearest_driver.full_name} ({distance:.1f}km away)")
    
    # Switch back to smart matching
    system.matching_algorithm = SmartMatching()
    print(f"   Switched back to: {system.matching_algorithm.get_algorithm_name()}")
    
    print()
    
    # Show comprehensive analytics
    print("14. SYSTEM ANALYTICS:")
    
    analytics = system.get_system_analytics()
    
    print(f"   Users:")
    print(f"     Total: {analytics['total_users']}")
    print(f"     Drivers: {analytics['total_drivers']}")
    print(f"     Riders: {analytics['total_riders']}")
    
    print(f"\n   Trips:")
    print(f"     Total: {analytics['total_trips']}")
    print(f"     Completed: {analytics['completed_trips']}")
    print(f"     Cancelled: {analytics['cancelled_trips']}")
    print(f"     Completion Rate: {analytics['completion_rate']:.1f}%")
    
    print(f"\n   Performance:")
    print(f"     Online Drivers: {analytics['online_drivers']}")
    print(f"     Active Trips: {analytics['active_trips']}")
    print(f"     Average Wait Time: {analytics['average_wait_time']:.1f} minutes")
    print(f"     Average Trip Duration: {analytics['average_trip_duration']:.1f} minutes")
    
    print(f"\n   Financial:")
    print(f"     Total Revenue: ${analytics['total_revenue']:.2f}")
    print(f"     Average Fare: ${analytics['total_revenue'] / max(1, analytics['completed_trips']):.2f}")
    
    print()
    
    # Show driver and rider statistics
    print("15. USER STATISTICS:")
    
    print("   Driver Statistics:")
    for driver in drivers:
        print(f"     {driver.full_name}:")
        print(f"       Trips: {driver.total_trips}")
        print(f"       Earnings: ${driver.total_earnings:.2f}")
        print(f"       Rating: {driver.rating:.1f} ({driver.total_ratings} reviews)")
        print(f"       Status: {driver.status.value}")
    
    print("\n   Rider Statistics:")
    for rider in riders:
        print(f"     {rider.full_name}:")
        print(f"       Trips: {rider.total_trips}")
        print(f"       Rating: {rider.rating:.1f} ({rider.total_ratings} reviews)")
    
    print()
    
    # Show final system state
    print("16. FINAL SYSTEM STATE:")
    
    final_analytics = system.get_system_analytics()
    
    print(f"   Active System Components:")
    print(f"     Online Drivers: {final_analytics['online_drivers']}/{final_analytics['total_drivers']}")
    print(f"     Active Trips: {final_analytics['active_trips']}")
    print(f"     Pending Requests: {final_analytics['pending_requests']}")
    
    print(f"\n   City Coverage:")
    online_drivers_by_location = defaultdict(int)
    for driver in drivers:
        if driver.status == DriverStatus.ONLINE and driver.current_location:
            online_drivers_by_location[driver.current_location.city] += 1
    
    for city, count in online_drivers_by_location.items():
        print(f"     {city}: {count} drivers")
    
    print(f"\n   Service Quality:")
    avg_driver_rating = sum(d.rating for d in drivers) / len(drivers)
    avg_rider_rating = sum(r.rating for r in riders) / len(riders)
    print(f"     Average Driver Rating: {avg_driver_rating:.1f}⭐")
    print(f"     Average Rider Rating: {avg_rider_rating:.1f}⭐")
    print(f"     Service Reliability: {final_analytics['completion_rate']:.1f}%")
    
    print()
    print("=== RIDE SHARING SYSTEM DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_ride_sharing_system()
