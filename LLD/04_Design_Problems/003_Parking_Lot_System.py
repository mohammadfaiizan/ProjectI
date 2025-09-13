"""
PARKING LOT SYSTEM - Complete System Design
===========================================

Problem Statement:
Design a comprehensive parking lot management system that handles:
- Multiple vehicle types (Car, Motorcycle, Truck, Bus)
- Different parking spot types and sizes
- Entry and exit management with ticket system
- Payment processing with multiple payment methods
- Real-time availability tracking
- Pricing strategies (hourly, daily, monthly passes)
- Reserved parking and VIP sections
- Multi-level parking garage support
- Automated barrier control
- Revenue tracking and reporting

Requirements:
- Support different vehicle sizes and parking spot allocation
- Implement dynamic pricing based on time and demand
- Handle peak hours and special events pricing
- Generate parking tickets with QR codes
- Process payments via cash, card, and mobile payments
- Track occupancy in real-time
- Support handicapped parking spots
- Implement security features and surveillance integration
- Generate reports for management
- Handle system failures gracefully

Design Patterns Used:
- Factory: Vehicle and parking spot creation
- Strategy: Pricing and payment strategies
- Observer: Occupancy monitoring
- Command: Parking operations
- State: Parking spot states
- Singleton: Parking lot management
- Decorator: Premium parking features
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import time
from dataclasses import dataclass, field
import json
import qrcode
from io import BytesIO
import base64


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class VehicleType(Enum):
    MOTORCYCLE = "motorcycle"
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    ELECTRIC_CAR = "electric_car"


class SpotType(Enum):
    MOTORCYCLE = "motorcycle"
    COMPACT = "compact"
    REGULAR = "regular"
    LARGE = "large"
    HANDICAPPED = "handicapped"
    ELECTRIC = "electric"
    VIP = "vip"


class SpotStatus(Enum):
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    RESERVED = "reserved"
    OUT_OF_ORDER = "out_of_order"


class PaymentMethod(Enum):
    CASH = "cash"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    MOBILE_PAYMENT = "mobile_payment"
    MONTHLY_PASS = "monthly_pass"


class TicketStatus(Enum):
    ACTIVE = "active"
    PAID = "paid"
    LOST = "lost"
    EXPIRED = "expired"


@dataclass
class VehicleInfo:
    """Vehicle information."""
    license_plate: str
    vehicle_type: VehicleType
    owner_name: str = ""
    owner_phone: str = ""
    is_electric: bool = False
    is_handicapped: bool = False
    
    def __post_init__(self):
        self.license_plate = self.license_plate.upper()


@dataclass
class PaymentInfo:
    """Payment transaction information."""
    payment_id: str
    amount: float
    method: PaymentMethod
    timestamp: datetime
    transaction_reference: str = ""
    status: str = "completed"


# ============================================================================
# VEHICLE CLASSES
# ============================================================================

class Vehicle(ABC):
    """Abstract vehicle class."""
    
    def __init__(self, license_plate: str, vehicle_type: VehicleType):
        self.license_plate = license_plate.upper()
        self.vehicle_type = vehicle_type
        self.entry_time: Optional[datetime] = None
        self.exit_time: Optional[datetime] = None
    
    @abstractmethod
    def get_required_spot_types(self) -> List[SpotType]:
        """Get list of compatible parking spot types."""
        pass
    
    @abstractmethod
    def get_size_factor(self) -> float:
        """Get size factor for pricing calculations."""
        pass
    
    def get_parking_duration(self) -> timedelta:
        """Get parking duration."""
        if self.entry_time and self.exit_time:
            return self.exit_time - self.entry_time
        elif self.entry_time:
            return datetime.now() - self.entry_time
        return timedelta(0)
    
    def __str__(self) -> str:
        return f"{self.vehicle_type.value.title()} - {self.license_plate}"


class Motorcycle(Vehicle):
    """Motorcycle vehicle."""
    
    def __init__(self, license_plate: str):
        super().__init__(license_plate, VehicleType.MOTORCYCLE)
    
    def get_required_spot_types(self) -> List[SpotType]:
        return [SpotType.MOTORCYCLE, SpotType.COMPACT, SpotType.REGULAR]
    
    def get_size_factor(self) -> float:
        return 0.5


class Car(Vehicle):
    """Car vehicle."""
    
    def __init__(self, license_plate: str, is_electric: bool = False):
        super().__init__(license_plate, VehicleType.CAR)
        self.is_electric = is_electric
    
    def get_required_spot_types(self) -> List[SpotType]:
        base_types = [SpotType.COMPACT, SpotType.REGULAR]
        if self.is_electric:
            base_types.insert(0, SpotType.ELECTRIC)
        return base_types
    
    def get_size_factor(self) -> float:
        return 1.0


class Truck(Vehicle):
    """Truck vehicle."""
    
    def __init__(self, license_plate: str):
        super().__init__(license_plate, VehicleType.TRUCK)
    
    def get_required_spot_types(self) -> List[SpotType]:
        return [SpotType.LARGE]
    
    def get_size_factor(self) -> float:
        return 2.0


class Bus(Vehicle):
    """Bus vehicle."""
    
    def __init__(self, license_plate: str):
        super().__init__(license_plate, VehicleType.BUS)
    
    def get_required_spot_types(self) -> List[SpotType]:
        return [SpotType.LARGE]
    
    def get_size_factor(self) -> float:
        return 3.0


# ============================================================================
# VEHICLE FACTORY
# ============================================================================

class VehicleFactory:
    """Factory for creating vehicles."""
    
    @staticmethod
    def create_vehicle(vehicle_info: VehicleInfo) -> Vehicle:
        """Create vehicle based on vehicle info."""
        license_plate = vehicle_info.license_plate
        vehicle_type = vehicle_info.vehicle_type
        
        if vehicle_type == VehicleType.MOTORCYCLE:
            return Motorcycle(license_plate)
        elif vehicle_type == VehicleType.CAR:
            return Car(license_plate, vehicle_info.is_electric)
        elif vehicle_type == VehicleType.TRUCK:
            return Truck(license_plate)
        elif vehicle_type == VehicleType.BUS:
            return Bus(license_plate)
        else:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}")


# ============================================================================
# PARKING SPOT CLASSES
# ============================================================================

class ParkingSpot:
    """Individual parking spot."""
    
    def __init__(self, spot_id: str, spot_type: SpotType, floor: int, section: str):
        self.spot_id = spot_id
        self.spot_type = spot_type
        self.floor = floor
        self.section = section
        self.status = SpotStatus.AVAILABLE
        self.current_vehicle: Optional[Vehicle] = None
        self.reserved_until: Optional[datetime] = None
        self.last_occupied: Optional[datetime] = None
        self.total_usage_time = timedelta(0)
        self.usage_count = 0
        self._lock = threading.Lock()
    
    def is_available(self) -> bool:
        """Check if spot is available."""
        with self._lock:
            if self.status == SpotStatus.OUT_OF_ORDER:
                return False
            
            if self.status == SpotStatus.RESERVED:
                if self.reserved_until and datetime.now() > self.reserved_until:
                    self.status = SpotStatus.AVAILABLE
                    self.reserved_until = None
                else:
                    return False
            
            return self.status == SpotStatus.AVAILABLE
    
    def can_fit_vehicle(self, vehicle: Vehicle) -> bool:
        """Check if vehicle can fit in this spot."""
        if not self.is_available():
            return False
        
        compatible_spots = vehicle.get_required_spot_types()
        return self.spot_type in compatible_spots
    
    def occupy_spot(self, vehicle: Vehicle) -> bool:
        """Occupy the spot with a vehicle."""
        with self._lock:
            if not self.can_fit_vehicle(vehicle):
                return False
            
            self.status = SpotStatus.OCCUPIED
            self.current_vehicle = vehicle
            self.last_occupied = datetime.now()
            vehicle.entry_time = datetime.now()
            self.usage_count += 1
            
            return True
    
    def vacate_spot(self) -> Optional[Vehicle]:
        """Vacate the spot and return the vehicle."""
        with self._lock:
            if self.status != SpotStatus.OCCUPIED or not self.current_vehicle:
                return None
            
            vehicle = self.current_vehicle
            vehicle.exit_time = datetime.now()
            
            # Update usage statistics
            if self.last_occupied:
                usage_duration = datetime.now() - self.last_occupied
                self.total_usage_time += usage_duration
            
            self.status = SpotStatus.AVAILABLE
            self.current_vehicle = None
            self.last_occupied = None
            
            return vehicle
    
    def reserve_spot(self, duration_minutes: int = 30) -> bool:
        """Reserve the spot for a specified duration."""
        with self._lock:
            if not self.is_available():
                return False
            
            self.status = SpotStatus.RESERVED
            self.reserved_until = datetime.now() + timedelta(minutes=duration_minutes)
            return True
    
    def set_out_of_order(self, out_of_order: bool = True) -> None:
        """Set spot as out of order."""
        with self._lock:
            if out_of_order:
                if self.current_vehicle:
                    return  # Cannot set out of order if occupied
                self.status = SpotStatus.OUT_OF_ORDER
            else:
                if self.status == SpotStatus.OUT_OF_ORDER:
                    self.status = SpotStatus.AVAILABLE
    
    def get_spot_info(self) -> Dict[str, Any]:
        """Get spot information."""
        return {
            'spot_id': self.spot_id,
            'spot_type': self.spot_type.value,
            'floor': self.floor,
            'section': self.section,
            'status': self.status.value,
            'current_vehicle': str(self.current_vehicle) if self.current_vehicle else None,
            'reserved_until': self.reserved_until.isoformat() if self.reserved_until else None,
            'usage_count': self.usage_count,
            'total_usage_hours': self.total_usage_time.total_seconds() / 3600
        }
    
    def __str__(self) -> str:
        return f"Spot {self.spot_id} ({self.spot_type.value}) - Floor {self.floor}, Section {self.section}"


# ============================================================================
# PARKING TICKET
# ============================================================================

class ParkingTicket:
    """Parking ticket with QR code."""
    
    def __init__(self, vehicle: Vehicle, spot: ParkingSpot, entry_time: datetime = None):
        self.ticket_id = str(uuid.uuid4())
        self.vehicle = vehicle
        self.spot = spot
        self.entry_time = entry_time or datetime.now()
        self.exit_time: Optional[datetime] = None
        self.status = TicketStatus.ACTIVE
        self.payment_info: Optional[PaymentInfo] = None
        self.qr_code = self._generate_qr_code()
        self.lost_ticket_fee = 50.0  # Fee for lost tickets
    
    def _generate_qr_code(self) -> str:
        """Generate QR code for the ticket."""
        ticket_data = {
            'ticket_id': self.ticket_id,
            'license_plate': self.vehicle.license_plate,
            'spot_id': self.spot.spot_id,
            'entry_time': self.entry_time.isoformat()
        }
        
        # Create QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(json.dumps(ticket_data))
        qr.make(fit=True)
        
        # Convert to base64 string (simplified representation)
        return base64.b64encode(json.dumps(ticket_data).encode()).decode()
    
    def calculate_parking_duration(self) -> timedelta:
        """Calculate parking duration."""
        end_time = self.exit_time or datetime.now()
        return end_time - self.entry_time
    
    def mark_as_paid(self, payment_info: PaymentInfo) -> None:
        """Mark ticket as paid."""
        self.payment_info = payment_info
        self.status = TicketStatus.PAID
    
    def mark_as_lost(self) -> None:
        """Mark ticket as lost."""
        self.status = TicketStatus.LOST
    
    def is_expired(self, max_hours: int = 24) -> bool:
        """Check if ticket is expired."""
        if self.status != TicketStatus.ACTIVE:
            return False
        
        duration = self.calculate_parking_duration()
        return duration.total_seconds() > (max_hours * 3600)
    
    def get_ticket_info(self) -> Dict[str, Any]:
        """Get ticket information."""
        return {
            'ticket_id': self.ticket_id,
            'vehicle': str(self.vehicle),
            'spot': str(self.spot),
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'status': self.status.value,
            'duration_hours': self.calculate_parking_duration().total_seconds() / 3600,
            'payment_info': {
                'amount': self.payment_info.amount,
                'method': self.payment_info.method.value,
                'timestamp': self.payment_info.timestamp.isoformat()
            } if self.payment_info else None,
            'qr_code': self.qr_code
        }
    
    def __str__(self) -> str:
        return f"Ticket {self.ticket_id[:8]} - {self.vehicle.license_plate}"


# ============================================================================
# PRICING STRATEGIES
# ============================================================================

class PricingStrategy(ABC):
    """Abstract pricing strategy."""
    
    @abstractmethod
    def calculate_fee(self, ticket: ParkingTicket, exit_time: datetime = None) -> float:
        """Calculate parking fee."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class HourlyPricingStrategy(PricingStrategy):
    """Hourly pricing strategy."""
    
    def __init__(self, base_rate: float = 5.0, max_daily_rate: float = 50.0):
        self.base_rate = base_rate
        self.max_daily_rate = max_daily_rate
    
    def calculate_fee(self, ticket: ParkingTicket, exit_time: datetime = None) -> float:
        """Calculate hourly parking fee."""
        end_time = exit_time or datetime.now()
        duration = end_time - ticket.entry_time
        
        # Calculate hours (minimum 1 hour)
        hours = max(1, int(duration.total_seconds() / 3600))
        if duration.total_seconds() % 3600 > 0:
            hours += 1  # Round up partial hours
        
        # Apply vehicle size factor
        size_factor = ticket.vehicle.get_size_factor()
        
        # Calculate base fee
        fee = hours * self.base_rate * size_factor
        
        # Apply daily maximum
        daily_fee = min(fee, self.max_daily_rate * size_factor)
        
        # Apply spot type multiplier
        spot_multiplier = self._get_spot_multiplier(ticket.spot.spot_type)
        
        return daily_fee * spot_multiplier
    
    def _get_spot_multiplier(self, spot_type: SpotType) -> float:
        """Get pricing multiplier based on spot type."""
        multipliers = {
            SpotType.MOTORCYCLE: 0.5,
            SpotType.COMPACT: 0.8,
            SpotType.REGULAR: 1.0,
            SpotType.LARGE: 1.5,
            SpotType.HANDICAPPED: 0.5,  # Discounted
            SpotType.ELECTRIC: 0.9,     # Slight discount for electric
            SpotType.VIP: 2.0           # Premium pricing
        }
        return multipliers.get(spot_type, 1.0)
    
    def get_strategy_name(self) -> str:
        return "Hourly Pricing Strategy"


class DynamicPricingStrategy(PricingStrategy):
    """Dynamic pricing based on demand and time."""
    
    def __init__(self, base_rate: float = 5.0, peak_multiplier: float = 1.5):
        self.base_rate = base_rate
        self.peak_multiplier = peak_multiplier
        self.peak_hours = [(7, 10), (17, 20)]  # Morning and evening peaks
    
    def calculate_fee(self, ticket: ParkingTicket, exit_time: datetime = None) -> float:
        """Calculate dynamic parking fee."""
        end_time = exit_time or datetime.now()
        duration = end_time - ticket.entry_time
        
        # Calculate base fee
        hours = max(1, int(duration.total_seconds() / 3600))
        if duration.total_seconds() % 3600 > 0:
            hours += 1
        
        # Check if parking occurred during peak hours
        entry_hour = ticket.entry_time.hour
        is_peak = any(start <= entry_hour < end for start, end in self.peak_hours)
        
        # Apply peak pricing
        rate = self.base_rate * (self.peak_multiplier if is_peak else 1.0)
        
        # Apply vehicle and spot factors
        size_factor = ticket.vehicle.get_size_factor()
        spot_multiplier = self._get_spot_multiplier(ticket.spot.spot_type)
        
        return hours * rate * size_factor * spot_multiplier
    
    def _get_spot_multiplier(self, spot_type: SpotType) -> float:
        """Get pricing multiplier based on spot type."""
        multipliers = {
            SpotType.MOTORCYCLE: 0.5,
            SpotType.COMPACT: 0.8,
            SpotType.REGULAR: 1.0,
            SpotType.LARGE: 1.5,
            SpotType.HANDICAPPED: 0.0,  # Free for handicapped
            SpotType.ELECTRIC: 0.8,
            SpotType.VIP: 3.0
        }
        return multipliers.get(spot_type, 1.0)
    
    def get_strategy_name(self) -> str:
        return "Dynamic Pricing Strategy"


class FlatRatePricingStrategy(PricingStrategy):
    """Flat rate pricing strategy."""
    
    def __init__(self, daily_rate: float = 25.0):
        self.daily_rate = daily_rate
    
    def calculate_fee(self, ticket: ParkingTicket, exit_time: datetime = None) -> float:
        """Calculate flat rate parking fee."""
        size_factor = ticket.vehicle.get_size_factor()
        spot_multiplier = self._get_spot_multiplier(ticket.spot.spot_type)
        
        return self.daily_rate * size_factor * spot_multiplier
    
    def _get_spot_multiplier(self, spot_type: SpotType) -> float:
        """Get pricing multiplier based on spot type."""
        multipliers = {
            SpotType.MOTORCYCLE: 0.5,
            SpotType.COMPACT: 0.8,
            SpotType.REGULAR: 1.0,
            SpotType.LARGE: 1.2,
            SpotType.HANDICAPPED: 0.0,
            SpotType.ELECTRIC: 0.9,
            SpotType.VIP: 2.0
        }
        return multipliers.get(spot_type, 1.0)
    
    def get_strategy_name(self) -> str:
        return "Flat Rate Pricing Strategy"


# ============================================================================
# PAYMENT PROCESSING
# ============================================================================

class PaymentProcessor(ABC):
    """Abstract payment processor."""
    
    @abstractmethod
    def process_payment(self, amount: float, method: PaymentMethod, 
                       reference: str = "") -> PaymentInfo:
        """Process payment and return payment info."""
        pass
    
    @abstractmethod
    def refund_payment(self, payment_info: PaymentInfo) -> bool:
        """Process refund."""
        pass


class CashPaymentProcessor(PaymentProcessor):
    """Cash payment processor."""
    
    def process_payment(self, amount: float, method: PaymentMethod, 
                       reference: str = "") -> PaymentInfo:
        """Process cash payment."""
        if method != PaymentMethod.CASH:
            raise ValueError("This processor only handles cash payments")
        
        # Simulate cash payment processing
        payment_id = f"CASH_{uuid.uuid4().hex[:8]}"
        
        return PaymentInfo(
            payment_id=payment_id,
            amount=amount,
            method=method,
            timestamp=datetime.now(),
            transaction_reference=reference,
            status="completed"
        )
    
    def refund_payment(self, payment_info: PaymentInfo) -> bool:
        """Process cash refund."""
        # Cash refunds require manual handling
        return True


class CardPaymentProcessor(PaymentProcessor):
    """Card payment processor."""
    
    def process_payment(self, amount: float, method: PaymentMethod, 
                       reference: str = "") -> PaymentInfo:
        """Process card payment."""
        if method not in [PaymentMethod.CREDIT_CARD, PaymentMethod.DEBIT_CARD]:
            raise ValueError("This processor only handles card payments")
        
        # Simulate card payment processing
        payment_id = f"CARD_{uuid.uuid4().hex[:8]}"
        
        # Simulate payment gateway interaction
        time.sleep(0.1)  # Simulate processing delay
        
        return PaymentInfo(
            payment_id=payment_id,
            amount=amount,
            method=method,
            timestamp=datetime.now(),
            transaction_reference=reference,
            status="completed"
        )
    
    def refund_payment(self, payment_info: PaymentInfo) -> bool:
        """Process card refund."""
        # Simulate refund processing
        return True


class MobilePaymentProcessor(PaymentProcessor):
    """Mobile payment processor."""
    
    def process_payment(self, amount: float, method: PaymentMethod, 
                       reference: str = "") -> PaymentInfo:
        """Process mobile payment."""
        if method != PaymentMethod.MOBILE_PAYMENT:
            raise ValueError("This processor only handles mobile payments")
        
        payment_id = f"MOBILE_{uuid.uuid4().hex[:8]}"
        
        return PaymentInfo(
            payment_id=payment_id,
            amount=amount,
            method=method,
            timestamp=datetime.now(),
            transaction_reference=reference,
            status="completed"
        )
    
    def refund_payment(self, payment_info: PaymentInfo) -> bool:
        """Process mobile payment refund."""
        return True


# ============================================================================
# PARKING LOT MANAGEMENT
# ============================================================================

class ParkingFloor:
    """Represents a floor in the parking garage."""
    
    def __init__(self, floor_number: int):
        self.floor_number = floor_number
        self.spots: Dict[str, ParkingSpot] = {}
        self.sections: Set[str] = set()
        self._lock = threading.Lock()
    
    def add_spot(self, spot: ParkingSpot) -> None:
        """Add a parking spot to this floor."""
        with self._lock:
            self.spots[spot.spot_id] = spot
            self.sections.add(spot.section)
    
    def get_available_spots(self, vehicle_type: VehicleType = None) -> List[ParkingSpot]:
        """Get available spots, optionally filtered by vehicle type."""
        available_spots = []
        
        for spot in self.spots.values():
            if spot.is_available():
                if vehicle_type is None:
                    available_spots.append(spot)
                else:
                    # Create a temporary vehicle to check compatibility
                    temp_vehicle = VehicleFactory.create_vehicle(
                        VehicleInfo("TEMP", vehicle_type)
                    )
                    if spot.can_fit_vehicle(temp_vehicle):
                        available_spots.append(spot)
        
        return available_spots
    
    def get_spot_counts_by_type(self) -> Dict[SpotType, Dict[str, int]]:
        """Get spot counts by type and status."""
        counts = {}
        
        for spot in self.spots.values():
            spot_type = spot.spot_type
            if spot_type not in counts:
                counts[spot_type] = {
                    'total': 0,
                    'available': 0,
                    'occupied': 0,
                    'reserved': 0,
                    'out_of_order': 0
                }
            
            counts[spot_type]['total'] += 1
            
            if spot.status == SpotStatus.AVAILABLE:
                counts[spot_type]['available'] += 1
            elif spot.status == SpotStatus.OCCUPIED:
                counts[spot_type]['occupied'] += 1
            elif spot.status == SpotStatus.RESERVED:
                counts[spot_type]['reserved'] += 1
            elif spot.status == SpotStatus.OUT_OF_ORDER:
                counts[spot_type]['out_of_order'] += 1
        
        return counts
    
    def get_floor_info(self) -> Dict[str, Any]:
        """Get floor information."""
        spot_counts = self.get_spot_counts_by_type()
        total_spots = len(self.spots)
        available_spots = sum(counts['available'] for counts in spot_counts.values())
        
        return {
            'floor_number': self.floor_number,
            'total_spots': total_spots,
            'available_spots': available_spots,
            'occupancy_rate': (total_spots - available_spots) / total_spots if total_spots > 0 else 0,
            'sections': list(self.sections),
            'spot_counts': {
                spot_type.value: counts 
                for spot_type, counts in spot_counts.items()
            }
        }


class ParkingLot:
    """Main parking lot management system."""
    
    def __init__(self, name: str, address: str):
        self.name = name
        self.address = address
        self.floors: Dict[int, ParkingFloor] = {}
        self.active_tickets: Dict[str, ParkingTicket] = {}
        self.completed_tickets: List[ParkingTicket] = []
        
        # Payment processing
        self.payment_processors = {
            PaymentMethod.CASH: CashPaymentProcessor(),
            PaymentMethod.CREDIT_CARD: CardPaymentProcessor(),
            PaymentMethod.DEBIT_CARD: CardPaymentProcessor(),
            PaymentMethod.MOBILE_PAYMENT: MobilePaymentProcessor()
        }
        
        # Pricing strategy
        self.pricing_strategy: PricingStrategy = HourlyPricingStrategy()
        
        # Statistics
        self.total_revenue = 0.0
        self.daily_revenue = 0.0
        self.last_revenue_reset = datetime.now().date()
        
        self._lock = threading.Lock()
        
        print(f"🅿️ Parking Lot '{name}' initialized at {address}")
    
    def add_floor(self, floor: ParkingFloor) -> None:
        """Add a floor to the parking lot."""
        with self._lock:
            self.floors[floor.floor_number] = floor
    
    def set_pricing_strategy(self, strategy: PricingStrategy) -> None:
        """Set the pricing strategy."""
        self.pricing_strategy = strategy
        print(f"Pricing strategy changed to: {strategy.get_strategy_name()}")
    
    def find_available_spot(self, vehicle: Vehicle) -> Optional[ParkingSpot]:
        """Find an available spot for a vehicle."""
        # Get compatible spot types
        compatible_types = vehicle.get_required_spot_types()
        
        # Search floors in order
        for floor_num in sorted(self.floors.keys()):
            floor = self.floors[floor_num]
            
            for spot in floor.spots.values():
                if spot.can_fit_vehicle(vehicle):
                    return spot
        
        return None
    
    def park_vehicle(self, vehicle_info: VehicleInfo) -> Optional[ParkingTicket]:
        """Park a vehicle and return a ticket."""
        with self._lock:
            # Create vehicle
            vehicle = VehicleFactory.create_vehicle(vehicle_info)
            
            # Check if vehicle is already parked
            for ticket in self.active_tickets.values():
                if ticket.vehicle.license_plate == vehicle.license_plate:
                    return None  # Vehicle already parked
            
            # Find available spot
            spot = self.find_available_spot(vehicle)
            if not spot:
                return None  # No available spots
            
            # Occupy the spot
            if spot.occupy_spot(vehicle):
                # Create ticket
                ticket = ParkingTicket(vehicle, spot)
                self.active_tickets[ticket.ticket_id] = ticket
                
                print(f"🚗 Vehicle {vehicle.license_plate} parked at {spot}")
                return ticket
            
            return None
    
    def exit_vehicle(self, ticket_id: str, payment_method: PaymentMethod, 
                    payment_reference: str = "") -> Tuple[bool, str, float]:
        """Process vehicle exit and payment."""
        with self._lock:
            ticket = self.active_tickets.get(ticket_id)
            if not ticket:
                return False, "Invalid ticket", 0.0
            
            if ticket.status != TicketStatus.ACTIVE:
                return False, "Ticket is not active", 0.0
            
            # Calculate fee
            fee = self.pricing_strategy.calculate_fee(ticket)
            
            # Handle lost ticket
            if ticket.status == TicketStatus.LOST:
                fee += ticket.lost_ticket_fee
            
            # Process payment
            try:
                processor = self.payment_processors.get(payment_method)
                if not processor:
                    return False, "Payment method not supported", fee
                
                payment_info = processor.process_payment(fee, payment_method, payment_reference)
                ticket.mark_as_paid(payment_info)
                
                # Vacate spot
                vehicle = ticket.spot.vacate_spot()
                if vehicle:
                    vehicle.exit_time = datetime.now()
                    ticket.exit_time = datetime.now()
                
                # Move ticket to completed
                del self.active_tickets[ticket_id]
                self.completed_tickets.append(ticket)
                
                # Update revenue
                self._update_revenue(fee)
                
                print(f"🚗 Vehicle {vehicle.license_plate} exited. Fee: ${fee:.2f}")
                return True, "Payment successful", fee
                
            except Exception as e:
                return False, f"Payment failed: {str(e)}", fee
    
    def report_lost_ticket(self, license_plate: str) -> Optional[str]:
        """Report a lost ticket and return new ticket ID."""
        license_plate = license_plate.upper()
        
        for ticket in self.active_tickets.values():
            if ticket.vehicle.license_plate == license_plate:
                ticket.mark_as_lost()
                return ticket.ticket_id
        
        return None
    
    def get_parking_fee(self, ticket_id: str) -> Tuple[bool, float, str]:
        """Get parking fee for a ticket."""
        ticket = self.active_tickets.get(ticket_id)
        if not ticket:
            return False, 0.0, "Invalid ticket"
        
        fee = self.pricing_strategy.calculate_fee(ticket)
        
        if ticket.status == TicketStatus.LOST:
            fee += ticket.lost_ticket_fee
            return True, fee, f"Parking fee: ${fee - ticket.lost_ticket_fee:.2f}, Lost ticket fee: ${ticket.lost_ticket_fee:.2f}"
        
        return True, fee, f"Parking fee: ${fee:.2f}"
    
    def get_occupancy_status(self) -> Dict[str, Any]:
        """Get current occupancy status."""
        total_spots = 0
        available_spots = 0
        floor_info = {}
        
        for floor_num, floor in self.floors.items():
            floor_data = floor.get_floor_info()
            floor_info[floor_num] = floor_data
            total_spots += floor_data['total_spots']
            available_spots += floor_data['available_spots']
        
        occupancy_rate = (total_spots - available_spots) / total_spots if total_spots > 0 else 0
        
        return {
            'total_spots': total_spots,
            'available_spots': available_spots,
            'occupied_spots': total_spots - available_spots,
            'occupancy_rate': occupancy_rate,
            'floors': floor_info,
            'active_vehicles': len(self.active_tickets)
        }
    
    def get_revenue_report(self) -> Dict[str, Any]:
        """Get revenue report."""
        self._reset_daily_revenue_if_needed()
        
        # Calculate revenue by payment method
        payment_method_revenue = {}
        for ticket in self.completed_tickets:
            if ticket.payment_info:
                method = ticket.payment_info.method.value
                payment_method_revenue[method] = payment_method_revenue.get(method, 0) + ticket.payment_info.amount
        
        # Calculate average parking duration
        total_duration = sum(
            ticket.calculate_parking_duration().total_seconds() 
            for ticket in self.completed_tickets
        )
        avg_duration_hours = (total_duration / len(self.completed_tickets) / 3600) if self.completed_tickets else 0
        
        return {
            'total_revenue': self.total_revenue,
            'daily_revenue': self.daily_revenue,
            'total_transactions': len(self.completed_tickets),
            'active_tickets': len(self.active_tickets),
            'payment_method_breakdown': payment_method_revenue,
            'average_parking_duration_hours': avg_duration_hours,
            'pricing_strategy': self.pricing_strategy.get_strategy_name()
        }
    
    def _update_revenue(self, amount: float) -> None:
        """Update revenue tracking."""
        self._reset_daily_revenue_if_needed()
        self.total_revenue += amount
        self.daily_revenue += amount
    
    def _reset_daily_revenue_if_needed(self) -> None:
        """Reset daily revenue if new day."""
        if self.last_revenue_reset < datetime.now().date():
            self.daily_revenue = 0.0
            self.last_revenue_reset = datetime.now().date()
    
    def get_parking_lot_info(self) -> Dict[str, Any]:
        """Get comprehensive parking lot information."""
        occupancy = self.get_occupancy_status()
        revenue = self.get_revenue_report()
        
        return {
            'name': self.name,
            'address': self.address,
            'total_floors': len(self.floors),
            'occupancy': occupancy,
            'revenue': revenue,
            'pricing_strategy': self.pricing_strategy.get_strategy_name()
        }


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_parking_lot_system():
    """Demonstrate the parking lot system."""
    print("=== PARKING LOT SYSTEM DEMONSTRATION ===\n")
    
    # Initialize parking lot
    parking_lot = ParkingLot("Downtown Parking Garage", "123 Main Street")
    
    print("1. PARKING LOT SETUP:")
    
    # Create floors and spots
    for floor_num in range(1, 4):  # 3 floors
        floor = ParkingFloor(floor_num)
        
        # Add different types of spots
        spot_configs = [
            (SpotType.MOTORCYCLE, 10, "A"),
            (SpotType.COMPACT, 20, "B"),
            (SpotType.REGULAR, 30, "C"),
            (SpotType.LARGE, 5, "D"),
            (SpotType.HANDICAPPED, 3, "E"),
            (SpotType.ELECTRIC, 8, "F"),
            (SpotType.VIP, 4, "G")
        ]
        
        for spot_type, count, section in spot_configs:
            for i in range(count):
                spot_id = f"F{floor_num}-{section}{i+1:02d}"
                spot = ParkingSpot(spot_id, spot_type, floor_num, section)
                floor.add_spot(spot)
        
        parking_lot.add_floor(floor)
        print(f"   ✓ Floor {floor_num} created with {len(floor.spots)} spots")
    
    print()
    
    # Set pricing strategy
    print("2. PRICING STRATEGY SETUP:")
    parking_lot.set_pricing_strategy(DynamicPricingStrategy(base_rate=3.0, peak_multiplier=2.0))
    
    # Show initial status
    occupancy = parking_lot.get_occupancy_status()
    print(f"   ✓ Total spots: {occupancy['total_spots']}")
    print(f"   ✓ Available spots: {occupancy['available_spots']}")
    print()
    
    # Simulate vehicle parking
    print("3. VEHICLE PARKING SIMULATION:")
    
    vehicles_to_park = [
        VehicleInfo("ABC123", VehicleType.CAR, "John Doe", "555-0001"),
        VehicleInfo("XYZ789", VehicleType.MOTORCYCLE, "Jane Smith", "555-0002"),
        VehicleInfo("TRK456", VehicleType.TRUCK, "Bob Johnson", "555-0003"),
        VehicleInfo("ELC001", VehicleType.CAR, "Alice Green", "555-0004", is_electric=True),
        VehicleInfo("HND001", VehicleType.CAR, "Charlie Brown", "555-0005", is_handicapped=True),
        VehicleInfo("BUS999", VehicleType.BUS, "Metro Transit", "555-0006"),
        VehicleInfo("CAR001", VehicleType.CAR, "David Wilson", "555-0007"),
        VehicleInfo("CAR002", VehicleType.CAR, "Emma Davis", "555-0008"),
        VehicleInfo("MTR002", VehicleType.MOTORCYCLE, "Frank Miller", "555-0009"),
        VehicleInfo("CAR003", VehicleType.CAR, "Grace Taylor", "555-0010")
    ]
    
    parked_tickets = []
    
    for vehicle_info in vehicles_to_park:
        ticket = parking_lot.park_vehicle(vehicle_info)
        if ticket:
            parked_tickets.append(ticket)
            print(f"   ✓ Parked {vehicle_info.license_plate} ({vehicle_info.vehicle_type.value}) at {ticket.spot.spot_id}")
        else:
            print(f"   ✗ Failed to park {vehicle_info.license_plate} - no available spots")
    
    print()
    
    # Show updated occupancy
    print("4. OCCUPANCY STATUS AFTER PARKING:")
    occupancy = parking_lot.get_occupancy_status()
    print(f"   Available spots: {occupancy['available_spots']}")
    print(f"   Occupied spots: {occupancy['occupied_spots']}")
    print(f"   Occupancy rate: {occupancy['occupancy_rate']:.1%}")
    
    # Show floor-wise breakdown
    for floor_num, floor_info in occupancy['floors'].items():
        print(f"   Floor {floor_num}: {floor_info['available_spots']}/{floor_info['total_spots']} available")
    
    print()
    
    # Simulate some time passing
    print("5. SIMULATING TIME PASSAGE (2 hours):")
    for ticket in parked_tickets:
        # Simulate 2 hours of parking
        ticket.entry_time = datetime.now() - timedelta(hours=2)
    
    # Check parking fees
    print("   Current parking fees:")
    for ticket in parked_tickets[:5]:  # Show first 5
        success, fee, message = parking_lot.get_parking_fee(ticket.ticket_id)
        if success:
            print(f"   {ticket.vehicle.license_plate}: ${fee:.2f}")
    
    print()
    
    # Simulate vehicle exits
    print("6. VEHICLE EXIT SIMULATION:")
    
    payment_methods = [
        PaymentMethod.CASH,
        PaymentMethod.CREDIT_CARD,
        PaymentMethod.DEBIT_CARD,
        PaymentMethod.MOBILE_PAYMENT
    ]
    
    # Exit first 6 vehicles
    for i, ticket in enumerate(parked_tickets[:6]):
        payment_method = payment_methods[i % len(payment_methods)]
        success, message, fee = parking_lot.exit_vehicle(
            ticket.ticket_id, 
            payment_method, 
            f"REF{i+1:03d}"
        )
        
        if success:
            print(f"   ✓ {ticket.vehicle.license_plate} exited - {message} (${fee:.2f})")
        else:
            print(f"   ✗ {ticket.vehicle.license_plate} exit failed - {message}")
    
    print()
    
    # Test lost ticket scenario
    print("7. LOST TICKET SCENARIO:")
    remaining_tickets = parked_tickets[6:]
    if remaining_tickets:
        lost_ticket = remaining_tickets[0]
        lost_ticket_id = parking_lot.report_lost_ticket(lost_ticket.vehicle.license_plate)
        
        if lost_ticket_id:
            print(f"   ✓ Reported lost ticket for {lost_ticket.vehicle.license_plate}")
            
            # Check fee with lost ticket penalty
            success, fee, message = parking_lot.get_parking_fee(lost_ticket_id)
            print(f"   Fee with lost ticket penalty: ${fee:.2f}")
            
            # Process exit with lost ticket
            success, exit_message, exit_fee = parking_lot.exit_vehicle(
                lost_ticket_id, 
                PaymentMethod.CASH
            )
            print(f"   Exit result: {exit_message}")
    
    print()
    
    # Test different pricing strategies
    print("8. PRICING STRATEGY COMPARISON:")
    
    # Create a test ticket for comparison
    test_vehicle = VehicleFactory.create_vehicle(
        VehicleInfo("TEST123", VehicleType.CAR)
    )
    test_spot = ParkingSpot("TEST01", SpotType.REGULAR, 1, "TEST")
    test_ticket = ParkingTicket(test_vehicle, test_spot, datetime.now() - timedelta(hours=3))
    
    strategies = [
        HourlyPricingStrategy(base_rate=5.0),
        DynamicPricingStrategy(base_rate=4.0, peak_multiplier=1.8),
        FlatRatePricingStrategy(daily_rate=30.0)
    ]
    
    print("   3-hour parking fees for regular car:")
    for strategy in strategies:
        fee = strategy.calculate_fee(test_ticket)
        print(f"   {strategy.get_strategy_name()}: ${fee:.2f}")
    
    print()
    
    # Show final reports
    print("9. FINAL REPORTS:")
    
    # Revenue report
    revenue_report = parking_lot.get_revenue_report()
    print(f"   Total Revenue: ${revenue_report['total_revenue']:.2f}")
    print(f"   Daily Revenue: ${revenue_report['daily_revenue']:.2f}")
    print(f"   Total Transactions: {revenue_report['total_transactions']}")
    print(f"   Active Tickets: {revenue_report['active_tickets']}")
    print(f"   Average Parking Duration: {revenue_report['average_parking_duration_hours']:.1f} hours")
    
    # Payment method breakdown
    if revenue_report['payment_method_breakdown']:
        print(f"   Payment Method Breakdown:")
        for method, amount in revenue_report['payment_method_breakdown'].items():
            print(f"     {method.replace('_', ' ').title()}: ${amount:.2f}")
    
    print()
    
    # Final occupancy
    final_occupancy = parking_lot.get_occupancy_status()
    print(f"   Final Occupancy: {final_occupancy['occupied_spots']}/{final_occupancy['total_spots']} spots")
    print(f"   Final Occupancy Rate: {final_occupancy['occupancy_rate']:.1%}")
    
    # Spot type utilization
    print(f"   Spot Type Utilization:")
    for floor_num, floor_info in final_occupancy['floors'].items():
        print(f"     Floor {floor_num}:")
        for spot_type, counts in floor_info['spot_counts'].items():
            utilization = counts['occupied'] / counts['total'] if counts['total'] > 0 else 0
            print(f"       {spot_type.replace('_', ' ').title()}: {counts['occupied']}/{counts['total']} ({utilization:.1%})")
    
    print()
    print("=== PARKING LOT SYSTEM DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_parking_lot_system()
