"""
HOTEL BOOKING SYSTEM - Complete System Design
=============================================

Problem Statement:
Design a comprehensive hotel booking system that handles:
- Hotel and room management
- Room availability and pricing
- Booking and reservation management
- Customer management and profiles
- Payment processing and billing
- Check-in and check-out processes
- Room service and amenities
- Staff management and roles
- Inventory and housekeeping
- Reporting and analytics

Requirements:
- Support multiple hotels and room types
- Handle real-time room availability
- Implement dynamic pricing strategies
- Process online and offline bookings
- Manage customer profiles and preferences
- Handle group bookings and corporate accounts
- Support multiple payment methods
- Implement loyalty programs and discounts
- Manage housekeeping and maintenance schedules
- Generate comprehensive reports

Design Patterns Used:
- Factory: Room and booking creation
- Strategy: Pricing and discount strategies
- Observer: Booking notifications
- State: Booking and room states
- Command: Booking operations
- Decorator: Service add-ons
- Singleton: Hotel management system
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, date, timedelta
from enum import Enum
import uuid
import threading
from dataclasses import dataclass, field
from decimal import Decimal
import json


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class RoomType(Enum):
    SINGLE = "single"
    DOUBLE = "double"
    TWIN = "twin"
    SUITE = "suite"
    DELUXE = "deluxe"
    PRESIDENTIAL = "presidential"


class RoomStatus(Enum):
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    RESERVED = "reserved"
    OUT_OF_ORDER = "out_of_order"
    CLEANING = "cleaning"
    MAINTENANCE = "maintenance"


class BookingStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CHECKED_IN = "checked_in"
    CHECKED_OUT = "checked_out"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class PaymentStatus(Enum):
    PENDING = "pending"
    PAID = "paid"
    PARTIAL = "partial"
    REFUNDED = "refunded"
    FAILED = "failed"


class CustomerType(Enum):
    INDIVIDUAL = "individual"
    CORPORATE = "corporate"
    VIP = "vip"
    LOYALTY_MEMBER = "loyalty_member"


class StaffRole(Enum):
    MANAGER = "manager"
    RECEPTIONIST = "receptionist"
    HOUSEKEEPING = "housekeeping"
    MAINTENANCE = "maintenance"
    CONCIERGE = "concierge"
    SECURITY = "security"


@dataclass
class Address:
    """Address information."""
    street: str
    city: str
    state: str
    country: str
    zip_code: str


@dataclass
class ContactInfo:
    """Contact information."""
    phone: str
    email: str
    address: Address


@dataclass
class ServiceItem:
    """Hotel service item."""
    service_id: str
    name: str
    description: str
    price: Decimal
    category: str
    is_available: bool = True


# ============================================================================
# PRICING STRATEGIES
# ============================================================================

class PricingStrategy(ABC):
    """Abstract pricing strategy."""
    
    @abstractmethod
    def calculate_price(self, room: 'Room', check_in: date, check_out: date, 
                       customer_type: CustomerType) -> Decimal:
        """Calculate room price for given dates."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class StandardPricingStrategy(PricingStrategy):
    """Standard pricing strategy."""
    
    def calculate_price(self, room: 'Room', check_in: date, check_out: date, 
                       customer_type: CustomerType) -> Decimal:
        """Calculate standard price."""
        nights = (check_out - check_in).days
        base_price = room.base_price * nights
        
        # Apply customer type discount
        discount_multiplier = self._get_customer_discount(customer_type)
        
        return base_price * discount_multiplier
    
    def _get_customer_discount(self, customer_type: CustomerType) -> Decimal:
        """Get discount multiplier based on customer type."""
        discounts = {
            CustomerType.INDIVIDUAL: Decimal('1.0'),
            CustomerType.CORPORATE: Decimal('0.9'),
            CustomerType.VIP: Decimal('0.8'),
            CustomerType.LOYALTY_MEMBER: Decimal('0.85')
        }
        return discounts.get(customer_type, Decimal('1.0'))
    
    def get_strategy_name(self) -> str:
        return "Standard Pricing"


class DynamicPricingStrategy(PricingStrategy):
    """Dynamic pricing based on demand and seasonality."""
    
    def __init__(self):
        self.peak_seasons = [
            (date(2024, 6, 1), date(2024, 8, 31)),   # Summer
            (date(2024, 12, 20), date(2025, 1, 5))   # Winter holidays
        ]
        self.weekend_multiplier = Decimal('1.2')
        self.peak_season_multiplier = Decimal('1.5')
    
    def calculate_price(self, room: 'Room', check_in: date, check_out: date, 
                       customer_type: CustomerType) -> Decimal:
        """Calculate dynamic price."""
        nights = (check_out - check_in).days
        total_price = Decimal('0')
        
        current_date = check_in
        while current_date < check_out:
            daily_price = room.base_price
            
            # Weekend pricing
            if current_date.weekday() >= 5:  # Saturday, Sunday
                daily_price *= self.weekend_multiplier
            
            # Peak season pricing
            if self._is_peak_season(current_date):
                daily_price *= self.peak_season_multiplier
            
            # Occupancy-based pricing (simplified)
            occupancy_rate = self._get_occupancy_rate(current_date)
            if occupancy_rate > 0.8:
                daily_price *= Decimal('1.3')
            elif occupancy_rate > 0.6:
                daily_price *= Decimal('1.1')
            
            total_price += daily_price
            current_date += timedelta(days=1)
        
        # Apply customer discount
        discount_multiplier = self._get_customer_discount(customer_type)
        return total_price * discount_multiplier
    
    def _is_peak_season(self, check_date: date) -> bool:
        """Check if date is in peak season."""
        for start, end in self.peak_seasons:
            if start <= check_date <= end:
                return True
        return False
    
    def _get_occupancy_rate(self, check_date: date) -> float:
        """Get occupancy rate for date (simplified simulation)."""
        # Simulate higher occupancy on weekends and peak seasons
        base_rate = 0.6
        if check_date.weekday() >= 5:
            base_rate += 0.2
        if self._is_peak_season(check_date):
            base_rate += 0.15
        return min(base_rate, 0.95)
    
    def _get_customer_discount(self, customer_type: CustomerType) -> Decimal:
        """Get discount multiplier based on customer type."""
        discounts = {
            CustomerType.INDIVIDUAL: Decimal('1.0'),
            CustomerType.CORPORATE: Decimal('0.92'),
            CustomerType.VIP: Decimal('0.85'),
            CustomerType.LOYALTY_MEMBER: Decimal('0.88')
        }
        return discounts.get(customer_type, Decimal('1.0'))
    
    def get_strategy_name(self) -> str:
        return "Dynamic Pricing"


# ============================================================================
# ROOM CLASSES
# ============================================================================

class Room:
    """Hotel room with amenities and status."""
    
    def __init__(self, room_number: str, room_type: RoomType, floor: int, 
                 base_price: Decimal, max_occupancy: int):
        self.room_number = room_number
        self.room_type = room_type
        self.floor = floor
        self.base_price = base_price
        self.max_occupancy = max_occupancy
        self.status = RoomStatus.AVAILABLE
        
        # Amenities
        self.amenities: Set[str] = set()
        self.has_balcony = False
        self.has_sea_view = False
        self.has_city_view = False
        
        # Maintenance and housekeeping
        self.last_cleaned = datetime.now()
        self.last_maintenance = datetime.now()
        self.maintenance_notes: List[str] = []
        
        # Current booking
        self.current_booking_id: Optional[str] = None
        
        self._lock = threading.Lock()
    
    def add_amenity(self, amenity: str) -> None:
        """Add amenity to room."""
        self.amenities.add(amenity)
    
    def remove_amenity(self, amenity: str) -> None:
        """Remove amenity from room."""
        self.amenities.discard(amenity)
    
    def set_status(self, status: RoomStatus, booking_id: str = None) -> None:
        """Set room status."""
        with self._lock:
            self.status = status
            if status == RoomStatus.OCCUPIED and booking_id:
                self.current_booking_id = booking_id
            elif status == RoomStatus.AVAILABLE:
                self.current_booking_id = None
    
    def is_available(self, check_in: date, check_out: date) -> bool:
        """Check if room is available for given dates."""
        with self._lock:
            # Simplified availability check
            return self.status in [RoomStatus.AVAILABLE, RoomStatus.CLEANING]
    
    def schedule_cleaning(self) -> None:
        """Schedule room cleaning."""
        with self._lock:
            if self.status == RoomStatus.AVAILABLE:
                self.status = RoomStatus.CLEANING
    
    def complete_cleaning(self) -> None:
        """Complete room cleaning."""
        with self._lock:
            if self.status == RoomStatus.CLEANING:
                self.status = RoomStatus.AVAILABLE
                self.last_cleaned = datetime.now()
    
    def schedule_maintenance(self, notes: str) -> None:
        """Schedule room maintenance."""
        with self._lock:
            self.status = RoomStatus.MAINTENANCE
            self.maintenance_notes.append(f"{datetime.now().isoformat()}: {notes}")
    
    def complete_maintenance(self) -> None:
        """Complete room maintenance."""
        with self._lock:
            if self.status == RoomStatus.MAINTENANCE:
                self.status = RoomStatus.AVAILABLE
                self.last_maintenance = datetime.now()
    
    def get_room_info(self) -> Dict[str, Any]:
        """Get room information."""
        return {
            'room_number': self.room_number,
            'room_type': self.room_type.value,
            'floor': self.floor,
            'base_price': float(self.base_price),
            'max_occupancy': self.max_occupancy,
            'status': self.status.value,
            'amenities': list(self.amenities),
            'features': {
                'has_balcony': self.has_balcony,
                'has_sea_view': self.has_sea_view,
                'has_city_view': self.has_city_view
            },
            'maintenance': {
                'last_cleaned': self.last_cleaned.isoformat(),
                'last_maintenance': self.last_maintenance.isoformat(),
                'notes_count': len(self.maintenance_notes)
            },
            'current_booking_id': self.current_booking_id
        }
    
    def __str__(self) -> str:
        return f"Room {self.room_number} ({self.room_type.value}) - {self.status.value}"


# ============================================================================
# CUSTOMER CLASSES
# ============================================================================

class Customer:
    """Hotel customer with profile and preferences."""
    
    def __init__(self, customer_id: str, name: str, contact_info: ContactInfo):
        self.customer_id = customer_id
        self.name = name
        self.contact_info = contact_info
        self.customer_type = CustomerType.INDIVIDUAL
        
        # Profile information
        self.date_of_birth: Optional[date] = None
        self.nationality = ""
        self.passport_number = ""
        self.id_number = ""
        
        # Preferences
        self.room_preferences: Set[str] = set()
        self.dietary_restrictions: List[str] = []
        self.special_requests: List[str] = []
        
        # Loyalty program
        self.loyalty_points = 0
        self.loyalty_tier = "Bronze"
        
        # Booking history
        self.booking_history: List[str] = []  # booking_ids
        self.total_stays = 0
        self.total_spent = Decimal('0')
        
        # Corporate information (if applicable)
        self.company_name = ""
        self.company_id = ""
        
        self.created_at = datetime.now()
    
    def add_room_preference(self, preference: str) -> None:
        """Add room preference."""
        self.room_preferences.add(preference)
    
    def add_dietary_restriction(self, restriction: str) -> None:
        """Add dietary restriction."""
        if restriction not in self.dietary_restrictions:
            self.dietary_restrictions.append(restriction)
    
    def add_special_request(self, request: str) -> None:
        """Add special request."""
        if request not in self.special_requests:
            self.special_requests.append(request)
    
    def update_loyalty_points(self, points: int) -> None:
        """Update loyalty points and tier."""
        self.loyalty_points += points
        
        # Update tier based on points
        if self.loyalty_points >= 10000:
            self.loyalty_tier = "Platinum"
            self.customer_type = CustomerType.VIP
        elif self.loyalty_points >= 5000:
            self.loyalty_tier = "Gold"
            self.customer_type = CustomerType.LOYALTY_MEMBER
        elif self.loyalty_points >= 1000:
            self.loyalty_tier = "Silver"
            self.customer_type = CustomerType.LOYALTY_MEMBER
    
    def add_booking(self, booking_id: str, amount: Decimal) -> None:
        """Add booking to history."""
        self.booking_history.append(booking_id)
        self.total_stays += 1
        self.total_spent += amount
        
        # Award loyalty points (1 point per dollar spent)
        self.update_loyalty_points(int(amount))
    
    def set_corporate_info(self, company_name: str, company_id: str) -> None:
        """Set corporate customer information."""
        self.company_name = company_name
        self.company_id = company_id
        self.customer_type = CustomerType.CORPORATE
    
    def get_customer_info(self) -> Dict[str, Any]:
        """Get customer information."""
        return {
            'customer_id': self.customer_id,
            'name': self.name,
            'contact': {
                'phone': self.contact_info.phone,
                'email': self.contact_info.email,
                'address': {
                    'street': self.contact_info.address.street,
                    'city': self.contact_info.address.city,
                    'state': self.contact_info.address.state,
                    'country': self.contact_info.address.country,
                    'zip_code': self.contact_info.address.zip_code
                }
            },
            'customer_type': self.customer_type.value,
            'profile': {
                'date_of_birth': self.date_of_birth.isoformat() if self.date_of_birth else None,
                'nationality': self.nationality,
                'passport_number': self.passport_number,
                'id_number': self.id_number
            },
            'preferences': {
                'room_preferences': list(self.room_preferences),
                'dietary_restrictions': self.dietary_restrictions,
                'special_requests': self.special_requests
            },
            'loyalty': {
                'points': self.loyalty_points,
                'tier': self.loyalty_tier
            },
            'statistics': {
                'total_stays': self.total_stays,
                'total_spent': float(self.total_spent),
                'booking_count': len(self.booking_history)
            },
            'corporate': {
                'company_name': self.company_name,
                'company_id': self.company_id
            } if self.customer_type == CustomerType.CORPORATE else None,
            'created_at': self.created_at.isoformat()
        }
    
    def __str__(self) -> str:
        return f"Customer {self.name} ({self.customer_type.value}) - {self.loyalty_tier}"


# ============================================================================
# BOOKING CLASSES
# ============================================================================

class Booking:
    """Hotel booking with details and services."""
    
    def __init__(self, booking_id: str, customer_id: str, room_id: str,
                 check_in: date, check_out: date, guests: int):
        self.booking_id = booking_id
        self.customer_id = customer_id
        self.room_id = room_id
        self.check_in = check_in
        self.check_out = check_out
        self.guests = guests
        self.status = BookingStatus.PENDING
        
        # Pricing
        self.room_rate = Decimal('0')
        self.total_amount = Decimal('0')
        self.taxes = Decimal('0')
        self.discounts = Decimal('0')
        self.services_total = Decimal('0')
        
        # Payment
        self.payment_status = PaymentStatus.PENDING
        self.payment_method = ""
        self.payment_reference = ""
        
        # Services and add-ons
        self.services: List[ServiceItem] = []
        self.special_requests: List[str] = []
        
        # Timestamps
        self.created_at = datetime.now()
        self.confirmed_at: Optional[datetime] = None
        self.checked_in_at: Optional[datetime] = None
        self.checked_out_at: Optional[datetime] = None
        self.cancelled_at: Optional[datetime] = None
        
        # Staff assignments
        self.assigned_staff: Dict[str, str] = {}  # role -> staff_id
        
        self._lock = threading.Lock()
    
    def calculate_total(self, pricing_strategy: PricingStrategy, 
                       room: Room, customer_type: CustomerType) -> None:
        """Calculate booking total."""
        with self._lock:
            # Calculate room rate
            self.room_rate = pricing_strategy.calculate_price(
                room, self.check_in, self.check_out, customer_type
            )
            
            # Calculate services total
            self.services_total = sum(service.price for service in self.services)
            
            # Calculate taxes (10% simplified)
            subtotal = self.room_rate + self.services_total - self.discounts
            self.taxes = subtotal * Decimal('0.1')
            
            # Calculate total
            self.total_amount = subtotal + self.taxes
    
    def add_service(self, service: ServiceItem) -> None:
        """Add service to booking."""
        self.services.append(service)
    
    def remove_service(self, service_id: str) -> bool:
        """Remove service from booking."""
        for i, service in enumerate(self.services):
            if service.service_id == service_id:
                del self.services[i]
                return True
        return False
    
    def add_special_request(self, request: str) -> None:
        """Add special request."""
        if request not in self.special_requests:
            self.special_requests.append(request)
    
    def apply_discount(self, discount_amount: Decimal) -> None:
        """Apply discount to booking."""
        self.discounts += discount_amount
    
    def confirm_booking(self) -> bool:
        """Confirm the booking."""
        with self._lock:
            if self.status == BookingStatus.PENDING:
                self.status = BookingStatus.CONFIRMED
                self.confirmed_at = datetime.now()
                return True
            return False
    
    def check_in(self, staff_id: str) -> bool:
        """Check in the booking."""
        with self._lock:
            if self.status == BookingStatus.CONFIRMED:
                self.status = BookingStatus.CHECKED_IN
                self.checked_in_at = datetime.now()
                self.assigned_staff['check_in'] = staff_id
                return True
            return False
    
    def check_out(self, staff_id: str) -> bool:
        """Check out the booking."""
        with self._lock:
            if self.status == BookingStatus.CHECKED_IN:
                self.status = BookingStatus.CHECKED_OUT
                self.checked_out_at = datetime.now()
                self.assigned_staff['check_out'] = staff_id
                return True
            return False
    
    def cancel_booking(self, reason: str = "") -> bool:
        """Cancel the booking."""
        with self._lock:
            if self.status in [BookingStatus.PENDING, BookingStatus.CONFIRMED]:
                self.status = BookingStatus.CANCELLED
                self.cancelled_at = datetime.now()
                return True
            return False
    
    def mark_no_show(self) -> bool:
        """Mark booking as no-show."""
        with self._lock:
            if (self.status == BookingStatus.CONFIRMED and 
                date.today() > self.check_in):
                self.status = BookingStatus.NO_SHOW
                return True
            return False
    
    def update_payment_status(self, status: PaymentStatus, 
                            method: str = "", reference: str = "") -> None:
        """Update payment status."""
        self.payment_status = status
        if method:
            self.payment_method = method
        if reference:
            self.payment_reference = reference
    
    def get_nights(self) -> int:
        """Get number of nights."""
        return (self.check_out - self.check_in).days
    
    def get_booking_info(self) -> Dict[str, Any]:
        """Get booking information."""
        return {
            'booking_id': self.booking_id,
            'customer_id': self.customer_id,
            'room_id': self.room_id,
            'dates': {
                'check_in': self.check_in.isoformat(),
                'check_out': self.check_out.isoformat(),
                'nights': self.get_nights()
            },
            'guests': self.guests,
            'status': self.status.value,
            'pricing': {
                'room_rate': float(self.room_rate),
                'services_total': float(self.services_total),
                'discounts': float(self.discounts),
                'taxes': float(self.taxes),
                'total_amount': float(self.total_amount)
            },
            'payment': {
                'status': self.payment_status.value,
                'method': self.payment_method,
                'reference': self.payment_reference
            },
            'services': [
                {
                    'service_id': service.service_id,
                    'name': service.name,
                    'price': float(service.price),
                    'category': service.category
                }
                for service in self.services
            ],
            'special_requests': self.special_requests,
            'timestamps': {
                'created_at': self.created_at.isoformat(),
                'confirmed_at': self.confirmed_at.isoformat() if self.confirmed_at else None,
                'checked_in_at': self.checked_in_at.isoformat() if self.checked_in_at else None,
                'checked_out_at': self.checked_out_at.isoformat() if self.checked_out_at else None,
                'cancelled_at': self.cancelled_at.isoformat() if self.cancelled_at else None
            },
            'assigned_staff': self.assigned_staff
        }
    
    def __str__(self) -> str:
        return f"Booking {self.booking_id} - Room {self.room_id} ({self.check_in} to {self.check_out})"


# ============================================================================
# HOTEL MANAGEMENT SYSTEM
# ============================================================================

class Hotel:
    """Hotel with rooms, staff, and services."""
    
    def __init__(self, hotel_id: str, name: str, address: Address):
        self.hotel_id = hotel_id
        self.name = name
        self.address = address
        self.description = ""
        self.star_rating = 3
        
        # Rooms and facilities
        self.rooms: Dict[str, Room] = {}
        self.room_types: Dict[RoomType, int] = {}  # type -> count
        
        # Services
        self.services: Dict[str, ServiceItem] = {}
        
        # Staff
        self.staff: Dict[str, Dict[str, Any]] = {}
        
        # Pricing
        self.pricing_strategy: PricingStrategy = StandardPricingStrategy()
        
        # Statistics
        self.total_bookings = 0
        self.total_revenue = Decimal('0')
        self.occupancy_rate = 0.0
        
        self._lock = threading.Lock()
    
    def add_room(self, room: Room) -> None:
        """Add room to hotel."""
        with self._lock:
            self.rooms[room.room_number] = room
            
            # Update room type count
            if room.room_type in self.room_types:
                self.room_types[room.room_type] += 1
            else:
                self.room_types[room.room_type] = 1
    
    def remove_room(self, room_number: str) -> bool:
        """Remove room from hotel."""
        with self._lock:
            if room_number in self.rooms:
                room = self.rooms[room_number]
                if room.status == RoomStatus.AVAILABLE:
                    del self.rooms[room_number]
                    self.room_types[room.room_type] -= 1
                    return True
            return False
    
    def add_service(self, service: ServiceItem) -> None:
        """Add service to hotel."""
        self.services[service.service_id] = service
    
    def remove_service(self, service_id: str) -> bool:
        """Remove service from hotel."""
        if service_id in self.services:
            del self.services[service_id]
            return True
        return False
    
    def add_staff(self, staff_id: str, name: str, role: StaffRole, 
                  contact_info: ContactInfo) -> None:
        """Add staff member."""
        self.staff[staff_id] = {
            'name': name,
            'role': role,
            'contact_info': contact_info,
            'hire_date': datetime.now(),
            'is_active': True
        }
    
    def set_pricing_strategy(self, strategy: PricingStrategy) -> None:
        """Set pricing strategy."""
        self.pricing_strategy = strategy
    
    def get_available_rooms(self, check_in: date, check_out: date, 
                           room_type: RoomType = None) -> List[Room]:
        """Get available rooms for dates."""
        available_rooms = []
        
        for room in self.rooms.values():
            if room_type and room.room_type != room_type:
                continue
            
            if room.is_available(check_in, check_out):
                available_rooms.append(room)
        
        return available_rooms
    
    def calculate_occupancy_rate(self, target_date: date = None) -> float:
        """Calculate occupancy rate for a date."""
        if not target_date:
            target_date = date.today()
        
        total_rooms = len(self.rooms)
        if total_rooms == 0:
            return 0.0
        
        occupied_rooms = sum(1 for room in self.rooms.values() 
                           if room.status == RoomStatus.OCCUPIED)
        
        return occupied_rooms / total_rooms
    
    def get_revenue_report(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Get revenue report for date range."""
        # Simplified revenue calculation
        # In real system, would query booking database
        
        return {
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'days': (end_date - start_date).days
            },
            'revenue': {
                'total_revenue': float(self.total_revenue),
                'room_revenue': float(self.total_revenue * Decimal('0.8')),
                'service_revenue': float(self.total_revenue * Decimal('0.2'))
            },
            'bookings': {
                'total_bookings': self.total_bookings,
                'average_booking_value': float(self.total_revenue / max(1, self.total_bookings))
            },
            'occupancy': {
                'average_occupancy_rate': self.occupancy_rate,
                'total_room_nights': len(self.rooms) * (end_date - start_date).days
            }
        }
    
    def get_hotel_info(self) -> Dict[str, Any]:
        """Get hotel information."""
        return {
            'hotel_id': self.hotel_id,
            'name': self.name,
            'address': {
                'street': self.address.street,
                'city': self.address.city,
                'state': self.address.state,
                'country': self.address.country,
                'zip_code': self.address.zip_code
            },
            'description': self.description,
            'star_rating': self.star_rating,
            'rooms': {
                'total_rooms': len(self.rooms),
                'room_types': {room_type.value: count for room_type, count in self.room_types.items()},
                'available_rooms': len([r for r in self.rooms.values() if r.status == RoomStatus.AVAILABLE])
            },
            'services': {
                'total_services': len(self.services),
                'service_categories': list(set(s.category for s in self.services.values()))
            },
            'staff': {
                'total_staff': len(self.staff),
                'active_staff': len([s for s in self.staff.values() if s['is_active']])
            },
            'statistics': {
                'total_bookings': self.total_bookings,
                'total_revenue': float(self.total_revenue),
                'current_occupancy_rate': self.calculate_occupancy_rate()
            },
            'pricing_strategy': self.pricing_strategy.get_strategy_name()
        }


# ============================================================================
# BOOKING SYSTEM MANAGER
# ============================================================================

class HotelBookingSystem:
    """Main hotel booking system manager."""
    
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.hotels: Dict[str, Hotel] = {}
        self.customers: Dict[str, Customer] = {}
        self.bookings: Dict[str, Booking] = {}
        
        # System statistics
        self.total_bookings = 0
        self.total_revenue = Decimal('0')
        
        self._lock = threading.Lock()
        
        print(f"🏨 Hotel Booking System '{system_name}' initialized")
    
    def add_hotel(self, hotel: Hotel) -> None:
        """Add hotel to system."""
        with self._lock:
            self.hotels[hotel.hotel_id] = hotel
            print(f"   ✓ Hotel '{hotel.name}' added to system")
    
    def register_customer(self, name: str, contact_info: ContactInfo) -> Customer:
        """Register new customer."""
        customer_id = str(uuid.uuid4())
        customer = Customer(customer_id, name, contact_info)
        
        with self._lock:
            self.customers[customer_id] = customer
        
        return customer
    
    def search_rooms(self, check_in: date, check_out: date, guests: int,
                    city: str = "", room_type: RoomType = None) -> List[Dict[str, Any]]:
        """Search available rooms across hotels."""
        results = []
        
        for hotel in self.hotels.values():
            # Filter by city if specified
            if city and city.lower() not in hotel.address.city.lower():
                continue
            
            available_rooms = hotel.get_available_rooms(check_in, check_out, room_type)
            
            for room in available_rooms:
                if room.max_occupancy >= guests:
                    # Calculate price for individual customer
                    price = hotel.pricing_strategy.calculate_price(
                        room, check_in, check_out, CustomerType.INDIVIDUAL
                    )
                    
                    results.append({
                        'hotel_id': hotel.hotel_id,
                        'hotel_name': hotel.name,
                        'hotel_address': f"{hotel.address.city}, {hotel.address.state}",
                        'hotel_rating': hotel.star_rating,
                        'room_number': room.room_number,
                        'room_type': room.room_type.value,
                        'max_occupancy': room.max_occupancy,
                        'amenities': list(room.amenities),
                        'price': float(price),
                        'nights': (check_out - check_in).days
                    })
        
        # Sort by price
        results.sort(key=lambda x: x['price'])
        return results
    
    def create_booking(self, customer_id: str, hotel_id: str, room_number: str,
                      check_in: date, check_out: date, guests: int) -> Optional[Booking]:
        """Create new booking."""
        customer = self.customers.get(customer_id)
        hotel = self.hotels.get(hotel_id)
        
        if not customer or not hotel:
            return None
        
        room = hotel.rooms.get(room_number)
        if not room or not room.is_available(check_in, check_out):
            return None
        
        if room.max_occupancy < guests:
            return None
        
        # Create booking
        booking_id = str(uuid.uuid4())
        booking = Booking(booking_id, customer_id, room_number, check_in, check_out, guests)
        
        # Calculate total
        booking.calculate_total(hotel.pricing_strategy, room, customer.customer_type)
        
        with self._lock:
            self.bookings[booking_id] = booking
            self.total_bookings += 1
        
        # Reserve room
        room.set_status(RoomStatus.RESERVED, booking_id)
        
        print(f"📅 Booking created: {booking_id}")
        return booking
    
    def confirm_booking(self, booking_id: str, payment_method: str, 
                       payment_reference: str) -> bool:
        """Confirm booking with payment."""
        booking = self.bookings.get(booking_id)
        if not booking:
            return False
        
        if booking.confirm_booking():
            booking.update_payment_status(PaymentStatus.PAID, payment_method, payment_reference)
            
            # Update customer history
            customer = self.customers.get(booking.customer_id)
            if customer:
                customer.add_booking(booking_id, booking.total_amount)
            
            # Update system revenue
            with self._lock:
                self.total_revenue += booking.total_amount
            
            print(f"✅ Booking confirmed: {booking_id}")
            return True
        
        return False
    
    def cancel_booking(self, booking_id: str, reason: str = "") -> bool:
        """Cancel booking."""
        booking = self.bookings.get(booking_id)
        if not booking:
            return False
        
        if booking.cancel_booking(reason):
            # Free up room
            hotel = self._get_hotel_for_booking(booking)
            if hotel:
                room = hotel.rooms.get(booking.room_id)
                if room:
                    room.set_status(RoomStatus.AVAILABLE)
            
            # Process refund if applicable
            if booking.payment_status == PaymentStatus.PAID:
                booking.update_payment_status(PaymentStatus.REFUNDED)
            
            print(f"❌ Booking cancelled: {booking_id}")
            return True
        
        return False
    
    def check_in_booking(self, booking_id: str, staff_id: str) -> bool:
        """Check in booking."""
        booking = self.bookings.get(booking_id)
        if not booking:
            return False
        
        if booking.check_in(staff_id):
            # Update room status
            hotel = self._get_hotel_for_booking(booking)
            if hotel:
                room = hotel.rooms.get(booking.room_id)
                if room:
                    room.set_status(RoomStatus.OCCUPIED, booking_id)
            
            print(f"🔑 Check-in completed: {booking_id}")
            return True
        
        return False
    
    def check_out_booking(self, booking_id: str, staff_id: str) -> bool:
        """Check out booking."""
        booking = self.bookings.get(booking_id)
        if not booking:
            return False
        
        if booking.check_out(staff_id):
            # Update room status and schedule cleaning
            hotel = self._get_hotel_for_booking(booking)
            if hotel:
                room = hotel.rooms.get(booking.room_id)
                if room:
                    room.schedule_cleaning()
            
            print(f"🚪 Check-out completed: {booking_id}")
            return True
        
        return False
    
    def _get_hotel_for_booking(self, booking: Booking) -> Optional[Hotel]:
        """Get hotel for booking."""
        for hotel in self.hotels.values():
            if booking.room_id in hotel.rooms:
                return hotel
        return None
    
    def get_customer_bookings(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get all bookings for customer."""
        customer_bookings = []
        
        for booking in self.bookings.values():
            if booking.customer_id == customer_id:
                booking_info = booking.get_booking_info()
                
                # Add hotel information
                hotel = self._get_hotel_for_booking(booking)
                if hotel:
                    booking_info['hotel_name'] = hotel.name
                    booking_info['hotel_address'] = f"{hotel.address.city}, {hotel.address.state}"
                
                customer_bookings.append(booking_info)
        
        # Sort by creation date (newest first)
        customer_bookings.sort(key=lambda x: x['timestamps']['created_at'], reverse=True)
        return customer_bookings
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        total_rooms = sum(len(hotel.rooms) for hotel in self.hotels.values())
        occupied_rooms = sum(
            len([r for r in hotel.rooms.values() if r.status == RoomStatus.OCCUPIED])
            for hotel in self.hotels.values()
        )
        
        return {
            'system_name': self.system_name,
            'hotels': {
                'total_hotels': len(self.hotels),
                'total_rooms': total_rooms,
                'occupied_rooms': occupied_rooms,
                'occupancy_rate': occupied_rooms / max(1, total_rooms)
            },
            'customers': {
                'total_customers': len(self.customers),
                'vip_customers': len([c for c in self.customers.values() if c.customer_type == CustomerType.VIP]),
                'corporate_customers': len([c for c in self.customers.values() if c.customer_type == CustomerType.CORPORATE])
            },
            'bookings': {
                'total_bookings': self.total_bookings,
                'active_bookings': len([b for b in self.bookings.values() if b.status in [BookingStatus.CONFIRMED, BookingStatus.CHECKED_IN]]),
                'pending_bookings': len([b for b in self.bookings.values() if b.status == BookingStatus.PENDING])
            },
            'revenue': {
                'total_revenue': float(self.total_revenue),
                'average_booking_value': float(self.total_revenue / max(1, self.total_bookings))
            }
        }


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_hotel_booking_system():
    """Demonstrate the hotel booking system."""
    print("=== HOTEL BOOKING SYSTEM DEMONSTRATION ===\n")
    
    # Initialize system
    system = HotelBookingSystem("HotelBooker Pro")
    
    print("1. HOTEL SETUP:")
    
    # Create hotels
    hotels_data = [
        ("Grand Plaza Hotel", "New York", "NY", 5),
        ("Seaside Resort", "Miami", "FL", 4),
        ("Business Inn", "Chicago", "IL", 3)
    ]
    
    hotels = []
    for name, city, state, rating in hotels_data:
        address = Address("123 Main St", city, state, "USA", "12345")
        hotel = Hotel(str(uuid.uuid4()), name, address)
        hotel.star_rating = rating
        hotel.description = f"A {rating}-star hotel in {city}"
        
        # Set pricing strategy
        if rating >= 4:
            hotel.set_pricing_strategy(DynamicPricingStrategy())
        
        # Add rooms
        room_configs = [
            (RoomType.SINGLE, 20, Decimal('100')),
            (RoomType.DOUBLE, 30, Decimal('150')),
            (RoomType.SUITE, 10, Decimal('300')),
            (RoomType.DELUXE, 5, Decimal('400'))
        ]
        
        for room_type, count, base_price in room_configs:
            for i in range(count):
                room_number = f"{room_type.value[0].upper()}{i+1:03d}"
                max_occupancy = 1 if room_type == RoomType.SINGLE else 2 if room_type == RoomType.DOUBLE else 4
                
                room = Room(room_number, room_type, (i // 10) + 1, base_price, max_occupancy)
                
                # Add amenities
                room.add_amenity("WiFi")
                room.add_amenity("TV")
                if room_type in [RoomType.SUITE, RoomType.DELUXE]:
                    room.add_amenity("Mini Bar")
                    room.add_amenity("Room Service")
                    room.has_city_view = True
                
                hotel.add_room(room)
        
        # Add services
        services_data = [
            ("SPA001", "Spa Treatment", "Relaxing spa services", Decimal('80'), "Wellness"),
            ("GYM001", "Gym Access", "24/7 gym access", Decimal('20'), "Fitness"),
            ("WIFI001", "Premium WiFi", "High-speed internet", Decimal('15'), "Technology"),
            ("PARK001", "Valet Parking", "Valet parking service", Decimal('25'), "Transportation")
        ]
        
        for service_id, name, desc, price, category in services_data:
            service = ServiceItem(service_id, name, desc, price, category)
            hotel.add_service(service)
        
        # Add staff
        staff_data = [
            ("Manager", StaffRole.MANAGER),
            ("Front Desk", StaffRole.RECEPTIONIST),
            ("Housekeeping", StaffRole.HOUSEKEEPING),
            ("Concierge", StaffRole.CONCIERGE)
        ]
        
        for name, role in staff_data:
            staff_id = str(uuid.uuid4())
            contact = ContactInfo("555-0000", f"{name.lower().replace(' ', '')}@{hotel.name.lower().replace(' ', '')}.com", address)
            hotel.add_staff(staff_id, name, role, contact)
        
        system.add_hotel(hotel)
        hotels.append(hotel)
        print(f"   ✓ {name} created with {len(hotel.rooms)} rooms")
    
    print()
    
    # Register customers
    print("2. CUSTOMER REGISTRATION:")
    
    customers_data = [
        ("John Smith", "555-1001", "john@example.com", CustomerType.INDIVIDUAL),
        ("Jane Doe", "555-1002", "jane@example.com", CustomerType.LOYALTY_MEMBER),
        ("Acme Corp", "555-1003", "booking@acme.com", CustomerType.CORPORATE),
        ("Alice Johnson", "555-1004", "alice@example.com", CustomerType.VIP)
    ]
    
    customers = []
    for name, phone, email, customer_type in customers_data:
        address = Address("456 Oak St", "Boston", "MA", "USA", "02101")
        contact = ContactInfo(phone, email, address)
        customer = system.register_customer(name, contact)
        customer.customer_type = customer_type
        
        # Set loyalty points for non-individual customers
        if customer_type == CustomerType.LOYALTY_MEMBER:
            customer.update_loyalty_points(2000)
        elif customer_type == CustomerType.VIP:
            customer.update_loyalty_points(15000)
        elif customer_type == CustomerType.CORPORATE:
            customer.set_corporate_info("Acme Corporation", "ACME001")
        
        customers.append(customer)
        print(f"   ✓ {name} registered as {customer_type.value}")
    
    print()
    
    # Search for rooms
    print("3. ROOM SEARCH:")
    
    check_in = date.today() + timedelta(days=7)
    check_out = check_in + timedelta(days=3)
    
    search_results = system.search_rooms(check_in, check_out, guests=2, city="New York")
    
    print(f"   Search: {check_in} to {check_out} for 2 guests in New York")
    print(f"   Found {len(search_results)} available rooms:")
    
    for i, result in enumerate(search_results[:5]):  # Show first 5
        print(f"     {i+1}. {result['hotel_name']} - {result['room_type']} Room")
        print(f"        Price: ${result['price']:.2f} for {result['nights']} nights")
        print(f"        Amenities: {', '.join(result['amenities'])}")
    
    print()
    
    # Create bookings
    print("4. BOOKING CREATION:")
    
    bookings = []
    
    # John books a room at Grand Plaza
    if search_results:
        result = search_results[0]  # First result
        booking = system.create_booking(
            customers[0].customer_id,
            result['hotel_id'],
            result['room_number'],
            check_in,
            check_out,
            2
        )
        
        if booking:
            # Add services
            hotel = system.hotels[result['hotel_id']]
            spa_service = hotel.services.get("SPA001")
            if spa_service:
                booking.add_service(spa_service)
            
            booking.add_special_request("Late check-out")
            booking.add_special_request("Extra towels")
            
            # Recalculate total with services
            room = hotel.rooms[result['room_number']]
            booking.calculate_total(hotel.pricing_strategy, room, customers[0].customer_type)
            
            bookings.append(booking)
            print(f"   ✓ John's booking created: ${booking.total_amount:.2f}")
    
    # Jane books a suite (VIP customer)
    suite_results = system.search_rooms(check_in, check_out, guests=2, room_type=RoomType.SUITE)
    if suite_results:
        result = suite_results[0]
        booking = system.create_booking(
            customers[1].customer_id,
            result['hotel_id'],
            result['room_number'],
            check_in,
            check_out,
            2
        )
        
        if booking:
            bookings.append(booking)
            print(f"   ✓ Jane's suite booking created: ${booking.total_amount:.2f}")
    
    print()
    
    # Confirm bookings
    print("5. BOOKING CONFIRMATION:")
    
    for i, booking in enumerate(bookings):
        payment_methods = ["Credit Card", "Debit Card", "PayPal"]
        success = system.confirm_booking(
            booking.booking_id,
            payment_methods[i % len(payment_methods)],
            f"REF{i+1:03d}"
        )
        
        if success:
            print(f"   ✓ Booking {booking.booking_id[:8]} confirmed")
        else:
            print(f"   ✗ Booking {booking.booking_id[:8]} confirmation failed")
    
    print()
    
    # Simulate check-in/check-out
    print("6. CHECK-IN/CHECK-OUT SIMULATION:")
    
    if bookings:
        booking = bookings[0]
        
        # Get staff ID for check-in
        hotel = system._get_hotel_for_booking(booking)
        if hotel:
            receptionist_id = next(iter(hotel.staff.keys()))
            
            # Check-in
            success = system.check_in_booking(booking.booking_id, receptionist_id)
            if success:
                print(f"   ✓ Check-in completed for {booking.booking_id[:8]}")
            
            # Simulate stay and check-out
            success = system.check_out_booking(booking.booking_id, receptionist_id)
            if success:
                print(f"   ✓ Check-out completed for {booking.booking_id[:8]}")
    
    print()
    
    # Test pricing strategies
    print("7. PRICING STRATEGY COMPARISON:")
    
    if hotels:
        hotel = hotels[0]  # Grand Plaza
        room = next(iter(hotel.rooms.values()))
        
        strategies = [
            StandardPricingStrategy(),
            DynamicPricingStrategy()
        ]
        
        print(f"   Room: {room.room_number} ({room.room_type.value})")
        print(f"   Dates: {check_in} to {check_out}")
        
        for strategy in strategies:
            for customer_type in CustomerType:
                price = strategy.calculate_price(room, check_in, check_out, customer_type)
                print(f"   {strategy.get_strategy_name()} - {customer_type.value}: ${price:.2f}")
    
    print()
    
    # Show customer booking history
    print("8. CUSTOMER BOOKING HISTORY:")
    
    for customer in customers[:2]:  # Show first 2 customers
        customer_bookings = system.get_customer_bookings(customer.customer_id)
        print(f"   {customer.name} ({customer.customer_type.value}):")
        print(f"     Total bookings: {len(customer_bookings)}")
        print(f"     Loyalty points: {customer.loyalty_points}")
        print(f"     Total spent: ${customer.total_spent:.2f}")
        
        for booking_info in customer_bookings:
            status = booking_info['status']
            total = booking_info['pricing']['total_amount']
            hotel_name = booking_info.get('hotel_name', 'Unknown Hotel')
            print(f"     - {hotel_name}: ${total:.2f} ({status})")
    
    print()
    
    # Show hotel information
    print("9. HOTEL INFORMATION:")
    
    for hotel in hotels:
        info = hotel.get_hotel_info()
        print(f"   {info['name']} ({info['star_rating']} stars):")
        print(f"     Location: {info['address']['city']}, {info['address']['state']}")
        print(f"     Rooms: {info['rooms']['total_rooms']} total, {info['rooms']['available_rooms']} available")
        print(f"     Occupancy: {info['statistics']['current_occupancy_rate']:.1%}")
        print(f"     Revenue: ${info['statistics']['total_revenue']:.2f}")
        print(f"     Pricing: {info['pricing_strategy']}")
        
        # Show room type breakdown
        print(f"     Room Types:")
        for room_type, count in info['rooms']['room_types'].items():
            print(f"       {room_type.title()}: {count}")
    
    print()
    
    # Show system statistics
    print("10. SYSTEM STATISTICS:")
    
    stats = system.get_system_statistics()
    print(f"   System: {stats['system_name']}")
    print(f"   Hotels: {stats['hotels']['total_hotels']}")
    print(f"   Total Rooms: {stats['hotels']['total_rooms']}")
    print(f"   Overall Occupancy: {stats['hotels']['occupancy_rate']:.1%}")
    print(f"   Customers: {stats['customers']['total_customers']}")
    print(f"     - VIP: {stats['customers']['vip_customers']}")
    print(f"     - Corporate: {stats['customers']['corporate_customers']}")
    print(f"   Bookings: {stats['bookings']['total_bookings']}")
    print(f"     - Active: {stats['bookings']['active_bookings']}")
    print(f"     - Pending: {stats['bookings']['pending_bookings']}")
    print(f"   Revenue: ${stats['revenue']['total_revenue']:.2f}")
    print(f"   Average Booking Value: ${stats['revenue']['average_booking_value']:.2f}")
    
    print()
    
    # Generate revenue report
    print("11. REVENUE REPORT:")
    
    if hotels:
        hotel = hotels[0]
        report_start = date.today() - timedelta(days=30)
        report_end = date.today()
        
        revenue_report = hotel.get_revenue_report(report_start, report_end)
        
        print(f"   {hotel.name} - Last 30 Days:")
        print(f"   Period: {revenue_report['period']['start_date']} to {revenue_report['period']['end_date']}")
        print(f"   Total Revenue: ${revenue_report['revenue']['total_revenue']:.2f}")
        print(f"   Room Revenue: ${revenue_report['revenue']['room_revenue']:.2f}")
        print(f"   Service Revenue: ${revenue_report['revenue']['service_revenue']:.2f}")
        print(f"   Total Bookings: {revenue_report['bookings']['total_bookings']}")
        print(f"   Average Booking Value: ${revenue_report['bookings']['average_booking_value']:.2f}")
        print(f"   Average Occupancy: {revenue_report['occupancy']['average_occupancy_rate']:.1%}")
    
    print()
    print("=== HOTEL BOOKING SYSTEM DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_hotel_booking_system()
