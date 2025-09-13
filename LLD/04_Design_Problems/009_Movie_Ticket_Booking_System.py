"""
MOVIE TICKET BOOKING SYSTEM - Complete System Design
====================================================

Problem Statement:
Design a comprehensive movie ticket booking system that handles:
- Movie catalog and show scheduling
- Theater and screen management
- Seat selection and booking
- Pricing strategies and discounts
- Payment processing
- User management and profiles
- Booking confirmation and tickets
- Cancellation and refunds
- Real-time seat availability
- Multi-location theater chains

Requirements:
- Support multiple theaters and screens
- Handle different movie formats (2D, 3D, IMAX)
- Implement dynamic pricing based on time and demand
- Provide real-time seat availability
- Support group bookings and corporate accounts
- Handle various payment methods
- Generate digital and physical tickets
- Implement loyalty programs and discounts
- Support advance booking and walk-ins
- Provide comprehensive reporting

Design Patterns Used:
- Factory: Movie and booking creation
- Strategy: Pricing strategies
- Observer: Booking notifications
- State: Booking and seat states
- Command: Booking operations
- Decorator: Premium features
- Singleton: Theater management system
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, date, time, timedelta
from enum import Enum
import uuid
import threading
from dataclasses import dataclass, field
from decimal import Decimal
import json
import qrcode
from io import BytesIO
import base64


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class MovieGenre(Enum):
    ACTION = "action"
    COMEDY = "comedy"
    DRAMA = "drama"
    HORROR = "horror"
    ROMANCE = "romance"
    THRILLER = "thriller"
    SCI_FI = "sci_fi"
    FANTASY = "fantasy"
    DOCUMENTARY = "documentary"
    ANIMATION = "animation"


class MovieRating(Enum):
    G = "g"
    PG = "pg"
    PG_13 = "pg_13"
    R = "r"
    NC_17 = "nc_17"


class ShowFormat(Enum):
    STANDARD_2D = "standard_2d"
    DIGITAL_3D = "digital_3d"
    IMAX_2D = "imax_2d"
    IMAX_3D = "imax_3d"
    DOLBY_ATMOS = "dolby_atmos"
    VIP = "vip"


class SeatType(Enum):
    REGULAR = "regular"
    PREMIUM = "premium"
    VIP = "vip"
    WHEELCHAIR = "wheelchair"
    COUPLE = "couple"


class SeatStatus(Enum):
    AVAILABLE = "available"
    SELECTED = "selected"
    BOOKED = "booked"
    BLOCKED = "blocked"
    MAINTENANCE = "maintenance"


class BookingStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    EXPIRED = "expired"


class PaymentMethod(Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    DIGITAL_WALLET = "digital_wallet"
    CASH = "cash"
    GIFT_CARD = "gift_card"


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


# ============================================================================
# PRICING STRATEGIES
# ============================================================================

class PricingStrategy(ABC):
    """Abstract pricing strategy."""
    
    @abstractmethod
    def calculate_price(self, base_price: Decimal, show_time: datetime, 
                       seat_type: SeatType, show_format: ShowFormat,
                       customer_type: str = "regular") -> Decimal:
        """Calculate ticket price."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class StandardPricingStrategy(PricingStrategy):
    """Standard pricing strategy."""
    
    def calculate_price(self, base_price: Decimal, show_time: datetime, 
                       seat_type: SeatType, show_format: ShowFormat,
                       customer_type: str = "regular") -> Decimal:
        """Calculate standard price."""
        price = base_price
        
        # Seat type multiplier
        seat_multipliers = {
            SeatType.REGULAR: Decimal('1.0'),
            SeatType.PREMIUM: Decimal('1.3'),
            SeatType.VIP: Decimal('1.8'),
            SeatType.WHEELCHAIR: Decimal('1.0'),
            SeatType.COUPLE: Decimal('1.5')
        }
        price *= seat_multipliers.get(seat_type, Decimal('1.0'))
        
        # Format multiplier
        format_multipliers = {
            ShowFormat.STANDARD_2D: Decimal('1.0'),
            ShowFormat.DIGITAL_3D: Decimal('1.4'),
            ShowFormat.IMAX_2D: Decimal('1.6'),
            ShowFormat.IMAX_3D: Decimal('2.0'),
            ShowFormat.DOLBY_ATMOS: Decimal('1.3'),
            ShowFormat.VIP: Decimal('2.5')
        }
        price *= format_multipliers.get(show_format, Decimal('1.0'))
        
        # Customer type discount
        customer_discounts = {
            "regular": Decimal('1.0'),
            "student": Decimal('0.8'),
            "senior": Decimal('0.7'),
            "child": Decimal('0.6'),
            "military": Decimal('0.75')
        }
        price *= customer_discounts.get(customer_type, Decimal('1.0'))
        
        return price
    
    def get_strategy_name(self) -> str:
        return "Standard Pricing"


class DynamicPricingStrategy(PricingStrategy):
    """Dynamic pricing based on time and demand."""
    
    def __init__(self):
        self.peak_hours = [(time(18, 0), time(22, 0))]  # Evening shows
        self.weekend_multiplier = Decimal('1.2')
        self.peak_hour_multiplier = Decimal('1.3')
        self.advance_booking_discount = Decimal('0.9')  # 10% off for advance booking
    
    def calculate_price(self, base_price: Decimal, show_time: datetime, 
                       seat_type: SeatType, show_format: ShowFormat,
                       customer_type: str = "regular") -> Decimal:
        """Calculate dynamic price."""
        # Start with standard pricing
        standard_strategy = StandardPricingStrategy()
        price = standard_strategy.calculate_price(
            base_price, show_time, seat_type, show_format, customer_type
        )
        
        # Weekend pricing
        if show_time.weekday() >= 5:  # Saturday, Sunday
            price *= self.weekend_multiplier
        
        # Peak hour pricing
        show_time_only = show_time.time()
        for start_time, end_time in self.peak_hours:
            if start_time <= show_time_only <= end_time:
                price *= self.peak_hour_multiplier
                break
        
        # Advance booking discount (more than 7 days in advance)
        days_in_advance = (show_time.date() - date.today()).days
        if days_in_advance > 7:
            price *= self.advance_booking_discount
        
        return price
    
    def get_strategy_name(self) -> str:
        return "Dynamic Pricing"


# ============================================================================
# MOVIE CLASSES
# ============================================================================

class Movie:
    """Movie with details and showtimes."""
    
    def __init__(self, movie_id: str, title: str, description: str, 
                 duration_minutes: int, rating: MovieRating):
        self.movie_id = movie_id
        self.title = title
        self.description = description
        self.duration_minutes = duration_minutes
        self.rating = rating
        
        # Movie details
        self.genres: Set[MovieGenre] = set()
        self.director = ""
        self.cast: List[str] = []
        self.language = "English"
        self.subtitles: List[str] = []
        
        # Release information
        self.release_date: Optional[date] = None
        self.country = ""
        
        # Media
        self.poster_url = ""
        self.trailer_url = ""
        self.images: List[str] = []
        
        # Ratings and reviews
        self.imdb_rating = 0.0
        self.user_rating = 0.0
        self.user_rating_count = 0
        
        # Availability
        self.is_active = True
        self.is_coming_soon = False
        
        self.created_at = datetime.now()
    
    def add_genre(self, genre: MovieGenre) -> None:
        """Add genre to movie."""
        self.genres.add(genre)
    
    def add_cast_member(self, actor: str) -> None:
        """Add cast member."""
        if actor not in self.cast:
            self.cast.append(actor)
    
    def add_subtitle_language(self, language: str) -> None:
        """Add subtitle language."""
        if language not in self.subtitles:
            self.subtitles.append(language)
    
    def add_user_rating(self, rating: float) -> None:
        """Add user rating (1-10)."""
        if 1 <= rating <= 10:
            total_rating = self.user_rating * self.user_rating_count + rating
            self.user_rating_count += 1
            self.user_rating = total_rating / self.user_rating_count
    
    def get_movie_info(self) -> Dict[str, Any]:
        """Get movie information."""
        return {
            'movie_id': self.movie_id,
            'title': self.title,
            'description': self.description,
            'duration_minutes': self.duration_minutes,
            'rating': self.rating.value,
            'genres': [genre.value for genre in self.genres],
            'director': self.director,
            'cast': self.cast,
            'language': self.language,
            'subtitles': self.subtitles,
            'release_date': self.release_date.isoformat() if self.release_date else None,
            'country': self.country,
            'poster_url': self.poster_url,
            'trailer_url': self.trailer_url,
            'images': self.images,
            'ratings': {
                'imdb_rating': self.imdb_rating,
                'user_rating': self.user_rating,
                'user_rating_count': self.user_rating_count
            },
            'is_active': self.is_active,
            'is_coming_soon': self.is_coming_soon,
            'created_at': self.created_at.isoformat()
        }
    
    def __str__(self) -> str:
        return f"{self.title} ({self.duration_minutes}min) - {self.rating.value.upper()}"


# ============================================================================
# THEATER CLASSES
# ============================================================================

class Seat:
    """Individual seat in a screen."""
    
    def __init__(self, seat_id: str, row: str, number: int, seat_type: SeatType):
        self.seat_id = seat_id
        self.row = row
        self.number = number
        self.seat_type = seat_type
        self.status = SeatStatus.AVAILABLE
        
        # Booking information
        self.current_booking_id: Optional[str] = None
        self.hold_expiry: Optional[datetime] = None
        
        self._lock = threading.Lock()
    
    def is_available(self) -> bool:
        """Check if seat is available."""
        with self._lock:
            if self.status in [SeatStatus.BOOKED, SeatStatus.BLOCKED, SeatStatus.MAINTENANCE]:
                return False
            
            # Check if hold has expired
            if self.status == SeatStatus.SELECTED and self.hold_expiry:
                if datetime.now() > self.hold_expiry:
                    self.status = SeatStatus.AVAILABLE
                    self.current_booking_id = None
                    self.hold_expiry = None
            
            return self.status == SeatStatus.AVAILABLE
    
    def hold_seat(self, booking_id: str, hold_duration_minutes: int = 10) -> bool:
        """Hold seat for booking."""
        with self._lock:
            if not self.is_available():
                return False
            
            self.status = SeatStatus.SELECTED
            self.current_booking_id = booking_id
            self.hold_expiry = datetime.now() + timedelta(minutes=hold_duration_minutes)
            return True
    
    def book_seat(self, booking_id: str) -> bool:
        """Book the seat."""
        with self._lock:
            if (self.status == SeatStatus.SELECTED and 
                self.current_booking_id == booking_id):
                self.status = SeatStatus.BOOKED
                self.hold_expiry = None
                return True
            return False
    
    def release_seat(self) -> bool:
        """Release the seat."""
        with self._lock:
            if self.status in [SeatStatus.SELECTED, SeatStatus.BOOKED]:
                self.status = SeatStatus.AVAILABLE
                self.current_booking_id = None
                self.hold_expiry = None
                return True
            return False
    
    def block_seat(self) -> None:
        """Block seat (maintenance, etc.)."""
        with self._lock:
            if self.status == SeatStatus.AVAILABLE:
                self.status = SeatStatus.BLOCKED
    
    def unblock_seat(self) -> None:
        """Unblock seat."""
        with self._lock:
            if self.status == SeatStatus.BLOCKED:
                self.status = SeatStatus.AVAILABLE
    
    def get_seat_info(self) -> Dict[str, Any]:
        """Get seat information."""
        return {
            'seat_id': self.seat_id,
            'row': self.row,
            'number': self.number,
            'seat_type': self.seat_type.value,
            'status': self.status.value,
            'is_available': self.is_available(),
            'current_booking_id': self.current_booking_id,
            'hold_expiry': self.hold_expiry.isoformat() if self.hold_expiry else None
        }
    
    def __str__(self) -> str:
        return f"Seat {self.row}{self.number} ({self.seat_type.value}) - {self.status.value}"


class Screen:
    """Movie screen with seats and shows."""
    
    def __init__(self, screen_id: str, name: str, capacity: int):
        self.screen_id = screen_id
        self.name = name
        self.capacity = capacity
        
        # Seats layout
        self.seats: Dict[str, Seat] = {}  # seat_id -> Seat
        self.rows: Dict[str, List[Seat]] = {}  # row -> [seats]
        
        # Screen capabilities
        self.supported_formats: Set[ShowFormat] = {ShowFormat.STANDARD_2D}
        
        # Technical specifications
        self.screen_size = ""
        self.sound_system = ""
        self.projection_type = ""
        
        # Statistics
        self.total_shows = 0
        self.total_bookings = 0
        
        self._lock = threading.Lock()
    
    def add_seat(self, seat: Seat) -> None:
        """Add seat to screen."""
        with self._lock:
            self.seats[seat.seat_id] = seat
            
            if seat.row not in self.rows:
                self.rows[seat.row] = []
            
            self.rows[seat.row].append(seat)
            # Sort seats by number
            self.rows[seat.row].sort(key=lambda s: s.number)
    
    def add_supported_format(self, format_type: ShowFormat) -> None:
        """Add supported show format."""
        self.supported_formats.add(format_type)
    
    def get_available_seats(self) -> List[Seat]:
        """Get all available seats."""
        return [seat for seat in self.seats.values() if seat.is_available()]
    
    def get_seats_by_type(self, seat_type: SeatType) -> List[Seat]:
        """Get seats by type."""
        return [seat for seat in self.seats.values() if seat.seat_type == seat_type]
    
    def get_seat_layout(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get seat layout by rows."""
        layout = {}
        for row, seats in self.rows.items():
            layout[row] = [seat.get_seat_info() for seat in seats]
        return layout
    
    def get_occupancy_rate(self) -> float:
        """Get current occupancy rate."""
        total_seats = len(self.seats)
        if total_seats == 0:
            return 0.0
        
        occupied_seats = len([s for s in self.seats.values() 
                            if s.status in [SeatStatus.BOOKED, SeatStatus.SELECTED]])
        
        return occupied_seats / total_seats
    
    def get_screen_info(self) -> Dict[str, Any]:
        """Get screen information."""
        available_seats = len(self.get_available_seats())
        
        return {
            'screen_id': self.screen_id,
            'name': self.name,
            'capacity': self.capacity,
            'available_seats': available_seats,
            'occupancy_rate': self.get_occupancy_rate(),
            'supported_formats': [fmt.value for fmt in self.supported_formats],
            'technical_specs': {
                'screen_size': self.screen_size,
                'sound_system': self.sound_system,
                'projection_type': self.projection_type
            },
            'statistics': {
                'total_shows': self.total_shows,
                'total_bookings': self.total_bookings
            },
            'seat_types': {
                seat_type.value: len(self.get_seats_by_type(seat_type))
                for seat_type in SeatType
            }
        }
    
    def __str__(self) -> str:
        return f"Screen {self.name} (Capacity: {self.capacity})"


class Show:
    """Movie show with timing and pricing."""
    
    def __init__(self, show_id: str, movie: Movie, screen: Screen, 
                 show_time: datetime, show_format: ShowFormat):
        self.show_id = show_id
        self.movie = movie
        self.screen = screen
        self.show_time = show_time
        self.show_format = show_format
        
        # Pricing
        self.base_price = Decimal('10.00')  # Default base price
        
        # Show status
        self.is_active = True
        self.is_sold_out = False
        
        # Booking information
        self.total_bookings = 0
        self.total_revenue = Decimal('0')
        
        # Calculate end time
        self.end_time = show_time + timedelta(minutes=movie.duration_minutes + 30)  # +30 for ads/cleanup
        
        self.created_at = datetime.now()
    
    def get_available_seats(self) -> List[Seat]:
        """Get available seats for this show."""
        return self.screen.get_available_seats()
    
    def get_available_seat_count(self) -> int:
        """Get count of available seats."""
        return len(self.get_available_seats())
    
    def is_booking_allowed(self) -> bool:
        """Check if booking is allowed for this show."""
        # Don't allow booking if show has already started
        if datetime.now() >= self.show_time:
            return False
        
        # Don't allow booking if show is inactive or sold out
        if not self.is_active or self.is_sold_out:
            return False
        
        return True
    
    def update_sold_out_status(self) -> None:
        """Update sold out status based on available seats."""
        available_seats = self.get_available_seat_count()
        self.is_sold_out = available_seats == 0
    
    def add_booking(self, booking_amount: Decimal) -> None:
        """Add booking statistics."""
        self.total_bookings += 1
        self.total_revenue += booking_amount
        self.update_sold_out_status()
    
    def get_show_info(self) -> Dict[str, Any]:
        """Get show information."""
        return {
            'show_id': self.show_id,
            'movie': self.movie.get_movie_info(),
            'screen': {
                'screen_id': self.screen.screen_id,
                'name': self.screen.name,
                'capacity': self.screen.capacity
            },
            'show_time': self.show_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'show_format': self.show_format.value,
            'base_price': float(self.base_price),
            'is_active': self.is_active,
            'is_sold_out': self.is_sold_out,
            'is_booking_allowed': self.is_booking_allowed(),
            'available_seats': self.get_available_seat_count(),
            'statistics': {
                'total_bookings': self.total_bookings,
                'total_revenue': float(self.total_revenue),
                'occupancy_rate': self.screen.get_occupancy_rate()
            },
            'created_at': self.created_at.isoformat()
        }
    
    def __str__(self) -> str:
        return f"{self.movie.title} - {self.show_time.strftime('%Y-%m-%d %H:%M')} ({self.show_format.value})"


# ============================================================================
# BOOKING CLASSES
# ============================================================================

class Booking:
    """Movie ticket booking."""
    
    def __init__(self, booking_id: str, show: Show, customer_name: str, 
                 customer_email: str, customer_phone: str):
        self.booking_id = booking_id
        self.show = show
        self.customer_name = customer_name
        self.customer_email = customer_email
        self.customer_phone = customer_phone
        self.status = BookingStatus.PENDING
        
        # Booked seats
        self.booked_seats: List[Seat] = []
        
        # Pricing
        self.seat_prices: Dict[str, Decimal] = {}  # seat_id -> price
        self.total_amount = Decimal('0')
        self.discount_amount = Decimal('0')
        self.tax_amount = Decimal('0')
        self.convenience_fee = Decimal('2.00')  # Booking fee
        
        # Payment information
        self.payment_method: Optional[PaymentMethod] = None
        self.payment_reference = ""
        
        # Timestamps
        self.created_at = datetime.now()
        self.confirmed_at: Optional[datetime] = None
        self.cancelled_at: Optional[datetime] = None
        
        # Booking expiry (for payment)
        self.expires_at = datetime.now() + timedelta(minutes=15)
        
        # Tickets
        self.ticket_codes: List[str] = []
        self.qr_codes: List[str] = []
    
    def add_seat(self, seat: Seat, price: Decimal) -> bool:
        """Add seat to booking."""
        if seat.hold_seat(self.booking_id):
            self.booked_seats.append(seat)
            self.seat_prices[seat.seat_id] = price
            self._calculate_total()
            return True
        return False
    
    def remove_seat(self, seat_id: str) -> bool:
        """Remove seat from booking."""
        for i, seat in enumerate(self.booked_seats):
            if seat.seat_id == seat_id:
                seat.release_seat()
                del self.booked_seats[i]
                del self.seat_prices[seat_id]
                self._calculate_total()
                return True
        return False
    
    def apply_discount(self, discount_amount: Decimal) -> None:
        """Apply discount to booking."""
        self.discount_amount = discount_amount
        self._calculate_total()
    
    def _calculate_total(self) -> None:
        """Calculate total booking amount."""
        subtotal = sum(self.seat_prices.values())
        
        # Apply discount
        discounted_subtotal = subtotal - self.discount_amount
        
        # Calculate tax (8% on discounted subtotal)
        self.tax_amount = discounted_subtotal * Decimal('0.08')
        
        # Calculate total
        self.total_amount = discounted_subtotal + self.tax_amount + self.convenience_fee
    
    def confirm_booking(self, payment_method: PaymentMethod, 
                       payment_reference: str) -> bool:
        """Confirm booking with payment."""
        if self.status != BookingStatus.PENDING:
            return False
        
        if datetime.now() > self.expires_at:
            self.cancel_booking("Booking expired")
            return False
        
        # Confirm seat bookings
        for seat in self.booked_seats:
            if not seat.book_seat(self.booking_id):
                # Rollback if any seat booking fails
                self.cancel_booking("Seat booking failed")
                return False
        
        # Generate tickets
        self._generate_tickets()
        
        # Update booking status
        self.status = BookingStatus.CONFIRMED
        self.confirmed_at = datetime.now()
        self.payment_method = payment_method
        self.payment_reference = payment_reference
        
        # Update show statistics
        self.show.add_booking(self.total_amount)
        
        return True
    
    def cancel_booking(self, reason: str = "") -> bool:
        """Cancel booking."""
        if self.status == BookingStatus.CANCELLED:
            return False
        
        # Release all seats
        for seat in self.booked_seats:
            seat.release_seat()
        
        self.status = BookingStatus.CANCELLED
        self.cancelled_at = datetime.now()
        
        return True
    
    def is_expired(self) -> bool:
        """Check if booking is expired."""
        return datetime.now() > self.expires_at
    
    def _generate_tickets(self) -> None:
        """Generate ticket codes and QR codes."""
        for seat in self.booked_seats:
            # Generate ticket code
            ticket_code = f"TKT{self.booking_id[:8]}{seat.seat_id}"
            self.ticket_codes.append(ticket_code)
            
            # Generate QR code data
            qr_data = {
                'booking_id': self.booking_id,
                'show_id': self.show.show_id,
                'movie_title': self.show.movie.title,
                'show_time': self.show.show_time.isoformat(),
                'screen': self.show.screen.name,
                'seat': f"{seat.row}{seat.number}",
                'ticket_code': ticket_code
            }
            
            # Create QR code (simplified as base64 encoded JSON)
            qr_code = base64.b64encode(json.dumps(qr_data).encode()).decode()
            self.qr_codes.append(qr_code)
    
    def get_booking_info(self) -> Dict[str, Any]:
        """Get booking information."""
        return {
            'booking_id': self.booking_id,
            'show': self.show.get_show_info(),
            'customer': {
                'name': self.customer_name,
                'email': self.customer_email,
                'phone': self.customer_phone
            },
            'seats': [
                {
                    'seat_info': seat.get_seat_info(),
                    'price': float(self.seat_prices[seat.seat_id])
                }
                for seat in self.booked_seats
            ],
            'pricing': {
                'subtotal': float(sum(self.seat_prices.values())),
                'discount_amount': float(self.discount_amount),
                'tax_amount': float(self.tax_amount),
                'convenience_fee': float(self.convenience_fee),
                'total_amount': float(self.total_amount)
            },
            'status': self.status.value,
            'payment': {
                'method': self.payment_method.value if self.payment_method else None,
                'reference': self.payment_reference
            },
            'tickets': {
                'ticket_codes': self.ticket_codes,
                'qr_codes': self.qr_codes
            },
            'timestamps': {
                'created_at': self.created_at.isoformat(),
                'confirmed_at': self.confirmed_at.isoformat() if self.confirmed_at else None,
                'cancelled_at': self.cancelled_at.isoformat() if self.cancelled_at else None,
                'expires_at': self.expires_at.isoformat()
            }
        }
    
    def __str__(self) -> str:
        return f"Booking {self.booking_id} - {len(self.booked_seats)} seats (${self.total_amount})"


# ============================================================================
# THEATER MANAGEMENT SYSTEM
# ============================================================================

class Theater:
    """Movie theater with screens and shows."""
    
    def __init__(self, theater_id: str, name: str, address: Address):
        self.theater_id = theater_id
        self.name = name
        self.address = address
        
        # Contact information
        self.phone = ""
        self.email = ""
        self.website = ""
        
        # Screens and shows
        self.screens: Dict[str, Screen] = {}
        self.shows: Dict[str, Show] = {}
        self.movies: Dict[str, Movie] = {}
        
        # Bookings
        self.bookings: Dict[str, Booking] = {}
        
        # Pricing strategy
        self.pricing_strategy: PricingStrategy = StandardPricingStrategy()
        
        # Operating hours
        self.opening_time = time(9, 0)
        self.closing_time = time(23, 30)
        
        # Statistics
        self.total_bookings = 0
        self.total_revenue = Decimal('0')
        
        self._lock = threading.Lock()
    
    def add_screen(self, screen: Screen) -> None:
        """Add screen to theater."""
        self.screens[screen.screen_id] = screen
    
    def add_movie(self, movie: Movie) -> None:
        """Add movie to theater."""
        self.movies[movie.movie_id] = movie
    
    def set_pricing_strategy(self, strategy: PricingStrategy) -> None:
        """Set pricing strategy."""
        self.pricing_strategy = strategy
    
    def create_show(self, movie_id: str, screen_id: str, show_time: datetime,
                   show_format: ShowFormat, base_price: Decimal) -> Optional[Show]:
        """Create new show."""
        movie = self.movies.get(movie_id)
        screen = self.screens.get(screen_id)
        
        if not movie or not screen:
            return None
        
        # Check if screen supports the format
        if show_format not in screen.supported_formats:
            return None
        
        # Check for scheduling conflicts
        if self._has_scheduling_conflict(screen_id, show_time, movie.duration_minutes):
            return None
        
        show_id = str(uuid.uuid4())
        show = Show(show_id, movie, screen, show_time, show_format)
        show.base_price = base_price
        
        with self._lock:
            self.shows[show_id] = show
            screen.total_shows += 1
        
        return show
    
    def _has_scheduling_conflict(self, screen_id: str, show_time: datetime, 
                               duration_minutes: int) -> bool:
        """Check for scheduling conflicts."""
        show_end_time = show_time + timedelta(minutes=duration_minutes + 30)
        
        for show in self.shows.values():
            if show.screen.screen_id == screen_id:
                # Check if times overlap
                if (show_time < show.end_time and show_end_time > show.show_time):
                    return True
        
        return False
    
    def get_shows_by_movie(self, movie_id: str, target_date: date = None) -> List[Show]:
        """Get shows for a movie on a specific date."""
        if not target_date:
            target_date = date.today()
        
        shows = []
        for show in self.shows.values():
            if (show.movie.movie_id == movie_id and 
                show.show_time.date() == target_date and
                show.is_active):
                shows.append(show)
        
        # Sort by show time
        shows.sort(key=lambda s: s.show_time)
        return shows
    
    def get_shows_by_date(self, target_date: date) -> List[Show]:
        """Get all shows for a specific date."""
        shows = []
        for show in self.shows.values():
            if show.show_time.date() == target_date and show.is_active:
                shows.append(show)
        
        # Sort by show time
        shows.sort(key=lambda s: s.show_time)
        return shows
    
    def search_shows(self, movie_title: str = None, genre: MovieGenre = None,
                    show_date: date = None, show_format: ShowFormat = None) -> List[Show]:
        """Search shows by criteria."""
        results = []
        
        for show in self.shows.values():
            if not show.is_active:
                continue
            
            # Filter by movie title
            if movie_title and movie_title.lower() not in show.movie.title.lower():
                continue
            
            # Filter by genre
            if genre and genre not in show.movie.genres:
                continue
            
            # Filter by date
            if show_date and show.show_time.date() != show_date:
                continue
            
            # Filter by format
            if show_format and show.show_format != show_format:
                continue
            
            results.append(show)
        
        # Sort by show time
        results.sort(key=lambda s: s.show_time)
        return results
    
    def create_booking(self, show_id: str, customer_name: str, customer_email: str,
                      customer_phone: str, seat_ids: List[str], 
                      customer_type: str = "regular") -> Optional[Booking]:
        """Create new booking."""
        show = self.shows.get(show_id)
        if not show or not show.is_booking_allowed():
            return None
        
        booking_id = str(uuid.uuid4())
        booking = Booking(booking_id, show, customer_name, customer_email, customer_phone)
        
        # Add seats to booking
        for seat_id in seat_ids:
            seat = show.screen.seats.get(seat_id)
            if not seat or not seat.is_available():
                # Cancel booking if any seat is not available
                booking.cancel_booking("Seat not available")
                return None
            
            # Calculate price for this seat
            price = self.pricing_strategy.calculate_price(
                show.base_price, show.show_time, seat.seat_type, 
                show.show_format, customer_type
            )
            
            if not booking.add_seat(seat, price):
                # Cancel booking if seat cannot be added
                booking.cancel_booking("Failed to reserve seat")
                return None
        
        with self._lock:
            self.bookings[booking_id] = booking
        
        return booking
    
    def confirm_booking(self, booking_id: str, payment_method: PaymentMethod,
                       payment_reference: str) -> bool:
        """Confirm booking with payment."""
        booking = self.bookings.get(booking_id)
        if not booking:
            return False
        
        if booking.confirm_booking(payment_method, payment_reference):
            with self._lock:
                self.total_bookings += 1
                self.total_revenue += booking.total_amount
            return True
        
        return False
    
    def cancel_booking(self, booking_id: str, reason: str = "") -> bool:
        """Cancel booking."""
        booking = self.bookings.get(booking_id)
        if not booking:
            return False
        
        return booking.cancel_booking(reason)
    
    def cleanup_expired_bookings(self) -> int:
        """Clean up expired bookings."""
        expired_count = 0
        
        with self._lock:
            for booking in list(self.bookings.values()):
                if booking.is_expired() and booking.status == BookingStatus.PENDING:
                    booking.cancel_booking("Expired")
                    expired_count += 1
        
        return expired_count
    
    def get_theater_info(self) -> Dict[str, Any]:
        """Get theater information."""
        return {
            'theater_id': self.theater_id,
            'name': self.name,
            'address': {
                'street': self.address.street,
                'city': self.address.city,
                'state': self.address.state,
                'country': self.address.country,
                'zip_code': self.address.zip_code
            },
            'contact': {
                'phone': self.phone,
                'email': self.email,
                'website': self.website
            },
            'screens_count': len(self.screens),
            'movies_count': len(self.movies),
            'active_shows': len([s for s in self.shows.values() if s.is_active]),
            'total_capacity': sum(screen.capacity for screen in self.screens.values()),
            'statistics': {
                'total_bookings': self.total_bookings,
                'total_revenue': float(self.total_revenue)
            },
            'pricing_strategy': self.pricing_strategy.get_strategy_name(),
            'operating_hours': {
                'opening_time': self.opening_time.strftime('%H:%M'),
                'closing_time': self.closing_time.strftime('%H:%M')
            }
        }
    
    def get_daily_report(self, target_date: date) -> Dict[str, Any]:
        """Get daily theater report."""
        daily_shows = self.get_shows_by_date(target_date)
        daily_bookings = [
            b for b in self.bookings.values()
            if b.show.show_time.date() == target_date and b.status == BookingStatus.CONFIRMED
        ]
        
        total_tickets = sum(len(b.booked_seats) for b in daily_bookings)
        total_revenue = sum(b.total_amount for b in daily_bookings)
        
        # Calculate occupancy by show
        show_occupancy = {}
        for show in daily_shows:
            total_seats = show.screen.capacity
            booked_seats = len([
                seat for booking in daily_bookings
                if booking.show.show_id == show.show_id
                for seat in booking.booked_seats
            ])
            occupancy_rate = booked_seats / total_seats if total_seats > 0 else 0
            show_occupancy[show.show_id] = {
                'movie_title': show.movie.title,
                'show_time': show.show_time.strftime('%H:%M'),
                'screen': show.screen.name,
                'occupancy_rate': occupancy_rate,
                'tickets_sold': booked_seats
            }
        
        return {
            'date': target_date.isoformat(),
            'shows_count': len(daily_shows),
            'bookings_count': len(daily_bookings),
            'tickets_sold': total_tickets,
            'revenue': float(total_revenue),
            'average_ticket_price': float(total_revenue / max(1, total_tickets)),
            'show_occupancy': show_occupancy
        }


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_movie_booking_system():
    """Demonstrate the movie ticket booking system."""
    print("=== MOVIE TICKET BOOKING SYSTEM DEMONSTRATION ===\n")
    
    # Initialize theater
    address = Address("123 Cinema Blvd", "Hollywood", "CA", "USA", "90210")
    theater = Theater("THR001", "Cineplex Downtown", address)
    theater.phone = "555-MOVIE"
    theater.email = "info@cineplex.com"
    
    print("1. THEATER SETUP:")
    
    # Create screens
    screens_data = [
        ("SCR001", "Screen 1", 150, [ShowFormat.STANDARD_2D, ShowFormat.DIGITAL_3D]),
        ("SCR002", "Screen 2", 200, [ShowFormat.STANDARD_2D, ShowFormat.IMAX_2D, ShowFormat.IMAX_3D]),
        ("SCR003", "Screen 3", 100, [ShowFormat.STANDARD_2D, ShowFormat.VIP])
    ]
    
    for screen_id, name, capacity, formats in screens_data:
        screen = Screen(screen_id, name, capacity)
        
        # Add supported formats
        for fmt in formats:
            screen.add_supported_format(fmt)
        
        # Create seat layout
        rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        seats_per_row = capacity // len(rows)
        
        for i, row in enumerate(rows):
            for seat_num in range(1, seats_per_row + 1):
                seat_id = f"{screen_id}_{row}{seat_num:02d}"
                
                # Determine seat type
                if row in ['A', 'B']:  # Front rows
                    seat_type = SeatType.REGULAR
                elif row in ['H', 'I', 'J']:  # Back rows
                    seat_type = SeatType.PREMIUM
                elif seat_num in [1, seats_per_row]:  # Aisle seats
                    seat_type = SeatType.WHEELCHAIR if row == 'E' else SeatType.REGULAR
                else:
                    seat_type = SeatType.REGULAR
                
                # VIP screen has all VIP seats
                if "Screen 3" in name:
                    seat_type = SeatType.VIP
                
                seat = Seat(seat_id, row, seat_num, seat_type)
                screen.add_seat(seat)
        
        theater.add_screen(screen)
        print(f"   ✓ {name} created with {capacity} seats")
    
    print()
    
    # Create movies
    print("2. MOVIE CATALOG SETUP:")
    
    movies_data = [
        ("Superhero Adventure", "Epic superhero movie with amazing action", 150, MovieRating.PG_13, 
         [MovieGenre.ACTION, MovieGenre.SCI_FI], "John Director", ["Hero Actor", "Villain Actor"]),
        ("Romantic Comedy", "Heartwarming romantic comedy", 120, MovieRating.PG,
         [MovieGenre.COMEDY, MovieGenre.ROMANCE], "Jane Director", ["Lead Actor", "Lead Actress"]),
        ("Horror Thriller", "Spine-chilling horror experience", 105, MovieRating.R,
         [MovieGenre.HORROR, MovieGenre.THRILLER], "Scary Director", ["Scream Actor", "Final Girl"]),
        ("Family Animation", "Fun animated movie for the whole family", 95, MovieRating.G,
         [MovieGenre.ANIMATION, MovieGenre.COMEDY], "Animation Studio", ["Voice Actor 1", "Voice Actor 2"])
    ]
    
    movies = []
    for title, desc, duration, rating, genres, director, cast in movies_data:
        movie_id = str(uuid.uuid4())
        movie = Movie(movie_id, title, desc, duration, rating)
        movie.director = director
        movie.cast = cast
        movie.release_date = date.today() - timedelta(days=30)
        movie.imdb_rating = 7.5 + (len(movies) * 0.3)  # Varying ratings
        
        for genre in genres:
            movie.add_genre(genre)
        
        # Add some user ratings
        for _ in range(100):
            movie.add_user_rating(movie.imdb_rating + ((-1) ** (_ % 2)) * 0.5)
        
        theater.add_movie(movie)
        movies.append(movie)
        print(f"   ✓ {title} added ({duration}min, {rating.value.upper()})")
    
    print()
    
    # Set pricing strategy
    print("3. PRICING STRATEGY SETUP:")
    theater.set_pricing_strategy(DynamicPricingStrategy())
    print(f"   ✓ Pricing strategy: {theater.pricing_strategy.get_strategy_name()}")
    
    print()
    
    # Create shows
    print("4. SHOW SCHEDULING:")
    
    # Schedule shows for today and tomorrow
    base_date = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
    
    show_times = [
        timedelta(hours=0),   # 2:00 PM
        timedelta(hours=3),   # 5:00 PM
        timedelta(hours=6),   # 8:00 PM
        timedelta(hours=9),   # 11:00 PM
    ]
    
    shows = []
    screen_ids = list(theater.screens.keys())
    
    for day_offset in range(2):  # Today and tomorrow
        for i, movie in enumerate(movies):
            for j, time_offset in enumerate(show_times):
                show_time = base_date + timedelta(days=day_offset) + time_offset
                screen_id = screen_ids[i % len(screen_ids)]
                
                # Vary formats
                formats = [ShowFormat.STANDARD_2D, ShowFormat.DIGITAL_3D, ShowFormat.IMAX_2D]
                show_format = formats[j % len(formats)]
                
                # Vary pricing
                base_prices = [Decimal('12.00'), Decimal('15.00'), Decimal('18.00'), Decimal('25.00')]
                base_price = base_prices[j % len(base_prices)]
                
                show = theater.create_show(movie.movie_id, screen_id, show_time, show_format, base_price)
                if show:
                    shows.append(show)
                    print(f"   ✓ {movie.title} - {show_time.strftime('%m/%d %H:%M')} ({show_format.value})")
    
    print()
    
    # Test seat selection and booking
    print("5. BOOKING PROCESS:")
    
    # Get a show for booking
    test_show = shows[0] if shows else None
    if test_show:
        print(f"   Booking for: {test_show}")
        
        # Get available seats
        available_seats = test_show.get_available_seats()[:5]  # First 5 seats
        seat_ids = [seat.seat_id for seat in available_seats]
        
        print(f"   Selected seats: {[f'{s.row}{s.number}' for s in available_seats]}")
        
        # Create booking
        booking = theater.create_booking(
            test_show.show_id,
            "John Customer",
            "john@example.com",
            "555-1234",
            seat_ids,
            "regular"
        )
        
        if booking:
            print(f"   ✓ Booking created: {booking.booking_id[:8]}")
            print(f"   Total amount: ${booking.total_amount:.2f}")
            
            # Show booking details
            booking_info = booking.get_booking_info()
            print(f"   Seats booked:")
            for seat_info in booking_info['seats']:
                seat = seat_info['seat_info']
                price = seat_info['price']
                print(f"     - {seat['row']}{seat['number']} ({seat['seat_type']}): ${price:.2f}")
            
            # Confirm booking
            success = theater.confirm_booking(
                booking.booking_id,
                PaymentMethod.CREDIT_CARD,
                "CC123456789"
            )
            
            if success:
                print(f"   ✓ Booking confirmed with payment")
                print(f"   Ticket codes: {booking.ticket_codes}")
            else:
                print(f"   ✗ Booking confirmation failed")
        else:
            print(f"   ✗ Booking creation failed")
    
    print()
    
    # Test different customer types and pricing
    print("6. PRICING VARIATIONS:")
    
    if test_show:
        sample_seat = available_seats[0] if available_seats else None
        if sample_seat:
            customer_types = ["regular", "student", "senior", "child", "military"]
            
            print(f"   Pricing for {sample_seat.row}{sample_seat.number} ({sample_seat.seat_type.value}):")
            
            for customer_type in customer_types:
                price = theater.pricing_strategy.calculate_price(
                    test_show.base_price,
                    test_show.show_time,
                    sample_seat.seat_type,
                    test_show.show_format,
                    customer_type
                )
                print(f"     {customer_type.title()}: ${price:.2f}")
    
    print()
    
    # Search functionality
    print("7. SEARCH FUNCTIONALITY:")
    
    # Search by movie title
    search_results = theater.search_shows(movie_title="Superhero")
    print(f"   Search 'Superhero': {len(search_results)} shows found")
    
    # Search by genre
    search_results = theater.search_shows(genre=MovieGenre.COMEDY)
    print(f"   Search Comedy genre: {len(search_results)} shows found")
    
    # Search by date
    search_results = theater.search_shows(show_date=date.today())
    print(f"   Search today's shows: {len(search_results)} shows found")
    
    # Search by format
    search_results = theater.search_shows(show_format=ShowFormat.DIGITAL_3D)
    print(f"   Search 3D shows: {len(search_results)} shows found")
    
    print()
    
    # Show theater statistics
    print("8. THEATER STATISTICS:")
    
    theater_info = theater.get_theater_info()
    
    print(f"   Theater: {theater_info['name']}")
    print(f"   Location: {theater_info['address']['city']}, {theater_info['address']['state']}")
    print(f"   Screens: {theater_info['screens_count']}")
    print(f"   Movies: {theater_info['movies_count']}")
    print(f"   Active Shows: {theater_info['active_shows']}")
    print(f"   Total Capacity: {theater_info['total_capacity']} seats")
    print(f"   Total Bookings: {theater_info['statistics']['total_bookings']}")
    print(f"   Total Revenue: ${theater_info['statistics']['total_revenue']:.2f}")
    
    print()
    
    # Show screen information
    print("9. SCREEN INFORMATION:")
    
    for screen in theater.screens.values():
        screen_info = screen.get_screen_info()
        print(f"   {screen_info['name']}:")
        print(f"     Capacity: {screen_info['capacity']}")
        print(f"     Available: {screen_info['available_seats']}")
        print(f"     Occupancy: {screen_info['occupancy_rate']:.1%}")
        print(f"     Formats: {', '.join(screen_info['supported_formats'])}")
        
        # Show seat type breakdown
        print(f"     Seat Types:")
        for seat_type, count in screen_info['seat_types'].items():
            if count > 0:
                print(f"       {seat_type.title()}: {count}")
    
    print()
    
    # Daily report
    print("10. DAILY REPORT:")
    
    daily_report = theater.get_daily_report(date.today())
    
    print(f"   Date: {daily_report['date']}")
    print(f"   Shows: {daily_report['shows_count']}")
    print(f"   Bookings: {daily_report['bookings_count']}")
    print(f"   Tickets Sold: {daily_report['tickets_sold']}")
    print(f"   Revenue: ${daily_report['revenue']:.2f}")
    print(f"   Average Ticket Price: ${daily_report['average_ticket_price']:.2f}")
    
    print(f"   Show Performance:")
    for show_id, occupancy_info in daily_report['show_occupancy'].items():
        movie_title = occupancy_info['movie_title']
        show_time = occupancy_info['show_time']
        occupancy_rate = occupancy_info['occupancy_rate']
        tickets_sold = occupancy_info['tickets_sold']
        
        print(f"     {movie_title} ({show_time}): {occupancy_rate:.1%} occupancy ({tickets_sold} tickets)")
    
    print()
    
    # Test booking expiration cleanup
    print("11. BOOKING MANAGEMENT:")
    
    # Create a test booking that will expire
    if shows:
        expired_booking = theater.create_booking(
            shows[1].show_id,
            "Test Customer",
            "test@example.com",
            "555-9999",
            [available_seats[0].seat_id] if available_seats else [],
            "regular"
        )
        
        if expired_booking:
            # Force expiration
            expired_booking.expires_at = datetime.now() - timedelta(minutes=1)
            
            # Cleanup expired bookings
            cleaned_count = theater.cleanup_expired_bookings()
            print(f"   ✓ Cleaned up {cleaned_count} expired bookings")
    
    # Show movie details
    print("12. MOVIE DETAILS:")
    
    for movie in movies[:2]:  # Show first 2 movies
        movie_info = movie.get_movie_info()
        print(f"   {movie_info['title']}:")
        print(f"     Duration: {movie_info['duration_minutes']} minutes")
        print(f"     Rating: {movie_info['rating'].upper()}")
        print(f"     Genres: {', '.join(movie_info['genres'])}")
        print(f"     Director: {movie_info['director']}")
        print(f"     Cast: {', '.join(movie_info['cast'][:2])}...")
        print(f"     IMDB Rating: {movie_info['ratings']['imdb_rating']:.1f}")
        print(f"     User Rating: {movie_info['ratings']['user_rating']:.1f} ({movie_info['ratings']['user_rating_count']} reviews)")
    
    print()
    print("=== MOVIE TICKET BOOKING SYSTEM DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_movie_booking_system()
