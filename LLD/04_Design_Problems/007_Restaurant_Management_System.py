"""
RESTAURANT MANAGEMENT SYSTEM - Complete System Design
=====================================================

Problem Statement:
Design a comprehensive restaurant management system that handles:
- Table reservation and management
- Menu management and pricing
- Order taking and kitchen management
- Staff management and roles
- Inventory and supply chain
- Billing and payment processing
- Customer management and loyalty
- Reporting and analytics
- Multi-location support
- Integration with delivery services

Requirements:
- Support table reservations and walk-ins
- Manage complex menu items with variations
- Handle order workflow from taking to serving
- Track inventory and automatic reordering
- Support multiple payment methods
- Manage staff schedules and roles
- Generate comprehensive reports
- Support loyalty programs and promotions
- Handle takeout and delivery orders
- Integrate with POS systems

Design Patterns Used:
- Factory: Order and menu item creation
- Strategy: Pricing and discount strategies
- Observer: Order status notifications
- State: Order and table states
- Command: Kitchen operations
- Chain of Responsibility: Order processing
- Decorator: Menu item customizations
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
from collections import defaultdict, deque


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class TableStatus(Enum):
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    RESERVED = "reserved"
    CLEANING = "cleaning"
    OUT_OF_ORDER = "out_of_order"


class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PREPARING = "preparing"
    READY = "ready"
    SERVED = "served"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class OrderType(Enum):
    DINE_IN = "dine_in"
    TAKEOUT = "takeout"
    DELIVERY = "delivery"


class MenuCategory(Enum):
    APPETIZER = "appetizer"
    MAIN_COURSE = "main_course"
    DESSERT = "dessert"
    BEVERAGE = "beverage"
    SIDE_DISH = "side_dish"
    SPECIAL = "special"


class StaffRole(Enum):
    MANAGER = "manager"
    WAITER = "waiter"
    CHEF = "chef"
    CASHIER = "cashier"
    HOST = "host"
    KITCHEN_STAFF = "kitchen_staff"
    DELIVERY = "delivery"


class PaymentMethod(Enum):
    CASH = "cash"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    MOBILE_PAYMENT = "mobile_payment"
    GIFT_CARD = "gift_card"


class InventoryStatus(Enum):
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    EXPIRED = "expired"


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
class MenuItemCustomization:
    """Menu item customization option."""
    name: str
    options: List[str]
    price_modifier: Decimal = Decimal('0')
    is_required: bool = False


@dataclass
class NutritionalInfo:
    """Nutritional information for menu items."""
    calories: int
    protein: float
    carbs: float
    fat: float
    fiber: float
    sodium: float


# ============================================================================
# PRICING STRATEGIES
# ============================================================================

class PricingStrategy(ABC):
    """Abstract pricing strategy."""
    
    @abstractmethod
    def calculate_price(self, base_price: Decimal, customizations: List[str], 
                       customer_type: str, time_of_day: time) -> Decimal:
        """Calculate final price."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class StandardPricingStrategy(PricingStrategy):
    """Standard pricing strategy."""
    
    def calculate_price(self, base_price: Decimal, customizations: List[str], 
                       customer_type: str, time_of_day: time) -> Decimal:
        """Calculate standard price."""
        final_price = base_price
        
        # Apply customer discounts
        if customer_type == "senior":
            final_price *= Decimal('0.9')
        elif customer_type == "student":
            final_price *= Decimal('0.95')
        elif customer_type == "vip":
            final_price *= Decimal('0.85')
        
        return final_price
    
    def get_strategy_name(self) -> str:
        return "Standard Pricing"


class HappyHourPricingStrategy(PricingStrategy):
    """Happy hour pricing strategy."""
    
    def __init__(self):
        self.happy_hour_start = time(15, 0)  # 3 PM
        self.happy_hour_end = time(18, 0)    # 6 PM
        self.happy_hour_discount = Decimal('0.8')  # 20% off
    
    def calculate_price(self, base_price: Decimal, customizations: List[str], 
                       customer_type: str, time_of_day: time) -> Decimal:
        """Calculate price with happy hour discount."""
        final_price = base_price
        
        # Apply happy hour discount
        if self.happy_hour_start <= time_of_day <= self.happy_hour_end:
            final_price *= self.happy_hour_discount
        
        # Apply customer discounts
        if customer_type == "senior":
            final_price *= Decimal('0.9')
        elif customer_type == "student":
            final_price *= Decimal('0.95')
        elif customer_type == "vip":
            final_price *= Decimal('0.85')
        
        return final_price
    
    def get_strategy_name(self) -> str:
        return "Happy Hour Pricing"


class DynamicPricingStrategy(PricingStrategy):
    """Dynamic pricing based on demand and time."""
    
    def __init__(self):
        self.peak_hours = [(time(12, 0), time(14, 0)), (time(18, 0), time(21, 0))]
        self.peak_multiplier = Decimal('1.1')
    
    def calculate_price(self, base_price: Decimal, customizations: List[str], 
                       customer_type: str, time_of_day: time) -> Decimal:
        """Calculate dynamic price."""
        final_price = base_price
        
        # Apply peak hour pricing
        for start_time, end_time in self.peak_hours:
            if start_time <= time_of_day <= end_time:
                final_price *= self.peak_multiplier
                break
        
        # Apply customer discounts
        if customer_type == "senior":
            final_price *= Decimal('0.9')
        elif customer_type == "student":
            final_price *= Decimal('0.95')
        elif customer_type == "vip":
            final_price *= Decimal('0.85')
        
        return final_price
    
    def get_strategy_name(self) -> str:
        return "Dynamic Pricing"


# ============================================================================
# MENU CLASSES
# ============================================================================

class MenuItem:
    """Restaurant menu item with customizations."""
    
    def __init__(self, item_id: str, name: str, description: str, 
                 base_price: Decimal, category: MenuCategory):
        self.item_id = item_id
        self.name = name
        self.description = description
        self.base_price = base_price
        self.category = category
        
        # Availability and timing
        self.is_available = True
        self.available_times: List[Tuple[time, time]] = []  # (start, end) times
        self.available_days: Set[int] = set(range(7))  # 0=Monday, 6=Sunday
        
        # Customizations
        self.customizations: List[MenuItemCustomization] = []
        self.allergens: Set[str] = set()
        self.dietary_tags: Set[str] = set()  # vegetarian, vegan, gluten-free, etc.
        
        # Nutritional information
        self.nutritional_info: Optional[NutritionalInfo] = None
        
        # Preparation details
        self.prep_time_minutes = 15
        self.ingredients: Dict[str, float] = {}  # ingredient_id -> quantity
        
        # Statistics
        self.times_ordered = 0
        self.total_revenue = Decimal('0')
        self.average_rating = 0.0
        self.ratings_count = 0
        
        self.created_at = datetime.now()
    
    def add_customization(self, customization: MenuItemCustomization) -> None:
        """Add customization option."""
        self.customizations.append(customization)
    
    def add_allergen(self, allergen: str) -> None:
        """Add allergen information."""
        self.allergens.add(allergen)
    
    def add_dietary_tag(self, tag: str) -> None:
        """Add dietary tag."""
        self.dietary_tags.add(tag)
    
    def add_ingredient(self, ingredient_id: str, quantity: float) -> None:
        """Add ingredient requirement."""
        self.ingredients[ingredient_id] = quantity
    
    def set_availability_hours(self, start_time: time, end_time: time) -> None:
        """Set availability hours."""
        self.available_times.append((start_time, end_time))
    
    def is_available_now(self) -> bool:
        """Check if item is available now."""
        if not self.is_available:
            return False
        
        now = datetime.now()
        current_time = now.time()
        current_day = now.weekday()
        
        # Check day availability
        if current_day not in self.available_days:
            return False
        
        # Check time availability
        if self.available_times:
            for start_time, end_time in self.available_times:
                if start_time <= current_time <= end_time:
                    return True
            return False
        
        return True
    
    def calculate_price(self, customizations: List[str], pricing_strategy: PricingStrategy,
                       customer_type: str = "regular") -> Decimal:
        """Calculate price with customizations."""
        current_time = datetime.now().time()
        base_price = pricing_strategy.calculate_price(
            self.base_price, customizations, customer_type, current_time
        )
        
        # Add customization costs
        customization_cost = Decimal('0')
        for customization in self.customizations:
            for option in customizations:
                if option in customization.options:
                    customization_cost += customization.price_modifier
        
        return base_price + customization_cost
    
    def add_rating(self, rating: float) -> None:
        """Add customer rating."""
        total_rating = self.average_rating * self.ratings_count + rating
        self.ratings_count += 1
        self.average_rating = total_rating / self.ratings_count
    
    def record_order(self, quantity: int, total_price: Decimal) -> None:
        """Record order statistics."""
        self.times_ordered += quantity
        self.total_revenue += total_price
    
    def get_item_info(self) -> Dict[str, Any]:
        """Get menu item information."""
        return {
            'item_id': self.item_id,
            'name': self.name,
            'description': self.description,
            'base_price': float(self.base_price),
            'category': self.category.value,
            'is_available': self.is_available,
            'is_available_now': self.is_available_now(),
            'prep_time_minutes': self.prep_time_minutes,
            'customizations': [
                {
                    'name': c.name,
                    'options': c.options,
                    'price_modifier': float(c.price_modifier),
                    'is_required': c.is_required
                }
                for c in self.customizations
            ],
            'allergens': list(self.allergens),
            'dietary_tags': list(self.dietary_tags),
            'nutritional_info': {
                'calories': self.nutritional_info.calories,
                'protein': self.nutritional_info.protein,
                'carbs': self.nutritional_info.carbs,
                'fat': self.nutritional_info.fat,
                'fiber': self.nutritional_info.fiber,
                'sodium': self.nutritional_info.sodium
            } if self.nutritional_info else None,
            'statistics': {
                'times_ordered': self.times_ordered,
                'total_revenue': float(self.total_revenue),
                'average_rating': self.average_rating,
                'ratings_count': self.ratings_count
            },
            'created_at': self.created_at.isoformat()
        }
    
    def __str__(self) -> str:
        return f"{self.name} - ${self.base_price} ({self.category.value})"


# ============================================================================
# TABLE MANAGEMENT
# ============================================================================

class Table:
    """Restaurant table with capacity and status."""
    
    def __init__(self, table_number: str, capacity: int, section: str = ""):
        self.table_number = table_number
        self.capacity = capacity
        self.section = section
        self.status = TableStatus.AVAILABLE
        
        # Current occupancy
        self.current_party_size = 0
        self.current_order_id: Optional[str] = None
        self.occupied_since: Optional[datetime] = None
        
        # Reservations
        self.reservations: List[Dict[str, Any]] = []
        
        # Features
        self.is_outdoor = False
        self.has_view = False
        self.is_accessible = False
        
        # Statistics
        self.total_seatings = 0
        self.total_revenue = Decimal('0')
        self.average_dining_time = timedelta(hours=1)
        
        self._lock = threading.Lock()
    
    def reserve_table(self, reservation_id: str, party_size: int, 
                     reservation_time: datetime, customer_name: str) -> bool:
        """Reserve table for specific time."""
        with self._lock:
            # Check if table can accommodate party
            if party_size > self.capacity:
                return False
            
            # Check for conflicts
            for reservation in self.reservations:
                existing_time = datetime.fromisoformat(reservation['reservation_time'])
                time_diff = abs((existing_time - reservation_time).total_seconds())
                if time_diff < 7200:  # 2 hours buffer
                    return False
            
            reservation = {
                'reservation_id': reservation_id,
                'party_size': party_size,
                'reservation_time': reservation_time.isoformat(),
                'customer_name': customer_name,
                'status': 'confirmed'
            }
            
            self.reservations.append(reservation)
            return True
    
    def occupy_table(self, party_size: int, order_id: str = None) -> bool:
        """Occupy table with party."""
        with self._lock:
            if self.status != TableStatus.AVAILABLE or party_size > self.capacity:
                return False
            
            self.status = TableStatus.OCCUPIED
            self.current_party_size = party_size
            self.current_order_id = order_id
            self.occupied_since = datetime.now()
            self.total_seatings += 1
            
            return True
    
    def free_table(self, revenue: Decimal = Decimal('0')) -> bool:
        """Free up table after dining."""
        with self._lock:
            if self.status != TableStatus.OCCUPIED:
                return False
            
            # Calculate dining time
            if self.occupied_since:
                dining_time = datetime.now() - self.occupied_since
                # Update average dining time
                total_time = self.average_dining_time * (self.total_seatings - 1) + dining_time
                self.average_dining_time = total_time / self.total_seatings
            
            # Update statistics
            self.total_revenue += revenue
            
            # Reset table
            self.status = TableStatus.CLEANING
            self.current_party_size = 0
            self.current_order_id = None
            self.occupied_since = None
            
            return True
    
    def complete_cleaning(self) -> None:
        """Complete table cleaning."""
        with self._lock:
            if self.status == TableStatus.CLEANING:
                self.status = TableStatus.AVAILABLE
    
    def set_out_of_order(self, out_of_order: bool = True) -> None:
        """Set table out of order."""
        with self._lock:
            if out_of_order and self.status == TableStatus.AVAILABLE:
                self.status = TableStatus.OUT_OF_ORDER
            elif not out_of_order and self.status == TableStatus.OUT_OF_ORDER:
                self.status = TableStatus.AVAILABLE
    
    def is_available_at(self, check_time: datetime, party_size: int) -> bool:
        """Check if table is available at specific time."""
        if party_size > self.capacity:
            return False
        
        if self.status == TableStatus.OUT_OF_ORDER:
            return False
        
        # Check reservations
        for reservation in self.reservations:
            reservation_time = datetime.fromisoformat(reservation['reservation_time'])
            time_diff = abs((reservation_time - check_time).total_seconds())
            if time_diff < 7200:  # 2 hours buffer
                return False
        
        return True
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get table information."""
        return {
            'table_number': self.table_number,
            'capacity': self.capacity,
            'section': self.section,
            'status': self.status.value,
            'current_party_size': self.current_party_size,
            'current_order_id': self.current_order_id,
            'occupied_since': self.occupied_since.isoformat() if self.occupied_since else None,
            'features': {
                'is_outdoor': self.is_outdoor,
                'has_view': self.has_view,
                'is_accessible': self.is_accessible
            },
            'reservations_count': len(self.reservations),
            'statistics': {
                'total_seatings': self.total_seatings,
                'total_revenue': float(self.total_revenue),
                'average_dining_time_minutes': self.average_dining_time.total_seconds() / 60
            }
        }
    
    def __str__(self) -> str:
        return f"Table {self.table_number} (Cap: {self.capacity}) - {self.status.value}"


# ============================================================================
# ORDER MANAGEMENT
# ============================================================================

class OrderItem:
    """Individual item in an order."""
    
    def __init__(self, menu_item: MenuItem, quantity: int, 
                 customizations: List[str] = None, special_instructions: str = ""):
        self.menu_item = menu_item
        self.quantity = quantity
        self.customizations = customizations or []
        self.special_instructions = special_instructions
        
        # Pricing
        self.unit_price = Decimal('0')
        self.total_price = Decimal('0')
        
        # Status
        self.status = "pending"
        self.prepared_quantity = 0
        
        self.created_at = datetime.now()
    
    def calculate_price(self, pricing_strategy: PricingStrategy, customer_type: str = "regular") -> None:
        """Calculate item price."""
        self.unit_price = self.menu_item.calculate_price(
            self.customizations, pricing_strategy, customer_type
        )
        self.total_price = self.unit_price * self.quantity
    
    def mark_prepared(self, quantity: int = None) -> None:
        """Mark quantity as prepared."""
        if quantity is None:
            quantity = self.quantity
        
        self.prepared_quantity = min(self.prepared_quantity + quantity, self.quantity)
        
        if self.prepared_quantity >= self.quantity:
            self.status = "ready"
    
    def get_item_info(self) -> Dict[str, Any]:
        """Get order item information."""
        return {
            'menu_item_id': self.menu_item.item_id,
            'menu_item_name': self.menu_item.name,
            'quantity': self.quantity,
            'unit_price': float(self.unit_price),
            'total_price': float(self.total_price),
            'customizations': self.customizations,
            'special_instructions': self.special_instructions,
            'status': self.status,
            'prepared_quantity': self.prepared_quantity,
            'prep_time_minutes': self.menu_item.prep_time_minutes
        }


class Order:
    """Restaurant order with items and status tracking."""
    
    def __init__(self, order_id: str, order_type: OrderType, customer_name: str = "",
                 table_number: str = "", phone_number: str = ""):
        self.order_id = order_id
        self.order_type = order_type
        self.customer_name = customer_name
        self.table_number = table_number
        self.phone_number = phone_number
        self.status = OrderStatus.PENDING
        
        # Order items
        self.items: List[OrderItem] = []
        
        # Pricing
        self.subtotal = Decimal('0')
        self.tax_amount = Decimal('0')
        self.tip_amount = Decimal('0')
        self.discount_amount = Decimal('0')
        self.total_amount = Decimal('0')
        
        # Timing
        self.estimated_prep_time = timedelta(minutes=0)
        self.actual_prep_time: Optional[timedelta] = None
        
        # Staff assignments
        self.waiter_id: Optional[str] = None
        self.chef_id: Optional[str] = None
        
        # Delivery information (if applicable)
        self.delivery_address: Optional[Address] = None
        self.delivery_fee = Decimal('0')
        self.delivery_time: Optional[datetime] = None
        
        # Special requests
        self.special_requests: List[str] = []
        
        # Timestamps
        self.created_at = datetime.now()
        self.confirmed_at: Optional[datetime] = None
        self.started_at: Optional[datetime] = None
        self.ready_at: Optional[datetime] = None
        self.served_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        
        self._lock = threading.Lock()
    
    def add_item(self, menu_item: MenuItem, quantity: int, 
                customizations: List[str] = None, special_instructions: str = "") -> None:
        """Add item to order."""
        order_item = OrderItem(menu_item, quantity, customizations, special_instructions)
        self.items.append(order_item)
    
    def remove_item(self, menu_item_id: str) -> bool:
        """Remove item from order."""
        for i, item in enumerate(self.items):
            if item.menu_item.item_id == menu_item_id:
                del self.items[i]
                return True
        return False
    
    def calculate_total(self, pricing_strategy: PricingStrategy, 
                       tax_rate: Decimal = Decimal('0.08'), customer_type: str = "regular") -> None:
        """Calculate order total."""
        with self._lock:
            self.subtotal = Decimal('0')
            
            # Calculate item prices
            for item in self.items:
                item.calculate_price(pricing_strategy, customer_type)
                self.subtotal += item.total_price
            
            # Calculate estimated prep time
            self.estimated_prep_time = timedelta(minutes=max(
                item.menu_item.prep_time_minutes for item in self.items
            )) if self.items else timedelta(0)
            
            # Apply discount
            discounted_subtotal = self.subtotal - self.discount_amount
            
            # Calculate tax
            self.tax_amount = discounted_subtotal * tax_rate
            
            # Calculate total
            self.total_amount = discounted_subtotal + self.tax_amount + self.delivery_fee
    
    def apply_discount(self, discount_amount: Decimal) -> None:
        """Apply discount to order."""
        self.discount_amount = discount_amount
    
    def set_delivery_info(self, address: Address, delivery_fee: Decimal) -> None:
        """Set delivery information."""
        self.delivery_address = address
        self.delivery_fee = delivery_fee
        self.order_type = OrderType.DELIVERY
    
    def add_special_request(self, request: str) -> None:
        """Add special request."""
        if request not in self.special_requests:
            self.special_requests.append(request)
    
    def confirm_order(self, waiter_id: str = None) -> bool:
        """Confirm the order."""
        with self._lock:
            if self.status == OrderStatus.PENDING:
                self.status = OrderStatus.CONFIRMED
                self.confirmed_at = datetime.now()
                self.waiter_id = waiter_id
                return True
            return False
    
    def start_preparation(self, chef_id: str = None) -> bool:
        """Start order preparation."""
        with self._lock:
            if self.status == OrderStatus.CONFIRMED:
                self.status = OrderStatus.PREPARING
                self.started_at = datetime.now()
                self.chef_id = chef_id
                return True
            return False
    
    def mark_ready(self) -> bool:
        """Mark order as ready."""
        with self._lock:
            if self.status == OrderStatus.PREPARING:
                # Check if all items are ready
                all_ready = all(item.status == "ready" for item in self.items)
                if all_ready:
                    self.status = OrderStatus.READY
                    self.ready_at = datetime.now()
                    
                    # Calculate actual prep time
                    if self.started_at:
                        self.actual_prep_time = self.ready_at - self.started_at
                    
                    return True
            return False
    
    def mark_served(self) -> bool:
        """Mark order as served."""
        with self._lock:
            if self.status == OrderStatus.READY:
                self.status = OrderStatus.SERVED
                self.served_at = datetime.now()
                return True
            return False
    
    def complete_order(self) -> bool:
        """Complete the order."""
        with self._lock:
            if self.status == OrderStatus.SERVED:
                self.status = OrderStatus.COMPLETED
                self.completed_at = datetime.now()
                
                # Update menu item statistics
                for item in self.items:
                    item.menu_item.record_order(item.quantity, item.total_price)
                
                return True
            return False
    
    def cancel_order(self, reason: str = "") -> bool:
        """Cancel the order."""
        with self._lock:
            if self.status in [OrderStatus.PENDING, OrderStatus.CONFIRMED]:
                self.status = OrderStatus.CANCELLED
                return True
            return False
    
    def get_order_info(self) -> Dict[str, Any]:
        """Get order information."""
        return {
            'order_id': self.order_id,
            'order_type': self.order_type.value,
            'customer_name': self.customer_name,
            'table_number': self.table_number,
            'phone_number': self.phone_number,
            'status': self.status.value,
            'items': [item.get_item_info() for item in self.items],
            'pricing': {
                'subtotal': float(self.subtotal),
                'tax_amount': float(self.tax_amount),
                'tip_amount': float(self.tip_amount),
                'discount_amount': float(self.discount_amount),
                'delivery_fee': float(self.delivery_fee),
                'total_amount': float(self.total_amount)
            },
            'timing': {
                'estimated_prep_time_minutes': self.estimated_prep_time.total_seconds() / 60,
                'actual_prep_time_minutes': self.actual_prep_time.total_seconds() / 60 if self.actual_prep_time else None
            },
            'staff': {
                'waiter_id': self.waiter_id,
                'chef_id': self.chef_id
            },
            'delivery_address': {
                'street': self.delivery_address.street,
                'city': self.delivery_address.city,
                'state': self.delivery_address.state,
                'zip_code': self.delivery_address.zip_code
            } if self.delivery_address else None,
            'special_requests': self.special_requests,
            'timestamps': {
                'created_at': self.created_at.isoformat(),
                'confirmed_at': self.confirmed_at.isoformat() if self.confirmed_at else None,
                'started_at': self.started_at.isoformat() if self.started_at else None,
                'ready_at': self.ready_at.isoformat() if self.ready_at else None,
                'served_at': self.served_at.isoformat() if self.served_at else None,
                'completed_at': self.completed_at.isoformat() if self.completed_at else None
            }
        }
    
    def __str__(self) -> str:
        return f"Order {self.order_id} - {self.order_type.value} ({self.status.value})"


# ============================================================================
# RESTAURANT MANAGEMENT SYSTEM
# ============================================================================

class Restaurant:
    """Main restaurant management system."""
    
    def __init__(self, restaurant_id: str, name: str, address: Address):
        self.restaurant_id = restaurant_id
        self.name = name
        self.address = address
        self.phone = ""
        self.email = ""
        self.website = ""
        
        # Operating hours
        self.operating_hours: Dict[int, Tuple[time, time]] = {}  # day -> (open, close)
        
        # Menu and tables
        self.menu_items: Dict[str, MenuItem] = {}
        self.tables: Dict[str, Table] = {}
        
        # Orders and reservations
        self.orders: Dict[str, Order] = {}
        self.reservations: Dict[str, Dict[str, Any]] = {}
        
        # Staff
        self.staff: Dict[str, Dict[str, Any]] = {}
        
        # Pricing strategy
        self.pricing_strategy: PricingStrategy = StandardPricingStrategy()
        
        # Statistics
        self.total_orders = 0
        self.total_revenue = Decimal('0')
        self.daily_revenue = Decimal('0')
        self.last_revenue_reset = date.today()
        
        # Settings
        self.tax_rate = Decimal('0.08')
        self.service_charge_rate = Decimal('0.15')
        self.delivery_fee = Decimal('5.00')
        
        self._lock = threading.Lock()
        
        print(f"🍽️ Restaurant '{name}' initialized")
    
    def set_operating_hours(self, day: int, open_time: time, close_time: time) -> None:
        """Set operating hours for a day (0=Monday, 6=Sunday)."""
        self.operating_hours[day] = (open_time, close_time)
    
    def add_menu_item(self, menu_item: MenuItem) -> None:
        """Add menu item."""
        self.menu_items[menu_item.item_id] = menu_item
    
    def remove_menu_item(self, item_id: str) -> bool:
        """Remove menu item."""
        if item_id in self.menu_items:
            del self.menu_items[item_id]
            return True
        return False
    
    def add_table(self, table: Table) -> None:
        """Add table."""
        self.tables[table.table_number] = table
    
    def remove_table(self, table_number: str) -> bool:
        """Remove table."""
        table = self.tables.get(table_number)
        if table and table.status == TableStatus.AVAILABLE:
            del self.tables[table_number]
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
            'is_active': True,
            'shift_start': None,
            'shift_end': None
        }
    
    def set_pricing_strategy(self, strategy: PricingStrategy) -> None:
        """Set pricing strategy."""
        self.pricing_strategy = strategy
    
    def get_available_tables(self, party_size: int, reservation_time: datetime = None) -> List[Table]:
        """Get available tables for party size."""
        available_tables = []
        check_time = reservation_time or datetime.now()
        
        for table in self.tables.values():
            if table.is_available_at(check_time, party_size):
                available_tables.append(table)
        
        # Sort by capacity (prefer smaller tables that fit)
        available_tables.sort(key=lambda t: (t.capacity >= party_size, t.capacity))
        return available_tables
    
    def make_reservation(self, customer_name: str, party_size: int, 
                        reservation_time: datetime, phone: str = "", 
                        special_requests: List[str] = None) -> Optional[str]:
        """Make table reservation."""
        available_tables = self.get_available_tables(party_size, reservation_time)
        
        if not available_tables:
            return None
        
        # Select best table
        table = available_tables[0]
        reservation_id = str(uuid.uuid4())
        
        if table.reserve_table(reservation_id, party_size, reservation_time, customer_name):
            reservation = {
                'reservation_id': reservation_id,
                'customer_name': customer_name,
                'party_size': party_size,
                'reservation_time': reservation_time.isoformat(),
                'table_number': table.table_number,
                'phone': phone,
                'special_requests': special_requests or [],
                'status': 'confirmed',
                'created_at': datetime.now().isoformat()
            }
            
            with self._lock:
                self.reservations[reservation_id] = reservation
            
            return reservation_id
        
        return None
    
    def create_order(self, order_type: OrderType, customer_name: str = "",
                    table_number: str = "", phone_number: str = "") -> Order:
        """Create new order."""
        order_id = str(uuid.uuid4())
        order = Order(order_id, order_type, customer_name, table_number, phone_number)
        
        with self._lock:
            self.orders[order_id] = order
            self.total_orders += 1
        
        return order
    
    def add_item_to_order(self, order_id: str, menu_item_id: str, quantity: int,
                         customizations: List[str] = None, special_instructions: str = "") -> bool:
        """Add item to order."""
        order = self.orders.get(order_id)
        menu_item = self.menu_items.get(menu_item_id)
        
        if not order or not menu_item or not menu_item.is_available_now():
            return False
        
        order.add_item(menu_item, quantity, customizations, special_instructions)
        return True
    
    def calculate_order_total(self, order_id: str, customer_type: str = "regular") -> bool:
        """Calculate order total."""
        order = self.orders.get(order_id)
        if not order:
            return False
        
        order.calculate_total(self.pricing_strategy, self.tax_rate, customer_type)
        return True
    
    def confirm_order(self, order_id: str, waiter_id: str = None) -> bool:
        """Confirm order."""
        order = self.orders.get(order_id)
        if not order:
            return False
        
        if order.confirm_order(waiter_id):
            # Occupy table if dine-in
            if order.order_type == OrderType.DINE_IN and order.table_number:
                table = self.tables.get(order.table_number)
                if table:
                    table.occupy_table(1, order_id)  # Simplified party size
            
            return True
        
        return False
    
    def process_payment(self, order_id: str, payment_method: PaymentMethod,
                       tip_amount: Decimal = Decimal('0')) -> bool:
        """Process order payment."""
        order = self.orders.get(order_id)
        if not order:
            return False
        
        order.tip_amount = tip_amount
        order.total_amount += tip_amount
        
        # Update revenue
        with self._lock:
            self._reset_daily_revenue_if_needed()
            self.total_revenue += order.total_amount
            self.daily_revenue += order.total_amount
        
        # Complete order
        order.complete_order()
        
        # Free table if dine-in
        if order.order_type == OrderType.DINE_IN and order.table_number:
            table = self.tables.get(order.table_number)
            if table:
                table.free_table(order.total_amount)
        
        return True
    
    def get_menu_by_category(self, category: MenuCategory = None) -> List[Dict[str, Any]]:
        """Get menu items by category."""
        items = []
        
        for item in self.menu_items.values():
            if category is None or item.category == category:
                items.append(item.get_item_info())
        
        # Sort by category and name
        items.sort(key=lambda x: (x['category'], x['name']))
        return items
    
    def search_menu_items(self, query: str, dietary_filter: str = None) -> List[Dict[str, Any]]:
        """Search menu items."""
        results = []
        query_lower = query.lower()
        
        for item in self.menu_items.values():
            # Text search
            if (query_lower in item.name.lower() or 
                query_lower in item.description.lower()):
                
                # Dietary filter
                if dietary_filter and dietary_filter not in item.dietary_tags:
                    continue
                
                results.append(item.get_item_info())
        
        return results
    
    def get_kitchen_orders(self) -> List[Dict[str, Any]]:
        """Get orders for kitchen display."""
        kitchen_orders = []
        
        for order in self.orders.values():
            if order.status in [OrderStatus.CONFIRMED, OrderStatus.PREPARING]:
                kitchen_orders.append(order.get_order_info())
        
        # Sort by order time
        kitchen_orders.sort(key=lambda x: x['timestamps']['created_at'])
        return kitchen_orders
    
    def get_table_status(self) -> Dict[str, Any]:
        """Get current table status."""
        status = {
            'total_tables': len(self.tables),
            'available': 0,
            'occupied': 0,
            'reserved': 0,
            'cleaning': 0,
            'out_of_order': 0,
            'tables': []
        }
        
        for table in self.tables.values():
            table_info = table.get_table_info()
            status['tables'].append(table_info)
            
            # Count by status
            if table.status == TableStatus.AVAILABLE:
                status['available'] += 1
            elif table.status == TableStatus.OCCUPIED:
                status['occupied'] += 1
            elif table.status == TableStatus.RESERVED:
                status['reserved'] += 1
            elif table.status == TableStatus.CLEANING:
                status['cleaning'] += 1
            elif table.status == TableStatus.OUT_OF_ORDER:
                status['out_of_order'] += 1
        
        status['occupancy_rate'] = status['occupied'] / max(1, status['total_tables'])
        return status
    
    def get_daily_report(self, target_date: date = None) -> Dict[str, Any]:
        """Get daily operations report."""
        if not target_date:
            target_date = date.today()
        
        # Filter orders for the day
        daily_orders = []
        for order in self.orders.values():
            order_date = order.created_at.date()
            if order_date == target_date:
                daily_orders.append(order)
        
        # Calculate statistics
        total_orders = len(daily_orders)
        completed_orders = len([o for o in daily_orders if o.status == OrderStatus.COMPLETED])
        cancelled_orders = len([o for o in daily_orders if o.status == OrderStatus.CANCELLED])
        
        daily_revenue = sum(o.total_amount for o in daily_orders if o.status == OrderStatus.COMPLETED)
        
        # Order type breakdown
        dine_in_orders = len([o for o in daily_orders if o.order_type == OrderType.DINE_IN])
        takeout_orders = len([o for o in daily_orders if o.order_type == OrderType.TAKEOUT])
        delivery_orders = len([o for o in daily_orders if o.order_type == OrderType.DELIVERY])
        
        # Popular items
        item_counts = defaultdict(int)
        for order in daily_orders:
            for item in order.items:
                item_counts[item.menu_item.name] += item.quantity
        
        popular_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'date': target_date.isoformat(),
            'orders': {
                'total_orders': total_orders,
                'completed_orders': completed_orders,
                'cancelled_orders': cancelled_orders,
                'completion_rate': completed_orders / max(1, total_orders)
            },
            'order_types': {
                'dine_in': dine_in_orders,
                'takeout': takeout_orders,
                'delivery': delivery_orders
            },
            'revenue': {
                'daily_revenue': float(daily_revenue),
                'average_order_value': float(daily_revenue / max(1, completed_orders))
            },
            'popular_items': [
                {'item_name': name, 'quantity_sold': count}
                for name, count in popular_items
            ]
        }
    
    def _reset_daily_revenue_if_needed(self) -> None:
        """Reset daily revenue if new day."""
        if self.last_revenue_reset < date.today():
            self.daily_revenue = Decimal('0')
            self.last_revenue_reset = date.today()
    
    def get_restaurant_info(self) -> Dict[str, Any]:
        """Get restaurant information."""
        return {
            'restaurant_id': self.restaurant_id,
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
            'menu_items_count': len(self.menu_items),
            'tables_count': len(self.tables),
            'staff_count': len([s for s in self.staff.values() if s['is_active']]),
            'statistics': {
                'total_orders': self.total_orders,
                'total_revenue': float(self.total_revenue),
                'daily_revenue': float(self.daily_revenue)
            },
            'pricing_strategy': self.pricing_strategy.get_strategy_name()
        }


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_restaurant_system():
    """Demonstrate the restaurant management system."""
    print("=== RESTAURANT MANAGEMENT SYSTEM DEMONSTRATION ===\n")
    
    # Initialize restaurant
    address = Address("123 Food Street", "Culinary City", "CA", "USA", "90210")
    restaurant = Restaurant("REST001", "The Gourmet Kitchen", address)
    restaurant.phone = "555-FOOD"
    restaurant.email = "info@gourmetkitchen.com"
    
    print("1. RESTAURANT SETUP:")
    
    # Set operating hours
    for day in range(7):  # Monday to Sunday
        if day < 5:  # Weekdays
            restaurant.set_operating_hours(day, time(11, 0), time(22, 0))
        else:  # Weekends
            restaurant.set_operating_hours(day, time(10, 0), time(23, 0))
    
    print(f"   ✓ Operating hours set for all days")
    
    # Add tables
    table_configs = [
        ("T01", 2, "Main"), ("T02", 2, "Main"), ("T03", 4, "Main"),
        ("T04", 4, "Main"), ("T05", 6, "Main"), ("T06", 8, "Main"),
        ("P01", 4, "Patio"), ("P02", 4, "Patio"), ("P03", 6, "Patio")
    ]
    
    for table_num, capacity, section in table_configs:
        table = Table(table_num, capacity, section)
        if section == "Patio":
            table.is_outdoor = True
            table.has_view = True
        restaurant.add_table(table)
    
    print(f"   ✓ {len(restaurant.tables)} tables added")
    
    # Add staff
    staff_data = [
        ("Alice Manager", StaffRole.MANAGER),
        ("Bob Waiter", StaffRole.WAITER),
        ("Carol Chef", StaffRole.CHEF),
        ("David Cashier", StaffRole.CASHIER),
        ("Eve Host", StaffRole.HOST)
    ]
    
    for name, role in staff_data:
        staff_id = str(uuid.uuid4())
        contact = ContactInfo("555-0000", f"{name.lower().replace(' ', '')}@restaurant.com", address)
        restaurant.add_staff(staff_id, name, role, contact)
    
    print(f"   ✓ {len(restaurant.staff)} staff members added")
    
    print()
    
    # Create menu
    print("2. MENU CREATION:")
    
    menu_data = [
        # Appetizers
        ("Caesar Salad", "Fresh romaine with parmesan and croutons", Decimal('12.99'), MenuCategory.APPETIZER, 10),
        ("Bruschetta", "Toasted bread with tomato and basil", Decimal('9.99'), MenuCategory.APPETIZER, 8),
        ("Calamari Rings", "Crispy fried squid with marinara sauce", Decimal('14.99'), MenuCategory.APPETIZER, 12),
        
        # Main Courses
        ("Grilled Salmon", "Atlantic salmon with lemon herb butter", Decimal('24.99'), MenuCategory.MAIN_COURSE, 20),
        ("Ribeye Steak", "12oz ribeye with garlic mashed potatoes", Decimal('32.99'), MenuCategory.MAIN_COURSE, 25),
        ("Chicken Parmesan", "Breaded chicken with marinara and mozzarella", Decimal('19.99'), MenuCategory.MAIN_COURSE, 18),
        ("Vegetable Pasta", "Penne with seasonal vegetables", Decimal('16.99'), MenuCategory.MAIN_COURSE, 15),
        
        # Desserts
        ("Chocolate Cake", "Rich chocolate cake with vanilla ice cream", Decimal('8.99'), MenuCategory.DESSERT, 5),
        ("Tiramisu", "Classic Italian dessert", Decimal('7.99'), MenuCategory.DESSERT, 5),
        
        # Beverages
        ("House Wine", "Red or white wine by the glass", Decimal('8.99'), MenuCategory.BEVERAGE, 2),
        ("Craft Beer", "Local brewery selection", Decimal('5.99'), MenuCategory.BEVERAGE, 2),
        ("Fresh Juice", "Orange, apple, or cranberry", Decimal('3.99'), MenuCategory.BEVERAGE, 2)
    ]
    
    for name, description, price, category, prep_time in menu_data:
        item_id = str(uuid.uuid4())
        menu_item = MenuItem(item_id, name, description, price, category)
        menu_item.prep_time_minutes = prep_time
        
        # Add some customizations
        if category == MenuCategory.MAIN_COURSE:
            cooking_custom = MenuItemCustomization("Cooking Level", ["Rare", "Medium Rare", "Medium", "Well Done"])
            menu_item.add_customization(cooking_custom)
            
            sides_custom = MenuItemCustomization("Side Dish", ["Fries", "Salad", "Rice", "Vegetables"], Decimal('2.00'))
            menu_item.add_customization(sides_custom)
        
        # Add dietary tags
        if "Vegetable" in name or "Pasta" in name:
            menu_item.add_dietary_tag("vegetarian")
        
        if "Salmon" in name or "Calamari" in name:
            menu_item.add_allergen("seafood")
        
        # Add nutritional info (simplified)
        if category == MenuCategory.MAIN_COURSE:
            nutrition = NutritionalInfo(
                calories=450 + (50 if "Steak" in name else 0),
                protein=35.0,
                carbs=25.0,
                fat=15.0,
                fiber=5.0,
                sodium=800.0
            )
            menu_item.nutritional_info = nutrition
        
        restaurant.add_menu_item(menu_item)
    
    print(f"   ✓ {len(restaurant.menu_items)} menu items created")
    
    print()
    
    # Test pricing strategies
    print("3. PRICING STRATEGY TESTING:")
    
    strategies = [
        StandardPricingStrategy(),
        HappyHourPricingStrategy(),
        DynamicPricingStrategy()
    ]
    
    # Get a sample menu item
    sample_item = next(iter(restaurant.menu_items.values()))
    
    for strategy in strategies:
        restaurant.set_pricing_strategy(strategy)
        
        # Test different customer types
        for customer_type in ["regular", "senior", "student", "vip"]:
            price = sample_item.calculate_price([], strategy, customer_type)
            print(f"   {strategy.get_strategy_name()} - {customer_type}: ${price:.2f}")
    
    # Set back to standard pricing
    restaurant.set_pricing_strategy(StandardPricingStrategy())
    
    print()
    
    # Make reservations
    print("4. RESERVATION SYSTEM:")
    
    reservation_data = [
        ("John Smith", 4, datetime.now() + timedelta(hours=2), "555-1001"),
        ("Jane Doe", 2, datetime.now() + timedelta(hours=3), "555-1002"),
        ("Bob Johnson", 6, datetime.now() + timedelta(days=1, hours=1), "555-1003")
    ]
    
    reservations = []
    for name, party_size, res_time, phone in reservation_data:
        reservation_id = restaurant.make_reservation(name, party_size, res_time, phone, ["Window seat"])
        if reservation_id:
            reservations.append(reservation_id)
            print(f"   ✓ Reservation for {name} (party of {party_size}): {reservation_id[:8]}")
        else:
            print(f"   ✗ Failed to make reservation for {name}")
    
    print()
    
    # Create orders
    print("5. ORDER MANAGEMENT:")
    
    # Dine-in order
    dine_in_order = restaurant.create_order(OrderType.DINE_IN, "Alice Customer", "T03", "555-2001")
    
    # Add items to order
    menu_items = list(restaurant.menu_items.values())
    
    # Add Caesar Salad
    caesar_item = next(item for item in menu_items if "Caesar" in item.name)
    restaurant.add_item_to_order(dine_in_order.order_id, caesar_item.item_id, 1)
    
    # Add Grilled Salmon with customizations
    salmon_item = next(item for item in menu_items if "Salmon" in item.name)
    restaurant.add_item_to_order(
        dine_in_order.order_id, 
        salmon_item.item_id, 
        2, 
        ["Medium", "Vegetables"], 
        "Extra lemon on the side"
    )
    
    # Add wine
    wine_item = next(item for item in menu_items if "Wine" in item.name)
    restaurant.add_item_to_order(dine_in_order.order_id, wine_item.item_id, 2)
    
    # Calculate total
    restaurant.calculate_order_total(dine_in_order.order_id, "regular")
    
    print(f"   ✓ Dine-in order created: ${dine_in_order.total_amount:.2f}")
    
    # Takeout order
    takeout_order = restaurant.create_order(OrderType.TAKEOUT, "Bob Customer", "", "555-2002")
    
    # Add items
    pasta_item = next(item for item in menu_items if "Pasta" in item.name)
    restaurant.add_item_to_order(takeout_order.order_id, pasta_item.item_id, 1)
    
    cake_item = next(item for item in menu_items if "Cake" in item.name)
    restaurant.add_item_to_order(takeout_order.order_id, cake_item.item_id, 1)
    
    restaurant.calculate_order_total(takeout_order.order_id, "student")
    
    print(f"   ✓ Takeout order created: ${takeout_order.total_amount:.2f}")
    
    # Delivery order
    delivery_order = restaurant.create_order(OrderType.DELIVERY, "Carol Customer", "", "555-2003")
    delivery_address = Address("456 Home St", "Delivery City", "CA", "USA", "90211")
    delivery_order.set_delivery_info(delivery_address, restaurant.delivery_fee)
    
    # Add items
    steak_item = next(item for item in menu_items if "Steak" in item.name)
    restaurant.add_item_to_order(
        delivery_order.order_id, 
        steak_item.item_id, 
        1, 
        ["Medium Rare", "Fries"]
    )
    
    restaurant.calculate_order_total(delivery_order.order_id, "vip")
    
    print(f"   ✓ Delivery order created: ${delivery_order.total_amount:.2f}")
    
    print()
    
    # Confirm orders and simulate workflow
    print("6. ORDER WORKFLOW SIMULATION:")
    
    orders = [dine_in_order, takeout_order, delivery_order]
    
    # Get staff IDs
    waiter_id = next(staff_id for staff_id, staff in restaurant.staff.items() if staff['role'] == StaffRole.WAITER)
    chef_id = next(staff_id for staff_id, staff in restaurant.staff.items() if staff['role'] == StaffRole.CHEF)
    
    for i, order in enumerate(orders):
        # Confirm order
        restaurant.confirm_order(order.order_id, waiter_id)
        print(f"   ✓ Order {order.order_id[:8]} confirmed")
        
        # Start preparation
        order.start_preparation(chef_id)
        print(f"   ✓ Order {order.order_id[:8]} preparation started")
        
        # Mark items as prepared
        for item in order.items:
            item.mark_prepared()
        
        # Mark order ready
        order.mark_ready()
        print(f"   ✓ Order {order.order_id[:8]} ready")
        
        # Serve order
        order.mark_served()
        print(f"   ✓ Order {order.order_id[:8]} served")
    
    print()
    
    # Process payments
    print("7. PAYMENT PROCESSING:")
    
    payment_methods = [PaymentMethod.CREDIT_CARD, PaymentMethod.CASH, PaymentMethod.MOBILE_PAYMENT]
    tips = [Decimal('5.00'), Decimal('3.00'), Decimal('7.50')]
    
    for i, order in enumerate(orders):
        success = restaurant.process_payment(
            order.order_id, 
            payment_methods[i], 
            tips[i]
        )
        
        if success:
            final_total = order.total_amount
            print(f"   ✓ Payment processed for {order.order_id[:8]}: ${final_total:.2f}")
        else:
            print(f"   ✗ Payment failed for {order.order_id[:8]}")
    
    print()
    
    # Show menu by category
    print("8. MENU DISPLAY:")
    
    for category in MenuCategory:
        items = restaurant.get_menu_by_category(category)
        if items:
            print(f"   {category.value.replace('_', ' ').title()}:")
            for item in items[:3]:  # Show first 3 items
                price = item['base_price']
                availability = "✓" if item['is_available_now'] else "✗"
                print(f"     {availability} {item['name']} - ${price:.2f}")
                if item['dietary_tags']:
                    print(f"       Tags: {', '.join(item['dietary_tags'])}")
    
    print()
    
    # Search menu
    print("9. MENU SEARCH:")
    
    search_queries = ["salmon", "vegetarian", "chocolate"]
    
    for query in search_queries:
        results = restaurant.search_menu_items(query)
        print(f"   Search '{query}': {len(results)} results")
        for result in results[:2]:  # Show first 2 results
            print(f"     - {result['name']} (${result['base_price']:.2f})")
    
    print()
    
    # Show table status
    print("10. TABLE STATUS:")
    
    table_status = restaurant.get_table_status()
    print(f"   Total Tables: {table_status['total_tables']}")
    print(f"   Available: {table_status['available']}")
    print(f"   Occupied: {table_status['occupied']}")
    print(f"   Occupancy Rate: {table_status['occupancy_rate']:.1%}")
    
    print(f"   Table Details:")
    for table_info in table_status['tables'][:5]:  # Show first 5 tables
        table_num = table_info['table_number']
        capacity = table_info['capacity']
        status = table_info['status']
        section = table_info['section']
        print(f"     {table_num} (Cap: {capacity}, {section}): {status}")
    
    print()
    
    # Kitchen display
    print("11. KITCHEN DISPLAY:")
    
    kitchen_orders = restaurant.get_kitchen_orders()
    print(f"   Active Kitchen Orders: {len(kitchen_orders)}")
    
    for order_info in kitchen_orders:
        order_id = order_info['order_id'][:8]
        order_type = order_info['order_type']
        status = order_info['status']
        item_count = len(order_info['items'])
        print(f"     {order_id} ({order_type}): {item_count} items - {status}")
    
    print()
    
    # Daily report
    print("12. DAILY REPORT:")
    
    daily_report = restaurant.get_daily_report()
    
    print(f"   Date: {daily_report['date']}")
    print(f"   Orders:")
    print(f"     Total: {daily_report['orders']['total_orders']}")
    print(f"     Completed: {daily_report['orders']['completed_orders']}")
    print(f"     Completion Rate: {daily_report['orders']['completion_rate']:.1%}")
    
    print(f"   Order Types:")
    print(f"     Dine-in: {daily_report['order_types']['dine_in']}")
    print(f"     Takeout: {daily_report['order_types']['takeout']}")
    print(f"     Delivery: {daily_report['order_types']['delivery']}")
    
    print(f"   Revenue:")
    print(f"     Daily Revenue: ${daily_report['revenue']['daily_revenue']:.2f}")
    print(f"     Average Order Value: ${daily_report['revenue']['average_order_value']:.2f}")
    
    print(f"   Popular Items:")
    for item in daily_report['popular_items']:
        print(f"     {item['item_name']}: {item['quantity_sold']} sold")
    
    print()
    
    # Restaurant summary
    print("13. RESTAURANT SUMMARY:")
    
    restaurant_info = restaurant.get_restaurant_info()
    
    print(f"   Restaurant: {restaurant_info['name']}")
    print(f"   Location: {restaurant_info['address']['city']}, {restaurant_info['address']['state']}")
    print(f"   Contact: {restaurant_info['contact']['phone']}")
    print(f"   Menu Items: {restaurant_info['menu_items_count']}")
    print(f"   Tables: {restaurant_info['tables_count']}")
    print(f"   Active Staff: {restaurant_info['staff_count']}")
    print(f"   Total Orders: {restaurant_info['statistics']['total_orders']}")
    print(f"   Total Revenue: ${restaurant_info['statistics']['total_revenue']:.2f}")
    print(f"   Daily Revenue: ${restaurant_info['statistics']['daily_revenue']:.2f}")
    print(f"   Pricing Strategy: {restaurant_info['pricing_strategy']}")
    
    print()
    print("=== RESTAURANT MANAGEMENT SYSTEM DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_restaurant_system()
