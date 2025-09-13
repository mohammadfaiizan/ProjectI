"""
FOOD DELIVERY SYSTEM - Complete System Design
============================================

Problem Statement:
Design a comprehensive food delivery system that handles:
- Restaurant and menu management
- Customer ordering and cart management
- Order processing and kitchen management
- Delivery partner assignment and tracking
- Real-time order tracking and notifications
- Payment processing and billing
- Rating and review system
- Inventory management and availability
- Promotional campaigns and discounts
- Multi-restaurant operations
- Analytics and reporting

Requirements:
- Support restaurant registration and menu management
- Handle customer browsing, ordering, and payment
- Manage order workflow from placement to delivery
- Implement efficient delivery partner assignment
- Provide real-time tracking and notifications
- Support multiple payment methods and billing
- Handle inventory management and availability
- Implement rating and review system
- Support promotional campaigns and discounts
- Provide comprehensive analytics and reporting
- Scale to handle high-volume concurrent orders
- Support multi-city operations

Design Patterns Used:
- Factory: Order and delivery creation
- State: Order state management
- Observer: Real-time tracking and notifications
- Strategy: Delivery assignment and pricing strategies
- Command: Order operations with history
- Template Method: Order processing workflow
- Decorator: Promotions and discounts
- Facade: Simplified ordering interface
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
    CUSTOMER = "customer"
    RESTAURANT_OWNER = "restaurant_owner"
    DELIVERY_PARTNER = "delivery_partner"
    ADMIN = "admin"


class OrderStatus(Enum):
    PLACED = "placed"
    CONFIRMED = "confirmed"
    PREPARING = "preparing"
    READY = "ready"
    PICKED_UP = "picked_up"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class DeliveryPartnerStatus(Enum):
    OFFLINE = "offline"
    AVAILABLE = "available"
    BUSY = "busy"
    ON_BREAK = "on_break"


class PaymentStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class CuisineType(Enum):
    ITALIAN = "italian"
    CHINESE = "chinese"
    INDIAN = "indian"
    MEXICAN = "mexican"
    AMERICAN = "american"
    THAI = "thai"
    JAPANESE = "japanese"
    MIDDLE_EASTERN = "middle_eastern"
    FAST_FOOD = "fast_food"
    DESSERTS = "desserts"


class PromotionType(Enum):
    PERCENTAGE_DISCOUNT = "percentage_discount"
    FIXED_DISCOUNT = "fixed_discount"
    FREE_DELIVERY = "free_delivery"
    BUY_ONE_GET_ONE = "buy_one_get_one"
    MINIMUM_ORDER_DISCOUNT = "minimum_order_discount"


@dataclass
class Location:
    """Geographic location."""
    latitude: float
    longitude: float
    address: str
    city: str
    postal_code: str = ""
    
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
    is_verified: bool = False
    
    def __post_init__(self):
        if not self.user_id:
            self.user_id = str(uuid.uuid4())
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass
class Customer(User):
    """Customer information."""
    delivery_addresses: List[Location] = field(default_factory=list)
    favorite_restaurants: Set[str] = field(default_factory=set)
    
    # Order history stats
    total_orders: int = 0
    total_spent: float = 0.0
    
    # Preferences
    dietary_preferences: List[str] = field(default_factory=list)
    preferred_cuisines: List[CuisineType] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        self.user_type = UserType.CUSTOMER


@dataclass
class MenuItem:
    """Restaurant menu item."""
    item_id: str
    name: str
    description: str
    price: float
    category: str
    
    # Availability
    is_available: bool = True
    preparation_time: int = 15  # minutes
    
    # Nutritional info
    calories: Optional[int] = None
    is_vegetarian: bool = False
    is_vegan: bool = False
    allergens: List[str] = field(default_factory=list)
    
    # Images and extras
    image_url: str = ""
    customization_options: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.item_id:
            self.item_id = str(uuid.uuid4())


@dataclass
class Restaurant:
    """Restaurant information."""
    restaurant_id: str
    name: str
    owner_id: str
    location: Location
    cuisine_types: List[CuisineType]
    
    # Contact and operational info
    phone: str = ""
    email: str = ""
    description: str = ""
    
    # Operational settings
    is_open: bool = True
    operating_hours: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    delivery_radius: float = 10.0  # kilometers
    minimum_order_amount: float = 10.0
    delivery_fee: float = 3.0
    
    # Menu and capacity
    menu_items: Dict[str, MenuItem] = field(default_factory=dict)
    max_orders_per_hour: int = 30
    current_orders_count: int = 0
    
    # Ratings and reviews
    rating: float = 5.0
    total_reviews: int = 0
    
    # Business metrics
    total_orders: int = 0
    total_revenue: float = 0.0
    
    def __post_init__(self):
        if not self.restaurant_id:
            self.restaurant_id = str(uuid.uuid4())
        
        if not self.operating_hours:
            # Default operating hours (9 AM - 11 PM)
            for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                self.operating_hours[day] = ('09:00', '23:00')
    
    def is_open_now(self) -> bool:
        """Check if restaurant is currently open."""
        if not self.is_open:
            return False
        
        now = datetime.now()
        day_name = now.strftime('%A').lower()
        current_time = now.strftime('%H:%M')
        
        if day_name in self.operating_hours:
            open_time, close_time = self.operating_hours[day_name]
            return open_time <= current_time <= close_time
        
        return False
    
    def can_deliver_to(self, location: Location) -> bool:
        """Check if restaurant can deliver to location."""
        distance = self.location.distance_to(location)
        return distance <= self.delivery_radius
    
    def add_menu_item(self, item: MenuItem) -> None:
        """Add item to menu."""
        self.menu_items[item.item_id] = item
    
    def update_availability(self, item_id: str, available: bool) -> bool:
        """Update menu item availability."""
        if item_id in self.menu_items:
            self.menu_items[item_id].is_available = available
            return True
        return False


@dataclass
class CartItem:
    """Item in shopping cart."""
    menu_item: MenuItem
    quantity: int
    customizations: Dict[str, str] = field(default_factory=dict)
    special_instructions: str = ""
    
    @property
    def total_price(self) -> float:
        return self.menu_item.price * self.quantity


@dataclass
class Cart:
    """Shopping cart."""
    customer_id: str
    restaurant_id: str
    items: List[CartItem] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def subtotal(self) -> float:
        return sum(item.total_price for item in self.items)
    
    @property
    def total_items(self) -> int:
        return sum(item.quantity for item in self.items)
    
    def add_item(self, menu_item: MenuItem, quantity: int = 1, **kwargs) -> None:
        """Add item to cart."""
        cart_item = CartItem(menu_item=menu_item, quantity=quantity, **kwargs)
        self.items.append(cart_item)
    
    def remove_item(self, item_index: int) -> bool:
        """Remove item from cart."""
        if 0 <= item_index < len(self.items):
            self.items.pop(item_index)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all items from cart."""
        self.items.clear()


@dataclass
class Promotion:
    """Promotional offer."""
    promotion_id: str
    name: str
    description: str
    promotion_type: PromotionType
    
    # Discount values
    discount_percentage: float = 0.0
    discount_amount: float = 0.0
    minimum_order_amount: float = 0.0
    
    # Validity
    is_active: bool = True
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    
    # Restrictions
    applicable_restaurants: Set[str] = field(default_factory=set)
    applicable_items: Set[str] = field(default_factory=set)
    max_uses: Optional[int] = None
    uses_count: int = 0
    
    def __post_init__(self):
        if not self.promotion_id:
            self.promotion_id = str(uuid.uuid4())
    
    def is_valid(self) -> bool:
        """Check if promotion is currently valid."""
        if not self.is_active:
            return False
        
        now = datetime.now()
        if now < self.start_date:
            return False
        
        if self.end_date and now > self.end_date:
            return False
        
        if self.max_uses and self.uses_count >= self.max_uses:
            return False
        
        return True
    
    def calculate_discount(self, cart: Cart) -> float:
        """Calculate discount amount for cart."""
        if not self.is_valid():
            return 0.0
        
        if cart.subtotal < self.minimum_order_amount:
            return 0.0
        
        if self.promotion_type == PromotionType.PERCENTAGE_DISCOUNT:
            return cart.subtotal * (self.discount_percentage / 100)
        elif self.promotion_type == PromotionType.FIXED_DISCOUNT:
            return min(self.discount_amount, cart.subtotal)
        
        return 0.0


@dataclass
class Order:
    """Food order."""
    order_id: str
    customer_id: str
    restaurant_id: str
    items: List[CartItem]
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    confirmed_at: Optional[datetime] = None
    prepared_at: Optional[datetime] = None
    picked_up_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    
    # Status
    status: OrderStatus = OrderStatus.PLACED
    
    # Delivery info
    delivery_address: Optional[Location] = None
    delivery_partner_id: Optional[str] = None
    estimated_delivery_time: Optional[datetime] = None
    
    # Financial
    subtotal: float = 0.0
    delivery_fee: float = 0.0
    taxes: float = 0.0
    discount: float = 0.0
    total_amount: float = 0.0
    
    # Special instructions and notes
    special_instructions: str = ""
    restaurant_notes: str = ""
    delivery_notes: str = ""
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = str(uuid.uuid4())
        
        if self.subtotal == 0.0:
            self.subtotal = sum(item.total_price for item in self.items)
    
    def calculate_total(self, delivery_fee: float = 0.0, tax_rate: float = 0.08, 
                       discount: float = 0.0) -> None:
        """Calculate order total."""
        self.delivery_fee = delivery_fee
        self.taxes = self.subtotal * tax_rate
        self.discount = discount
        self.total_amount = self.subtotal + self.delivery_fee + self.taxes - self.discount


@dataclass
class DeliveryPartner(User):
    """Delivery partner information."""
    vehicle_type: str = "bike"  # bike, scooter, car
    status: DeliveryPartnerStatus = DeliveryPartnerStatus.OFFLINE
    current_location: Optional[Location] = None
    
    # Ratings and performance
    rating: float = 5.0
    total_ratings: int = 0
    total_deliveries: int = 0
    total_earnings: float = 0.0
    
    # Operational metrics
    acceptance_rate: float = 1.0
    completion_rate: float = 1.0
    average_delivery_time: float = 30.0  # minutes
    
    # Current delivery
    current_order_id: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.user_type = UserType.DELIVERY_PARTNER
    
    def update_location(self, location: Location) -> None:
        """Update delivery partner's location."""
        self.current_location = location
        self.last_active = datetime.now()


@dataclass
class Delivery:
    """Delivery information."""
    delivery_id: str
    order_id: str
    delivery_partner_id: str
    
    # Locations
    pickup_location: Location
    delivery_location: Location
    
    # Timestamps
    assigned_at: datetime = field(default_factory=datetime.now)
    picked_up_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    
    # Tracking
    route_points: List[Location] = field(default_factory=list)
    estimated_delivery_time: Optional[datetime] = None
    actual_delivery_time: Optional[datetime] = None
    
    # Financial
    delivery_fee: float = 0.0
    partner_earnings: float = 0.0
    
    def __post_init__(self):
        if not self.delivery_id:
            self.delivery_id = str(uuid.uuid4())


# ============================================================================
# ORDER STATE MANAGEMENT
# ============================================================================

class OrderStateMachine:
    """Manage order state transitions."""
    
    def __init__(self, order: Order):
        self.order = order
        self.valid_transitions = {
            OrderStatus.PLACED: [OrderStatus.CONFIRMED, OrderStatus.CANCELLED],
            OrderStatus.CONFIRMED: [OrderStatus.PREPARING, OrderStatus.CANCELLED],
            OrderStatus.PREPARING: [OrderStatus.READY, OrderStatus.CANCELLED],
            OrderStatus.READY: [OrderStatus.PICKED_UP, OrderStatus.CANCELLED],
            OrderStatus.PICKED_UP: [OrderStatus.OUT_FOR_DELIVERY],
            OrderStatus.OUT_FOR_DELIVERY: [OrderStatus.DELIVERED, OrderStatus.CANCELLED],
            OrderStatus.DELIVERED: [OrderStatus.REFUNDED],
            OrderStatus.CANCELLED: [OrderStatus.REFUNDED],
            OrderStatus.REFUNDED: []  # Terminal state
        }
    
    def can_transition_to(self, new_status: OrderStatus) -> bool:
        """Check if transition is valid."""
        return new_status in self.valid_transitions.get(self.order.status, [])
    
    def transition_to(self, new_status: OrderStatus) -> bool:
        """Transition to new status."""
        if not self.can_transition_to(new_status):
            return False
        
        old_status = self.order.status
        self.order.status = new_status
        
        # Update timestamps
        now = datetime.now()
        
        if new_status == OrderStatus.CONFIRMED:
            self.order.confirmed_at = now
        elif new_status == OrderStatus.READY:
            self.order.prepared_at = now
        elif new_status == OrderStatus.PICKED_UP:
            self.order.picked_up_at = now
        elif new_status == OrderStatus.DELIVERED:
            self.order.delivered_at = now
        
        print(f"Order {self.order.order_id[:8]} transitioned: {old_status.value} → {new_status.value}")
        return True


# ============================================================================
# DELIVERY ASSIGNMENT STRATEGIES
# ============================================================================

class DeliveryAssignmentStrategy(ABC):
    """Abstract delivery assignment strategy."""
    
    @abstractmethod
    def assign_delivery_partner(self, order: Order, available_partners: List[DeliveryPartner],
                              restaurant_location: Location) -> Optional[DeliveryPartner]:
        """Assign delivery partner to order."""
        pass


class NearestPartnerStrategy(DeliveryAssignmentStrategy):
    """Assign to nearest available partner."""
    
    def assign_delivery_partner(self, order: Order, available_partners: List[DeliveryPartner],
                              restaurant_location: Location) -> Optional[DeliveryPartner]:
        """Find nearest partner to restaurant."""
        if not available_partners:
            return None
        
        nearest_partner = None
        min_distance = float('inf')
        
        for partner in available_partners:
            if not partner.current_location:
                continue
            
            distance = partner.current_location.distance_to(restaurant_location)
            
            if distance < min_distance:
                min_distance = distance
                nearest_partner = partner
        
        return nearest_partner


class BalancedAssignmentStrategy(DeliveryAssignmentStrategy):
    """Assign based on multiple factors."""
    
    def assign_delivery_partner(self, order: Order, available_partners: List[DeliveryPartner],
                              restaurant_location: Location) -> Optional[DeliveryPartner]:
        """Assign partner based on distance, rating, and performance."""
        if not available_partners:
            return None
        
        scored_partners = []
        
        for partner in available_partners:
            if not partner.current_location:
                continue
            
            score = self._calculate_partner_score(partner, restaurant_location, order.delivery_address)
            scored_partners.append((partner, score))
        
        if not scored_partners:
            return None
        
        # Sort by score (higher is better)
        scored_partners.sort(key=lambda x: x[1], reverse=True)
        return scored_partners[0][0]
    
    def _calculate_partner_score(self, partner: DeliveryPartner, restaurant_location: Location,
                               delivery_location: Optional[Location]) -> float:
        """Calculate partner assignment score."""
        score = 100.0  # Base score
        
        # Distance factor (closer to restaurant is better)
        distance_to_restaurant = partner.current_location.distance_to(restaurant_location)
        distance_score = max(0, 50 - distance_to_restaurant * 5)
        
        # Rating factor
        rating_score = partner.rating * 10
        
        # Performance factors
        acceptance_score = partner.acceptance_rate * 20
        completion_score = partner.completion_rate * 15
        
        # Experience factor
        experience_score = min(partner.total_deliveries * 0.1, 10)
        
        # Delivery time factor (faster is better)
        speed_score = max(0, 20 - partner.average_delivery_time * 0.5)
        
        total_score = (distance_score + rating_score + acceptance_score + 
                      completion_score + experience_score + speed_score)
        
        return max(0, total_score)


# ============================================================================
# MAIN FOOD DELIVERY SYSTEM
# ============================================================================

class FoodDeliverySystem:
    """Main food delivery system."""
    
    def __init__(self, city_name: str = "Metro City"):
        self.city_name = city_name
        
        # Data storage
        self.users: Dict[str, User] = {}
        self.customers: Dict[str, Customer] = {}
        self.restaurants: Dict[str, Restaurant] = {}
        self.delivery_partners: Dict[str, DeliveryPartner] = {}
        self.orders: Dict[str, Order] = {}
        self.deliveries: Dict[str, Delivery] = {}
        self.promotions: Dict[str, Promotion] = {}
        
        # Active carts
        self.carts: Dict[str, Cart] = {}  # customer_id -> Cart
        
        # System components
        self.assignment_strategy = BalancedAssignmentStrategy()
        
        # Analytics
        self.analytics = {
            'total_orders': 0,
            'completed_orders': 0,
            'cancelled_orders': 0,
            'total_revenue': 0.0,
            'total_deliveries': 0,
            'average_delivery_time': 0.0,
            'customer_satisfaction': 0.0
        }
        
        # Threading
        self._lock = threading.RLock()
        
        print(f"🍕 Food Delivery System initialized for {city_name}")
    
    def register_customer(self, email: str, phone: str, first_name: str, last_name: str) -> Customer:
        """Register a new customer."""
        customer = Customer(
            user_id=str(uuid.uuid4()),
            email=email,
            phone=phone,
            first_name=first_name,
            last_name=last_name
        )
        
        with self._lock:
            self.users[customer.user_id] = customer
            self.customers[customer.user_id] = customer
        
        return customer
    
    def register_restaurant(self, name: str, owner_id: str, location: Location,
                           cuisine_types: List[CuisineType], **kwargs) -> Restaurant:
        """Register a new restaurant."""
        restaurant = Restaurant(
            restaurant_id=str(uuid.uuid4()),
            name=name,
            owner_id=owner_id,
            location=location,
            cuisine_types=cuisine_types,
            **kwargs
        )
        
        with self._lock:
            self.restaurants[restaurant.restaurant_id] = restaurant
        
        return restaurant
    
    def register_delivery_partner(self, email: str, phone: str, first_name: str, 
                                last_name: str, vehicle_type: str = "bike") -> DeliveryPartner:
        """Register a new delivery partner."""
        partner = DeliveryPartner(
            user_id=str(uuid.uuid4()),
            email=email,
            phone=phone,
            first_name=first_name,
            last_name=last_name,
            vehicle_type=vehicle_type
        )
        
        with self._lock:
            self.users[partner.user_id] = partner
            self.delivery_partners[partner.user_id] = partner
        
        return partner
    
    def add_menu_item(self, restaurant_id: str, name: str, description: str, 
                     price: float, category: str, **kwargs) -> Optional[MenuItem]:
        """Add menu item to restaurant."""
        if restaurant_id not in self.restaurants:
            return None
        
        menu_item = MenuItem(
            item_id=str(uuid.uuid4()),
            name=name,
            description=description,
            price=price,
            category=category,
            **kwargs
        )
        
        restaurant = self.restaurants[restaurant_id]
        restaurant.add_menu_item(menu_item)
        
        return menu_item
    
    def search_restaurants(self, customer_location: Location, cuisine_type: CuisineType = None,
                          search_query: str = "", radius: float = 20.0) -> List[Restaurant]:
        """Search for restaurants."""
        matching_restaurants = []
        
        for restaurant in self.restaurants.values():
            # Check if restaurant can deliver to customer
            if not restaurant.can_deliver_to(customer_location):
                continue
            
            # Check if restaurant is open
            if not restaurant.is_open_now():
                continue
            
            # Filter by cuisine type
            if cuisine_type and cuisine_type not in restaurant.cuisine_types:
                continue
            
            # Search by name or description
            if search_query:
                query_lower = search_query.lower()
                if (query_lower not in restaurant.name.lower() and 
                    query_lower not in restaurant.description.lower()):
                    continue
            
            matching_restaurants.append(restaurant)
        
        # Sort by rating and distance
        matching_restaurants.sort(
            key=lambda r: (r.rating, -r.location.distance_to(customer_location)),
            reverse=True
        )
        
        return matching_restaurants
    
    def get_cart(self, customer_id: str, restaurant_id: str = None) -> Cart:
        """Get or create cart for customer."""
        if customer_id not in self.carts:
            if not restaurant_id:
                raise ValueError("Restaurant ID required for new cart")
            
            self.carts[customer_id] = Cart(
                customer_id=customer_id,
                restaurant_id=restaurant_id
            )
        
        cart = self.carts[customer_id]
        
        # If switching restaurants, clear cart
        if restaurant_id and cart.restaurant_id != restaurant_id:
            cart.clear()
            cart.restaurant_id = restaurant_id
        
        return cart
    
    def add_to_cart(self, customer_id: str, restaurant_id: str, item_id: str,
                   quantity: int = 1, **kwargs) -> bool:
        """Add item to customer's cart."""
        if restaurant_id not in self.restaurants:
            return False
        
        restaurant = self.restaurants[restaurant_id]
        
        if item_id not in restaurant.menu_items:
            return False
        
        menu_item = restaurant.menu_items[item_id]
        
        if not menu_item.is_available:
            return False
        
        cart = self.get_cart(customer_id, restaurant_id)
        cart.add_item(menu_item, quantity, **kwargs)
        
        return True
    
    def create_promotion(self, name: str, description: str, promotion_type: PromotionType,
                        **kwargs) -> Promotion:
        """Create a new promotion."""
        promotion = Promotion(
            promotion_id=str(uuid.uuid4()),
            name=name,
            description=description,
            promotion_type=promotion_type,
            **kwargs
        )
        
        with self._lock:
            self.promotions[promotion.promotion_id] = promotion
        
        return promotion
    
    def place_order(self, customer_id: str, delivery_address: Location,
                   promotion_code: str = None, special_instructions: str = "") -> Order:
        """Place an order."""
        if customer_id not in self.customers:
            raise ValueError("Customer not found")
        
        if customer_id not in self.carts:
            raise ValueError("No cart found")
        
        cart = self.carts[customer_id]
        
        if not cart.items:
            raise ValueError("Cart is empty")
        
        restaurant = self.restaurants[cart.restaurant_id]
        
        # Check minimum order amount
        if cart.subtotal < restaurant.minimum_order_amount:
            raise ValueError(f"Minimum order amount is ${restaurant.minimum_order_amount}")
        
        # Check restaurant capacity
        if restaurant.current_orders_count >= restaurant.max_orders_per_hour:
            raise ValueError("Restaurant is currently busy, please try later")
        
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            customer_id=customer_id,
            restaurant_id=cart.restaurant_id,
            items=cart.items.copy(),
            delivery_address=delivery_address,
            special_instructions=special_instructions
        )
        
        # Calculate pricing
        delivery_fee = restaurant.delivery_fee
        discount = 0.0
        
        # Apply promotion if provided
        if promotion_code:
            promotion = self._find_promotion_by_code(promotion_code)
            if promotion and promotion.is_valid():
                discount = promotion.calculate_discount(cart)
                promotion.uses_count += 1
        
        order.calculate_total(delivery_fee, discount=discount)
        
        # Estimate delivery time
        prep_time = max(item.menu_item.preparation_time for item in cart.items)
        delivery_distance = restaurant.location.distance_to(delivery_address)
        delivery_time = delivery_distance * 3  # Assume 3 minutes per km
        
        order.estimated_delivery_time = datetime.now() + timedelta(
            minutes=prep_time + delivery_time + 10  # 10 min buffer
        )
        
        with self._lock:
            self.orders[order.order_id] = order
            restaurant.current_orders_count += 1
            self.analytics['total_orders'] += 1
        
        # Clear cart
        cart.clear()
        
        # Start order processing
        self._process_order(order)
        
        return order
    
    def confirm_order(self, restaurant_id: str, order_id: str) -> bool:
        """Restaurant confirms order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.restaurant_id != restaurant_id:
            return False
        
        state_machine = OrderStateMachine(order)
        return state_machine.transition_to(OrderStatus.CONFIRMED)
    
    def update_order_status(self, order_id: str, new_status: OrderStatus,
                           user_id: str = None) -> bool:
        """Update order status."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        state_machine = OrderStateMachine(order)
        
        success = state_machine.transition_to(new_status)
        
        if success and new_status == OrderStatus.READY:
            # Assign delivery partner
            self._assign_delivery_partner(order)
        
        return success
    
    def set_delivery_partner_status(self, partner_id: str, status: DeliveryPartnerStatus) -> bool:
        """Set delivery partner status."""
        if partner_id not in self.delivery_partners:
            return False
        
        partner = self.delivery_partners[partner_id]
        partner.status = status
        
        return True
    
    def update_delivery_partner_location(self, partner_id: str, location: Location) -> bool:
        """Update delivery partner location."""
        if partner_id not in self.delivery_partners:
            return False
        
        partner = self.delivery_partners[partner_id]
        partner.update_location(location)
        
        return True
    
    def get_order_tracking(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order tracking information."""
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        restaurant = self.restaurants[order.restaurant_id]
        
        tracking_info = {
            'order_id': order.order_id,
            'status': order.status.value,
            'restaurant': {
                'name': restaurant.name,
                'location': restaurant.location
            },
            'estimated_delivery_time': order.estimated_delivery_time.isoformat() if order.estimated_delivery_time else None,
            'delivery_partner': None
        }
        
        # Add delivery partner info if assigned
        if order.delivery_partner_id:
            partner = self.delivery_partners.get(order.delivery_partner_id)
            if partner:
                tracking_info['delivery_partner'] = {
                    'name': partner.full_name,
                    'phone': partner.phone,
                    'vehicle_type': partner.vehicle_type,
                    'current_location': partner.current_location,
                    'rating': partner.rating
                }
        
        return tracking_info
    
    def rate_order(self, customer_id: str, order_id: str, restaurant_rating: int,
                  delivery_rating: int = None, feedback: str = "") -> bool:
        """Rate completed order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.customer_id != customer_id or order.status != OrderStatus.DELIVERED:
            return False
        
        # Update restaurant rating
        restaurant = self.restaurants[order.restaurant_id]
        total_rating_points = restaurant.rating * restaurant.total_reviews + restaurant_rating
        restaurant.total_reviews += 1
        restaurant.rating = total_rating_points / restaurant.total_reviews
        
        # Update delivery partner rating if provided
        if delivery_rating and order.delivery_partner_id:
            partner = self.delivery_partners[order.delivery_partner_id]
            total_rating_points = partner.rating * partner.total_ratings + delivery_rating
            partner.total_ratings += 1
            partner.rating = total_rating_points / partner.total_ratings
        
        return True
    
    def get_order_history(self, customer_id: str, limit: int = 20) -> List[Order]:
        """Get customer's order history."""
        customer_orders = []
        
        for order in self.orders.values():
            if order.customer_id == customer_id:
                customer_orders.append(order)
        
        # Sort by creation time, newest first
        customer_orders.sort(key=lambda o: o.created_at, reverse=True)
        
        return customer_orders[:limit]
    
    def get_restaurant_orders(self, restaurant_id: str, status: OrderStatus = None) -> List[Order]:
        """Get orders for a restaurant."""
        restaurant_orders = []
        
        for order in self.orders.values():
            if order.restaurant_id == restaurant_id:
                if status is None or order.status == status:
                    restaurant_orders.append(order)
        
        # Sort by creation time, newest first
        restaurant_orders.sort(key=lambda o: o.created_at, reverse=True)
        
        return restaurant_orders
    
    def _process_order(self, order: Order) -> None:
        """Process order automatically."""
        def auto_process():
            import time
            
            # Auto-confirm after 30 seconds
            time.sleep(0.5)
            
            state_machine = OrderStateMachine(order)
            state_machine.transition_to(OrderStatus.CONFIRMED)
            
            # Auto-start preparing
            time.sleep(0.5)
            state_machine.transition_to(OrderStatus.PREPARING)
            
            # Auto-ready after prep time
            prep_time = max(item.menu_item.preparation_time for item in order.items)
            time.sleep(prep_time / 30)  # Speed up for demo
            
            state_machine.transition_to(OrderStatus.READY)
        
        threading.Thread(target=auto_process, daemon=True).start()
    
    def _assign_delivery_partner(self, order: Order) -> bool:
        """Assign delivery partner to order."""
        restaurant = self.restaurants[order.restaurant_id]
        
        # Get available delivery partners
        available_partners = [
            partner for partner in self.delivery_partners.values()
            if partner.status == DeliveryPartnerStatus.AVAILABLE and partner.current_location
        ]
        
        if not available_partners:
            print(f"No available delivery partners for order {order.order_id[:8]}")
            return False
        
        # Use assignment strategy
        selected_partner = self.assignment_strategy.assign_delivery_partner(
            order, available_partners, restaurant.location
        )
        
        if selected_partner:
            # Assign partner to order
            order.delivery_partner_id = selected_partner.user_id
            selected_partner.status = DeliveryPartnerStatus.BUSY
            selected_partner.current_order_id = order.order_id
            
            # Create delivery record
            delivery = Delivery(
                delivery_id=str(uuid.uuid4()),
                order_id=order.order_id,
                delivery_partner_id=selected_partner.user_id,
                pickup_location=restaurant.location,
                delivery_location=order.delivery_address,
                delivery_fee=order.delivery_fee,
                partner_earnings=order.delivery_fee * 0.8  # 80% to partner
            )
            
            with self._lock:
                self.deliveries[delivery.delivery_id] = delivery
            
            # Auto-simulate delivery
            self._simulate_delivery(order, delivery, selected_partner)
            
            print(f"Delivery partner {selected_partner.full_name} assigned to order {order.order_id[:8]}")
            return True
        
        return False
    
    def _simulate_delivery(self, order: Order, delivery: Delivery, partner: DeliveryPartner) -> None:
        """Simulate delivery process."""
        def deliver():
            import time
            
            # Simulate pickup
            time.sleep(1)
            
            state_machine = OrderStateMachine(order)
            state_machine.transition_to(OrderStatus.PICKED_UP)
            delivery.picked_up_at = datetime.now()
            
            # Simulate delivery
            time.sleep(1)
            state_machine.transition_to(OrderStatus.OUT_FOR_DELIVERY)
            
            # Simulate arrival
            time.sleep(2)
            
            # Complete delivery
            state_machine.transition_to(OrderStatus.DELIVERED)
            delivery.delivered_at = datetime.now()
            
            # Update partner status
            partner.status = DeliveryPartnerStatus.AVAILABLE
            partner.current_order_id = None
            partner.total_deliveries += 1
            partner.total_earnings += delivery.partner_earnings
            
            # Update restaurant
            restaurant = self.restaurants[order.restaurant_id]
            restaurant.current_orders_count = max(0, restaurant.current_orders_count - 1)
            restaurant.total_orders += 1
            restaurant.total_revenue += order.total_amount
            
            # Update analytics
            with self._lock:
                self.analytics['completed_orders'] += 1
                self.analytics['total_revenue'] += order.total_amount
                self.analytics['total_deliveries'] += 1
            
            print(f"Order {order.order_id[:8]} delivered by {partner.full_name}")
        
        threading.Thread(target=deliver, daemon=True).start()
    
    def _find_promotion_by_code(self, code: str) -> Optional[Promotion]:
        """Find promotion by code (simplified)."""
        for promotion in self.promotions.values():
            if promotion.name.lower() == code.lower():
                return promotion
        return None
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics."""
        with self._lock:
            # Calculate additional metrics
            if self.analytics['completed_orders'] > 0:
                avg_order_value = self.analytics['total_revenue'] / self.analytics['completed_orders']
            else:
                avg_order_value = 0.0
            
            completion_rate = 0.0
            if self.analytics['total_orders'] > 0:
                completion_rate = (self.analytics['completed_orders'] / self.analytics['total_orders']) * 100
            
            return {
                **self.analytics,
                'total_customers': len(self.customers),
                'total_restaurants': len(self.restaurants),
                'total_delivery_partners': len(self.delivery_partners),
                'active_restaurants': len([r for r in self.restaurants.values() if r.is_open]),
                'available_partners': len([p for p in self.delivery_partners.values() 
                                         if p.status == DeliveryPartnerStatus.AVAILABLE]),
                'busy_partners': len([p for p in self.delivery_partners.values() 
                                    if p.status == DeliveryPartnerStatus.BUSY]),
                'average_order_value': avg_order_value,
                'completion_rate': completion_rate,
                'active_orders': len([o for o in self.orders.values() 
                                    if o.status not in [OrderStatus.DELIVERED, OrderStatus.CANCELLED]])
            }


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_food_delivery_system():
    """Demonstrate the food delivery system."""
    print("=== FOOD DELIVERY SYSTEM DEMONSTRATION ===\n")
    
    # Initialize system
    print("1. SYSTEM INITIALIZATION:")
    
    system = FoodDeliverySystem("Food City")
    print("   ✓ Food delivery system initialized")
    print()
    
    # Register customers
    print("2. CUSTOMER REGISTRATION:")
    
    customers = []
    customer_data = [
        ("alice@example.com", "+1234567890", "Alice", "Johnson"),
        ("bob@example.com", "+1234567891", "Bob", "Smith"),
        ("charlie@example.com", "+1234567892", "Charlie", "Brown")
    ]
    
    for email, phone, first_name, last_name in customer_data:
        customer = system.register_customer(email, phone, first_name, last_name)
        customers.append(customer)
        print(f"   ✓ Registered customer: {customer.full_name}")
    
    print()
    
    # Register restaurants
    print("3. RESTAURANT REGISTRATION:")
    
    restaurants = []
    restaurant_data = [
        ("Mario's Pizza", "owner1", Location(40.7128, -74.0060, "123 Main St", "Food City"), 
         [CuisineType.ITALIAN], {"phone": "+1111111111", "description": "Authentic Italian pizza"}),
        ("Dragon Palace", "owner2", Location(40.7589, -73.9851, "456 Oak Ave", "Food City"),
         [CuisineType.CHINESE], {"phone": "+1111111112", "description": "Traditional Chinese cuisine"}),
        ("Spice Garden", "owner3", Location(40.6782, -73.9442, "789 Pine Rd", "Food City"),
         [CuisineType.INDIAN], {"phone": "+1111111113", "description": "Spicy Indian dishes"}),
        ("Burger Hub", "owner4", Location(40.7831, -73.9712, "321 Elm St", "Food City"),
         [CuisineType.FAST_FOOD, CuisineType.AMERICAN], {"phone": "+1111111114", "description": "Gourmet burgers"})
    ]
    
    for name, owner_id, location, cuisines, kwargs in restaurant_data:
        restaurant = system.register_restaurant(name, owner_id, location, cuisines, **kwargs)
        restaurants.append(restaurant)
        print(f"   ✓ Registered restaurant: {name} ({', '.join(c.value for c in cuisines)})")
    
    print()
    
    # Add menu items
    print("4. MENU SETUP:")
    
    # Mario's Pizza menu
    pizza_items = [
        ("Margherita Pizza", "Classic pizza with tomato, mozzarella, and basil", 12.99, "Pizza"),
        ("Pepperoni Pizza", "Pizza with pepperoni and cheese", 14.99, "Pizza"),
        ("Caesar Salad", "Fresh romaine lettuce with Caesar dressing", 8.99, "Salad"),
        ("Garlic Bread", "Toasted bread with garlic and herbs", 4.99, "Appetizer")
    ]
    
    for name, desc, price, category in pizza_items:
        item = system.add_menu_item(restaurants[0].restaurant_id, name, desc, price, category)
        print(f"   ✓ Added to Mario's Pizza: {name} - ${price}")
    
    # Dragon Palace menu
    chinese_items = [
        ("Sweet and Sour Chicken", "Crispy chicken with sweet and sour sauce", 13.99, "Main Course"),
        ("Beef Lo Mein", "Noodles with beef and vegetables", 12.99, "Main Course"),
        ("Spring Rolls", "Crispy vegetable spring rolls", 6.99, "Appetizer"),
        ("Fried Rice", "Wok-fried rice with eggs and vegetables", 9.99, "Rice")
    ]
    
    for name, desc, price, category in chinese_items:
        system.add_menu_item(restaurants[1].restaurant_id, name, desc, price, category)
        print(f"   ✓ Added to Dragon Palace: {name} - ${price}")
    
    print()
    
    # Register delivery partners
    print("5. DELIVERY PARTNER REGISTRATION:")
    
    partners = []
    partner_data = [
        ("driver1@example.com", "+1234567893", "David", "Wilson", "bike"),
        ("driver2@example.com", "+1234567894", "Emma", "Davis", "scooter"),
        ("driver3@example.com", "+1234567895", "Frank", "Miller", "car")
    ]
    
    for email, phone, first_name, last_name, vehicle in partner_data:
        partner = system.register_delivery_partner(email, phone, first_name, last_name, vehicle)
        partners.append(partner)
        print(f"   ✓ Registered delivery partner: {partner.full_name} ({vehicle})")
    
    # Set partners online and update locations
    locations = [
        Location(40.7200, -74.0000, "Area 1", "Food City"),
        Location(40.7500, -73.9800, "Area 2", "Food City"),
        Location(40.6800, -73.9500, "Area 3", "Food City")
    ]
    
    for i, partner in enumerate(partners):
        system.set_delivery_partner_status(partner.user_id, DeliveryPartnerStatus.AVAILABLE)
        system.update_delivery_partner_location(partner.user_id, locations[i])
        print(f"   ✓ {partner.full_name} online at {locations[i].address}")
    
    print()
    
    # Test restaurant search
    print("6. RESTAURANT SEARCH:")
    
    customer_location = Location(40.7300, -74.0100, "Customer Area", "Food City")
    
    # Search all restaurants
    all_restaurants = system.search_restaurants(customer_location)
    print(f"   All available restaurants ({len(all_restaurants)}):")
    for restaurant in all_restaurants:
        distance = restaurant.location.distance_to(customer_location)
        print(f"     {restaurant.name}: {distance:.1f}km away, rating: {restaurant.rating:.1f}⭐")
    
    # Search by cuisine
    italian_restaurants = system.search_restaurants(customer_location, CuisineType.ITALIAN)
    print(f"\n   Italian restaurants ({len(italian_restaurants)}):")
    for restaurant in italian_restaurants:
        print(f"     {restaurant.name}")
    
    print()
    
    # Test cart and ordering
    print("7. CART AND ORDERING:")
    
    alice = customers[0]
    mario_pizza = restaurants[0]
    
    print(f"   {alice.full_name} ordering from {mario_pizza.name}:")
    
    # Add items to cart
    pizza_item = list(mario_pizza.menu_items.values())[0]  # Margherita Pizza
    salad_item = list(mario_pizza.menu_items.values())[2]  # Caesar Salad
    
    system.add_to_cart(alice.user_id, mario_pizza.restaurant_id, pizza_item.item_id, 2)
    system.add_to_cart(alice.user_id, mario_pizza.restaurant_id, salad_item.item_id, 1)
    
    cart = system.get_cart(alice.user_id)
    print(f"     Added 2x {pizza_item.name}, 1x {salad_item.name}")
    print(f"     Cart total: ${cart.subtotal:.2f} ({cart.total_items} items)")
    
    # Place order
    delivery_address = Location(40.7300, -74.0100, "Alice's Home", "Food City")
    
    order = system.place_order(
        alice.user_id,
        delivery_address,
        special_instructions="Please ring the doorbell"
    )
    
    print(f"   ✓ Order placed: {order.order_id[:8]} - Total: ${order.total_amount:.2f}")
    print(f"   ✓ Estimated delivery: {order.estimated_delivery_time.strftime('%H:%M')}")
    
    print()
    
    # Test promotions
    print("8. PROMOTIONS:")
    
    # Create promotions
    promo1 = system.create_promotion(
        "SAVE20",
        "20% off orders over $15",
        PromotionType.PERCENTAGE_DISCOUNT,
        discount_percentage=20.0,
        minimum_order_amount=15.0
    )
    
    promo2 = system.create_promotion(
        "FREEDEL",
        "Free delivery on any order",
        PromotionType.FREE_DELIVERY
    )
    
    print(f"   ✓ Created promotion: {promo1.name} - {promo1.description}")
    print(f"   ✓ Created promotion: {promo2.name} - {promo2.description}")
    
    # Bob places order with promotion
    bob = customers[1]
    dragon_palace = restaurants[1]
    
    # Add items to Bob's cart
    chinese_item = list(dragon_palace.menu_items.values())[0]  # Sweet and Sour Chicken
    rice_item = list(dragon_palace.menu_items.values())[3]     # Fried Rice
    
    system.add_to_cart(bob.user_id, dragon_palace.restaurant_id, chinese_item.item_id, 1)
    system.add_to_cart(bob.user_id, dragon_palace.restaurant_id, rice_item.item_id, 1)
    
    bob_order = system.place_order(
        bob.user_id,
        Location(40.7400, -73.9900, "Bob's Office", "Food City"),
        promotion_code="SAVE20"
    )
    
    print(f"   ✓ Bob's order with SAVE20: ${bob_order.total_amount:.2f} (discount: ${bob_order.discount:.2f})")
    
    print()
    
    # Test order tracking
    print("9. ORDER TRACKING:")
    
    # Wait for orders to process
    import time
    time.sleep(3)
    
    for order_id in [order.order_id, bob_order.order_id]:
        tracking = system.get_order_tracking(order_id)
        if tracking:
            print(f"   Order {order_id[:8]}:")
            print(f"     Status: {tracking['status']}")
            print(f"     Restaurant: {tracking['restaurant']['name']}")
            
            if tracking['delivery_partner']:
                partner = tracking['delivery_partner']
                print(f"     Delivery partner: {partner['name']} ({partner['vehicle_type']})")
    
    print()
    
    # Test order completion and rating
    print("10. ORDER COMPLETION AND RATING:")
    
    # Wait for delivery completion
    time.sleep(5)
    
    # Check order statuses
    for order_id, customer_name in [(order.order_id, alice.full_name), (bob_order.order_id, bob.full_name)]:
        current_order = system.orders[order_id]
        print(f"   {customer_name}'s order: {current_order.status.value}")
        
        if current_order.status == OrderStatus.DELIVERED:
            # Rate the order
            restaurant_rating = random.randint(4, 5)
            delivery_rating = random.randint(4, 5)
            
            system.rate_order(
                current_order.customer_id,
                order_id,
                restaurant_rating,
                delivery_rating,
                "Great food and fast delivery!"
            )
            
            print(f"     Rated: Restaurant {restaurant_rating}⭐, Delivery {delivery_rating}⭐")
    
    print()
    
    # Test order history
    print("11. ORDER HISTORY:")
    
    alice_history = system.get_order_history(alice.user_id)
    print(f"   {alice.full_name}'s order history ({len(alice_history)} orders):")
    
    for hist_order in alice_history:
        restaurant = system.restaurants[hist_order.restaurant_id]
        print(f"     {hist_order.created_at.strftime('%Y-%m-%d %H:%M')} - "
              f"{restaurant.name} - ${hist_order.total_amount:.2f} - {hist_order.status.value}")
    
    print()
    
    # Test restaurant dashboard
    print("12. RESTAURANT DASHBOARD:")
    
    for restaurant in restaurants[:2]:  # Show first 2 restaurants
        rest_orders = system.get_restaurant_orders(restaurant.restaurant_id)
        print(f"   {restaurant.name}:")
        print(f"     Total orders: {restaurant.total_orders}")
        print(f"     Total revenue: ${restaurant.total_revenue:.2f}")
        print(f"     Rating: {restaurant.rating:.1f}⭐ ({restaurant.total_reviews} reviews)")
        print(f"     Current orders: {restaurant.current_orders_count}")
        print(f"     Recent orders: {len(rest_orders)}")
    
    print()
    
    # Show delivery partner performance
    print("13. DELIVERY PARTNER PERFORMANCE:")
    
    for partner in partners:
        print(f"   {partner.full_name}:")
        print(f"     Status: {partner.status.value}")
        print(f"     Total deliveries: {partner.total_deliveries}")
        print(f"     Total earnings: ${partner.total_earnings:.2f}")
        print(f"     Rating: {partner.rating:.1f}⭐ ({partner.total_ratings} reviews)")
        print(f"     Acceptance rate: {partner.acceptance_rate:.1%}")
    
    print()
    
    # Show comprehensive analytics
    print("14. SYSTEM ANALYTICS:")
    
    analytics = system.get_system_analytics()
    
    print(f"   Business Metrics:")
    print(f"     Total customers: {analytics['total_customers']}")
    print(f"     Total restaurants: {analytics['total_restaurants']}")
    print(f"     Total delivery partners: {analytics['total_delivery_partners']}")
    
    print(f"\n   Order Metrics:")
    print(f"     Total orders: {analytics['total_orders']}")
    print(f"     Completed orders: {analytics['completed_orders']}")
    print(f"     Cancelled orders: {analytics['cancelled_orders']}")
    print(f"     Completion rate: {analytics['completion_rate']:.1f}%")
    print(f"     Average order value: ${analytics['average_order_value']:.2f}")
    
    print(f"\n   Operational Metrics:")
    print(f"     Active restaurants: {analytics['active_restaurants']}")
    print(f"     Available partners: {analytics['available_partners']}")
    print(f"     Busy partners: {analytics['busy_partners']}")
    print(f"     Active orders: {analytics['active_orders']}")
    
    print(f"\n   Financial Metrics:")
    print(f"     Total revenue: ${analytics['total_revenue']:.2f}")
    print(f"     Total deliveries: {analytics['total_deliveries']}")
    
    print()
    
    # Show customer preferences and recommendations
    print("15. CUSTOMER INSIGHTS:")
    
    for customer in customers:
        if customer.total_orders > 0:
            avg_order_value = customer.total_spent / customer.total_orders
            print(f"   {customer.full_name}:")
            print(f"     Total orders: {customer.total_orders}")
            print(f"     Total spent: ${customer.total_spent:.2f}")
            print(f"     Average order value: ${avg_order_value:.2f}")
            print(f"     Favorite restaurants: {len(customer.favorite_restaurants)}")
    
    print()
    
    # Show final system state
    print("16. FINAL SYSTEM STATE:")
    
    final_analytics = system.get_system_analytics()
    
    print(f"   Platform Overview:")
    print(f"     Users: {final_analytics['total_customers']} customers, "
           f"{final_analytics['total_restaurants']} restaurants, "
           f"{final_analytics['total_delivery_partners']} partners")
    
    print(f"   Current Activity:")
    print(f"     Active orders: {final_analytics['active_orders']}")
    print(f"     Available delivery partners: {final_analytics['available_partners']}")
    print(f"     Open restaurants: {final_analytics['active_restaurants']}")
    
    print(f"   Performance:")
    print(f"     Order completion rate: {final_analytics['completion_rate']:.1f}%")
    print(f"     Platform revenue: ${final_analytics['total_revenue']:.2f}")
    
    # Show top performing restaurants
    top_restaurants = sorted(restaurants, key=lambda r: r.total_revenue, reverse=True)[:3]
    print(f"\n   Top Performing Restaurants:")
    for i, restaurant in enumerate(top_restaurants):
        print(f"     {i+1}. {restaurant.name}: ${restaurant.total_revenue:.2f} revenue, "
              f"{restaurant.rating:.1f}⭐ rating")
    
    print()
    print("=== FOOD DELIVERY SYSTEM DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_food_delivery_system()
