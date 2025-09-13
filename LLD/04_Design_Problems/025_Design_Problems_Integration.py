"""
DESIGN PROBLEMS INTEGRATION - Complete System Integration
======================================================

This file demonstrates how multiple design patterns and systems work together
in a comprehensive application. It integrates various design problems into
a unified e-commerce platform that showcases:

- User Management System
- Product Catalog with Search
- Shopping Cart and Order Processing
- Payment Processing
- Inventory Management
- Notification System
- Logging and Monitoring
- Caching Layer
- Rate Limiting
- File Management
- Analytics and Reporting

Key Integration Patterns:
- Facade: Simplified system interfaces
- Mediator: Component communication
- Observer: Event-driven architecture
- Strategy: Pluggable algorithms
- Factory: Object creation
- Singleton: Shared resources
- Decorator: Feature enhancement
- Command: Operation encapsulation
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import time
import json
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib


# ============================================================================
# SHARED ENUMS AND INTERFACES
# ============================================================================

class EventType(Enum):
    USER_REGISTERED = "user_registered"
    USER_LOGIN = "user_login"
    PRODUCT_VIEWED = "product_viewed"
    CART_UPDATED = "cart_updated"
    ORDER_PLACED = "order_placed"
    PAYMENT_PROCESSED = "payment_processed"
    ORDER_SHIPPED = "order_shipped"
    INVENTORY_LOW = "inventory_low"
    SYSTEM_ERROR = "system_error"


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"


# ============================================================================
# EVENT SYSTEM (OBSERVER PATTERN)
# ============================================================================

@dataclass
class Event:
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class EventListener(ABC):
    """Abstract event listener."""
    
    @abstractmethod
    def handle_event(self, event: Event) -> None:
        """Handle an event."""
        pass


class EventBus:
    """Central event bus for system-wide communication."""
    
    def __init__(self):
        self.listeners: Dict[EventType, List[EventListener]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def subscribe(self, event_type: EventType, listener: EventListener) -> None:
        """Subscribe to an event type."""
        with self._lock:
            self.listeners[event_type].append(listener)
    
    def publish(self, event: Event) -> None:
        """Publish an event to all listeners."""
        with self._lock:
            listeners = self.listeners.get(event.event_type, [])
            
        for listener in listeners:
            try:
                listener.handle_event(event)
            except Exception as e:
                print(f"Error in event listener: {e}")


# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class Logger:
    """Centralized logging system."""
    
    def __init__(self):
        self.logs = deque(maxlen=10000)
        self._lock = threading.RLock()
    
    def log(self, level: LogLevel, message: str, **context) -> None:
        """Log a message."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level.value,
            'message': message,
            'context': context
        }
        
        with self._lock:
            self.logs.append(log_entry)
        
        # Print to console for demo
        print(f"[{level.value.upper()}] {message}")
    
    def get_logs(self, level: LogLevel = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent logs."""
        with self._lock:
            logs = list(self.logs)
        
        if level:
            logs = [log for log in logs if log['level'] == level.value]
        
        return logs[-limit:]


# ============================================================================
# CACHING SYSTEM
# ============================================================================

class CacheEntry:
    """Cache entry with TTL support."""
    
    def __init__(self, value: Any, ttl: int = 300):
        self.value = value
        self.created_at = datetime.now()
        self.ttl = ttl
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)


class Cache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.data: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self.data.get(key)
            
            if entry is None:
                self.misses += 1
                return None
            
            if entry.is_expired:
                del self.data[key]
                self.misses += 1
                return None
            
            self.hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set value in cache."""
        with self._lock:
            # Evict if at capacity
            if len(self.data) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.data.keys(), key=lambda k: self.data[k].created_at)
                del self.data[oldest_key]
            
            self.data[key] = CacheEntry(value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self.data:
                del self.data[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.data.clear()
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / max(1, total)


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, capacity: int, refill_rate: int):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = datetime.now()
        self._lock = threading.RLock()
    
    def allow_request(self, tokens_required: int = 1) -> bool:
        """Check if request is allowed."""
        with self._lock:
            self._refill_tokens()
            
            if self.tokens >= tokens_required:
                self.tokens -= tokens_required
                return True
            
            return False
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on time elapsed."""
        now = datetime.now()
        elapsed = (now - self.last_refill).total_seconds()
        
        tokens_to_add = int(elapsed * self.refill_rate)
        if tokens_to_add > 0:
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now


# ============================================================================
# USER MANAGEMENT
# ============================================================================

@dataclass
class User:
    user_id: str
    email: str
    name: str
    phone: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    preferences: Dict[str, Any] = field(default_factory=dict)


class UserService:
    """User management service."""
    
    def __init__(self, event_bus: EventBus, logger: Logger, cache: Cache):
        self.users: Dict[str, User] = {}
        self.email_to_id: Dict[str, str] = {}
        self.event_bus = event_bus
        self.logger = logger
        self.cache = cache
        self._lock = threading.RLock()
    
    def create_user(self, email: str, name: str, phone: str = "") -> User:
        """Create a new user."""
        user = User(
            user_id=str(uuid.uuid4()),
            email=email,
            name=name,
            phone=phone
        )
        
        with self._lock:
            self.users[user.user_id] = user
            self.email_to_id[email] = user.user_id
        
        # Cache user
        self.cache.set(f"user:{user.user_id}", user, ttl=600)
        
        # Log and publish event
        self.logger.log(LogLevel.INFO, f"User created: {email}")
        self.event_bus.publish(Event(
            event_type=EventType.USER_REGISTERED,
            data={'user_id': user.user_id, 'email': email, 'name': name}
        ))
        
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        # Try cache first
        cached_user = self.cache.get(f"user:{user_id}")
        if cached_user:
            return cached_user
        
        # Get from storage
        user = self.users.get(user_id)
        if user:
            self.cache.set(f"user:{user_id}", user, ttl=600)
        
        return user
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user (simplified)."""
        user_id = self.email_to_id.get(email)
        if user_id:
            user = self.get_user(user_id)
            if user and user.is_active:
                # Log login
                self.logger.log(LogLevel.INFO, f"User login: {email}")
                self.event_bus.publish(Event(
                    event_type=EventType.USER_LOGIN,
                    data={'user_id': user_id, 'email': email}
                ))
                return user
        
        return None


# ============================================================================
# PRODUCT CATALOG
# ============================================================================

@dataclass
class Product:
    product_id: str
    name: str
    description: str
    price: float
    category: str
    sku: str
    inventory_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True


class ProductService:
    """Product catalog service."""
    
    def __init__(self, event_bus: EventBus, logger: Logger, cache: Cache):
        self.products: Dict[str, Product] = {}
        self.sku_to_id: Dict[str, str] = {}
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
        self.event_bus = event_bus
        self.logger = logger
        self.cache = cache
        self._lock = threading.RLock()
    
    def add_product(self, name: str, description: str, price: float,
                   category: str, sku: str, inventory_count: int = 0) -> Product:
        """Add a new product."""
        product = Product(
            product_id=str(uuid.uuid4()),
            name=name,
            description=description,
            price=price,
            category=category,
            sku=sku,
            inventory_count=inventory_count
        )
        
        with self._lock:
            self.products[product.product_id] = product
            self.sku_to_id[sku] = product.product_id
            self.category_index[category].add(product.product_id)
        
        # Cache product
        self.cache.set(f"product:{product.product_id}", product, ttl=1800)
        
        self.logger.log(LogLevel.INFO, f"Product added: {name} (SKU: {sku})")
        
        return product
    
    def get_product(self, product_id: str) -> Optional[Product]:
        """Get product by ID."""
        # Try cache first
        cached_product = self.cache.get(f"product:{product_id}")
        if cached_product:
            return cached_product
        
        # Get from storage
        product = self.products.get(product_id)
        if product:
            self.cache.set(f"product:{product_id}", product, ttl=1800)
        
        return product
    
    def search_products(self, query: str = "", category: str = "",
                       limit: int = 20) -> List[Product]:
        """Search products."""
        results = []
        
        if category:
            # Search by category
            product_ids = self.category_index.get(category, set())
            for product_id in list(product_ids)[:limit]:
                product = self.get_product(product_id)
                if product and product.is_active:
                    results.append(product)
        
        elif query:
            # Simple text search
            query_lower = query.lower()
            for product in self.products.values():
                if (product.is_active and 
                    (query_lower in product.name.lower() or 
                     query_lower in product.description.lower())):
                    results.append(product)
                    if len(results) >= limit:
                        break
        
        else:
            # Return all active products
            for product in list(self.products.values())[:limit]:
                if product.is_active:
                    results.append(product)
        
        return results
    
    def update_inventory(self, product_id: str, quantity_change: int) -> bool:
        """Update product inventory."""
        product = self.get_product(product_id)
        if not product:
            return False
        
        new_count = product.inventory_count + quantity_change
        if new_count < 0:
            return False
        
        with self._lock:
            product.inventory_count = new_count
        
        # Update cache
        self.cache.set(f"product:{product_id}", product, ttl=1800)
        
        # Check for low inventory
        if new_count <= 5:
            self.event_bus.publish(Event(
                event_type=EventType.INVENTORY_LOW,
                data={'product_id': product_id, 'current_count': new_count}
            ))
        
        return True


# ============================================================================
# SHOPPING CART
# ============================================================================

@dataclass
class CartItem:
    product_id: str
    quantity: int
    price_at_time: float


class ShoppingCart:
    """Shopping cart implementation."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.items: Dict[str, CartItem] = {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_item(self, product_id: str, quantity: int, price: float) -> None:
        """Add item to cart."""
        if product_id in self.items:
            self.items[product_id].quantity += quantity
        else:
            self.items[product_id] = CartItem(
                product_id=product_id,
                quantity=quantity,
                price_at_time=price
            )
        
        self.updated_at = datetime.now()
    
    def remove_item(self, product_id: str) -> bool:
        """Remove item from cart."""
        if product_id in self.items:
            del self.items[product_id]
            self.updated_at = datetime.now()
            return True
        return False
    
    def update_quantity(self, product_id: str, quantity: int) -> bool:
        """Update item quantity."""
        if product_id in self.items:
            if quantity <= 0:
                return self.remove_item(product_id)
            else:
                self.items[product_id].quantity = quantity
                self.updated_at = datetime.now()
                return True
        return False
    
    def get_total(self) -> float:
        """Calculate cart total."""
        return sum(item.quantity * item.price_at_time for item in self.items.values())
    
    def get_item_count(self) -> int:
        """Get total item count."""
        return sum(item.quantity for item in self.items.values())


class CartService:
    """Shopping cart service."""
    
    def __init__(self, product_service: ProductService, event_bus: EventBus,
                 logger: Logger, cache: Cache):
        self.carts: Dict[str, ShoppingCart] = {}
        self.product_service = product_service
        self.event_bus = event_bus
        self.logger = logger
        self.cache = cache
        self._lock = threading.RLock()
    
    def get_cart(self, user_id: str) -> ShoppingCart:
        """Get or create user's cart."""
        # Try cache first
        cached_cart = self.cache.get(f"cart:{user_id}")
        if cached_cart:
            return cached_cart
        
        # Get or create cart
        if user_id not in self.carts:
            self.carts[user_id] = ShoppingCart(user_id)
        
        cart = self.carts[user_id]
        self.cache.set(f"cart:{user_id}", cart, ttl=1800)
        
        return cart
    
    def add_to_cart(self, user_id: str, product_id: str, quantity: int) -> bool:
        """Add product to cart."""
        product = self.product_service.get_product(product_id)
        if not product or not product.is_active:
            return False
        
        # Check inventory
        if product.inventory_count < quantity:
            self.logger.log(LogLevel.WARNING, 
                          f"Insufficient inventory for product {product_id}")
            return False
        
        cart = self.get_cart(user_id)
        cart.add_item(product_id, quantity, product.price)
        
        # Update cache
        self.cache.set(f"cart:{user_id}", cart, ttl=1800)
        
        # Publish event
        self.event_bus.publish(Event(
            event_type=EventType.CART_UPDATED,
            data={
                'user_id': user_id,
                'product_id': product_id,
                'quantity': quantity,
                'action': 'add'
            }
        ))
        
        self.logger.log(LogLevel.INFO, 
                       f"Added {quantity} of {product_id} to cart for user {user_id}")
        
        return True


# ============================================================================
# ORDER PROCESSING
# ============================================================================

@dataclass
class Order:
    order_id: str
    user_id: str
    items: List[CartItem]
    total_amount: float
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    shipping_address: str = ""
    payment_method: str = ""


class OrderService:
    """Order processing service."""
    
    def __init__(self, product_service: ProductService, cart_service: CartService,
                 event_bus: EventBus, logger: Logger):
        self.orders: Dict[str, Order] = {}
        self.product_service = product_service
        self.cart_service = cart_service
        self.event_bus = event_bus
        self.logger = logger
        self._lock = threading.RLock()
    
    def place_order(self, user_id: str, shipping_address: str,
                   payment_method: str) -> Optional[Order]:
        """Place an order from user's cart."""
        cart = self.cart_service.get_cart(user_id)
        
        if not cart.items:
            self.logger.log(LogLevel.WARNING, f"Empty cart for user {user_id}")
            return None
        
        # Validate inventory
        for item in cart.items.values():
            product = self.product_service.get_product(item.product_id)
            if not product or product.inventory_count < item.quantity:
                self.logger.log(LogLevel.ERROR, 
                              f"Insufficient inventory for product {item.product_id}")
                return None
        
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            user_id=user_id,
            items=list(cart.items.values()),
            total_amount=cart.get_total(),
            shipping_address=shipping_address,
            payment_method=payment_method
        )
        
        with self._lock:
            self.orders[order.order_id] = order
        
        # Update inventory
        for item in cart.items.values():
            self.product_service.update_inventory(item.product_id, -item.quantity)
        
        # Clear cart
        cart.items.clear()
        self.cart_service.cache.delete(f"cart:{user_id}")
        
        # Publish event
        self.event_bus.publish(Event(
            event_type=EventType.ORDER_PLACED,
            data={
                'order_id': order.order_id,
                'user_id': user_id,
                'total_amount': order.total_amount,
                'item_count': len(order.items)
            }
        ))
        
        self.logger.log(LogLevel.INFO, 
                       f"Order placed: {order.order_id} for user {user_id}")
        
        return order


# ============================================================================
# NOTIFICATION SERVICE
# ============================================================================

class NotificationService(EventListener):
    """Notification service that responds to events."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.sent_notifications = []
    
    def handle_event(self, event: Event) -> None:
        """Handle events and send appropriate notifications."""
        if event.event_type == EventType.USER_REGISTERED:
            self._send_welcome_notification(event.data)
        
        elif event.event_type == EventType.ORDER_PLACED:
            self._send_order_confirmation(event.data)
        
        elif event.event_type == EventType.INVENTORY_LOW:
            self._send_inventory_alert(event.data)
    
    def _send_welcome_notification(self, data: Dict[str, Any]) -> None:
        """Send welcome notification to new user."""
        notification = {
            'type': 'welcome',
            'user_id': data['user_id'],
            'channel': NotificationChannel.EMAIL.value,
            'subject': f"Welcome {data['name']}!",
            'message': f"Welcome to our platform, {data['name']}!"
        }
        
        self.sent_notifications.append(notification)
        self.logger.log(LogLevel.INFO, f"Sent welcome notification to {data['email']}")
    
    def _send_order_confirmation(self, data: Dict[str, Any]) -> None:
        """Send order confirmation."""
        notification = {
            'type': 'order_confirmation',
            'user_id': data['user_id'],
            'channel': NotificationChannel.EMAIL.value,
            'subject': f"Order Confirmation #{data['order_id'][:8]}",
            'message': f"Your order for ${data['total_amount']:.2f} has been confirmed!"
        }
        
        self.sent_notifications.append(notification)
        self.logger.log(LogLevel.INFO, f"Sent order confirmation for {data['order_id']}")
    
    def _send_inventory_alert(self, data: Dict[str, Any]) -> None:
        """Send inventory alert to admin."""
        notification = {
            'type': 'inventory_alert',
            'user_id': 'admin',
            'channel': NotificationChannel.EMAIL.value,
            'subject': 'Low Inventory Alert',
            'message': f"Product {data['product_id']} has only {data['current_count']} items left"
        }
        
        self.sent_notifications.append(notification)
        self.logger.log(LogLevel.WARNING, f"Sent inventory alert for {data['product_id']}")


# ============================================================================
# ANALYTICS SERVICE
# ============================================================================

class AnalyticsService(EventListener):
    """Analytics service that tracks system metrics."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.metrics = {
            'user_registrations': 0,
            'user_logins': 0,
            'orders_placed': 0,
            'total_revenue': 0.0,
            'products_viewed': 0,
            'cart_updates': 0
        }
        self.daily_metrics = defaultdict(lambda: defaultdict(int))
    
    def handle_event(self, event: Event) -> None:
        """Track events for analytics."""
        today = event.timestamp.date().isoformat()
        
        if event.event_type == EventType.USER_REGISTERED:
            self.metrics['user_registrations'] += 1
            self.daily_metrics[today]['registrations'] += 1
        
        elif event.event_type == EventType.USER_LOGIN:
            self.metrics['user_logins'] += 1
            self.daily_metrics[today]['logins'] += 1
        
        elif event.event_type == EventType.ORDER_PLACED:
            self.metrics['orders_placed'] += 1
            self.metrics['total_revenue'] += event.data['total_amount']
            self.daily_metrics[today]['orders'] += 1
            self.daily_metrics[today]['revenue'] += event.data['total_amount']
        
        elif event.event_type == EventType.PRODUCT_VIEWED:
            self.metrics['products_viewed'] += 1
            self.daily_metrics[today]['product_views'] += 1
        
        elif event.event_type == EventType.CART_UPDATED:
            self.metrics['cart_updates'] += 1
            self.daily_metrics[today]['cart_updates'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            'total_metrics': self.metrics,
            'daily_metrics': dict(self.daily_metrics)
        }


# ============================================================================
# MAIN INTEGRATED SYSTEM
# ============================================================================

class ECommerceSystem:
    """Integrated e-commerce system demonstrating multiple design patterns."""
    
    def __init__(self):
        # Core infrastructure
        self.event_bus = EventBus()
        self.logger = Logger()
        self.cache = Cache(max_size=5000)
        self.rate_limiter = RateLimiter(capacity=100, refill_rate=10)
        
        # Services
        self.user_service = UserService(self.event_bus, self.logger, self.cache)
        self.product_service = ProductService(self.event_bus, self.logger, self.cache)
        self.cart_service = CartService(
            self.product_service, self.event_bus, self.logger, self.cache
        )
        self.order_service = OrderService(
            self.product_service, self.cart_service, self.event_bus, self.logger
        )
        
        # Event listeners
        self.notification_service = NotificationService(self.logger)
        self.analytics_service = AnalyticsService(self.logger)
        
        # Register event listeners
        self._register_event_listeners()
        
        self.logger.log(LogLevel.INFO, "E-Commerce System initialized")
    
    def _register_event_listeners(self) -> None:
        """Register all event listeners."""
        for event_type in EventType:
            self.event_bus.subscribe(event_type, self.notification_service)
            self.event_bus.subscribe(event_type, self.analytics_service)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'cache_hit_rate': f"{self.cache.hit_rate:.2%}",
            'rate_limiter_tokens': self.rate_limiter.tokens,
            'total_users': len(self.user_service.users),
            'total_products': len(self.product_service.products),
            'total_orders': len(self.order_service.orders),
            'notifications_sent': len(self.notification_service.sent_notifications),
            'metrics': self.analytics_service.get_metrics(),
            'recent_logs': self.logger.get_logs(limit=5)
        }


def demonstrate_integrated_system():
    """Demonstrate the integrated e-commerce system."""
    print("=== INTEGRATED E-COMMERCE SYSTEM DEMONSTRATION ===\n")
    
    # Initialize system
    system = ECommerceSystem()
    print("✓ System initialized with all components")
    print()
    
    # Create users
    print("1. USER REGISTRATION:")
    users = []
    user_data = [
        ("alice@example.com", "Alice Johnson", "+1234567890"),
        ("bob@example.com", "Bob Smith", "+1987654321"),
        ("charlie@example.com", "Charlie Brown", "+1555123456")
    ]
    
    for email, name, phone in user_data:
        user = system.user_service.create_user(email, name, phone)
        users.append(user)
        print(f"   ✓ Created user: {name}")
    
    print()
    
    # Add products
    print("2. PRODUCT CATALOG:")
    products = []
    product_data = [
        ("Laptop", "High-performance laptop", 999.99, "Electronics", "LAP001", 10),
        ("Smartphone", "Latest smartphone", 699.99, "Electronics", "PHN001", 25),
        ("Headphones", "Wireless headphones", 199.99, "Electronics", "HDP001", 50),
        ("T-Shirt", "Cotton t-shirt", 19.99, "Clothing", "TSH001", 100),
        ("Jeans", "Denim jeans", 59.99, "Clothing", "JNS001", 75)
    ]
    
    for name, desc, price, category, sku, inventory in product_data:
        product = system.product_service.add_product(
            name, desc, price, category, sku, inventory
        )
        products.append(product)
        print(f"   ✓ Added product: {name} (${price})")
    
    print()
    
    # Simulate user activity
    print("3. USER ACTIVITY SIMULATION:")
    
    # User login
    user = system.user_service.authenticate_user("alice@example.com", "password")
    print(f"   ✓ User authenticated: {user.email}")
    
    # Product browsing
    search_results = system.product_service.search_products(category="Electronics")
    print(f"   ✓ Found {len(search_results)} electronics products")
    
    # Add to cart
    laptop = products[0]  # Laptop
    phone = products[1]   # Smartphone
    
    system.cart_service.add_to_cart(user.user_id, laptop.product_id, 1)
    system.cart_service.add_to_cart(user.user_id, phone.product_id, 2)
    print(f"   ✓ Added products to cart")
    
    # View cart
    cart = system.cart_service.get_cart(user.user_id)
    print(f"   ✓ Cart total: ${cart.get_total():.2f} ({cart.get_item_count()} items)")
    
    # Place order
    order = system.order_service.place_order(
        user.user_id,
        "123 Main St, City, State",
        "credit_card"
    )
    if order:
        print(f"   ✓ Order placed: {order.order_id[:8]} (${order.total_amount:.2f})")
    
    print()
    
    # Simulate more activity
    print("4. ADDITIONAL ACTIVITY:")
    
    # More users and orders
    for i, (email, name, phone) in enumerate(user_data[1:], 1):
        user = system.user_service.authenticate_user(email, "password")
        
        # Add random products to cart
        selected_products = products[i:i+2]
        for product in selected_products:
            system.cart_service.add_to_cart(user.user_id, product.product_id, 1)
        
        # Place order
        order = system.order_service.place_order(
            user.user_id,
            f"{100+i*10} Main St, City, State",
            "credit_card"
        )
        
        print(f"   ✓ {name} placed order: ${order.total_amount:.2f}")
    
    print()
    
    # Show system status
    print("5. SYSTEM STATUS:")
    status = system.get_system_status()
    
    print(f"   Cache hit rate: {status['cache_hit_rate']}")
    print(f"   Total users: {status['total_users']}")
    print(f"   Total products: {status['total_products']}")
    print(f"   Total orders: {status['total_orders']}")
    print(f"   Notifications sent: {status['notifications_sent']}")
    
    # Show analytics
    metrics = status['metrics']['total_metrics']
    print(f"\n   Analytics:")
    print(f"     User registrations: {metrics['user_registrations']}")
    print(f"     Orders placed: {metrics['orders_placed']}")
    print(f"     Total revenue: ${metrics['total_revenue']:.2f}")
    print(f"     Cart updates: {metrics['cart_updates']}")
    
    print()
    
    # Show recent activity
    print("6. RECENT ACTIVITY:")
    recent_logs = status['recent_logs']
    for log_entry in recent_logs:
        print(f"   [{log_entry['level'].upper()}] {log_entry['message']}")
    
    print()
    
    # Show notifications
    print("7. NOTIFICATIONS SENT:")
    notifications = system.notification_service.sent_notifications
    for notif in notifications[-5:]:  # Last 5 notifications
        print(f"   {notif['type']}: {notif['subject']}")
    
    print()
    
    # Performance metrics
    print("8. PERFORMANCE METRICS:")
    print(f"   Cache entries: {len(system.cache.data)}")
    print(f"   Cache hit rate: {system.cache.hit_rate:.2%}")
    print(f"   Rate limiter tokens: {system.rate_limiter.tokens}")
    print(f"   Event bus listeners: {len(system.event_bus.listeners)}")
    
    print()
    print("=== INTEGRATED SYSTEM DEMONSTRATION COMPLETE ===")
    print("\nThis demonstration showcased:")
    print("• Observer Pattern (Event Bus)")
    print("• Facade Pattern (Service Interfaces)")
    print("• Strategy Pattern (Multiple Services)")
    print("• Singleton Pattern (Shared Resources)")
    print("• Factory Pattern (Object Creation)")
    print("• Decorator Pattern (Service Enhancement)")
    print("• Cache Implementation with TTL")
    print("• Rate Limiting with Token Bucket")
    print("• Comprehensive Logging")
    print("• Event-Driven Architecture")
    print("• Service Integration and Communication")


if __name__ == "__main__":
    demonstrate_integrated_system()
