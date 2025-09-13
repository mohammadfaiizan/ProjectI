"""
SOLID IN REAL SYSTEMS - Enterprise Application Architecture
===========================================================

Problem Statement:
Demonstrate SOLID principles in real-world enterprise systems:
- Complete e-commerce platform following all SOLID principles
- Microservices architecture with SOLID compliance
- Domain-driven design with SOLID principles
- Enterprise patterns and SOLID integration
- Scalable and maintainable system design

Learning Objectives:
- Apply SOLID principles in complex real-world systems
- Design enterprise-grade applications
- Integrate SOLID with architectural patterns
- Build scalable and maintainable systems
- Understand SOLID in microservices context
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Protocol
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import uuid
import json


# ============================================================================
# DOMAIN VALUE OBJECTS AND ENUMS
# ============================================================================

class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class UserRole(Enum):
    CUSTOMER = "customer"
    ADMIN = "admin"
    VENDOR = "vendor"
    SUPPORT = "support"


@dataclass(frozen=True)
class Money:
    """Value object for money (SRP - single responsibility for money operations)."""
    amount: float
    currency: str = "USD"
    
    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Amount cannot be negative")
        if not self.currency or len(self.currency) != 3:
            raise ValueError("Currency must be 3-letter code")
    
    def add(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)
    
    def subtract(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError("Cannot subtract different currencies")
        return Money(self.amount - other.amount, self.currency)
    
    def multiply(self, factor: float) -> 'Money':
        return Money(self.amount * factor, self.currency)
    
    def __str__(self) -> str:
        return f"{self.currency} {self.amount:.2f}"


@dataclass(frozen=True)
class Address:
    """Value object for address (SRP - address representation only)."""
    street: str
    city: str
    state: str
    postal_code: str
    country: str = "USA"
    
    def __post_init__(self):
        if not all([self.street, self.city, self.state, self.postal_code]):
            raise ValueError("All address fields are required")


# ============================================================================
# DOMAIN ENTITIES (SRP - Single Responsibility)
# ============================================================================

class User:
    """User entity (SRP - user data and basic operations only)."""
    
    def __init__(self, user_id: str, username: str, email: str, role: UserRole):
        self.id = user_id
        self.username = username
        self.email = email
        self.role = role
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.is_active = True
        self.last_login: Optional[datetime] = None
    
    def update_last_login(self) -> None:
        """Update last login timestamp."""
        self.last_login = datetime.now()
        self.updated_at = datetime.now()
    
    def deactivate(self) -> None:
        """Deactivate user account."""
        self.is_active = False
        self.updated_at = datetime.now()
    
    def change_role(self, new_role: UserRole) -> None:
        """Change user role."""
        self.role = new_role
        self.updated_at = datetime.now()


class Product:
    """Product entity (SRP - product data and operations only)."""
    
    def __init__(self, product_id: str, name: str, description: str, 
                 price: Money, category: str, vendor_id: str):
        self.id = product_id
        self.name = name
        self.description = description
        self.price = price
        self.category = category
        self.vendor_id = vendor_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.is_active = True
        self.stock_quantity = 0
    
    def update_price(self, new_price: Money) -> None:
        """Update product price."""
        self.price = new_price
        self.updated_at = datetime.now()
    
    def update_stock(self, quantity: int) -> None:
        """Update stock quantity."""
        if quantity < 0:
            raise ValueError("Stock quantity cannot be negative")
        self.stock_quantity = quantity
        self.updated_at = datetime.now()
    
    def is_available(self, required_quantity: int = 1) -> bool:
        """Check if product is available."""
        return self.is_active and self.stock_quantity >= required_quantity


class OrderItem:
    """Order item entity (SRP - order item data only)."""
    
    def __init__(self, product: Product, quantity: int, unit_price: Money):
        self.product = product
        self.quantity = quantity
        self.unit_price = unit_price
        self.total_price = unit_price.multiply(quantity)
    
    def update_quantity(self, new_quantity: int) -> None:
        """Update item quantity."""
        if new_quantity <= 0:
            raise ValueError("Quantity must be positive")
        self.quantity = new_quantity
        self.total_price = self.unit_price.multiply(new_quantity)


class Order:
    """Order entity (SRP - order data and basic operations only)."""
    
    def __init__(self, order_id: str, customer: User, shipping_address: Address):
        self.id = order_id
        self.customer = customer
        self.shipping_address = shipping_address
        self.items: List[OrderItem] = []
        self.status = OrderStatus.PENDING
        self.payment_status = PaymentStatus.PENDING
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.subtotal = Money(0)
        self.tax_amount = Money(0)
        self.shipping_cost = Money(0)
        self.total_amount = Money(0)
    
    def add_item(self, item: OrderItem) -> None:
        """Add item to order."""
        self.items.append(item)
        self._recalculate_totals()
    
    def remove_item(self, product_id: str) -> bool:
        """Remove item from order."""
        for i, item in enumerate(self.items):
            if item.product.id == product_id:
                self.items.pop(i)
                self._recalculate_totals()
                return True
        return False
    
    def update_status(self, new_status: OrderStatus) -> None:
        """Update order status."""
        self.status = new_status
        self.updated_at = datetime.now()
    
    def update_payment_status(self, new_status: PaymentStatus) -> None:
        """Update payment status."""
        self.payment_status = new_status
        self.updated_at = datetime.now()
    
    def _recalculate_totals(self) -> None:
        """Recalculate order totals."""
        if not self.items:
            self.subtotal = Money(0)
            self.tax_amount = Money(0)
            self.total_amount = Money(0)
            return
        
        currency = self.items[0].total_price.currency
        subtotal_amount = sum(item.total_price.amount for item in self.items)
        
        self.subtotal = Money(subtotal_amount, currency)
        self.tax_amount = Money(subtotal_amount * 0.08, currency)  # 8% tax
        self.shipping_cost = Money(10.0, currency) if subtotal_amount < 100 else Money(0, currency)
        self.total_amount = self.subtotal.add(self.tax_amount).add(self.shipping_cost)
        
        self.updated_at = datetime.now()


# ============================================================================
# REPOSITORY INTERFACES (DIP - Dependency Inversion)
# ============================================================================

class UserRepository(ABC):
    """User repository interface (DIP - abstraction)."""
    
    @abstractmethod
    def save(self, user: User) -> bool:
        pass
    
    @abstractmethod
    def find_by_id(self, user_id: str) -> Optional[User]:
        pass
    
    @abstractmethod
    def find_by_email(self, email: str) -> Optional[User]:
        pass
    
    @abstractmethod
    def find_by_role(self, role: UserRole) -> List[User]:
        pass


class ProductRepository(ABC):
    """Product repository interface (DIP - abstraction)."""
    
    @abstractmethod
    def save(self, product: Product) -> bool:
        pass
    
    @abstractmethod
    def find_by_id(self, product_id: str) -> Optional[Product]:
        pass
    
    @abstractmethod
    def find_by_category(self, category: str) -> List[Product]:
        pass
    
    @abstractmethod
    def find_available_products(self) -> List[Product]:
        pass


class OrderRepository(ABC):
    """Order repository interface (DIP - abstraction)."""
    
    @abstractmethod
    def save(self, order: Order) -> bool:
        pass
    
    @abstractmethod
    def find_by_id(self, order_id: str) -> Optional[Order]:
        pass
    
    @abstractmethod
    def find_by_customer(self, customer_id: str) -> List[Order]:
        pass
    
    @abstractmethod
    def find_by_status(self, status: OrderStatus) -> List[Order]:
        pass


# ============================================================================
# SERVICE INTERFACES (ISP - Interface Segregation)
# ============================================================================

class AuthenticationService(ABC):
    """Authentication service interface (ISP - focused interface)."""
    
    @abstractmethod
    def authenticate(self, username: str, password: str) -> Optional[User]:
        pass
    
    @abstractmethod
    def generate_token(self, user: User) -> str:
        pass
    
    @abstractmethod
    def validate_token(self, token: str) -> Optional[User]:
        pass


class AuthorizationService(ABC):
    """Authorization service interface (ISP - focused interface)."""
    
    @abstractmethod
    def can_access_resource(self, user: User, resource: str, action: str) -> bool:
        pass
    
    @abstractmethod
    def get_user_permissions(self, user: User) -> List[str]:
        pass


class NotificationService(ABC):
    """Notification service interface (ISP - focused interface)."""
    
    @abstractmethod
    def send_order_confirmation(self, order: Order) -> bool:
        pass
    
    @abstractmethod
    def send_payment_notification(self, order: Order, payment_status: PaymentStatus) -> bool:
        pass
    
    @abstractmethod
    def send_shipping_notification(self, order: Order, tracking_number: str) -> bool:
        pass


class PaymentService(ABC):
    """Payment service interface (ISP - focused interface)."""
    
    @abstractmethod
    def process_payment(self, order: Order, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def refund_payment(self, order: Order, amount: Money) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_payment_status(self, transaction_id: str) -> PaymentStatus:
        pass


class InventoryService(ABC):
    """Inventory service interface (ISP - focused interface)."""
    
    @abstractmethod
    def reserve_items(self, order: Order) -> bool:
        pass
    
    @abstractmethod
    def release_reservation(self, order: Order) -> bool:
        pass
    
    @abstractmethod
    def update_stock_levels(self, product_updates: Dict[str, int]) -> bool:
        pass


class ShippingService(ABC):
    """Shipping service interface (ISP - focused interface)."""
    
    @abstractmethod
    def calculate_shipping_cost(self, order: Order) -> Money:
        pass
    
    @abstractmethod
    def create_shipment(self, order: Order) -> str:  # Returns tracking number
        pass
    
    @abstractmethod
    def track_shipment(self, tracking_number: str) -> Dict[str, Any]:
        pass


class AuditService(ABC):
    """Audit service interface (ISP - focused interface)."""
    
    @abstractmethod
    def log_user_action(self, user: User, action: str, details: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def log_system_event(self, event: str, details: Dict[str, Any]) -> None:
        pass


# ============================================================================
# PRICING STRATEGIES (OCP - Open/Closed Principle)
# ============================================================================

class PricingStrategy(ABC):
    """Abstract pricing strategy (OCP - open for extension)."""
    
    @abstractmethod
    def calculate_discount(self, order: Order, user: User) -> Money:
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        pass


class StandardPricing(PricingStrategy):
    """Standard pricing with no discounts."""
    
    def calculate_discount(self, order: Order, user: User) -> Money:
        return Money(0, order.subtotal.currency)
    
    def get_description(self) -> str:
        return "Standard pricing"


class VIPCustomerPricing(PricingStrategy):
    """VIP customer pricing strategy."""
    
    def __init__(self, discount_percentage: float):
        self.discount_percentage = discount_percentage
    
    def calculate_discount(self, order: Order, user: User) -> Money:
        if user.role == UserRole.CUSTOMER:  # Assume VIP logic here
            discount_amount = order.subtotal.amount * (self.discount_percentage / 100)
            return Money(discount_amount, order.subtotal.currency)
        return Money(0, order.subtotal.currency)
    
    def get_description(self) -> str:
        return f"VIP customer {self.discount_percentage}% discount"


class BulkOrderPricing(PricingStrategy):
    """Bulk order pricing strategy."""
    
    def __init__(self, min_items: int, discount_percentage: float):
        self.min_items = min_items
        self.discount_percentage = discount_percentage
    
    def calculate_discount(self, order: Order, user: User) -> Money:
        total_items = sum(item.quantity for item in order.items)
        if total_items >= self.min_items:
            discount_amount = order.subtotal.amount * (self.discount_percentage / 100)
            return Money(discount_amount, order.subtotal.currency)
        return Money(0, order.subtotal.currency)
    
    def get_description(self) -> str:
        return f"Bulk order discount: {self.discount_percentage}% off {self.min_items}+ items"


class SeasonalPricing(PricingStrategy):
    """Seasonal pricing strategy (OCP - new strategy without modifying existing code)."""
    
    def __init__(self, season_multiplier: float, valid_until: datetime):
        self.season_multiplier = season_multiplier
        self.valid_until = valid_until
    
    def calculate_discount(self, order: Order, user: User) -> Money:
        if datetime.now() <= self.valid_until:
            discount_amount = order.subtotal.amount * self.season_multiplier
            return Money(discount_amount, order.subtotal.currency)
        return Money(0, order.subtotal.currency)
    
    def get_description(self) -> str:
        return f"Seasonal discount: {self.season_multiplier*100:.0f}% off until {self.valid_until.strftime('%Y-%m-%d')}"


# ============================================================================
# CONCRETE IMPLEMENTATIONS (LSP - Liskov Substitution)
# ============================================================================

class InMemoryUserRepository(UserRepository):
    """In-memory user repository (LSP - substitutable for UserRepository)."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.email_index: Dict[str, str] = {}
    
    def save(self, user: User) -> bool:
        self.users[user.id] = user
        self.email_index[user.email] = user.id
        return True
    
    def find_by_id(self, user_id: str) -> Optional[User]:
        return self.users.get(user_id)
    
    def find_by_email(self, email: str) -> Optional[User]:
        user_id = self.email_index.get(email)
        return self.users.get(user_id) if user_id else None
    
    def find_by_role(self, role: UserRole) -> List[User]:
        return [user for user in self.users.values() if user.role == role]


class InMemoryProductRepository(ProductRepository):
    """In-memory product repository (LSP - substitutable for ProductRepository)."""
    
    def __init__(self):
        self.products: Dict[str, Product] = {}
        self.category_index: Dict[str, List[str]] = {}
    
    def save(self, product: Product) -> bool:
        self.products[product.id] = product
        
        # Update category index
        if product.category not in self.category_index:
            self.category_index[product.category] = []
        if product.id not in self.category_index[product.category]:
            self.category_index[product.category].append(product.id)
        
        return True
    
    def find_by_id(self, product_id: str) -> Optional[Product]:
        return self.products.get(product_id)
    
    def find_by_category(self, category: str) -> List[Product]:
        product_ids = self.category_index.get(category, [])
        return [self.products[pid] for pid in product_ids if pid in self.products]
    
    def find_available_products(self) -> List[Product]:
        return [p for p in self.products.values() if p.is_active and p.stock_quantity > 0]


class InMemoryOrderRepository(OrderRepository):
    """In-memory order repository (LSP - substitutable for OrderRepository)."""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.customer_index: Dict[str, List[str]] = {}
        self.status_index: Dict[OrderStatus, List[str]] = {}
    
    def save(self, order: Order) -> bool:
        self.orders[order.id] = order
        
        # Update customer index
        if order.customer.id not in self.customer_index:
            self.customer_index[order.customer.id] = []
        if order.id not in self.customer_index[order.customer.id]:
            self.customer_index[order.customer.id].append(order.id)
        
        # Update status index
        if order.status not in self.status_index:
            self.status_index[order.status] = []
        if order.id not in self.status_index[order.status]:
            self.status_index[order.status].append(order.id)
        
        return True
    
    def find_by_id(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)
    
    def find_by_customer(self, customer_id: str) -> List[Order]:
        order_ids = self.customer_index.get(customer_id, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]
    
    def find_by_status(self, status: OrderStatus) -> List[Order]:
        order_ids = self.status_index.get(status, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]


class EmailNotificationService(NotificationService):
    """Email notification service (LSP - substitutable for NotificationService)."""
    
    def __init__(self):
        self.sent_notifications = []
    
    def send_order_confirmation(self, order: Order) -> bool:
        notification = {
            'type': 'order_confirmation',
            'recipient': order.customer.email,
            'order_id': order.id,
            'total': str(order.total_amount),
            'sent_at': datetime.now().isoformat()
        }
        self.sent_notifications.append(notification)
        print(f"Order confirmation sent to {order.customer.email} for order {order.id}")
        return True
    
    def send_payment_notification(self, order: Order, payment_status: PaymentStatus) -> bool:
        notification = {
            'type': 'payment_notification',
            'recipient': order.customer.email,
            'order_id': order.id,
            'payment_status': payment_status.value,
            'sent_at': datetime.now().isoformat()
        }
        self.sent_notifications.append(notification)
        print(f"Payment notification sent to {order.customer.email}: {payment_status.value}")
        return True
    
    def send_shipping_notification(self, order: Order, tracking_number: str) -> bool:
        notification = {
            'type': 'shipping_notification',
            'recipient': order.customer.email,
            'order_id': order.id,
            'tracking_number': tracking_number,
            'sent_at': datetime.now().isoformat()
        }
        self.sent_notifications.append(notification)
        print(f"Shipping notification sent to {order.customer.email}: {tracking_number}")
        return True


class MockPaymentService(PaymentService):
    """Mock payment service for testing (LSP - substitutable for PaymentService)."""
    
    def __init__(self):
        self.processed_payments = []
    
    def process_payment(self, order: Order, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        transaction_id = str(uuid.uuid4())
        payment_record = {
            'transaction_id': transaction_id,
            'order_id': order.id,
            'amount': order.total_amount.amount,
            'currency': order.total_amount.currency,
            'status': 'completed',
            'processed_at': datetime.now().isoformat()
        }
        self.processed_payments.append(payment_record)
        print(f"Payment processed for order {order.id}: {order.total_amount}")
        return payment_record
    
    def refund_payment(self, order: Order, amount: Money) -> Dict[str, Any]:
        refund_record = {
            'refund_id': str(uuid.uuid4()),
            'order_id': order.id,
            'amount': amount.amount,
            'currency': amount.currency,
            'status': 'completed',
            'processed_at': datetime.now().isoformat()
        }
        print(f"Refund processed for order {order.id}: {amount}")
        return refund_record
    
    def get_payment_status(self, transaction_id: str) -> PaymentStatus:
        for payment in self.processed_payments:
            if payment['transaction_id'] == transaction_id:
                return PaymentStatus.COMPLETED
        return PaymentStatus.PENDING


class SimpleInventoryService(InventoryService):
    """Simple inventory service (LSP - substitutable for InventoryService)."""
    
    def __init__(self, product_repository: ProductRepository):
        self.product_repository = product_repository
        self.reservations: Dict[str, Dict[str, int]] = {}  # order_id -> {product_id: quantity}
    
    def reserve_items(self, order: Order) -> bool:
        # Check availability first
        for item in order.items:
            product = self.product_repository.find_by_id(item.product.id)
            if not product or not product.is_available(item.quantity):
                return False
        
        # Reserve items
        reservations = {}
        for item in order.items:
            product = self.product_repository.find_by_id(item.product.id)
            product.stock_quantity -= item.quantity
            self.product_repository.save(product)
            reservations[item.product.id] = item.quantity
        
        self.reservations[order.id] = reservations
        print(f"Items reserved for order {order.id}")
        return True
    
    def release_reservation(self, order: Order) -> bool:
        if order.id not in self.reservations:
            return False
        
        reservations = self.reservations[order.id]
        for product_id, quantity in reservations.items():
            product = self.product_repository.find_by_id(product_id)
            if product:
                product.stock_quantity += quantity
                self.product_repository.save(product)
        
        del self.reservations[order.id]
        print(f"Reservation released for order {order.id}")
        return True
    
    def update_stock_levels(self, product_updates: Dict[str, int]) -> bool:
        for product_id, new_quantity in product_updates.items():
            product = self.product_repository.find_by_id(product_id)
            if product:
                product.update_stock(new_quantity)
                self.product_repository.save(product)
        print(f"Stock levels updated for {len(product_updates)} products")
        return True


# ============================================================================
# HIGH-LEVEL BUSINESS SERVICES (DIP - Dependency Inversion)
# ============================================================================

class OrderManagementService:
    """
    High-level order management service (DIP - depends on abstractions).
    Demonstrates all SOLID principles working together.
    """
    
    def __init__(self,
                 order_repository: OrderRepository,
                 product_repository: ProductRepository,
                 user_repository: UserRepository,
                 inventory_service: InventoryService,
                 payment_service: PaymentService,
                 notification_service: NotificationService,
                 audit_service: AuditService,
                 pricing_strategy: PricingStrategy):
        # DIP: All dependencies are abstractions
        self.order_repository = order_repository
        self.product_repository = product_repository
        self.user_repository = user_repository
        self.inventory_service = inventory_service
        self.payment_service = payment_service
        self.notification_service = notification_service
        self.audit_service = audit_service
        self.pricing_strategy = pricing_strategy
    
    def create_order(self, customer_id: str, shipping_address: Address,
                    items: List[Dict[str, Any]]) -> Optional[Order]:
        """Create new order with full business logic."""
        
        # Find customer
        customer = self.user_repository.find_by_id(customer_id)
        if not customer:
            self.audit_service.log_system_event("order_creation_failed", 
                                               {"reason": "customer_not_found", "customer_id": customer_id})
            return None
        
        # Create order
        order_id = str(uuid.uuid4())
        order = Order(order_id, customer, shipping_address)
        
        # Add items
        for item_data in items:
            product = self.product_repository.find_by_id(item_data['product_id'])
            if not product:
                continue
            
            if not product.is_available(item_data['quantity']):
                self.audit_service.log_system_event("order_creation_failed",
                                                   {"reason": "insufficient_stock", "product_id": product.id})
                return None
            
            order_item = OrderItem(product, item_data['quantity'], product.price)
            order.add_item(order_item)
        
        if not order.items:
            return None
        
        # Apply pricing strategy (OCP - open for extension)
        discount = self.pricing_strategy.calculate_discount(order, customer)
        if discount.amount > 0:
            order.total_amount = order.total_amount.subtract(discount)
        
        # Reserve inventory
        if not self.inventory_service.reserve_items(order):
            self.audit_service.log_system_event("order_creation_failed",
                                               {"reason": "inventory_reservation_failed", "order_id": order.id})
            return None
        
        # Save order
        if self.order_repository.save(order):
            # Send confirmation
            self.notification_service.send_order_confirmation(order)
            
            # Audit log
            self.audit_service.log_user_action(customer, "order_created", 
                                             {"order_id": order.id, "total": str(order.total_amount)})
            
            return order
        
        return None
    
    def process_payment(self, order_id: str, payment_details: Dict[str, Any]) -> bool:
        """Process payment for order."""
        
        order = self.order_repository.find_by_id(order_id)
        if not order:
            return False
        
        if order.payment_status != PaymentStatus.PENDING:
            return False
        
        # Update payment status
        order.update_payment_status(PaymentStatus.PROCESSING)
        self.order_repository.save(order)
        
        try:
            # Process payment
            payment_result = self.payment_service.process_payment(order, payment_details)
            
            if payment_result.get('status') == 'completed':
                order.update_payment_status(PaymentStatus.COMPLETED)
                order.update_status(OrderStatus.CONFIRMED)
                self.order_repository.save(order)
                
                # Send notification
                self.notification_service.send_payment_notification(order, PaymentStatus.COMPLETED)
                
                # Audit log
                self.audit_service.log_user_action(order.customer, "payment_completed",
                                                 {"order_id": order.id, "transaction_id": payment_result.get('transaction_id')})
                
                return True
            else:
                order.update_payment_status(PaymentStatus.FAILED)
                self.order_repository.save(order)
                
                # Release inventory reservation
                self.inventory_service.release_reservation(order)
                
                # Send failure notification
                self.notification_service.send_payment_notification(order, PaymentStatus.FAILED)
                
                return False
                
        except Exception as e:
            order.update_payment_status(PaymentStatus.FAILED)
            self.order_repository.save(order)
            
            self.audit_service.log_system_event("payment_processing_error",
                                               {"order_id": order.id, "error": str(e)})
            return False
    
    def cancel_order(self, order_id: str, user_id: str) -> bool:
        """Cancel order."""
        
        order = self.order_repository.find_by_id(order_id)
        if not order:
            return False
        
        user = self.user_repository.find_by_id(user_id)
        if not user:
            return False
        
        # Check if order can be cancelled
        if order.status in [OrderStatus.SHIPPED, OrderStatus.DELIVERED]:
            return False
        
        # Process refund if payment was completed
        if order.payment_status == PaymentStatus.COMPLETED:
            refund_result = self.payment_service.refund_payment(order, order.total_amount)
            if refund_result.get('status') == 'completed':
                order.update_payment_status(PaymentStatus.REFUNDED)
        
        # Release inventory reservation
        self.inventory_service.release_reservation(order)
        
        # Update order status
        order.update_status(OrderStatus.CANCELLED)
        self.order_repository.save(order)
        
        # Audit log
        self.audit_service.log_user_action(user, "order_cancelled", {"order_id": order.id})
        
        return True
    
    def get_order_summary(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive order summary."""
        
        order = self.order_repository.find_by_id(order_id)
        if not order:
            return None
        
        return {
            'order_id': order.id,
            'customer': {
                'id': order.customer.id,
                'username': order.customer.username,
                'email': order.customer.email
            },
            'status': order.status.value,
            'payment_status': order.payment_status.value,
            'items': [
                {
                    'product_id': item.product.id,
                    'product_name': item.product.name,
                    'quantity': item.quantity,
                    'unit_price': str(item.unit_price),
                    'total_price': str(item.total_price)
                }
                for item in order.items
            ],
            'pricing': {
                'subtotal': str(order.subtotal),
                'tax_amount': str(order.tax_amount),
                'shipping_cost': str(order.shipping_cost),
                'total_amount': str(order.total_amount),
                'pricing_strategy': self.pricing_strategy.get_description()
            },
            'shipping_address': {
                'street': order.shipping_address.street,
                'city': order.shipping_address.city,
                'state': order.shipping_address.state,
                'postal_code': order.shipping_address.postal_code,
                'country': order.shipping_address.country
            },
            'timestamps': {
                'created_at': order.created_at.isoformat(),
                'updated_at': order.updated_at.isoformat()
            }
        }


class SimpleAuditService(AuditService):
    """Simple audit service implementation."""
    
    def __init__(self):
        self.audit_logs = []
    
    def log_user_action(self, user: User, action: str, details: Dict[str, Any]) -> None:
        log_entry = {
            'type': 'user_action',
            'user_id': user.id,
            'username': user.username,
            'action': action,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.audit_logs.append(log_entry)
        print(f"AUDIT: User {user.username} performed {action}")
    
    def log_system_event(self, event: str, details: Dict[str, Any]) -> None:
        log_entry = {
            'type': 'system_event',
            'event': event,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.audit_logs.append(log_entry)
        print(f"AUDIT: System event {event}")


def demonstrate_solid_in_real_systems():
    """
    Demonstrate SOLID principles in a complete real-world e-commerce system.
    """
    print("=== SOLID PRINCIPLES IN REAL SYSTEMS DEMONSTRATION ===\n")
    
    print("Building Complete E-commerce Platform with SOLID Principles:")
    print("• SRP: Each class has single, well-defined responsibility")
    print("• OCP: Pricing strategies extensible without modification")
    print("• LSP: All implementations fully substitutable")
    print("• ISP: Focused interfaces for specific capabilities")
    print("• DIP: High-level services depend on abstractions")
    print()
    
    # 1. Setup all dependencies (DIP - Dependency Injection)
    print("1. SETTING UP ENTERPRISE SYSTEM DEPENDENCIES:")
    
    # Repositories
    user_repository = InMemoryUserRepository()
    product_repository = InMemoryProductRepository()
    order_repository = InMemoryOrderRepository()
    
    # Services
    inventory_service = SimpleInventoryService(product_repository)
    payment_service = MockPaymentService()
    notification_service = EmailNotificationService()
    audit_service = SimpleAuditService()
    
    # Pricing strategy (OCP - can be changed without modifying code)
    pricing_strategy = VIPCustomerPricing(15.0)  # 15% VIP discount
    
    # High-level service
    order_service = OrderManagementService(
        order_repository, product_repository, user_repository,
        inventory_service, payment_service, notification_service,
        audit_service, pricing_strategy
    )
    
    print("   ✓ All services initialized with dependency injection")
    print("   ✓ System ready for enterprise-level operations")
    print()
    
    # 2. Create test data (SRP - each entity has single responsibility)
    print("2. CREATING ENTERPRISE TEST DATA:")
    
    # Create users
    customer = User("CUST001", "john_doe", "john@example.com", UserRole.CUSTOMER)
    admin = User("ADMIN001", "admin_user", "admin@example.com", UserRole.ADMIN)
    
    user_repository.save(customer)
    user_repository.save(admin)
    
    # Create products
    products = [
        Product("PROD001", "Gaming Laptop", "High-performance gaming laptop", 
                Money(1299.99), "Electronics", "VENDOR001"),
        Product("PROD002", "Wireless Mouse", "Ergonomic wireless mouse", 
                Money(49.99), "Electronics", "VENDOR001"),
        Product("PROD003", "Mechanical Keyboard", "RGB mechanical keyboard", 
                Money(129.99), "Electronics", "VENDOR002"),
        Product("PROD004", "4K Monitor", "27-inch 4K gaming monitor", 
                Money(399.99), "Electronics", "VENDOR002")
    ]
    
    for product in products:
        product.update_stock(50)  # Add stock
        product_repository.save(product)
    
    print(f"   ✓ Created {len(products)} products with inventory")
    print(f"   ✓ Created {len([customer, admin])} users with different roles")
    print()
    
    # 3. Test complete order workflow
    print("3. COMPLETE ORDER WORKFLOW DEMONSTRATION:")
    
    # Create shipping address
    shipping_address = Address(
        "123 Main Street",
        "New York",
        "NY",
        "10001",
        "USA"
    )
    
    # Create order with multiple items
    order_items = [
        {'product_id': 'PROD001', 'quantity': 1},  # Gaming Laptop
        {'product_id': 'PROD002', 'quantity': 2},  # Wireless Mouse x2
        {'product_id': 'PROD003', 'quantity': 1}   # Mechanical Keyboard
    ]
    
    print("   Creating order with multiple items...")
    order = order_service.create_order(customer.id, shipping_address, order_items)
    
    if order:
        print(f"   ✓ Order created successfully: {order.id}")
        print(f"   ✓ Order total: {order.total_amount}")
        print(f"   ✓ Applied pricing strategy: {pricing_strategy.get_description()}")
        
        # Process payment
        print("\n   Processing payment...")
        payment_details = {
            'card_number': '1234567890123456',
            'expiry': '12/25',
            'cvv': '123',
            'cardholder': 'John Doe'
        }
        
        payment_success = order_service.process_payment(order.id, payment_details)
        print(f"   ✓ Payment processing: {'Success' if payment_success else 'Failed'}")
        
        # Get order summary
        summary = order_service.get_order_summary(order.id)
        if summary:
            print(f"   ✓ Order status: {summary['status']}")
            print(f"   ✓ Payment status: {summary['payment_status']}")
            print(f"   ✓ Items count: {len(summary['items'])}")
    else:
        print("   ✗ Order creation failed")
    
    print()
    
    # 4. Test different pricing strategies (OCP - Open/Closed Principle)
    print("4. TESTING DIFFERENT PRICING STRATEGIES (OCP):")
    
    pricing_strategies = [
        StandardPricing(),
        VIPCustomerPricing(20.0),
        BulkOrderPricing(3, 10.0),
        SeasonalPricing(0.25, datetime.now() + timedelta(days=30))
    ]
    
    test_order_items = [
        {'product_id': 'PROD001', 'quantity': 1},
        {'product_id': 'PROD002', 'quantity': 3}
    ]
    
    for i, strategy in enumerate(pricing_strategies):
        print(f"\n   Strategy {i+1}: {strategy.get_description()}")
        
        # Create new order service with different pricing strategy
        test_order_service = OrderManagementService(
            order_repository, product_repository, user_repository,
            inventory_service, payment_service, notification_service,
            audit_service, strategy
        )
        
        test_order = test_order_service.create_order(customer.id, shipping_address, test_order_items)
        if test_order:
            print(f"     Order total: {test_order.total_amount}")
            # Clean up test order
            order_repository.orders.pop(test_order.id, None)
    
    print()
    
    # 5. Test system with different implementations (LSP - Liskov Substitution)
    print("5. TESTING SUBSTITUTABILITY (LSP):")
    
    # Test with different notification service
    class SMSNotificationService(NotificationService):
        def __init__(self):
            self.sent_messages = []
        
        def send_order_confirmation(self, order: Order) -> bool:
            self.sent_messages.append(f"SMS: Order {order.id} confirmed")
            print(f"SMS sent to {order.customer.username}: Order confirmed")
            return True
        
        def send_payment_notification(self, order: Order, payment_status: PaymentStatus) -> bool:
            self.sent_messages.append(f"SMS: Payment {payment_status.value}")
            print(f"SMS sent: Payment {payment_status.value}")
            return True
        
        def send_shipping_notification(self, order: Order, tracking_number: str) -> bool:
            self.sent_messages.append(f"SMS: Shipped {tracking_number}")
            print(f"SMS sent: Shipped {tracking_number}")
            return True
    
    # Replace notification service (LSP - fully substitutable)
    sms_service = SMSNotificationService()
    order_service.notification_service = sms_service
    
    print("   ✓ Replaced EmailNotificationService with SMSNotificationService")
    print("   ✓ System continues to work without any changes (LSP compliance)")
    
    # Test order creation with new notification service
    test_order = order_service.create_order(customer.id, shipping_address, [{'product_id': 'PROD004', 'quantity': 1}])
    if test_order:
        print(f"   ✓ Order created with SMS notifications: {test_order.id}")
    
    print()
    
    # 6. Show interface segregation (ISP)
    print("6. INTERFACE SEGREGATION BENEFITS (ISP):")
    
    print("   ✓ AuthenticationService: Only authentication methods")
    print("   ✓ AuthorizationService: Only authorization methods")
    print("   ✓ NotificationService: Only notification methods")
    print("   ✓ PaymentService: Only payment processing methods")
    print("   ✓ InventoryService: Only inventory management methods")
    print("   ✓ AuditService: Only audit logging methods")
    print("   ✓ Each service interface is focused and cohesive")
    print("   ✓ Clients only depend on methods they actually use")
    print()
    
    # 7. System statistics and audit trail
    print("7. ENTERPRISE SYSTEM STATISTICS:")
    
    all_orders = order_repository.find_all()
    all_products = product_repository.find_available_products()
    all_customers = user_repository.find_by_role(UserRole.CUSTOMER)
    
    print(f"   Total Orders: {len(all_orders)}")
    print(f"   Available Products: {len(all_products)}")
    print(f"   Total Customers: {len(all_customers)}")
    print(f"   Notifications Sent: {len(notification_service.sent_notifications) + len(sms_service.sent_messages)}")
    print(f"   Payments Processed: {len(payment_service.processed_payments)}")
    print(f"   Audit Log Entries: {len(audit_service.audit_logs)}")
    
    print("\n   Recent Audit Trail:")
    for log_entry in audit_service.audit_logs[-5:]:  # Show last 5 entries
        print(f"     {log_entry['timestamp'][:19]}: {log_entry.get('action', log_entry.get('event'))}")
    
    print()
    
    # 8. Benefits in enterprise systems
    print("8. SOLID BENEFITS IN ENTERPRISE SYSTEMS:")
    print("   Scalability:")
    print("   • Easy to add new services without changing existing code")
    print("   • Microservices can be developed independently")
    print("   • Horizontal scaling of individual components")
    print()
    print("   Maintainability:")
    print("   • Clear separation of concerns")
    print("   • Changes are localized to specific components")
    print("   • Easy to understand and debug")
    print()
    print("   Testability:")
    print("   • Each component can be tested in isolation")
    print("   • Mock implementations for external dependencies")
    print("   • Comprehensive test coverage possible")
    print()
    print("   Flexibility:")
    print("   • Easy to swap implementations")
    print("   • Support for different deployment environments")
    print("   • Plugin architectures and extensibility")
    print()
    print("   Team Productivity:")
    print("   • Multiple teams can work on different components")
    print("   • Clear interfaces reduce integration issues")
    print("   • Faster development and deployment cycles")
    print()
    
    print("=== SOLID IN REAL SYSTEMS DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_solid_in_real_systems()
