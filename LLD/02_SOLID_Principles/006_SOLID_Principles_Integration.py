"""
SOLID PRINCIPLES INTEGRATION - Combining All SOLID Principles
=============================================================

Problem Statement:
Demonstrate how all SOLID principles work together:
- Integrating SRP, OCP, LSP, ISP, and DIP in a cohesive system
- Building a complete application following all SOLID principles
- Showing how principles complement each other
- Real-world example with e-commerce system
- Best practices for SOLID implementation

Learning Objectives:
- Understand how SOLID principles work together
- Build systems that follow all SOLID principles
- See the synergy between different principles
- Apply SOLID principles in complex scenarios
- Design maintainable and extensible architectures
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import uuid


# ENUMS AND VALUE OBJECTS
class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class PaymentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


@dataclass
class Money:
    """Value object for money (SRP - single responsibility for money representation)."""
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


# INTERFACES (ISP - Interface Segregation Principle)

class Identifiable(Protocol):
    """Interface for objects with ID."""
    id: str


class Timestamped(Protocol):
    """Interface for objects with timestamps."""
    created_at: datetime
    updated_at: datetime


class Persistable(Protocol):
    """Interface for objects that can be persisted."""
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Persistable': ...


class Validatable(Protocol):
    """Interface for objects that can be validated."""
    def validate(self) -> bool: ...
    def get_validation_errors(self) -> List[str]: ...


# REPOSITORY INTERFACES (DIP - Dependency Inversion Principle)

class Repository(ABC):
    """Generic repository interface."""
    
    @abstractmethod
    def save(self, entity: Any) -> bool:
        """Save entity."""
        pass
    
    @abstractmethod
    def find_by_id(self, entity_id: str) -> Optional[Any]:
        """Find entity by ID."""
        pass
    
    @abstractmethod
    def find_all(self) -> List[Any]:
        """Find all entities."""
        pass
    
    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """Delete entity."""
        pass


class ProductRepository(Repository):
    """Product repository interface."""
    
    @abstractmethod
    def find_by_category(self, category: str) -> List['Product']:
        """Find products by category."""
        pass
    
    @abstractmethod
    def find_in_stock(self) -> List['Product']:
        """Find products in stock."""
        pass


class OrderRepository(Repository):
    """Order repository interface."""
    
    @abstractmethod
    def find_by_customer(self, customer_id: str) -> List['Order']:
        """Find orders by customer."""
        pass
    
    @abstractmethod
    def find_by_status(self, status: OrderStatus) -> List['Order']:
        """Find orders by status."""
        pass


# SERVICE INTERFACES (ISP - Interface Segregation)

class NotificationService(ABC):
    """Notification service interface."""
    
    @abstractmethod
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        """Send notification."""
        pass


class PaymentProcessor(ABC):
    """Payment processor interface."""
    
    @abstractmethod
    def process_payment(self, amount: Money, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        """Process payment."""
        pass
    
    @abstractmethod
    def refund_payment(self, transaction_id: str, amount: Money) -> Dict[str, Any]:
        """Refund payment."""
        pass


class InventoryService(ABC):
    """Inventory service interface."""
    
    @abstractmethod
    def check_availability(self, product_id: str, quantity: int) -> bool:
        """Check product availability."""
        pass
    
    @abstractmethod
    def reserve_items(self, product_id: str, quantity: int) -> bool:
        """Reserve items."""
        pass
    
    @abstractmethod
    def release_reservation(self, product_id: str, quantity: int) -> bool:
        """Release reservation."""
        pass


class Logger(ABC):
    """Logger interface."""
    
    @abstractmethod
    def log(self, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log message."""
        pass


# DOMAIN ENTITIES (SRP - Single Responsibility Principle)

class Product:
    """Product entity (SRP - represents product data only)."""
    
    def __init__(self, product_id: str, name: str, price: Money, category: str, stock_quantity: int = 0):
        self.id = product_id
        self.name = name
        self.price = price
        self.category = category
        self.stock_quantity = stock_quantity
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.is_active = True
    
    def update_price(self, new_price: Money) -> None:
        """Update product price."""
        self.price = new_price
        self.updated_at = datetime.now()
    
    def update_stock(self, new_quantity: int) -> None:
        """Update stock quantity."""
        if new_quantity < 0:
            raise ValueError("Stock quantity cannot be negative")
        self.stock_quantity = new_quantity
        self.updated_at = datetime.now()
    
    def is_in_stock(self, required_quantity: int = 1) -> bool:
        """Check if product is in stock."""
        return self.stock_quantity >= required_quantity and self.is_active
    
    def validate(self) -> bool:
        """Validate product data."""
        return (self.name.strip() != "" and 
                self.price.amount > 0 and 
                self.category.strip() != "" and
                self.stock_quantity >= 0)
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        errors = []
        if not self.name.strip():
            errors.append("Product name cannot be empty")
        if self.price.amount <= 0:
            errors.append("Product price must be positive")
        if not self.category.strip():
            errors.append("Product category cannot be empty")
        if self.stock_quantity < 0:
            errors.append("Stock quantity cannot be negative")
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'price': {'amount': self.price.amount, 'currency': self.price.currency},
            'category': self.category,
            'stock_quantity': self.stock_quantity,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Product':
        """Create from dictionary."""
        price_data = data['price']
        price = Money(price_data['amount'], price_data['currency'])
        
        product = cls(data['id'], data['name'], price, data['category'], data['stock_quantity'])
        product.created_at = datetime.fromisoformat(data['created_at'])
        product.updated_at = datetime.fromisoformat(data['updated_at'])
        product.is_active = data['is_active']
        return product


class OrderItem:
    """Order item entity (SRP - represents order item data only)."""
    
    def __init__(self, product: Product, quantity: int, unit_price: Money):
        self.product = product
        self.quantity = quantity
        self.unit_price = unit_price
        self.total_price = unit_price.multiply(quantity)
    
    def validate(self) -> bool:
        """Validate order item."""
        return self.quantity > 0 and self.unit_price.amount > 0
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        errors = []
        if self.quantity <= 0:
            errors.append("Quantity must be positive")
        if self.unit_price.amount <= 0:
            errors.append("Unit price must be positive")
        return errors


class Customer:
    """Customer entity (SRP - represents customer data only)."""
    
    def __init__(self, customer_id: str, name: str, email: str, phone: str):
        self.id = customer_id
        self.name = name
        self.email = email
        self.phone = phone
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.is_active = True
    
    def update_contact_info(self, email: str = None, phone: str = None) -> None:
        """Update contact information."""
        if email:
            self.email = email
        if phone:
            self.phone = phone
        self.updated_at = datetime.now()
    
    def validate(self) -> bool:
        """Validate customer data."""
        return (self.name.strip() != "" and 
                "@" in self.email and 
                self.phone.strip() != "")
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        errors = []
        if not self.name.strip():
            errors.append("Customer name cannot be empty")
        if "@" not in self.email:
            errors.append("Invalid email format")
        if not self.phone.strip():
            errors.append("Phone number cannot be empty")
        return errors


class Order:
    """Order entity (SRP - represents order data and basic operations)."""
    
    def __init__(self, order_id: str, customer: Customer):
        self.id = order_id
        self.customer = customer
        self.items: List[OrderItem] = []
        self.status = OrderStatus.PENDING
        self.payment_status = PaymentStatus.PENDING
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.subtotal = Money(0)
        self.tax_amount = Money(0)
        self.total_amount = Money(0)
    
    def add_item(self, product: Product, quantity: int) -> bool:
        """Add item to order."""
        if not product.is_in_stock(quantity):
            return False
        
        order_item = OrderItem(product, quantity, product.price)
        if order_item.validate():
            self.items.append(order_item)
            self._recalculate_totals()
            self.updated_at = datetime.now()
            return True
        return False
    
    def remove_item(self, product_id: str) -> bool:
        """Remove item from order."""
        for i, item in enumerate(self.items):
            if item.product.id == product_id:
                self.items.pop(i)
                self._recalculate_totals()
                self.updated_at = datetime.now()
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
        self.total_amount = self.subtotal.add(self.tax_amount)
    
    def validate(self) -> bool:
        """Validate order."""
        return (len(self.items) > 0 and 
                self.customer.validate() and
                all(item.validate() for item in self.items))
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        errors = []
        if len(self.items) == 0:
            errors.append("Order must have at least one item")
        
        customer_errors = self.customer.get_validation_errors()
        errors.extend([f"Customer: {error}" for error in customer_errors])
        
        for i, item in enumerate(self.items):
            item_errors = item.get_validation_errors()
            errors.extend([f"Item {i+1}: {error}" for error in item_errors])
        
        return errors


# DISCOUNT STRATEGIES (OCP - Open/Closed Principle)

class DiscountStrategy(ABC):
    """Abstract discount strategy (OCP - open for extension)."""
    
    @abstractmethod
    def calculate_discount(self, order: Order) -> Money:
        """Calculate discount amount."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get discount description."""
        pass


class NoDiscount(DiscountStrategy):
    """No discount strategy."""
    
    def calculate_discount(self, order: Order) -> Money:
        return Money(0, order.subtotal.currency)
    
    def get_description(self) -> str:
        return "No discount"


class PercentageDiscount(DiscountStrategy):
    """Percentage discount strategy."""
    
    def __init__(self, percentage: float, min_amount: Money = None):
        self.percentage = percentage
        self.min_amount = min_amount or Money(0)
    
    def calculate_discount(self, order: Order) -> Money:
        if order.subtotal.amount >= self.min_amount.amount:
            discount_amount = order.subtotal.amount * (self.percentage / 100)
            return Money(discount_amount, order.subtotal.currency)
        return Money(0, order.subtotal.currency)
    
    def get_description(self) -> str:
        if self.min_amount.amount > 0:
            return f"{self.percentage}% off orders over {self.min_amount}"
        return f"{self.percentage}% off"


class FixedAmountDiscount(DiscountStrategy):
    """Fixed amount discount strategy."""
    
    def __init__(self, discount_amount: Money, min_amount: Money = None):
        self.discount_amount = discount_amount
        self.min_amount = min_amount or Money(0)
    
    def calculate_discount(self, order: Order) -> Money:
        if order.subtotal.amount >= self.min_amount.amount:
            return Money(
                min(self.discount_amount.amount, order.subtotal.amount),
                order.subtotal.currency
            )
        return Money(0, order.subtotal.currency)
    
    def get_description(self) -> str:
        if self.min_amount.amount > 0:
            return f"{self.discount_amount} off orders over {self.min_amount}"
        return f"{self.discount_amount} off"


# CONCRETE IMPLEMENTATIONS (LSP - Liskov Substitution Principle)

class InMemoryProductRepository(ProductRepository):
    """In-memory product repository (LSP - substitutable for ProductRepository)."""
    
    def __init__(self):
        self.products: Dict[str, Product] = {}
    
    def save(self, product: Product) -> bool:
        if product.validate():
            self.products[product.id] = product
            return True
        return False
    
    def find_by_id(self, product_id: str) -> Optional[Product]:
        return self.products.get(product_id)
    
    def find_all(self) -> List[Product]:
        return list(self.products.values())
    
    def delete(self, product_id: str) -> bool:
        if product_id in self.products:
            del self.products[product_id]
            return True
        return False
    
    def find_by_category(self, category: str) -> List[Product]:
        return [p for p in self.products.values() if p.category == category and p.is_active]
    
    def find_in_stock(self) -> List[Product]:
        return [p for p in self.products.values() if p.stock_quantity > 0 and p.is_active]


class InMemoryOrderRepository(OrderRepository):
    """In-memory order repository (LSP - substitutable for OrderRepository)."""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
    
    def save(self, order: Order) -> bool:
        if order.validate():
            self.orders[order.id] = order
            return True
        return False
    
    def find_by_id(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)
    
    def find_all(self) -> List[Order]:
        return list(self.orders.values())
    
    def delete(self, order_id: str) -> bool:
        if order_id in self.orders:
            del self.orders[order_id]
            return True
        return False
    
    def find_by_customer(self, customer_id: str) -> List[Order]:
        return [o for o in self.orders.values() if o.customer.id == customer_id]
    
    def find_by_status(self, status: OrderStatus) -> List[Order]:
        return [o for o in self.orders.values() if o.status == status]


class EmailNotificationService(NotificationService):
    """Email notification service (LSP - substitutable for NotificationService)."""
    
    def __init__(self):
        self.sent_notifications = []
    
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        notification = {
            'recipient': recipient,
            'subject': subject,
            'message': message,
            'sent_at': datetime.now().isoformat(),
            'type': 'email'
        }
        self.sent_notifications.append(notification)
        print(f"Email sent to {recipient}: {subject}")
        return True


class MockPaymentProcessor(PaymentProcessor):
    """Mock payment processor for testing (LSP - substitutable for PaymentProcessor)."""
    
    def __init__(self):
        self.processed_payments = []
    
    def process_payment(self, amount: Money, payment_details: Dict[str, Any]) -> Dict[str, Any]:
        transaction_id = str(uuid.uuid4())
        payment_record = {
            'transaction_id': transaction_id,
            'amount': amount.amount,
            'currency': amount.currency,
            'status': 'completed',
            'processed_at': datetime.now().isoformat()
        }
        self.processed_payments.append(payment_record)
        print(f"Payment processed: {amount}")
        return payment_record
    
    def refund_payment(self, transaction_id: str, amount: Money) -> Dict[str, Any]:
        refund_record = {
            'refund_id': str(uuid.uuid4()),
            'original_transaction_id': transaction_id,
            'amount': amount.amount,
            'currency': amount.currency,
            'status': 'completed',
            'processed_at': datetime.now().isoformat()
        }
        print(f"Refund processed: {amount}")
        return refund_record


class SimpleInventoryService(InventoryService):
    """Simple inventory service (LSP - substitutable for InventoryService)."""
    
    def __init__(self, product_repository: ProductRepository):
        self.product_repository = product_repository
        self.reservations: Dict[str, int] = {}
    
    def check_availability(self, product_id: str, quantity: int) -> bool:
        product = self.product_repository.find_by_id(product_id)
        if not product:
            return False
        
        reserved = self.reservations.get(product_id, 0)
        available = product.stock_quantity - reserved
        return available >= quantity
    
    def reserve_items(self, product_id: str, quantity: int) -> bool:
        if self.check_availability(product_id, quantity):
            self.reservations[product_id] = self.reservations.get(product_id, 0) + quantity
            return True
        return False
    
    def release_reservation(self, product_id: str, quantity: int) -> bool:
        if product_id in self.reservations:
            current_reservation = self.reservations[product_id]
            if current_reservation >= quantity:
                self.reservations[product_id] = current_reservation - quantity
                if self.reservations[product_id] == 0:
                    del self.reservations[product_id]
                return True
        return False


class ConsoleLogger(Logger):
    """Console logger (LSP - substitutable for Logger)."""
    
    def __init__(self):
        self.log_entries = []
    
    def log(self, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"
        if context:
            log_entry += f" | Context: {context}"
        
        self.log_entries.append(log_entry)
        print(log_entry)


# HIGH-LEVEL SERVICES (DIP - Dependency Inversion Principle)

class OrderService:
    """
    Order service (DIP - depends on abstractions, not concretions).
    Integrates all SOLID principles.
    """
    
    def __init__(self,
                 order_repository: OrderRepository,
                 product_repository: ProductRepository,
                 inventory_service: InventoryService,
                 payment_processor: PaymentProcessor,
                 notification_service: NotificationService,
                 logger: Logger):
        # DIP: Depending on abstractions
        self.order_repository = order_repository
        self.product_repository = product_repository
        self.inventory_service = inventory_service
        self.payment_processor = payment_processor
        self.notification_service = notification_service
        self.logger = logger
        self.discount_strategy: DiscountStrategy = NoDiscount()
    
    def set_discount_strategy(self, strategy: DiscountStrategy) -> None:
        """Set discount strategy (OCP - open for extension)."""
        self.discount_strategy = strategy
    
    def create_order(self, customer: Customer, items: List[Dict[str, Any]]) -> Optional[Order]:
        """Create new order (SRP - single responsibility for order creation)."""
        
        self.logger.log("INFO", f"Creating order for customer {customer.id}")
        
        # Validate customer
        if not customer.validate():
            errors = customer.get_validation_errors()
            self.logger.log("ERROR", f"Invalid customer data: {errors}")
            return None
        
        # Create order
        order_id = str(uuid.uuid4())
        order = Order(order_id, customer)
        
        # Add items and check inventory
        for item_data in items:
            product_id = item_data['product_id']
            quantity = item_data['quantity']
            
            product = self.product_repository.find_by_id(product_id)
            if not product:
                self.logger.log("ERROR", f"Product not found: {product_id}")
                return None
            
            if not self.inventory_service.check_availability(product_id, quantity):
                self.logger.log("ERROR", f"Insufficient inventory for product {product_id}")
                return None
            
            if not order.add_item(product, quantity):
                self.logger.log("ERROR", f"Failed to add item to order: {product_id}")
                return None
            
            # Reserve inventory
            if not self.inventory_service.reserve_items(product_id, quantity):
                self.logger.log("ERROR", f"Failed to reserve inventory for {product_id}")
                return None
        
        # Apply discount
        discount_amount = self.discount_strategy.calculate_discount(order)
        if discount_amount.amount > 0:
            order.total_amount = order.total_amount.subtract(discount_amount)
            self.logger.log("INFO", f"Applied discount: {self.discount_strategy.get_description()}")
        
        # Save order
        if self.order_repository.save(order):
            self.logger.log("INFO", f"Order created successfully: {order_id}")
            
            # Send confirmation
            self.notification_service.send_notification(
                customer.email,
                "Order Confirmation",
                f"Your order {order_id} has been created. Total: {order.total_amount}"
            )
            
            return order
        else:
            self.logger.log("ERROR", f"Failed to save order: {order_id}")
            return None
    
    def process_payment(self, order_id: str, payment_details: Dict[str, Any]) -> bool:
        """Process payment for order."""
        
        self.logger.log("INFO", f"Processing payment for order {order_id}")
        
        order = self.order_repository.find_by_id(order_id)
        if not order:
            self.logger.log("ERROR", f"Order not found: {order_id}")
            return False
        
        if order.payment_status != PaymentStatus.PENDING:
            self.logger.log("ERROR", f"Order payment already processed: {order_id}")
            return False
        
        # Process payment
        order.update_payment_status(PaymentStatus.PROCESSING)
        self.order_repository.save(order)
        
        try:
            payment_result = self.payment_processor.process_payment(order.total_amount, payment_details)
            
            if payment_result.get('status') == 'completed':
                order.update_payment_status(PaymentStatus.COMPLETED)
                order.update_status(OrderStatus.CONFIRMED)
                self.order_repository.save(order)
                
                self.logger.log("INFO", f"Payment completed for order {order_id}")
                
                # Send confirmation
                self.notification_service.send_notification(
                    order.customer.email,
                    "Payment Confirmation",
                    f"Payment for order {order_id} has been processed successfully."
                )
                
                return True
            else:
                order.update_payment_status(PaymentStatus.FAILED)
                self.order_repository.save(order)
                self.logger.log("ERROR", f"Payment failed for order {order_id}")
                return False
                
        except Exception as e:
            order.update_payment_status(PaymentStatus.FAILED)
            self.order_repository.save(order)
            self.logger.log("ERROR", f"Payment processing error for order {order_id}: {str(e)}")
            return False
    
    def get_order_summary(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order summary."""
        order = self.order_repository.find_by_id(order_id)
        if not order:
            return None
        
        return {
            'order_id': order.id,
            'customer': order.customer.name,
            'status': order.status.value,
            'payment_status': order.payment_status.value,
            'items': [
                {
                    'product_name': item.product.name,
                    'quantity': item.quantity,
                    'unit_price': str(item.unit_price),
                    'total_price': str(item.total_price)
                }
                for item in order.items
            ],
            'subtotal': str(order.subtotal),
            'tax_amount': str(order.tax_amount),
            'total_amount': str(order.total_amount),
            'discount_applied': self.discount_strategy.get_description(),
            'created_at': order.created_at.isoformat()
        }


def demonstrate_solid_principles_integration():
    """
    Demonstrate how all SOLID principles work together in a complete system.
    """
    print("=== SOLID PRINCIPLES INTEGRATION DEMONSTRATION ===\n")
    
    print("Building an E-commerce System Following All SOLID Principles:")
    print("• SRP: Each class has a single responsibility")
    print("• OCP: Discount strategies are open for extension")
    print("• LSP: All implementations are substitutable")
    print("• ISP: Interfaces are focused and specific")
    print("• DIP: High-level modules depend on abstractions")
    print()
    
    # 1. Setup dependencies (DIP - Dependency Injection)
    print("1. SETTING UP DEPENDENCIES (DIP):")
    
    product_repository = InMemoryProductRepository()
    order_repository = InMemoryOrderRepository()
    inventory_service = SimpleInventoryService(product_repository)
    payment_processor = MockPaymentProcessor()
    notification_service = EmailNotificationService()
    logger = ConsoleLogger()
    
    print("   ✓ Created all service implementations")
    print("   ✓ Services depend on abstractions, not concrete classes")
    print()
    
    # 2. Create products (SRP - Product entity)
    print("2. CREATING PRODUCTS (SRP):")
    
    products = [
        Product("PROD001", "Laptop", Money(999.99), "Electronics", 10),
        Product("PROD002", "Mouse", Money(29.99), "Electronics", 50),
        Product("PROD003", "Keyboard", Money(79.99), "Electronics", 25),
        Product("PROD004", "Monitor", Money(299.99), "Electronics", 15)
    ]
    
    for product in products:
        product_repository.save(product)
        print(f"   ✓ Created product: {product.name} - {product.price}")
    
    print()
    
    # 3. Create order service with all dependencies
    print("3. CREATING ORDER SERVICE (DIP + All Principles):")
    
    order_service = OrderService(
        order_repository,
        product_repository,
        inventory_service,
        payment_processor,
        notification_service,
        logger
    )
    
    print("   ✓ OrderService created with injected dependencies")
    print("   ✓ Service can work with any implementation of the interfaces")
    print()
    
    # 4. Test different discount strategies (OCP)
    print("4. TESTING DISCOUNT STRATEGIES (OCP):")
    
    discount_strategies = [
        NoDiscount(),
        PercentageDiscount(10, Money(100)),
        FixedAmountDiscount(Money(50), Money(200))
    ]
    
    customer = Customer("CUST001", "John Doe", "john@example.com", "+1234567890")
    
    for i, strategy in enumerate(discount_strategies):
        print(f"\n   Test {i+1}: {strategy.get_description()}")
        
        # Set discount strategy (OCP - open for extension)
        order_service.set_discount_strategy(strategy)
        
        # Create order
        order_items = [
            {'product_id': 'PROD001', 'quantity': 1},  # $999.99
            {'product_id': 'PROD002', 'quantity': 2},  # $59.98
            {'product_id': 'PROD003', 'quantity': 1}   # $79.99
        ]
        
        order = order_service.create_order(customer, order_items)
        
        if order:
            print(f"     Order created: {order.id}")
            print(f"     Subtotal: {order.subtotal}")
            print(f"     Total: {order.total_amount}")
            
            # Process payment
            payment_success = order_service.process_payment(order.id, {
                'card_number': '1234567890123456',
                'expiry': '12/25',
                'cvv': '123'
            })
            
            print(f"     Payment: {'Success' if payment_success else 'Failed'}")
            
            # Get order summary
            summary = order_service.get_order_summary(order.id)
            if summary:
                print(f"     Status: {summary['status']}")
                print(f"     Payment Status: {summary['payment_status']}")
        else:
            print("     Order creation failed")
    
    print()
    
    # 5. Show substitutability (LSP)
    print("5. DEMONSTRATING SUBSTITUTABILITY (LSP):")
    
    # Create different implementations
    class SMSNotificationService(NotificationService):
        def __init__(self):
            self.sent_messages = []
        
        def send_notification(self, recipient: str, subject: str, message: str) -> bool:
            self.sent_messages.append({
                'recipient': recipient,
                'subject': subject,
                'message': message,
                'sent_at': datetime.now().isoformat()
            })
            print(f"SMS sent to {recipient}: {subject}")
            return True
    
    # Replace notification service (LSP - substitutable)
    sms_service = SMSNotificationService()
    order_service.notification_service = sms_service
    
    print("   ✓ Replaced EmailNotificationService with SMSNotificationService")
    print("   ✓ OrderService works without any changes (LSP compliance)")
    
    # Test with new service
    test_order = order_service.create_order(customer, [{'product_id': 'PROD004', 'quantity': 1}])
    if test_order:
        print(f"   ✓ Order created with SMS notifications: {test_order.id}")
    
    print()
    
    # 6. Show interface segregation (ISP)
    print("6. INTERFACE SEGREGATION (ISP):")
    
    print("   ✓ NotificationService: Only notification methods")
    print("   ✓ PaymentProcessor: Only payment methods")
    print("   ✓ InventoryService: Only inventory methods")
    print("   ✓ Repository: Only data persistence methods")
    print("   ✓ Logger: Only logging methods")
    print("   ✓ Each interface is focused and cohesive")
    print()
    
    # 7. Show single responsibility (SRP)
    print("7. SINGLE RESPONSIBILITY PRINCIPLE (SRP):")
    
    print("   ✓ Product: Only product data and operations")
    print("   ✓ Order: Only order data and operations")
    print("   ✓ Customer: Only customer data and operations")
    print("   ✓ OrderService: Only order business logic")
    print("   ✓ Repository: Only data persistence")
    print("   ✓ NotificationService: Only sending notifications")
    print("   ✓ PaymentProcessor: Only payment processing")
    print("   ✓ Each class has one reason to change")
    print()
    
    # 8. Show system statistics
    print("8. SYSTEM STATISTICS:")
    
    all_orders = order_repository.find_all()
    all_products = product_repository.find_all()
    
    print(f"   Total Orders: {len(all_orders)}")
    print(f"   Total Products: {len(all_products)}")
    print(f"   Notifications Sent: {len(notification_service.sent_notifications) + len(sms_service.sent_messages)}")
    print(f"   Payments Processed: {len(payment_processor.processed_payments)}")
    print(f"   Log Entries: {len(logger.log_entries)}")
    
    print()
    
    # 9. Benefits summary
    print("9. SOLID PRINCIPLES BENEFITS IN THIS SYSTEM:")
    print("   ✓ SRP: Easy to understand and maintain each class")
    print("   ✓ OCP: Can add new discount strategies without changing existing code")
    print("   ✓ LSP: Can swap implementations without breaking functionality")
    print("   ✓ ISP: Clients only depend on methods they actually use")
    print("   ✓ DIP: High-level logic is independent of implementation details")
    print()
    print("   System Benefits:")
    print("   • Highly testable with mock implementations")
    print("   • Easy to extend with new features")
    print("   • Loose coupling between components")
    print("   • Clear separation of concerns")
    print("   • Maintainable and flexible architecture")
    print()
    
    print("=== SOLID PRINCIPLES INTEGRATION DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_solid_principles_integration()
