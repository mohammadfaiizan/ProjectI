"""
OOPS BEST PRACTICES - Clean Object-Oriented Design
==================================================

Problem Statement:
Demonstrate object-oriented programming best practices:
- SOLID principles in practice
- Clean code principles for OOP
- Design patterns usage guidelines
- Common OOP anti-patterns to avoid
- Testing strategies for OOP code

Learning Objectives:
- Apply SOLID principles consistently
- Write clean, maintainable OOP code
- Use design patterns appropriately
- Avoid common OOP pitfalls
- Design testable object-oriented systems
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol, Union
from datetime import datetime
from enum import Enum
import logging
from dataclasses import dataclass
import json


# Best Practice 1: Use Enums for Constants
class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class PaymentStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


# Best Practice 2: Use Dataclasses for Simple Data Containers
@dataclass
class Address:
    """Address data class following best practices."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"
    
    def __post_init__(self):
        """Validate data after initialization."""
        if not self.street.strip():
            raise ValueError("Street cannot be empty")
        if not self.city.strip():
            raise ValueError("City cannot be empty")
        if len(self.zip_code) != 5 or not self.zip_code.isdigit():
            raise ValueError("Zip code must be 5 digits")
    
    def format_address(self) -> str:
        """Format address for display."""
        return f"{self.street}, {self.city}, {self.state} {self.zip_code}, {self.country}"


@dataclass
class Money:
    """Money value object following best practices."""
    amount: float
    currency: str = "USD"
    
    def __post_init__(self):
        """Validate money values."""
        if self.amount < 0:
            raise ValueError("Amount cannot be negative")
        if not self.currency or len(self.currency) != 3:
            raise ValueError("Currency must be 3-letter code")
    
    def add(self, other: 'Money') -> 'Money':
        """Add two money amounts."""
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)
    
    def subtract(self, other: 'Money') -> 'Money':
        """Subtract two money amounts."""
        if self.currency != other.currency:
            raise ValueError("Cannot subtract different currencies")
        return Money(self.amount - other.amount, self.currency)
    
    def __str__(self) -> str:
        return f"{self.currency} {self.amount:.2f}"


# Best Practice 3: Single Responsibility Principle
class EmailValidator:
    """Single responsibility: email validation only."""
    
    @staticmethod
    def is_valid(email: str) -> bool:
        """Validate email format."""
        return "@" in email and "." in email.split("@")[1]
    
    @staticmethod
    def normalize(email: str) -> str:
        """Normalize email to lowercase."""
        return email.lower().strip()


class PasswordValidator:
    """Single responsibility: password validation only."""
    
    @staticmethod
    def is_strong(password: str) -> bool:
        """Check if password is strong."""
        return (len(password) >= 8 and
                any(c.isupper() for c in password) and
                any(c.islower() for c in password) and
                any(c.isdigit() for c in password))
    
    @staticmethod
    def get_strength_score(password: str) -> int:
        """Get password strength score (0-5)."""
        score = 0
        if len(password) >= 8:
            score += 1
        if any(c.isupper() for c in password):
            score += 1
        if any(c.islower() for c in password):
            score += 1
        if any(c.isdigit() for c in password):
            score += 1
        if any(c in "!@#$%^&*" for c in password):
            score += 1
        return score


# Best Practice 4: Open/Closed Principle with Strategy Pattern
class DiscountStrategy(ABC):
    """Abstract discount strategy."""
    
    @abstractmethod
    def calculate_discount(self, amount: Money) -> Money:
        """Calculate discount amount."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get discount description."""
        pass


class PercentageDiscount(DiscountStrategy):
    """Percentage-based discount."""
    
    def __init__(self, percentage: float):
        if not 0 <= percentage <= 100:
            raise ValueError("Percentage must be between 0 and 100")
        self.percentage = percentage
    
    def calculate_discount(self, amount: Money) -> Money:
        """Calculate percentage discount."""
        discount_amount = amount.amount * (self.percentage / 100)
        return Money(discount_amount, amount.currency)
    
    def get_description(self) -> str:
        """Get discount description."""
        return f"{self.percentage}% discount"


class FixedAmountDiscount(DiscountStrategy):
    """Fixed amount discount."""
    
    def __init__(self, discount_amount: Money):
        self.discount_amount = discount_amount
    
    def calculate_discount(self, amount: Money) -> Money:
        """Calculate fixed discount."""
        if amount.currency != self.discount_amount.currency:
            raise ValueError("Currency mismatch")
        
        discount = min(self.discount_amount.amount, amount.amount)
        return Money(discount, amount.currency)
    
    def get_description(self) -> str:
        """Get discount description."""
        return f"{self.discount_amount} off"


class BuyOneGetOneDiscount(DiscountStrategy):
    """Buy one get one discount."""
    
    def __init__(self, item_price: Money):
        self.item_price = item_price
    
    def calculate_discount(self, amount: Money) -> Money:
        """Calculate BOGO discount."""
        if amount.currency != self.item_price.currency:
            raise ValueError("Currency mismatch")
        
        # Simple BOGO: 50% off if amount is at least 2x item price
        if amount.amount >= self.item_price.amount * 2:
            return Money(self.item_price.amount, amount.currency)
        return Money(0, amount.currency)
    
    def get_description(self) -> str:
        """Get discount description."""
        return "Buy One Get One Free"


# Best Practice 5: Dependency Inversion Principle
class Logger(ABC):
    """Abstract logger interface."""
    
    @abstractmethod
    def log(self, level: str, message: str) -> None:
        """Log message."""
        pass


class ConsoleLogger(Logger):
    """Console logger implementation."""
    
    def log(self, level: str, message: str) -> None:
        """Log to console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")


class FileLogger(Logger):
    """File logger implementation."""
    
    def __init__(self, filename: str):
        self.filename = filename
    
    def log(self, level: str, message: str) -> None:
        """Log to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filename, 'a') as f:
            f.write(f"[{timestamp}] {level}: {message}\n")


class NotificationService(ABC):
    """Abstract notification service."""
    
    @abstractmethod
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        """Send notification."""
        pass


class EmailNotificationService(NotificationService):
    """Email notification service."""
    
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        """Send email notification."""
        print(f"Email sent to {recipient}: {subject}")
        return True


# Best Practice 6: Composition over Inheritance
class Customer:
    """Customer class using composition."""
    
    def __init__(self, customer_id: str, name: str, email: str, address: Address):
        self.customer_id = customer_id
        self.name = name
        self.email = email
        self.address = address  # Composition
        self.created_at = datetime.now()
        self.is_active = True
    
    def update_address(self, new_address: Address) -> None:
        """Update customer address."""
        self.address = new_address
    
    def deactivate(self) -> None:
        """Deactivate customer."""
        self.is_active = False
    
    def get_display_name(self) -> str:
        """Get customer display name."""
        return f"{self.name} ({self.customer_id})"
    
    def __str__(self) -> str:
        return f"Customer({self.customer_id}, {self.name})"


class OrderItem:
    """Order item class."""
    
    def __init__(self, product_id: str, product_name: str, price: Money, quantity: int):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        self.product_id = product_id
        self.product_name = product_name
        self.price = price
        self.quantity = quantity
    
    def get_total_price(self) -> Money:
        """Get total price for this item."""
        return Money(self.price.amount * self.quantity, self.price.currency)
    
    def __str__(self) -> str:
        return f"{self.product_name} x{self.quantity} @ {self.price}"


class Order:
    """Order class demonstrating best practices."""
    
    def __init__(self, order_id: str, customer: Customer, logger: Logger):
        self.order_id = order_id
        self.customer = customer  # Composition
        self.items: List[OrderItem] = []
        self.status = OrderStatus.PENDING
        self.payment_status = PaymentStatus.PENDING
        self.created_at = datetime.now()
        self.discount_strategy: Optional[DiscountStrategy] = None
        self._logger = logger  # Dependency injection
    
    def add_item(self, item: OrderItem) -> None:
        """Add item to order."""
        if self.status != OrderStatus.PENDING:
            raise ValueError("Cannot modify confirmed order")
        
        self.items.append(item)
        self._logger.log("INFO", f"Added item {item.product_name} to order {self.order_id}")
    
    def remove_item(self, product_id: str) -> bool:
        """Remove item from order."""
        if self.status != OrderStatus.PENDING:
            raise ValueError("Cannot modify confirmed order")
        
        for i, item in enumerate(self.items):
            if item.product_id == product_id:
                removed_item = self.items.pop(i)
                self._logger.log("INFO", f"Removed item {removed_item.product_name} from order {self.order_id}")
                return True
        return False
    
    def apply_discount(self, discount_strategy: DiscountStrategy) -> None:
        """Apply discount strategy."""
        self.discount_strategy = discount_strategy
        self._logger.log("INFO", f"Applied discount: {discount_strategy.get_description()}")
    
    def calculate_subtotal(self) -> Money:
        """Calculate order subtotal."""
        if not self.items:
            return Money(0)
        
        total = Money(0, self.items[0].price.currency)
        for item in self.items:
            total = total.add(item.get_total_price())
        return total
    
    def calculate_discount(self) -> Money:
        """Calculate discount amount."""
        if not self.discount_strategy:
            return Money(0)
        
        subtotal = self.calculate_subtotal()
        return self.discount_strategy.calculate_discount(subtotal)
    
    def calculate_total(self) -> Money:
        """Calculate order total."""
        subtotal = self.calculate_subtotal()
        discount = self.calculate_discount()
        return subtotal.subtract(discount)
    
    def confirm_order(self) -> bool:
        """Confirm the order."""
        if not self.items:
            raise ValueError("Cannot confirm empty order")
        
        if self.status != OrderStatus.PENDING:
            raise ValueError("Order already confirmed")
        
        self.status = OrderStatus.CONFIRMED
        self._logger.log("INFO", f"Order {self.order_id} confirmed")
        return True
    
    def ship_order(self) -> bool:
        """Ship the order."""
        if self.status != OrderStatus.CONFIRMED:
            raise ValueError("Order must be confirmed before shipping")
        
        if self.payment_status != PaymentStatus.COMPLETED:
            raise ValueError("Payment must be completed before shipping")
        
        self.status = OrderStatus.SHIPPED
        self._logger.log("INFO", f"Order {self.order_id} shipped")
        return True
    
    def cancel_order(self) -> bool:
        """Cancel the order."""
        if self.status in [OrderStatus.SHIPPED, OrderStatus.DELIVERED]:
            raise ValueError("Cannot cancel shipped or delivered order")
        
        self.status = OrderStatus.CANCELLED
        self._logger.log("INFO", f"Order {self.order_id} cancelled")
        return True
    
    def get_order_summary(self) -> Dict[str, Any]:
        """Get order summary."""
        return {
            'order_id': self.order_id,
            'customer': self.customer.get_display_name(),
            'status': self.status.value,
            'payment_status': self.payment_status.value,
            'item_count': len(self.items),
            'subtotal': str(self.calculate_subtotal()),
            'discount': str(self.calculate_discount()),
            'total': str(self.calculate_total()),
            'created_at': self.created_at.isoformat()
        }
    
    def __str__(self) -> str:
        return f"Order({self.order_id}, {self.status.value}, {len(self.items)} items)"


# Best Practice 7: Factory Pattern for Complex Object Creation
class OrderFactory:
    """Factory for creating orders."""
    
    def __init__(self, logger: Logger, notification_service: NotificationService):
        self._logger = logger
        self._notification_service = notification_service
        self._order_counter = 0
    
    def create_order(self, customer: Customer) -> Order:
        """Create new order."""
        self._order_counter += 1
        order_id = f"ORD{self._order_counter:06d}"
        
        order = Order(order_id, customer, self._logger)
        self._logger.log("INFO", f"Created order {order_id} for customer {customer.name}")
        
        return order
    
    def create_order_with_items(self, customer: Customer, items: List[OrderItem]) -> Order:
        """Create order with initial items."""
        order = self.create_order(customer)
        
        for item in items:
            order.add_item(item)
        
        return order


# Best Practice 8: Service Layer Pattern
class OrderService:
    """Order service implementing business logic."""
    
    def __init__(self, order_factory: OrderFactory, notification_service: NotificationService, logger: Logger):
        self._order_factory = order_factory
        self._notification_service = notification_service
        self._logger = logger
        self._orders: Dict[str, Order] = {}
    
    def create_order(self, customer: Customer, items: List[OrderItem]) -> Order:
        """Create new order with items."""
        order = self._order_factory.create_order_with_items(customer, items)
        self._orders[order.order_id] = order
        
        # Send confirmation email
        self._notification_service.send_notification(
            customer.email,
            "Order Created",
            f"Your order {order.order_id} has been created."
        )
        
        return order
    
    def apply_discount_to_order(self, order_id: str, discount_strategy: DiscountStrategy) -> bool:
        """Apply discount to order."""
        order = self._orders.get(order_id)
        if not order:
            return False
        
        order.apply_discount(discount_strategy)
        return True
    
    def confirm_order(self, order_id: str) -> bool:
        """Confirm order."""
        order = self._orders.get(order_id)
        if not order:
            return False
        
        try:
            order.confirm_order()
            
            # Send confirmation email
            self._notification_service.send_notification(
                order.customer.email,
                "Order Confirmed",
                f"Your order {order_id} has been confirmed."
            )
            
            return True
        except ValueError as e:
            self._logger.log("ERROR", f"Failed to confirm order {order_id}: {str(e)}")
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_customer_orders(self, customer_id: str) -> List[Order]:
        """Get all orders for customer."""
        return [order for order in self._orders.values() 
                if order.customer.customer_id == customer_id]


# Best Practice 9: Value Objects and Immutability
class ProductId:
    """Value object for product ID."""
    
    def __init__(self, value: str):
        if not value or not value.strip():
            raise ValueError("Product ID cannot be empty")
        self._value = value.strip().upper()
    
    @property
    def value(self) -> str:
        """Get product ID value."""
        return self._value
    
    def __eq__(self, other) -> bool:
        """Check equality."""
        return isinstance(other, ProductId) and self._value == other._value
    
    def __hash__(self) -> int:
        """Get hash value."""
        return hash(self._value)
    
    def __str__(self) -> str:
        return self._value


# Anti-pattern examples to avoid
class BadOrder:
    """Example of what NOT to do - violates multiple principles."""
    
    def __init__(self):
        # Violates SRP - too many responsibilities
        self.order_id = None
        self.customer_name = None
        self.customer_email = None
        self.items = []
        self.total = 0
        self.discount = 0
        self.tax = 0
        self.shipping = 0
        
        # Violates DIP - depends on concrete classes
        self.logger = ConsoleLogger()  # Hard dependency
        
        # Violates OCP - hard to extend
        self.discount_type = "percentage"  # Magic string
    
    def calculate_total(self):
        """Violates SRP - business logic mixed with data."""
        # Complex calculation logic here
        pass
    
    def send_email(self):
        """Violates SRP - order shouldn't send emails."""
        # Email sending logic here
        pass
    
    def save_to_database(self):
        """Violates SRP - order shouldn't know about persistence."""
        # Database logic here
        pass


def demonstrate_oops_best_practices():
    """
    Demonstrate object-oriented programming best practices.
    """
    print("=== OOPS BEST PRACTICES DEMONSTRATION ===\n")
    
    # 1. Using Value Objects and Data Classes
    print("1. VALUE OBJECTS AND DATA CLASSES:")
    
    address = Address("123 Main St", "New York", "NY", "10001")
    print(f"Address: {address.format_address()}")
    
    price = Money(99.99, "USD")
    discount_amount = Money(10.00, "USD")
    final_price = price.subtract(discount_amount)
    print(f"Price: {price}, Discount: {discount_amount}, Final: {final_price}")
    print()
    
    # 2. Single Responsibility Principle
    print("2. SINGLE RESPONSIBILITY PRINCIPLE:")
    
    email = "USER@EXAMPLE.COM"
    if EmailValidator.is_valid(email):
        normalized_email = EmailValidator.normalize(email)
        print(f"Valid email: {normalized_email}")
    
    password = "MySecurePass123"
    if PasswordValidator.is_strong(password):
        strength = PasswordValidator.get_strength_score(password)
        print(f"Strong password with strength score: {strength}/5")
    print()
    
    # 3. Strategy Pattern (Open/Closed Principle)
    print("3. STRATEGY PATTERN FOR DISCOUNTS:")
    
    order_amount = Money(100.00, "USD")
    
    # Different discount strategies
    percentage_discount = PercentageDiscount(20)
    fixed_discount = FixedAmountDiscount(Money(15.00, "USD"))
    bogo_discount = BuyOneGetOneDiscount(Money(25.00, "USD"))
    
    strategies = [percentage_discount, fixed_discount, bogo_discount]
    
    for strategy in strategies:
        discount = strategy.calculate_discount(order_amount)
        final_amount = order_amount.subtract(discount)
        print(f"{strategy.get_description()}: {discount} off, Final: {final_amount}")
    print()
    
    # 4. Dependency Injection
    print("4. DEPENDENCY INJECTION:")
    
    # Create dependencies
    logger = ConsoleLogger()
    notification_service = EmailNotificationService()
    order_factory = OrderFactory(logger, notification_service)
    order_service = OrderService(order_factory, notification_service, logger)
    
    # Create customer
    customer_address = Address("456 Oak Ave", "San Francisco", "CA", "94102")
    customer = Customer("CUST001", "Alice Johnson", "alice@example.com", customer_address)
    
    # Create order items
    items = [
        OrderItem("PROD001", "Laptop", Money(999.99, "USD"), 1),
        OrderItem("PROD002", "Mouse", Money(29.99, "USD"), 2)
    ]
    
    # Create order through service
    order = order_service.create_order(customer, items)
    print(f"Created: {order}")
    print(f"Subtotal: {order.calculate_subtotal()}")
    
    # Apply discount
    discount_strategy = PercentageDiscount(10)
    order_service.apply_discount_to_order(order.order_id, discount_strategy)
    print(f"After discount: {order.calculate_total()}")
    
    # Confirm order
    success = order_service.confirm_order(order.order_id)
    print(f"Order confirmed: {success}")
    print()
    
    # 5. Order Summary
    print("5. ORDER SUMMARY:")
    summary = order.get_order_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()
    
    # 6. Best Practices Summary
    print("6. OOPS BEST PRACTICES SUMMARY:")
    print("✓ SOLID Principles:")
    print("  - Single Responsibility: Each class has one reason to change")
    print("  - Open/Closed: Open for extension, closed for modification")
    print("  - Liskov Substitution: Subtypes must be substitutable")
    print("  - Interface Segregation: Many specific interfaces > one general")
    print("  - Dependency Inversion: Depend on abstractions, not concretions")
    
    print("\n✓ Design Principles:")
    print("  - Composition over inheritance")
    print("  - Program to interfaces, not implementations")
    print("  - Encapsulate what varies")
    print("  - Favor immutable objects when possible")
    print("  - Use value objects for domain concepts")
    
    print("\n✓ Code Quality:")
    print("  - Clear and descriptive naming")
    print("  - Small, focused methods and classes")
    print("  - Proper error handling and validation")
    print("  - Comprehensive logging and monitoring")
    print("  - Testable design with dependency injection")
    
    print("\n✓ Anti-patterns to Avoid:")
    print("  - God objects (classes that do too much)")
    print("  - Tight coupling between classes")
    print("  - Magic numbers and strings")
    print("  - Deep inheritance hierarchies")
    print("  - Circular dependencies")
    print()
    
    print("=== OOPS BEST PRACTICES DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_oops_best_practices()
