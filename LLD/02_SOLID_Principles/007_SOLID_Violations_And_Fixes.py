"""
SOLID VIOLATIONS AND FIXES - Common Anti-patterns and Solutions
===============================================================

Problem Statement:
Demonstrate common SOLID principle violations and their fixes:
- Identifying SOLID violations in existing code
- Refactoring techniques to fix violations
- Before and after comparisons
- Common anti-patterns and code smells
- Step-by-step refactoring process

Learning Objectives:
- Recognize SOLID principle violations
- Apply refactoring techniques to fix violations
- Understand common anti-patterns
- Learn systematic refactoring approaches
- Improve code quality through SOLID principles
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from datetime import datetime
from enum import Enum
import json


# ============================================================================
# VIOLATION 1: SRP VIOLATION - God Class
# ============================================================================

class BadUserManager:
    """
    SRP VIOLATION: This class has too many responsibilities:
    1. User data management
    2. User validation
    3. Password hashing
    4. Email sending
    5. Database operations
    6. Logging
    7. Authentication
    8. Authorization
    """
    
    def __init__(self):
        self.users = {}
        self.logged_in_users = set()
        self.user_permissions = {}
        self.email_templates = {}
        self.log_entries = []
    
    def create_user(self, username: str, email: str, password: str, role: str) -> bool:
        """Creates user - doing too many things!"""
        
        # Responsibility 1: Logging
        self.log_entries.append(f"[{datetime.now()}] Creating user: {username}")
        
        # Responsibility 2: Validation
        if not self.validate_email(email):
            self.log_entries.append(f"[{datetime.now()}] Invalid email: {email}")
            return False
        
        if not self.validate_password(password):
            self.log_entries.append(f"[{datetime.now()}] Invalid password for: {username}")
            return False
        
        # Responsibility 3: Password hashing
        hashed_password = self.hash_password(password)
        
        # Responsibility 4: User data management
        user_data = {
            'username': username,
            'email': email,
            'password': hashed_password,
            'role': role,
            'created_at': datetime.now().isoformat()
        }
        
        # Responsibility 5: Database operations
        self.save_to_database(user_data)
        
        # Responsibility 6: Permission setup
        self.setup_user_permissions(username, role)
        
        # Responsibility 7: Email sending
        self.send_welcome_email(email, username)
        
        self.users[username] = user_data
        return True
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """Authentication logic mixed with other concerns."""
        self.log_entries.append(f"[{datetime.now()}] Auth attempt: {username}")
        
        if username not in self.users:
            return False
        
        user = self.users[username]
        if self.verify_password(password, user['password']):
            self.logged_in_users.add(username)
            self.log_entries.append(f"[{datetime.now()}] User logged in: {username}")
            return True
        
        return False
    
    def authorize_action(self, username: str, action: str) -> bool:
        """Authorization logic mixed with other concerns."""
        if username not in self.logged_in_users:
            return False
        
        user_perms = self.user_permissions.get(username, [])
        return action in user_perms
    
    # Too many methods in one class...
    def validate_email(self, email: str) -> bool:
        return "@" in email
    
    def validate_password(self, password: str) -> bool:
        return len(password) >= 8
    
    def hash_password(self, password: str) -> str:
        return f"hashed_{password}"
    
    def verify_password(self, password: str, hashed: str) -> bool:
        return f"hashed_{password}" == hashed
    
    def save_to_database(self, user_data: dict) -> None:
        print(f"Saving to database: {user_data['username']}")
    
    def setup_user_permissions(self, username: str, role: str) -> None:
        if role == "admin":
            self.user_permissions[username] = ["read", "write", "delete", "manage"]
        elif role == "user":
            self.user_permissions[username] = ["read", "write"]
    
    def send_welcome_email(self, email: str, username: str) -> None:
        print(f"Sending welcome email to {email}")


# SRP FIX: Separate responsibilities into focused classes

class User:
    """SRP FIX: Only responsible for user data representation."""
    
    def __init__(self, username: str, email: str, password_hash: str, role: str):
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.role = role
        self.created_at = datetime.now()


class UserValidator:
    """SRP FIX: Only responsible for user data validation."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        return "@" in email and "." in email.split("@")[1]
    
    @staticmethod
    def validate_password(password: str) -> bool:
        return (len(password) >= 8 and
                any(c.isupper() for c in password) and
                any(c.islower() for c in password) and
                any(c.isdigit() for c in password))


class PasswordManager:
    """SRP FIX: Only responsible for password operations."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        return f"secure_hash_{password}"
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return PasswordManager.hash_password(password) == hashed


class UserRepository:
    """SRP FIX: Only responsible for user data persistence."""
    
    def __init__(self):
        self.users = {}
    
    def save(self, user: User) -> bool:
        self.users[user.username] = user
        print(f"User saved: {user.username}")
        return True
    
    def find_by_username(self, username: str) -> Optional[User]:
        return self.users.get(username)


class EmailService:
    """SRP FIX: Only responsible for email operations."""
    
    def send_welcome_email(self, email: str, username: str) -> bool:
        print(f"Welcome email sent to {email}")
        return True


class AuthenticationService:
    """SRP FIX: Only responsible for authentication."""
    
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository
        self.logged_in_users = set()
    
    def authenticate(self, username: str, password: str) -> bool:
        user = self.user_repository.find_by_username(username)
        if user and PasswordManager.verify_password(password, user.password_hash):
            self.logged_in_users.add(username)
            return True
        return False
    
    def is_logged_in(self, username: str) -> bool:
        return username in self.logged_in_users


class AuthorizationService:
    """SRP FIX: Only responsible for authorization."""
    
    def __init__(self):
        self.permissions = {
            "admin": ["read", "write", "delete", "manage"],
            "user": ["read", "write"]
        }
    
    def authorize(self, user: User, action: str) -> bool:
        user_permissions = self.permissions.get(user.role, [])
        return action in user_permissions


class Logger:
    """SRP FIX: Only responsible for logging."""
    
    def __init__(self):
        self.log_entries = []
    
    def log(self, message: str) -> None:
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {message}"
        self.log_entries.append(log_entry)
        print(log_entry)


class GoodUserService:
    """SRP FIX: Coordinates other services but has single responsibility."""
    
    def __init__(self, user_repository: UserRepository, email_service: EmailService,
                 auth_service: AuthenticationService, authz_service: AuthorizationService,
                 logger: Logger):
        self.user_repository = user_repository
        self.email_service = email_service
        self.auth_service = auth_service
        self.authz_service = authz_service
        self.logger = logger
    
    def create_user(self, username: str, email: str, password: str, role: str) -> bool:
        self.logger.log(f"Creating user: {username}")
        
        if not UserValidator.validate_email(email):
            self.logger.log(f"Invalid email: {email}")
            return False
        
        if not UserValidator.validate_password(password):
            self.logger.log(f"Invalid password for: {username}")
            return False
        
        password_hash = PasswordManager.hash_password(password)
        user = User(username, email, password_hash, role)
        
        if self.user_repository.save(user):
            self.email_service.send_welcome_email(email, username)
            self.logger.log(f"User created successfully: {username}")
            return True
        
        return False


# ============================================================================
# VIOLATION 2: OCP VIOLATION - Modification for Extension
# ============================================================================

class BadDiscountCalculator:
    """
    OCP VIOLATION: Adding new discount types requires modifying this class.
    """
    
    def calculate_discount(self, order_amount: float, discount_type: str, **kwargs) -> float:
        """OCP VIOLATION: Must modify this method for each new discount type."""
        
        if discount_type == "percentage":
            return order_amount * (kwargs.get("percentage", 0) / 100)
        elif discount_type == "fixed":
            return min(kwargs.get("amount", 0), order_amount)
        elif discount_type == "buy_one_get_one":
            item_price = kwargs.get("item_price", 0)
            if order_amount >= item_price * 2:
                return item_price
            return 0
        # Adding new discount types requires modifying this method!
        # elif discount_type == "seasonal":
        #     return order_amount * 0.15
        else:
            return 0


# OCP FIX: Use strategy pattern for extensibility

class DiscountStrategy(ABC):
    """OCP FIX: Abstract strategy allows extension without modification."""
    
    @abstractmethod
    def calculate_discount(self, order_amount: float) -> float:
        pass


class PercentageDiscount(DiscountStrategy):
    """OCP FIX: Concrete strategy implementation."""
    
    def __init__(self, percentage: float):
        self.percentage = percentage
    
    def calculate_discount(self, order_amount: float) -> float:
        return order_amount * (self.percentage / 100)


class FixedDiscount(DiscountStrategy):
    """OCP FIX: Another concrete strategy."""
    
    def __init__(self, amount: float):
        self.amount = amount
    
    def calculate_discount(self, order_amount: float) -> float:
        return min(self.amount, order_amount)


class SeasonalDiscount(DiscountStrategy):
    """OCP FIX: New discount type added without modifying existing code."""
    
    def __init__(self, season_multiplier: float):
        self.season_multiplier = season_multiplier
    
    def calculate_discount(self, order_amount: float) -> float:
        return order_amount * self.season_multiplier


class GoodDiscountCalculator:
    """OCP FIX: Uses strategy pattern, closed for modification, open for extension."""
    
    def __init__(self, strategy: DiscountStrategy):
        self.strategy = strategy
    
    def calculate_discount(self, order_amount: float) -> float:
        return self.strategy.calculate_discount(order_amount)
    
    def set_strategy(self, strategy: DiscountStrategy) -> None:
        self.strategy = strategy


# ============================================================================
# VIOLATION 3: LSP VIOLATION - Behavioral Incompatibility
# ============================================================================

class BadRectangle:
    """LSP VIOLATION: Base class that will be violated by Square."""
    
    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height
    
    def set_width(self, width: float) -> None:
        self._width = width
    
    def set_height(self, height: float) -> None:
        self._height = height
    
    def get_area(self) -> float:
        return self._width * self._height


class BadSquare(BadRectangle):
    """LSP VIOLATION: Changes behavior of parent class methods."""
    
    def set_width(self, width: float) -> None:
        # LSP VIOLATION: Changes both width and height
        self._width = width
        self._height = width  # Violates expected behavior!
    
    def set_height(self, height: float) -> None:
        # LSP VIOLATION: Changes both width and height
        self._width = height   # Violates expected behavior!
        self._height = height


def test_rectangle_behavior(rect: BadRectangle) -> None:
    """This function expects rectangle behavior but fails with BadSquare."""
    rect.set_width(5)
    rect.set_height(10)
    expected_area = 5 * 10  # 50
    actual_area = rect.get_area()
    
    # This assertion fails for BadSquare!
    assert actual_area == expected_area, f"Expected {expected_area}, got {actual_area}"


# LSP FIX: Proper inheritance hierarchy

class Shape(ABC):
    """LSP FIX: Abstract base class with proper contract."""
    
    @abstractmethod
    def get_area(self) -> float:
        pass


class GoodRectangle(Shape):
    """LSP FIX: Rectangle with immutable dimensions."""
    
    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height
    
    @property
    def width(self) -> float:
        return self._width
    
    @property
    def height(self) -> float:
        return self._height
    
    def get_area(self) -> float:
        return self._width * self._height
    
    def resize(self, width: float, height: float) -> 'GoodRectangle':
        """Returns new rectangle with different dimensions."""
        return GoodRectangle(width, height)


class GoodSquare(Shape):
    """LSP FIX: Square as separate class, not inheriting from Rectangle."""
    
    def __init__(self, side: float):
        self._side = side
    
    @property
    def side(self) -> float:
        return self._side
    
    def get_area(self) -> float:
        return self._side * self._side
    
    def resize(self, side: float) -> 'GoodSquare':
        """Returns new square with different side."""
        return GoodSquare(side)


# ============================================================================
# VIOLATION 4: ISP VIOLATION - Fat Interface
# ============================================================================

class BadPrinter(ABC):
    """ISP VIOLATION: Fat interface forcing all implementations to support all methods."""
    
    @abstractmethod
    def print_document(self, document: str) -> bool:
        pass
    
    @abstractmethod
    def scan_document(self) -> str:
        """ISP VIOLATION: Not all printers can scan!"""
        pass
    
    @abstractmethod
    def fax_document(self, number: str, document: str) -> bool:
        """ISP VIOLATION: Not all printers can fax!"""
        pass
    
    @abstractmethod
    def copy_document(self, copies: int) -> bool:
        """ISP VIOLATION: Not all printers can copy!"""
        pass


class BadSimplePrinter(BadPrinter):
    """ISP VIOLATION: Forced to implement methods it doesn't support."""
    
    def print_document(self, document: str) -> bool:
        print(f"Printing: {document}")
        return True
    
    def scan_document(self) -> str:
        # ISP VIOLATION: Simple printer can't scan!
        raise NotImplementedError("This printer cannot scan")
    
    def fax_document(self, number: str, document: str) -> bool:
        # ISP VIOLATION: Simple printer can't fax!
        raise NotImplementedError("This printer cannot fax")
    
    def copy_document(self, copies: int) -> bool:
        # ISP VIOLATION: Simple printer can't copy!
        raise NotImplementedError("This printer cannot copy")


# ISP FIX: Segregated interfaces

class Printer(Protocol):
    """ISP FIX: Focused interface for printing."""
    def print_document(self, document: str) -> bool: ...


class Scanner(Protocol):
    """ISP FIX: Focused interface for scanning."""
    def scan_document(self) -> str: ...


class FaxMachine(Protocol):
    """ISP FIX: Focused interface for faxing."""
    def fax_document(self, number: str, document: str) -> bool: ...


class Copier(Protocol):
    """ISP FIX: Focused interface for copying."""
    def copy_document(self, copies: int) -> bool: ...


class GoodSimplePrinter:
    """ISP FIX: Only implements what it actually supports."""
    
    def print_document(self, document: str) -> bool:
        print(f"Printing: {document}")
        return True


class GoodMultiFunctionPrinter:
    """ISP FIX: Implements multiple interfaces as needed."""
    
    def print_document(self, document: str) -> bool:
        print(f"MFP Printing: {document}")
        return True
    
    def scan_document(self) -> str:
        return "Scanned document content"
    
    def fax_document(self, number: str, document: str) -> bool:
        print(f"Faxing to {number}: {document}")
        return True
    
    def copy_document(self, copies: int) -> bool:
        print(f"Making {copies} copies")
        return True


# ============================================================================
# VIOLATION 5: DIP VIOLATION - High-level depending on low-level
# ============================================================================

class BadEmailSender:
    """DIP VIOLATION: Concrete low-level module."""
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        print(f"Sending email to {to}: {subject}")
        return True


class BadFileLogger:
    """DIP VIOLATION: Concrete low-level module."""
    
    def log_to_file(self, message: str) -> None:
        print(f"Logging to file: {message}")


class BadOrderService:
    """
    DIP VIOLATION: High-level module depending directly on low-level modules.
    """
    
    def __init__(self):
        # DIP VIOLATION: Direct dependency on concrete classes
        self.email_sender = BadEmailSender()
        self.logger = BadFileLogger()
    
    def process_order(self, order_id: str, customer_email: str) -> bool:
        # DIP VIOLATION: Tightly coupled to specific implementations
        self.logger.log_to_file(f"Processing order: {order_id}")
        
        # Process order logic here...
        
        success = self.email_sender.send_email(
            customer_email,
            "Order Confirmation",
            f"Your order {order_id} has been processed."
        )
        
        if success:
            self.logger.log_to_file(f"Order processed successfully: {order_id}")
        
        return success


# DIP FIX: Depend on abstractions

class NotificationService(ABC):
    """DIP FIX: Abstract interface for notifications."""
    
    @abstractmethod
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        pass


class LoggingService(ABC):
    """DIP FIX: Abstract interface for logging."""
    
    @abstractmethod
    def log(self, message: str) -> None:
        pass


class GoodEmailSender(NotificationService):
    """DIP FIX: Concrete implementation depending on abstraction."""
    
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        print(f"Email sent to {recipient}: {subject}")
        return True


class GoodFileLogger(LoggingService):
    """DIP FIX: Concrete implementation depending on abstraction."""
    
    def log(self, message: str) -> None:
        print(f"File log: {message}")


class GoodOrderService:
    """DIP FIX: High-level module depending on abstractions."""
    
    def __init__(self, notification_service: NotificationService, logger: LoggingService):
        # DIP FIX: Depending on abstractions, not concretions
        self.notification_service = notification_service
        self.logger = logger
    
    def process_order(self, order_id: str, customer_email: str) -> bool:
        self.logger.log(f"Processing order: {order_id}")
        
        # Process order logic here...
        
        success = self.notification_service.send_notification(
            customer_email,
            "Order Confirmation",
            f"Your order {order_id} has been processed."
        )
        
        if success:
            self.logger.log(f"Order processed successfully: {order_id}")
        
        return success


def demonstrate_solid_violations_and_fixes():
    """
    Demonstrate SOLID violations and their fixes.
    """
    print("=== SOLID VIOLATIONS AND FIXES DEMONSTRATION ===\n")
    
    # 1. SRP Violation and Fix
    print("1. SRP VIOLATION AND FIX:")
    print("   BEFORE: BadUserManager class has 8+ responsibilities")
    print("   AFTER: Separated into focused classes:")
    print("   • User (data representation)")
    print("   • UserValidator (validation)")
    print("   • PasswordManager (password operations)")
    print("   • UserRepository (data persistence)")
    print("   • EmailService (email operations)")
    print("   • AuthenticationService (authentication)")
    print("   • AuthorizationService (authorization)")
    print("   • Logger (logging)")
    print("   • GoodUserService (coordination)")
    
    # Test the fix
    user_repo = UserRepository()
    email_service = EmailService()
    auth_service = AuthenticationService(user_repo)
    authz_service = AuthorizationService()
    logger = Logger()
    
    good_user_service = GoodUserService(user_repo, email_service, auth_service, authz_service, logger)
    success = good_user_service.create_user("john_doe", "john@example.com", "SecurePass123", "user")
    print(f"   ✓ User creation with SRP-compliant design: {'Success' if success else 'Failed'}")
    print()
    
    # 2. OCP Violation and Fix
    print("2. OCP VIOLATION AND FIX:")
    print("   BEFORE: BadDiscountCalculator requires modification for new discount types")
    print("   AFTER: Strategy pattern allows extension without modification")
    
    # Test the fix
    strategies = [
        PercentageDiscount(10),
        FixedDiscount(50),
        SeasonalDiscount(0.15)  # New strategy added without modifying existing code
    ]
    
    order_amount = 200.0
    for strategy in strategies:
        calculator = GoodDiscountCalculator(strategy)
        discount = calculator.calculate_discount(order_amount)
        print(f"   ✓ {strategy.__class__.__name__}: ${discount:.2f} discount on ${order_amount}")
    print()
    
    # 3. LSP Violation and Fix
    print("3. LSP VIOLATION AND FIX:")
    print("   BEFORE: BadSquare violates LSP by changing Rectangle behavior")
    print("   AFTER: Separate Shape hierarchy with proper substitutability")
    
    # Test the fix
    shapes = [GoodRectangle(5, 10), GoodSquare(7)]
    for shape in shapes:
        area = shape.get_area()
        print(f"   ✓ {shape.__class__.__name__} area: {area}")
    print()
    
    # 4. ISP Violation and Fix
    print("4. ISP VIOLATION AND FIX:")
    print("   BEFORE: BadPrinter forces all implementations to support all methods")
    print("   AFTER: Segregated interfaces for specific capabilities")
    
    # Test the fix
    simple_printer = GoodSimplePrinter()
    mfp = GoodMultiFunctionPrinter()
    
    # Simple printer only prints
    simple_printer.print_document("Simple document")
    
    # MFP supports all operations
    mfp.print_document("MFP document")
    mfp.scan_document()
    mfp.copy_document(3)
    
    print("   ✓ Simple printer only implements printing")
    print("   ✓ MFP implements all interfaces it supports")
    print()
    
    # 5. DIP Violation and Fix
    print("5. DIP VIOLATION AND FIX:")
    print("   BEFORE: BadOrderService depends directly on concrete classes")
    print("   AFTER: Depends on abstractions, implementations injected")
    
    # Test the fix
    email_sender = GoodEmailSender()
    file_logger = GoodFileLogger()
    good_order_service = GoodOrderService(email_sender, file_logger)
    
    success = good_order_service.process_order("ORD123", "customer@example.com")
    print(f"   ✓ Order processing with DIP-compliant design: {'Success' if success else 'Failed'}")
    print()
    
    # 6. Summary of Benefits
    print("6. BENEFITS OF FIXING SOLID VIOLATIONS:")
    print("   SRP Benefits:")
    print("   • Each class has a single, clear purpose")
    print("   • Easier to understand and maintain")
    print("   • Changes are localized to specific responsibilities")
    print()
    print("   OCP Benefits:")
    print("   • New functionality added without modifying existing code")
    print("   • Reduced risk of introducing bugs")
    print("   • Better extensibility")
    print()
    print("   LSP Benefits:")
    print("   • Subclasses can be used interchangeably")
    print("   • Polymorphism works correctly")
    print("   • No unexpected behavior from substitutions")
    print()
    print("   ISP Benefits:")
    print("   • Clients only depend on methods they use")
    print("   • Interfaces are focused and cohesive")
    print("   • Easier to implement and test")
    print()
    print("   DIP Benefits:")
    print("   • High-level logic independent of implementation details")
    print("   • Easy to swap implementations")
    print("   • Better testability with mocks")
    print("   • Loose coupling between modules")
    print()
    
    # 7. Refactoring Process
    print("7. SYSTEMATIC REFACTORING PROCESS:")
    print("   Step 1: Identify SOLID violations")
    print("   • Look for classes with multiple responsibilities (SRP)")
    print("   • Find code that requires modification for extension (OCP)")
    print("   • Check for behavioral incompatibilities in inheritance (LSP)")
    print("   • Identify fat interfaces with unrelated methods (ISP)")
    print("   • Find high-level modules depending on low-level details (DIP)")
    print()
    print("   Step 2: Plan the refactoring")
    print("   • Determine which principle is violated")
    print("   • Design the target structure")
    print("   • Plan the migration strategy")
    print()
    print("   Step 3: Apply the fixes")
    print("   • Extract classes for SRP")
    print("   • Introduce abstractions for OCP")
    print("   • Redesign inheritance for LSP")
    print("   • Split interfaces for ISP")
    print("   • Inject dependencies for DIP")
    print()
    print("   Step 4: Verify the improvements")
    print("   • Test that functionality still works")
    print("   • Verify SOLID compliance")
    print("   • Check for improved maintainability")
    print()
    
    print("=== SOLID VIOLATIONS AND FIXES DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_solid_violations_and_fixes()
