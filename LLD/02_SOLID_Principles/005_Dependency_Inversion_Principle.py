"""
DEPENDENCY INVERSION PRINCIPLE - DIP and Dependency Injection
=============================================================

Problem Statement:
Demonstrate the Dependency Inversion Principle (DIP):
- High-level modules should not depend on low-level modules
- Both should depend on abstractions (interfaces)
- Abstractions should not depend on details
- Details should depend on abstractions
- Dependency injection patterns and IoC containers

Learning Objectives:
- Understand the Dependency Inversion Principle
- Identify dependency violations and tight coupling
- Design systems with proper dependency direction
- Implement dependency injection patterns
- Create flexible and testable architectures
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from datetime import datetime
from enum import Enum
import json


# VIOLATION EXAMPLE - High-level module depending on low-level modules
class BadEmailService:
    """BAD EXAMPLE: Concrete email service (low-level module)."""
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        print(f"Sending email to {to}: {subject}")
        return True


class BadSMSService:
    """BAD EXAMPLE: Concrete SMS service (low-level module)."""
    
    def send_sms(self, phone: str, message: str) -> bool:
        print(f"Sending SMS to {phone}: {message}")
        return True


class BadFileLogger:
    """BAD EXAMPLE: Concrete file logger (low-level module)."""
    
    def log(self, message: str) -> None:
        print(f"Logging to file: {message}")


class BadUserService:
    """
    BAD EXAMPLE: High-level module depending directly on low-level modules.
    This violates DIP because it's tightly coupled to concrete implementations.
    """
    
    def __init__(self):
        # DIP VIOLATION: Depending on concrete classes
        self.email_service = BadEmailService()
        self.sms_service = BadSMSService()
        self.logger = BadFileLogger()
        self.users = {}
    
    def create_user(self, username: str, email: str, phone: str) -> bool:
        """Create user - tightly coupled to concrete services."""
        
        # DIP VIOLATION: Cannot easily change logging implementation
        self.logger.log(f"Creating user: {username}")
        
        user = {
            'username': username,
            'email': email,
            'phone': phone,
            'created_at': datetime.now().isoformat()
        }
        
        self.users[username] = user
        
        # DIP VIOLATION: Cannot easily change notification method
        self.email_service.send_email(email, "Welcome!", "Welcome to our platform!")
        self.sms_service.send_sms(phone, "Welcome! Your account is ready.")
        
        return True


# GOOD EXAMPLE - DIP-compliant design with abstractions

# 1. Abstract interfaces (high-level abstractions)
class NotificationService(ABC):
    """Abstract notification service interface."""
    
    @abstractmethod
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        """Send notification to recipient."""
        pass
    
    @abstractmethod
    def get_service_name(self) -> str:
        """Get service name."""
        pass


class Logger(ABC):
    """Abstract logger interface."""
    
    @abstractmethod
    def log(self, level: str, message: str) -> None:
        """Log message with specified level."""
        pass
    
    @abstractmethod
    def get_logs(self) -> List[str]:
        """Get all log entries."""
        pass


class UserRepository(ABC):
    """Abstract user repository interface."""
    
    @abstractmethod
    def save_user(self, user: Dict[str, Any]) -> bool:
        """Save user to storage."""
        pass
    
    @abstractmethod
    def find_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Find user by username."""
        pass
    
    @abstractmethod
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users."""
        pass
    
    @abstractmethod
    def delete_user(self, username: str) -> bool:
        """Delete user."""
        pass


# 2. Concrete implementations (low-level modules) depending on abstractions
class EmailNotificationService(NotificationService):
    """Email notification implementation."""
    
    def __init__(self, smtp_server: str = "localhost"):
        self.smtp_server = smtp_server
        self.sent_emails = []
    
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        """Send email notification."""
        email_data = {
            'to': recipient,
            'subject': subject,
            'body': message,
            'sent_at': datetime.now().isoformat()
        }
        
        self.sent_emails.append(email_data)
        print(f"Email sent to {recipient}: {subject}")
        return True
    
    def get_service_name(self) -> str:
        """Get service name."""
        return "Email Service"
    
    def get_sent_emails(self) -> List[Dict[str, Any]]:
        """Get sent emails."""
        return self.sent_emails.copy()


class SMSNotificationService(NotificationService):
    """SMS notification implementation."""
    
    def __init__(self, api_key: str = "default_key"):
        self.api_key = api_key
        self.sent_messages = []
    
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        """Send SMS notification."""
        sms_data = {
            'phone': recipient,
            'message': f"{subject}: {message}",
            'sent_at': datetime.now().isoformat()
        }
        
        self.sent_messages.append(sms_data)
        print(f"SMS sent to {recipient}: {subject}")
        return True
    
    def get_service_name(self) -> str:
        """Get service name."""
        return "SMS Service"
    
    def get_sent_messages(self) -> List[Dict[str, Any]]:
        """Get sent messages."""
        return self.sent_messages.copy()


class SlackNotificationService(NotificationService):
    """Slack notification implementation - new service without changing high-level code."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.sent_notifications = []
    
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        """Send Slack notification."""
        slack_data = {
            'channel': recipient,
            'title': subject,
            'text': message,
            'sent_at': datetime.now().isoformat()
        }
        
        self.sent_notifications.append(slack_data)
        print(f"Slack notification sent to {recipient}: {subject}")
        return True
    
    def get_service_name(self) -> str:
        """Get service name."""
        return "Slack Service"


class FileLogger(Logger):
    """File logger implementation."""
    
    def __init__(self, filename: str = "app.log"):
        self.filename = filename
        self.log_entries = []
    
    def log(self, level: str, message: str) -> None:
        """Log to file."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_entries.append(log_entry)
        print(f"File log: {log_entry}")
    
    def get_logs(self) -> List[str]:
        """Get all log entries."""
        return self.log_entries.copy()


class ConsoleLogger(Logger):
    """Console logger implementation."""
    
    def __init__(self):
        self.log_entries = []
    
    def log(self, level: str, message: str) -> None:
        """Log to console."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_entries.append(log_entry)
        print(f"Console log: {log_entry}")
    
    def get_logs(self) -> List[str]:
        """Get all log entries."""
        return self.log_entries.copy()


class DatabaseLogger(Logger):
    """Database logger implementation."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.log_entries = []
    
    def log(self, level: str, message: str) -> None:
        """Log to database."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_entries.append(log_entry)
        print(f"Database log: {log_entry}")
    
    def get_logs(self) -> List[str]:
        """Get all log entries."""
        return self.log_entries.copy()


class InMemoryUserRepository(UserRepository):
    """In-memory user repository implementation."""
    
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
    
    def save_user(self, user: Dict[str, Any]) -> bool:
        """Save user to memory."""
        username = user.get('username')
        if username:
            self.users[username] = user
            return True
        return False
    
    def find_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Find user by username."""
        return self.users.get(username)
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users."""
        return list(self.users.values())
    
    def delete_user(self, username: str) -> bool:
        """Delete user."""
        if username in self.users:
            del self.users[username]
            return True
        return False


class DatabaseUserRepository(UserRepository):
    """Database user repository implementation."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.users: Dict[str, Dict[str, Any]] = {}  # Simulated database
    
    def save_user(self, user: Dict[str, Any]) -> bool:
        """Save user to database."""
        username = user.get('username')
        if username:
            self.users[username] = user
            print(f"User saved to database: {username}")
            return True
        return False
    
    def find_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Find user in database."""
        user = self.users.get(username)
        if user:
            print(f"User found in database: {username}")
        return user
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users from database."""
        return list(self.users.values())
    
    def delete_user(self, username: str) -> bool:
        """Delete user from database."""
        if username in self.users:
            del self.users[username]
            print(f"User deleted from database: {username}")
            return True
        return False


# 3. High-level module depending on abstractions (DIP compliant)
class UserService:
    """
    DIP-COMPLIANT: High-level module depending only on abstractions.
    Can work with any implementation of the abstract interfaces.
    """
    
    def __init__(self, 
                 repository: UserRepository,
                 notification_service: NotificationService,
                 logger: Logger):
        # DIP COMPLIANT: Depending on abstractions, not concrete classes
        self.repository = repository
        self.notification_service = notification_service
        self.logger = logger
    
    def create_user(self, username: str, email: str, phone: str) -> bool:
        """Create user using injected dependencies."""
        
        self.logger.log("INFO", f"Creating user: {username}")
        
        # Check if user already exists
        existing_user = self.repository.find_user(username)
        if existing_user:
            self.logger.log("WARNING", f"User already exists: {username}")
            return False
        
        # Create user
        user = {
            'username': username,
            'email': email,
            'phone': phone,
            'created_at': datetime.now().isoformat()
        }
        
        # Save user
        if self.repository.save_user(user):
            self.logger.log("INFO", f"User saved successfully: {username}")
            
            # Send welcome notification
            welcome_message = "Welcome to our platform! Your account is ready."
            notification_sent = self.notification_service.send_notification(
                email, "Welcome!", welcome_message
            )
            
            if notification_sent:
                self.logger.log("INFO", f"Welcome notification sent via {self.notification_service.get_service_name()}")
            else:
                self.logger.log("ERROR", f"Failed to send welcome notification")
            
            return True
        else:
            self.logger.log("ERROR", f"Failed to save user: {username}")
            return False
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        self.logger.log("INFO", f"Retrieving user: {username}")
        return self.repository.find_user(username)
    
    def delete_user(self, username: str) -> bool:
        """Delete user."""
        self.logger.log("INFO", f"Deleting user: {username}")
        
        if self.repository.delete_user(username):
            self.logger.log("INFO", f"User deleted successfully: {username}")
            return True
        else:
            self.logger.log("ERROR", f"Failed to delete user: {username}")
            return False
    
    def get_user_statistics(self) -> Dict[str, Any]:
        """Get user statistics."""
        all_users = self.repository.get_all_users()
        return {
            'total_users': len(all_users),
            'repository_type': self.repository.__class__.__name__,
            'notification_service': self.notification_service.get_service_name(),
            'logger_type': self.logger.__class__.__name__
        }


# DEPENDENCY INJECTION PATTERNS

# 1. Constructor Injection (already shown above)
class OrderService:
    """Service using constructor injection."""
    
    def __init__(self, logger: Logger, notification_service: NotificationService):
        self.logger = logger
        self.notification_service = notification_service
        self.orders = {}
    
    def create_order(self, order_id: str, customer_email: str, items: List[Dict[str, Any]]) -> bool:
        """Create order with injected dependencies."""
        self.logger.log("INFO", f"Creating order: {order_id}")
        
        order = {
            'order_id': order_id,
            'customer_email': customer_email,
            'items': items,
            'total': sum(item['price'] * item['quantity'] for item in items),
            'created_at': datetime.now().isoformat()
        }
        
        self.orders[order_id] = order
        
        # Send order confirmation
        self.notification_service.send_notification(
            customer_email,
            "Order Confirmation",
            f"Your order {order_id} has been created."
        )
        
        self.logger.log("INFO", f"Order created successfully: {order_id}")
        return True


# 2. Setter Injection
class PaymentService:
    """Service using setter injection."""
    
    def __init__(self):
        self._logger: Optional[Logger] = None
        self._notification_service: Optional[NotificationService] = None
        self.payments = {}
    
    def set_logger(self, logger: Logger) -> None:
        """Setter injection for logger."""
        self._logger = logger
    
    def set_notification_service(self, notification_service: NotificationService) -> None:
        """Setter injection for notification service."""
        self._notification_service = notification_service
    
    def process_payment(self, payment_id: str, amount: float, customer_email: str) -> bool:
        """Process payment with injected dependencies."""
        if self._logger:
            self._logger.log("INFO", f"Processing payment: {payment_id}")
        
        payment = {
            'payment_id': payment_id,
            'amount': amount,
            'customer_email': customer_email,
            'status': 'completed',
            'processed_at': datetime.now().isoformat()
        }
        
        self.payments[payment_id] = payment
        
        # Send payment confirmation
        if self._notification_service:
            self._notification_service.send_notification(
                customer_email,
                "Payment Processed",
                f"Your payment of ${amount:.2f} has been processed."
            )
        
        if self._logger:
            self._logger.log("INFO", f"Payment processed successfully: {payment_id}")
        
        return True


# 3. Interface Injection
class ServiceConfigurable(Protocol):
    """Protocol for services that can be configured with dependencies."""
    
    def configure(self, logger: Logger, notification_service: NotificationService) -> None:
        """Configure service with dependencies."""
        ...


class InventoryService:
    """Service using interface injection."""
    
    def __init__(self):
        self._logger: Optional[Logger] = None
        self._notification_service: Optional[NotificationService] = None
        self.inventory = {}
    
    def configure(self, logger: Logger, notification_service: NotificationService) -> None:
        """Interface injection method."""
        self._logger = logger
        self._notification_service = notification_service
    
    def update_stock(self, product_id: str, quantity: int, admin_email: str) -> bool:
        """Update stock with injected dependencies."""
        if self._logger:
            self._logger.log("INFO", f"Updating stock for product: {product_id}")
        
        if product_id in self.inventory:
            self.inventory[product_id] += quantity
        else:
            self.inventory[product_id] = quantity
        
        # Notify admin of stock update
        if self._notification_service:
            self._notification_service.send_notification(
                admin_email,
                "Stock Updated",
                f"Stock for {product_id} updated to {self.inventory[product_id]} units."
            )
        
        if self._logger:
            self._logger.log("INFO", f"Stock updated successfully: {product_id}")
        
        return True


# Simple IoC Container
class DIContainer:
    """Simple Dependency Injection Container."""
    
    def __init__(self):
        self._services = {}
        self._singletons = {}
    
    def register(self, interface: type, implementation: type, singleton: bool = False) -> None:
        """Register a service implementation."""
        self._services[interface] = {
            'implementation': implementation,
            'singleton': singleton
        }
    
    def register_instance(self, interface: type, instance: Any) -> None:
        """Register a service instance."""
        self._singletons[interface] = instance
    
    def resolve(self, interface: type) -> Any:
        """Resolve a service instance."""
        # Check for registered instance
        if interface in self._singletons:
            return self._singletons[interface]
        
        # Check for registered service
        if interface in self._services:
            service_info = self._services[interface]
            implementation = service_info['implementation']
            
            # Create instance
            instance = implementation()
            
            # Store as singleton if required
            if service_info['singleton']:
                self._singletons[interface] = instance
            
            return instance
        
        raise ValueError(f"Service not registered: {interface}")
    
    def create_user_service(self) -> UserService:
        """Factory method to create UserService with all dependencies."""
        repository = self.resolve(UserRepository)
        notification_service = self.resolve(NotificationService)
        logger = self.resolve(Logger)
        
        return UserService(repository, notification_service, logger)


def demonstrate_dependency_inversion_principle():
    """
    Demonstrate Dependency Inversion Principle with practical examples.
    """
    print("=== DEPENDENCY INVERSION PRINCIPLE DEMONSTRATION ===\n")
    
    # 1. Show DIP violation problem
    print("1. DIP VIOLATION PROBLEM:")
    print("   BadUserService depends directly on concrete classes:")
    print("   - BadEmailService (low-level)")
    print("   - BadSMSService (low-level)")
    print("   - BadFileLogger (low-level)")
    print("   This creates tight coupling and makes testing/changes difficult.")
    print()
    
    # 2. DIP-compliant design with different implementations
    print("2. DIP-COMPLIANT DESIGN:")
    
    # Create different combinations of services
    configurations = [
        {
            'name': 'Development Configuration',
            'repository': InMemoryUserRepository(),
            'notification': EmailNotificationService(),
            'logger': ConsoleLogger()
        },
        {
            'name': 'Production Configuration',
            'repository': DatabaseUserRepository("prod_db_connection"),
            'notification': SlackNotificationService("https://hooks.slack.com/webhook"),
            'logger': DatabaseLogger("log_db_connection")
        },
        {
            'name': 'Testing Configuration',
            'repository': InMemoryUserRepository(),
            'notification': SMSNotificationService("test_api_key"),
            'logger': FileLogger("test.log")
        }
    ]
    
    print("   Testing different configurations:")
    
    for config in configurations:
        print(f"\n   {config['name']}:")
        
        # Create user service with injected dependencies
        user_service = UserService(
            config['repository'],
            config['notification'],
            config['logger']
        )
        
        # Use the service (same interface, different implementations)
        success = user_service.create_user(
            f"user_{config['name'].lower().replace(' ', '_')}",
            "user@example.com",
            "+1234567890"
        )
        
        print(f"     User creation: {'Success' if success else 'Failed'}")
        
        # Get statistics
        stats = user_service.get_user_statistics()
        print(f"     Repository: {stats['repository_type']}")
        print(f"     Notification: {stats['notification_service']}")
        print(f"     Logger: {stats['logger_type']}")
    
    print()
    
    # 3. Dependency Injection Patterns
    print("3. DEPENDENCY INJECTION PATTERNS:")
    
    # Constructor Injection
    print("   Constructor Injection:")
    logger = FileLogger("orders.log")
    email_service = EmailNotificationService()
    order_service = OrderService(logger, email_service)
    
    order_service.create_order("ORD001", "customer@example.com", [
        {'name': 'Product A', 'price': 29.99, 'quantity': 2},
        {'name': 'Product B', 'price': 49.99, 'quantity': 1}
    ])
    
    # Setter Injection
    print("\n   Setter Injection:")
    payment_service = PaymentService()
    payment_service.set_logger(ConsoleLogger())
    payment_service.set_notification_service(SMSNotificationService())
    
    payment_service.process_payment("PAY001", 109.97, "customer@example.com")
    
    # Interface Injection
    print("\n   Interface Injection:")
    inventory_service = InventoryService()
    inventory_service.configure(DatabaseLogger("inventory.db"), SlackNotificationService("webhook"))
    
    inventory_service.update_stock("PROD001", 50, "admin@example.com")
    
    print()
    
    # 4. IoC Container Usage
    print("4. IOC CONTAINER USAGE:")
    
    # Create and configure container
    container = DIContainer()
    
    # Register services
    container.register(UserRepository, InMemoryUserRepository, singleton=True)
    container.register(NotificationService, EmailNotificationService)
    container.register(Logger, FileLogger)
    
    # Create service using container
    user_service = container.create_user_service()
    
    print("   Created UserService using IoC container")
    success = user_service.create_user("container_user", "container@example.com", "+1111111111")
    print(f"   User creation via container: {'Success' if success else 'Failed'}")
    
    # Show statistics
    stats = user_service.get_user_statistics()
    print(f"   Container resolved services:")
    for key, value in stats.items():
        print(f"     {key}: {value}")
    
    print()
    
    # 5. Benefits of DIP
    print("5. BENEFITS OF DEPENDENCY INVERSION PRINCIPLE:")
    print("   ✓ High-level modules are independent of low-level implementation details")
    print("   ✓ Easy to swap implementations without changing high-level code")
    print("   ✓ Better testability with mock implementations")
    print("   ✓ Loose coupling between modules")
    print("   ✓ Follows Hollywood Principle: 'Don't call us, we'll call you'")
    print("   ✓ Supports plugin architectures")
    print("   ✓ Easier to maintain and extend")
    print()
    
    print("   DIP Implementation Techniques:")
    print("   • Constructor injection (most common)")
    print("   • Setter injection (for optional dependencies)")
    print("   • Interface injection (for configurable services)")
    print("   • Service locator pattern")
    print("   • IoC containers for automatic dependency resolution")
    print("   • Factory patterns for complex object creation")
    print()
    
    # 6. Testing Benefits
    print("6. TESTING BENEFITS:")
    print("   DIP makes testing easier by allowing mock implementations:")
    
    class MockLogger(Logger):
        def __init__(self):
            self.logged_messages = []
        
        def log(self, level: str, message: str) -> None:
            self.logged_messages.append(f"{level}: {message}")
        
        def get_logs(self) -> List[str]:
            return self.logged_messages
    
    class MockNotificationService(NotificationService):
        def __init__(self):
            self.sent_notifications = []
        
        def send_notification(self, recipient: str, subject: str, message: str) -> bool:
            self.sent_notifications.append((recipient, subject, message))
            return True
        
        def get_service_name(self) -> str:
            return "Mock Service"
    
    # Create service with mocks for testing
    mock_logger = MockLogger()
    mock_notification = MockNotificationService()
    test_repository = InMemoryUserRepository()
    
    test_service = UserService(test_repository, mock_notification, mock_logger)
    test_service.create_user("test_user", "test@example.com", "+1234567890")
    
    print(f"   Mock logger captured {len(mock_logger.logged_messages)} messages")
    print(f"   Mock notification sent {len(mock_notification.sent_notifications)} notifications")
    print("   Testing is much easier with dependency injection!")
    
    print()
    
    print("=== DEPENDENCY INVERSION PRINCIPLE DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_dependency_inversion_principle()
