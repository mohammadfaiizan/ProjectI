"""
DEPENDENCY INJECTION PATTERNS - DI Implementation Techniques
============================================================

Problem Statement:
Demonstrate various dependency injection patterns and techniques:
- Constructor injection, setter injection, interface injection
- Service locator pattern
- IoC containers and dependency resolution
- Factory patterns for dependency creation
- Dependency injection frameworks simulation
- Testing with dependency injection

Learning Objectives:
- Master different dependency injection techniques
- Understand IoC containers and service locators
- Implement factory patterns for complex dependencies
- Design testable systems with DI
- Choose appropriate DI patterns for different scenarios
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, TypeVar, Type
from datetime import datetime
from enum import Enum
import inspect


# ============================================================================
# INTERFACES AND ABSTRACTIONS
# ============================================================================

class Logger(ABC):
    """Abstract logger interface."""
    
    @abstractmethod
    def log(self, level: str, message: str) -> None:
        pass


class NotificationService(ABC):
    """Abstract notification service interface."""
    
    @abstractmethod
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        pass


class DataRepository(ABC):
    """Abstract data repository interface."""
    
    @abstractmethod
    def save(self, entity: Any) -> bool:
        pass
    
    @abstractmethod
    def find_by_id(self, entity_id: str) -> Optional[Any]:
        pass


class ConfigurationService(ABC):
    """Abstract configuration service interface."""
    
    @abstractmethod
    def get_setting(self, key: str, default: Any = None) -> Any:
        pass


# ============================================================================
# CONCRETE IMPLEMENTATIONS
# ============================================================================

class ConsoleLogger(Logger):
    """Console logger implementation."""
    
    def __init__(self, prefix: str = "APP"):
        self.prefix = prefix
        self.log_entries = []
    
    def log(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {self.prefix} {level}: {message}"
        self.log_entries.append(log_entry)
        print(log_entry)


class FileLogger(Logger):
    """File logger implementation."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.log_entries = []
    
    def log(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] FILE {level}: {message}"
        self.log_entries.append(log_entry)
        print(f"Writing to {self.filename}: {log_entry}")


class EmailNotificationService(NotificationService):
    """Email notification service implementation."""
    
    def __init__(self, smtp_server: str = "localhost"):
        self.smtp_server = smtp_server
        self.sent_notifications = []
    
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        notification = {
            'recipient': recipient,
            'subject': subject,
            'message': message,
            'sent_at': datetime.now().isoformat(),
            'service': 'email'
        }
        self.sent_notifications.append(notification)
        print(f"Email sent via {self.smtp_server} to {recipient}: {subject}")
        return True


class SMSNotificationService(NotificationService):
    """SMS notification service implementation."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.sent_notifications = []
    
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        notification = {
            'recipient': recipient,
            'subject': subject,
            'message': message,
            'sent_at': datetime.now().isoformat(),
            'service': 'sms'
        }
        self.sent_notifications.append(notification)
        print(f"SMS sent to {recipient}: {subject}")
        return True


class InMemoryRepository(DataRepository):
    """In-memory data repository implementation."""
    
    def __init__(self):
        self.data = {}
        self.next_id = 1
    
    def save(self, entity: Any) -> bool:
        entity_id = getattr(entity, 'id', str(self.next_id))
        self.data[entity_id] = entity
        self.next_id += 1
        print(f"Entity saved to memory: {entity_id}")
        return True
    
    def find_by_id(self, entity_id: str) -> Optional[Any]:
        return self.data.get(entity_id)


class DatabaseRepository(DataRepository):
    """Database repository implementation."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.data = {}  # Simulated database
    
    def save(self, entity: Any) -> bool:
        entity_id = getattr(entity, 'id', str(len(self.data) + 1))
        self.data[entity_id] = entity
        print(f"Entity saved to database ({self.connection_string}): {entity_id}")
        return True
    
    def find_by_id(self, entity_id: str) -> Optional[Any]:
        return self.data.get(entity_id)


class JsonConfigurationService(ConfigurationService):
    """JSON-based configuration service."""
    
    def __init__(self, config_data: Dict[str, Any] = None):
        self.config_data = config_data or {
            'database_url': 'localhost:5432',
            'smtp_server': 'smtp.example.com',
            'api_timeout': 30,
            'debug_mode': True
        }
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        return self.config_data.get(key, default)


# ============================================================================
# PATTERN 1: CONSTRUCTOR INJECTION
# ============================================================================

class UserService:
    """Service using constructor injection (most common pattern)."""
    
    def __init__(self, 
                 repository: DataRepository,
                 notification_service: NotificationService,
                 logger: Logger):
        # Dependencies injected through constructor
        self.repository = repository
        self.notification_service = notification_service
        self.logger = logger
        self.users = {}
    
    def create_user(self, username: str, email: str) -> bool:
        """Create user using injected dependencies."""
        self.logger.log("INFO", f"Creating user: {username}")
        
        user = {
            'id': username,
            'username': username,
            'email': email,
            'created_at': datetime.now().isoformat()
        }
        
        if self.repository.save(user):
            self.notification_service.send_notification(
                email,
                "Welcome!",
                f"Welcome {username}! Your account has been created."
            )
            self.logger.log("INFO", f"User created successfully: {username}")
            return True
        else:
            self.logger.log("ERROR", f"Failed to create user: {username}")
            return False


# ============================================================================
# PATTERN 2: SETTER INJECTION
# ============================================================================

class OrderService:
    """Service using setter injection (for optional dependencies)."""
    
    def __init__(self):
        # Dependencies will be set via setters
        self._repository: Optional[DataRepository] = None
        self._notification_service: Optional[NotificationService] = None
        self._logger: Optional[Logger] = None
        self.orders = {}
    
    # Setter methods for dependency injection
    def set_repository(self, repository: DataRepository) -> None:
        """Setter injection for repository."""
        self._repository = repository
    
    def set_notification_service(self, notification_service: NotificationService) -> None:
        """Setter injection for notification service."""
        self._notification_service = notification_service
    
    def set_logger(self, logger: Logger) -> None:
        """Setter injection for logger."""
        self._logger = logger
    
    def create_order(self, order_id: str, customer_email: str, items: List[Dict[str, Any]]) -> bool:
        """Create order using injected dependencies."""
        if self._logger:
            self._logger.log("INFO", f"Creating order: {order_id}")
        
        order = {
            'id': order_id,
            'customer_email': customer_email,
            'items': items,
            'total': sum(item['price'] * item['quantity'] for item in items),
            'created_at': datetime.now().isoformat()
        }
        
        success = True
        if self._repository:
            success = self._repository.save(order)
        else:
            # Fallback behavior when repository is not injected
            self.orders[order_id] = order
        
        if success and self._notification_service:
            self._notification_service.send_notification(
                customer_email,
                "Order Confirmation",
                f"Your order {order_id} has been created."
            )
        
        if self._logger:
            status = "successfully" if success else "failed"
            self._logger.log("INFO", f"Order creation {status}: {order_id}")
        
        return success


# ============================================================================
# PATTERN 3: INTERFACE INJECTION
# ============================================================================

class Injectable(ABC):
    """Interface for objects that can receive dependencies."""
    
    @abstractmethod
    def inject_dependencies(self, **dependencies) -> None:
        """Inject dependencies into the object."""
        pass


class PaymentService(Injectable):
    """Service using interface injection."""
    
    def __init__(self):
        self._logger: Optional[Logger] = None
        self._notification_service: Optional[NotificationService] = None
        self._config_service: Optional[ConfigurationService] = None
        self.payments = {}
    
    def inject_dependencies(self, **dependencies) -> None:
        """Interface injection method."""
        self._logger = dependencies.get('logger')
        self._notification_service = dependencies.get('notification_service')
        self._config_service = dependencies.get('config_service')
    
    def process_payment(self, payment_id: str, amount: float, customer_email: str) -> bool:
        """Process payment using injected dependencies."""
        if self._logger:
            self._logger.log("INFO", f"Processing payment: {payment_id}")
        
        # Use configuration if available
        timeout = 30
        if self._config_service:
            timeout = self._config_service.get_setting('api_timeout', 30)
        
        payment = {
            'id': payment_id,
            'amount': amount,
            'customer_email': customer_email,
            'status': 'completed',
            'timeout_used': timeout,
            'processed_at': datetime.now().isoformat()
        }
        
        self.payments[payment_id] = payment
        
        if self._notification_service:
            self._notification_service.send_notification(
                customer_email,
                "Payment Processed",
                f"Your payment of ${amount:.2f} has been processed."
            )
        
        if self._logger:
            self._logger.log("INFO", f"Payment processed successfully: {payment_id}")
        
        return True


# ============================================================================
# PATTERN 4: SERVICE LOCATOR
# ============================================================================

class ServiceLocator:
    """Service locator pattern implementation."""
    
    _instance = None
    _services: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register_service(cls, service_name: str, service_instance: Any) -> None:
        """Register a service instance."""
        cls._services[service_name] = service_instance
    
    @classmethod
    def get_service(cls, service_name: str) -> Any:
        """Get a service instance."""
        service = cls._services.get(service_name)
        if service is None:
            raise ValueError(f"Service not registered: {service_name}")
        return service
    
    @classmethod
    def clear_services(cls) -> None:
        """Clear all registered services."""
        cls._services.clear()


class InventoryService:
    """Service using service locator pattern."""
    
    def __init__(self):
        self.inventory = {}
    
    def update_stock(self, product_id: str, quantity: int) -> bool:
        """Update stock using service locator."""
        # Get dependencies from service locator
        logger = ServiceLocator.get_service('logger')
        repository = ServiceLocator.get_service('repository')
        
        logger.log("INFO", f"Updating stock for product: {product_id}")
        
        self.inventory[product_id] = self.inventory.get(product_id, 0) + quantity
        
        stock_record = {
            'id': f"stock_{product_id}",
            'product_id': product_id,
            'quantity': self.inventory[product_id],
            'updated_at': datetime.now().isoformat()
        }
        
        if repository.save(stock_record):
            logger.log("INFO", f"Stock updated successfully: {product_id}")
            return True
        else:
            logger.log("ERROR", f"Failed to update stock: {product_id}")
            return False


# ============================================================================
# PATTERN 5: IOC CONTAINER
# ============================================================================

T = TypeVar('T')

class DIContainer:
    """Dependency Injection Container with automatic resolution."""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._transient_types: Dict[Type, Type] = {}
    
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a singleton service."""
        self._services[interface] = implementation
        self._singletons[interface] = None  # Will be created on first access
    
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a transient service (new instance each time)."""
        self._transient_types[interface] = implementation
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function for service creation."""
        self._factories[interface] = factory
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register a specific instance."""
        self._singletons[interface] = instance
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service instance."""
        # Check for registered instance
        if interface in self._singletons:
            if self._singletons[interface] is None:
                # Create singleton instance
                implementation = self._services[interface]
                self._singletons[interface] = self._create_instance(implementation)
            return self._singletons[interface]
        
        # Check for factory
        if interface in self._factories:
            return self._factories[interface]()
        
        # Check for transient
        if interface in self._transient_types:
            implementation = self._transient_types[interface]
            return self._create_instance(implementation)
        
        # Try to create directly if it's a concrete class
        if not inspect.isabstract(interface):
            return self._create_instance(interface)
        
        raise ValueError(f"Service not registered: {interface}")
    
    def _create_instance(self, cls: Type[T]) -> T:
        """Create instance with automatic dependency injection."""
        # Get constructor signature
        sig = inspect.signature(cls.__init__)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Try to resolve parameter type
            param_type = param.annotation
            if param_type != inspect.Parameter.empty:
                try:
                    params[param_name] = self.resolve(param_type)
                except ValueError:
                    # If we can't resolve, check if parameter has default value
                    if param.default == inspect.Parameter.empty:
                        raise ValueError(f"Cannot resolve parameter {param_name} of type {param_type}")
        
        return cls(**params)


# ============================================================================
# PATTERN 6: FACTORY PATTERN FOR DEPENDENCIES
# ============================================================================

class ServiceFactory:
    """Factory for creating services with proper dependencies."""
    
    def __init__(self, container: DIContainer):
        self.container = container
    
    def create_user_service(self, service_type: str = "standard") -> UserService:
        """Factory method to create user service with different configurations."""
        
        if service_type == "standard":
            repository = InMemoryRepository()
            notification = EmailNotificationService()
            logger = ConsoleLogger("USER_SERVICE")
        elif service_type == "production":
            repository = DatabaseRepository("prod://database:5432")
            notification = EmailNotificationService("smtp.production.com")
            logger = FileLogger("user_service.log")
        elif service_type == "testing":
            repository = InMemoryRepository()
            notification = SMSNotificationService("test_api_key")
            logger = ConsoleLogger("TEST")
        else:
            raise ValueError(f"Unknown service type: {service_type}")
        
        return UserService(repository, notification, logger)
    
    def create_order_service_with_all_dependencies(self) -> OrderService:
        """Create order service with all dependencies set."""
        order_service = OrderService()
        order_service.set_repository(self.container.resolve(DataRepository))
        order_service.set_notification_service(self.container.resolve(NotificationService))
        order_service.set_logger(self.container.resolve(Logger))
        return order_service


# ============================================================================
# PATTERN 7: DEPENDENCY INJECTION FOR TESTING
# ============================================================================

class MockLogger(Logger):
    """Mock logger for testing."""
    
    def __init__(self):
        self.logged_messages = []
    
    def log(self, level: str, message: str) -> None:
        self.logged_messages.append(f"{level}: {message}")


class MockNotificationService(NotificationService):
    """Mock notification service for testing."""
    
    def __init__(self):
        self.sent_notifications = []
    
    def send_notification(self, recipient: str, subject: str, message: str) -> bool:
        self.sent_notifications.append({
            'recipient': recipient,
            'subject': subject,
            'message': message
        })
        return True


class MockRepository(DataRepository):
    """Mock repository for testing."""
    
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.saved_entities = []
    
    def save(self, entity: Any) -> bool:
        if self.should_fail:
            return False
        self.saved_entities.append(entity)
        return True
    
    def find_by_id(self, entity_id: str) -> Optional[Any]:
        for entity in self.saved_entities:
            if getattr(entity, 'id', None) == entity_id:
                return entity
        return None


def demonstrate_dependency_injection_patterns():
    """
    Demonstrate various dependency injection patterns.
    """
    print("=== DEPENDENCY INJECTION PATTERNS DEMONSTRATION ===\n")
    
    # 1. Constructor Injection
    print("1. CONSTRUCTOR INJECTION PATTERN:")
    print("   Most common and recommended pattern")
    print("   Dependencies are required and immutable after construction")
    
    repository = InMemoryRepository()
    email_service = EmailNotificationService()
    logger = ConsoleLogger("USER")
    
    user_service = UserService(repository, email_service, logger)
    success = user_service.create_user("john_doe", "john@example.com")
    print(f"   ✓ User creation: {'Success' if success else 'Failed'}")
    print()
    
    # 2. Setter Injection
    print("2. SETTER INJECTION PATTERN:")
    print("   Good for optional dependencies or when dependencies change")
    
    order_service = OrderService()
    order_service.set_repository(DatabaseRepository("test://db"))
    order_service.set_notification_service(SMSNotificationService("api_key"))
    order_service.set_logger(FileLogger("orders.log"))
    
    order_items = [
        {'name': 'Product A', 'price': 29.99, 'quantity': 2},
        {'name': 'Product B', 'price': 49.99, 'quantity': 1}
    ]
    
    success = order_service.create_order("ORD001", "customer@example.com", order_items)
    print(f"   ✓ Order creation: {'Success' if success else 'Failed'}")
    print()
    
    # 3. Interface Injection
    print("3. INTERFACE INJECTION PATTERN:")
    print("   Dependencies injected through a common interface")
    
    payment_service = PaymentService()
    payment_service.inject_dependencies(
        logger=ConsoleLogger("PAYMENT"),
        notification_service=EmailNotificationService(),
        config_service=JsonConfigurationService()
    )
    
    success = payment_service.process_payment("PAY001", 109.97, "customer@example.com")
    print(f"   ✓ Payment processing: {'Success' if success else 'Failed'}")
    print()
    
    # 4. Service Locator Pattern
    print("4. SERVICE LOCATOR PATTERN:")
    print("   Services are retrieved from a central registry")
    
    # Register services
    ServiceLocator.register_service('logger', ConsoleLogger("INVENTORY"))
    ServiceLocator.register_service('repository', InMemoryRepository())
    
    inventory_service = InventoryService()
    success = inventory_service.update_stock("PROD001", 50)
    print(f"   ✓ Stock update: {'Success' if success else 'Failed'}")
    
    # Clean up service locator
    ServiceLocator.clear_services()
    print()
    
    # 5. IoC Container
    print("5. IOC CONTAINER PATTERN:")
    print("   Automatic dependency resolution and lifecycle management")
    
    container = DIContainer()
    
    # Register services
    container.register_singleton(Logger, ConsoleLogger)
    container.register_transient(NotificationService, EmailNotificationService)
    container.register_instance(DataRepository, InMemoryRepository())
    
    # Resolve services automatically
    resolved_logger = container.resolve(Logger)
    resolved_notification = container.resolve(NotificationService)
    resolved_repository = container.resolve(DataRepository)
    
    print(f"   ✓ Resolved Logger: {type(resolved_logger).__name__}")
    print(f"   ✓ Resolved NotificationService: {type(resolved_notification).__name__}")
    print(f"   ✓ Resolved Repository: {type(resolved_repository).__name__}")
    
    # Test singleton behavior
    logger1 = container.resolve(Logger)
    logger2 = container.resolve(Logger)
    print(f"   ✓ Singleton behavior: {logger1 is logger2}")
    
    # Test transient behavior
    notification1 = container.resolve(NotificationService)
    notification2 = container.resolve(NotificationService)
    print(f"   ✓ Transient behavior: {notification1 is not notification2}")
    print()
    
    # 6. Factory Pattern
    print("6. FACTORY PATTERN FOR DEPENDENCIES:")
    print("   Creating services with different configurations")
    
    factory = ServiceFactory(container)
    
    # Create different service configurations
    standard_service = factory.create_user_service("standard")
    production_service = factory.create_user_service("production")
    testing_service = factory.create_user_service("testing")
    
    print(f"   ✓ Standard service created: {type(standard_service.logger).__name__}")
    print(f"   ✓ Production service created: {type(production_service.repository).__name__}")
    print(f"   ✓ Testing service created: {type(testing_service.notification_service).__name__}")
    print()
    
    # 7. Testing with Dependency Injection
    print("7. TESTING WITH DEPENDENCY INJECTION:")
    print("   Using mocks and stubs for isolated testing")
    
    # Create mocks
    mock_logger = MockLogger()
    mock_notification = MockNotificationService()
    mock_repository = MockRepository()
    
    # Test successful scenario
    test_service = UserService(mock_repository, mock_notification, mock_logger)
    success = test_service.create_user("test_user", "test@example.com")
    
    print(f"   ✓ Test user creation: {'Success' if success else 'Failed'}")
    print(f"   ✓ Logged messages: {len(mock_logger.logged_messages)}")
    print(f"   ✓ Sent notifications: {len(mock_notification.sent_notifications)}")
    print(f"   ✓ Saved entities: {len(mock_repository.saved_entities)}")
    
    # Test failure scenario
    failing_repository = MockRepository(should_fail=True)
    failing_service = UserService(failing_repository, mock_notification, mock_logger)
    failure = failing_service.create_user("failing_user", "fail@example.com")
    
    print(f"   ✓ Test failure scenario: {'Failed as expected' if not failure else 'Unexpected success'}")
    print()
    
    # 8. Comparison of Patterns
    print("8. DEPENDENCY INJECTION PATTERNS COMPARISON:")
    print()
    print("   Constructor Injection:")
    print("   • Pros: Required dependencies, immutable, fail-fast")
    print("   • Cons: Can lead to large constructors")
    print("   • Best for: Required dependencies, most common pattern")
    print()
    print("   Setter Injection:")
    print("   • Pros: Optional dependencies, can change after construction")
    print("   • Cons: Mutable state, dependencies might not be set")
    print("   • Best for: Optional dependencies, configuration changes")
    print()
    print("   Interface Injection:")
    print("   • Pros: Explicit injection contract, flexible")
    print("   • Cons: More complex, requires additional interface")
    print("   • Best for: Framework-level injection, plugin architectures")
    print()
    print("   Service Locator:")
    print("   • Pros: Simple to use, centralized service registry")
    print("   • Cons: Hidden dependencies, harder to test, service locator dependency")
    print("   • Best for: Legacy systems, when DI is not feasible")
    print()
    print("   IoC Container:")
    print("   • Pros: Automatic resolution, lifecycle management, configuration")
    print("   • Cons: Complex setup, magic behavior, learning curve")
    print("   • Best for: Large applications, enterprise systems")
    print()
    print("   Factory Pattern:")
    print("   • Pros: Controlled creation, different configurations")
    print("   • Cons: Additional abstraction layer")
    print("   • Best for: Complex object creation, multiple configurations")
    print()
    
    # 9. Best Practices
    print("9. DEPENDENCY INJECTION BEST PRACTICES:")
    print("   ✓ Prefer constructor injection for required dependencies")
    print("   ✓ Use setter injection for optional dependencies")
    print("   ✓ Keep constructors simple and focused")
    print("   ✓ Avoid service locator anti-pattern when possible")
    print("   ✓ Use IoC containers for complex dependency graphs")
    print("   ✓ Design for testability with mock implementations")
    print("   ✓ Follow single responsibility principle in services")
    print("   ✓ Use factory patterns for complex object creation")
    print("   ✓ Document dependency requirements clearly")
    print("   ✓ Avoid circular dependencies")
    print()
    
    print("=== DEPENDENCY INJECTION PATTERNS DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_dependency_injection_patterns()
