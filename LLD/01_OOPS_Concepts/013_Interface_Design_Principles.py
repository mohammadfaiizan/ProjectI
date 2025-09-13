"""
INTERFACE DESIGN PRINCIPLES - Contract-based Programming
========================================================

Problem Statement:
Demonstrate interface design principles and patterns:
- Abstract base classes as interfaces
- Protocol-based interfaces (Python 3.8+)
- Interface segregation principle
- Dependency inversion through interfaces
- Contract-based programming

Learning Objectives:
- Design clean and focused interfaces
- Implement interface segregation principle
- Use interfaces for dependency inversion
- Create flexible and testable code
- Apply contract-based programming
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from datetime import datetime
from enum import Enum


# Interface Segregation Principle - Split large interfaces into smaller ones
class Readable(ABC):
    """Interface for readable objects."""
    
    @abstractmethod
    def read(self) -> str:
        """Read content from the object."""
        pass


class Writable(ABC):
    """Interface for writable objects."""
    
    @abstractmethod
    def write(self, content: str) -> bool:
        """Write content to the object."""
        pass


class Seekable(ABC):
    """Interface for seekable objects."""
    
    @abstractmethod
    def seek(self, position: int) -> bool:
        """Seek to position in the object."""
        pass
    
    @abstractmethod
    def tell(self) -> int:
        """Get current position."""
        pass


# Protocol-based interfaces (Python 3.8+)
@runtime_checkable
class Drawable(Protocol):
    """Protocol for drawable objects."""
    
    def draw(self) -> str:
        """Draw the object."""
        ...
    
    def get_area(self) -> float:
        """Get the area of the object."""
        ...


@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable objects."""
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize object to dictionary."""
        ...
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Serializable':
        """Deserialize dictionary to object."""
        ...


# Payment processing interfaces
class PaymentMethod(ABC):
    """Abstract interface for payment methods."""
    
    @abstractmethod
    def validate_payment_details(self, details: Dict[str, Any]) -> bool:
        """Validate payment details."""
        pass
    
    @abstractmethod
    def process_payment(self, amount: float, details: Dict[str, Any]) -> Dict[str, Any]:
        """Process payment and return result."""
        pass
    
    @abstractmethod
    def refund_payment(self, transaction_id: str, amount: float) -> Dict[str, Any]:
        """Refund payment."""
        pass


class NotificationSender(ABC):
    """Interface for notification senders."""
    
    @abstractmethod
    def send_notification(self, recipient: str, message: str, subject: str = "") -> bool:
        """Send notification to recipient."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if notification service is available."""
        pass


# Concrete implementations
class CreditCardPayment(PaymentMethod):
    """Credit card payment implementation."""
    
    def __init__(self, gateway_url: str, merchant_id: str):
        self.gateway_url = gateway_url
        self.merchant_id = merchant_id
    
    def validate_payment_details(self, details: Dict[str, Any]) -> bool:
        """Validate credit card details."""
        required_fields = ['card_number', 'expiry_month', 'expiry_year', 'cvv']
        
        for field in required_fields:
            if field not in details:
                return False
        
        # Validate card number (simplified)
        card_number = details['card_number'].replace(' ', '')
        if not (card_number.isdigit() and len(card_number) == 16):
            return False
        
        return True
    
    def process_payment(self, amount: float, details: Dict[str, Any]) -> Dict[str, Any]:
        """Process credit card payment."""
        if not self.validate_payment_details(details):
            return {'success': False, 'error': 'Invalid payment details'}
        
        # Simulate payment processing
        transaction_id = f"CC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            'success': True,
            'transaction_id': transaction_id,
            'amount': amount,
            'method': 'credit_card',
            'message': 'Payment processed successfully'
        }
    
    def refund_payment(self, transaction_id: str, amount: float) -> Dict[str, Any]:
        """Refund credit card payment."""
        return {
            'success': True,
            'refund_id': f"REF_{transaction_id}",
            'amount': amount,
            'message': 'Refund processed'
        }


class PayPalPayment(PaymentMethod):
    """PayPal payment implementation."""
    
    def __init__(self, api_key: str, secret: str):
        self.api_key = api_key
        self.secret = secret
    
    def validate_payment_details(self, details: Dict[str, Any]) -> bool:
        """Validate PayPal details."""
        return 'email' in details and '@' in details['email']
    
    def process_payment(self, amount: float, details: Dict[str, Any]) -> Dict[str, Any]:
        """Process PayPal payment."""
        if not self.validate_payment_details(details):
            return {'success': False, 'error': 'Invalid PayPal email'}
        
        transaction_id = f"PP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            'success': True,
            'transaction_id': transaction_id,
            'amount': amount,
            'method': 'paypal',
            'email': details['email'],
            'message': 'PayPal payment completed'
        }
    
    def refund_payment(self, transaction_id: str, amount: float) -> Dict[str, Any]:
        """Refund PayPal payment."""
        return {
            'success': True,
            'refund_id': f"PPREF_{transaction_id}",
            'amount': amount,
            'message': 'PayPal refund initiated'
        }


class EmailNotification(NotificationSender):
    """Email notification implementation."""
    
    def __init__(self, smtp_server: str, port: int):
        self.smtp_server = smtp_server
        self.port = port
        self._is_connected = False
    
    def send_notification(self, recipient: str, message: str, subject: str = "") -> bool:
        """Send email notification."""
        if not self.is_available():
            return False
        
        print(f"Email sent to {recipient}")
        print(f"Subject: {subject}")
        print(f"Message: {message}")
        return True
    
    def is_available(self) -> bool:
        """Check if email service is available."""
        # Simulate connection check
        return True


class SMSNotification(NotificationSender):
    """SMS notification implementation."""
    
    def __init__(self, api_key: str, service_url: str):
        self.api_key = api_key
        self.service_url = service_url
    
    def send_notification(self, recipient: str, message: str, subject: str = "") -> bool:
        """Send SMS notification."""
        if not self.is_available():
            return False
        
        print(f"SMS sent to {recipient}: {message}")
        return True
    
    def is_available(self) -> bool:
        """Check if SMS service is available."""
        return True


# File system interfaces with segregation
class TextFile(Readable, Writable, Seekable):
    """Text file implementing multiple interfaces."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.content = ""
        self.position = 0
    
    def read(self) -> str:
        """Read content from file."""
        return self.content[self.position:]
    
    def write(self, content: str) -> bool:
        """Write content to file."""
        self.content = self.content[:self.position] + content + self.content[self.position:]
        self.position += len(content)
        return True
    
    def seek(self, position: int) -> bool:
        """Seek to position."""
        if 0 <= position <= len(self.content):
            self.position = position
            return True
        return False
    
    def tell(self) -> int:
        """Get current position."""
        return self.position


class ReadOnlyFile(Readable, Seekable):
    """Read-only file implementing only necessary interfaces."""
    
    def __init__(self, filename: str, content: str):
        self.filename = filename
        self.content = content
        self.position = 0
    
    def read(self) -> str:
        """Read content from file."""
        return self.content[self.position:]
    
    def seek(self, position: int) -> bool:
        """Seek to position."""
        if 0 <= position <= len(self.content):
            self.position = position
            return True
        return False
    
    def tell(self) -> int:
        """Get current position."""
        return self.position


# Protocol implementations
class Circle:
    """Circle implementing Drawable protocol."""
    
    def __init__(self, radius: float):
        self.radius = radius
    
    def draw(self) -> str:
        """Draw the circle."""
        return f"Drawing circle with radius {self.radius}"
    
    def get_area(self) -> float:
        """Get circle area."""
        import math
        return math.pi * self.radius ** 2
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize circle."""
        return {
            'type': 'circle',
            'radius': self.radius
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Circle':
        """Deserialize circle."""
        return cls(data['radius'])


class Rectangle:
    """Rectangle implementing Drawable protocol."""
    
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def draw(self) -> str:
        """Draw the rectangle."""
        return f"Drawing rectangle {self.width}x{self.height}"
    
    def get_area(self) -> float:
        """Get rectangle area."""
        return self.width * self.height
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize rectangle."""
        return {
            'type': 'rectangle',
            'width': self.width,
            'height': self.height
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Rectangle':
        """Deserialize rectangle."""
        return cls(data['width'], data['height'])


# Service classes using dependency inversion
class PaymentProcessor:
    """Payment processor using dependency inversion."""
    
    def __init__(self, payment_method: PaymentMethod, notification_sender: NotificationSender):
        self.payment_method = payment_method  # Depends on interface, not concrete class
        self.notification_sender = notification_sender  # Depends on interface
        self.transaction_history: List[Dict[str, Any]] = []
    
    def process_payment(self, amount: float, payment_details: Dict[str, Any], 
                       customer_email: str) -> Dict[str, Any]:
        """Process payment using injected payment method."""
        # Process payment
        result = self.payment_method.process_payment(amount, payment_details)
        
        # Log transaction
        self.transaction_history.append({
            'timestamp': datetime.now(),
            'result': result,
            'customer_email': customer_email
        })
        
        # Send notification
        if result['success']:
            message = f"Payment of ${amount:.2f} processed successfully. Transaction ID: {result['transaction_id']}"
            self.notification_sender.send_notification(
                customer_email, 
                message, 
                "Payment Confirmation"
            )
        else:
            message = f"Payment of ${amount:.2f} failed: {result.get('error', 'Unknown error')}"
            self.notification_sender.send_notification(
                customer_email, 
                message, 
                "Payment Failed"
            )
        
        return result
    
    def refund_payment(self, transaction_id: str, amount: float, customer_email: str) -> Dict[str, Any]:
        """Process refund."""
        result = self.payment_method.refund_payment(transaction_id, amount)
        
        if result['success']:
            message = f"Refund of ${amount:.2f} processed. Refund ID: {result['refund_id']}"
            self.notification_sender.send_notification(
                customer_email, 
                message, 
                "Refund Processed"
            )
        
        return result


class DrawingCanvas:
    """Drawing canvas that works with any drawable object."""
    
    def __init__(self):
        self.shapes: List[Drawable] = []
    
    def add_shape(self, shape: Drawable) -> None:
        """Add drawable shape to canvas."""
        if isinstance(shape, Drawable):  # Runtime check for protocol
            self.shapes.append(shape)
        else:
            raise TypeError("Object must implement Drawable protocol")
    
    def draw_all(self) -> List[str]:
        """Draw all shapes on canvas."""
        drawings = []
        for shape in self.shapes:
            drawings.append(shape.draw())
        return drawings
    
    def get_total_area(self) -> float:
        """Get total area of all shapes."""
        return sum(shape.get_area() for shape in self.shapes)
    
    def serialize_canvas(self) -> Dict[str, Any]:
        """Serialize entire canvas."""
        serialized_shapes = []
        for shape in self.shapes:
            if isinstance(shape, Serializable):
                serialized_shapes.append(shape.serialize())
        
        return {
            'shapes': serialized_shapes,
            'total_area': self.get_total_area()
        }


# Interface for data access layer
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


class InMemoryUserRepository(Repository):
    """In-memory implementation of user repository."""
    
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
    
    def save(self, user: Dict[str, Any]) -> bool:
        """Save user."""
        if 'id' not in user:
            return False
        
        self.users[user['id']] = user
        return True
    
    def find_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Find user by ID."""
        return self.users.get(user_id)
    
    def find_all(self) -> List[Dict[str, Any]]:
        """Find all users."""
        return list(self.users.values())
    
    def delete(self, user_id: str) -> bool:
        """Delete user."""
        if user_id in self.users:
            del self.users[user_id]
            return True
        return False


class UserService:
    """User service using repository interface."""
    
    def __init__(self, repository: Repository):
        self.repository = repository  # Depends on interface
    
    def create_user(self, name: str, email: str) -> Dict[str, Any]:
        """Create new user."""
        user = {
            'id': f"USER_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'name': name,
            'email': email,
            'created_at': datetime.now().isoformat()
        }
        
        if self.repository.save(user):
            return user
        else:
            raise Exception("Failed to save user")
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        return self.repository.find_by_id(user_id)
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users."""
        return self.repository.find_all()
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        return self.repository.delete(user_id)


def demonstrate_interface_design_principles():
    """
    Demonstrate interface design principles and patterns.
    """
    print("=== INTERFACE DESIGN PRINCIPLES DEMONSTRATION ===\n")
    
    # 1. Interface Segregation Principle
    print("1. INTERFACE SEGREGATION PRINCIPLE:")
    print("   Splitting large interfaces into smaller, focused ones")
    
    # Text file implements all interfaces
    text_file = TextFile("document.txt")
    text_file.write("Hello, World!")
    text_file.seek(0)
    content = text_file.read()
    print(f"Text file content: {content}")
    print(f"Current position: {text_file.tell()}")
    
    # Read-only file implements only necessary interfaces
    readonly_file = ReadOnlyFile("readonly.txt", "This is read-only content")
    readonly_content = readonly_file.read()
    print(f"Read-only content: {readonly_content}")
    
    # Cannot write to read-only file (doesn't implement Writable)
    print(f"Text file is writable: {isinstance(text_file, Writable)}")
    print(f"Read-only file is writable: {isinstance(readonly_file, Writable)}")
    print()
    
    # 2. Protocol-based Interfaces
    print("2. PROTOCOL-BASED INTERFACES:")
    print("   Using protocols for duck typing with type checking")
    
    canvas = DrawingCanvas()
    
    # Create shapes that implement Drawable protocol
    circle = Circle(5.0)
    rectangle = Rectangle(4.0, 6.0)
    
    # Check if objects implement protocol
    print(f"Circle implements Drawable: {isinstance(circle, Drawable)}")
    print(f"Rectangle implements Drawable: {isinstance(rectangle, Drawable)}")
    
    # Add shapes to canvas
    canvas.add_shape(circle)
    canvas.add_shape(rectangle)
    
    # Draw all shapes
    drawings = canvas.draw_all()
    for drawing in drawings:
        print(f"  {drawing}")
    
    print(f"Total area: {canvas.get_total_area():.2f}")
    
    # Serialize canvas
    serialized = canvas.serialize_canvas()
    print(f"Serialized canvas: {serialized}")
    print()
    
    # 3. Dependency Inversion Principle
    print("3. DEPENDENCY INVERSION PRINCIPLE:")
    print("   High-level modules depend on abstractions, not concretions")
    
    # Create different payment methods
    credit_card = CreditCardPayment("https://gateway.visa.com", "MERCHANT123")
    paypal = PayPalPayment("api_key_123", "secret_456")
    
    # Create different notification methods
    email_notifier = EmailNotification("smtp.gmail.com", 587)
    sms_notifier = SMSNotification("sms_api_key", "https://sms.service.com")
    
    # Create payment processors with different combinations
    cc_email_processor = PaymentProcessor(credit_card, email_notifier)
    paypal_sms_processor = PaymentProcessor(paypal, sms_notifier)
    
    # Process payments using different implementations
    cc_payment_details = {
        'card_number': '1234567890123456',
        'expiry_month': 12,
        'expiry_year': 2025,
        'cvv': '123'
    }
    
    paypal_payment_details = {
        'email': 'customer@example.com'
    }
    
    print("Processing credit card payment with email notification:")
    cc_result = cc_email_processor.process_payment(100.0, cc_payment_details, "customer@example.com")
    print(f"Result: {cc_result['success']}")
    
    print("\nProcessing PayPal payment with SMS notification:")
    pp_result = paypal_sms_processor.process_payment(75.0, paypal_payment_details, "+1234567890")
    print(f"Result: {pp_result['success']}")
    
    # Process refunds
    if cc_result['success']:
        print("\nProcessing refund:")
        refund_result = cc_email_processor.refund_payment(cc_result['transaction_id'], 50.0, "customer@example.com")
        print(f"Refund result: {refund_result['success']}")
    
    print()
    
    # 4. Repository Pattern with Interface
    print("4. REPOSITORY PATTERN WITH INTERFACE:")
    print("   Data access layer abstraction")
    
    # Create repository and service
    user_repository = InMemoryUserRepository()
    user_service = UserService(user_repository)
    
    # Create users
    user1 = user_service.create_user("Alice Johnson", "alice@example.com")
    user2 = user_service.create_user("Bob Smith", "bob@example.com")
    
    print(f"Created user: {user1['name']} ({user1['id']})")
    print(f"Created user: {user2['name']} ({user2['id']})")
    
    # Retrieve users
    retrieved_user = user_service.get_user(user1['id'])
    if retrieved_user:
        print(f"Retrieved user: {retrieved_user['name']}")
    
    # Get all users
    all_users = user_service.get_all_users()
    print(f"Total users: {len(all_users)}")
    
    # Delete user
    deleted = user_service.delete_user(user2['id'])
    print(f"User deleted: {deleted}")
    print(f"Remaining users: {len(user_service.get_all_users())}")
    print()
    
    # 5. Interface Design Best Practices
    print("5. INTERFACE DESIGN BEST PRACTICES:")
    print("✓ Keep interfaces small and focused (Interface Segregation)")
    print("✓ Define clear contracts with abstract methods")
    print("✓ Use protocols for duck typing with type safety")
    print("✓ Depend on abstractions, not concretions (Dependency Inversion)")
    print("✓ Make interfaces easy to implement and test")
    print("✓ Document interface contracts clearly")
    print("✓ Use composition over inheritance for flexibility")
    print("✓ Design for extensibility and maintainability")
    print()
    
    # 6. Benefits of Interface-based Design
    print("6. BENEFITS OF INTERFACE-BASED DESIGN:")
    print("• Loose coupling between components")
    print("• Easy to test with mock implementations")
    print("• Flexible and extensible architecture")
    print("• Clear separation of concerns")
    print("• Support for multiple implementations")
    print("• Better code organization and maintainability")
    print()
    
    print("=== INTERFACE DESIGN PRINCIPLES DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_interface_design_principles()
