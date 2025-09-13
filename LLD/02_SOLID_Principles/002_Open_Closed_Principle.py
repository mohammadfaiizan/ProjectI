"""
OPEN CLOSED PRINCIPLE - OCP Implementation Strategies
====================================================

Problem Statement:
Demonstrate the Open/Closed Principle (OCP):
- Software entities should be open for extension but closed for modification
- Adding new functionality without changing existing code
- Using abstraction and polymorphism for extensibility
- Strategy pattern and other OCP-enabling patterns
- Plugin architecture and extensible systems

Learning Objectives:
- Understand the Open/Closed Principle
- Design extensible systems without modifying existing code
- Use abstraction to enable extension
- Implement strategy pattern for OCP compliance
- Create plugin-based architectures
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import math


# VIOLATION EXAMPLE - Modifying existing code for new functionality
class BadShapeCalculator:
    """
    BAD EXAMPLE: Violates OCP because adding new shapes requires modifying this class.
    """
    
    def calculate_area(self, shape_type: str, **kwargs) -> float:
        """Calculate area - requires modification for each new shape type."""
        
        if shape_type == "rectangle":
            return kwargs["width"] * kwargs["height"]
        elif shape_type == "circle":
            return math.pi * kwargs["radius"] ** 2
        elif shape_type == "triangle":
            return 0.5 * kwargs["base"] * kwargs["height"]
        # Adding new shapes requires modifying this method!
        # elif shape_type == "square":
        #     return kwargs["side"] ** 2
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")


# GOOD EXAMPLE - Following OCP with abstraction and polymorphism

# 1. Abstract base class for shapes (closed for modification)
class Shape(ABC):
    """
    Abstract shape class - closed for modification, open for extension.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.created_at = datetime.now()
    
    @abstractmethod
    def calculate_area(self) -> float:
        """Calculate area of the shape."""
        pass
    
    @abstractmethod
    def calculate_perimeter(self) -> float:
        """Calculate perimeter of the shape."""
        pass
    
    def get_shape_info(self) -> Dict[str, Any]:
        """Get shape information."""
        return {
            'name': self.name,
            'area': self.calculate_area(),
            'perimeter': self.calculate_perimeter(),
            'created_at': self.created_at.isoformat()
        }
    
    def __str__(self) -> str:
        return f"{self.name}(area={self.calculate_area():.2f})"


# 2. Concrete shape implementations (extensions)
class Rectangle(Shape):
    """Rectangle shape implementation."""
    
    def __init__(self, width: float, height: float):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def calculate_area(self) -> float:
        """Calculate rectangle area."""
        return self.width * self.height
    
    def calculate_perimeter(self) -> float:
        """Calculate rectangle perimeter."""
        return 2 * (self.width + self.height)


class Circle(Shape):
    """Circle shape implementation."""
    
    def __init__(self, radius: float):
        super().__init__("Circle")
        self.radius = radius
    
    def calculate_area(self) -> float:
        """Calculate circle area."""
        return math.pi * self.radius ** 2
    
    def calculate_perimeter(self) -> float:
        """Calculate circle perimeter (circumference)."""
        return 2 * math.pi * self.radius


class Triangle(Shape):
    """Triangle shape implementation."""
    
    def __init__(self, base: float, height: float, side1: float, side2: float):
        super().__init__("Triangle")
        self.base = base
        self.height = height
        self.side1 = side1
        self.side2 = side2
    
    def calculate_area(self) -> float:
        """Calculate triangle area."""
        return 0.5 * self.base * self.height
    
    def calculate_perimeter(self) -> float:
        """Calculate triangle perimeter."""
        return self.base + self.side1 + self.side2


# 3. NEW SHAPES can be added without modifying existing code
class Square(Shape):
    """Square shape - new extension without modifying existing code."""
    
    def __init__(self, side: float):
        super().__init__("Square")
        self.side = side
    
    def calculate_area(self) -> float:
        """Calculate square area."""
        return self.side ** 2
    
    def calculate_perimeter(self) -> float:
        """Calculate square perimeter."""
        return 4 * self.side


class Pentagon(Shape):
    """Pentagon shape - another new extension."""
    
    def __init__(self, side: float):
        super().__init__("Pentagon")
        self.side = side
    
    def calculate_area(self) -> float:
        """Calculate pentagon area."""
        return (1/4) * math.sqrt(25 + 10 * math.sqrt(5)) * self.side ** 2
    
    def calculate_perimeter(self) -> float:
        """Calculate pentagon perimeter."""
        return 5 * self.side


# 4. Shape calculator that works with any shape (OCP compliant)
class ShapeCalculator:
    """
    Shape calculator that follows OCP.
    Can work with any shape without modification.
    """
    
    def __init__(self):
        self.calculations_performed = 0
    
    def calculate_total_area(self, shapes: List[Shape]) -> float:
        """Calculate total area of all shapes."""
        total_area = 0
        for shape in shapes:
            total_area += shape.calculate_area()
            self.calculations_performed += 1
        return total_area
    
    def calculate_total_perimeter(self, shapes: List[Shape]) -> float:
        """Calculate total perimeter of all shapes."""
        total_perimeter = 0
        for shape in shapes:
            total_perimeter += shape.calculate_perimeter()
            self.calculations_performed += 1
        return total_perimeter
    
    def get_shape_statistics(self, shapes: List[Shape]) -> Dict[str, Any]:
        """Get statistics for all shapes."""
        if not shapes:
            return {}
        
        areas = [shape.calculate_area() for shape in shapes]
        perimeters = [shape.calculate_perimeter() for shape in shapes]
        
        return {
            'total_shapes': len(shapes),
            'total_area': sum(areas),
            'average_area': sum(areas) / len(areas),
            'largest_area': max(areas),
            'smallest_area': min(areas),
            'total_perimeter': sum(perimeters),
            'shape_types': list(set(shape.name for shape in shapes))
        }


# STRATEGY PATTERN EXAMPLE - Another OCP implementation

# 1. Abstract strategy for discount calculation
class DiscountStrategy(ABC):
    """Abstract discount strategy - closed for modification, open for extension."""
    
    @abstractmethod
    def calculate_discount(self, amount: float) -> float:
        """Calculate discount amount."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get discount description."""
        pass


# 2. Concrete discount strategies (extensions)
class NoDiscount(DiscountStrategy):
    """No discount strategy."""
    
    def calculate_discount(self, amount: float) -> float:
        """No discount applied."""
        return 0.0
    
    def get_description(self) -> str:
        """Get description."""
        return "No discount"


class PercentageDiscount(DiscountStrategy):
    """Percentage-based discount strategy."""
    
    def __init__(self, percentage: float):
        if not 0 <= percentage <= 100:
            raise ValueError("Percentage must be between 0 and 100")
        self.percentage = percentage
    
    def calculate_discount(self, amount: float) -> float:
        """Calculate percentage discount."""
        return amount * (self.percentage / 100)
    
    def get_description(self) -> str:
        """Get description."""
        return f"{self.percentage}% discount"


class FixedAmountDiscount(DiscountStrategy):
    """Fixed amount discount strategy."""
    
    def __init__(self, discount_amount: float):
        if discount_amount < 0:
            raise ValueError("Discount amount cannot be negative")
        self.discount_amount = discount_amount
    
    def calculate_discount(self, amount: float) -> float:
        """Calculate fixed discount."""
        return min(self.discount_amount, amount)
    
    def get_description(self) -> str:
        """Get description."""
        return f"${self.discount_amount:.2f} off"


class TieredDiscount(DiscountStrategy):
    """Tiered discount strategy - new extension."""
    
    def __init__(self, tiers: List[tuple]):
        """
        Initialize with tiers as list of (threshold, percentage) tuples.
        Example: [(100, 5), (500, 10), (1000, 15)]
        """
        self.tiers = sorted(tiers, key=lambda x: x[0])  # Sort by threshold
    
    def calculate_discount(self, amount: float) -> float:
        """Calculate tiered discount."""
        for threshold, percentage in reversed(self.tiers):
            if amount >= threshold:
                return amount * (percentage / 100)
        return 0.0
    
    def get_description(self) -> str:
        """Get description."""
        tier_descriptions = [f"${threshold}+: {percentage}%" for threshold, percentage in self.tiers]
        return f"Tiered discount ({', '.join(tier_descriptions)})"


class BuyOneGetOneDiscount(DiscountStrategy):
    """BOGO discount strategy - another new extension."""
    
    def __init__(self, item_price: float):
        self.item_price = item_price
    
    def calculate_discount(self, amount: float) -> float:
        """Calculate BOGO discount."""
        if amount >= self.item_price * 2:
            # Get one item free for every two items
            pairs = int(amount // (self.item_price * 2))
            return pairs * self.item_price
        return 0.0
    
    def get_description(self) -> str:
        """Get description."""
        return f"Buy One Get One Free (item price: ${self.item_price:.2f})"


# 3. Order class using discount strategy
class Order:
    """Order class that uses discount strategy (OCP compliant)."""
    
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.items: List[Dict[str, Any]] = []
        self.discount_strategy: DiscountStrategy = NoDiscount()
        self.created_at = datetime.now()
    
    def add_item(self, name: str, price: float, quantity: int = 1) -> None:
        """Add item to order."""
        self.items.append({
            'name': name,
            'price': price,
            'quantity': quantity,
            'total': price * quantity
        })
    
    def set_discount_strategy(self, strategy: DiscountStrategy) -> None:
        """Set discount strategy."""
        self.discount_strategy = strategy
    
    def calculate_subtotal(self) -> float:
        """Calculate order subtotal."""
        return sum(item['total'] for item in self.items)
    
    def calculate_discount(self) -> float:
        """Calculate discount amount."""
        subtotal = self.calculate_subtotal()
        return self.discount_strategy.calculate_discount(subtotal)
    
    def calculate_total(self) -> float:
        """Calculate final total."""
        return self.calculate_subtotal() - self.calculate_discount()
    
    def get_order_summary(self) -> Dict[str, Any]:
        """Get order summary."""
        return {
            'order_id': self.order_id,
            'items': self.items,
            'subtotal': self.calculate_subtotal(),
            'discount_description': self.discount_strategy.get_description(),
            'discount_amount': self.calculate_discount(),
            'total': self.calculate_total(),
            'created_at': self.created_at.isoformat()
        }


# PLUGIN ARCHITECTURE EXAMPLE - Advanced OCP implementation

class PaymentProcessor(ABC):
    """Abstract payment processor for plugin architecture."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get processor name."""
        pass
    
    @abstractmethod
    def validate_payment_data(self, payment_data: Dict[str, Any]) -> bool:
        """Validate payment data."""
        pass
    
    @abstractmethod
    def process_payment(self, amount: float, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process payment."""
        pass
    
    @abstractmethod
    def get_supported_currencies(self) -> List[str]:
        """Get supported currencies."""
        pass


class CreditCardProcessor(PaymentProcessor):
    """Credit card payment processor plugin."""
    
    def get_name(self) -> str:
        """Get processor name."""
        return "Credit Card"
    
    def validate_payment_data(self, payment_data: Dict[str, Any]) -> bool:
        """Validate credit card data."""
        required_fields = ['card_number', 'expiry_month', 'expiry_year', 'cvv']
        return all(field in payment_data for field in required_fields)
    
    def process_payment(self, amount: float, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process credit card payment."""
        if not self.validate_payment_data(payment_data):
            return {'success': False, 'error': 'Invalid payment data'}
        
        # Simulate processing
        return {
            'success': True,
            'transaction_id': f"CC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'amount': amount,
            'processor': self.get_name()
        }
    
    def get_supported_currencies(self) -> List[str]:
        """Get supported currencies."""
        return ['USD', 'EUR', 'GBP', 'CAD']


class PayPalProcessor(PaymentProcessor):
    """PayPal payment processor plugin."""
    
    def get_name(self) -> str:
        """Get processor name."""
        return "PayPal"
    
    def validate_payment_data(self, payment_data: Dict[str, Any]) -> bool:
        """Validate PayPal data."""
        return 'email' in payment_data and '@' in payment_data['email']
    
    def process_payment(self, amount: float, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process PayPal payment."""
        if not self.validate_payment_data(payment_data):
            return {'success': False, 'error': 'Invalid PayPal email'}
        
        return {
            'success': True,
            'transaction_id': f"PP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'amount': amount,
            'processor': self.get_name(),
            'email': payment_data['email']
        }
    
    def get_supported_currencies(self) -> List[str]:
        """Get supported currencies."""
        return ['USD', 'EUR', 'GBP', 'AUD', 'JPY']


class CryptocurrencyProcessor(PaymentProcessor):
    """Cryptocurrency payment processor plugin - new extension."""
    
    def get_name(self) -> str:
        """Get processor name."""
        return "Cryptocurrency"
    
    def validate_payment_data(self, payment_data: Dict[str, Any]) -> bool:
        """Validate cryptocurrency data."""
        return 'wallet_address' in payment_data and 'currency' in payment_data
    
    def process_payment(self, amount: float, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process cryptocurrency payment."""
        if not self.validate_payment_data(payment_data):
            return {'success': False, 'error': 'Invalid cryptocurrency data'}
        
        return {
            'success': True,
            'transaction_id': f"CRYPTO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'amount': amount,
            'processor': self.get_name(),
            'currency': payment_data['currency'],
            'wallet_address': payment_data['wallet_address']
        }
    
    def get_supported_currencies(self) -> List[str]:
        """Get supported currencies."""
        return ['BTC', 'ETH', 'LTC', 'ADA']


class PaymentGateway:
    """Payment gateway that supports multiple processors (OCP compliant)."""
    
    def __init__(self):
        self.processors: Dict[str, PaymentProcessor] = {}
        self.transaction_history: List[Dict[str, Any]] = []
    
    def register_processor(self, processor: PaymentProcessor) -> None:
        """Register a payment processor plugin."""
        self.processors[processor.get_name()] = processor
    
    def get_available_processors(self) -> List[str]:
        """Get list of available processors."""
        return list(self.processors.keys())
    
    def process_payment(self, processor_name: str, amount: float, 
                       payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process payment using specified processor."""
        if processor_name not in self.processors:
            return {'success': False, 'error': f'Processor {processor_name} not available'}
        
        processor = self.processors[processor_name]
        result = processor.process_payment(amount, payment_data)
        
        # Log transaction
        self.transaction_history.append({
            'timestamp': datetime.now().isoformat(),
            'processor': processor_name,
            'amount': amount,
            'result': result
        })
        
        return result
    
    def get_supported_currencies(self, processor_name: str) -> List[str]:
        """Get supported currencies for processor."""
        if processor_name in self.processors:
            return self.processors[processor_name].get_supported_currencies()
        return []


def demonstrate_open_closed_principle():
    """
    Demonstrate Open/Closed Principle with practical examples.
    """
    print("=== OPEN CLOSED PRINCIPLE DEMONSTRATION ===\n")
    
    # 1. Show the problem with OCP violation
    print("1. OCP VIOLATION EXAMPLE:")
    print("   BadShapeCalculator requires modification for each new shape type.")
    print("   Adding a new shape means changing existing, tested code.\n")
    
    # 2. Demonstrate OCP-compliant shape system
    print("2. OCP-COMPLIANT SHAPE SYSTEM:")
    
    # Create various shapes
    shapes = [
        Rectangle(5, 3),
        Circle(4),
        Triangle(6, 4, 5, 5),
        Square(4),  # New shape added without modifying existing code
        Pentagon(3)  # Another new shape
    ]
    
    print("   Created shapes:")
    for shape in shapes:
        print(f"     {shape}")
    
    # Use shape calculator (works with all shapes without modification)
    calculator = ShapeCalculator()
    total_area = calculator.calculate_total_area(shapes)
    total_perimeter = calculator.calculate_total_perimeter(shapes)
    
    print(f"\n   Total area of all shapes: {total_area:.2f}")
    print(f"   Total perimeter of all shapes: {total_perimeter:.2f}")
    
    # Get statistics
    stats = calculator.get_shape_statistics(shapes)
    print(f"   Shape statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"     {key}: {value:.2f}")
        else:
            print(f"     {key}: {value}")
    
    print()
    
    # 3. Strategy Pattern Example
    print("3. STRATEGY PATTERN FOR DISCOUNTS:")
    
    # Create order
    order = Order("ORD001")
    order.add_item("Laptop", 1000.00, 1)
    order.add_item("Mouse", 25.00, 2)
    order.add_item("Keyboard", 75.00, 1)
    
    print(f"   Order subtotal: ${order.calculate_subtotal():.2f}")
    
    # Test different discount strategies
    discount_strategies = [
        NoDiscount(),
        PercentageDiscount(10),
        FixedAmountDiscount(50),
        TieredDiscount([(100, 5), (500, 10), (1000, 15)]),
        BuyOneGetOneDiscount(25.00)  # New strategy added without modifying existing code
    ]
    
    print("\n   Testing different discount strategies:")
    for strategy in discount_strategies:
        order.set_discount_strategy(strategy)
        discount = order.calculate_discount()
        total = order.calculate_total()
        print(f"     {strategy.get_description()}: -${discount:.2f} = ${total:.2f}")
    
    print()
    
    # 4. Plugin Architecture Example
    print("4. PLUGIN ARCHITECTURE FOR PAYMENT PROCESSING:")
    
    # Create payment gateway
    gateway = PaymentGateway()
    
    # Register payment processors (plugins)
    gateway.register_processor(CreditCardProcessor())
    gateway.register_processor(PayPalProcessor())
    gateway.register_processor(CryptocurrencyProcessor())  # New processor added
    
    print("   Available payment processors:")
    for processor in gateway.get_available_processors():
        currencies = gateway.get_supported_currencies(processor)
        print(f"     {processor}: {', '.join(currencies)}")
    
    # Process payments with different processors
    print("\n   Processing payments:")
    
    # Credit card payment
    cc_result = gateway.process_payment("Credit Card", 100.00, {
        'card_number': '1234567890123456',
        'expiry_month': 12,
        'expiry_year': 2025,
        'cvv': '123'
    })
    print(f"     Credit Card: {'Success' if cc_result['success'] else 'Failed'}")
    
    # PayPal payment
    pp_result = gateway.process_payment("PayPal", 75.00, {
        'email': 'user@example.com'
    })
    print(f"     PayPal: {'Success' if pp_result['success'] else 'Failed'}")
    
    # Cryptocurrency payment
    crypto_result = gateway.process_payment("Cryptocurrency", 200.00, {
        'wallet_address': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
        'currency': 'BTC'
    })
    print(f"     Cryptocurrency: {'Success' if crypto_result['success'] else 'Failed'}")
    
    print(f"\n   Total transactions processed: {len(gateway.transaction_history)}")
    
    print()
    
    # 5. Benefits of OCP
    print("5. BENEFITS OF OPEN/CLOSED PRINCIPLE:")
    print("   ✓ New functionality can be added without modifying existing code")
    print("   ✓ Existing code remains stable and tested")
    print("   ✓ Reduces risk of introducing bugs in working code")
    print("   ✓ Supports plugin architectures and extensible systems")
    print("   ✓ Follows the principle of 'closed for modification, open for extension'")
    print("   ✓ Enables polymorphism and abstraction")
    print("   ✓ Makes systems more maintainable and flexible")
    print()
    
    # 6. OCP Implementation Techniques
    print("6. OCP IMPLEMENTATION TECHNIQUES:")
    print("   • Abstract base classes and interfaces")
    print("   • Strategy pattern for algorithm variations")
    print("   • Template method pattern")
    print("   • Observer pattern for event handling")
    print("   • Plugin architectures")
    print("   • Dependency injection")
    print("   • Factory patterns for object creation")
    print()
    
    print("=== OPEN CLOSED PRINCIPLE DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_open_closed_principle()
