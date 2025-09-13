"""
LISKOV SUBSTITUTION PRINCIPLE - LSP and Proper Inheritance
==========================================================

Problem Statement:
Demonstrate the Liskov Substitution Principle (LSP):
- Objects of a superclass should be replaceable with objects of subclasses
- Subclasses must be substitutable for their base classes
- Behavioral subtyping and contract compliance
- Avoiding LSP violations in inheritance hierarchies
- Designing proper inheritance relationships

Learning Objectives:
- Understand the Liskov Substitution Principle
- Identify LSP violations in inheritance hierarchies
- Design proper inheritance relationships
- Ensure behavioral compatibility in subclasses
- Apply LSP in real-world scenarios
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import math


# VIOLATION EXAMPLE - LSP violation with improper inheritance
class BadRectangle:
    """
    BAD EXAMPLE: Base rectangle class that will be violated by Square.
    """
    
    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height
    
    @property
    def width(self) -> float:
        return self._width
    
    @width.setter
    def width(self, value: float) -> None:
        self._width = value
    
    @property
    def height(self) -> float:
        return self._height
    
    @height.setter
    def height(self, value: float) -> None:
        self._height = value
    
    def calculate_area(self) -> float:
        return self._width * self._height


class BadSquare(BadRectangle):
    """
    BAD EXAMPLE: Violates LSP because it changes the behavior of width/height setters.
    """
    
    def __init__(self, side: float):
        super().__init__(side, side)
    
    @property
    def width(self) -> float:
        return self._width
    
    @width.setter
    def width(self, value: float) -> None:
        # LSP VIOLATION: Changes both width and height, violating expected behavior
        self._width = value
        self._height = value  # This breaks the expected behavior!
    
    @property
    def height(self) -> float:
        return self._height
    
    @height.setter
    def height(self, value: float) -> None:
        # LSP VIOLATION: Changes both width and height, violating expected behavior
        self._width = value   # This breaks the expected behavior!
        self._height = value


def demonstrate_lsp_violation():
    """Demonstrate how BadSquare violates LSP."""
    
    def test_rectangle_behavior(rectangle: BadRectangle) -> None:
        """Function that expects rectangle behavior."""
        rectangle.width = 5
        rectangle.height = 10
        expected_area = 5 * 10  # 50
        actual_area = rectangle.calculate_area()
        
        print(f"Expected area: {expected_area}, Actual area: {actual_area}")
        print(f"Width: {rectangle.width}, Height: {rectangle.height}")
        
        # This assertion will fail for BadSquare!
        assert actual_area == expected_area, "LSP Violation: Unexpected behavior!"
    
    print("Testing with BadRectangle:")
    rect = BadRectangle(3, 4)
    test_rectangle_behavior(rect)  # Works fine
    
    print("\nTesting with BadSquare (LSP Violation):")
    square = BadSquare(3)
    try:
        test_rectangle_behavior(square)  # Will fail!
    except AssertionError as e:
        print(f"LSP Violation detected: {e}")


# GOOD EXAMPLE - LSP-compliant design using proper abstraction

# 1. Abstract base class defining the contract
class Shape(ABC):
    """
    Abstract shape class that defines the contract all shapes must follow.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.created_at = datetime.now()
    
    @abstractmethod
    def calculate_area(self) -> float:
        """Calculate the area of the shape."""
        pass
    
    @abstractmethod
    def calculate_perimeter(self) -> float:
        """Calculate the perimeter of the shape."""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> Dict[str, float]:
        """Get shape dimensions."""
        pass
    
    def get_shape_info(self) -> Dict[str, Any]:
        """Get complete shape information."""
        return {
            'name': self.name,
            'area': self.calculate_area(),
            'perimeter': self.calculate_perimeter(),
            'dimensions': self.get_dimensions(),
            'created_at': self.created_at.isoformat()
        }
    
    def __str__(self) -> str:
        return f"{self.name}(area={self.calculate_area():.2f})"


# 2. LSP-compliant Rectangle implementation
class Rectangle(Shape):
    """Rectangle implementation that follows LSP."""
    
    def __init__(self, width: float, height: float):
        super().__init__("Rectangle")
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        self._width = width
        self._height = height
    
    @property
    def width(self) -> float:
        return self._width
    
    @property
    def height(self) -> float:
        return self._height
    
    def calculate_area(self) -> float:
        """Calculate rectangle area."""
        return self._width * self._height
    
    def calculate_perimeter(self) -> float:
        """Calculate rectangle perimeter."""
        return 2 * (self._width + self._height)
    
    def get_dimensions(self) -> Dict[str, float]:
        """Get rectangle dimensions."""
        return {'width': self._width, 'height': self._height}
    
    def resize(self, width: float, height: float) -> 'Rectangle':
        """Create new rectangle with different dimensions (immutable approach)."""
        return Rectangle(width, height)


# 3. LSP-compliant Square implementation
class Square(Shape):
    """Square implementation that follows LSP by not inheriting from Rectangle."""
    
    def __init__(self, side: float):
        super().__init__("Square")
        if side <= 0:
            raise ValueError("Side must be positive")
        self._side = side
    
    @property
    def side(self) -> float:
        return self._side
    
    def calculate_area(self) -> float:
        """Calculate square area."""
        return self._side ** 2
    
    def calculate_perimeter(self) -> float:
        """Calculate square perimeter."""
        return 4 * self._side
    
    def get_dimensions(self) -> Dict[str, float]:
        """Get square dimensions."""
        return {'side': self._side}
    
    def resize(self, side: float) -> 'Square':
        """Create new square with different side (immutable approach)."""
        return Square(side)


# 4. Circle implementation following the same contract
class Circle(Shape):
    """Circle implementation that follows LSP."""
    
    def __init__(self, radius: float):
        super().__init__("Circle")
        if radius <= 0:
            raise ValueError("Radius must be positive")
        self._radius = radius
    
    @property
    def radius(self) -> float:
        return self._radius
    
    def calculate_area(self) -> float:
        """Calculate circle area."""
        return math.pi * self._radius ** 2
    
    def calculate_perimeter(self) -> float:
        """Calculate circle perimeter (circumference)."""
        return 2 * math.pi * self._radius
    
    def get_dimensions(self) -> Dict[str, float]:
        """Get circle dimensions."""
        return {'radius': self._radius}
    
    def resize(self, radius: float) -> 'Circle':
        """Create new circle with different radius."""
        return Circle(radius)


# BIRD EXAMPLE - Classic LSP violation and fix

# Violation example
class BadBird:
    """BAD EXAMPLE: Base bird class that will be violated by penguin."""
    
    def __init__(self, name: str):
        self.name = name
    
    def fly(self) -> str:
        """All birds can fly - this assumption violates LSP."""
        return f"{self.name} is flying"
    
    def eat(self) -> str:
        """All birds can eat."""
        return f"{self.name} is eating"


class BadPenguin(BadBird):
    """BAD EXAMPLE: Penguin violates LSP because it can't fly."""
    
    def fly(self) -> str:
        """LSP VIOLATION: Penguin can't fly but must implement fly method."""
        raise NotImplementedError("Penguins cannot fly!")


# LSP-compliant bird hierarchy
class Bird(ABC):
    """Abstract bird class with common bird behaviors."""
    
    def __init__(self, name: str, species: str):
        self.name = name
        self.species = species
    
    @abstractmethod
    def eat(self) -> str:
        """All birds can eat."""
        pass
    
    @abstractmethod
    def make_sound(self) -> str:
        """All birds make sounds."""
        pass
    
    def sleep(self) -> str:
        """All birds sleep."""
        return f"{self.name} is sleeping"


class FlyingBird(Bird):
    """Abstract class for birds that can fly."""
    
    @abstractmethod
    def fly(self) -> str:
        """Flying birds can fly."""
        pass
    
    @abstractmethod
    def get_flight_speed(self) -> float:
        """Get flight speed in km/h."""
        pass


class SwimmingBird(Bird):
    """Abstract class for birds that can swim."""
    
    @abstractmethod
    def swim(self) -> str:
        """Swimming birds can swim."""
        pass
    
    @abstractmethod
    def get_swim_speed(self) -> float:
        """Get swimming speed in km/h."""
        pass


# Concrete implementations following LSP
class Eagle(FlyingBird):
    """Eagle - a flying bird."""
    
    def eat(self) -> str:
        return f"{self.name} the eagle is hunting for prey"
    
    def make_sound(self) -> str:
        return f"{self.name} screeches loudly"
    
    def fly(self) -> str:
        return f"{self.name} soars high in the sky"
    
    def get_flight_speed(self) -> float:
        return 80.0  # km/h


class Sparrow(FlyingBird):
    """Sparrow - a flying bird."""
    
    def eat(self) -> str:
        return f"{self.name} the sparrow is eating seeds"
    
    def make_sound(self) -> str:
        return f"{self.name} chirps melodiously"
    
    def fly(self) -> str:
        return f"{self.name} flies quickly between trees"
    
    def get_flight_speed(self) -> float:
        return 25.0  # km/h


class Penguin(SwimmingBird):
    """Penguin - a swimming bird that follows LSP."""
    
    def eat(self) -> str:
        return f"{self.name} the penguin is eating fish"
    
    def make_sound(self) -> str:
        return f"{self.name} makes penguin sounds"
    
    def swim(self) -> str:
        return f"{self.name} swims gracefully underwater"
    
    def get_swim_speed(self) -> float:
        return 8.0  # km/h


class Duck(FlyingBird, SwimmingBird):
    """Duck - can both fly and swim."""
    
    def eat(self) -> str:
        return f"{self.name} the duck is eating aquatic plants"
    
    def make_sound(self) -> str:
        return f"{self.name} quacks loudly"
    
    def fly(self) -> str:
        return f"{self.name} flies over the pond"
    
    def get_flight_speed(self) -> float:
        return 50.0  # km/h
    
    def swim(self) -> str:
        return f"{self.name} swims on the water surface"
    
    def get_swim_speed(self) -> float:
        return 3.0  # km/h


# VEHICLE EXAMPLE - Proper LSP implementation

class Vehicle(ABC):
    """Abstract vehicle class defining the contract."""
    
    def __init__(self, make: str, model: str, year: int):
        self.make = make
        self.model = model
        self.year = year
        self.is_running = False
        self.fuel_level = 0.0
    
    @abstractmethod
    def start_engine(self) -> bool:
        """Start the vehicle engine."""
        pass
    
    @abstractmethod
    def stop_engine(self) -> bool:
        """Stop the vehicle engine."""
        pass
    
    @abstractmethod
    def get_max_speed(self) -> float:
        """Get maximum speed in km/h."""
        pass
    
    @abstractmethod
    def get_fuel_efficiency(self) -> float:
        """Get fuel efficiency in km/L."""
        pass
    
    def get_vehicle_info(self) -> Dict[str, Any]:
        """Get vehicle information."""
        return {
            'make': self.make,
            'model': self.model,
            'year': self.year,
            'is_running': self.is_running,
            'fuel_level': self.fuel_level,
            'max_speed': self.get_max_speed(),
            'fuel_efficiency': self.get_fuel_efficiency()
        }


class Car(Vehicle):
    """Car implementation following LSP."""
    
    def __init__(self, make: str, model: str, year: int, engine_size: float):
        super().__init__(make, model, year)
        self.engine_size = engine_size
        self.max_fuel_capacity = 50.0
    
    def start_engine(self) -> bool:
        """Start car engine."""
        if not self.is_running and self.fuel_level > 0:
            self.is_running = True
            return True
        return False
    
    def stop_engine(self) -> bool:
        """Stop car engine."""
        if self.is_running:
            self.is_running = False
            return True
        return False
    
    def get_max_speed(self) -> float:
        """Get car maximum speed."""
        return 180.0  # km/h
    
    def get_fuel_efficiency(self) -> float:
        """Get car fuel efficiency."""
        return 12.0  # km/L
    
    def add_fuel(self, amount: float) -> bool:
        """Add fuel to car."""
        if self.fuel_level + amount <= self.max_fuel_capacity:
            self.fuel_level += amount
            return True
        return False


class ElectricCar(Vehicle):
    """Electric car implementation following LSP."""
    
    def __init__(self, make: str, model: str, year: int, battery_capacity: float):
        super().__init__(make, model, year)
        self.battery_capacity = battery_capacity
        self.battery_level = 0.0
    
    def start_engine(self) -> bool:
        """Start electric motor."""
        if not self.is_running and self.battery_level > 0:
            self.is_running = True
            return True
        return False
    
    def stop_engine(self) -> bool:
        """Stop electric motor."""
        if self.is_running:
            self.is_running = False
            return True
        return False
    
    def get_max_speed(self) -> float:
        """Get electric car maximum speed."""
        return 200.0  # km/h
    
    def get_fuel_efficiency(self) -> float:
        """Get electric car efficiency (km per kWh equivalent)."""
        return 25.0  # km/kWh (equivalent)
    
    def charge_battery(self, amount: float) -> bool:
        """Charge the battery."""
        if self.battery_level + amount <= self.battery_capacity:
            self.battery_level += amount
            # Update fuel_level for consistency with base class contract
            self.fuel_level = (self.battery_level / self.battery_capacity) * 100
            return True
        return False


class Motorcycle(Vehicle):
    """Motorcycle implementation following LSP."""
    
    def __init__(self, make: str, model: str, year: int, engine_size: int):
        super().__init__(make, model, year)
        self.engine_size = engine_size
        self.max_fuel_capacity = 15.0
    
    def start_engine(self) -> bool:
        """Start motorcycle engine."""
        if not self.is_running and self.fuel_level > 0:
            self.is_running = True
            return True
        return False
    
    def stop_engine(self) -> bool:
        """Stop motorcycle engine."""
        if self.is_running:
            self.is_running = False
            return True
        return False
    
    def get_max_speed(self) -> float:
        """Get motorcycle maximum speed."""
        return 250.0  # km/h
    
    def get_fuel_efficiency(self) -> float:
        """Get motorcycle fuel efficiency."""
        return 20.0  # km/L


# Functions that work with any LSP-compliant objects
def test_shape_behavior(shapes: List[Shape]) -> None:
    """Test that all shapes behave consistently (LSP compliance)."""
    print("Testing shape behavior (LSP compliance):")
    
    for shape in shapes:
        info = shape.get_shape_info()
        print(f"  {shape.name}: Area={info['area']:.2f}, Perimeter={info['perimeter']:.2f}")
        
        # All shapes should have positive area and perimeter
        assert info['area'] > 0, f"Shape {shape.name} has invalid area"
        assert info['perimeter'] > 0, f"Shape {shape.name} has invalid perimeter"
        
        # All shapes should have dimensions
        assert len(info['dimensions']) > 0, f"Shape {shape.name} has no dimensions"


def test_vehicle_behavior(vehicles: List[Vehicle]) -> None:
    """Test that all vehicles behave consistently (LSP compliance)."""
    print("Testing vehicle behavior (LSP compliance):")
    
    for vehicle in vehicles:
        # Add some fuel/charge
        if isinstance(vehicle, Car):
            vehicle.add_fuel(20)
        elif isinstance(vehicle, ElectricCar):
            vehicle.charge_battery(50)
        elif isinstance(vehicle, Motorcycle):
            vehicle.fuel_level = 10
        
        # Test start/stop behavior
        started = vehicle.start_engine()
        print(f"  {vehicle.make} {vehicle.model}: Started={started}")
        
        if started:
            info = vehicle.get_vehicle_info()
            print(f"    Max Speed: {info['max_speed']} km/h")
            print(f"    Fuel Efficiency: {info['fuel_efficiency']} km/L")
            
            stopped = vehicle.stop_engine()
            print(f"    Stopped: {stopped}")
        
        # All vehicles should follow the same contract
        assert hasattr(vehicle, 'is_running'), "Vehicle missing is_running attribute"
        assert hasattr(vehicle, 'fuel_level'), "Vehicle missing fuel_level attribute"


def test_bird_behavior(birds: List[Bird]) -> None:
    """Test that all birds behave consistently (LSP compliance)."""
    print("Testing bird behavior (LSP compliance):")
    
    for bird in birds:
        print(f"  {bird.name} ({bird.species}):")
        print(f"    Eating: {bird.eat()}")
        print(f"    Sound: {bird.make_sound()}")
        print(f"    Sleeping: {bird.sleep()}")
        
        # Test specific abilities
        if isinstance(bird, FlyingBird):
            print(f"    Flying: {bird.fly()}")
            print(f"    Flight Speed: {bird.get_flight_speed()} km/h")
        
        if isinstance(bird, SwimmingBird):
            print(f"    Swimming: {bird.swim()}")
            print(f"    Swim Speed: {bird.get_swim_speed()} km/h")


def demonstrate_liskov_substitution_principle():
    """
    Demonstrate Liskov Substitution Principle with practical examples.
    """
    print("=== LISKOV SUBSTITUTION PRINCIPLE DEMONSTRATION ===\n")
    
    # 1. Show LSP violation
    print("1. LSP VIOLATION EXAMPLE:")
    demonstrate_lsp_violation()
    print()
    
    # 2. LSP-compliant shape hierarchy
    print("2. LSP-COMPLIANT SHAPE HIERARCHY:")
    
    shapes = [
        Rectangle(5, 3),
        Square(4),
        Circle(3)
    ]
    
    test_shape_behavior(shapes)
    print()
    
    # 3. LSP-compliant bird hierarchy
    print("3. LSP-COMPLIANT BIRD HIERARCHY:")
    
    birds = [
        Eagle("Eddie", "Bald Eagle"),
        Sparrow("Sparky", "House Sparrow"),
        Penguin("Penny", "Emperor Penguin"),
        Duck("Daffy", "Mallard Duck")
    ]
    
    test_bird_behavior(birds)
    print()
    
    # 4. LSP-compliant vehicle hierarchy
    print("4. LSP-COMPLIANT VEHICLE HIERARCHY:")
    
    vehicles = [
        Car("Toyota", "Camry", 2023, 2.5),
        ElectricCar("Tesla", "Model 3", 2023, 75.0),
        Motorcycle("Harley-Davidson", "Street 750", 2023, 750)
    ]
    
    test_vehicle_behavior(vehicles)
    print()
    
    # 5. Polymorphic usage demonstration
    print("5. POLYMORPHIC USAGE (LSP in Action):")
    
    def calculate_total_area(shapes: List[Shape]) -> float:
        """Function that works with any Shape subclass."""
        return sum(shape.calculate_area() for shape in shapes)
    
    def start_all_vehicles(vehicles: List[Vehicle]) -> int:
        """Function that works with any Vehicle subclass."""
        started_count = 0
        for vehicle in vehicles:
            if vehicle.start_engine():
                started_count += 1
        return started_count
    
    total_area = calculate_total_area(shapes)
    print(f"   Total area of all shapes: {total_area:.2f}")
    
    # Reset vehicles and add fuel/charge
    for vehicle in vehicles:
        vehicle.stop_engine()
        if isinstance(vehicle, Car):
            vehicle.add_fuel(30)
        elif isinstance(vehicle, ElectricCar):
            vehicle.charge_battery(60)
        elif isinstance(vehicle, Motorcycle):
            vehicle.fuel_level = 12
    
    started_vehicles = start_all_vehicles(vehicles)
    print(f"   Started vehicles: {started_vehicles}/{len(vehicles)}")
    print()
    
    # 6. LSP Benefits and Guidelines
    print("6. LSP BENEFITS AND GUIDELINES:")
    print("   ✓ Subclasses can be used wherever base class is expected")
    print("   ✓ Polymorphism works correctly without surprises")
    print("   ✓ Client code doesn't need to know about specific subclasses")
    print("   ✓ Inheritance hierarchies are logically consistent")
    print("   ✓ Code is more maintainable and extensible")
    print()
    
    print("   LSP Guidelines:")
    print("   • Preconditions cannot be strengthened in subclasses")
    print("   • Postconditions cannot be weakened in subclasses")
    print("   • Invariants of the base class must be preserved")
    print("   • History constraint (new methods shouldn't change state unexpectedly)")
    print("   • Subclasses should not throw new exceptions")
    print("   • Behavioral compatibility is more important than structural compatibility")
    print()
    
    print("=== LISKOV SUBSTITUTION PRINCIPLE DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_liskov_substitution_principle()
