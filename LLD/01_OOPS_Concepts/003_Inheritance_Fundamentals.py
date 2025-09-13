"""
INHERITANCE FUNDAMENTALS - Single and Multiple Inheritance
==========================================================

Problem Statement:
Demonstrate inheritance concepts including:
- Single inheritance and class hierarchies
- Multiple inheritance and method resolution order
- Super() method usage and constructor chaining
- Inheritance vs composition trade-offs
- Abstract base classes and concrete implementations

Learning Objectives:
- Understand inheritance relationships and hierarchies
- Master super() method for parent class access
- Handle multiple inheritance complexities
- Design proper class hierarchies
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
import math


# Base class for single inheritance demonstration
class Vehicle:
    """
    Base Vehicle class demonstrating fundamental inheritance concepts.
    """
    
    def __init__(self, make: str, model: str, year: int, color: str):
        """
        Initialize vehicle with basic properties.
        
        Args:
            make: Vehicle manufacturer
            model: Vehicle model
            year: Manufacturing year
            color: Vehicle color
        """
        self.make = make
        self.model = model
        self.year = year
        self.color = color
        self.mileage = 0.0
        self.is_running = False
        self.fuel_level = 0.0
        self.max_fuel_capacity = 50.0  # Default capacity
    
    def start_engine(self) -> bool:
        """Start the vehicle engine."""
        if not self.is_running:
            self.is_running = True
            print(f"{self.make} {self.model} engine started.")
            return True
        else:
            print(f"{self.make} {self.model} engine is already running.")
            return False
    
    def stop_engine(self) -> bool:
        """Stop the vehicle engine."""
        if self.is_running:
            self.is_running = False
            print(f"{self.make} {self.model} engine stopped.")
            return True
        else:
            print(f"{self.make} {self.model} engine is already stopped.")
            return False
    
    def add_fuel(self, amount: float) -> bool:
        """Add fuel to the vehicle."""
        if amount <= 0:
            print("Fuel amount must be positive.")
            return False
        
        if self.fuel_level + amount > self.max_fuel_capacity:
            amount = self.max_fuel_capacity - self.fuel_level
        
        self.fuel_level += amount
        print(f"Added {amount:.1f}L fuel. Current level: {self.fuel_level:.1f}L")
        return True
    
    def drive(self, distance: float) -> bool:
        """Drive the vehicle for a given distance."""
        if not self.is_running:
            print("Cannot drive. Engine is not running.")
            return False
        
        if distance <= 0:
            print("Distance must be positive.")
            return False
        
        fuel_needed = distance * 0.1  # 0.1L per km
        if self.fuel_level < fuel_needed:
            print("Insufficient fuel for the journey.")
            return False
        
        self.mileage += distance
        self.fuel_level -= fuel_needed
        print(f"Drove {distance}km. Total mileage: {self.mileage}km")
        return True
    
    def get_vehicle_info(self) -> Dict[str, Any]:
        """Get vehicle information."""
        return {
            'make': self.make,
            'model': self.model,
            'year': self.year,
            'color': self.color,
            'mileage': self.mileage,
            'fuel_level': self.fuel_level,
            'is_running': self.is_running
        }
    
    def __str__(self) -> str:
        return f"{self.year} {self.make} {self.model} ({self.color})"


# Single inheritance examples
class Car(Vehicle):
    """
    Car class inheriting from Vehicle.
    Demonstrates single inheritance and method extension.
    """
    
    def __init__(self, make: str, model: str, year: int, color: str, 
                 num_doors: int, transmission: str):
        """
        Initialize car with additional car-specific properties.
        
        Args:
            make: Car manufacturer
            model: Car model
            year: Manufacturing year
            color: Car color
            num_doors: Number of doors
            transmission: Transmission type (manual/automatic)
        """
        # Call parent constructor using super()
        super().__init__(make, model, year, color)
        
        # Car-specific attributes
        self.num_doors = num_doors
        self.transmission = transmission
        self.max_fuel_capacity = 60.0  # Override parent value
        self.air_conditioning = False
        self.radio_on = False
    
    def turn_on_ac(self) -> bool:
        """Turn on air conditioning."""
        if self.is_running:
            self.air_conditioning = True
            print("Air conditioning turned on.")
            return True
        else:
            print("Cannot turn on AC. Engine is not running.")
            return False
    
    def turn_off_ac(self) -> bool:
        """Turn off air conditioning."""
        self.air_conditioning = False
        print("Air conditioning turned off.")
        return True
    
    def turn_on_radio(self) -> bool:
        """Turn on radio."""
        self.radio_on = True
        print("Radio turned on.")
        return True
    
    def turn_off_radio(self) -> bool:
        """Turn off radio."""
        self.radio_on = False
        print("Radio turned off.")
        return True
    
    # Override parent method to add car-specific behavior
    def start_engine(self) -> bool:
        """Start car engine with additional checks."""
        if super().start_engine():  # Call parent method
            print("Car systems initialized.")
            return True
        return False
    
    def get_vehicle_info(self) -> Dict[str, Any]:
        """Get car information including parent info."""
        info = super().get_vehicle_info()  # Get parent info
        info.update({
            'num_doors': self.num_doors,
            'transmission': self.transmission,
            'air_conditioning': self.air_conditioning,
            'radio_on': self.radio_on
        })
        return info


class Motorcycle(Vehicle):
    """
    Motorcycle class inheriting from Vehicle.
    Demonstrates inheritance with different specialization.
    """
    
    def __init__(self, make: str, model: str, year: int, color: str, 
                 engine_size: int, has_sidecar: bool = False):
        """
        Initialize motorcycle with specific properties.
        
        Args:
            make: Motorcycle manufacturer
            model: Motorcycle model
            year: Manufacturing year
            color: Motorcycle color
            engine_size: Engine size in CC
            has_sidecar: Whether motorcycle has a sidecar
        """
        super().__init__(make, model, year, color)
        
        self.engine_size = engine_size
        self.has_sidecar = has_sidecar
        self.max_fuel_capacity = 20.0  # Smaller tank
        self.helmet_count = 0
    
    def add_helmet(self) -> None:
        """Add a helmet."""
        self.helmet_count += 1
        print(f"Helmet added. Total helmets: {self.helmet_count}")
    
    def remove_helmet(self) -> bool:
        """Remove a helmet."""
        if self.helmet_count > 0:
            self.helmet_count -= 1
            print(f"Helmet removed. Remaining helmets: {self.helmet_count}")
            return True
        else:
            print("No helmets to remove.")
            return False
    
    def wheelie(self) -> bool:
        """Perform a wheelie (motorcycle-specific method)."""
        if self.is_running and not self.has_sidecar:
            print(f"{self.make} {self.model} performing wheelie!")
            return True
        elif self.has_sidecar:
            print("Cannot perform wheelie with sidecar.")
            return False
        else:
            print("Engine must be running to perform wheelie.")
            return False
    
    def get_vehicle_info(self) -> Dict[str, Any]:
        """Get motorcycle information."""
        info = super().get_vehicle_info()
        info.update({
            'engine_size': self.engine_size,
            'has_sidecar': self.has_sidecar,
            'helmet_count': self.helmet_count
        })
        return info


# Multiple inheritance demonstration
class Electric:
    """
    Mixin class for electric vehicle functionality.
    """
    
    def __init__(self, battery_capacity: float, charging_speed: float):
        """
        Initialize electric vehicle components.
        
        Args:
            battery_capacity: Battery capacity in kWh
            charging_speed: Charging speed in kW
        """
        self.battery_capacity = battery_capacity
        self.battery_level = 0.0
        self.charging_speed = charging_speed
        self.is_charging = False
    
    def charge_battery(self, hours: float) -> float:
        """
        Charge the battery for given hours.
        
        Args:
            hours: Charging time in hours
            
        Returns:
            float: Amount of energy added
        """
        if self.is_charging:
            energy_added = min(
                self.charging_speed * hours,
                self.battery_capacity - self.battery_level
            )
            self.battery_level += energy_added
            print(f"Charged {energy_added:.1f}kWh. Battery level: {self.battery_level:.1f}kWh")
            return energy_added
        else:
            print("Not connected to charger.")
            return 0.0
    
    def start_charging(self) -> bool:
        """Start charging process."""
        if self.battery_level < self.battery_capacity:
            self.is_charging = True
            print("Charging started.")
            return True
        else:
            print("Battery is already full.")
            return False
    
    def stop_charging(self) -> bool:
        """Stop charging process."""
        self.is_charging = False
        print("Charging stopped.")
        return True
    
    def get_range(self) -> float:
        """Get estimated range based on battery level."""
        return self.battery_level * 5  # 5km per kWh


class GPS:
    """
    Mixin class for GPS functionality.
    """
    
    def __init__(self):
        """Initialize GPS system."""
        self.current_location = (0.0, 0.0)  # (latitude, longitude)
        self.destination = None
        self.gps_enabled = True
    
    def set_destination(self, latitude: float, longitude: float) -> None:
        """Set GPS destination."""
        self.destination = (latitude, longitude)
        print(f"Destination set to: ({latitude}, {longitude})")
    
    def get_current_location(self) -> tuple:
        """Get current GPS location."""
        if self.gps_enabled:
            return self.current_location
        else:
            print("GPS is disabled.")
            return None
    
    def calculate_distance_to_destination(self) -> Optional[float]:
        """Calculate distance to destination."""
        if self.destination and self.gps_enabled:
            # Simplified distance calculation
            lat_diff = self.destination[0] - self.current_location[0]
            lon_diff = self.destination[1] - self.current_location[1]
            distance = math.sqrt(lat_diff**2 + lon_diff**2) * 111  # Rough km conversion
            return distance
        return None
    
    def navigate(self) -> bool:
        """Start navigation to destination."""
        if self.destination and self.gps_enabled:
            distance = self.calculate_distance_to_destination()
            print(f"Navigation started. Distance to destination: {distance:.1f}km")
            return True
        else:
            print("Set destination first or enable GPS.")
            return False


# Multiple inheritance example
class ElectricCar(Car, Electric, GPS):
    """
    Electric Car class demonstrating multiple inheritance.
    Inherits from Car, Electric, and GPS classes.
    """
    
    def __init__(self, make: str, model: str, year: int, color: str,
                 num_doors: int, transmission: str, battery_capacity: float,
                 charging_speed: float):
        """
        Initialize electric car with all parent class features.
        """
        # Initialize all parent classes
        Car.__init__(self, make, model, year, color, num_doors, transmission)
        Electric.__init__(self, battery_capacity, charging_speed)
        GPS.__init__(self)
        
        # Electric car specific attributes
        self.regenerative_braking = True
        self.eco_mode = False
    
    def enable_eco_mode(self) -> None:
        """Enable eco mode for better efficiency."""
        self.eco_mode = True
        print("Eco mode enabled.")
    
    def disable_eco_mode(self) -> None:
        """Disable eco mode."""
        self.eco_mode = False
        print("Eco mode disabled.")
    
    def regenerative_brake(self, energy_recovered: float) -> None:
        """Recover energy through regenerative braking."""
        if self.regenerative_braking:
            self.battery_level = min(
                self.battery_level + energy_recovered,
                self.battery_capacity
            )
            print(f"Recovered {energy_recovered:.1f}kWh through regenerative braking.")
    
    # Override drive method to use electric power
    def drive(self, distance: float) -> bool:
        """Drive electric car using battery power."""
        if not self.is_running:
            print("Cannot drive. Car is not started.")
            return False
        
        if distance <= 0:
            print("Distance must be positive.")
            return False
        
        energy_needed = distance * 0.2  # 0.2kWh per km
        if self.eco_mode:
            energy_needed *= 0.8  # 20% more efficient in eco mode
        
        if self.battery_level < energy_needed:
            print("Insufficient battery charge for the journey.")
            return False
        
        self.mileage += distance
        self.battery_level -= energy_needed
        
        # Simulate regenerative braking (recover 10% of energy)
        if self.regenerative_braking:
            self.regenerative_brake(energy_needed * 0.1)
        
        print(f"Drove {distance}km electrically. Battery level: {self.battery_level:.1f}kWh")
        return True
    
    def get_vehicle_info(self) -> Dict[str, Any]:
        """Get complete electric car information."""
        info = Car.get_vehicle_info(self)  # Get car info
        info.update({
            'battery_capacity': self.battery_capacity,
            'battery_level': self.battery_level,
            'charging_speed': self.charging_speed,
            'is_charging': self.is_charging,
            'estimated_range': self.get_range(),
            'eco_mode': self.eco_mode,
            'regenerative_braking': self.regenerative_braking,
            'gps_enabled': self.gps_enabled,
            'current_location': self.current_location
        })
        return info


# Abstract base class demonstration
class Shape(ABC):
    """
    Abstract base class for geometric shapes.
    Demonstrates abstract methods and inheritance contracts.
    """
    
    def __init__(self, name: str):
        """Initialize shape with name."""
        self.name = name
        self.created_at = datetime.now()
    
    @abstractmethod
    def calculate_area(self) -> float:
        """Calculate area of the shape (must be implemented by subclasses)."""
        pass
    
    @abstractmethod
    def calculate_perimeter(self) -> float:
        """Calculate perimeter of the shape (must be implemented by subclasses)."""
        pass
    
    def get_shape_info(self) -> Dict[str, Any]:
        """Get basic shape information."""
        return {
            'name': self.name,
            'area': self.calculate_area(),
            'perimeter': self.calculate_perimeter(),
            'created_at': self.created_at.isoformat()
        }
    
    def __str__(self) -> str:
        return f"{self.name} (Area: {self.calculate_area():.2f}, Perimeter: {self.calculate_perimeter():.2f})"


class Rectangle(Shape):
    """Rectangle class inheriting from abstract Shape."""
    
    def __init__(self, width: float, height: float):
        """Initialize rectangle with dimensions."""
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
    """Circle class inheriting from abstract Shape."""
    
    def __init__(self, radius: float):
        """Initialize circle with radius."""
        super().__init__("Circle")
        self.radius = radius
    
    def calculate_area(self) -> float:
        """Calculate circle area."""
        return math.pi * self.radius ** 2
    
    def calculate_perimeter(self) -> float:
        """Calculate circle perimeter (circumference)."""
        return 2 * math.pi * self.radius


def demonstrate_inheritance():
    """
    Demonstrate inheritance concepts with practical examples.
    """
    print("=== INHERITANCE FUNDAMENTALS DEMONSTRATION ===\n")
    
    # 1. Single Inheritance
    print("1. Single Inheritance - Car inherits from Vehicle:")
    car = Car("Toyota", "Camry", 2023, "Blue", 4, "Automatic")
    print(f"Created: {car}")
    
    car.add_fuel(50)
    car.start_engine()
    car.turn_on_ac()
    car.drive(100)
    
    print(f"Car info: {car.get_vehicle_info()}")
    print()
    
    # 2. Another Single Inheritance Example
    print("2. Motorcycle inherits from Vehicle:")
    motorcycle = Motorcycle("Harley-Davidson", "Street 750", 2023, "Black", 750)
    print(f"Created: {motorcycle}")
    
    motorcycle.add_fuel(15)
    motorcycle.start_engine()
    motorcycle.add_helmet()
    motorcycle.add_helmet()
    motorcycle.wheelie()
    
    print(f"Motorcycle info: {motorcycle.get_vehicle_info()}")
    print()
    
    # 3. Multiple Inheritance
    print("3. Multiple Inheritance - ElectricCar:")
    electric_car = ElectricCar("Tesla", "Model 3", 2023, "White", 4, "Automatic", 75.0, 11.0)
    print(f"Created: {electric_car}")
    
    # Use Car methods
    electric_car.start_engine()
    electric_car.turn_on_ac()
    
    # Use Electric methods
    electric_car.start_charging()
    electric_car.charge_battery(2.0)  # Charge for 2 hours
    electric_car.stop_charging()
    
    # Use GPS methods
    electric_car.set_destination(40.7128, -74.0060)  # New York coordinates
    electric_car.navigate()
    
    # Use electric car specific methods
    electric_car.enable_eco_mode()
    electric_car.drive(50)
    
    print(f"Electric car info: {electric_car.get_vehicle_info()}")
    print()
    
    # 4. Method Resolution Order (MRO)
    print("4. Method Resolution Order:")
    print(f"ElectricCar MRO: {[cls.__name__ for cls in ElectricCar.__mro__]}")
    print()
    
    # 5. Abstract Base Classes
    print("5. Abstract Base Classes:")
    
    # Create concrete shapes
    rectangle = Rectangle(5.0, 3.0)
    circle = Circle(4.0)
    
    shapes = [rectangle, circle]
    
    for shape in shapes:
        print(f"{shape}")
        print(f"  Shape info: {shape.get_shape_info()}")
    
    # Try to create abstract shape (will fail)
    try:
        # This will raise TypeError
        abstract_shape = Shape("Abstract")
    except TypeError as e:
        print(f"Cannot instantiate abstract class: {e}")
    print()
    
    # 6. Super() Method Usage
    print("6. Super() Method Usage:")
    print("Car.start_engine() calls Vehicle.start_engine() using super()")
    car2 = Car("Honda", "Civic", 2023, "Red", 4, "Manual")
    car2.start_engine()  # Shows both parent and child behavior
    print()
    
    # 7. Inheritance vs Composition
    print("7. Inheritance Relationships:")
    print(f"isinstance(car, Vehicle): {isinstance(car, Vehicle)}")
    print(f"isinstance(car, Car): {isinstance(car, Car)}")
    print(f"isinstance(electric_car, Car): {isinstance(electric_car, Car)}")
    print(f"isinstance(electric_car, Electric): {isinstance(electric_car, Electric)}")
    print(f"isinstance(electric_car, GPS): {isinstance(electric_car, GPS)}")
    print()
    
    print("=== INHERITANCE DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_inheritance()
