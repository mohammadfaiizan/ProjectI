"""
COMPOSITION VS INHERITANCE - Object Relationship Patterns
=========================================================

Problem Statement:
Demonstrate the differences between composition and inheritance:
- When to use inheritance vs composition
- "Has-a" vs "Is-a" relationships
- Favor composition over inheritance principle
- Mixing composition and inheritance effectively
- Avoiding inheritance pitfalls

Learning Objectives:
- Understand composition and inheritance trade-offs
- Design flexible object relationships
- Apply "favor composition over inheritance" principle
- Create maintainable class hierarchies
- Use delegation and aggregation patterns
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from datetime import datetime
from enum import Enum


# Example 1: Inheritance-based approach (traditional)
class InheritanceVehicle:
    """Base vehicle class for inheritance example."""
    
    def __init__(self, make: str, model: str, year: int):
        self.make = make
        self.model = model
        self.year = year
        self.is_running = False
    
    def start(self) -> str:
        self.is_running = True
        return f"{self.make} {self.model} started"
    
    def stop(self) -> str:
        self.is_running = False
        return f"{self.make} {self.model} stopped"


class InheritanceCar(InheritanceVehicle):
    """Car using inheritance - becomes rigid."""
    
    def __init__(self, make: str, model: str, year: int, doors: int):
        super().__init__(make, model, year)
        self.doors = doors
        self.trunk_open = False
    
    def open_trunk(self) -> str:
        self.trunk_open = True
        return "Trunk opened"
    
    def honk(self) -> str:
        return "Beep beep!"


class InheritanceMotorcycle(InheritanceVehicle):
    """Motorcycle using inheritance."""
    
    def __init__(self, make: str, model: str, year: int, engine_size: int):
        super().__init__(make, model, year)
        self.engine_size = engine_size
    
    def wheelie(self) -> str:
        if self.is_running:
            return "Performing wheelie!"
        return "Cannot wheelie - engine not running"


# Problem with inheritance: What if we want a flying car?
# We'd need multiple inheritance or complex hierarchies
class InheritanceFlyingCar(InheritanceCar):
    """Flying car - inheritance becomes complex."""
    
    def __init__(self, make: str, model: str, year: int, doors: int, max_altitude: int):
        super().__init__(make, model, year, doors)
        self.max_altitude = max_altitude
        self.is_flying = False
    
    def take_off(self) -> str:
        if self.is_running:
            self.is_flying = True
            return "Taking off!"
        return "Cannot take off - engine not running"
    
    def land(self) -> str:
        self.is_flying = False
        return "Landing"


# Example 2: Composition-based approach (flexible)
class Engine:
    """Engine component for composition."""
    
    def __init__(self, horsepower: int, fuel_type: str):
        self.horsepower = horsepower
        self.fuel_type = fuel_type
        self.is_running = False
        self.temperature = 20  # Celsius
    
    def start(self) -> str:
        if not self.is_running:
            self.is_running = True
            self.temperature = 90
            return f"{self.horsepower}HP {self.fuel_type} engine started"
        return "Engine already running"
    
    def stop(self) -> str:
        if self.is_running:
            self.is_running = False
            self.temperature = 20
            return "Engine stopped"
        return "Engine already stopped"
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'running': self.is_running,
            'temperature': self.temperature,
            'horsepower': self.horsepower,
            'fuel_type': self.fuel_type
        }


class Transmission:
    """Transmission component."""
    
    def __init__(self, transmission_type: str, gears: int):
        self.type = transmission_type  # "manual", "automatic", "cvt"
        self.gears = gears
        self.current_gear = 0  # 0 = park/neutral
    
    def shift_up(self) -> str:
        if self.current_gear < self.gears:
            self.current_gear += 1
            return f"Shifted to gear {self.current_gear}"
        return "Already in highest gear"
    
    def shift_down(self) -> str:
        if self.current_gear > 0:
            self.current_gear -= 1
            return f"Shifted to gear {self.current_gear}"
        return "Already in lowest gear"
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'current_gear': self.current_gear,
            'max_gears': self.gears
        }


class GPS:
    """GPS navigation component."""
    
    def __init__(self):
        self.current_location = (0.0, 0.0)
        self.destination = None
        self.is_navigating = False
    
    def set_destination(self, latitude: float, longitude: float) -> str:
        self.destination = (latitude, longitude)
        return f"Destination set to ({latitude}, {longitude})"
    
    def start_navigation(self) -> str:
        if self.destination:
            self.is_navigating = True
            return "Navigation started"
        return "Please set destination first"
    
    def stop_navigation(self) -> str:
        self.is_navigating = False
        return "Navigation stopped"
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'current_location': self.current_location,
            'destination': self.destination,
            'navigating': self.is_navigating
        }


class AirConditioner:
    """Air conditioning component."""
    
    def __init__(self, max_cooling_power: int):
        self.max_power = max_cooling_power
        self.is_on = False
        self.temperature_setting = 22  # Celsius
        self.current_power = 0
    
    def turn_on(self) -> str:
        self.is_on = True
        self.current_power = self.max_power // 2
        return f"AC turned on at {self.temperature_setting}°C"
    
    def turn_off(self) -> str:
        self.is_on = False
        self.current_power = 0
        return "AC turned off"
    
    def set_temperature(self, temperature: int) -> str:
        if 16 <= temperature <= 30:
            self.temperature_setting = temperature
            return f"Temperature set to {temperature}°C"
        return "Temperature must be between 16-30°C"
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'is_on': self.is_on,
            'temperature_setting': self.temperature_setting,
            'current_power': self.current_power,
            'max_power': self.max_power
        }


class FlightSystem:
    """Flight system for flying vehicles."""
    
    def __init__(self, max_altitude: int, max_speed: int):
        self.max_altitude = max_altitude
        self.max_speed = max_speed
        self.current_altitude = 0
        self.is_flying = False
        self.autopilot = False
    
    def take_off(self) -> str:
        if not self.is_flying:
            self.is_flying = True
            self.current_altitude = 100  # Initial climb
            return "Taking off!"
        return "Already flying"
    
    def land(self) -> str:
        if self.is_flying:
            self.is_flying = False
            self.current_altitude = 0
            self.autopilot = False
            return "Landing complete"
        return "Already on ground"
    
    def climb(self, altitude: int) -> str:
        if not self.is_flying:
            return "Cannot climb - not flying"
        
        target_altitude = min(altitude, self.max_altitude)
        self.current_altitude = target_altitude
        return f"Climbed to {target_altitude} feet"
    
    def enable_autopilot(self) -> str:
        if self.is_flying:
            self.autopilot = True
            return "Autopilot engaged"
        return "Cannot engage autopilot - not flying"
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'flying': self.is_flying,
            'altitude': self.current_altitude,
            'max_altitude': self.max_altitude,
            'autopilot': self.autopilot
        }


# Composition-based vehicle design
class CompositionVehicle:
    """
    Vehicle using composition - much more flexible.
    Uses "has-a" relationships instead of "is-a".
    """
    
    def __init__(self, make: str, model: str, year: int, engine: Engine):
        self.make = make
        self.model = model
        self.year = year
        self.engine = engine  # Composition: Vehicle HAS an Engine
        
        # Optional components (can be None)
        self.transmission: Optional[Transmission] = None
        self.gps: Optional[GPS] = None
        self.air_conditioner: Optional[AirConditioner] = None
        self.flight_system: Optional[FlightSystem] = None
        
        # Vehicle-specific attributes
        self.mileage = 0.0
        self.fuel_level = 50.0
    
    def add_transmission(self, transmission: Transmission) -> str:
        """Add transmission component."""
        self.transmission = transmission
        return f"Added {transmission.type} transmission"
    
    def add_gps(self, gps: GPS) -> str:
        """Add GPS component."""
        self.gps = gps
        return "GPS system installed"
    
    def add_air_conditioner(self, ac: AirConditioner) -> str:
        """Add air conditioning component."""
        self.air_conditioner = ac
        return "Air conditioning system installed"
    
    def add_flight_system(self, flight_system: FlightSystem) -> str:
        """Add flight system component."""
        self.flight_system = flight_system
        return "Flight system installed"
    
    def start(self) -> str:
        """Start the vehicle (delegates to engine)."""
        return self.engine.start()
    
    def stop(self) -> str:
        """Stop the vehicle (delegates to engine)."""
        result = self.engine.stop()
        
        # Stop other systems when engine stops
        if self.air_conditioner and self.air_conditioner.is_on:
            result += " | " + self.air_conditioner.turn_off()
        
        if self.gps and self.gps.is_navigating:
            result += " | " + self.gps.stop_navigation()
        
        return result
    
    def drive(self, distance: float) -> str:
        """Drive the vehicle."""
        if not self.engine.is_running:
            return "Cannot drive - engine not running"
        
        self.mileage += distance
        fuel_consumed = distance * 0.1  # Simple fuel consumption
        self.fuel_level = max(0, self.fuel_level - fuel_consumed)
        
        return f"Drove {distance}km. Total mileage: {self.mileage}km"
    
    def fly(self, altitude: int) -> str:
        """Fly the vehicle (if flight system available)."""
        if not self.flight_system:
            return "This vehicle cannot fly"
        
        if not self.engine.is_running:
            return "Cannot fly - engine not running"
        
        if not self.flight_system.is_flying:
            result = self.flight_system.take_off()
            if altitude > 100:
                result += " | " + self.flight_system.climb(altitude)
            return result
        else:
            return self.flight_system.climb(altitude)
    
    def land(self) -> str:
        """Land the vehicle (if flying)."""
        if not self.flight_system:
            return "This vehicle cannot fly"
        
        return self.flight_system.land()
    
    def get_full_status(self) -> Dict[str, Any]:
        """Get complete vehicle status."""
        status = {
            'vehicle': f"{self.year} {self.make} {self.model}",
            'mileage': self.mileage,
            'fuel_level': self.fuel_level,
            'engine': self.engine.get_status()
        }
        
        if self.transmission:
            status['transmission'] = self.transmission.get_status()
        
        if self.gps:
            status['gps'] = self.gps.get_status()
        
        if self.air_conditioner:
            status['air_conditioner'] = self.air_conditioner.get_status()
        
        if self.flight_system:
            status['flight_system'] = self.flight_system.get_status()
        
        return status
    
    def __str__(self) -> str:
        components = []
        if self.transmission:
            components.append(f"{self.transmission.type} transmission")
        if self.gps:
            components.append("GPS")
        if self.air_conditioner:
            components.append("AC")
        if self.flight_system:
            components.append("Flight System")
        
        component_str = f" with {', '.join(components)}" if components else ""
        return f"{self.year} {self.make} {self.model}{component_str}"


# Factory functions for creating different vehicle types
def create_basic_car(make: str, model: str, year: int) -> CompositionVehicle:
    """Create a basic car with essential components."""
    engine = Engine(150, "gasoline")
    car = CompositionVehicle(make, model, year, engine)
    
    # Add basic car components
    car.add_transmission(Transmission("automatic", 6))
    
    return car


def create_luxury_car(make: str, model: str, year: int) -> CompositionVehicle:
    """Create a luxury car with premium components."""
    engine = Engine(300, "gasoline")
    car = CompositionVehicle(make, model, year, engine)
    
    # Add luxury components
    car.add_transmission(Transmission("automatic", 8))
    car.add_gps(GPS())
    car.add_air_conditioner(AirConditioner(5000))
    
    return car


def create_flying_car(make: str, model: str, year: int) -> CompositionVehicle:
    """Create a flying car - easy with composition!"""
    engine = Engine(400, "hybrid")
    car = CompositionVehicle(make, model, year, engine)
    
    # Add all components including flight system
    car.add_transmission(Transmission("cvt", 1))
    car.add_gps(GPS())
    car.add_air_conditioner(AirConditioner(3000))
    car.add_flight_system(FlightSystem(10000, 200))
    
    return car


def create_motorcycle(make: str, model: str, year: int) -> CompositionVehicle:
    """Create a motorcycle with minimal components."""
    engine = Engine(100, "gasoline")
    motorcycle = CompositionVehicle(make, model, year, engine)
    
    # Motorcycles typically don't have automatic transmission or AC
    motorcycle.add_transmission(Transmission("manual", 6))
    
    return motorcycle


# Aggregation example (weaker form of composition)
class Driver:
    """Driver class for aggregation example."""
    
    def __init__(self, name: str, license_number: str):
        self.name = name
        self.license_number = license_number
        self.driving_experience = 0
    
    def drive_vehicle(self, vehicle: CompositionVehicle, distance: float) -> str:
        """Drive a vehicle (aggregation - driver can exist without vehicle)."""
        result = vehicle.start()
        result += " | " + vehicle.drive(distance)
        return f"{self.name} driving: {result}"
    
    def __str__(self) -> str:
        return f"Driver({self.name}, License: {self.license_number})"


# Mixin pattern (composition alternative to multiple inheritance)
class SecurityMixin:
    """Security features mixin."""
    
    def __init__(self):
        self.alarm_armed = False
        self.doors_locked = False
    
    def arm_alarm(self) -> str:
        self.alarm_armed = True
        return "Security alarm armed"
    
    def disarm_alarm(self) -> str:
        self.alarm_armed = False
        return "Security alarm disarmed"
    
    def lock_doors(self) -> str:
        self.doors_locked = True
        return "Doors locked"
    
    def unlock_doors(self) -> str:
        self.doors_locked = False
        return "Doors unlocked"


class EntertainmentMixin:
    """Entertainment features mixin."""
    
    def __init__(self):
        self.radio_on = False
        self.volume = 5
        self.current_station = "FM 101.5"
    
    def turn_on_radio(self) -> str:
        self.radio_on = True
        return f"Radio on - {self.current_station} at volume {self.volume}"
    
    def turn_off_radio(self) -> str:
        self.radio_on = False
        return "Radio off"
    
    def change_station(self, station: str) -> str:
        self.current_station = station
        return f"Changed to {station}"
    
    def adjust_volume(self, volume: int) -> str:
        if 0 <= volume <= 10:
            self.volume = volume
            return f"Volume set to {volume}"
        return "Volume must be between 0-10"


class PremiumVehicle(CompositionVehicle, SecurityMixin, EntertainmentMixin):
    """
    Premium vehicle combining composition with mixins.
    Shows how to mix composition and inheritance effectively.
    """
    
    def __init__(self, make: str, model: str, year: int, engine: Engine):
        CompositionVehicle.__init__(self, make, model, year, engine)
        SecurityMixin.__init__(self)
        EntertainmentMixin.__init__(self)
    
    def get_premium_status(self) -> Dict[str, Any]:
        """Get status including premium features."""
        status = self.get_full_status()
        status['security'] = {
            'alarm_armed': self.alarm_armed,
            'doors_locked': self.doors_locked
        }
        status['entertainment'] = {
            'radio_on': self.radio_on,
            'volume': self.volume,
            'station': self.current_station
        }
        return status


def demonstrate_composition_vs_inheritance():
    """
    Demonstrate composition vs inheritance with practical examples.
    """
    print("=== COMPOSITION VS INHERITANCE DEMONSTRATION ===\n")
    
    # 1. Inheritance Approach Problems
    print("1. Inheritance Approach - Rigid Structure:")
    
    inheritance_car = InheritanceCar("Toyota", "Camry", 2023, 4)
    inheritance_motorcycle = InheritanceMotorcycle("Harley", "Street 750", 2023, 750)
    inheritance_flying_car = InheritanceFlyingCar("Future Motors", "SkyRider", 2030, 2, 10000)
    
    print(f"Car: {inheritance_car.start()}")
    print(f"Car: {inheritance_car.honk()}")
    
    print(f"Motorcycle: {inheritance_motorcycle.start()}")
    print(f"Motorcycle: {inheritance_motorcycle.wheelie()}")
    
    print(f"Flying Car: {inheritance_flying_car.start()}")
    print(f"Flying Car: {inheritance_flying_car.take_off()}")
    print()
    
    # 2. Composition Approach - Flexible Structure
    print("2. Composition Approach - Flexible Structure:")
    
    # Create different types of vehicles easily
    basic_car = create_basic_car("Honda", "Civic", 2023)
    luxury_car = create_luxury_car("BMW", "X5", 2023)
    flying_car = create_flying_car("AeroMobil", "4.0", 2025)
    motorcycle = create_motorcycle("Yamaha", "R1", 2023)
    
    vehicles = [basic_car, luxury_car, flying_car, motorcycle]
    
    for vehicle in vehicles:
        print(f"Vehicle: {vehicle}")
        print(f"  Start: {vehicle.start()}")
        print(f"  Drive: {vehicle.drive(50)}")
        
        # Try flying (only works for flying car)
        if hasattr(vehicle, 'flight_system') and vehicle.flight_system:
            print(f"  Fly: {vehicle.fly(5000)}")
            print(f"  Land: {vehicle.land()}")
        
        print(f"  Stop: {vehicle.stop()}")
        print()
    
    # 3. Component Reusability
    print("3. Component Reusability:")
    
    # Create components that can be reused
    powerful_engine = Engine(500, "electric")
    sport_transmission = Transmission("manual", 7)
    premium_gps = GPS()
    climate_control = AirConditioner(8000)
    
    # Create custom vehicle with specific components
    custom_vehicle = CompositionVehicle("Custom", "Speedster", 2024, powerful_engine)
    print(f"Custom vehicle: {custom_vehicle.add_transmission(sport_transmission)}")
    print(f"Custom vehicle: {custom_vehicle.add_gps(premium_gps)}")
    print(f"Custom vehicle: {custom_vehicle.add_air_conditioner(climate_control)}")
    
    # Test the custom vehicle
    custom_vehicle.start()
    if custom_vehicle.gps:
        custom_vehicle.gps.set_destination(40.7128, -74.0060)
        custom_vehicle.gps.start_navigation()
    
    if custom_vehicle.air_conditioner:
        custom_vehicle.air_conditioner.turn_on()
        custom_vehicle.air_conditioner.set_temperature(20)
    
    print(f"Custom vehicle status: {custom_vehicle.get_full_status()}")
    print()
    
    # 4. Aggregation Example
    print("4. Aggregation Example (Driver and Vehicle):")
    
    driver1 = Driver("Alice Johnson", "DL123456")
    driver2 = Driver("Bob Smith", "DL789012")
    
    # Drivers can drive different vehicles (aggregation)
    print(f"{driver1.drive_vehicle(luxury_car, 100)}")
    print(f"{driver2.drive_vehicle(motorcycle, 75)}")
    
    # Driver exists independently of vehicle
    print(f"Driver 1: {driver1}")
    print(f"Driver 2: {driver2}")
    print()
    
    # 5. Mixin Pattern (Composition Alternative to Multiple Inheritance)
    print("5. Mixin Pattern - Premium Vehicle:")
    
    premium_engine = Engine(350, "hybrid")
    premium_vehicle = PremiumVehicle("Mercedes", "S-Class", 2024, premium_engine)
    premium_vehicle.add_transmission(Transmission("automatic", 9))
    premium_vehicle.add_gps(GPS())
    premium_vehicle.add_air_conditioner(AirConditioner(6000))
    
    # Use security features
    print(f"Security: {premium_vehicle.arm_alarm()}")
    print(f"Security: {premium_vehicle.lock_doors()}")
    
    # Use entertainment features
    print(f"Entertainment: {premium_vehicle.turn_on_radio()}")
    print(f"Entertainment: {premium_vehicle.change_station('Spotify Premium')}")
    print(f"Entertainment: {premium_vehicle.adjust_volume(8)}")
    
    # Get complete status
    premium_status = premium_vehicle.get_premium_status()
    print(f"Premium vehicle has {len(premium_status)} feature categories")
    print()
    
    # 6. Flexibility Comparison
    print("6. Flexibility Comparison:")
    print("Inheritance Issues:")
    print("  - Rigid class hierarchies")
    print("  - Diamond problem with multiple inheritance")
    print("  - Difficult to add new features")
    print("  - Tight coupling between classes")
    
    print("\nComposition Benefits:")
    print("  - Flexible object construction")
    print("  - Easy to add/remove features")
    print("  - Loose coupling between components")
    print("  - Better testability")
    print("  - Runtime behavior modification")
    print()
    
    print("=== COMPOSITION VS INHERITANCE DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_composition_vs_inheritance()
