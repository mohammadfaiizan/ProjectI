"""
Python Design Patterns: Common Design Patterns Implemented in Pythonic Ways
Implementation-focused with minimal comments, maximum functionality coverage
"""

import weakref
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod
from functools import wraps
import copy

# Singleton Pattern - Multiple Implementations
class SingletonMeta(type):
    """Metaclass-based singleton"""
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class MetaSingleton(metaclass=SingletonMeta):
    def __init__(self, name="default"):
        if not hasattr(self, 'initialized'):
            self.name = name
            self.data = {}
            self.initialized = True
    
    def set_data(self, key, value):
        self.data[key] = value
    
    def get_data(self, key):
        return self.data.get(key)

class DecoratorSingleton:
    """Decorator-based singleton"""
    _instances = {}
    _lock = threading.Lock()
    
    def __init__(self, cls):
        self.cls = cls
    
    def __call__(self, *args, **kwargs):
        if self.cls not in self._instances:
            with self._lock:
                if self.cls not in self._instances:
                    self._instances[self.cls] = self.cls(*args, **kwargs)
        return self._instances[self.cls]

@DecoratorSingleton
class ConfigManager:
    def __init__(self):
        self.config = {"debug": False, "version": "1.0"}
    
    def set_config(self, key, value):
        self.config[key] = value
    
    def get_config(self, key):
        return self.config.get(key)

def singleton_demo():
    # Test metaclass singleton
    meta1 = MetaSingleton("instance1")
    meta2 = MetaSingleton("instance2")  # Should be same instance
    
    meta1.set_data("test", "value1")
    
    # Test decorator singleton
    config1 = ConfigManager()
    config2 = ConfigManager()  # Should be same instance
    
    config1.set_config("new_key", "new_value")
    
    return {
        "metaclass_singleton": {
            "same_instance": meta1 is meta2,
            "name": meta2.name,  # Should be "instance1"
            "shared_data": meta2.get_data("test")
        },
        "decorator_singleton": {
            "same_instance": config1 is config2,
            "shared_config": config2.get_config("new_key")
        }
    }

# Factory Pattern
class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass
    
    @abstractmethod
    def get_info(self):
        pass

class Dog(Animal):
    def __init__(self, breed="Mixed"):
        self.breed = breed
    
    def make_sound(self):
        return "Woof!"
    
    def get_info(self):
        return f"Dog ({self.breed})"

class Cat(Animal):
    def __init__(self, color="Unknown"):
        self.color = color
    
    def make_sound(self):
        return "Meow!"
    
    def get_info(self):
        return f"Cat ({self.color})"

class Bird(Animal):
    def __init__(self, species="Generic"):
        self.species = species
    
    def make_sound(self):
        return "Tweet!"
    
    def get_info(self):
        return f"Bird ({self.species})"

class AnimalFactory:
    """Simple Factory Pattern"""
    
    _animals = {
        "dog": Dog,
        "cat": Cat,
        "bird": Bird
    }
    
    @classmethod
    def create_animal(cls, animal_type, **kwargs):
        animal_class = cls._animals.get(animal_type.lower())
        if animal_class:
            return animal_class(**kwargs)
        raise ValueError(f"Unknown animal type: {animal_type}")
    
    @classmethod
    def register_animal(cls, animal_type, animal_class):
        """Register new animal type"""
        cls._animals[animal_type.lower()] = animal_class
    
    @classmethod
    def get_available_types(cls):
        return list(cls._animals.keys())

# Abstract Factory Pattern
class GUIFactory(ABC):
    @abstractmethod
    def create_button(self):
        pass
    
    @abstractmethod
    def create_window(self):
        pass

class WindowsFactory(GUIFactory):
    def create_button(self):
        return WindowsButton()
    
    def create_window(self):
        return WindowsWindow()

class MacFactory(GUIFactory):
    def create_button(self):
        return MacButton()
    
    def create_window(self):
        return MacWindow()

class WindowsButton:
    def click(self):
        return "Windows button clicked"
    
    def style(self):
        return "Windows button style"

class MacButton:
    def click(self):
        return "Mac button clicked"
    
    def style(self):
        return "Mac button style"

class WindowsWindow:
    def open(self):
        return "Windows window opened"
    
    def close(self):
        return "Windows window closed"

class MacWindow:
    def open(self):
        return "Mac window opened"
    
    def close(self):
        return "Mac window closed"

def factory_pattern_demo():
    # Simple Factory
    animals = []
    animal_types = ["dog", "cat", "bird"]
    
    for animal_type in animal_types:
        if animal_type == "dog":
            animal = AnimalFactory.create_animal(animal_type, breed="Labrador")
        elif animal_type == "cat":
            animal = AnimalFactory.create_animal(animal_type, color="Orange")
        else:
            animal = AnimalFactory.create_animal(animal_type, species="Parrot")
        
        animals.append({
            "type": animal_type,
            "info": animal.get_info(),
            "sound": animal.make_sound()
        })
    
    # Abstract Factory
    factories = {"windows": WindowsFactory(), "mac": MacFactory()}
    gui_results = {}
    
    for platform, factory in factories.items():
        button = factory.create_button()
        window = factory.create_window()
        
        gui_results[platform] = {
            "button_click": button.click(),
            "button_style": button.style(),
            "window_open": window.open(),
            "window_close": window.close()
        }
    
    return {
        "simple_factory": {
            "animals": animals,
            "available_types": AnimalFactory.get_available_types()
        },
        "abstract_factory": gui_results
    }

# Observer Pattern
class Observable:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer):
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, *args, **kwargs):
        for observer in self._observers:
            observer.update(self, *args, **kwargs)

class WeatherStation(Observable):
    def __init__(self):
        super().__init__()
        self._temperature = 20
        self._humidity = 50
        self._pressure = 1013
    
    @property
    def temperature(self):
        return self._temperature
    
    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        self.notify("temperature", value)
    
    @property
    def humidity(self):
        return self._humidity
    
    @humidity.setter
    def humidity(self, value):
        self._humidity = value
        self.notify("humidity", value)
    
    def get_measurements(self):
        return {
            "temperature": self._temperature,
            "humidity": self._humidity,
            "pressure": self._pressure
        }
    
    def set_measurements(self, temp, humidity, pressure):
        self._temperature = temp
        self._humidity = humidity
        self._pressure = pressure
        self.notify("all_measurements", self.get_measurements())

class Display:
    def __init__(self, name):
        self.name = name
        self.updates = []
    
    def update(self, observable, event_type, data):
        update_info = {
            "display": self.name,
            "event": event_type,
            "data": data,
            "timestamp": time.time()
        }
        self.updates.append(update_info)

class PhoneApp(Display):
    def update(self, observable, event_type, data):
        super().update(observable, event_type, data)
        # Additional phone-specific logic

def observer_pattern_demo():
    # Create weather station and displays
    weather = WeatherStation()
    
    phone_display = PhoneApp("Phone")
    desktop_display = Display("Desktop")
    tablet_display = Display("Tablet")
    
    # Attach observers
    weather.attach(phone_display)
    weather.attach(desktop_display)
    weather.attach(tablet_display)
    
    # Make changes
    weather.temperature = 25
    weather.humidity = 60
    weather.set_measurements(30, 70, 1020)
    
    # Detach one observer
    weather.detach(tablet_display)
    weather.temperature = 35
    
    observer_results = {
        "phone_updates": len(phone_display.updates),
        "desktop_updates": len(desktop_display.updates),
        "tablet_updates": len(tablet_display.updates),  # Should be less than others
        "latest_phone_update": phone_display.updates[-1] if phone_display.updates else None,
        "latest_desktop_update": desktop_display.updates[-1] if desktop_display.updates else None
    }
    
    return observer_results

# Strategy Pattern
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass

class BubbleSort(SortStrategy):
    def sort(self, data):
        data = data.copy()
        n = len(data)
        for i in range(n):
            for j in range(0, n - i - 1):
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
        return data

class QuickSort(SortStrategy):
    def sort(self, data):
        if len(data) <= 1:
            return data
        
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        
        return self.sort(left) + middle + self.sort(right)

class MergeSort(SortStrategy):
    def sort(self, data):
        if len(data) <= 1:
            return data
        
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        
        return self._merge(left, right)
    
    def _merge(self, left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result

class Sorter:
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: SortStrategy):
        self._strategy = strategy
    
    def sort(self, data):
        return self._strategy.sort(data)

def strategy_pattern_demo():
    data = [64, 34, 25, 12, 22, 11, 90]
    
    # Test different strategies
    strategies = {
        "bubble": BubbleSort(),
        "quick": QuickSort(),
        "merge": MergeSort()
    }
    
    strategy_results = {}
    sorter = Sorter(BubbleSort())  # Initial strategy
    
    for name, strategy in strategies.items():
        sorter.set_strategy(strategy)
        sorted_data = sorter.sort(data)
        strategy_results[name] = {
            "sorted": sorted_data,
            "correct": sorted_data == sorted(data)
        }
    
    return {
        "original_data": data,
        "strategy_results": strategy_results
    }

# Command Pattern
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass

class Light:
    def __init__(self, location):
        self.location = location
        self.is_on = False
    
    def turn_on(self):
        self.is_on = True
        return f"{self.location} light is ON"
    
    def turn_off(self):
        self.is_on = False
        return f"{self.location} light is OFF"

class LightOnCommand(Command):
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        return self.light.turn_on()
    
    def undo(self):
        return self.light.turn_off()

class LightOffCommand(Command):
    def __init__(self, light):
        self.light = light
    
    def execute(self):
        return self.light.turn_off()
    
    def undo(self):
        return self.light.turn_on()

class Fan:
    def __init__(self, location):
        self.location = location
        self.speed = 0  # 0=off, 1=low, 2=medium, 3=high
    
    def set_speed(self, speed):
        old_speed = self.speed
        self.speed = max(0, min(3, speed))
        return f"{self.location} fan speed: {self.speed}", old_speed

class FanCommand(Command):
    def __init__(self, fan, speed):
        self.fan = fan
        self.speed = speed
        self.previous_speed = 0
    
    def execute(self):
        result, self.previous_speed = self.fan.set_speed(self.speed)
        return result
    
    def undo(self):
        result, _ = self.fan.set_speed(self.previous_speed)
        return result

class RemoteControl:
    def __init__(self):
        self.commands = {}
        self.last_command = None
    
    def set_command(self, slot, command):
        self.commands[slot] = command
    
    def press_button(self, slot):
        if slot in self.commands:
            self.last_command = self.commands[slot]
            return self.last_command.execute()
        return "No command assigned"
    
    def press_undo(self):
        if self.last_command:
            return self.last_command.undo()
        return "No command to undo"

def command_pattern_demo():
    # Create devices
    living_room_light = Light("Living Room")
    bedroom_light = Light("Bedroom")
    ceiling_fan = Fan("Living Room")
    
    # Create commands
    living_light_on = LightOnCommand(living_room_light)
    living_light_off = LightOffCommand(living_room_light)
    bedroom_light_on = LightOnCommand(bedroom_light)
    fan_high = FanCommand(ceiling_fan, 3)
    fan_medium = FanCommand(ceiling_fan, 2)
    fan_off = FanCommand(ceiling_fan, 0)
    
    # Set up remote
    remote = RemoteControl()
    remote.set_command(1, living_light_on)
    remote.set_command(2, living_light_off)
    remote.set_command(3, bedroom_light_on)
    remote.set_command(4, fan_high)
    remote.set_command(5, fan_medium)
    remote.set_command(6, fan_off)
    
    # Execute commands
    results = []
    
    results.append(remote.press_button(1))  # Living room light on
    results.append(remote.press_button(4))  # Fan high
    results.append(remote.press_undo())     # Undo fan (back to 0)
    results.append(remote.press_button(5))  # Fan medium
    results.append(remote.press_button(3))  # Bedroom light on
    results.append(remote.press_undo())     # Undo bedroom light
    
    device_states = {
        "living_room_light": living_room_light.is_on,
        "bedroom_light": bedroom_light.is_on,
        "fan_speed": ceiling_fan.speed
    }
    
    return {
        "command_results": results,
        "device_states": device_states
    }

# Decorator Pattern (not Python decorators)
class Coffee(ABC):
    @abstractmethod
    def cost(self):
        pass
    
    @abstractmethod
    def description(self):
        pass

class SimpleCoffee(Coffee):
    def cost(self):
        return 2.0
    
    def description(self):
        return "Simple coffee"

class CoffeeDecorator(Coffee):
    def __init__(self, coffee):
        self._coffee = coffee
    
    def cost(self):
        return self._coffee.cost()
    
    def description(self):
        return self._coffee.description()

class MilkDecorator(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 0.5
    
    def description(self):
        return self._coffee.description() + ", milk"

class SugarDecorator(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 0.2
    
    def description(self):
        return self._coffee.description() + ", sugar"

class WhipDecorator(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 0.7
    
    def description(self):
        return self._coffee.description() + ", whipped cream"

def decorator_pattern_demo():
    # Create base coffee
    coffee = SimpleCoffee()
    
    coffee_options = [
        ("base", coffee.description(), coffee.cost()),
    ]
    
    # Add milk
    coffee = MilkDecorator(coffee)
    coffee_options.append(("with milk", coffee.description(), coffee.cost()))
    
    # Add sugar
    coffee = SugarDecorator(coffee)
    coffee_options.append(("with sugar", coffee.description(), coffee.cost()))
    
    # Add whip
    coffee = WhipDecorator(coffee)
    coffee_options.append(("with whip", coffee.description(), coffee.cost()))
    
    # Create another combination
    coffee2 = WhipDecorator(SugarDecorator(SimpleCoffee()))
    
    return {
        "coffee_progression": coffee_options,
        "final_coffee": {
            "description": coffee.description(),
            "cost": coffee.cost()
        },
        "alternative_coffee": {
            "description": coffee2.description(),
            "cost": coffee2.cost()
        }
    }

# Builder Pattern
class Pizza:
    def __init__(self):
        self.size = ""
        self.crust = ""
        self.toppings = []
        self.sauce = ""
        self.cheese = ""
    
    def __str__(self):
        return (f"{self.size} {self.crust} pizza with {self.sauce} sauce, "
                f"{self.cheese} cheese, and toppings: {', '.join(self.toppings)}")

class PizzaBuilder:
    def __init__(self):
        self.pizza = Pizza()
    
    def set_size(self, size):
        self.pizza.size = size
        return self
    
    def set_crust(self, crust):
        self.pizza.crust = crust
        return self
    
    def set_sauce(self, sauce):
        self.pizza.sauce = sauce
        return self
    
    def set_cheese(self, cheese):
        self.pizza.cheese = cheese
        return self
    
    def add_topping(self, topping):
        self.pizza.toppings.append(topping)
        return self
    
    def add_toppings(self, toppings):
        self.pizza.toppings.extend(toppings)
        return self
    
    def build(self):
        return self.pizza

class PizzaDirector:
    @staticmethod
    def make_margherita(builder):
        return (builder
                .set_size("Medium")
                .set_crust("Thin")
                .set_sauce("Tomato")
                .set_cheese("Mozzarella")
                .add_topping("Basil")
                .build())
    
    @staticmethod
    def make_pepperoni(builder):
        return (builder
                .set_size("Large")
                .set_crust("Thick")
                .set_sauce("Tomato")
                .set_cheese("Mozzarella")
                .add_toppings(["Pepperoni", "Oregano"])
                .build())
    
    @staticmethod
    def make_veggie(builder):
        return (builder
                .set_size("Medium")
                .set_crust("Whole Wheat")
                .set_sauce("Pesto")
                .set_cheese("Goat Cheese")
                .add_toppings(["Bell Peppers", "Mushrooms", "Olives", "Onions"])
                .build())

def builder_pattern_demo():
    # Create predefined pizzas
    director = PizzaDirector()
    
    margherita = director.make_margherita(PizzaBuilder())
    pepperoni = director.make_pepperoni(PizzaBuilder())
    veggie = director.make_veggie(PizzaBuilder())
    
    # Create custom pizza
    custom = (PizzaBuilder()
              .set_size("Large")
              .set_crust("Stuffed")
              .set_sauce("BBQ")
              .set_cheese("Cheddar")
              .add_topping("Chicken")
              .add_topping("Bacon")
              .add_topping("Red Onions")
              .build())
    
    return {
        "predefined_pizzas": {
            "margherita": str(margherita),
            "pepperoni": str(pepperoni),
            "veggie": str(veggie)
        },
        "custom_pizza": str(custom)
    }

# Adapter Pattern
class EuropeanSocket:
    def voltage(self):
        return 230
    
    def live(self):
        return 1
    
    def neutral(self):
        return -1
    
    def earth(self):
        return 0

class USASocket:
    def voltage(self):
        return 120
    
    def live(self):
        return 1
    
    def neutral(self):
        return -1

class SocketAdapter:
    def __init__(self, socket):
        self.socket = socket
    
    def voltage(self):
        # Convert voltage if needed
        if hasattr(self.socket, 'earth'):
            # European socket - convert to US voltage
            return self.socket.voltage() * 120 / 230
        else:
            # US socket - convert to European voltage
            return self.socket.voltage() * 230 / 120
    
    def live(self):
        return self.socket.live()
    
    def neutral(self):
        return self.socket.neutral()

class ElectricDevice:
    def __init__(self, required_voltage=120):
        self.required_voltage = required_voltage
    
    def plug_in(self, socket):
        voltage = socket.voltage()
        if abs(voltage - self.required_voltage) <= 10:  # 10V tolerance
            return f"Device working at {voltage}V"
        else:
            return f"Voltage mismatch: need {self.required_voltage}V, got {voltage}V"

def adapter_pattern_demo():
    # Create sockets
    eu_socket = EuropeanSocket()
    us_socket = USASocket()
    
    # Create devices
    us_device = ElectricDevice(120)  # US device
    eu_device = ElectricDevice(230)  # European device
    
    # Direct connections
    direct_results = {
        "us_device_us_socket": us_device.plug_in(us_socket),
        "eu_device_eu_socket": eu_device.plug_in(eu_socket),
        "us_device_eu_socket": us_device.plug_in(eu_socket),  # Should fail
        "eu_device_us_socket": eu_device.plug_in(us_socket)   # Should fail
    }
    
    # Adapted connections
    us_to_eu_adapter = SocketAdapter(us_socket)
    eu_to_us_adapter = SocketAdapter(eu_socket)
    
    adapted_results = {
        "us_device_adapted_eu": us_device.plug_in(eu_to_us_adapter),
        "eu_device_adapted_us": eu_device.plug_in(us_to_eu_adapter),
        "adapter_voltages": {
            "us_to_eu": us_to_eu_adapter.voltage(),
            "eu_to_us": eu_to_us_adapter.voltage()
        }
    }
    
    return {
        "direct_connections": direct_results,
        "adapted_connections": adapted_results
    }

# Template Method Pattern
class DataProcessor(ABC):
    def process(self, data):
        """Template method defining the algorithm structure"""
        validated_data = self.validate_data(data)
        processed_data = self.transform_data(validated_data)
        result = self.save_data(processed_data)
        self.log_result(result)
        return result
    
    @abstractmethod
    def validate_data(self, data):
        pass
    
    @abstractmethod
    def transform_data(self, data):
        pass
    
    @abstractmethod
    def save_data(self, data):
        pass
    
    def log_result(self, result):
        """Hook method - can be overridden but has default implementation"""
        print(f"Processing completed: {len(result)} items")

class CSVProcessor(DataProcessor):
    def validate_data(self, data):
        # Simulate CSV validation
        if not isinstance(data, list):
            raise ValueError("CSV data must be a list")
        return data
    
    def transform_data(self, data):
        # Convert to dictionary format
        if not data:
            return []
        
        headers = data[0]
        return [dict(zip(headers, row)) for row in data[1:]]
    
    def save_data(self, data):
        # Simulate saving to database
        return {"saved_to": "csv_database", "records": len(data), "data": data}

class JSONProcessor(DataProcessor):
    def validate_data(self, data):
        if not isinstance(data, dict):
            raise ValueError("JSON data must be a dictionary")
        return data
    
    def transform_data(self, data):
        # Flatten nested dictionaries
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    result[f"{key}_{nested_key}"] = nested_value
            else:
                result[key] = value
        return result
    
    def save_data(self, data):
        return {"saved_to": "json_database", "fields": len(data), "data": data}
    
    def log_result(self, result):
        # Override hook method
        print(f"JSON processing completed: {result['fields']} fields saved")

def template_method_demo():
    # Test CSV processor
    csv_data = [
        ["name", "age", "city"],
        ["Alice", "30", "New York"],
        ["Bob", "25", "Los Angeles"]
    ]
    
    csv_processor = CSVProcessor()
    csv_result = csv_processor.process(csv_data)
    
    # Test JSON processor
    json_data = {
        "user": {"name": "Charlie", "age": 35},
        "location": {"city": "Chicago", "state": "IL"},
        "active": True
    }
    
    json_processor = JSONProcessor()
    json_result = json_processor.process(json_data)
    
    return {
        "csv_processing": csv_result,
        "json_processing": json_result
    }

# Comprehensive testing
def run_all_design_pattern_demos():
    """Execute all design pattern demonstrations"""
    demo_functions = [
        ('singleton', singleton_demo),
        ('factory', factory_pattern_demo),
        ('observer', observer_pattern_demo),
        ('strategy', strategy_pattern_demo),
        ('command', command_pattern_demo),
        ('decorator', decorator_pattern_demo),
        ('builder', builder_pattern_demo),
        ('adapter', adapter_pattern_demo),
        ('template_method', template_method_demo)
    ]
    
    results = {}
    for name, func in demo_functions:
        try:
            result = func()
            results[name] = result
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    print("=== Python Design Patterns Demo ===")
    
    # Run all demonstrations
    all_results = run_all_design_pattern_demos()
    
    for category, data in all_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        
        if 'error' in data:
            print(f"  Error: {data['error']}")
            continue
        
        # Display results
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and len(value) > 3:
                    print(f"  {key}: {dict(list(value.items())[:3])}... (truncated)")
                elif isinstance(value, list) and len(value) > 5:
                    print(f"  {key}: {value[:5]}... (showing first 5)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {data}")
    
    print("\n=== DESIGN PATTERNS CATEGORIES ===")
    
    categories = {
        "Creational Patterns": [
            "Singleton - Ensure single instance",
            "Factory - Create objects without specifying exact class",
            "Abstract Factory - Create families of related objects",
            "Builder - Construct complex objects step by step"
        ],
        "Structural Patterns": [
            "Adapter - Make incompatible interfaces work together",
            "Decorator - Add behavior to objects dynamically",
            "Facade - Provide unified interface to subsystem",
            "Composite - Treat individual and composite objects uniformly"
        ],
        "Behavioral Patterns": [
            "Observer - Define subscription mechanism for notifications",
            "Strategy - Define family of algorithms and make them interchangeable",
            "Command - Encapsulate requests as objects",
            "Template Method - Define skeleton of algorithm in base class"
        ]
    }
    
    for category, patterns in categories.items():
        print(f"  {category}:")
        for pattern in patterns:
            print(f"    • {pattern}")
    
    print("\n=== PYTHONIC IMPLEMENTATIONS ===")
    
    pythonic_notes = {
        "Singleton": "Use modules, metaclasses, or decorators instead of classic GoF implementation",
        "Factory": "Use functions, classes, or registries - very natural in Python",
        "Observer": "Consider using built-in weakref for automatic cleanup",
        "Strategy": "Functions are first-class objects - can often just pass functions",
        "Command": "Functions and closures often simpler than full command objects",
        "Decorator": "Python has built-in decorator syntax for different purpose",
        "Builder": "Method chaining with fluent interface works well",
        "Adapter": "Duck typing reduces need, but still useful for incompatible interfaces"
    }
    
    for pattern, note in pythonic_notes.items():
        print(f"  {pattern}: {note}")
    
    print("\n=== WHEN TO USE PATTERNS ===")
    
    usage_guidelines = [
        "Don't force patterns - use when they solve actual problems",
        "Consider simpler Python alternatives before complex patterns",
        "Patterns should make code more maintainable, not more complex",
        "Some patterns are less relevant in Python due to language features",
        "Focus on the problem the pattern solves, not the pattern itself",
        "Combine patterns when appropriate - they often work together",
        "Document why you chose a particular pattern",
        "Refactor to patterns when complexity emerges, don't start with them"
    ]
    
    for guideline in usage_guidelines:
        print(f"  • {guideline}")
    
    print("\n=== Design Patterns Complete! ===")
    print("  Common design patterns implemented in Pythonic ways")
