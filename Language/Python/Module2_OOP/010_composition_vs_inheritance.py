"""
Python Composition vs Inheritance: Design Principles and Refactoring Patterns
Implementation-focused with minimal comments, maximum functionality coverage
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
import time
from dataclasses import dataclass
from enum import Enum

# Inheritance example - Traditional approach
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
        self.energy = 100
        self.health = 100
    
    def eat(self, food_amount):
        self.energy = min(100, self.energy + food_amount)
        return f"{self.name} ate and gained {food_amount} energy"
    
    def sleep(self, hours):
        self.health = min(100, self.health + hours * 5)
        return f"{self.name} slept for {hours} hours"
    
    def make_sound(self):
        return f"{self.name} makes a sound"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")
        self.breed = breed
        self.loyalty = 100
    
    def make_sound(self):
        return f"{self.name} barks: Woof!"
    
    def fetch(self, item):
        if self.energy > 10:
            self.energy -= 10
            return f"{self.name} fetched the {item}"
        return f"{self.name} is too tired to fetch"
    
    def guard(self):
        if self.energy > 20:
            self.energy -= 20
            return f"{self.name} is guarding the house"
        return f"{self.name} is too tired to guard"

class Cat(Animal):
    def __init__(self, name, color):
        super().__init__(name, "Cat")
        self.color = color
        self.independence = 100
    
    def make_sound(self):
        return f"{self.name} meows: Meow!"
    
    def hunt(self, prey):
        if self.energy > 15:
            self.energy -= 15
            return f"{self.name} hunted a {prey}"
        return f"{self.name} is too tired to hunt"
    
    def climb(self):
        if self.energy > 10:
            self.energy -= 10
            return f"{self.name} climbed up high"
        return f"{self.name} is too tired to climb"

class Bird(Animal):
    def __init__(self, name, wing_span):
        super().__init__(name, "Bird")
        self.wing_span = wing_span
        self.altitude = 0
    
    def make_sound(self):
        return f"{self.name} chirps: Tweet!"
    
    def fly(self, distance):
        if self.energy > distance:
            self.energy -= distance
            self.altitude = distance * 10
            return f"{self.name} flew {distance} units, altitude: {self.altitude}"
        return f"{self.name} is too tired to fly"

def inheritance_example():
    # Create animals using inheritance
    dog = Dog("Buddy", "Golden Retriever")
    cat = Cat("Whiskers", "Orange")
    bird = Bird("Tweety", 15.0)
    
    animals = [dog, cat, bird]
    
    # Test common behavior
    common_results = []
    for animal in animals:
        animal.eat(20)
        animal.sleep(2)
        common_results.append({
            "name": animal.name,
            "sound": animal.make_sound(),
            "energy": animal.energy,
            "health": animal.health
        })
    
    # Test specific behaviors
    specific_results = {
        "dog_fetch": dog.fetch("ball"),
        "dog_guard": dog.guard(),
        "cat_hunt": cat.hunt("mouse"),
        "cat_climb": cat.climb(),
        "bird_fly": bird.fly(5)
    }
    
    return {
        "common_behavior": common_results,
        "specific_behavior": specific_results,
        "inheritance_hierarchy": {
            "dog_is_animal": isinstance(dog, Animal),
            "cat_is_animal": isinstance(cat, Animal),
            "bird_is_animal": isinstance(bird, Animal)
        }
    }

# Composition example - Modern approach
class EnergySystem:
    def __init__(self, initial_energy=100):
        self.energy = initial_energy
        self.max_energy = 100
    
    def consume(self, amount):
        self.energy = max(0, self.energy - amount)
        return self.energy
    
    def restore(self, amount):
        self.energy = min(self.max_energy, self.energy + amount)
        return self.energy
    
    def has_energy(self, required):
        return self.energy >= required

class HealthSystem:
    def __init__(self, initial_health=100):
        self.health = initial_health
        self.max_health = 100
    
    def damage(self, amount):
        self.health = max(0, self.health - amount)
        return self.health
    
    def heal(self, amount):
        self.health = min(self.max_health, self.health + amount)
        return self.health
    
    def is_healthy(self):
        return self.health > 50

class SoundSystem:
    def __init__(self, default_sound="makes a sound"):
        self.default_sound = default_sound
        self.sound_variations = []
    
    def make_sound(self, creature_name):
        return f"{creature_name} {self.default_sound}"
    
    def add_sound_variation(self, variation):
        self.sound_variations.append(variation)

class MovementSystem:
    def __init__(self):
        self.position = {"x": 0, "y": 0, "z": 0}
        self.movement_history = []
    
    def move(self, dx, dy, dz=0):
        self.position["x"] += dx
        self.position["y"] += dy
        self.position["z"] += dz
        self.movement_history.append(self.position.copy())
        return self.position.copy()

# Composition-based creatures
class ComposedCreature:
    def __init__(self, name, species):
        self.name = name
        self.species = species
        self.energy_system = EnergySystem()
        self.health_system = HealthSystem()
        self.sound_system = SoundSystem()
        self.movement_system = MovementSystem()
        self.abilities = []
    
    def eat(self, food_amount):
        self.energy_system.restore(food_amount)
        return f"{self.name} ate and gained {food_amount} energy"
    
    def sleep(self, hours):
        self.health_system.heal(hours * 5)
        return f"{self.name} slept for {hours} hours"
    
    def make_sound(self):
        return self.sound_system.make_sound(self.name)
    
    def move(self, dx, dy, dz=0):
        if self.energy_system.has_energy(abs(dx) + abs(dy) + abs(dz)):
            self.energy_system.consume(abs(dx) + abs(dy) + abs(dz))
            new_pos = self.movement_system.move(dx, dy, dz)
            return f"{self.name} moved to {new_pos}"
        return f"{self.name} is too tired to move"
    
    def add_ability(self, ability):
        self.abilities.append(ability)
    
    def use_ability(self, ability_name, *args, **kwargs):
        for ability in self.abilities:
            if ability.name == ability_name:
                return ability.use(self, *args, **kwargs)
        return f"{self.name} doesn't have the ability {ability_name}"

# Ability system using composition
class Ability:
    def __init__(self, name, energy_cost=10, description=""):
        self.name = name
        self.energy_cost = energy_cost
        self.description = description
    
    def use(self, creature, *args, **kwargs):
        if creature.energy_system.has_energy(self.energy_cost):
            creature.energy_system.consume(self.energy_cost)
            return self._execute(creature, *args, **kwargs)
        return f"{creature.name} doesn't have enough energy for {self.name}"
    
    def _execute(self, creature, *args, **kwargs):
        return f"{creature.name} used {self.name}"

class FetchAbility(Ability):
    def __init__(self):
        super().__init__("fetch", 10, "Fetch an item")
    
    def _execute(self, creature, item="ball"):
        return f"{creature.name} fetched the {item}"

class HuntAbility(Ability):
    def __init__(self):
        super().__init__("hunt", 15, "Hunt prey")
    
    def _execute(self, creature, prey="mouse"):
        return f"{creature.name} hunted a {prey}"

class FlyAbility(Ability):
    def __init__(self):
        super().__init__("fly", 0, "Fly through the air")  # Uses move energy
    
    def _execute(self, creature, distance=5):
        creature.movement_system.position["z"] = distance * 10
        return f"{creature.name} flew to altitude {creature.movement_system.position['z']}"

class GuardAbility(Ability):
    def __init__(self):
        super().__init__("guard", 20, "Guard an area")
    
    def _execute(self, creature):
        return f"{creature.name} is guarding the area"

class ClimbAbility(Ability):
    def __init__(self):
        super().__init__("climb", 10, "Climb up high")
    
    def _execute(self, creature):
        creature.movement_system.position["z"] += 5
        return f"{creature.name} climbed up high"

def composition_example():
    # Create creatures using composition
    dog = ComposedCreature("Buddy", "Dog")
    dog.sound_system = SoundSystem("barks: Woof!")
    dog.add_ability(FetchAbility())
    dog.add_ability(GuardAbility())
    
    cat = ComposedCreature("Whiskers", "Cat")
    cat.sound_system = SoundSystem("meows: Meow!")
    cat.add_ability(HuntAbility())
    cat.add_ability(ClimbAbility())
    
    bird = ComposedCreature("Tweety", "Bird")
    bird.sound_system = SoundSystem("chirps: Tweet!")
    bird.add_ability(FlyAbility())
    
    creatures = [dog, cat, bird]
    
    # Test common behavior
    common_results = []
    for creature in creatures:
        creature.eat(20)
        creature.sleep(2)
        common_results.append({
            "name": creature.name,
            "sound": creature.make_sound(),
            "energy": creature.energy_system.energy,
            "health": creature.health_system.health
        })
    
    # Test specific abilities
    specific_results = {
        "dog_fetch": dog.use_ability("fetch", "stick"),
        "dog_guard": dog.use_ability("guard"),
        "cat_hunt": cat.use_ability("hunt", "bird"),
        "cat_climb": cat.use_ability("climb"),
        "bird_fly": bird.use_ability("fly", 8)
    }
    
    return {
        "common_behavior": common_results,
        "specific_behavior": specific_results,
        "composition_structure": {
            "dog_abilities": [ability.name for ability in dog.abilities],
            "cat_abilities": [ability.name for ability in cat.abilities],
            "bird_abilities": [ability.name for ability in bird.abilities]
        }
    }

# Mixin pattern - Best of both worlds
class MovementMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position = {"x": 0, "y": 0}
    
    def move_to(self, x, y):
        self.position = {"x": x, "y": y}
        return f"Moved to ({x}, {y})"

class EnergyMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy = 100
    
    def use_energy(self, amount):
        if self.energy >= amount:
            self.energy -= amount
            return True
        return False
    
    def restore_energy(self, amount):
        self.energy = min(100, self.energy + amount)

class SoundMixin:
    def __init__(self, sound="makes a sound", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sound = sound
    
    def make_sound(self):
        return f"{self.name} {self.sound}"

class MixinCreature(MovementMixin, EnergyMixin, SoundMixin):
    def __init__(self, name, sound="makes a sound"):
        self.name = name
        super().__init__(sound=sound)

class MixinDog(MixinCreature):
    def __init__(self, name):
        super().__init__(name, "barks: Woof!")
    
    def fetch(self):
        if self.use_energy(10):
            return f"{self.name} fetched the ball"
        return f"{self.name} is too tired"

class MixinCat(MixinCreature):
    def __init__(self, name):
        super().__init__(name, "meows: Meow!")
    
    def hunt(self):
        if self.use_energy(15):
            return f"{self.name} caught a mouse"
        return f"{self.name} is too tired"

def mixin_example():
    dog = MixinDog("Rex")
    cat = MixinCat("Mittens")
    
    # Test mixin functionality
    mixin_results = {
        "dog_sound": dog.make_sound(),
        "cat_sound": cat.make_sound(),
        "dog_move": dog.move_to(10, 5),
        "cat_move": cat.move_to(3, 7),
        "dog_fetch": dog.fetch(),
        "cat_hunt": cat.hunt(),
        "dog_energy": dog.energy,
        "cat_energy": cat.energy
    }
    
    # Test energy system
    dog.restore_energy(50)
    cat.restore_energy(30)
    
    mixin_results.update({
        "dog_energy_after_restore": dog.energy,
        "cat_energy_after_restore": cat.energy
    })
    
    return mixin_results

# Protocol-based composition
class Flyable(Protocol):
    def fly(self, distance: int) -> str: ...

class Swimmable(Protocol):
    def swim(self, distance: int) -> str: ...

class Walkable(Protocol):
    def walk(self, distance: int) -> str: ...

class FlyingSystem:
    def __init__(self, max_altitude=1000):
        self.altitude = 0
        self.max_altitude = max_altitude
    
    def fly(self, distance):
        self.altitude = min(self.max_altitude, self.altitude + distance * 10)
        return f"Flying at altitude {self.altitude}"

class SwimmingSystem:
    def __init__(self, max_depth=100):
        self.depth = 0
        self.max_depth = max_depth
    
    def swim(self, distance):
        self.depth = min(self.max_depth, self.depth + distance)
        return f"Swimming at depth {self.depth}"

class WalkingSystem:
    def __init__(self):
        self.distance_walked = 0
    
    def walk(self, distance):
        self.distance_walked += distance
        return f"Walked {distance} units, total: {self.distance_walked}"

class ProtocolCreature:
    def __init__(self, name):
        self.name = name
        self.movement_systems = {}
    
    def add_movement_system(self, system_type, system):
        self.movement_systems[system_type] = system
    
    def move(self, movement_type, distance):
        if movement_type in self.movement_systems:
            system = self.movement_systems[movement_type]
            if hasattr(system, movement_type):
                method = getattr(system, movement_type)
                return f"{self.name}: {method(distance)}"
        return f"{self.name} cannot {movement_type}"

def protocol_composition_example():
    # Create creatures with different movement capabilities
    duck = ProtocolCreature("Duck")
    duck.add_movement_system("fly", FlyingSystem(500))
    duck.add_movement_system("swim", SwimmingSystem(50))
    duck.add_movement_system("walk", WalkingSystem())
    
    fish = ProtocolCreature("Fish")
    fish.add_movement_system("swim", SwimmingSystem(200))
    
    bird = ProtocolCreature("Eagle")
    bird.add_movement_system("fly", FlyingSystem(2000))
    
    # Test movement capabilities
    protocol_results = {
        "duck_fly": duck.move("fly", 10),
        "duck_swim": duck.move("swim", 5),
        "duck_walk": duck.move("walk", 3),
        "fish_swim": fish.move("swim", 15),
        "fish_fly": fish.move("fly", 10),  # Should fail
        "bird_fly": bird.move("fly", 20),
        "bird_swim": bird.move("swim", 5)   # Should fail
    }
    
    return protocol_results

# Refactoring from inheritance to composition
class LegacyVehicle:
    """Legacy inheritance-based vehicle system"""
    def __init__(self, make, model):
        self.make = make
        self.model = model
        self.fuel = 100
        self.speed = 0
    
    def start_engine(self):
        return f"{self.make} {self.model} engine started"
    
    def accelerate(self, amount):
        if self.fuel > 0:
            self.speed += amount
            self.fuel -= amount // 10
            return f"Accelerated to {self.speed} mph"
        return "Out of fuel"

class LegacyCar(LegacyVehicle):
    def __init__(self, make, model, doors):
        super().__init__(make, model)
        self.doors = doors
    
    def open_trunk(self):
        return f"{self.make} {self.model} trunk opened"

class LegacyMotorcycle(LegacyVehicle):
    def __init__(self, make, model, engine_size):
        super().__init__(make, model)
        self.engine_size = engine_size
    
    def wheelie(self):
        if self.speed > 20:
            return f"{self.make} {self.model} doing a wheelie!"
        return "Need more speed for wheelie"

# Refactored composition-based system
class Engine:
    def __init__(self, engine_type="gasoline"):
        self.engine_type = engine_type
        self.running = False
    
    def start(self):
        self.running = True
        return f"{self.engine_type.title()} engine started"
    
    def stop(self):
        self.running = False
        return f"Engine stopped"

class FuelSystem:
    def __init__(self, capacity=100, fuel_type="gasoline"):
        self.capacity = capacity
        self.fuel = capacity
        self.fuel_type = fuel_type
    
    def consume(self, amount):
        consumed = min(self.fuel, amount)
        self.fuel -= consumed
        return consumed
    
    def refuel(self, amount):
        added = min(self.capacity - self.fuel, amount)
        self.fuel += added
        return added

class DriveSystem:
    def __init__(self):
        self.speed = 0
        self.distance = 0
    
    def accelerate(self, amount, fuel_system):
        fuel_needed = amount // 10
        if fuel_system.consume(fuel_needed) > 0:
            self.speed += amount
            return f"Accelerated to {self.speed} mph"
        return "Insufficient fuel"
    
    def brake(self, amount):
        self.speed = max(0, self.speed - amount)
        return f"Slowed to {self.speed} mph"

class ModernVehicle:
    def __init__(self, make, model):
        self.make = make
        self.model = model
        self.engine = Engine()
        self.fuel_system = FuelSystem()
        self.drive_system = DriveSystem()
        self.features = []
    
    def add_feature(self, feature):
        self.features.append(feature)
    
    def use_feature(self, feature_name, *args, **kwargs):
        for feature in self.features:
            if feature.name == feature_name:
                return feature.use(*args, **kwargs)
        return f"Feature {feature_name} not available"
    
    def start(self):
        return self.engine.start()
    
    def accelerate(self, amount):
        if self.engine.running:
            return self.drive_system.accelerate(amount, self.fuel_system)
        return "Engine not running"

class VehicleFeature:
    def __init__(self, name):
        self.name = name
    
    def use(self, *args, **kwargs):
        return f"Used {self.name}"

class TrunkFeature(VehicleFeature):
    def __init__(self):
        super().__init__("trunk")
        self.open = False
    
    def use(self):
        self.open = not self.open
        state = "opened" if self.open else "closed"
        return f"Trunk {state}"

class WheelieFeatue(VehicleFeature):
    def __init__(self, vehicle):
        super().__init__("wheelie")
        self.vehicle = vehicle
    
    def use(self):
        if self.vehicle.drive_system.speed > 20:
            return f"{self.vehicle.make} {self.vehicle.model} doing a wheelie!"
        return "Need more speed for wheelie"

def refactoring_example():
    # Legacy system
    legacy_car = LegacyCar("Toyota", "Camry", 4)
    legacy_motorcycle = LegacyMotorcycle("Harley", "Sportster", 883)
    
    legacy_results = {
        "car_start": legacy_car.start_engine(),
        "car_accelerate": legacy_car.accelerate(30),
        "car_trunk": legacy_car.open_trunk(),
        "motorcycle_start": legacy_motorcycle.start_engine(),
        "motorcycle_accelerate": legacy_motorcycle.accelerate(40),
        "motorcycle_wheelie": legacy_motorcycle.wheelie()
    }
    
    # Modern composition system
    modern_car = ModernVehicle("Honda", "Civic")
    modern_car.add_feature(TrunkFeature())
    
    modern_motorcycle = ModernVehicle("Yamaha", "YZF")
    modern_motorcycle.add_feature(WheelieFeatue(modern_motorcycle))
    
    modern_results = {
        "car_start": modern_car.start(),
        "car_accelerate": modern_car.accelerate(30),
        "car_trunk": modern_car.use_feature("trunk"),
        "motorcycle_start": modern_motorcycle.start(),
        "motorcycle_accelerate": modern_motorcycle.accelerate(40),
        "motorcycle_wheelie": modern_motorcycle.use_feature("wheelie")
    }
    
    return {
        "legacy_system": legacy_results,
        "modern_system": modern_results,
        "benefits": [
            "Modern system is more flexible and extensible",
            "Features can be mixed and matched",
            "Systems can be reused across different vehicle types",
            "Easier to test individual components",
            "No deep inheritance hierarchies"
        ]
    }

# Composition patterns comparison
def composition_patterns_comparison():
    patterns = {
        "Pure Composition": {
            "description": "Objects contain other objects as components",
            "pros": ["Maximum flexibility", "Easy to test", "No inheritance issues"],
            "cons": ["More verbose", "Requires more setup code"],
            "use_when": "Complex systems with many interchangeable parts"
        },
        "Mixins": {
            "description": "Multiple inheritance with focused functionality",
            "pros": ["Reusable functionality", "Less verbose than composition", "Pythonic"],
            "cons": ["Multiple inheritance complexity", "Method resolution order issues"],
            "use_when": "Cross-cutting concerns across multiple classes"
        },
        "Protocol-based": {
            "description": "Duck typing with explicit interfaces",
            "pros": ["Type safety", "Clear contracts", "Flexible implementations"],
            "cons": ["Requires Python 3.8+", "More abstract"],
            "use_when": "Large codebases with type checking requirements"
        },
        "Hybrid Approach": {
            "description": "Combines inheritance, composition, and mixins",
            "pros": ["Best of all worlds", "Gradual refactoring possible"],
            "cons": ["Can be complex", "Requires careful design"],
            "use_when": "Refactoring legacy code or complex domains"
        }
    }
    
    return patterns

# Design principles demonstration
def design_principles_demo():
    # Single Responsibility Principle
    class EmailValidator:
        def validate(self, email):
            return "@" in email and "." in email
    
    class UserCreator:
        def __init__(self, validator):
            self.validator = validator
        
        def create_user(self, name, email):
            if self.validator.validate(email):
                return {"name": name, "email": email, "created": True}
            return {"error": "Invalid email"}
    
    # Open/Closed Principle with composition
    class NotificationSender:
        def __init__(self):
            self.channels = []
        
        def add_channel(self, channel):
            self.channels.append(channel)
        
        def send(self, message):
            results = []
            for channel in self.channels:
                results.append(channel.send(message))
            return results
    
    class EmailChannel:
        def send(self, message):
            return f"Email sent: {message}"
    
    class SMSChannel:
        def send(self, message):
            return f"SMS sent: {message}"
    
    class SlackChannel:
        def send(self, message):
            return f"Slack sent: {message}"
    
    # Dependency Inversion with composition
    class OrderProcessor:
        def __init__(self, payment_gateway, inventory_service):
            self.payment_gateway = payment_gateway
            self.inventory_service = inventory_service
        
        def process_order(self, order):
            if self.inventory_service.check_availability(order["item"]):
                payment_result = self.payment_gateway.process_payment(order["amount"])
                if payment_result["success"]:
                    return {"status": "success", "order_id": hash(str(order))}
            return {"status": "failed", "reason": "Processing failed"}
    
    class MockPaymentGateway:
        def process_payment(self, amount):
            return {"success": amount > 0, "transaction_id": f"txn_{amount}"}
    
    class MockInventoryService:
        def check_availability(self, item):
            return item != "out_of_stock_item"
    
    # Test design principles
    validator = EmailValidator()
    user_creator = UserCreator(validator)
    
    notification_sender = NotificationSender()
    notification_sender.add_channel(EmailChannel())
    notification_sender.add_channel(SMSChannel())
    notification_sender.add_channel(SlackChannel())
    
    order_processor = OrderProcessor(
        MockPaymentGateway(),
        MockInventoryService()
    )
    
    results = {
        "srp_example": user_creator.create_user("John", "john@example.com"),
        "ocp_example": notification_sender.send("Hello World"),
        "dip_example": order_processor.process_order({"item": "laptop", "amount": 999})
    }
    
    return results

# Comprehensive testing
def run_all_composition_demos():
    """Execute all composition vs inheritance demonstrations"""
    demo_functions = [
        ('inheritance_example', inheritance_example),
        ('composition_example', composition_example),
        ('mixin_example', mixin_example),
        ('protocol_composition', protocol_composition_example),
        ('refactoring_example', refactoring_example),
        ('composition_patterns', composition_patterns_comparison),
        ('design_principles', design_principles_demo)
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
    print("=== Python Composition vs Inheritance Demo ===")
    
    # Run all demonstrations
    all_results = run_all_composition_demos()
    
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
    
    print("\n=== INHERITANCE vs COMPOSITION ===")
    
    comparison = {
        "Inheritance (is-a)": [
            "Models is-a relationships naturally",
            "Code reuse through parent classes",
            "Polymorphism through method overriding",
            "Can lead to deep hierarchies",
            "Tight coupling between parent and child",
            "Hard to change base class without affecting children"
        ],
        "Composition (has-a)": [
            "Models has-a relationships naturally",
            "Flexible runtime configuration",
            "Easy to test individual components",
            "Loose coupling between components",
            "Can change behavior by swapping components",
            "More verbose but more explicit"
        ]
    }
    
    for approach, characteristics in comparison.items():
        print(f"  {approach}:")
        for char in characteristics:
            print(f"    • {char}")
    
    print("\n=== DESIGN PRINCIPLES ===")
    
    principles = {
        "Favor Composition Over Inheritance": "Use composition when possible, inheritance when necessary",
        "Single Responsibility": "Each class should have one reason to change",
        "Open/Closed Principle": "Open for extension, closed for modification",
        "Liskov Substitution": "Subclasses should be substitutable for their base classes",
        "Interface Segregation": "Many specific interfaces better than one general interface",
        "Dependency Inversion": "Depend on abstractions, not concretions"
    }
    
    for principle, description in principles.items():
        print(f"  {principle}: {description}")
    
    print("\n=== WHEN TO USE WHAT ===")
    
    guidelines = {
        "Use Inheritance When": [
            "True is-a relationship exists",
            "You need polymorphic behavior",
            "Base class provides significant shared functionality",
            "The hierarchy is stable and won't change often"
        ],
        "Use Composition When": [
            "Has-a relationship is more appropriate",
            "You need runtime flexibility",
            "Components can be reused in different contexts",
            "You want to avoid deep inheritance hierarchies"
        ],
        "Use Mixins When": [
            "You have cross-cutting concerns",
            "You need multiple inheritance of behavior",
            "The functionality is orthogonal to the main class purpose",
            "You're working in a Python-centric environment"
        ],
        "Use Protocols When": [
            "You need duck typing with type safety",
            "Working with large codebases",
            "You want clear interface contracts",
            "Using static type checkers like MyPy"
        ]
    }
    
    for category, items in guidelines.items():
        print(f"  {category}:")
        for item in items:
            print(f"    • {item}")
    
    print("\n=== REFACTORING STRATEGIES ===")
    
    strategies = [
        "Start with inheritance for simple cases",
        "Refactor to composition when complexity grows",
        "Extract common functionality into mixins",
        "Use dependency injection for external dependencies",
        "Prefer protocols over abstract base classes for interfaces",
        "Test each component independently",
        "Document architectural decisions and trade-offs",
        "Gradually migrate from inheritance to composition"
    ]
    
    for strategy in strategies:
        print(f"  • {strategy}")
    
    print("\n=== Composition vs Inheritance Complete! ===")
    print("  Design principles and architectural patterns mastered")
