"""
POLYMORPHISM IMPLEMENTATION - Method Overriding and Overloading
==============================================================

Problem Statement:
Demonstrate polymorphism concepts including:
- Method overriding and dynamic dispatch
- Method overloading simulation in Python
- Runtime polymorphism with inheritance
- Duck typing and protocol-based polymorphism
- Operator overloading

Learning Objectives:
- Understand runtime polymorphism mechanisms
- Implement method overriding effectively
- Simulate method overloading in Python
- Use duck typing for flexible interfaces
- Overload operators for custom classes
"""

from abc import ABC, abstractmethod
from typing import Union, List, Any, Protocol
from functools import singledispatch
import math


# Abstract base class for polymorphism demonstration
class Animal(ABC):
    """
    Abstract Animal class demonstrating polymorphic behavior.
    """
    
    def __init__(self, name: str, species: str, age: int):
        """Initialize animal with basic properties."""
        self.name = name
        self.species = species
        self.age = age
        self.energy = 100
        self.hunger = 0
    
    @abstractmethod
    def make_sound(self) -> str:
        """Abstract method that must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def move(self) -> str:
        """Abstract method for movement behavior."""
        pass
    
    def eat(self, food_amount: int = 10) -> str:
        """Common eating behavior (can be overridden)."""
        self.hunger = max(0, self.hunger - food_amount)
        self.energy = min(100, self.energy + food_amount // 2)
        return f"{self.name} is eating. Hunger: {self.hunger}, Energy: {self.energy}"
    
    def sleep(self, hours: int = 8) -> str:
        """Common sleeping behavior."""
        self.energy = min(100, self.energy + hours * 5)
        return f"{self.name} slept for {hours} hours. Energy: {self.energy}"
    
    def get_info(self) -> str:
        """Get animal information."""
        return f"{self.name} ({self.species}), Age: {self.age}, Energy: {self.energy}"


# Concrete implementations demonstrating method overriding
class Dog(Animal):
    """Dog class with specific implementations."""
    
    def __init__(self, name: str, breed: str, age: int):
        """Initialize dog with breed information."""
        super().__init__(name, "Dog", age)
        self.breed = breed
        self.loyalty = 100
        self.tricks_known = []
    
    def make_sound(self) -> str:
        """Override abstract method - dogs bark."""
        return f"{self.name} says: Woof! Woof!"
    
    def move(self) -> str:
        """Override abstract method - dogs run."""
        self.energy -= 5
        return f"{self.name} runs around energetically!"
    
    def eat(self, food_amount: int = 15) -> str:
        """Override eating behavior - dogs eat more enthusiastically."""
        result = super().eat(food_amount)
        return result + " *wags tail*"
    
    def fetch(self, item: str = "ball") -> str:
        """Dog-specific behavior."""
        self.energy -= 10
        self.loyalty += 5
        return f"{self.name} fetches the {item}!"
    
    def learn_trick(self, trick: str) -> str:
        """Teach dog a new trick."""
        if trick not in self.tricks_known:
            self.tricks_known.append(trick)
            return f"{self.name} learned {trick}!"
        return f"{self.name} already knows {trick}."
    
    def perform_trick(self, trick: str) -> str:
        """Perform a learned trick."""
        if trick in self.tricks_known:
            return f"{self.name} performs {trick}!"
        return f"{self.name} doesn't know {trick} yet."


class Cat(Animal):
    """Cat class with specific implementations."""
    
    def __init__(self, name: str, breed: str, age: int):
        """Initialize cat with breed information."""
        super().__init__(name, "Cat", age)
        self.breed = breed
        self.independence = 80
        self.lives_remaining = 9
    
    def make_sound(self) -> str:
        """Override abstract method - cats meow."""
        return f"{self.name} says: Meow!"
    
    def move(self) -> str:
        """Override abstract method - cats prowl."""
        self.energy -= 3
        return f"{self.name} prowls silently."
    
    def eat(self, food_amount: int = 8) -> str:
        """Override eating behavior - cats are picky eaters."""
        if food_amount > 12:
            return f"{self.name} sniffs the food and walks away."
        result = super().eat(food_amount)
        return result + " *purrs contentedly*"
    
    def climb(self, height: str = "tree") -> str:
        """Cat-specific behavior."""
        self.energy -= 8
        return f"{self.name} climbs up the {height}."
    
    def purr(self) -> str:
        """Cat-specific behavior."""
        self.energy += 2
        return f"{self.name} purrs softly."


class Bird(Animal):
    """Bird class with flying capabilities."""
    
    def __init__(self, name: str, species: str, age: int, can_fly: bool = True):
        """Initialize bird with flight capability."""
        super().__init__(name, species, age)
        self.can_fly = can_fly
        self.altitude = 0
    
    def make_sound(self) -> str:
        """Override abstract method - birds chirp."""
        return f"{self.name} says: Tweet! Tweet!"
    
    def move(self) -> str:
        """Override abstract method - birds can fly or hop."""
        if self.can_fly:
            self.energy -= 7
            self.altitude += 10
            return f"{self.name} flies gracefully!"
        else:
            self.energy -= 2
            return f"{self.name} hops around."
    
    def fly(self, distance: int = 100) -> str:
        """Bird-specific flying behavior."""
        if not self.can_fly:
            return f"{self.name} cannot fly."
        
        self.energy -= distance // 10
        self.altitude += distance // 20
        return f"{self.name} flies {distance} meters!"


# Method overloading simulation using default parameters and *args
class Calculator:
    """
    Calculator class demonstrating method overloading simulation.
    Python doesn't support true method overloading, but we can simulate it.
    """
    
    def add(self, *args) -> Union[int, float]:
        """
        Add method with variable arguments (simulates overloading).
        Can handle different numbers of arguments.
        """
        if len(args) == 0:
            return 0
        elif len(args) == 1:
            return args[0]
        else:
            return sum(args)
    
    def multiply(self, a: Union[int, float], b: Union[int, float] = 1) -> Union[int, float]:
        """Multiply with optional second parameter."""
        return a * b
    
    def power(self, base: Union[int, float], exponent: Union[int, float] = 2) -> Union[int, float]:
        """Power function with default exponent of 2."""
        return base ** exponent
    
    def divide(self, dividend: Union[int, float], divisor: Union[int, float] = 1) -> Union[int, float]:
        """Division with default divisor of 1."""
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        return dividend / divisor


# Using singledispatch for method overloading based on type
@singledispatch
def process_data(data):
    """Generic data processing function."""
    raise NotImplementedError(f"Unsupported type: {type(data)}")

@process_data.register
def _(data: int):
    """Process integer data."""
    return f"Processing integer: {data * 2}"

@process_data.register
def _(data: str):
    """Process string data."""
    return f"Processing string: {data.upper()}"

@process_data.register
def _(data: list):
    """Process list data."""
    return f"Processing list of {len(data)} items: {sum(data) if all(isinstance(x, (int, float)) for x in data) else 'mixed types'}"


# Duck typing demonstration
class Duck:
    """Duck class for duck typing demonstration."""
    
    def quack(self) -> str:
        return "Quack!"
    
    def fly(self) -> str:
        return "Duck flies!"


class Airplane:
    """Airplane class that also 'flies' - duck typing."""
    
    def quack(self) -> str:
        return "Airplane engine noise!"
    
    def fly(self) -> str:
        return "Airplane flies high!"


class Robot:
    """Robot class that can mimic duck behavior."""
    
    def quack(self) -> str:
        return "Beep! Simulating quack!"
    
    def fly(self) -> str:
        return "Robot activates propellers!"


def make_it_quack_and_fly(duck_like_object):
    """
    Function demonstrating duck typing.
    Accepts any object that has quack() and fly() methods.
    """
    print(duck_like_object.quack())
    print(duck_like_object.fly())


# Protocol-based polymorphism (Python 3.8+)
class Drawable(Protocol):
    """Protocol defining drawable interface."""
    
    def draw(self) -> str:
        """Draw the object."""
        ...


class Circle:
    """Circle class implementing Drawable protocol."""
    
    def __init__(self, radius: float):
        self.radius = radius
    
    def draw(self) -> str:
        return f"Drawing circle with radius {self.radius}"
    
    def area(self) -> float:
        return math.pi * self.radius ** 2


class Rectangle:
    """Rectangle class implementing Drawable protocol."""
    
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def draw(self) -> str:
        return f"Drawing rectangle {self.width}x{self.height}"
    
    def area(self) -> float:
        return self.width * self.height


def draw_shape(shape: Drawable) -> str:
    """Function that works with any Drawable object."""
    return shape.draw()


# Operator overloading demonstration
class Vector:
    """
    Vector class demonstrating operator overloading.
    """
    
    def __init__(self, x: float, y: float):
        """Initialize vector with x and y components."""
        self.x = x
        self.y = y
    
    def __add__(self, other: 'Vector') -> 'Vector':
        """Overload + operator for vector addition."""
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        raise TypeError("Can only add Vector to Vector")
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        """Overload - operator for vector subtraction."""
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        raise TypeError("Can only subtract Vector from Vector")
    
    def __mul__(self, scalar: Union[int, float]) -> 'Vector':
        """Overload * operator for scalar multiplication."""
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        raise TypeError("Can only multiply Vector by scalar")
    
    def __rmul__(self, scalar: Union[int, float]) -> 'Vector':
        """Overload right multiplication (scalar * vector)."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: Union[int, float]) -> 'Vector':
        """Overload / operator for scalar division."""
        if isinstance(scalar, (int, float)) and scalar != 0:
            return Vector(self.x / scalar, self.y / scalar)
        raise ValueError("Cannot divide by zero or non-scalar")
    
    def __eq__(self, other: 'Vector') -> bool:
        """Overload == operator for vector equality."""
        if isinstance(other, Vector):
            return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10
        return False
    
    def __ne__(self, other: 'Vector') -> bool:
        """Overload != operator."""
        return not self.__eq__(other)
    
    def __abs__(self) -> float:
        """Overload abs() function for vector magnitude."""
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def __str__(self) -> str:
        """String representation of vector."""
        return f"Vector({self.x:.2f}, {self.y:.2f})"
    
    def __repr__(self) -> str:
        """Developer representation of vector."""
        return f"Vector({self.x}, {self.y})"
    
    def dot_product(self, other: 'Vector') -> float:
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y
    
    def magnitude(self) -> float:
        """Get vector magnitude."""
        return abs(self)
    
    def normalize(self) -> 'Vector':
        """Get normalized vector."""
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0)
        return self / mag


def demonstrate_polymorphism():
    """
    Demonstrate polymorphism concepts with practical examples.
    """
    print("=== POLYMORPHISM IMPLEMENTATION DEMONSTRATION ===\n")
    
    # 1. Runtime Polymorphism with Method Overriding
    print("1. Runtime Polymorphism - Method Overriding:")
    animals = [
        Dog("Buddy", "Golden Retriever", 3),
        Cat("Whiskers", "Persian", 2),
        Bird("Tweety", "Canary", 1)
    ]
    
    for animal in animals:
        print(f"Animal: {animal.get_info()}")
        print(f"  Sound: {animal.make_sound()}")
        print(f"  Movement: {animal.move()}")
        print(f"  Eating: {animal.eat()}")
        print()
    
    # 2. Specific behaviors (method overriding with extensions)
    print("2. Specific Animal Behaviors:")
    dog = animals[0]
    cat = animals[1]
    bird = animals[2]
    
    print(f"Dog specific: {dog.fetch()}")
    dog.learn_trick("sit")
    dog.learn_trick("roll over")
    print(f"Dog tricks: {dog.perform_trick('sit')}")
    
    print(f"Cat specific: {cat.climb()}")
    print(f"Cat behavior: {cat.purr()}")
    
    print(f"Bird specific: {bird.fly(200)}")
    print()
    
    # 3. Method Overloading Simulation
    print("3. Method Overloading Simulation:")
    calc = Calculator()
    
    print(f"add(): {calc.add()}")
    print(f"add(5): {calc.add(5)}")
    print(f"add(1, 2): {calc.add(1, 2)}")
    print(f"add(1, 2, 3, 4): {calc.add(1, 2, 3, 4)}")
    
    print(f"multiply(5): {calc.multiply(5)}")
    print(f"multiply(5, 3): {calc.multiply(5, 3)}")
    
    print(f"power(2): {calc.power(2)}")
    print(f"power(2, 3): {calc.power(2, 3)}")
    print()
    
    # 4. Single Dispatch (Type-based Method Overloading)
    print("4. Single Dispatch - Type-based Overloading:")
    print(process_data(42))
    print(process_data("hello world"))
    print(process_data([1, 2, 3, 4, 5]))
    print()
    
    # 5. Duck Typing
    print("5. Duck Typing:")
    duck_like_objects = [Duck(), Airplane(), Robot()]
    
    for obj in duck_like_objects:
        print(f"{obj.__class__.__name__}:")
        make_it_quack_and_fly(obj)
        print()
    
    # 6. Protocol-based Polymorphism
    print("6. Protocol-based Polymorphism:")
    shapes = [Circle(5.0), Rectangle(4.0, 6.0)]
    
    for shape in shapes:
        print(f"{shape.__class__.__name__}: {draw_shape(shape)}")
        if hasattr(shape, 'area'):
            print(f"  Area: {shape.area():.2f}")
    print()
    
    # 7. Operator Overloading
    print("7. Operator Overloading:")
    v1 = Vector(3.0, 4.0)
    v2 = Vector(1.0, 2.0)
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"v1 * 2 = {v1 * 2}")
    print(f"3 * v2 = {3 * v2}")
    print(f"v1 / 2 = {v1 / 2}")
    print(f"|v1| = {abs(v1):.2f}")
    print(f"v1 == v2: {v1 == v2}")
    print(f"v1 != v2: {v1 != v2}")
    print(f"v1 · v2 = {v1.dot_product(v2):.2f}")
    print(f"v1 normalized = {v1.normalize()}")
    print()
    
    # 8. Polymorphic Collections
    print("8. Polymorphic Collections:")
    drawable_objects = [Circle(3.0), Rectangle(2.0, 4.0)]
    
    total_area = 0
    for obj in drawable_objects:
        print(draw_shape(obj))
        total_area += obj.area()
    
    print(f"Total area of all shapes: {total_area:.2f}")
    print()
    
    print("=== POLYMORPHISM DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_polymorphism()
