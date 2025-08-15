"""
Python Inheritance and Polymorphism: Single/Multiple Inheritance, MRO, and Polymorphic Patterns
Implementation-focused with minimal comments, maximum functionality coverage
"""

import abc
from typing import List, Any, Protocol

# Single inheritance basics
class Animal:
    species_count = 0
    
    def __init__(self, name, species):
        self.name = name
        self.species = species
        self.age = 0
        Animal.species_count += 1
    
    def make_sound(self):
        return f"{self.name} makes a sound"
    
    def get_info(self):
        return f"{self.name} is a {self.species}, age {self.age}"
    
    def birthday(self):
        self.age += 1
        return f"{self.name} is now {self.age} years old"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")  # Call parent constructor
        self.breed = breed
        self.tricks = []
    
    def make_sound(self):
        # Method overriding
        return f"{self.name} barks: Woof!"
    
    def learn_trick(self, trick):
        self.tricks.append(trick)
        return f"{self.name} learned {trick}"
    
    def perform_tricks(self):
        if not self.tricks:
            return f"{self.name} doesn't know any tricks yet"
        return f"{self.name} performs: {', '.join(self.tricks)}"
    
    def get_info(self):
        # Method overriding with extension
        base_info = super().get_info()
        return f"{base_info}, breed: {self.breed}, tricks: {len(self.tricks)}"

class Cat(Animal):
    def __init__(self, name, indoor=True):
        super().__init__(name, "Cat")
        self.indoor = indoor
        self.lives = 9
    
    def make_sound(self):
        return f"{self.name} meows: Meow!"
    
    def lose_life(self):
        if self.lives > 0:
            self.lives -= 1
            return f"{self.name} lost a life, {self.lives} remaining"
        return f"{self.name} has no more lives!"
    
    def get_info(self):
        base_info = super().get_info()
        location = "indoor" if self.indoor else "outdoor"
        return f"{base_info}, {location} cat, lives: {self.lives}"

def single_inheritance_demo():
    # Create instances
    dog = Dog("Buddy", "Golden Retriever")
    cat = Cat("Whiskers", indoor=True)
    
    # Test inherited methods
    dog.birthday()
    cat.birthday()
    
    # Test overridden methods
    dog_sound = dog.make_sound()
    cat_sound = cat.make_sound()
    
    # Test extended functionality
    dog.learn_trick("sit")
    dog.learn_trick("roll over")
    cat.lose_life()
    
    return {
        "dog_info": dog.get_info(),
        "cat_info": cat.get_info(),
        "dog_sound": dog_sound,
        "cat_sound": cat_sound,
        "dog_tricks": dog.perform_tricks(),
        "cat_lives": cat.lives,
        "total_animals": Animal.species_count,
        "inheritance_working": isinstance(dog, Animal) and isinstance(cat, Animal)
    }

# Multiple inheritance and Method Resolution Order (MRO)
class Flyable:
    def __init__(self):
        self.altitude = 0
        self.flying = False
    
    def take_off(self):
        self.flying = True
        self.altitude = 100
        return f"Taking off, altitude: {self.altitude}ft"
    
    def land(self):
        self.flying = False
        self.altitude = 0
        return "Landed safely"
    
    def get_flight_info(self):
        status = "flying" if self.flying else "grounded"
        return f"Status: {status}, Altitude: {self.altitude}ft"

class Swimmable:
    def __init__(self):
        self.depth = 0
        self.swimming = False
    
    def dive(self):
        self.swimming = True
        self.depth = 10
        return f"Diving, depth: {self.depth}ft"
    
    def surface(self):
        self.swimming = False
        self.depth = 0
        return "Surfaced"
    
    def get_swim_info(self):
        status = "swimming" if self.swimming else "on surface"
        return f"Status: {status}, Depth: {self.depth}ft"

class Duck(Animal, Flyable, Swimmable):
    def __init__(self, name):
        # Multiple inheritance requires careful initialization
        Animal.__init__(self, name, "Duck")
        Flyable.__init__(self)
        Swimmable.__init__(self)
        self.feather_color = "brown"
    
    def make_sound(self):
        return f"{self.name} quacks: Quack!"
    
    def get_all_info(self):
        return {
            "animal_info": self.get_info(),
            "flight_info": self.get_flight_info(),
            "swim_info": self.get_swim_info(),
            "feather_color": self.feather_color
        }

class Penguin(Animal, Swimmable):
    def __init__(self, name):
        Animal.__init__(self, name, "Penguin")
        Swimmable.__init__(self)
        self.can_fly = False
    
    def make_sound(self):
        return f"{self.name} honks: Honk!"
    
    def slide_on_ice(self):
        return f"{self.name} slides gracefully on ice"

def multiple_inheritance_demo():
    duck = Duck("Quackers")
    penguin = Penguin("Pingu")
    
    # Test multiple inheritance capabilities
    duck_flight = duck.take_off()
    duck_swim = duck.dive()
    duck_sound = duck.make_sound()
    
    penguin_swim = penguin.dive()
    penguin_sound = penguin.make_sound()
    penguin_slide = penguin.slide_on_ice()
    
    # Method Resolution Order analysis
    duck_mro = [cls.__name__ for cls in Duck.__mro__]
    penguin_mro = [cls.__name__ for cls in Penguin.__mro__]
    
    return {
        "duck_capabilities": {
            "flight": duck_flight,
            "swim": duck_swim,
            "sound": duck_sound,
            "all_info": duck.get_all_info()
        },
        "penguin_capabilities": {
            "swim": penguin_swim,
            "sound": penguin_sound,
            "slide": penguin_slide,
            "can_fly": penguin.can_fly
        },
        "mro_analysis": {
            "duck_mro": duck_mro,
            "penguin_mro": penguin_mro
        },
        "inheritance_checks": {
            "duck_is_animal": isinstance(duck, Animal),
            "duck_is_flyable": isinstance(duck, Flyable),
            "duck_is_swimmable": isinstance(duck, Swimmable),
            "penguin_is_flyable": isinstance(penguin, Flyable)
        }
    }

# Abstract base classes
class Shape(abc.ABC):
    def __init__(self, name):
        self.name = name
    
    @abc.abstractmethod
    def area(self):
        """All shapes must implement area calculation"""
        pass
    
    @abc.abstractmethod
    def perimeter(self):
        """All shapes must implement perimeter calculation"""
        pass
    
    def get_info(self):
        return f"{self.name}: Area={self.area():.2f}, Perimeter={self.perimeter():.2f}"

class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

class Triangle(Shape):
    def __init__(self, side1, side2, side3):
        super().__init__("Triangle")
        self.side1 = side1
        self.side2 = side2
        self.side3 = side3
    
    def area(self):
        # Using Heron's formula
        s = self.perimeter() / 2
        return (s * (s - self.side1) * (s - self.side2) * (s - self.side3)) ** 0.5
    
    def perimeter(self):
        return self.side1 + self.side2 + self.side3

def abstract_classes_demo():
    # Cannot instantiate abstract class
    try:
        shape = Shape("Generic")
        abstract_instantiation = True
    except TypeError:
        abstract_instantiation = False
    
    # Create concrete implementations
    rectangle = Rectangle(10, 5)
    circle = Circle(7)
    triangle = Triangle(3, 4, 5)
    
    shapes = [rectangle, circle, triangle]
    
    # Polymorphic behavior
    shape_calculations = []
    for shape in shapes:
        shape_calculations.append({
            "name": shape.name,
            "area": shape.area(),
            "perimeter": shape.perimeter(),
            "info": shape.get_info()
        })
    
    return {
        "abstract_instantiation_failed": not abstract_instantiation,
        "shape_calculations": shape_calculations,
        "polymorphism_working": all(hasattr(shape, 'area') and hasattr(shape, 'perimeter') for shape in shapes)
    }

# Polymorphism with duck typing
class Document:
    def __init__(self, content):
        self.content = content
    
    def render(self):
        return f"Rendering document: {self.content[:50]}..."

class PDFDocument:
    def __init__(self, content, pages):
        self.content = content
        self.pages = pages
    
    def render(self):
        return f"Rendering PDF ({self.pages} pages): {self.content[:30]}..."

class HTMLDocument:
    def __init__(self, content, title):
        self.content = content
        self.title = title
    
    def render(self):
        return f"Rendering HTML '{self.title}': {self.content[:40]}..."

class ImageDocument:
    def __init__(self, filename, format_type):
        self.filename = filename
        self.format_type = format_type
    
    def render(self):
        return f"Rendering {self.format_type} image: {self.filename}"

# Document processor using duck typing
class DocumentProcessor:
    def __init__(self):
        self.processed_count = 0
    
    def process_document(self, document):
        # Duck typing - if it has render(), we can process it
        if hasattr(document, 'render') and callable(getattr(document, 'render')):
            result = document.render()
            self.processed_count += 1
            return result
        raise TypeError("Object must have a render() method")
    
    def process_batch(self, documents):
        results = []
        for doc in documents:
            try:
                results.append(self.process_document(doc))
            except TypeError as e:
                results.append(f"Error: {e}")
        return results

def duck_typing_demo():
    # Create various document types
    text_doc = Document("This is a simple text document with some content.")
    pdf_doc = PDFDocument("PDF document content with formatting.", 5)
    html_doc = HTMLDocument("<h1>Hello World</h1><p>HTML content</p>", "Welcome Page")
    image_doc = ImageDocument("sunset.jpg", "JPEG")
    
    # Create non-document object for testing
    class NonDocument:
        def __init__(self, data):
            self.data = data
    
    non_doc = NonDocument("Not a document")
    
    # Process documents
    processor = DocumentProcessor()
    documents = [text_doc, pdf_doc, html_doc, image_doc, non_doc]
    
    batch_results = processor.process_batch(documents)
    
    return {
        "batch_results": batch_results,
        "processed_count": processor.processed_count,
        "duck_typing_successful": processor.processed_count == 4,  # All except non_doc
        "polymorphism_demonstration": "Different classes with same interface worked together"
    }

# Composition vs Inheritance
class Engine:
    def __init__(self, horsepower, fuel_type):
        self.horsepower = horsepower
        self.fuel_type = fuel_type
        self.running = False
    
    def start(self):
        self.running = True
        return f"Engine started: {self.horsepower}HP {self.fuel_type}"
    
    def stop(self):
        self.running = False
        return "Engine stopped"
    
    def get_status(self):
        status = "running" if self.running else "stopped"
        return f"{self.horsepower}HP {self.fuel_type} engine: {status}"

class GPS:
    def __init__(self):
        self.current_location = "Unknown"
        self.destination = None
    
    def set_destination(self, location):
        self.destination = location
        return f"Destination set to: {location}"
    
    def navigate(self):
        if self.destination:
            return f"Navigating from {self.current_location} to {self.destination}"
        return "No destination set"

# Composition approach
class Car:
    def __init__(self, make, model, engine_hp, fuel_type):
        self.make = make
        self.model = model
        self.engine = Engine(engine_hp, fuel_type)  # Composition
        self.gps = GPS()  # Composition
        self.speed = 0
    
    def start_car(self):
        return self.engine.start()
    
    def stop_car(self):
        return self.engine.stop()
    
    def accelerate(self, speed_increase):
        if self.engine.running:
            self.speed += speed_increase
            return f"Accelerating to {self.speed} mph"
        return "Cannot accelerate: engine not running"
    
    def navigate_to(self, destination):
        nav_result = self.gps.set_destination(destination)
        return f"{nav_result} - {self.gps.navigate()}"
    
    def get_car_status(self):
        return {
            "car": f"{self.make} {self.model}",
            "engine": self.engine.get_status(),
            "speed": f"{self.speed} mph",
            "navigation": self.gps.navigate()
        }

# Inheritance approach (for comparison)
class Vehicle:
    def __init__(self, make, model):
        self.make = make
        self.model = model
        self.speed = 0
    
    def accelerate(self, speed_increase):
        self.speed += speed_increase
        return f"Accelerating to {self.speed} mph"

class InheritanceCar(Vehicle):
    def __init__(self, make, model, engine_hp, fuel_type):
        super().__init__(make, model)
        self.engine_hp = engine_hp
        self.fuel_type = fuel_type
        self.engine_running = False
    
    def start_engine(self):
        self.engine_running = True
        return f"Engine started: {self.engine_hp}HP {self.fuel_type}"
    
    def accelerate(self, speed_increase):
        if self.engine_running:
            return super().accelerate(speed_increase)
        return "Cannot accelerate: engine not running"

def composition_vs_inheritance_demo():
    # Composition approach
    composition_car = Car("Toyota", "Camry", 200, "Gasoline")
    composition_car.start_car()
    composition_car.accelerate(30)
    composition_car.navigate_to("Downtown")
    
    # Inheritance approach
    inheritance_car = InheritanceCar("Honda", "Civic", 180, "Gasoline")
    inheritance_car.start_engine()
    inheritance_car.accelerate(25)
    
    return {
        "composition_status": composition_car.get_car_status(),
        "inheritance_speed": inheritance_car.speed,
        "composition_benefits": [
            "Engine and GPS can be reused in other classes",
            "Can easily replace Engine or GPS implementations",
            "No inheritance hierarchy complexity",
            "Each component can be tested independently"
        ],
        "inheritance_benefits": [
            "Simpler for simple hierarchies",
            "Direct access to parent methods",
            "Clear is-a relationship"
        ],
        "recommendation": "Favor composition for complex systems with reusable components"
    }

# Mixin classes
class LoggingMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logs = []
    
    def log(self, message):
        import time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        return log_entry
    
    def get_logs(self):
        return self.logs.copy()

class ValidationMixin:
    def validate_positive_number(self, value, field_name):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{field_name} must be a positive number")
        return True
    
    def validate_string(self, value, field_name, min_length=1):
        if not isinstance(value, str) or len(value) < min_length:
            raise ValueError(f"{field_name} must be a string with at least {min_length} characters")
        return True

class BankAccountWithMixins(LoggingMixin, ValidationMixin):
    def __init__(self, account_number, owner, initial_balance=0):
        super().__init__()  # Initialize mixins
        self.validate_string(account_number, "Account number", 5)
        self.validate_string(owner, "Owner name", 2)
        self.validate_positive_number(initial_balance, "Initial balance")
        
        self.account_number = account_number
        self.owner = owner
        self.balance = initial_balance
        
        self.log(f"Account created for {owner} with balance ${initial_balance}")
    
    def deposit(self, amount):
        self.validate_positive_number(amount, "Deposit amount")
        self.balance += amount
        self.log(f"Deposited ${amount}, new balance: ${self.balance}")
        return self.balance
    
    def withdraw(self, amount):
        self.validate_positive_number(amount, "Withdrawal amount")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
        self.log(f"Withdrew ${amount}, new balance: ${self.balance}")
        return self.balance

def mixin_demo():
    # Create account with mixins
    account = BankAccountWithMixins("ACC12345", "Alice Smith", 1000)
    
    # Perform operations (will trigger logging)
    account.deposit(500)
    account.withdraw(200)
    
    # Test validation
    try:
        account.withdraw(-100)  # Should fail validation
        validation_works = False
    except ValueError:
        validation_works = True
    
    return {
        "account_balance": account.balance,
        "transaction_logs": account.get_logs(),
        "validation_works": validation_works,
        "mixin_benefits": [
            "Reusable functionality across multiple classes",
            "Single responsibility principle",
            "Easy to combine multiple behaviors",
            "Can be tested independently"
        ]
    }

# Diamond problem and resolution
class A:
    def method(self):
        return "A"
    
    def common_method(self):
        return "A.common_method"

class B(A):
    def method(self):
        return "B"
    
    def common_method(self):
        return f"B.common_method -> {super().common_method()}"

class C(A):
    def method(self):
        return "C"
    
    def common_method(self):
        return f"C.common_method -> {super().common_method()}"

class D(B, C):  # Diamond inheritance
    def method(self):
        return f"D -> B: {B.method(self)}, C: {C.method(self)}"
    
    def common_method(self):
        return f"D.common_method -> {super().common_method()}"

def diamond_problem_demo():
    d = D()
    
    # Method resolution
    method_result = d.method()
    common_method_result = d.common_method()
    
    # MRO analysis
    mro_chain = [cls.__name__ for cls in D.__mro__]
    
    # Demonstrate super() behavior
    class DiamondAnalysis(B, C):
        def analyze_super(self):
            return {
                "direct_super": super().common_method(),
                "mro": [cls.__name__ for cls in self.__class__.__mro__]
            }
    
    analysis = DiamondAnalysis()
    super_analysis = analysis.analyze_super()
    
    return {
        "method_result": method_result,
        "common_method_result": common_method_result,
        "mro_chain": mro_chain,
        "super_analysis": super_analysis,
        "diamond_resolution": "Python uses C3 linearization algorithm for MRO"
    }

# Comprehensive testing
def run_all_inheritance_demos():
    """Execute all inheritance and polymorphism demonstrations"""
    demo_functions = [
        ('single_inheritance', single_inheritance_demo),
        ('multiple_inheritance', multiple_inheritance_demo),
        ('abstract_classes', abstract_classes_demo),
        ('duck_typing', duck_typing_demo),
        ('composition_vs_inheritance', composition_vs_inheritance_demo),
        ('mixins', mixin_demo),
        ('diamond_problem', diamond_problem_demo)
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
    print("=== Python Inheritance and Polymorphism Demo ===")
    
    # Run all demonstrations
    all_results = run_all_inheritance_demos()
    
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
    
    print("\n=== INHERITANCE CONCEPTS SUMMARY ===")
    
    concepts = {
        "Single Inheritance": "Child class inherits from one parent class",
        "Multiple Inheritance": "Child class inherits from multiple parent classes",
        "Method Resolution Order": "Algorithm determining method lookup order",
        "Method Overriding": "Child class provides specific implementation of parent method",
        "super()": "Function to call parent class methods",
        "Abstract Base Classes": "Classes that cannot be instantiated directly",
        "Duck Typing": "If it walks like a duck and quacks like a duck, it's a duck",
        "Composition": "Has-a relationship using object composition",
        "Mixins": "Classes designed to provide reusable functionality"
    }
    
    for concept, description in concepts.items():
        print(f"  {concept}: {description}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Favor composition over inheritance for complex relationships",
        "Use abstract base classes to define interfaces",
        "Implement __str__ and __repr__ in base classes",
        "Use super() to call parent methods properly",
        "Keep inheritance hierarchies shallow",
        "Use mixins for cross-cutting concerns",
        "Design for polymorphism with consistent interfaces",
        "Document inheritance relationships clearly"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Inheritance and Polymorphism Complete! ===")
    print("  Advanced OOP concepts and design patterns mastered")
