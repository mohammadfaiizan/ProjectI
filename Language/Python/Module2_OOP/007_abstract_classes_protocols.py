"""
Python Abstract Classes and Protocols: ABC Module, Protocols, and Interface Design
Implementation-focused with minimal comments, maximum functionality coverage
"""

import abc
from abc import ABC, abstractmethod, abstractproperty, abstractclassmethod, abstractstaticmethod
from typing import Protocol, runtime_checkable, Any, List, Dict, Optional
import weakref

# Basic abstract base classes
class Shape(ABC):
    """Abstract base class defining shape interface"""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def area(self):
        """Calculate and return the area"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate and return the perimeter"""
        pass
    
    @abstractproperty
    def description(self):
        """Return shape description"""
        pass
    
    def get_info(self):
        """Concrete method using abstract methods"""
        return {
            "name": self.name,
            "description": self.description,
            "area": self.area(),
            "perimeter": self.perimeter()
        }

class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)
    
    @property
    def description(self):
        return f"Rectangle with width {self.width} and height {self.height}"

class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius
    
    @property
    def description(self):
        return f"Circle with radius {self.radius}"

class IncompleteShape(Shape):
    def __init__(self):
        super().__init__("Incomplete")
    
    def area(self):
        return 0
    
    # Missing perimeter and description - cannot be instantiated

def basic_abc_demo():
    # Test concrete implementations
    rectangle = Rectangle(10, 5)
    circle = Circle(7)
    
    shapes_info = {
        "rectangle": rectangle.get_info(),
        "circle": circle.get_info()
    }
    
    # Test abstract class instantiation
    try:
        shape = Shape("Generic")
        abstract_instantiation = True
    except TypeError as e:
        abstract_instantiation = False
        abstract_error = str(e)
    
    # Test incomplete implementation
    try:
        incomplete = IncompleteShape()
        incomplete_instantiation = True
    except TypeError as e:
        incomplete_instantiation = False
        incomplete_error = str(e)
    
    # Test isinstance checks
    inheritance_checks = {
        "rectangle_is_shape": isinstance(rectangle, Shape),
        "circle_is_shape": isinstance(circle, Shape),
        "rectangle_is_abc": isinstance(rectangle, ABC)
    }
    
    return {
        "shapes_info": shapes_info,
        "abstract_class_test": {
            "can_instantiate": abstract_instantiation,
            "error": abstract_error if not abstract_instantiation else None
        },
        "incomplete_implementation": {
            "can_instantiate": incomplete_instantiation,
            "error": incomplete_error if not incomplete_instantiation else None
        },
        "inheritance_checks": inheritance_checks
    }

# Abstract methods with different decorators
class DataProcessor(ABC):
    """Abstract base class for data processing"""
    
    @abstractmethod
    def process_data(self, data):
        """Process raw data"""
        pass
    
    @abstractclassmethod
    def get_supported_formats(cls):
        """Return list of supported data formats"""
        pass
    
    @abstractstaticmethod
    def validate_format(data_format):
        """Validate if format is supported"""
        pass
    
    @abstractproperty
    def processor_name(self):
        """Name of the processor"""
        pass
    
    def process_batch(self, data_list):
        """Concrete method that uses abstract methods"""
        results = []
        for data in data_list:
            if self.validate_format(data.get('format', 'unknown')):
                processed = self.process_data(data)
                results.append(processed)
            else:
                results.append({"error": "Unsupported format"})
        return results

class JSONProcessor(DataProcessor):
    def process_data(self, data):
        return {"processed": f"JSON: {data}", "processor": self.processor_name}
    
    @classmethod
    def get_supported_formats(cls):
        return ["json", "application/json"]
    
    @staticmethod
    def validate_format(data_format):
        return data_format.lower() in ["json", "application/json"]
    
    @property
    def processor_name(self):
        return "JSON Processor v1.0"

class XMLProcessor(DataProcessor):
    def process_data(self, data):
        return {"processed": f"XML: {data}", "processor": self.processor_name}
    
    @classmethod
    def get_supported_formats(cls):
        return ["xml", "application/xml", "text/xml"]
    
    @staticmethod
    def validate_format(data_format):
        return data_format.lower() in ["xml", "application/xml", "text/xml"]
    
    @property
    def processor_name(self):
        return "XML Processor v2.0"

def abstract_methods_demo():
    # Create processors
    json_proc = JSONProcessor()
    xml_proc = XMLProcessor()
    
    # Test individual methods
    method_tests = {
        "json_formats": JSONProcessor.get_supported_formats(),
        "xml_formats": XMLProcessor.get_supported_formats(),
        "json_validation": JSONProcessor.validate_format("json"),
        "xml_validation": XMLProcessor.validate_format("json"),
        "json_name": json_proc.processor_name,
        "xml_name": xml_proc.processor_name
    }
    
    # Test batch processing
    test_data = [
        {"format": "json", "content": "{'key': 'value'}"},
        {"format": "xml", "content": "<root><item>value</item></root>"},
        {"format": "csv", "content": "col1,col2\nval1,val2"},
        {"format": "JSON", "content": "{'another': 'object'}"}
    ]
    
    batch_results = {
        "json_processor": json_proc.process_batch(test_data),
        "xml_processor": xml_proc.process_batch(test_data)
    }
    
    return {
        "method_tests": method_tests,
        "batch_processing": batch_results
    }

# Protocol-based typing (Python 3.8+)
@runtime_checkable
class Drawable(Protocol):
    """Protocol for drawable objects"""
    
    def draw(self) -> str:
        """Draw the object"""
        ...
    
    def get_coordinates(self) -> tuple:
        """Get object coordinates"""
        ...

@runtime_checkable
class Resizable(Protocol):
    """Protocol for resizable objects"""
    
    def resize(self, factor: float) -> None:
        """Resize the object by factor"""
        ...
    
    @property
    def size(self) -> float:
        """Current size of the object"""
        ...

class Point:
    """Point class that implements Drawable protocol without inheritance"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def draw(self):
        return f"Drawing point at ({self.x}, {self.y})"
    
    def get_coordinates(self):
        return (self.x, self.y)

class ResizableCircle:
    """Circle that implements both Drawable and Resizable protocols"""
    
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
    
    def draw(self):
        return f"Drawing circle at ({self.x}, {self.y}) with radius {self.radius}"
    
    def get_coordinates(self):
        return (self.x, self.y)
    
    def resize(self, factor):
        self.radius *= factor
    
    @property
    def size(self):
        return self.radius

class NotDrawable:
    """Class that doesn't implement any protocol"""
    
    def __init__(self, data):
        self.data = data
    
    def process(self):
        return f"Processing {self.data}"

def protocol_demo():
    # Create objects
    point = Point(10, 20)
    circle = ResizableCircle(5, 5, 3.0)
    not_drawable = NotDrawable("test")
    
    # Test protocol compliance
    protocol_checks = {
        "point_is_drawable": isinstance(point, Drawable),
        "circle_is_drawable": isinstance(circle, Drawable),
        "circle_is_resizable": isinstance(circle, Resizable),
        "not_drawable_is_drawable": isinstance(not_drawable, Drawable),
        "not_drawable_is_resizable": isinstance(not_drawable, Resizable)
    }
    
    # Test protocol methods
    drawable_objects = [obj for obj in [point, circle, not_drawable] if isinstance(obj, Drawable)]
    resizable_objects = [obj for obj in [point, circle, not_drawable] if isinstance(obj, Resizable)]
    
    drawing_results = [obj.draw() for obj in drawable_objects]
    
    # Test resizing
    original_size = circle.size
    circle.resize(2.0)
    new_size = circle.size
    
    return {
        "protocol_checks": protocol_checks,
        "drawing_results": drawing_results,
        "resizing_test": {
            "original_size": original_size,
            "new_size": new_size,
            "resize_worked": new_size == original_size * 2
        },
        "object_counts": {
            "drawable_objects": len(drawable_objects),
            "resizable_objects": len(resizable_objects)
        }
    }

# Duck typing vs Protocol typing
class Duck:
    def quack(self):
        return "Quack!"
    
    def fly(self):
        return "Flying like a duck"

class Robot:
    def quack(self):
        return "Beep! Simulated quack!"
    
    def fly(self):
        return "Jet propulsion activated"

class Car:
    def start(self):
        return "Engine started"
    
    def drive(self):
        return "Driving on road"

@runtime_checkable
class Quacker(Protocol):
    def quack(self) -> str: ...

@runtime_checkable
class Flyer(Protocol):
    def fly(self) -> str: ...

def duck_typing_function(obj):
    """Function using duck typing - tries to call methods"""
    results = {}
    
    # Try quack
    if hasattr(obj, 'quack') and callable(getattr(obj, 'quack')):
        results['quack'] = obj.quack()
    else:
        results['quack'] = "Cannot quack"
    
    # Try fly
    if hasattr(obj, 'fly') and callable(getattr(obj, 'fly')):
        results['fly'] = obj.fly()
    else:
        results['fly'] = "Cannot fly"
    
    return results

def protocol_typing_function(obj):
    """Function using protocol typing - checks protocol compliance"""
    results = {}
    
    if isinstance(obj, Quacker):
        results['quack'] = obj.quack()
    else:
        results['quack'] = "Not a Quacker protocol"
    
    if isinstance(obj, Flyer):
        results['fly'] = obj.fly()
    else:
        results['fly'] = "Not a Flyer protocol"
    
    return results

def duck_typing_vs_protocol_demo():
    duck = Duck()
    robot = Robot()
    car = Car()
    
    objects = [duck, robot, car]
    
    # Test duck typing approach
    duck_typing_results = {
        f"{obj.__class__.__name__}": duck_typing_function(obj)
        for obj in objects
    }
    
    # Test protocol typing approach
    protocol_typing_results = {
        f"{obj.__class__.__name__}": protocol_typing_function(obj)
        for obj in objects
    }
    
    # Protocol compliance checks
    compliance_checks = {}
    for obj in objects:
        obj_name = obj.__class__.__name__
        compliance_checks[obj_name] = {
            "is_quacker": isinstance(obj, Quacker),
            "is_flyer": isinstance(obj, Flyer)
        }
    
    return {
        "duck_typing_results": duck_typing_results,
        "protocol_typing_results": protocol_typing_results,
        "compliance_checks": compliance_checks,
        "comparison": {
            "duck_typing": "Runtime attribute checking, more flexible",
            "protocol_typing": "Static type checking support, clearer contracts"
        }
    }

# Multiple inheritance with ABCs
class Saveable(ABC):
    @abstractmethod
    def save(self, filename):
        pass

class Loadable(ABC):
    @abstractmethod
    def load(self, filename):
        pass

class Serializable(ABC):
    @abstractmethod
    def serialize(self):
        pass
    
    @abstractmethod
    def deserialize(self, data):
        pass

class Document(Saveable, Loadable, Serializable):
    def __init__(self, content=""):
        self.content = content
        self.metadata = {"created": "now", "version": 1}
    
    def save(self, filename):
        return f"Saved document to {filename}"
    
    def load(self, filename):
        self.content = f"Content loaded from {filename}"
        return self.content
    
    def serialize(self):
        return {
            "content": self.content,
            "metadata": self.metadata
        }
    
    def deserialize(self, data):
        self.content = data.get("content", "")
        self.metadata = data.get("metadata", {})
        return self

class PartialDocument(Saveable):
    def __init__(self, content=""):
        self.content = content
    
    def save(self, filename):
        return f"Partial save to {filename}"
    
    # Missing load, serialize, deserialize - cannot be instantiated

def multiple_abc_inheritance_demo():
    # Test complete implementation
    doc = Document("Hello World")
    
    # Test all abstract methods
    operations = {
        "save_result": doc.save("test.txt"),
        "load_result": doc.load("data.txt"),
        "serialize_result": doc.serialize(),
    }
    
    # Test deserialization
    test_data = {"content": "Deserialized content", "metadata": {"version": 2}}
    doc.deserialize(test_data)
    operations["deserialize_result"] = doc.content
    
    # Test isinstance checks
    inheritance_checks = {
        "is_saveable": isinstance(doc, Saveable),
        "is_loadable": isinstance(doc, Loadable),
        "is_serializable": isinstance(doc, Serializable),
        "is_document": isinstance(doc, Document)
    }
    
    # Test partial implementation
    try:
        partial = PartialDocument("Partial content")
        partial_instantiation = True
    except TypeError as e:
        partial_instantiation = False
        partial_error = str(e)
    
    return {
        "operations": operations,
        "inheritance_checks": inheritance_checks,
        "partial_implementation": {
            "can_instantiate": partial_instantiation,
            "error": partial_error if not partial_instantiation else None
        }
    }

# ABC registration for existing classes
class ThirdPartyClass:
    """Simulate a third-party class we can't modify"""
    
    def __init__(self, value):
        self.value = value
    
    def process(self):
        return f"Processing {self.value}"
    
    def get_result(self):
        return self.value * 2

class Processor(ABC):
    @abstractmethod
    def process(self):
        pass

# Register third-party class as implementing our ABC
Processor.register(ThirdPartyClass)

class NativeProcessor(Processor):
    def __init__(self, data):
        self.data = data
    
    def process(self):
        return f"Native processing {self.data}"

def abc_registration_demo():
    # Create instances
    third_party = ThirdPartyClass(42)
    native = NativeProcessor("test")
    
    # Test registration
    registration_checks = {
        "third_party_is_processor": isinstance(third_party, Processor),
        "native_is_processor": isinstance(native, Processor),
        "third_party_is_subclass": issubclass(ThirdPartyClass, Processor),
        "native_is_subclass": issubclass(NativeProcessor, Processor)
    }
    
    # Test that registered class still works
    third_party_result = third_party.process()
    native_result = native.process()
    
    # Test polymorphic behavior
    processors = [third_party, native]
    polymorphic_results = [proc.process() for proc in processors]
    
    return {
        "registration_checks": registration_checks,
        "processing_results": {
            "third_party": third_party_result,
            "native": native_result
        },
        "polymorphic_results": polymorphic_results,
        "registration_note": "Third-party class registered without inheritance"
    }

# Interface design patterns
class Repository(ABC):
    """Repository pattern interface"""
    
    @abstractmethod
    def save(self, entity):
        pass
    
    @abstractmethod
    def find_by_id(self, entity_id):
        pass
    
    @abstractmethod
    def find_all(self):
        pass
    
    @abstractmethod
    def delete(self, entity_id):
        pass

class InMemoryRepository(Repository):
    def __init__(self):
        self.data = {}
        self.next_id = 1
    
    def save(self, entity):
        entity_id = getattr(entity, 'id', self.next_id)
        if not hasattr(entity, 'id'):
            entity.id = entity_id
            self.next_id += 1
        
        self.data[entity_id] = entity
        return entity_id
    
    def find_by_id(self, entity_id):
        return self.data.get(entity_id)
    
    def find_all(self):
        return list(self.data.values())
    
    def delete(self, entity_id):
        return self.data.pop(entity_id, None)

class FileRepository(Repository):
    def __init__(self, filename):
        self.filename = filename
        self.data = {}
        self.next_id = 1
    
    def save(self, entity):
        entity_id = getattr(entity, 'id', self.next_id)
        if not hasattr(entity, 'id'):
            entity.id = entity_id
            self.next_id += 1
        
        self.data[entity_id] = entity
        return entity_id
    
    def find_by_id(self, entity_id):
        return self.data.get(entity_id)
    
    def find_all(self):
        return list(self.data.values())
    
    def delete(self, entity_id):
        return self.data.pop(entity_id, None)

class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def __repr__(self):
        return f"User(id={getattr(self, 'id', 'None')}, name='{self.name}', email='{self.email}')"

class UserService:
    def __init__(self, repository: Repository):
        self.repository = repository
    
    def create_user(self, name, email):
        user = User(name, email)
        user_id = self.repository.save(user)
        return user_id
    
    def get_user(self, user_id):
        return self.repository.find_by_id(user_id)
    
    def list_users(self):
        return self.repository.find_all()
    
    def remove_user(self, user_id):
        return self.repository.delete(user_id)

def interface_design_demo():
    # Test with in-memory repository
    memory_repo = InMemoryRepository()
    memory_service = UserService(memory_repo)
    
    # Create users
    user1_id = memory_service.create_user("Alice", "alice@email.com")
    user2_id = memory_service.create_user("Bob", "bob@email.com")
    
    # Test operations
    memory_results = {
        "user1": repr(memory_service.get_user(user1_id)),
        "user2": repr(memory_service.get_user(user2_id)),
        "all_users": [repr(user) for user in memory_service.list_users()],
        "user_count": len(memory_service.list_users())
    }
    
    # Test with file repository (simulated)
    file_repo = FileRepository("users.txt")
    file_service = UserService(file_repo)
    
    # Same operations with different repository
    user3_id = file_service.create_user("Charlie", "charlie@email.com")
    user4_id = file_service.create_user("Diana", "diana@email.com")
    
    file_results = {
        "user3": repr(file_service.get_user(user3_id)),
        "user4": repr(file_service.get_user(user4_id)),
        "all_users": [repr(user) for user in file_service.list_users()],
        "user_count": len(file_service.list_users())
    }
    
    # Test polymorphism
    repositories = [memory_repo, file_repo]
    repo_types = [type(repo).__name__ for repo in repositories]
    
    return {
        "memory_repository": memory_results,
        "file_repository": file_results,
        "polymorphism": {
            "repository_types": repo_types,
            "same_interface": "Both repositories implement Repository ABC"
        },
        "pattern_benefits": [
            "Dependency inversion - service depends on abstraction",
            "Easy testing with mock repositories",
            "Can switch storage implementations without changing service",
            "Clear contract definition through ABC"
        ]
    }

# Protocol with runtime checking and validation
@runtime_checkable
class Validator(Protocol):
    def validate(self, data: Any) -> bool: ...
    def get_error_message(self) -> str: ...

class EmailValidator:
    def validate(self, data):
        return isinstance(data, str) and "@" in data and "." in data
    
    def get_error_message(self):
        return "Invalid email format"

class PasswordValidator:
    def __init__(self, min_length=8):
        self.min_length = min_length
    
    def validate(self, data):
        return isinstance(data, str) and len(data) >= self.min_length
    
    def get_error_message(self):
        return f"Password must be at least {self.min_length} characters"

class NotAValidator:
    def check(self, data):
        return True

def validate_user_data(data, validators):
    """Function that works with any object implementing Validator protocol"""
    results = {}
    
    for field, value in data.items():
        field_results = []
        
        for validator in validators.get(field, []):
            if isinstance(validator, Validator):
                is_valid = validator.validate(value)
                field_results.append({
                    "validator": type(validator).__name__,
                    "valid": is_valid,
                    "error": None if is_valid else validator.get_error_message()
                })
            else:
                field_results.append({
                    "validator": type(validator).__name__,
                    "valid": False,
                    "error": "Not a valid Validator protocol implementation"
                })
        
        results[field] = field_results
    
    return results

def protocol_validation_demo():
    # Setup validators
    email_validator = EmailValidator()
    password_validator = PasswordValidator(10)
    not_validator = NotAValidator()
    
    # Test protocol compliance
    protocol_checks = {
        "email_is_validator": isinstance(email_validator, Validator),
        "password_is_validator": isinstance(password_validator, Validator),
        "not_validator_is_validator": isinstance(not_validator, Validator)
    }
    
    # Test data validation
    user_data = {
        "email": "user@example.com",
        "password": "short"
    }
    
    invalid_data = {
        "email": "invalid-email",
        "password": "verylongpassword123"
    }
    
    validators = {
        "email": [email_validator, not_validator],
        "password": [password_validator]
    }
    
    validation_results = {
        "valid_data": validate_user_data(user_data, validators),
        "invalid_data": validate_user_data(invalid_data, validators)
    }
    
    return {
        "protocol_checks": protocol_checks,
        "validation_results": validation_results
    }

# Comprehensive testing
def run_all_abc_protocol_demos():
    """Execute all abstract class and protocol demonstrations"""
    demo_functions = [
        ('basic_abc', basic_abc_demo),
        ('abstract_methods', abstract_methods_demo),
        ('protocols', protocol_demo),
        ('duck_typing_vs_protocol', duck_typing_vs_protocol_demo),
        ('multiple_abc_inheritance', multiple_abc_inheritance_demo),
        ('abc_registration', abc_registration_demo),
        ('interface_design', interface_design_demo),
        ('protocol_validation', protocol_validation_demo)
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
    print("=== Python Abstract Classes and Protocols Demo ===")
    
    # Run all demonstrations
    all_results = run_all_abc_protocol_demos()
    
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
    
    print("\n=== ABSTRACT BASE CLASSES (ABC) ===")
    
    abc_concepts = {
        "@abstractmethod": "Methods that must be implemented by subclasses",
        "@abstractproperty": "Properties that must be implemented by subclasses",
        "@abstractclassmethod": "Class methods that must be implemented",
        "@abstractstaticmethod": "Static methods that must be implemented",
        "ABC.register()": "Register existing classes as implementing ABC",
        "Cannot instantiate": "Abstract classes cannot be instantiated directly",
        "Enforcement": "Python enforces implementation of abstract methods"
    }
    
    for concept, description in abc_concepts.items():
        print(f"  {concept}: {description}")
    
    print("\n=== PROTOCOLS (PEP 544) ===")
    
    protocol_concepts = {
        "Structural Subtyping": "Type compatibility based on structure, not inheritance",
        "@runtime_checkable": "Enable isinstance() checks for protocols",
        "Duck Typing": "If it walks like a duck and quacks like a duck...",
        "Static Type Checking": "MyPy and other tools can check protocol compliance",
        "No Inheritance": "Classes don't need to inherit from protocols",
        "Flexible Design": "Protocols define what methods/attributes are needed",
        "Gradual Typing": "Can be used with or without type annotations"
    }
    
    for concept, description in protocol_concepts.items():
        print(f"  {concept}: {description}")
    
    print("\n=== DESIGN PATTERNS ===")
    
    patterns = [
        "Repository Pattern: Abstract data access with multiple implementations",
        "Strategy Pattern: Define family of algorithms with common interface",
        "Template Method: Define algorithm skeleton with abstract steps",
        "Factory Pattern: Create objects through abstract factory interface",
        "Observer Pattern: Define update interface for observing objects",
        "Command Pattern: Encapsulate requests as objects with common interface",
        "Adapter Pattern: Convert interface to work with existing code"
    ]
    
    for pattern in patterns:
        print(f"  • {pattern}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Use ABC for is-a relationships and strict contracts",
        "Use Protocols for duck typing and structural subtyping",
        "Prefer composition over inheritance when possible",
        "Design interfaces with single responsibility",
        "Use @runtime_checkable for protocols that need isinstance() checks",
        "Document expected behavior, not just method signatures",
        "Keep interfaces small and focused (Interface Segregation)",
        "Use ABC.register() for third-party classes you can't modify",
        "Consider both static and runtime type checking needs",
        "Test interface implementations thoroughly"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== ABC vs PROTOCOLS COMPARISON ===")
    
    comparison = {
        "Abstract Base Classes": [
            "Nominal typing (inheritance-based)",
            "Runtime enforcement of method implementation",
            "Can provide concrete methods alongside abstract ones",
            "Good for is-a relationships",
            "Traditional OOP approach"
        ],
        "Protocols": [
            "Structural typing (duck typing)",
            "Optional runtime checking with @runtime_checkable",
            "No inheritance required",
            "Good for can-do relationships",
            "More flexible and Pythonic"
        ]
    }
    
    for approach, characteristics in comparison.items():
        print(f"  {approach}:")
        for char in characteristics:
            print(f"    • {char}")
    
    print("\n=== Abstract Classes and Protocols Complete! ===")
    print("  Interface design and contract definition mastered")
