"""
Python Metaclasses Deep Dive: Advanced Class Creation and Metaclass Patterns
Implementation-focused with minimal comments, maximum functionality coverage
"""

import types
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Type, Optional, Callable
import weakref
from functools import wraps
import threading
import time

# Understanding class creation process
def class_creation_demo():
    """Demonstrate how classes are created in Python"""
    
    # Method 1: Normal class definition
    class NormalClass:
        def __init__(self, value):
            self.value = value
        
        def get_value(self):
            return self.value
    
    # Method 2: Using type() constructor
    def init_method(self, value):
        self.value = value
    
    def get_value_method(self):
        return self.value
    
    DynamicClass = type(
        'DynamicClass',  # name
        (object,),       # bases
        {                # namespace
            '__init__': init_method,
            'get_value': get_value_method,
            'class_type': 'dynamic'
        }
    )
    
    # Method 3: Programmatic class creation
    class_namespace = {}
    exec("""
class ExecutedClass:
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
    
    class_type = 'executed'
""", class_namespace)
    
    ExecutedClass = class_namespace['ExecutedClass']
    
    # Test all three methods
    normal_obj = NormalClass(10)
    dynamic_obj = DynamicClass(20)
    executed_obj = ExecutedClass(30)
    
    return {
        "normal_class": {
            "type": type(NormalClass).__name__,
            "value": normal_obj.get_value(),
            "mro": [cls.__name__ for cls in NormalClass.__mro__]
        },
        "dynamic_class": {
            "type": type(DynamicClass).__name__,
            "value": dynamic_obj.get_value(),
            "class_type": getattr(DynamicClass, 'class_type', None),
            "mro": [cls.__name__ for cls in DynamicClass.__mro__]
        },
        "executed_class": {
            "type": type(ExecutedClass).__name__,
            "value": executed_obj.get_value(),
            "class_type": getattr(ExecutedClass, 'class_type', None),
            "mro": [cls.__name__ for cls in ExecutedClass.__mro__]
        }
    }

# Basic metaclass patterns
class BasicMetaclass(type):
    """Basic metaclass demonstrating __new__ and __init__"""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        print(f"Creating class {name} with metaclass {mcs.__name__}")
        
        # Modify the namespace before class creation
        namespace['_created_by'] = mcs.__name__
        namespace['_creation_time'] = time.time()
        
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Post-creation modifications
        cls._instances = weakref.WeakSet()
        
        return cls
    
    def __init__(cls, name, bases, namespace, **kwargs):
        print(f"Initializing class {name}")
        super().__init__(name, bases, namespace)
        
        # Store metadata
        cls._metaclass_init_called = True
    
    def __call__(cls, *args, **kwargs):
        """Control instance creation"""
        print(f"Creating instance of {cls.__name__}")
        
        # Create instance normally
        instance = super().__call__(*args, **kwargs)
        
        # Track instance
        cls._instances.add(instance)
        
        return instance

class AttributeValidationMeta(type):
    """Metaclass that validates class attributes"""
    
    def __new__(mcs, name, bases, namespace):
        # Validate required attributes
        required_attrs = namespace.get('_required_attrs', [])
        for attr in required_attrs:
            if attr not in namespace:
                raise TypeError(f"Class {name} missing required attribute: {attr}")
        
        # Validate method signatures
        for attr_name, attr_value in namespace.items():
            if callable(attr_value) and not attr_name.startswith('_'):
                sig = inspect.signature(attr_value)
                if len(sig.parameters) == 0:
                    raise TypeError(f"Method {attr_name} must have at least one parameter (self)")
        
        return super().__new__(mcs, name, bases, namespace)

class SingletonMeta(type):
    """Metaclass implementing singleton pattern"""
    
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

def basic_metaclass_demo():
    """Demonstrate basic metaclass functionality"""
    
    # Test BasicMetaclass
    class TestClass(metaclass=BasicMetaclass):
        def __init__(self, value):
            self.value = value
        
        def get_info(self):
            return f"Value: {self.value}, Created by: {self._created_by}"
    
    # Test AttributeValidationMeta
    try:
        class ValidClass(metaclass=AttributeValidationMeta):
            _required_attrs = ['important_method']
            
            def important_method(self):
                return "This method is required"
        
        validation_success = True
    except TypeError as e:
        validation_success = False
        validation_error = str(e)
    
    try:
        class InvalidClass(metaclass=AttributeValidationMeta):
            _required_attrs = ['missing_method']
            
            def some_other_method(self):
                return "This is not the required method"
        
        validation_should_fail = False
    except TypeError as e:
        validation_should_fail = True
        validation_error_msg = str(e)
    
    # Test SingletonMeta
    class DatabaseConnection(metaclass=SingletonMeta):
        def __init__(self):
            self.connection_id = id(self)
    
    # Create instances
    test_obj1 = TestClass(100)
    test_obj2 = TestClass(200)
    
    db1 = DatabaseConnection()
    db2 = DatabaseConnection()
    
    return {
        "basic_metaclass": {
            "obj1_info": test_obj1.get_info(),
            "obj2_info": test_obj2.get_info(),
            "instances_tracked": len(TestClass._instances),
            "creation_time_exists": hasattr(TestClass, '_creation_time')
        },
        "validation_metaclass": {
            "valid_class_created": validation_success,
            "invalid_class_failed": validation_should_fail,
            "error_message": validation_error_msg if 'validation_error_msg' in locals() else None
        },
        "singleton_metaclass": {
            "same_instance": db1 is db2,
            "connection_ids": [db1.connection_id, db2.connection_id]
        }
    }

# Advanced metaclass patterns
class RegistryMeta(type):
    """Metaclass that maintains a registry of all classes"""
    
    registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Register the class
        category = namespace.get('_category', 'default')
        if category not in mcs.registry:
            mcs.registry[category] = []
        mcs.registry[category].append(cls)
        
        return cls
    
    @classmethod
    def get_classes(mcs, category=None):
        """Get registered classes by category"""
        if category:
            return mcs.registry.get(category, [])
        return mcs.registry

class AutoPropertyMeta(type):
    """Metaclass that automatically creates properties for attributes"""
    
    def __new__(mcs, name, bases, namespace):
        auto_properties = namespace.get('_auto_properties', [])
        
        for prop_name in auto_properties:
            private_name = f'_{prop_name}'
            
            def make_getter(private_attr):
                def getter(self):
                    return getattr(self, private_attr, None)
                return getter
            
            def make_setter(private_attr):
                def setter(self, value):
                    setattr(self, private_attr, value)
                return setter
            
            # Create property
            prop = property(
                make_getter(private_name),
                make_setter(private_name),
                doc=f"Auto-generated property for {prop_name}"
            )
            
            namespace[prop_name] = prop
        
        return super().__new__(mcs, name, bases, namespace)

class MethodDecoratorMeta(type):
    """Metaclass that automatically decorates methods"""
    
    def __new__(mcs, name, bases, namespace):
        decorators = namespace.get('_method_decorators', {})
        
        for attr_name, attr_value in list(namespace.items()):
            if callable(attr_value) and not attr_name.startswith('_'):
                for decorator in decorators.get(attr_name, []):
                    attr_value = decorator(attr_value)
                    namespace[attr_name] = attr_value
        
        return super().__new__(mcs, name, bases, namespace)

def timing_decorator(func):
    """Decorator that measures method execution time"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        end = time.time()
        if not hasattr(self, '_method_times'):
            self._method_times = {}
        self._method_times[func.__name__] = end - start
        return result
    return wrapper

def logging_decorator(func):
    """Decorator that logs method calls"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_method_calls'):
            self._method_calls = []
        self._method_calls.append(f"{func.__name__}({args}, {kwargs})")
        return func(self, *args, **kwargs)
    return wrapper

def advanced_metaclass_demo():
    """Demonstrate advanced metaclass patterns"""
    
    # Test RegistryMeta
    class Animal(metaclass=RegistryMeta):
        _category = 'animals'
    
    class Dog(Animal):
        pass
    
    class Cat(Animal):
        pass
    
    class Vehicle(metaclass=RegistryMeta):
        _category = 'vehicles'
    
    class Car(Vehicle):
        pass
    
    # Test AutoPropertyMeta
    class Person(metaclass=AutoPropertyMeta):
        _auto_properties = ['name', 'age', 'email']
        
        def __init__(self, name, age, email):
            self.name = name
            self.age = age
            self.email = email
    
    # Test MethodDecoratorMeta
    class Calculator(metaclass=MethodDecoratorMeta):
        _method_decorators = {
            'add': [timing_decorator, logging_decorator],
            'multiply': [timing_decorator]
        }
        
        def add(self, a, b):
            time.sleep(0.01)  # Simulate work
            return a + b
        
        def multiply(self, a, b):
            time.sleep(0.005)  # Simulate work
            return a * b
        
        def divide(self, a, b):
            return a / b if b != 0 else None
    
    # Test instances
    person = Person("Alice", 30, "alice@example.com")
    calc = Calculator()
    
    # Test calculator methods
    result1 = calc.add(5, 3)
    result2 = calc.multiply(4, 7)
    result3 = calc.divide(10, 2)
    
    return {
        "registry_metaclass": {
            "animals": [cls.__name__ for cls in RegistryMeta.get_classes('animals')],
            "vehicles": [cls.__name__ for cls in RegistryMeta.get_classes('vehicles')],
            "all_categories": list(RegistryMeta.registry.keys())
        },
        "auto_property_metaclass": {
            "person_name": person.name,
            "person_age": person.age,
            "has_properties": all(hasattr(Person, prop) for prop in ['name', 'age', 'email'])
        },
        "method_decorator_metaclass": {
            "add_result": result1,
            "multiply_result": result2,
            "divide_result": result3,
            "method_times": getattr(calc, '_method_times', {}),
            "method_calls": getattr(calc, '_method_calls', [])
        }
    }

# Metaclass inheritance and conflicts
class MetaA(type):
    """First metaclass"""
    def __new__(mcs, name, bases, namespace):
        namespace['from_meta_a'] = True
        return super().__new__(mcs, name, bases, namespace)

class MetaB(type):
    """Second metaclass"""
    def __new__(mcs, name, bases, namespace):
        namespace['from_meta_b'] = True
        return super().__new__(mcs, name, bases, namespace)

class MetaAB(MetaA, MetaB):
    """Combined metaclass resolving conflicts"""
    def __new__(mcs, name, bases, namespace):
        namespace['from_meta_ab'] = True
        return super().__new__(mcs, name, bases, namespace)

class ConflictResolver(type):
    """Metaclass that resolves multiple inheritance conflicts"""
    
    def __new__(mcs, name, bases, namespace):
        # Check for metaclass conflicts
        metaclasses = set()
        for base in bases:
            metaclasses.add(type(base))
        
        if len(metaclasses) > 1 and type not in metaclasses:
            print(f"Warning: Multiple metaclasses detected for {name}: {[m.__name__ for m in metaclasses]}")
        
        return super().__new__(mcs, name, bases, namespace)

def metaclass_inheritance_demo():
    """Demonstrate metaclass inheritance patterns"""
    
    # Single metaclass inheritance
    class BaseA(metaclass=MetaA):
        pass
    
    class DerivedA(BaseA):
        pass
    
    # Multiple metaclass inheritance (resolved)
    class BaseAB(metaclass=MetaAB):
        pass
    
    # Conflict detection
    class Base1(metaclass=MetaA):
        pass
    
    class Base2(metaclass=MetaB):
        pass
    
    try:
        # This would normally cause a metaclass conflict
        class Conflicted(Base1, Base2):
            pass
        conflict_resolved = False
    except TypeError as e:
        conflict_resolved = True
        conflict_error = str(e)
    
    # Using conflict resolver
    class ResolvedBase1(Base1, metaclass=ConflictResolver):
        pass
    
    return {
        "single_inheritance": {
            "base_a_attrs": [attr for attr in dir(BaseA) if attr.startswith('from_meta')],
            "derived_a_attrs": [attr for attr in dir(DerivedA) if attr.startswith('from_meta')]
        },
        "combined_metaclass": {
            "base_ab_attrs": [attr for attr in dir(BaseAB) if attr.startswith('from_meta')]
        },
        "conflict_detection": {
            "conflict_occurred": conflict_resolved,
            "error_message": conflict_error if 'conflict_error' in locals() else None
        }
    }

# Dynamic class modification
class DynamicModificationMeta(type):
    """Metaclass that allows dynamic class modification"""
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Add modification capabilities
        cls._dynamic_methods = {}
        cls._dynamic_properties = {}
        
        return cls
    
    def add_method(cls, name, method):
        """Add a method dynamically"""
        setattr(cls, name, method)
        cls._dynamic_methods[name] = method
    
    def add_property(cls, name, getter=None, setter=None):
        """Add a property dynamically"""
        prop = property(getter, setter)
        setattr(cls, name, prop)
        cls._dynamic_properties[name] = prop
    
    def remove_method(cls, name):
        """Remove a method dynamically"""
        if hasattr(cls, name):
            delattr(cls, name)
            cls._dynamic_methods.pop(name, None)

class VersionedMeta(type):
    """Metaclass that tracks class versions"""
    
    _versions = {}
    
    def __new__(mcs, name, bases, namespace):
        # Track version
        version = namespace.get('__version__', '1.0.0')
        
        cls = super().__new__(mcs, name, bases, namespace)
        
        if name not in mcs._versions:
            mcs._versions[name] = []
        mcs._versions[name].append((version, cls))
        
        return cls
    
    @classmethod
    def get_version_history(mcs, class_name):
        """Get version history for a class"""
        return mcs._versions.get(class_name, [])

def dynamic_modification_demo():
    """Demonstrate dynamic class modification"""
    
    # Test dynamic modification
    class DynamicClass(metaclass=DynamicModificationMeta):
        def __init__(self, value):
            self.value = value
        
        def original_method(self):
            return f"Original: {self.value}"
    
    # Add methods dynamically
    def new_method(self):
        return f"Dynamic: {self.value * 2}"
    
    def another_method(self, multiplier):
        return f"Another: {self.value * multiplier}"
    
    DynamicClass.add_method('new_method', new_method)
    DynamicClass.add_method('another_method', another_method)
    
    # Add property dynamically
    def get_doubled(self):
        return self.value * 2
    
    def set_doubled(self, val):
        self.value = val // 2
    
    DynamicClass.add_property('doubled', get_doubled, set_doubled)
    
    # Test versioned classes
    class VersionedClass1(metaclass=VersionedMeta):
        __version__ = '1.0.0'
        
        def method_v1(self):
            return "Version 1.0.0"
    
    class VersionedClass1(metaclass=VersionedMeta):
        __version__ = '1.1.0'
        
        def method_v1(self):
            return "Version 1.1.0"
        
        def new_method(self):
            return "New in 1.1.0"
    
    class VersionedClass1(metaclass=VersionedMeta):
        __version__ = '2.0.0'
        
        def method_v2(self):
            return "Version 2.0.0"
    
    # Test instances
    dynamic_obj = DynamicClass(10)
    versioned_obj = VersionedClass1()
    
    return {
        "dynamic_modification": {
            "original_method": dynamic_obj.original_method(),
            "new_method": dynamic_obj.new_method(),
            "another_method": dynamic_obj.another_method(3),
            "doubled_property": dynamic_obj.doubled,
            "dynamic_methods": list(DynamicClass._dynamic_methods.keys()),
            "dynamic_properties": list(DynamicClass._dynamic_properties.keys())
        },
        "versioned_classes": {
            "version_history": VersionedMeta.get_version_history('VersionedClass1'),
            "current_version": getattr(VersionedClass1, '__version__', 'unknown'),
            "current_methods": [m for m in dir(VersionedClass1) if not m.startswith('_') and callable(getattr(VersionedClass1, m))]
        }
    }

# Metaclass-based design patterns
class ObserverMeta(type):
    """Metaclass implementing observer pattern"""
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Add observer functionality
        cls._observers = []
        
        # Wrap methods that should notify observers
        notify_methods = namespace.get('_notify_on', [])
        for method_name in notify_methods:
            if hasattr(cls, method_name):
                original_method = getattr(cls, method_name)
                
                def make_notifying_method(orig_method, method_name):
                    @wraps(orig_method)
                    def notifying_method(self, *args, **kwargs):
                        result = orig_method(self, *args, **kwargs)
                        self._notify_observers(method_name, args, kwargs, result)
                        return result
                    return notifying_method
                
                setattr(cls, method_name, make_notifying_method(original_method, method_name))
        
        return cls
    
    def add_observer(cls, observer):
        """Add an observer to the class"""
        cls._observers.append(observer)
    
    def remove_observer(cls, observer):
        """Remove an observer from the class"""
        if observer in cls._observers:
            cls._observers.remove(observer)

class FactoryMeta(type):
    """Metaclass implementing factory pattern"""
    
    _product_registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Register products
        product_type = namespace.get('_product_type')
        if product_type:
            mcs._product_registry[product_type] = cls
        
        return cls
    
    def create_product(mcs, product_type, *args, **kwargs):
        """Factory method to create products"""
        if product_type in mcs._product_registry:
            return mcs._product_registry[product_type](*args, **kwargs)
        raise ValueError(f"Unknown product type: {product_type}")
    
    def get_available_products(mcs):
        """Get list of available product types"""
        return list(mcs._product_registry.keys())

class CommandMeta(type):
    """Metaclass implementing command pattern"""
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Add command functionality
        cls._command_history = []
        cls._undo_stack = []
        
        # Wrap execute method
        if 'execute' in namespace:
            original_execute = namespace['execute']
            
            def tracked_execute(self, *args, **kwargs):
                result = original_execute(self, *args, **kwargs)
                self.__class__._command_history.append((self, args, kwargs))
                if hasattr(self, 'undo'):
                    self.__class__._undo_stack.append(self)
                return result
            
            cls.execute = tracked_execute
        
        return cls
    
    def undo_last(cls):
        """Undo the last command"""
        if cls._undo_stack:
            command = cls._undo_stack.pop()
            if hasattr(command, 'undo'):
                return command.undo()
        return None

def design_patterns_demo():
    """Demonstrate metaclass-based design patterns"""
    
    # Observer pattern
    class Subject(metaclass=ObserverMeta):
        _notify_on = ['update_value', 'set_name']
        
        def __init__(self, name, value):
            self.name = name
            self.value = value
        
        def update_value(self, new_value):
            old_value = self.value
            self.value = new_value
            return f"Updated from {old_value} to {new_value}"
        
        def set_name(self, new_name):
            old_name = self.name
            self.name = new_name
            return f"Name changed from {old_name} to {new_name}"
        
        def _notify_observers(self, method_name, args, kwargs, result):
            for observer in self._observers:
                observer.notify(self, method_name, args, kwargs, result)
    
    class Observer:
        def __init__(self, name):
            self.name = name
            self.notifications = []
        
        def notify(self, subject, method_name, args, kwargs, result):
            self.notifications.append({
                'method': method_name,
                'args': args,
                'kwargs': kwargs,
                'result': result
            })
    
    # Factory pattern
    class Product(metaclass=FactoryMeta):
        pass
    
    class ConcreteProductA(Product):
        _product_type = 'A'
        
        def __init__(self, config):
            self.config = config
            self.type = 'A'
    
    class ConcreteProductB(Product):
        _product_type = 'B'
        
        def __init__(self, config):
            self.config = config
            self.type = 'B'
    
    # Command pattern
    class Command(metaclass=CommandMeta):
        pass
    
    class IncrementCommand(Command):
        def __init__(self, target, amount):
            self.target = target
            self.amount = amount
            self.old_value = None
        
        def execute(self):
            self.old_value = self.target.value
            self.target.value += self.amount
            return f"Incremented by {self.amount}"
        
        def undo(self):
            if self.old_value is not None:
                self.target.value = self.old_value
                return f"Undone increment, restored to {self.old_value}"
    
    # Test observer pattern
    subject = Subject("TestSubject", 100)
    observer1 = Observer("Observer1")
    observer2 = Observer("Observer2")
    
    Subject.add_observer(observer1)
    Subject.add_observer(observer2)
    
    result1 = subject.update_value(200)
    result2 = subject.set_name("NewSubject")
    
    # Test factory pattern
    product_a = Product.create_product('A', {'setting': 'value_a'})
    product_b = Product.create_product('B', {'setting': 'value_b'})
    
    # Test command pattern
    class Target:
        def __init__(self, value):
            self.value = value
    
    target = Target(10)
    cmd1 = IncrementCommand(target, 5)
    cmd2 = IncrementCommand(target, 3)
    
    cmd1.execute()
    cmd2.execute()
    
    undo_result = Command.undo_last()
    
    return {
        "observer_pattern": {
            "observer1_notifications": len(observer1.notifications),
            "observer2_notifications": len(observer2.notifications),
            "last_notification": observer1.notifications[-1] if observer1.notifications else None
        },
        "factory_pattern": {
            "product_a_type": product_a.type,
            "product_b_type": product_b.type,
            "available_products": Product.get_available_products()
        },
        "command_pattern": {
            "target_final_value": target.value,
            "command_history_length": len(Command._command_history),
            "undo_result": undo_result
        }
    }

# Real-world metaclass examples
class ORMMeta(type):
    """Simplified ORM metaclass"""
    
    _model_registry = {}
    
    def __new__(mcs, name, bases, namespace):
        # Don't process the base Model class
        if name == 'Model':
            return super().__new__(mcs, name, bases, namespace)
        
        # Extract field definitions
        fields = {}
        for key, value in list(namespace.items()):
            if isinstance(value, Field):
                fields[key] = value
                value.name = key
        
        namespace['_fields'] = fields
        namespace['_table_name'] = namespace.get('_table_name', name.lower())
        
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Register model
        mcs._model_registry[name] = cls
        
        return cls
    
    def get_model(mcs, name):
        """Get a registered model by name"""
        return mcs._model_registry.get(name)

class Field:
    """Simple field descriptor for ORM"""
    
    def __init__(self, field_type, default=None, nullable=True):
        self.field_type = field_type
        self.default = default
        self.nullable = nullable
        self.name = None
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, self.default)
    
    def __set__(self, obj, value):
        if value is None and not self.nullable:
            raise ValueError(f"Field {self.name} cannot be null")
        
        if value is not None and not isinstance(value, self.field_type):
            try:
                value = self.field_type(value)
            except (ValueError, TypeError):
                raise TypeError(f"Field {self.name} must be of type {self.field_type.__name__}")
        
        obj.__dict__[self.name] = value

class ConfigurationMeta(type):
    """Metaclass for configuration classes with validation"""
    
    def __new__(mcs, name, bases, namespace):
        # Process configuration schema
        config_schema = namespace.get('_schema', {})
        
        # Create property descriptors for each config option
        for key, schema in config_schema.items():
            prop = mcs._create_config_property(key, schema)
            namespace[key] = prop
        
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Add validation method
        cls._validate_config = mcs._create_validator(config_schema)
        
        return cls
    
    @staticmethod
    def _create_config_property(key, schema):
        """Create a property with validation"""
        private_key = f'_{key}'
        
        def getter(self):
            return getattr(self, private_key, schema.get('default'))
        
        def setter(self, value):
            # Type validation
            expected_type = schema.get('type', str)
            if not isinstance(value, expected_type):
                raise TypeError(f"{key} must be of type {expected_type.__name__}")
            
            # Range validation
            if 'min' in schema and value < schema['min']:
                raise ValueError(f"{key} must be >= {schema['min']}")
            if 'max' in schema and value > schema['max']:
                raise ValueError(f"{key} must be <= {schema['max']}")
            
            # Choice validation
            if 'choices' in schema and value not in schema['choices']:
                raise ValueError(f"{key} must be one of {schema['choices']}")
            
            setattr(self, private_key, value)
        
        return property(getter, setter, doc=schema.get('doc', f'Configuration option: {key}'))
    
    @staticmethod
    def _create_validator(schema):
        """Create a validation method for the entire configuration"""
        def validate_config(self):
            errors = []
            for key, config in schema.items():
                try:
                    value = getattr(self, key)
                    if config.get('required', False) and value is None:
                        errors.append(f"{key} is required")
                except Exception as e:
                    errors.append(f"{key}: {e}")
            
            return errors
        
        return validate_config

def real_world_demo():
    """Demonstrate real-world metaclass applications"""
    
    # ORM example
    class Model(metaclass=ORMMeta):
        def __init__(self, **kwargs):
            for field_name, field in self._fields.items():
                value = kwargs.get(field_name, field.default)
                setattr(self, field_name, value)
        
        def to_dict(self):
            return {name: getattr(self, name) for name in self._fields}
    
    class User(Model):
        _table_name = 'users'
        
        id = Field(int, nullable=False)
        name = Field(str, nullable=False)
        email = Field(str)
        age = Field(int, default=0)
    
    class Product(Model):
        id = Field(int, nullable=False)
        name = Field(str, nullable=False)
        price = Field(float, default=0.0)
    
    # Configuration example
    class DatabaseConfig(metaclass=ConfigurationMeta):
        _schema = {
            'host': {
                'type': str,
                'default': 'localhost',
                'required': True,
                'doc': 'Database host address'
            },
            'port': {
                'type': int,
                'default': 5432,
                'min': 1,
                'max': 65535,
                'doc': 'Database port number'
            },
            'database': {
                'type': str,
                'required': True,
                'doc': 'Database name'
            },
            'ssl_mode': {
                'type': str,
                'default': 'prefer',
                'choices': ['disable', 'allow', 'prefer', 'require'],
                'doc': 'SSL connection mode'
            }
        }
        
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Test ORM
    user1 = User(id=1, name="Alice", email="alice@example.com", age=30)
    user2 = User(id=2, name="Bob", email="bob@example.com")
    
    product1 = Product(id=1, name="Laptop", price=999.99)
    
    # Test configuration
    config = DatabaseConfig(
        host="db.example.com",
        port=5432,
        database="myapp",
        ssl_mode="require"
    )
    
    validation_errors = config._validate_config()
    
    try:
        bad_config = DatabaseConfig(port=70000)  # Invalid port
        port_error = False
    except ValueError as e:
        port_error = str(e)
    
    return {
        "orm_demo": {
            "user1_dict": user1.to_dict(),
            "user2_dict": user2.to_dict(),
            "product1_dict": product1.to_dict(),
            "registered_models": list(ORMMeta._model_registry.keys()),
            "user_table_name": User._table_name,
            "user_fields": list(User._fields.keys())
        },
        "configuration_demo": {
            "config_values": {
                "host": config.host,
                "port": config.port,
                "database": config.database,
                "ssl_mode": config.ssl_mode
            },
            "validation_errors": validation_errors,
            "port_validation_error": port_error
        }
    }

# Comprehensive testing
def run_all_metaclass_demos():
    """Execute all metaclass demonstrations"""
    demo_functions = [
        ('class_creation', class_creation_demo),
        ('basic_metaclasses', basic_metaclass_demo),
        ('advanced_metaclasses', advanced_metaclass_demo),
        ('metaclass_inheritance', metaclass_inheritance_demo),
        ('dynamic_modification', dynamic_modification_demo),
        ('design_patterns', design_patterns_demo),
        ('real_world_examples', real_world_demo)
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
    print("=== Python Metaclasses Deep Dive Demo ===")
    
    # Run all demonstrations
    all_results = run_all_metaclass_demos()
    
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
    
    print("\n=== METACLASS CONCEPTS ===")
    
    concepts = {
        "Metaclass": "A class whose instances are classes themselves",
        "type": "The default metaclass, can be used to create classes dynamically",
        "__new__": "Controls class creation, called before __init__",
        "__init__": "Initializes the created class",
        "__call__": "Controls instance creation when class is called",
        "Class Creation Process": "1. Collect base classes 2. Prepare namespace 3. Execute body 4. Create class",
        "MRO": "Method Resolution Order determines metaclass inheritance",
        "Metaclass Conflicts": "Occur when multiple base classes have different metaclasses"
    }
    
    for concept, description in concepts.items():
        print(f"  {concept}: {description}")
    
    print("\n=== METACLASS USE CASES ===")
    
    use_cases = {
        "Singleton Pattern": "Ensure only one instance of a class exists",
        "Registry Pattern": "Automatically register classes when they're created",
        "Validation": "Validate class attributes and methods at creation time",
        "ORM Implementation": "Automatically create database mappings and methods",
        "Configuration Systems": "Add validation and type checking to config classes",
        "Design Pattern Implementation": "Observer, Factory, Command patterns",
        "API Frameworks": "Automatically generate REST endpoints from classes",
        "Aspect-Oriented Programming": "Add cross-cutting concerns like logging, timing"
    }
    
    for use_case, description in use_cases.items():
        print(f"  {use_case}: {description}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Use metaclasses sparingly - they make code harder to understand",
        "Consider class decorators as simpler alternatives",
        "Document metaclass behavior thoroughly",
        "Handle metaclass conflicts explicitly when using multiple inheritance",
        "Use descriptive names for metaclasses (ending with 'Meta')",
        "Keep metaclass logic simple and focused",
        "Test metaclass behavior extensively",
        "Consider the maintenance burden before using metaclasses",
        "Use __init_subclass__ instead of metaclasses when possible (Python 3.6+)",
        "Provide clear error messages for validation metaclasses"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Metaclasses Deep Dive Complete! ===")
    print("  Advanced metaclass patterns and class creation mastered")
