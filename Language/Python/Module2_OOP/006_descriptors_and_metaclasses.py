"""
Python Descriptors and Metaclasses: Advanced Object Model and Class Creation
Implementation-focused with minimal comments, maximum functionality coverage
"""

import weakref
from typing import Any, Dict, Type, Optional, Callable

# Basic descriptor protocol
class BasicDescriptor:
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.name = None
    
    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to class attribute"""
        self.name = name
        self.private_name = f"_{name}"
    
    def __get__(self, obj, objtype=None):
        """Called when attribute is accessed"""
        if obj is None:
            return self
        return getattr(obj, self.private_name, self.value)
    
    def __set__(self, obj, value):
        """Called when attribute is set"""
        setattr(obj, self.private_name, value)
    
    def __delete__(self, obj):
        """Called when attribute is deleted"""
        if hasattr(obj, self.private_name):
            delattr(obj, self.private_name)

class TrackingDescriptor:
    def __init__(self):
        self.data = weakref.WeakKeyDictionary()
        self.access_count = 0
        self.modification_count = 0
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        self.access_count += 1
        return self.data.get(obj, "Not set")
    
    def __set__(self, obj, value):
        self.modification_count += 1
        self.data[obj] = value
    
    def get_stats(self):
        return {
            "access_count": self.access_count,
            "modification_count": self.modification_count,
            "active_objects": len(self.data)
        }

class DescriptorExample:
    basic = BasicDescriptor("default")
    tracked = TrackingDescriptor()
    
    def __init__(self, name):
        self.name = name

def basic_descriptor_demo():
    # Create instances
    obj1 = DescriptorExample("Object1")
    obj2 = DescriptorExample("Object2")
    
    # Test basic descriptor
    obj1.basic = "value1"
    obj2.basic = "value2"
    
    basic_results = {
        "obj1_basic": obj1.basic,
        "obj2_basic": obj2.basic,
        "default_for_new": DescriptorExample("new").basic
    }
    
    # Test tracking descriptor
    obj1.tracked = "tracked_value1"
    obj2.tracked = "tracked_value2"
    
    # Multiple accesses
    access1 = obj1.tracked
    access2 = obj2.tracked
    access3 = obj1.tracked
    
    tracking_stats = DescriptorExample.tracked.get_stats()
    
    return {
        "basic_descriptor": basic_results,
        "tracking_stats": tracking_stats,
        "accessed_values": [access1, access2, access3]
    }

# Data vs non-data descriptors
class NonDataDescriptor:
    """Descriptor with only __get__ method"""
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.func(obj)

class DataDescriptor:
    """Descriptor with __get__ and __set__ methods"""
    def __init__(self):
        self.data = weakref.WeakKeyDictionary()
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.data.get(obj, "No data")
    
    def __set__(self, obj, value):
        self.data[obj] = f"Data: {value}"

class DescriptorPrecedence:
    # Data descriptor (higher precedence)
    data_desc = DataDescriptor()
    
    # Non-data descriptor (lower precedence)
    @NonDataDescriptor
    def computed_value(self):
        return f"Computed for {id(self)}"
    
    def __init__(self):
        # Instance attribute that might override descriptors
        self.instance_attr = "instance value"

def descriptor_precedence_demo():
    obj = DescriptorPrecedence()
    
    # Data descriptor behavior
    obj.data_desc = "test_value"
    data_desc_result = obj.data_desc
    
    # Non-data descriptor behavior
    computed_result = obj.computed_value
    
    # Instance attribute override attempt
    obj.__dict__["data_desc"] = "instance_override"  # Won't work - data descriptor has precedence
    obj.__dict__["computed_value"] = "instance_override"  # Will work - non-data descriptor
    
    after_override = {
        "data_desc": obj.data_desc,  # Still uses descriptor
        "computed_value": obj.computed_value  # Now uses instance attribute
    }
    
    return {
        "initial_results": {
            "data_desc": data_desc_result,
            "computed_value": computed_result
        },
        "after_override": after_override,
        "precedence_rules": {
            "data_descriptors": "Always take precedence over instance attributes",
            "non_data_descriptors": "Instance attributes take precedence if they exist"
        }
    }

# Advanced descriptors with validation
class ValidatedDescriptor:
    def __init__(self, validator=None, transform=None):
        self.validator = validator
        self.transform = transform
        self.data = weakref.WeakKeyDictionary()
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.data.get(obj)
    
    def __set__(self, obj, value):
        if self.validator:
            self.validator(value, self.name)
        
        if self.transform:
            value = self.transform(value)
        
        self.data[obj] = value

def positive_validator(value, field_name):
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field_name} must be a positive number")

def string_validator(value, field_name):
    if not isinstance(value, str) or len(value) < 1:
        raise ValueError(f"{field_name} must be a non-empty string")

def uppercase_transform(value):
    return value.upper() if isinstance(value, str) else value

class Person:
    name = ValidatedDescriptor(string_validator, uppercase_transform)
    age = ValidatedDescriptor(positive_validator)
    salary = ValidatedDescriptor(positive_validator)
    
    def __init__(self, name, age, salary):
        self.name = name
        self.age = age
        self.salary = salary
    
    def __repr__(self):
        return f"Person(name='{self.name}', age={self.age}, salary={self.salary})"

def validated_descriptor_demo():
    # Create valid person
    person = Person("john doe", 30, 50000)
    
    valid_creation = {
        "name": person.name,  # Should be uppercase
        "age": person.age,
        "salary": person.salary
    }
    
    # Test validation
    validation_tests = {}
    
    # Valid updates
    person.name = "jane smith"
    person.salary = 60000
    
    validation_tests["valid_updates"] = {
        "name": person.name,
        "salary": person.salary
    }
    
    # Invalid updates
    try:
        person.age = -5
        validation_tests["negative_age"] = "Failed to validate"
    except ValueError as e:
        validation_tests["negative_age"] = "Validation worked"
    
    try:
        person.name = ""
        validation_tests["empty_name"] = "Failed to validate"
    except ValueError as e:
        validation_tests["empty_name"] = "Validation worked"
    
    return {
        "valid_creation": valid_creation,
        "validation_tests": validation_tests,
        "final_person": repr(person)
    }

# Metaclass basics
class BasicMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        """Called when creating a new class"""
        # Modify class creation
        namespace['created_by_metaclass'] = True
        namespace['class_id'] = id(namespace)
        
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        return cls
    
    def __init__(cls, name, bases, namespace, **kwargs):
        """Called after class is created"""
        super().__init__(name, bases, namespace)
        cls.creation_info = {
            "name": name,
            "bases": [base.__name__ for base in bases],
            "methods": [key for key, value in namespace.items() if callable(value)]
        }

class MetaExample(metaclass=BasicMeta):
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value

def basic_metaclass_demo():
    # Create instance
    obj = MetaExample(42)
    
    # Check metaclass modifications
    metaclass_features = {
        "created_by_metaclass": hasattr(MetaExample, 'created_by_metaclass'),
        "has_class_id": hasattr(MetaExample, 'class_id'),
        "creation_info": getattr(MetaExample, 'creation_info', {}),
        "metaclass_type": type(MetaExample).__name__
    }
    
    # Normal functionality still works
    functionality_test = {
        "value": obj.get_value(),
        "instance_type": type(obj).__name__
    }
    
    return {
        "metaclass_features": metaclass_features,
        "normal_functionality": functionality_test
    }

# Advanced metaclass for automatic method decoration
class AutoLogMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Automatically wrap methods with logging
        for key, value in list(namespace.items()):
            if callable(value) and not key.startswith('_'):
                namespace[key] = mcs.add_logging(value, key)
        
        # Add method call statistics
        namespace['_method_calls'] = {}
        
        return super().__new__(mcs, name, bases, namespace)
    
    @staticmethod
    def add_logging(func, func_name):
        def wrapper(self, *args, **kwargs):
            # Track method calls
            if not hasattr(self, '_method_calls'):
                self._method_calls = {}
            
            if func_name not in self._method_calls:
                self._method_calls[func_name] = 0
            
            self._method_calls[func_name] += 1
            
            # Call original method
            result = func(self, *args, **kwargs)
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

class Calculator(metaclass=AutoLogMeta):
    def __init__(self, name):
        self.name = name
    
    def add(self, a, b):
        """Add two numbers"""
        return a + b
    
    def multiply(self, a, b):
        """Multiply two numbers"""
        return a * b
    
    def divide(self, a, b):
        """Divide two numbers"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def get_method_stats(self):
        """Get method call statistics"""
        return getattr(self, '_method_calls', {})

def auto_log_metaclass_demo():
    calc = Calculator("MyCalculator")
    
    # Perform operations
    results = {
        "add_result": calc.add(10, 5),
        "multiply_result": calc.multiply(4, 3),
        "divide_result": calc.divide(20, 4)
    }
    
    # Multiple calls
    calc.add(1, 2)
    calc.add(3, 4)
    calc.multiply(2, 5)
    
    # Get statistics
    method_stats = calc.get_method_stats()
    
    return {
        "calculation_results": results,
        "method_call_stats": method_stats,
        "auto_logging_working": len(method_stats) > 0
    }

# Metaclass for singleton pattern
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        """Control instance creation"""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
    def clear_instances(cls):
        """Clear singleton instances (for testing)"""
        if cls in cls._instances:
            del cls._instances[cls]

class DatabaseConnection(metaclass=SingletonMeta):
    def __init__(self, connection_string="default"):
        if not hasattr(self, 'initialized'):
            self.connection_string = connection_string
            self.connection_count = 0
            self.initialized = True
    
    def connect(self):
        self.connection_count += 1
        return f"Connected: {self.connection_string}"
    
    def get_info(self):
        return {
            "connection_string": self.connection_string,
            "connection_count": self.connection_count,
            "object_id": id(self)
        }

def singleton_metaclass_demo():
    # Create multiple "instances"
    db1 = DatabaseConnection("connection1")
    db2 = DatabaseConnection("connection2")  # Should be same instance
    db3 = DatabaseConnection("connection3")  # Should be same instance
    
    # Test singleton behavior
    singleton_test = {
        "same_instance": db1 is db2 is db3,
        "db1_id": id(db1),
        "db2_id": id(db2),
        "db3_id": id(db3)
    }
    
    # Operations on different references
    db1.connect()
    db2.connect()
    db3.connect()
    
    # All should show same state
    final_info = db1.get_info()
    
    return {
        "singleton_verification": singleton_test,
        "shared_state": final_info,
        "connection_count": final_info["connection_count"]  # Should be 3
    }

# Metaclass for attribute validation
class ValidatedMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Find validation rules
        validations = {}
        for key, value in list(namespace.items()):
            if key.endswith('_validator'):
                field_name = key[:-10]  # Remove '_validator' suffix
                validations[field_name] = value
                del namespace[key]  # Remove validator from class
        
        # Store validations
        namespace['_validations'] = validations
        
        # Create class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Add automatic validation to __setattr__
        original_setattr = cls.__setattr__
        
        def validated_setattr(self, name, value):
            if name in self._validations:
                self._validations[name](value)
            original_setattr(self, name, value)
        
        cls.__setattr__ = validated_setattr
        
        return cls

def validate_positive(value):
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("Value must be positive")

def validate_string(value):
    if not isinstance(value, str) or len(value) < 1:
        raise ValueError("Value must be non-empty string")

class Product(metaclass=ValidatedMeta):
    name_validator = validate_string
    price_validator = validate_positive
    quantity_validator = validate_positive
    
    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity
    
    def __repr__(self):
        return f"Product('{self.name}', ${self.price}, qty={self.quantity})"

def validated_metaclass_demo():
    # Create valid product
    product = Product("Laptop", 999.99, 5)
    
    valid_creation = {
        "name": product.name,
        "price": product.price,
        "quantity": product.quantity
    }
    
    # Test validation
    validation_tests = {}
    
    # Valid update
    product.price = 1199.99
    validation_tests["valid_price_update"] = product.price
    
    # Invalid updates
    try:
        product.price = -100
        validation_tests["negative_price"] = "Failed to validate"
    except ValueError:
        validation_tests["negative_price"] = "Validation worked"
    
    try:
        product.name = ""
        validation_tests["empty_name"] = "Failed to validate"
    except ValueError:
        validation_tests["empty_name"] = "Validation worked"
    
    return {
        "valid_creation": valid_creation,
        "validation_tests": validation_tests,
        "validations_available": hasattr(Product, '_validations')
    }

# Descriptors vs Properties comparison
class PropertyVersion:
    def __init__(self, value=0):
        self._value = value
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, val):
        if val < 0:
            raise ValueError("Value cannot be negative")
        self._value = val

class DescriptorVersion:
    value = ValidatedDescriptor(lambda v, n: None if v >= 0 else (_ for _ in ()).throw(ValueError("Value cannot be negative")))
    
    def __init__(self, value=0):
        self.value = value

def descriptor_vs_property_demo():
    # Property version
    prop_obj = PropertyVersion(10)
    prop_obj.value = 20
    prop_value = prop_obj.value
    
    # Descriptor version
    desc_obj = DescriptorVersion(10)
    desc_obj.value = 20
    desc_value = desc_obj.value
    
    # Test validation for both
    prop_validation = False
    try:
        prop_obj.value = -5
    except ValueError:
        prop_validation = True
    
    desc_validation = False
    try:
        desc_obj.value = -5
    except ValueError:
        desc_validation = True
    
    # Test reusability
    class AnotherClass:
        value = ValidatedDescriptor(lambda v, n: None if v >= 0 else (_ for _ in ()).throw(ValueError("Value cannot be negative")))
    
    another_obj = AnotherClass()
    another_obj.value = 30
    
    return {
        "property_version": {
            "value": prop_value,
            "validation_works": prop_validation
        },
        "descriptor_version": {
            "value": desc_value,
            "validation_works": desc_validation
        },
        "reusability": {
            "another_class_value": another_obj.value,
            "descriptor_advantage": "Descriptors can be reused across multiple classes"
        },
        "comparison": {
            "properties": "Good for single class, simple validation",
            "descriptors": "Better for reusable behavior across classes"
        }
    }

# Metaclass for automatic method registration
class RegistryMeta(type):
    registry = {}
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Register all public methods
        methods = {}
        for key, value in namespace.items():
            if callable(value) and not key.startswith('_'):
                methods[key] = value
        
        mcs.registry[name] = {
            'class': cls,
            'methods': methods,
            'base_classes': [base.__name__ for base in bases]
        }
        
        return cls
    
    @classmethod
    def get_registry(mcs):
        return mcs.registry.copy()
    
    @classmethod
    def find_classes_with_method(mcs, method_name):
        results = []
        for class_name, info in mcs.registry.items():
            if method_name in info['methods']:
                results.append(class_name)
        return results

class ServiceA(metaclass=RegistryMeta):
    def process(self):
        return "ServiceA processing"
    
    def validate(self):
        return "ServiceA validation"

class ServiceB(metaclass=RegistryMeta):
    def process(self):
        return "ServiceB processing"
    
    def transform(self):
        return "ServiceB transformation"

class ServiceC(ServiceA):  # Inheritance
    def process(self):
        return "ServiceC processing"
    
    def export(self):
        return "ServiceC export"

def registry_metaclass_demo():
    # Get registry information
    registry = RegistryMeta.get_registry()
    
    # Find classes with specific methods
    process_classes = RegistryMeta.find_classes_with_method('process')
    validate_classes = RegistryMeta.find_classes_with_method('validate')
    
    # Test services
    service_results = {
        "ServiceA": ServiceA().process(),
        "ServiceB": ServiceB().process(),
        "ServiceC": ServiceC().process()
    }
    
    return {
        "registry_info": {
            class_name: {
                'methods': list(info['methods'].keys()),
                'bases': info['base_classes']
            }
            for class_name, info in registry.items()
        },
        "method_search": {
            "classes_with_process": process_classes,
            "classes_with_validate": validate_classes
        },
        "service_results": service_results
    }

# Custom descriptor for caching
class CachedProperty:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.__doc__ = func.__doc__
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        
        # Check if cached value exists
        cache_name = f"_cached_{self.name}"
        if hasattr(obj, cache_name):
            return getattr(obj, cache_name)
        
        # Compute and cache the value
        value = self.func(obj)
        setattr(obj, cache_name, value)
        return value
    
    def __set__(self, obj, value):
        # Allow setting cached value
        cache_name = f"_cached_{self.name}"
        setattr(obj, cache_name, value)
    
    def __delete__(self, obj):
        # Clear cached value
        cache_name = f"_cached_{self.name}"
        if hasattr(obj, cache_name):
            delattr(obj, cache_name)

class ExpensiveCalculations:
    def __init__(self, base_value):
        self.base_value = base_value
        self.calculation_count = 0
    
    @CachedProperty
    def expensive_computation(self):
        """Simulate expensive computation"""
        self.calculation_count += 1
        result = sum(i ** 2 for i in range(self.base_value))
        return result
    
    @CachedProperty
    def another_computation(self):
        """Another expensive computation"""
        self.calculation_count += 1
        return self.base_value ** 3

def cached_property_demo():
    calc = ExpensiveCalculations(1000)
    
    # First access - should compute
    result1 = calc.expensive_computation
    count_after_first = calc.calculation_count
    
    # Second access - should use cache
    result2 = calc.expensive_computation
    count_after_second = calc.calculation_count
    
    # Third access - should still use cache
    result3 = calc.expensive_computation
    count_after_third = calc.calculation_count
    
    # Access different property
    another_result = calc.another_computation
    count_after_another = calc.calculation_count
    
    # Clear cache and access again
    del calc.expensive_computation
    result4 = calc.expensive_computation
    count_after_clear = calc.calculation_count
    
    return {
        "caching_behavior": {
            "first_access": result1,
            "second_access": result2,
            "third_access": result3,
            "results_equal": result1 == result2 == result3
        },
        "calculation_counts": {
            "after_first": count_after_first,
            "after_second": count_after_second,
            "after_third": count_after_third,
            "after_another": count_after_another,
            "after_clear": count_after_clear
        },
        "cache_effectiveness": {
            "cached_accesses": count_after_third == 1,
            "cache_cleared": count_after_clear == 3  # Should be 2 calculations total
        }
    }

# Comprehensive testing
def run_all_descriptor_metaclass_demos():
    """Execute all descriptor and metaclass demonstrations"""
    demo_functions = [
        ('basic_descriptors', basic_descriptor_demo),
        ('descriptor_precedence', descriptor_precedence_demo),
        ('validated_descriptors', validated_descriptor_demo),
        ('basic_metaclass', basic_metaclass_demo),
        ('auto_log_metaclass', auto_log_metaclass_demo),
        ('singleton_metaclass', singleton_metaclass_demo),
        ('validated_metaclass', validated_metaclass_demo),
        ('descriptor_vs_property', descriptor_vs_property_demo),
        ('registry_metaclass', registry_metaclass_demo),
        ('cached_property', cached_property_demo)
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
    print("=== Python Descriptors and Metaclasses Demo ===")
    
    # Run all demonstrations
    all_results = run_all_descriptor_metaclass_demos()
    
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
    
    print("\n=== DESCRIPTOR PROTOCOL ===")
    
    descriptor_info = {
        "__get__(self, obj, type)": "Called when attribute is accessed",
        "__set__(self, obj, value)": "Called when attribute is set",
        "__delete__(self, obj)": "Called when attribute is deleted",
        "__set_name__(self, owner, name)": "Called when descriptor is assigned to class",
        "Data Descriptors": "Have __get__ and __set__ methods",
        "Non-Data Descriptors": "Have only __get__ method",
        "Precedence": "Data descriptors > Instance dict > Non-data descriptors > Class dict"
    }
    
    for protocol, description in descriptor_info.items():
        print(f"  {protocol}: {description}")
    
    print("\n=== METACLASS CONCEPTS ===")
    
    metaclass_info = {
        "Metaclass": "A class whose instances are classes",
        "__new__": "Controls class creation, called before class exists",
        "__init__": "Initializes class after creation",
        "__call__": "Controls instance creation when class is called",
        "type()": "The default metaclass for all classes",
        "Use Cases": "Automatic method decoration, singletons, validation, registration",
        "When to Use": "When you need to modify class creation or behavior"
    }
    
    for concept, description in metaclass_info.items():
        print(f"  {concept}: {description}")
    
    print("\n=== ADVANCED PATTERNS ===")
    
    patterns = [
        "Validated attributes with descriptors for reusable validation",
        "Cached properties for expensive computations",
        "Automatic method logging with metaclasses",
        "Singleton pattern implementation with metaclasses",
        "Class registration systems for plugin architectures",
        "Attribute access tracking and statistics",
        "Dynamic method injection and modification",
        "Property vs descriptor trade-offs for different use cases"
    ]
    
    for pattern in patterns:
        print(f"  • {pattern}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Use descriptors for reusable attribute behavior across classes",
        "Prefer properties for simple single-class attribute control",
        "Use metaclasses sparingly - they make code harder to understand",
        "Document descriptor and metaclass behavior clearly",
        "Consider using __set_name__ for descriptor self-configuration",
        "Use weakref.WeakKeyDictionary for descriptor data storage",
        "Implement __delete__ in descriptors for complete protocol support",
        "Test edge cases like None objects and class access patterns"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Descriptors and Metaclasses Complete! ===")
    print("  Advanced Python object model and class creation mastered")
