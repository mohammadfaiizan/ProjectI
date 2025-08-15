"""
Python Classes and Objects: Fundamentals of Object-Oriented Programming
Implementation-focused with minimal comments, maximum functionality coverage
"""

import gc
import sys
import weakref
from typing import Any, Dict, List, Optional

# Basic class definition and instantiation
class Person:
    species = "Homo sapiens"  # Class attribute
    
    def __init__(self, name, age, email=None):
        self.name = name  # Instance attribute
        self.age = age
        self.email = email
        self.created_at = id(self)  # Unique identifier
    
    def greet(self):
        return f"Hello, I'm {self.name}"
    
    def celebrate_birthday(self):
        self.age += 1
        return f"{self.name} is now {self.age} years old"

def basic_class_operations():
    # Creating instances
    person1 = Person("Alice", 30, "alice@email.com")
    person2 = Person("Bob", 25)
    
    # Accessing attributes and methods
    basic_ops = {
        "person1_name": person1.name,
        "person1_greet": person1.greet(),
        "person2_birthday": person2.celebrate_birthday(),
        "class_attribute": Person.species,
        "instance_vs_class": person1.species == Person.species
    }
    
    # Dynamic attribute assignment
    person1.city = "New York"  # Dynamic attribute
    person2.skills = ["Python", "JavaScript"]
    
    basic_ops.update({
        "dynamic_city": getattr(person1, "city", "Not set"),
        "dynamic_skills": getattr(person2, "skills", []),
        "has_email": hasattr(person1, "email"),
        "has_phone": hasattr(person1, "phone")
    })
    
    return basic_ops

# Advanced class with multiple attribute types
class BankAccount:
    bank_name = "PyBank"
    interest_rate = 0.02
    
    def __init__(self, account_number, owner, initial_balance=0):
        self.account_number = account_number
        self.owner = owner
        self._balance = initial_balance  # Protected attribute
        self.__pin = "0000"  # Private attribute (name mangling)
        self.transaction_history = []
        self._add_transaction("OPEN", initial_balance)
    
    def _add_transaction(self, transaction_type, amount):
        # Protected method
        self.transaction_history.append({
            "type": transaction_type,
            "amount": amount,
            "balance": self._balance
        })
    
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            self._add_transaction("DEPOSIT", amount)
            return self._balance
        raise ValueError("Deposit amount must be positive")
    
    def withdraw(self, amount):
        if amount > 0 and amount <= self._balance:
            self._balance -= amount
            self._add_transaction("WITHDRAWAL", -amount)
            return self._balance
        raise ValueError("Invalid withdrawal amount")
    
    def get_balance(self):
        return self._balance
    
    def set_pin(self, old_pin, new_pin):
        if self.__pin == old_pin:
            self.__pin = new_pin
            return True
        return False
    
    def verify_pin(self, pin):
        return self.__pin == pin

def advanced_class_operations():
    # Creating and using advanced class
    account = BankAccount("ACC001", "Alice", 1000)
    
    # Perform operations
    account.deposit(500)
    account.withdraw(200)
    
    # Accessing different attribute types
    operations = {
        "final_balance": account.get_balance(),
        "transaction_count": len(account.transaction_history),
        "owner": account.owner,
        "bank_name": account.bank_name
    }
    
    # Demonstrate protected and private access
    operations.update({
        "protected_balance": account._balance,  # Accessible but discouraged
        "private_pin_access": hasattr(account, "_BankAccount__pin"),  # Name mangling
        "pin_verification": account.verify_pin("0000"),
        "pin_change": account.set_pin("0000", "1234")
    })
    
    return operations

# Class vs instance attributes demonstration
class Counter:
    total_instances = 0  # Class attribute
    
    def __init__(self, initial_value=0):
        self.value = initial_value  # Instance attribute
        Counter.total_instances += 1
        self.instance_id = Counter.total_instances
    
    def increment(self):
        self.value += 1
    
    @classmethod
    def get_total_instances(cls):
        return cls.total_instances
    
    def reset(self):
        self.value = 0

def class_vs_instance_attributes():
    # Create multiple instances
    counter1 = Counter(10)
    counter2 = Counter(20)
    counter3 = Counter()
    
    # Modify instance attributes
    counter1.increment()
    counter2.increment()
    
    # Modify class attribute
    Counter.total_instances += 5  # Direct modification
    
    attribute_demo = {
        "counter1_value": counter1.value,
        "counter2_value": counter2.value,
        "counter3_value": counter3.value,
        "total_instances": Counter.get_total_instances(),
        "instance_ids": [counter1.instance_id, counter2.instance_id, counter3.instance_id]
    }
    
    # Demonstrate attribute access patterns
    counter1.class_attr = "instance override"  # Creates instance attribute
    attribute_demo.update({
        "counter1_class_attr": counter1.class_attr,
        "counter2_class_attr": getattr(counter2, "class_attr", "Not set"),
        "class_total_via_instance": counter2.total_instances
    })
    
    return attribute_demo

# Object identity and equality
class Product:
    def __init__(self, name, price, category):
        self.name = name
        self.price = price
        self.category = category
    
    def __eq__(self, other):
        if isinstance(other, Product):
            return (self.name == other.name and 
                   self.price == other.price and 
                   self.category == other.category)
        return False
    
    def __hash__(self):
        return hash((self.name, self.price, self.category))
    
    def __repr__(self):
        return f"Product('{self.name}', {self.price}, '{self.category}')"

def object_identity_equality():
    # Create products
    product1 = Product("Laptop", 999.99, "Electronics")
    product2 = Product("Laptop", 999.99, "Electronics")
    product3 = product1  # Same object reference
    
    identity_tests = {
        "product1_id": id(product1),
        "product2_id": id(product2),
        "product3_id": id(product3),
        "identity_same": product1 is product3,
        "identity_different": product1 is product2,
        "equality_same": product1 == product2,  # Uses __eq__
        "equality_reference": product1 == product3
    }
    
    # Hash and set operations
    product_set = {product1, product2, product3}
    identity_tests.update({
        "set_size": len(product_set),  # Should be 1 due to __hash__ and __eq__
        "product1_hash": hash(product1),
        "product2_hash": hash(product2),
        "hashes_equal": hash(product1) == hash(product2)
    })
    
    return identity_tests

# Object lifecycle and garbage collection
class ResourceManager:
    active_resources = []
    
    def __init__(self, name):
        self.name = name
        self.resource_id = id(self)
        ResourceManager.active_resources.append(self)
    
    def __del__(self):
        # Destructor (finalizer)
        try:
            ResourceManager.active_resources.remove(self)
        except ValueError:
            pass  # Already removed
    
    def close(self):
        # Explicit cleanup
        if self in ResourceManager.active_resources:
            ResourceManager.active_resources.remove(self)

def object_lifecycle_demo():
    initial_count = len(ResourceManager.active_resources)
    
    # Create resources
    resource1 = ResourceManager("Resource1")
    resource2 = ResourceManager("Resource2")
    
    after_creation = len(ResourceManager.active_resources)
    
    # Explicit cleanup
    resource1.close()
    after_close = len(ResourceManager.active_resources)
    
    # Delete reference
    resource_id = resource2.resource_id
    del resource2
    gc.collect()  # Force garbage collection
    
    after_deletion = len(ResourceManager.active_resources)
    
    lifecycle_info = {
        "initial_count": initial_count,
        "after_creation": after_creation,
        "after_close": after_close,
        "after_deletion": after_deletion,
        "resource_cleanup": after_deletion < after_creation
    }
    
    return lifecycle_info

# Weak references
class Node:
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []
    
    def add_child(self, child):
        child.parent = self  # Strong reference
        self.children.append(child)

class WeakNode:
    def __init__(self, value):
        self.value = value
        self._parent = None
        self.children = []
    
    def add_child(self, child):
        child._parent = weakref.ref(self)  # Weak reference
        self.children.append(child)
    
    @property
    def parent(self):
        if self._parent is not None:
            return self._parent()  # Call weak reference
        return None

def weak_reference_demo():
    # Strong reference cycle (potential memory leak)
    parent = Node("Parent")
    child = Node("Child")
    parent.add_child(child)
    
    strong_refs = {
        "parent_children": len(parent.children),
        "child_has_parent": child.parent is not None,
        "circular_reference": parent.children[0].parent is parent
    }
    
    # Weak reference solution
    weak_parent = WeakNode("WeakParent")
    weak_child = WeakNode("WeakChild")
    weak_parent.add_child(weak_child)
    
    weak_refs = {
        "weak_parent_children": len(weak_parent.children),
        "weak_child_has_parent": weak_child.parent is not None,
        "weak_parent_accessible": weak_child.parent.value if weak_child.parent else None
    }
    
    # Test weak reference cleanup
    weak_parent_id = id(weak_parent)
    del weak_parent
    gc.collect()
    
    weak_refs.update({
        "parent_after_deletion": weak_child.parent,
        "weak_ref_broken": weak_child.parent is None
    })
    
    return {
        "strong_references": strong_refs,
        "weak_references": weak_refs
    }

# Class documentation and introspection
class DocumentedClass:
    """
    A well-documented class demonstrating Python documentation conventions.
    
    This class serves as an example of proper documentation practices,
    including class docstrings, method docstrings, and attribute documentation.
    
    Attributes:
        class_attribute (str): A class-level attribute for demonstration
        instance_count (int): Counter for created instances
    """
    
    class_attribute = "documented"
    instance_count = 0
    
    def __init__(self, name: str, value: int = 0):
        """
        Initialize a DocumentedClass instance.
        
        Args:
            name (str): The name identifier for this instance
            value (int, optional): Initial value. Defaults to 0.
        """
        self.name = name
        self.value = value
        DocumentedClass.instance_count += 1
    
    def process_data(self, data: List[Any]) -> Dict[str, Any]:
        """
        Process input data and return analysis.
        
        Args:
            data (List[Any]): Input data to process
            
        Returns:
            Dict[str, Any]: Analysis results containing processed information
            
        Raises:
            ValueError: If data is empty or invalid
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        return {
            "length": len(data),
            "types": list(set(type(item).__name__ for item in data)),
            "processed_by": self.name
        }

def class_introspection_demo():
    # Create documented instance
    doc_instance = DocumentedClass("Processor", 42)
    
    # Introspection
    introspection = {
        "class_name": doc_instance.__class__.__name__,
        "class_module": doc_instance.__class__.__module__,
        "class_doc": doc_instance.__class__.__doc__[:100] + "..." if doc_instance.__class__.__doc__ else None,
        "instance_dict": dict(doc_instance.__dict__),
        "class_dict_keys": list(DocumentedClass.__dict__.keys())
    }
    
    # Method introspection
    method = doc_instance.process_data
    introspection.update({
        "method_name": method.__name__,
        "method_doc": method.__doc__[:50] + "..." if method.__doc__ else None,
        "method_annotations": getattr(method, "__annotations__", {}),
        "callable_check": callable(method)
    })
    
    # Test the documented method
    test_data = [1, "hello", 3.14, True]
    result = doc_instance.process_data(test_data)
    introspection["method_result"] = result
    
    return introspection

# Dynamic class creation and modification
def dynamic_class_operations():
    # Create class dynamically
    def dynamic_init(self, name):
        self.name = name
    
    def dynamic_method(self):
        return f"Dynamic method called by {self.name}"
    
    # Create class using type()
    DynamicClass = type(
        'DynamicClass',  # Class name
        (object,),       # Base classes
        {               # Class dictionary
            '__init__': dynamic_init,
            'dynamic_method': dynamic_method,
            'class_var': 'dynamically created'
        }
    )
    
    # Use dynamic class
    dynamic_obj = DynamicClass("DynamicObject")
    
    # Add methods to existing class
    def new_method(self):
        return f"New method for {self.name}"
    
    DynamicClass.new_method = new_method
    
    dynamic_ops = {
        "dynamic_class_name": DynamicClass.__name__,
        "dynamic_obj_name": dynamic_obj.name,
        "dynamic_method_result": dynamic_obj.dynamic_method(),
        "new_method_result": dynamic_obj.new_method(),
        "class_var": DynamicClass.class_var
    }
    
    # Modify existing class
    original_method_count = len([attr for attr in dir(Person) if callable(getattr(Person, attr))])
    
    def get_age_group(self):
        if self.age < 18:
            return "Minor"
        elif self.age < 65:
            return "Adult"
        else:
            return "Senior"
    
    Person.get_age_group = get_age_group
    
    # Test modified class
    person = Person("TestPerson", 30)
    modified_method_count = len([attr for attr in dir(Person) if callable(getattr(Person, attr))])
    
    dynamic_ops.update({
        "original_method_count": original_method_count,
        "modified_method_count": modified_method_count,
        "new_method_works": person.get_age_group(),
        "method_added": modified_method_count > original_method_count
    })
    
    return dynamic_ops

# Memory and performance considerations
class MemoryEfficientClass:
    __slots__ = ['name', 'value', 'timestamp']  # Memory optimization
    
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.timestamp = id(self)

class RegularClass:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.timestamp = id(self)

def memory_performance_demo():
    # Create instances for comparison
    efficient_obj = MemoryEfficientClass("efficient", 42)
    regular_obj = RegularClass("regular", 42)
    
    # Memory usage comparison
    memory_comparison = {
        "efficient_size": sys.getsizeof(efficient_obj),
        "regular_size": sys.getsizeof(regular_obj),
        "has_dict_efficient": hasattr(efficient_obj, '__dict__'),
        "has_dict_regular": hasattr(regular_obj, '__dict__'),
        "slots_defined": hasattr(MemoryEfficientClass, '__slots__')
    }
    
    # Try to add dynamic attribute (should fail for __slots__)
    try:
        regular_obj.dynamic_attr = "works"
        regular_dynamic = True
    except:
        regular_dynamic = False
    
    try:
        efficient_obj.dynamic_attr = "fails"
        efficient_dynamic = True
    except AttributeError:
        efficient_dynamic = False
    
    memory_comparison.update({
        "regular_dynamic_attr": regular_dynamic,
        "efficient_dynamic_attr": efficient_dynamic,
        "memory_savings": memory_comparison["regular_size"] - memory_comparison["efficient_size"]
    })
    
    return memory_comparison

# Comprehensive testing
def run_all_class_demos():
    """Execute all class and object demonstrations"""
    demo_functions = [
        ('basic_operations', basic_class_operations),
        ('advanced_operations', advanced_class_operations),
        ('class_vs_instance', class_vs_instance_attributes),
        ('identity_equality', object_identity_equality),
        ('object_lifecycle', object_lifecycle_demo),
        ('weak_references', weak_reference_demo),
        ('class_introspection', class_introspection_demo),
        ('dynamic_classes', dynamic_class_operations),
        ('memory_performance', memory_performance_demo)
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
    print("=== Python Classes and Objects Demo ===")
    
    # Run all demonstrations
    all_results = run_all_class_demos()
    
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
                elif isinstance(value, list) and len(value) > 3:
                    print(f"  {key}: {value[:3]}... (showing first 3)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {data}")
    
    print("\n=== OOP FUNDAMENTALS SUMMARY ===")
    
    concepts = {
        "Class Definition": "Blueprint for creating objects with attributes and methods",
        "Instance Creation": "Objects created from classes with unique identity",
        "Attribute Types": "Class attributes (shared) vs instance attributes (unique)",
        "Encapsulation": "Public, protected (_), and private (__) attribute conventions",
        "Object Identity": "is operator for identity, == for equality comparison",
        "Garbage Collection": "Automatic memory management and object cleanup",
        "Introspection": "Runtime examination of objects and classes",
        "Dynamic Modification": "Adding/modifying classes and objects at runtime"
    }
    
    for concept, description in concepts.items():
        print(f"  {concept}: {description}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Use __init__ for object initialization",
        "Follow naming conventions (PEP 8)",
        "Document classes and methods with docstrings",
        "Use __slots__ for memory-critical applications",
        "Implement __repr__ and __str__ for better debugging",
        "Handle resource cleanup with __del__ or context managers",
        "Prefer composition over inheritance when appropriate",
        "Use weak references to avoid circular reference issues"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Classes and Objects Complete! ===")
    print("  Foundation concepts for Object-Oriented Programming established")
