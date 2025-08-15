"""
Python Special Methods (Magic Methods): Dunder Methods for Operator Overloading and Object Behavior
Implementation-focused with minimal comments, maximum functionality coverage
"""

import functools
from typing import Any, Iterator

# String representation methods
class Product:
    def __init__(self, name, price, category):
        self.name = name
        self.price = price
        self.category = category
        self.id = id(self)
    
    def __str__(self):
        """Human-readable string representation"""
        return f"{self.name} (${self.price:.2f})"
    
    def __repr__(self):
        """Developer-friendly representation"""
        return f"Product('{self.name}', {self.price}, '{self.category}')"
    
    def __format__(self, format_spec):
        """Custom formatting support"""
        if format_spec == 'full':
            return f"{self.name} - {self.category} - ${self.price:.2f}"
        elif format_spec == 'short':
            return f"{self.name[:10]}... ${self.price:.0f}"
        else:
            return str(self)

def string_representation_demo():
    product = Product("Wireless Headphones", 99.99, "Electronics")
    
    return {
        "str_output": str(product),
        "repr_output": repr(product),
        "format_full": f"{product:full}",
        "format_short": f"{product:short}",
        "format_default": f"{product}",
        "eval_repr": eval(repr(product)).name  # repr should be evaluable
    }

# Equality and hashing methods
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        """Equality comparison"""
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False
    
    def __ne__(self, other):
        """Not equal comparison (optional, defaults to not __eq__)"""
        return not self.__eq__(other)
    
    def __hash__(self):
        """Hash for use in sets and dict keys"""
        return hash((self.x, self.y))
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

def equality_hashing_demo():
    p1 = Point(1, 2)
    p2 = Point(1, 2)
    p3 = Point(2, 3)
    
    # Equality tests
    equality_results = {
        "p1_equals_p2": p1 == p2,
        "p1_equals_p3": p1 == p3,
        "p1_not_equals_p3": p1 != p3,
        "p1_hash": hash(p1),
        "p2_hash": hash(p2),
        "p3_hash": hash(p3),
        "same_hash": hash(p1) == hash(p2)
    }
    
    # Set operations
    point_set = {p1, p2, p3}
    equality_results.update({
        "set_size": len(point_set),  # Should be 2 since p1 == p2
        "p1_in_set": p1 in point_set,
        "point_12_in_set": Point(1, 2) in point_set
    })
    
    # Dictionary usage
    point_dict = {p1: "first", p2: "second", p3: "third"}
    equality_results.update({
        "dict_size": len(point_dict),
        "p1_value": point_dict.get(p1),
        "p2_value": point_dict.get(p2),
        "new_point_value": point_dict.get(Point(1, 2))
    })
    
    return equality_results

# Comparison methods
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius
    
    def __lt__(self, other):
        """Less than"""
        if isinstance(other, Temperature):
            return self.celsius < other.celsius
        return NotImplemented
    
    def __le__(self, other):
        """Less than or equal"""
        if isinstance(other, Temperature):
            return self.celsius <= other.celsius
        return NotImplemented
    
    def __gt__(self, other):
        """Greater than"""
        if isinstance(other, Temperature):
            return self.celsius > other.celsius
        return NotImplemented
    
    def __ge__(self, other):
        """Greater than or equal"""
        if isinstance(other, Temperature):
            return self.celsius >= other.celsius
        return NotImplemented
    
    def __eq__(self, other):
        """Equal"""
        if isinstance(other, Temperature):
            return self.celsius == other.celsius
        return NotImplemented
    
    def __repr__(self):
        return f"Temperature({self.celsius}°C)"

@functools.total_ordering
class OptimizedTemperature:
    """Using @total_ordering decorator to auto-generate comparison methods"""
    def __init__(self, celsius):
        self.celsius = celsius
    
    def __eq__(self, other):
        if isinstance(other, OptimizedTemperature):
            return self.celsius == other.celsius
        return NotImplemented
    
    def __lt__(self, other):
        if isinstance(other, OptimizedTemperature):
            return self.celsius < other.celsius
        return NotImplemented
    
    def __repr__(self):
        return f"OptimizedTemperature({self.celsius}°C)"

def comparison_methods_demo():
    temp1 = Temperature(20)
    temp2 = Temperature(25)
    temp3 = Temperature(20)
    
    # All comparison operations
    comparisons = {
        "temp1_lt_temp2": temp1 < temp2,
        "temp1_le_temp3": temp1 <= temp3,
        "temp2_gt_temp1": temp2 > temp1,
        "temp2_ge_temp1": temp2 >= temp1,
        "temp1_eq_temp3": temp1 == temp3,
        "temp1_ne_temp2": temp1 != temp2
    }
    
    # Sorting
    temperatures = [Temperature(30), Temperature(10), Temperature(20)]
    sorted_temps = sorted(temperatures)
    
    # Optimized version
    opt_temp1 = OptimizedTemperature(15)
    opt_temp2 = OptimizedTemperature(25)
    
    comparisons.update({
        "sorted_temperatures": [repr(t) for t in sorted_temps],
        "optimized_comparison": opt_temp1 < opt_temp2,
        "optimized_ge": opt_temp2 >= opt_temp1  # Auto-generated from __eq__ and __lt__
    })
    
    return comparisons

# Arithmetic operators
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """Addition: v1 + v2"""
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented
    
    def __sub__(self, other):
        """Subtraction: v1 - v2"""
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        return NotImplemented
    
    def __mul__(self, other):
        """Multiplication: v * scalar or v * v (dot product)"""
        if isinstance(other, (int, float)):
            return Vector(self.x * other, self.y * other)
        elif isinstance(other, Vector):
            return self.x * other.x + self.y * other.y  # Dot product
        return NotImplemented
    
    def __rmul__(self, other):
        """Right multiplication: scalar * v"""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Division: v / scalar"""
        if isinstance(other, (int, float)) and other != 0:
            return Vector(self.x / other, self.y / other)
        return NotImplemented
    
    def __pow__(self, other):
        """Power: v ** n"""
        if isinstance(other, (int, float)):
            return Vector(self.x ** other, self.y ** other)
        return NotImplemented
    
    def __neg__(self):
        """Unary negation: -v"""
        return Vector(-self.x, -self.y)
    
    def __abs__(self):
        """Absolute value (magnitude): abs(v)"""
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

def arithmetic_operators_demo():
    v1 = Vector(3, 4)
    v2 = Vector(1, 2)
    
    arithmetic_results = {
        "v1": repr(v1),
        "v2": repr(v2),
        "addition": repr(v1 + v2),
        "subtraction": repr(v1 - v2),
        "scalar_mult": repr(v1 * 2),
        "reverse_mult": repr(3 * v1),
        "dot_product": v1 * v2,
        "division": repr(v1 / 2),
        "power": repr(v1 ** 2),
        "negation": repr(-v1),
        "magnitude": abs(v1)
    }
    
    # Chained operations
    result = (v1 + v2) * 2 - Vector(1, 1)
    arithmetic_results["chained_ops"] = repr(result)
    
    return arithmetic_results

# Container methods
class ShoppingCart:
    def __init__(self):
        self.items = {}  # {item_name: quantity}
    
    def __len__(self):
        """Length: len(cart)"""
        return sum(self.items.values())
    
    def __getitem__(self, item_name):
        """Get item: cart[item_name]"""
        return self.items.get(item_name, 0)
    
    def __setitem__(self, item_name, quantity):
        """Set item: cart[item_name] = quantity"""
        if quantity <= 0:
            self.items.pop(item_name, None)
        else:
            self.items[item_name] = quantity
    
    def __delitem__(self, item_name):
        """Delete item: del cart[item_name]"""
        if item_name in self.items:
            del self.items[item_name]
        else:
            raise KeyError(f"Item '{item_name}' not in cart")
    
    def __contains__(self, item_name):
        """Membership test: item_name in cart"""
        return item_name in self.items
    
    def __iter__(self):
        """Iteration: for item in cart"""
        return iter(self.items.items())
    
    def __bool__(self):
        """Boolean conversion: bool(cart)"""
        return len(self.items) > 0
    
    def __repr__(self):
        return f"ShoppingCart({dict(self.items)})"

def container_methods_demo():
    cart = ShoppingCart()
    
    # Add items
    cart["apples"] = 5
    cart["bananas"] = 3
    cart["oranges"] = 2
    
    container_results = {
        "cart_length": len(cart),
        "apple_quantity": cart["apples"],
        "cart_contents": repr(cart),
        "has_apples": "apples" in cart,
        "has_grapes": "grapes" in cart,
        "cart_is_not_empty": bool(cart)
    }
    
    # Iteration
    items_list = list(cart)
    container_results["items_list"] = items_list
    
    # Modification
    cart["apples"] = 8  # Update
    del cart["bananas"]  # Delete
    cart["grapes"] = 0   # Should remove item
    
    container_results.update({
        "after_modification": repr(cart),
        "new_length": len(cart)
    })
    
    # Empty cart test
    cart.items.clear()
    container_results["empty_cart_bool"] = bool(cart)
    
    return container_results

# Callable objects
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
        self.call_count = 0
    
    def __call__(self, value):
        """Make object callable: multiplier(value)"""
        self.call_count += 1
        return value * self.factor
    
    def get_stats(self):
        return {"factor": self.factor, "call_count": self.call_count}

class StatefulFunction:
    def __init__(self):
        self.history = []
    
    def __call__(self, operation, a, b):
        """Callable object that maintains state"""
        if operation == "add":
            result = a + b
        elif operation == "multiply":
            result = a * b
        elif operation == "subtract":
            result = a - b
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        self.history.append({"operation": operation, "args": (a, b), "result": result})
        return result

def callable_objects_demo():
    # Multiplier function object
    double = Multiplier(2)
    triple = Multiplier(3)
    
    callable_results = {
        "double_5": double(5),
        "double_10": double(10),
        "triple_4": triple(4),
        "double_stats": double.get_stats(),
        "triple_stats": triple.get_stats()
    }
    
    # Stateful function
    calculator = StatefulFunction()
    result1 = calculator("add", 10, 5)
    result2 = calculator("multiply", 3, 4)
    result3 = calculator("subtract", 20, 8)
    
    callable_results.update({
        "calc_results": [result1, result2, result3],
        "calc_history": calculator.history,
        "is_callable": callable(calculator)
    })
    
    return callable_results

# Context manager methods
class FileManager:
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.file = None
        self.opened = False
    
    def __enter__(self):
        """Context manager entry"""
        try:
            self.file = open(self.filename, self.mode)
            self.opened = True
            return self.file
        except IOError as e:
            raise IOError(f"Cannot open {self.filename}: {e}")
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit"""
        if self.file:
            self.file.close()
            self.opened = False
        
        # Return False to propagate exceptions
        if exc_type:
            print(f"Exception occurred: {exc_type.__name__}: {exc_value}")
        return False

class DatabaseConnection:
    def __init__(self, db_name):
        self.db_name = db_name
        self.connected = False
        self.transactions = []
    
    def __enter__(self):
        """Simulate database connection"""
        self.connected = True
        self.transactions.append("Connected to database")
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup database connection"""
        if exc_type:
            self.transactions.append(f"Error occurred: {exc_type.__name__}")
            self.transactions.append("Rolling back transaction")
        else:
            self.transactions.append("Committing transaction")
        
        self.connected = False
        self.transactions.append("Connection closed")
        return False  # Don't suppress exceptions
    
    def execute(self, query):
        if not self.connected:
            raise RuntimeError("Not connected to database")
        self.transactions.append(f"Executed: {query}")
        return f"Result of: {query}"

def context_manager_demo():
    import tempfile
    import os
    
    # File manager context
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    temp_file.write("Hello, World!")
    temp_file.close()
    
    try:
        with FileManager(temp_file.name, 'r') as f:
            content = f.read()
        
        file_success = True
        file_content = content
    except Exception as e:
        file_success = False
        file_content = str(e)
    finally:
        os.unlink(temp_file.name)
    
    # Database context manager
    db_results = []
    try:
        with DatabaseConnection("test_db") as db:
            result1 = db.execute("SELECT * FROM users")
            result2 = db.execute("INSERT INTO logs VALUES ('test')")
            db_results = [result1, result2]
    except Exception as e:
        db_error = str(e)
    
    # Error handling context
    db_with_error = DatabaseConnection("error_db")
    try:
        with db_with_error as db:
            db.execute("SELECT * FROM valid_table")
            raise ValueError("Simulated error")  # This will trigger rollback
    except ValueError:
        pass  # Expected error
    
    return {
        "file_manager": {
            "success": file_success,
            "content": file_content
        },
        "database_success": {
            "results": db_results,
            "transactions": DatabaseConnection("test_db").transactions
        },
        "database_with_error": {
            "transactions": db_with_error.transactions
        }
    }

# Attribute access methods
class DynamicAttributes:
    def __init__(self):
        self._data = {}
        self._access_log = []
    
    def __getattr__(self, name):
        """Called when attribute is not found normally"""
        self._access_log.append(f"Accessing: {name}")
        if name.startswith("computed_"):
            # Dynamic computation
            return f"Computed value for {name}"
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Called on every attribute assignment"""
        if name.startswith("_"):
            # Private attributes go to __dict__
            super().__setattr__(name, value)
        else:
            # Public attributes go to _data
            if not hasattr(self, "_data"):
                super().__setattr__("_data", {})
            if not hasattr(self, "_access_log"):
                super().__setattr__("_access_log", [])
            self._data[name] = value
            self._access_log.append(f"Setting: {name} = {value}")
    
    def __getattribute__(self, name):
        """Called on every attribute access"""
        if name == "special_access":
            return "Special handling for this attribute"
        return super().__getattribute__(name)
    
    def __delattr__(self, name):
        """Called when deleting attributes"""
        if name in self._data:
            del self._data[name]
            self._access_log.append(f"Deleted: {name}")
        else:
            super().__delattr__(name)

def attribute_access_demo():
    obj = DynamicAttributes()
    
    # Set some attributes
    obj.name = "Python"
    obj.version = 3.9
    
    # Access attributes
    name_value = obj.name
    computed_value = obj.computed_something
    
    # Special access
    special_value = obj.special_access
    
    # Delete attribute
    del obj.version
    
    # Try to access non-existent attribute
    try:
        nonexistent = obj.nonexistent_attr
        access_error = False
    except AttributeError:
        access_error = True
    
    return {
        "stored_data": obj._data,
        "name_value": name_value,
        "computed_value": computed_value,
        "special_value": special_value,
        "access_log": obj._access_log,
        "access_error_raised": access_error
    }

# Copy and pickle methods
import copy

class CopyableObject:
    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = metadata or {}
        self.id = id(self)
    
    def __copy__(self):
        """Shallow copy implementation"""
        new_obj = self.__class__(self.data, self.metadata)
        new_obj.id = self.id  # Preserve original ID in copy
        return new_obj
    
    def __deepcopy__(self, memo):
        """Deep copy implementation"""
        new_data = copy.deepcopy(self.data, memo)
        new_metadata = copy.deepcopy(self.metadata, memo)
        new_obj = self.__class__(new_data, new_metadata)
        new_obj.id = self.id  # Preserve original ID
        return new_obj

def copy_methods_demo():
    original = CopyableObject([1, [2, 3]], {"key": [4, 5]})
    
    # Shallow copy
    shallow = copy.copy(original)
    
    # Deep copy
    deep = copy.deepcopy(original)
    
    # Modify nested structures
    original.data[1].append(6)
    original.metadata["key"].append(7)
    
    return {
        "original_data": original.data,
        "original_metadata": original.metadata,
        "shallow_data": shallow.data,
        "shallow_metadata": shallow.metadata,
        "deep_data": deep.data,
        "deep_metadata": deep.metadata,
        "shallow_affected": shallow.data[1] == original.data[1],  # Should be True
        "deep_unaffected": deep.data[1] != original.data[1]       # Should be True
    }

# Comprehensive testing
def run_all_magic_method_demos():
    """Execute all special methods demonstrations"""
    demo_functions = [
        ('string_representation', string_representation_demo),
        ('equality_hashing', equality_hashing_demo),
        ('comparison_methods', comparison_methods_demo),
        ('arithmetic_operators', arithmetic_operators_demo),
        ('container_methods', container_methods_demo),
        ('callable_objects', callable_objects_demo),
        ('context_managers', context_manager_demo),
        ('attribute_access', attribute_access_demo),
        ('copy_methods', copy_methods_demo)
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
    print("=== Python Special Methods (Magic Methods) Demo ===")
    
    # Run all demonstrations
    all_results = run_all_magic_method_demos()
    
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
    
    print("\n=== SPECIAL METHODS CATEGORIES ===")
    
    categories = {
        "String Representation": "__str__, __repr__, __format__",
        "Comparison": "__eq__, __ne__, __lt__, __le__, __gt__, __ge__",
        "Hashing": "__hash__ for use in sets and dictionaries",
        "Arithmetic": "__add__, __sub__, __mul__, __div__, __pow__, etc.",
        "Container": "__len__, __getitem__, __setitem__, __contains__, __iter__",
        "Callable": "__call__ to make objects callable like functions",
        "Context Manager": "__enter__, __exit__ for with statements",
        "Attribute Access": "__getattr__, __setattr__, __delattr__",
        "Copy/Pickle": "__copy__, __deepcopy__ for object duplication"
    }
    
    for category, methods in categories.items():
        print(f"  {category}: {methods}")
    
    print("\n=== IMPLEMENTATION GUIDELINES ===")
    
    guidelines = [
        "Always implement __repr__ for debugging",
        "Implement __str__ for user-friendly output",
        "If you implement __eq__, also implement __hash__",
        "Use @functools.total_ordering for comparison methods",
        "Return NotImplemented for unsupported operations",
        "Context managers should handle exceptions properly",
        "Arithmetic operations should work with appropriate types",
        "Container methods should behave like built-in containers"
    ]
    
    for guideline in guidelines:
        print(f"  • {guideline}")
    
    print("\n=== COMMON PATTERNS ===")
    
    patterns = [
        "Operator overloading for mathematical objects",
        "Container classes that behave like lists/dicts",
        "Context managers for resource management",
        "Callable objects as stateful functions",
        "Custom comparison logic for sorting",
        "String formatting for different output formats",
        "Attribute access control and validation"
    ]
    
    for pattern in patterns:
        print(f"  • {pattern}")
    
    print("\n=== Special Methods Complete! ===")
    print("  Object behavior customization through dunder methods mastered")
