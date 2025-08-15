"""
Python Methods and Properties: Instance, Class, Static Methods and Property Decorators
Implementation-focused with minimal comments, maximum functionality coverage
"""

import time
import math
from functools import wraps
from typing import Any, Optional, Union

# Instance methods fundamentals
class Calculator:
    def __init__(self, precision=2):
        self.precision = precision
        self.history = []
    
    def add(self, a, b):
        """Instance method - operates on instance data"""
        result = round(a + b, self.precision)
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        result = round(a * b, self.precision)
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def get_history(self):
        return self.history.copy()
    
    def clear_history(self):
        self.history.clear()
        return "History cleared"

def instance_methods_demo():
    calc1 = Calculator(3)
    calc2 = Calculator(1)
    
    # Each instance maintains its own state
    calc1.add(10.5, 20.3)
    calc1.multiply(5, 7)
    calc2.add(1.1, 2.2)
    
    return {
        "calc1_history": calc1.get_history(),
        "calc2_history": calc2.get_history(),
        "calc1_precision": calc1.precision,
        "calc2_precision": calc2.precision,
        "different_instances": calc1 is not calc2
    }

# Class methods with @classmethod
class Employee:
    company_name = "TechCorp"
    employee_count = 0
    salary_grades = {"junior": 50000, "senior": 80000, "lead": 120000}
    
    def __init__(self, name, grade="junior"):
        self.name = name
        self.grade = grade
        self.salary = self.salary_grades[grade]
        self.employee_id = Employee.employee_count + 1
        Employee.employee_count += 1
    
    @classmethod
    def get_company_info(cls):
        """Class method - operates on class data"""
        return {
            "company": cls.company_name,
            "total_employees": cls.employee_count,
            "available_grades": list(cls.salary_grades.keys())
        }
    
    @classmethod
    def create_senior_employee(cls, name):
        """Class method as alternative constructor"""
        return cls(name, "senior")
    
    @classmethod
    def create_lead_employee(cls, name):
        return cls(name, "lead")
    
    @classmethod
    def update_salary_grade(cls, grade, new_salary):
        """Class method to modify class data"""
        if grade in cls.salary_grades:
            cls.salary_grades[grade] = new_salary
            return True
        return False
    
    def get_employee_info(self):
        return {
            "name": self.name,
            "grade": self.grade,
            "salary": self.salary,
            "id": self.employee_id
        }

def class_methods_demo():
    # Regular instantiation
    emp1 = Employee("Alice")
    
    # Alternative constructors using class methods
    emp2 = Employee.create_senior_employee("Bob")
    emp3 = Employee.create_lead_employee("Charlie")
    
    # Class method to get company info
    company_info = Employee.get_company_info()
    
    # Update salary grades
    Employee.update_salary_grade("senior", 85000)
    
    # Create another employee after salary update
    emp4 = Employee("Diana", "senior")
    
    return {
        "company_info": company_info,
        "emp1_info": emp1.get_employee_info(),
        "emp2_info": emp2.get_employee_info(),
        "emp3_info": emp3.get_employee_info(),
        "emp4_salary": emp4.salary,  # Should reflect updated salary
        "total_employees": Employee.employee_count
    }

# Static methods with @staticmethod
class MathUtils:
    PI = 3.14159
    
    def __init__(self, name):
        self.name = name
    
    @staticmethod
    def factorial(n):
        """Static method - independent utility function"""
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n - 1)
    
    @staticmethod
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    @staticmethod
    def gcd(a, b):
        """Greatest common divisor using Euclidean algorithm"""
        while b:
            a, b = b, a % b
        return a
    
    @staticmethod
    def lcm(a, b):
        """Least common multiple"""
        return abs(a * b) // MathUtils.gcd(a, b)
    
    @staticmethod
    def distance_2d(x1, y1, x2, y2):
        """Calculate distance between two 2D points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def instance_method_using_static(self, n):
        """Instance method that uses static methods"""
        return {
            "calculator": self.name,
            "factorial": MathUtils.factorial(n),
            "is_prime": MathUtils.is_prime(n),
            "computation_time": time.time()
        }

def static_methods_demo():
    # Static methods can be called without creating instances
    static_results = {
        "factorial_5": MathUtils.factorial(5),
        "is_17_prime": MathUtils.is_prime(17),
        "is_20_prime": MathUtils.is_prime(20),
        "gcd_48_18": MathUtils.gcd(48, 18),
        "lcm_12_8": MathUtils.lcm(12, 8),
        "distance": MathUtils.distance_2d(0, 0, 3, 4)
    }
    
    # Static methods can also be called from instances
    math_util = MathUtils("MyCalculator")
    instance_result = math_util.instance_method_using_static(7)
    
    # Call static method from instance
    static_results.update({
        "factorial_from_instance": math_util.factorial(6),
        "instance_computation": instance_result
    })
    
    return static_results

# Property decorators - basic usage
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Getter for celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Setter for celsius with validation"""
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        """Computed property - no setter, read-only"""
        return (self._celsius * 9/5) + 32
    
    @property
    def kelvin(self):
        """Another computed property"""
        return self._celsius + 273.15
    
    @property
    def temperature_info(self):
        """Complex computed property"""
        return {
            "celsius": self.celsius,
            "fahrenheit": self.fahrenheit,
            "kelvin": self.kelvin,
            "is_freezing": self.celsius <= 0,
            "is_boiling": self.celsius >= 100
        }

def basic_properties_demo():
    temp = Temperature(25)
    
    # Read properties
    basic_props = {
        "initial_celsius": temp.celsius,
        "initial_fahrenheit": temp.fahrenheit,
        "initial_kelvin": temp.kelvin,
        "temp_info": temp.temperature_info
    }
    
    # Modify celsius via setter
    temp.celsius = 100
    
    basic_props.update({
        "after_change_celsius": temp.celsius,
        "after_change_fahrenheit": temp.fahrenheit,
        "after_change_kelvin": temp.kelvin,
        "boiling_point": temp.temperature_info["is_boiling"]
    })
    
    # Test validation
    try:
        temp.celsius = -300  # Should raise ValueError
        validation_error = False
    except ValueError:
        validation_error = True
    
    basic_props["validation_works"] = validation_error
    
    return basic_props

# Advanced property patterns
class SmartProperty:
    def __init__(self, initial_value=None):
        self._value = initial_value
        self._access_count = 0
        self._modification_count = 0
        self._history = []
    
    @property
    def value(self):
        """Property with access tracking"""
        self._access_count += 1
        return self._value
    
    @value.setter
    def value(self, new_value):
        """Setter with history tracking"""
        old_value = self._value
        self._value = new_value
        self._modification_count += 1
        self._history.append({
            "old": old_value,
            "new": new_value,
            "timestamp": time.time()
        })
    
    @value.deleter
    def value(self):
        """Deleter for property"""
        self._value = None
        self._modification_count += 1
        self._history.append({
            "action": "deleted",
            "timestamp": time.time()
        })
    
    @property
    def stats(self):
        """Read-only property with usage statistics"""
        return {
            "access_count": self._access_count,
            "modification_count": self._modification_count,
            "current_value": self._value,
            "history_length": len(self._history)
        }

def advanced_properties_demo():
    smart_prop = SmartProperty("initial")
    
    # Access value multiple times
    val1 = smart_prop.value
    val2 = smart_prop.value
    
    # Modify value
    smart_prop.value = "modified"
    smart_prop.value = "final"
    
    # Get stats
    stats_before_delete = smart_prop.stats
    
    # Delete value
    del smart_prop.value
    stats_after_delete = smart_prop.stats
    
    return {
        "stats_before_delete": stats_before_delete,
        "stats_after_delete": stats_after_delete,
        "history": smart_prop._history
    }

# Property with complex getter/setter logic
class BankAccountWithProperties:
    def __init__(self, account_number, initial_balance=0):
        self.account_number = account_number
        self._balance = initial_balance
        self._interest_rate = 0.05
        self._transaction_log = []
    
    @property
    def balance(self):
        """Balance with interest calculation"""
        return round(self._balance, 2)
    
    @balance.setter
    def balance(self, amount):
        """Balance setter with validation and logging"""
        if amount < 0:
            raise ValueError("Balance cannot be negative")
        
        old_balance = self._balance
        self._balance = amount
        self._transaction_log.append({
            "type": "balance_set",
            "old_balance": old_balance,
            "new_balance": amount,
            "timestamp": time.time()
        })
    
    @property
    def interest_rate(self):
        return self._interest_rate
    
    @interest_rate.setter
    def interest_rate(self, rate):
        if not 0 <= rate <= 1:
            raise ValueError("Interest rate must be between 0 and 1")
        self._interest_rate = rate
    
    @property
    def projected_balance(self):
        """Computed property - balance after one year of interest"""
        return round(self._balance * (1 + self._interest_rate), 2)
    
    @property
    def account_summary(self):
        """Complex read-only property"""
        return {
            "account_number": self.account_number,
            "current_balance": self.balance,
            "interest_rate": self.interest_rate * 100,  # As percentage
            "projected_balance": self.projected_balance,
            "transaction_count": len(self._transaction_log)
        }

def complex_properties_demo():
    account = BankAccountWithProperties("ACC123", 1000)
    
    # Read initial properties
    initial_summary = account.account_summary
    
    # Modify balance
    account.balance = 1500
    
    # Change interest rate
    account.interest_rate = 0.08
    
    # Get updated summary
    final_summary = account.account_summary
    
    return {
        "initial_summary": initial_summary,
        "final_summary": final_summary,
        "transaction_log": account._transaction_log
    }

# Method chaining patterns
class FluentCalculator:
    def __init__(self, initial_value=0):
        self.value = initial_value
        self.operations = []
    
    def add(self, amount):
        """Add and return self for chaining"""
        self.value += amount
        self.operations.append(f"add({amount})")
        return self
    
    def multiply(self, factor):
        """Multiply and return self for chaining"""
        self.value *= factor
        self.operations.append(f"multiply({factor})")
        return self
    
    def subtract(self, amount):
        """Subtract and return self for chaining"""
        self.value -= amount
        self.operations.append(f"subtract({amount})")
        return self
    
    def divide(self, divisor):
        """Divide and return self for chaining"""
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        self.value /= divisor
        self.operations.append(f"divide({divisor})")
        return self
    
    def power(self, exponent):
        """Power and return self for chaining"""
        self.value **= exponent
        self.operations.append(f"power({exponent})")
        return self
    
    def result(self):
        """Get final result (terminates chain)"""
        return round(self.value, 6)
    
    def reset(self):
        """Reset calculator and return self for chaining"""
        self.value = 0
        self.operations = []
        return self
    
    def get_history(self):
        """Get operation history"""
        return " -> ".join(self.operations)

def method_chaining_demo():
    calc = FluentCalculator(10)
    
    # Method chaining example
    result1 = calc.add(5).multiply(2).subtract(10).divide(2).result()
    history1 = calc.get_history()
    
    # Another chain after reset
    result2 = calc.reset().add(100).power(2).divide(4).subtract(2498).result()
    history2 = calc.get_history()
    
    # Complex chaining
    calc.reset()
    result3 = (calc
               .add(10)
               .multiply(3)
               .power(2)
               .divide(9)
               .subtract(90)
               .result())
    history3 = calc.get_history()
    
    return {
        "result1": result1,
        "history1": history1,
        "result2": result2,
        "history2": history2,
        "result3": result3,
        "history3": history3
    }

# When to use each method type
class MethodTypeExamples:
    class_counter = 0
    utility_functions = []
    
    def __init__(self, name):
        self.name = name
        self.instance_data = []
        MethodTypeExamples.class_counter += 1
    
    # Instance method - needs access to instance data
    def add_data(self, item):
        """Use when: Operating on instance-specific data"""
        self.instance_data.append(item)
        return len(self.instance_data)
    
    def get_instance_summary(self):
        """Use when: Returning instance-specific information"""
        return {
            "name": self.name,
            "data_count": len(self.instance_data),
            "data_preview": self.instance_data[:3]
        }
    
    # Class method - needs access to class data or alternative constructors
    @classmethod
    def get_class_stats(cls):
        """Use when: Operating on class-level data"""
        return {
            "total_instances": cls.class_counter,
            "utility_functions": len(cls.utility_functions)
        }
    
    @classmethod
    def create_with_initial_data(cls, name, initial_data):
        """Use when: Alternative constructor patterns"""
        instance = cls(name)
        for item in initial_data:
            instance.add_data(item)
        return instance
    
    @classmethod
    def register_utility(cls, func_name):
        """Use when: Modifying class-level collections"""
        cls.utility_functions.append(func_name)
    
    # Static method - independent utility functions
    @staticmethod
    def validate_data_format(data):
        """Use when: Utility function that doesn't need class/instance access"""
        if not isinstance(data, (list, tuple)):
            return False, "Data must be a list or tuple"
        if len(data) == 0:
            return False, "Data cannot be empty"
        return True, "Valid format"
    
    @staticmethod
    def calculate_statistics(numbers):
        """Use when: Mathematical/utility operations independent of class state"""
        if not numbers:
            return {}
        return {
            "count": len(numbers),
            "sum": sum(numbers),
            "average": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers)
        }

def method_types_comparison():
    # Create instances
    obj1 = MethodTypeExamples("Object1")
    obj2 = MethodTypeExamples.create_with_initial_data("Object2", [1, 2, 3])
    
    # Instance method usage
    obj1.add_data("item1")
    obj1.add_data("item2")
    
    # Class method usage
    MethodTypeExamples.register_utility("data_validator")
    class_stats = MethodTypeExamples.get_class_stats()
    
    # Static method usage (can be called from class or instance)
    validation_result = MethodTypeExamples.validate_data_format([1, 2, 3])
    stats_result = obj1.calculate_statistics([10, 20, 30, 40, 50])
    
    return {
        "obj1_summary": obj1.get_instance_summary(),
        "obj2_summary": obj2.get_instance_summary(),
        "class_statistics": class_stats,
        "validation_result": validation_result,
        "statistics_result": stats_result,
        "usage_recommendations": {
            "instance_methods": "When you need access to instance data (self)",
            "class_methods": "For alternative constructors or class data operations",
            "static_methods": "For utility functions that don't need class/instance access"
        }
    }

# Property vs direct attribute access
class PropertyVsAttribute:
    def __init__(self):
        self.direct_attribute = "Direct access"
        self._computed_count = 0
        self._validated_value = None
    
    # Direct attribute (no special behavior)
    simple_attr = "Simple class attribute"
    
    # Property with computation
    @property
    def computed_property(self):
        """Property that performs computation on access"""
        self._computed_count += 1
        return f"Computed {self._computed_count} times"
    
    # Property with validation
    @property
    def validated_value(self):
        return self._validated_value
    
    @validated_value.setter
    def validated_value(self, value):
        if not isinstance(value, str) or len(value) < 3:
            raise ValueError("Value must be a string with at least 3 characters")
        self._validated_value = value.upper()
    
    # Property that looks like attribute but has behavior
    @property
    def current_timestamp(self):
        """Always returns current time"""
        return time.time()

def property_vs_attribute_demo():
    obj = PropertyVsAttribute()
    
    # Direct attribute access
    obj.direct_attribute = "Modified directly"
    direct_result = obj.direct_attribute
    
    # Property access with computation
    computed1 = obj.computed_property
    computed2 = obj.computed_property
    computed3 = obj.computed_property
    
    # Property with validation
    obj.validated_value = "valid input"
    validated_result = obj.validated_value
    
    try:
        obj.validated_value = "no"  # Should raise ValueError
        validation_failed = False
    except ValueError:
        validation_failed = True
    
    # Timestamp property
    timestamp1 = obj.current_timestamp
    time.sleep(0.01)  # Small delay
    timestamp2 = obj.current_timestamp
    
    return {
        "direct_access": direct_result,
        "computed_calls": [computed1, computed2, computed3],
        "validated_result": validated_result,
        "validation_failed": validation_failed,
        "timestamps_different": timestamp1 != timestamp2,
        "recommendations": {
            "use_properties_when": [
                "You need validation on setting values",
                "You need computed/derived values",
                "You want to track access/modifications",
                "You need to maintain backward compatibility while changing internal implementation"
            ],
            "use_direct_attributes_when": [
                "Simple data storage with no special behavior",
                "Performance is critical and no computation needed",
                "No validation or transformation required"
            ]
        }
    }

# Comprehensive testing
def run_all_method_demos():
    """Execute all method and property demonstrations"""
    demo_functions = [
        ('instance_methods', instance_methods_demo),
        ('class_methods', class_methods_demo),
        ('static_methods', static_methods_demo),
        ('basic_properties', basic_properties_demo),
        ('advanced_properties', advanced_properties_demo),
        ('complex_properties', complex_properties_demo),
        ('method_chaining', method_chaining_demo),
        ('method_types_comparison', method_types_comparison),
        ('property_vs_attribute', property_vs_attribute_demo)
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
    print("=== Python Methods and Properties Demo ===")
    
    # Run all demonstrations
    all_results = run_all_method_demos()
    
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
    
    print("\n=== METHOD TYPES SUMMARY ===")
    
    method_types = {
        "Instance Methods": "Operate on instance data, require self parameter",
        "Class Methods": "Operate on class data, alternative constructors, use @classmethod",
        "Static Methods": "Utility functions, no access to self/cls, use @staticmethod",
        "Properties": "Attribute-like access with getter/setter/deleter behavior"
    }
    
    for method_type, description in method_types.items():
        print(f"  {method_type}: {description}")
    
    print("\n=== PROPERTY PATTERNS ===")
    
    property_patterns = [
        "Read-only properties: Only @property decorator",
        "Read-write properties: @property + @name.setter",
        "Deletable properties: Add @name.deleter",
        "Computed properties: Calculate value in getter",
        "Validated properties: Check values in setter",
        "Cached properties: Store computed results",
        "Tracked properties: Log access/modifications"
    ]
    
    for pattern in property_patterns:
        print(f"  • {pattern}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Use instance methods for operations on instance data",
        "Use class methods for alternative constructors",
        "Use static methods for utility functions",
        "Use properties for computed or validated attributes",
        "Implement method chaining by returning self",
        "Document the purpose of each method type",
        "Prefer properties over getter/setter methods",
        "Use descriptors for reusable property logic"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Methods and Properties Complete! ===")
    print("  Advanced method patterns and property usage mastered")
