"""
Python Dataclasses and Attrs: Modern Python Data Classes and Attribute Management
Implementation-focused with minimal comments, maximum functionality coverage
"""

from dataclasses import dataclass, field, fields, asdict, astuple, replace, make_dataclass, InitVar
from typing import List, Dict, Optional, Any, ClassVar, Union
import json
import time
from enum import Enum
from functools import total_ordering
import copy

# Basic dataclass usage
@dataclass
class Person:
    name: str
    age: int
    email: str = ""
    active: bool = True

@dataclass
class Employee(Person):
    employee_id: int
    department: str
    salary: float = 0.0

def basic_dataclass_demo():
    # Create instances
    person = Person("Alice", 30, "alice@email.com")
    employee = Employee("Bob", 25, "bob@company.com", True, 12345, "Engineering", 75000.0)
    
    # Auto-generated methods
    basic_operations = {
        "person_repr": repr(person),
        "employee_str": str(employee),
        "persons_equal": person == Person("Alice", 30, "alice@email.com"),
        "employees_different": employee != Employee("Charlie", 25, "charlie@company.com", True, 12346, "Marketing")
    }
    
    # Field access
    field_operations = {
        "person_name": person.name,
        "employee_salary": employee.salary,
        "person_default_active": person.active,
        "employee_inheritance": employee.name  # Inherited from Person
    }
    
    return {
        "basic_operations": basic_operations,
        "field_operations": field_operations
    }

# Dataclass with field customization
@dataclass
class Product:
    name: str
    price: float
    categories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    _internal_id: int = field(default=0, init=False, repr=False)
    
    def __post_init__(self):
        self._internal_id = hash(self.name + str(self.price))

@dataclass
class Inventory:
    items: List[Product] = field(default_factory=list)
    total_value: float = field(init=False)
    
    def __post_init__(self):
        self.total_value = sum(item.price for item in self.items)
    
    def add_item(self, item: Product):
        self.items.append(item)
        self.total_value += item.price

def field_customization_demo():
    # Create products with default factories
    product1 = Product("Laptop", 999.99, ["Electronics", "Computers"])
    product2 = Product("Book", 29.99)
    
    # Test default factories
    product1.categories.append("Gaming")
    product1.metadata["warranty"] = "2 years"
    
    field_results = {
        "product1_categories": product1.categories,
        "product2_categories": product2.categories,  # Should be empty list
        "product1_metadata": product1.metadata,
        "product2_metadata": product2.metadata,  # Should be empty dict
        "different_lists": product1.categories is not product2.categories,
        "internal_ids": [product1._internal_id, product2._internal_id]
    }
    
    # Test inventory with computed field
    inventory = Inventory([product1, product2])
    initial_value = inventory.total_value
    
    product3 = Product("Mouse", 49.99, ["Electronics", "Accessories"])
    inventory.add_item(product3)
    
    field_results.update({
        "initial_inventory_value": initial_value,
        "after_adding_item": inventory.total_value,
        "item_count": len(inventory.items)
    })
    
    return field_results

# Frozen dataclasses (immutable)
@dataclass(frozen=True)
class Point:
    x: float
    y: float
    
    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    def translate(self, dx: float, dy: float) -> 'Point':
        # Return new instance since frozen
        return Point(self.x + dx, self.y + dy)

@dataclass(frozen=True)
class Rectangle:
    top_left: Point
    bottom_right: Point
    
    @property
    def area(self) -> float:
        width = abs(self.bottom_right.x - self.top_left.x)
        height = abs(self.top_left.y - self.bottom_right.y)
        return width * height
    
    @property
    def center(self) -> Point:
        cx = (self.top_left.x + self.bottom_right.x) / 2
        cy = (self.top_left.y + self.bottom_right.y) / 2
        return Point(cx, cy)

def frozen_dataclass_demo():
    # Create immutable objects
    point1 = Point(3.0, 4.0)
    point2 = Point(0.0, 0.0)
    
    # Test immutability
    try:
        point1.x = 10  # Should raise error
        mutation_blocked = False
    except Exception:
        mutation_blocked = True
    
    # Test methods on frozen objects
    distance = point1.distance_from_origin()
    translated = point1.translate(1.0, 1.0)
    
    # Create rectangle
    rect = Rectangle(Point(0, 10), Point(10, 0))
    
    frozen_results = {
        "mutation_blocked": mutation_blocked,
        "point1_distance": distance,
        "translated_point": (translated.x, translated.y),
        "original_point_unchanged": (point1.x, point1.y),
        "rectangle_area": rect.area,
        "rectangle_center": (rect.center.x, rect.center.y),
        "points_hashable": hash(point1) != hash(point2)
    }
    
    return frozen_results

# Dataclass comparison and ordering
@total_ordering
@dataclass
class Student:
    name: str
    grade: float
    student_id: int = field(compare=False)  # Exclude from comparison
    
    def __eq__(self, other):
        if not isinstance(other, Student):
            return NotImplemented
        return self.grade == other.grade
    
    def __lt__(self, other):
        if not isinstance(other, Student):
            return NotImplemented
        return self.grade < other.grade

@dataclass(order=True)
class Task:
    priority: int
    name: str
    description: str = field(compare=False)

def comparison_ordering_demo():
    # Test student comparison (custom)
    student1 = Student("Alice", 85.5, 12345)
    student2 = Student("Bob", 92.0, 12346)
    student3 = Student("Charlie", 85.5, 12347)  # Same grade as Alice
    
    student_comparisons = {
        "alice_vs_bob": student1 < student2,
        "alice_vs_charlie": student1 == student3,
        "bob_vs_alice": student2 > student1,
        "student_id_ignored": student1 == Student("Different Name", 85.5, 99999)
    }
    
    # Test task comparison (automatic ordering)
    task1 = Task(1, "Critical Bug", "Fix production issue")
    task2 = Task(3, "Feature Request", "Add new feature")
    task3 = Task(2, "Code Review", "Review pull request")
    
    tasks = [task2, task1, task3]
    sorted_tasks = sorted(tasks)
    
    task_results = {
        "original_order": [task.name for task in tasks],
        "sorted_order": [task.name for task in sorted_tasks],
        "task1_lt_task2": task1 < task2,
        "priorities": [task.priority for task in sorted_tasks]
    }
    
    return {
        "student_comparisons": student_comparisons,
        "task_ordering": task_results
    }

# Advanced field features
class Status(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Job:
    name: str
    status: Status = Status.PENDING
    created_at: InitVar[Optional[float]] = None  # Init-only variable
    tags: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict, repr=False)
    _runtime_data: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    
    # Class variable (not a field)
    total_jobs: ClassVar[int] = 0
    
    def __post_init__(self, created_at: Optional[float]):
        if created_at is None:
            created_at = time.time()
        self._runtime_data["created_at"] = created_at
        Job.total_jobs += 1
    
    def add_tag(self, tag: str):
        if tag not in self.tags:
            self.tags.append(tag)
    
    def set_config(self, key: str, value: Any):
        self.config[key] = value
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self._runtime_data["created_at"]

def advanced_fields_demo():
    # Reset class counter
    Job.total_jobs = 0
    
    # Create jobs with different parameters
    job1 = Job("Data Processing")
    job2 = Job("Report Generation", Status.PROCESSING, created_at=time.time() - 3600)  # 1 hour ago
    job3 = Job("Backup", Status.COMPLETED, tags=["maintenance", "daily"])
    
    # Modify jobs
    job1.add_tag("important")
    job1.set_config("retry_count", 3)
    job2.set_config("output_format", "pdf")
    
    advanced_results = {
        "job_representations": [repr(job1), repr(job2), repr(job3)],
        "total_jobs_created": Job.total_jobs,
        "job1_tags": job1.tags,
        "job1_config": job1.config,
        "job2_older": job2.age_seconds > job1.age_seconds,
        "status_enum": job3.status.value
    }
    
    return advanced_results

# Dataclass serialization and deserialization
@dataclass
class Address:
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"

@dataclass
class Contact:
    name: str
    email: str
    phone: str
    address: Address
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_tuple(self) -> tuple:
        """Convert to tuple"""
        return astuple(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contact':
        """Create from dictionary"""
        # Handle nested Address
        if 'address' in data and isinstance(data['address'], dict):
            address_data = data.pop('address')
            address = Address(**address_data)
            return cls(address=address, **data)
        return cls(**data)

def serialization_demo():
    # Create complex object
    address = Address("123 Main St", "Anytown", "CA", "12345")
    contact = Contact(
        "John Doe",
        "john@example.com",
        "555-0123",
        address,
        ["friend", "colleague"],
        {"last_contact": "2023-01-15", "priority": "high"}
    )
    
    # Test serialization
    contact_dict = contact.to_dict()
    contact_tuple = contact.to_tuple()
    
    # JSON serialization
    contact_json = json.dumps(contact_dict, indent=2)
    
    # Deserialization
    parsed_dict = json.loads(contact_json)
    restored_contact = Contact.from_dict(parsed_dict)
    
    serialization_results = {
        "original_contact": repr(contact),
        "dict_keys": list(contact_dict.keys()),
        "tuple_length": len(contact_tuple),
        "json_length": len(contact_json),
        "restored_contact": repr(restored_contact),
        "serialization_successful": contact.name == restored_contact.name and contact.address.city == restored_contact.address.city
    }
    
    return serialization_results

# Dataclass inheritance patterns
@dataclass
class Vehicle:
    make: str
    model: str
    year: int
    color: str = "White"

@dataclass
class Car(Vehicle):
    doors: int = 4
    fuel_type: str = "Gasoline"

@dataclass
class ElectricCar(Car):
    battery_capacity: float
    range_miles: int
    fuel_type: str = field(default="Electric", init=False)  # Override parent field
    
    def __post_init__(self):
        # Ensure fuel_type is always Electric
        self.fuel_type = "Electric"

@dataclass
class Motorcycle(Vehicle):
    engine_size: float
    has_sidecar: bool = False

def inheritance_demo():
    # Create vehicles
    car = Car("Toyota", "Camry", 2023, "Blue", 4, "Hybrid")
    electric_car = ElectricCar("Tesla", "Model 3", 2023, "Red", 4, "N/A", 75.0, 300)
    motorcycle = Motorcycle("Harley", "Sportster", 2023, "Black", 883.0)
    
    # Test inheritance
    inheritance_results = {
        "car_details": asdict(car),
        "electric_car_details": asdict(electric_car),
        "motorcycle_details": asdict(motorcycle),
        "car_is_vehicle": isinstance(car, Vehicle),
        "electric_is_car": isinstance(electric_car, Car),
        "electric_fuel_type": electric_car.fuel_type,  # Should be "Electric"
        "field_inheritance": hasattr(motorcycle, 'make')  # Inherited from Vehicle
    }
    
    return inheritance_results

# Dataclass with validation
@dataclass
class ValidatedUser:
    username: str
    email: str
    age: int
    
    def __post_init__(self):
        self._validate_username()
        self._validate_email()
        self._validate_age()
    
    def _validate_username(self):
        if not isinstance(self.username, str) or len(self.username) < 3:
            raise ValueError("Username must be at least 3 characters")
    
    def _validate_email(self):
        if not isinstance(self.email, str) or "@" not in self.email:
            raise ValueError("Invalid email format")
    
    def _validate_age(self):
        if not isinstance(self.age, int) or self.age < 0 or self.age > 150:
            raise ValueError("Age must be between 0 and 150")

@dataclass
class BankAccount:
    account_number: str
    balance: float = 0.0
    _transactions: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        if self.balance < 0:
            raise ValueError("Initial balance cannot be negative")
    
    def deposit(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        
        self.balance += amount
        self._transactions.append({
            "type": "deposit",
            "amount": amount,
            "timestamp": time.time(),
            "balance_after": self.balance
        })
        return self.balance
    
    def withdraw(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        
        self.balance -= amount
        self._transactions.append({
            "type": "withdrawal",
            "amount": amount,
            "timestamp": time.time(),
            "balance_after": self.balance
        })
        return self.balance
    
    def get_transaction_history(self) -> List[Dict[str, Any]]:
        return self._transactions.copy()

def validation_demo():
    # Test user validation
    try:
        valid_user = ValidatedUser("alice123", "alice@example.com", 25)
        user_creation_success = True
    except ValueError:
        user_creation_success = False
    
    validation_errors = []
    
    # Test invalid username
    try:
        ValidatedUser("ab", "alice@example.com", 25)
    except ValueError as e:
        validation_errors.append(str(e))
    
    # Test invalid email
    try:
        ValidatedUser("alice123", "invalid-email", 25)
    except ValueError as e:
        validation_errors.append(str(e))
    
    # Test invalid age
    try:
        ValidatedUser("alice123", "alice@example.com", -5)
    except ValueError as e:
        validation_errors.append(str(e))
    
    # Test bank account
    account = BankAccount("ACC123456", 1000.0)
    account.deposit(500.0)
    account.withdraw(200.0)
    
    # Test bank account validation
    try:
        account.withdraw(2000.0)  # Insufficient funds
        withdrawal_error = None
    except ValueError as e:
        withdrawal_error = str(e)
    
    validation_results = {
        "user_creation_success": user_creation_success,
        "validation_errors": validation_errors,
        "account_balance": account.balance,
        "transaction_count": len(account.get_transaction_history()),
        "withdrawal_error": withdrawal_error
    }
    
    return validation_results

# Dynamic dataclass creation
def create_dynamic_dataclass():
    # Create dataclass programmatically
    DynamicPerson = make_dataclass(
        'DynamicPerson',
        [
            'name',
            'age',
            ('email', str, ''),  # With default
            ('active', bool, field(default=True)),  # With field
            ('metadata', dict, field(default_factory=dict))
        ]
    )
    
    # Create instance
    person = DynamicPerson("Dynamic User", 30, "user@example.com")
    person.metadata["created_dynamically"] = True
    
    # Get field information
    field_info = {}
    for f in fields(DynamicPerson):
        field_info[f.name] = {
            "type": f.type,
            "default": f.default if f.default != dataclass.MISSING else "NO_DEFAULT",
            "default_factory": f.default_factory if f.default_factory != dataclass.MISSING else "NO_FACTORY"
        }
    
    return {
        "person_dict": asdict(person),
        "field_info": field_info,
        "class_name": DynamicPerson.__name__
    }

# Dataclass modification with replace()
@dataclass(frozen=True)
class ImmutableConfig:
    debug: bool
    log_level: str
    max_connections: int
    timeout: float

def replacement_demo():
    # Create original config
    config = ImmutableConfig(True, "INFO", 100, 30.0)
    
    # Create modified versions using replace()
    prod_config = replace(config, debug=False, log_level="ERROR")
    high_perf_config = replace(config, max_connections=500, timeout=60.0)
    
    replacement_results = {
        "original_config": asdict(config),
        "prod_config": asdict(prod_config),
        "high_perf_config": asdict(high_perf_config),
        "configs_different": config is not prod_config,
        "original_unchanged": config.debug == True  # Original is unchanged
    }
    
    return replacement_results

# Performance comparison
def performance_comparison():
    import timeit
    
    # Regular class
    class RegularPerson:
        def __init__(self, name, age, email=""):
            self.name = name
            self.age = age
            self.email = email
        
        def __repr__(self):
            return f"RegularPerson(name='{self.name}', age={self.age}, email='{self.email}')"
        
        def __eq__(self, other):
            return (isinstance(other, RegularPerson) and 
                   self.name == other.name and 
                   self.age == other.age and 
                   self.email == other.email)
    
    @dataclass
    class DataclassPerson:
        name: str
        age: int
        email: str = ""
    
    # Test creation performance
    def create_regular():
        return RegularPerson("Test", 30, "test@example.com")
    
    def create_dataclass():
        return DataclassPerson("Test", 30, "test@example.com")
    
    # Measure performance
    regular_time = timeit.timeit(create_regular, number=10000)
    dataclass_time = timeit.timeit(create_dataclass, number=10000)
    
    # Test features
    regular_obj = create_regular()
    dataclass_obj = create_dataclass()
    
    return {
        "creation_times": {
            "regular_class": f"{regular_time:.6f}s",
            "dataclass": f"{dataclass_time:.6f}s",
            "difference": f"{abs(regular_time - dataclass_time):.6f}s"
        },
        "feature_comparison": {
            "regular_repr": repr(regular_obj),
            "dataclass_repr": repr(dataclass_obj),
            "regular_has_eq": hasattr(RegularPerson, '__eq__'),
            "dataclass_has_eq": hasattr(DataclassPerson, '__eq__')
        }
    }

# Comprehensive testing
def run_all_dataclass_demos():
    """Execute all dataclass demonstrations"""
    demo_functions = [
        ('basic_dataclass', basic_dataclass_demo),
        ('field_customization', field_customization_demo),
        ('frozen_dataclass', frozen_dataclass_demo),
        ('comparison_ordering', comparison_ordering_demo),
        ('advanced_fields', advanced_fields_demo),
        ('serialization', serialization_demo),
        ('inheritance', inheritance_demo),
        ('validation', validation_demo),
        ('dynamic_creation', create_dynamic_dataclass),
        ('replacement', replacement_demo),
        ('performance', performance_comparison)
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
    print("=== Python Dataclasses and Attrs Demo ===")
    
    # Run all demonstrations
    all_results = run_all_dataclass_demos()
    
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
    
    print("\n=== DATACLASS FEATURES ===")
    
    features = {
        "@dataclass": "Decorator to automatically generate special methods",
        "field()": "Customize field behavior (default, factory, compare, repr)",
        "frozen=True": "Make dataclass immutable",
        "order=True": "Auto-generate comparison methods",
        "InitVar": "Fields only used during initialization",
        "__post_init__": "Custom initialization after auto-generated __init__",
        "asdict()": "Convert dataclass instance to dictionary",
        "astuple()": "Convert dataclass instance to tuple",
        "replace()": "Create new instance with modified fields",
        "fields()": "Get field information for a dataclass"
    }
    
    for feature, description in features.items():
        print(f"  {feature}: {description}")
    
    print("\n=== FIELD OPTIONS ===")
    
    field_options = {
        "default": "Default value for the field",
        "default_factory": "Function to generate default value",
        "init": "Include field in __init__ method (default: True)",
        "repr": "Include field in __repr__ method (default: True)",
        "compare": "Include field in comparison methods (default: True)",
        "hash": "Include field in __hash__ method (default: None)",
        "metadata": "Additional field metadata (dict)"
    }
    
    for option, description in field_options.items():
        print(f"  {option}: {description}")
    
    print("\n=== DATACLASS PARAMETERS ===")
    
    parameters = {
        "init": "Generate __init__ method (default: True)",
        "repr": "Generate __repr__ method (default: True)",
        "eq": "Generate __eq__ method (default: True)",
        "order": "Generate comparison methods (default: False)",
        "unsafe_hash": "Generate __hash__ method (default: False)",
        "frozen": "Make instances immutable (default: False)",
        "match_args": "Add __match_args__ for pattern matching (default: True)",
        "kw_only": "Make all fields keyword-only (default: False)",
        "slots": "Generate __slots__ (default: False)"
    }
    
    for param, description in parameters.items():
        print(f"  {param}: {description}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Use dataclasses for simple data containers",
        "Use field(default_factory=list) for mutable defaults",
        "Use frozen=True for immutable data structures",
        "Implement __post_init__ for validation and computed fields",
        "Use InitVar for fields only needed during initialization",
        "Use typing annotations for better code documentation",
        "Consider slots=True for memory optimization",
        "Use asdict() and from_dict() patterns for serialization",
        "Inherit from dataclasses for code reuse",
        "Use replace() for functional-style updates on frozen dataclasses"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== WHEN TO USE DATACLASSES ===")
    
    use_cases = [
        "Simple data containers with automatic method generation",
        "Configuration objects with validation",
        "API response/request models",
        "Database entity representations",
        "Immutable value objects (with frozen=True)",
        "Replacing named tuples with more features",
        "Data transfer objects between layers",
        "Event and message objects in event-driven systems"
    ]
    
    for use_case in use_cases:
        print(f"  • {use_case}")
    
    print("\n=== DATACLASSES vs ALTERNATIVES ===")
    
    alternatives = {
        "Regular Classes": "More control, but more boilerplate code",
        "Named Tuples": "Immutable, but less flexible than dataclasses",
        "Attrs Library": "More features, but external dependency",
        "Pydantic": "Better for data validation and API models",
        "TypedDict": "For dictionary-like data with type hints"
    }
    
    for alternative, note in alternatives.items():
        print(f"  {alternative}: {note}")
    
    print("\n=== Dataclasses and Modern Python Complete! ===")
    print("  Modern data class patterns and attribute management mastered")
