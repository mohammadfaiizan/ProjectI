"""
Python Type Hints and MyPy: Advanced Typing, Static Analysis, and Type Safety
Implementation-focused with minimal comments, maximum functionality coverage
"""

from typing import (
    Any, Dict, List, Tuple, Set, Optional, Union, Callable, TypeVar, Generic,
    Protocol, TypedDict, Final, Literal, ClassVar, Type, cast, overload,
    ForwardRef, get_type_hints, get_origin, get_args
)
from typing_extensions import (
    NotRequired, Required, Self, TypeGuard, TypeAlias, Annotated, ParamSpec, Concatenate
)
from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass, field
import functools
import inspect
import sys
from enum import Enum, auto
import json

# Basic type hints
def basic_types_demo():
    """Demonstrate basic type hint usage"""
    
    # Primitive types
    def process_number(value: int) -> str:
        return f"Number: {value}"
    
    def calculate_percentage(part: float, total: float) -> float:
        return (part / total) * 100.0
    
    def greet_user(name: str, age: int, active: bool = True) -> str:
        status = "active" if active else "inactive"
        return f"Hello {name}, age {age}, status: {status}"
    
    # Collection types
    def process_items(items: List[str]) -> Dict[str, int]:
        return {item: len(item) for item in items}
    
    def get_unique_numbers(numbers: List[int]) -> Set[int]:
        return set(numbers)
    
    def create_coordinates(x: float, y: float) -> Tuple[float, float]:
        return (x, y)
    
    # Optional and Union types
    def find_user(user_id: int) -> Optional[Dict[str, str]]:
        users = {1: {"name": "Alice", "email": "alice@example.com"}}
        return users.get(user_id)
    
    def process_value(value: Union[int, str, float]) -> str:
        if isinstance(value, int):
            return f"Integer: {value}"
        elif isinstance(value, str):
            return f"String: {value}"
        else:
            return f"Float: {value}"
    
    # Function types
    def apply_operation(x: int, y: int, operation: Callable[[int, int], int]) -> int:
        return operation(x, y)
    
    def create_multiplier(factor: int) -> Callable[[int], int]:
        def multiply(value: int) -> int:
            return value * factor
        return multiply
    
    # Test basic types
    number_result = process_number(42)
    percentage_result = calculate_percentage(25.0, 100.0)
    greeting_result = greet_user("Alice", 30, True)
    
    items_result = process_items(["hello", "world", "python"])
    unique_result = get_unique_numbers([1, 2, 2, 3, 3, 4])
    coordinates_result = create_coordinates(10.5, 20.3)
    
    user_result = find_user(1)
    user_not_found = find_user(999)
    
    process_int = process_value(42)
    process_str = process_value("hello")
    process_float = process_value(3.14)
    
    add_operation = lambda x, y: x + y
    operation_result = apply_operation(10, 5, add_operation)
    
    double = create_multiplier(2)
    multiplier_result = double(15)
    
    return {
        'primitive_types': {
            'number_result': number_result,
            'percentage_result': percentage_result,
            'greeting_result': greeting_result
        },
        'collection_types': {
            'items_result': items_result,
            'unique_result': list(unique_result),
            'coordinates_result': coordinates_result
        },
        'optional_union': {
            'user_found': user_result,
            'user_not_found': user_not_found,
            'process_int': process_int,
            'process_str': process_str,
            'process_float': process_float
        },
        'function_types': {
            'operation_result': operation_result,
            'multiplier_result': multiplier_result
        }
    }

# Generic types and TypeVar
def generics_demo():
    """Demonstrate generic types and TypeVar usage"""
    
    T = TypeVar('T')
    U = TypeVar('U')
    K = TypeVar('K')
    V = TypeVar('V')
    
    # Generic functions
    def first_item(items: List[T]) -> Optional[T]:
        return items[0] if items else None
    
    def last_item(items: List[T]) -> Optional[T]:
        return items[-1] if items else None
    
    def map_list(items: List[T], func: Callable[[T], U]) -> List[U]:
        return [func(item) for item in items]
    
    def filter_list(items: List[T], predicate: Callable[[T], bool]) -> List[T]:
        return [item for item in items if predicate(item)]
    
    def pair_items(first: T, second: U) -> Tuple[T, U]:
        return (first, second)
    
    # Generic classes
    class Stack(Generic[T]):
        def __init__(self) -> None:
            self._items: List[T] = []
        
        def push(self, item: T) -> None:
            self._items.append(item)
        
        def pop(self) -> Optional[T]:
            return self._items.pop() if self._items else None
        
        def peek(self) -> Optional[T]:
            return self._items[-1] if self._items else None
        
        def is_empty(self) -> bool:
            return len(self._items) == 0
        
        def size(self) -> int:
            return len(self._items)
    
    class Pair(Generic[T, U]):
        def __init__(self, first: T, second: U) -> None:
            self.first = first
            self.second = second
        
        def get_first(self) -> T:
            return self.first
        
        def get_second(self) -> U:
            return self.second
        
        def swap(self) -> 'Pair[U, T]':
            return Pair(self.second, self.first)
    
    class Cache(Generic[K, V]):
        def __init__(self, max_size: int = 100) -> None:
            self._data: Dict[K, V] = {}
            self._max_size = max_size
        
        def get(self, key: K) -> Optional[V]:
            return self._data.get(key)
        
        def put(self, key: K, value: V) -> None:
            if len(self._data) >= self._max_size:
                # Remove oldest item (simplified LRU)
                oldest_key = next(iter(self._data))
                del self._data[oldest_key]
            self._data[key] = value
        
        def contains(self, key: K) -> bool:
            return key in self._data
        
        def size(self) -> int:
            return len(self._data)
    
    # Bounded TypeVar
    Number = TypeVar('Number', int, float)
    
    def add_numbers(x: Number, y: Number) -> Number:
        return x + y  # type: ignore
    
    # Test generic functions
    numbers = [1, 2, 3, 4, 5]
    strings = ["hello", "world", "python"]
    
    first_number = first_item(numbers)
    first_string = first_item(strings)
    
    squared_numbers = map_list(numbers, lambda x: x ** 2)
    upper_strings = map_list(strings, str.upper)
    
    even_numbers = filter_list(numbers, lambda x: x % 2 == 0)
    long_strings = filter_list(strings, lambda s: len(s) > 5)
    
    number_string_pair = pair_items(42, "answer")
    boolean_float_pair = pair_items(True, 3.14)
    
    # Test generic classes
    int_stack: Stack[int] = Stack()
    int_stack.push(10)
    int_stack.push(20)
    int_stack.push(30)
    
    stack_size = int_stack.size()
    stack_top = int_stack.peek()
    popped_value = int_stack.pop()
    
    string_int_pair: Pair[str, int] = Pair("count", 42)
    swapped_pair = string_int_pair.swap()
    
    cache: Cache[str, int] = Cache(max_size=3)
    cache.put("one", 1)
    cache.put("two", 2)
    cache.put("three", 3)
    
    cached_value = cache.get("two")
    cache_contains = cache.contains("four")
    
    # Test bounded TypeVar
    int_sum = add_numbers(10, 20)
    float_sum = add_numbers(1.5, 2.5)
    
    return {
        'generic_functions': {
            'first_number': first_number,
            'first_string': first_string,
            'squared_numbers': squared_numbers,
            'upper_strings': upper_strings,
            'even_numbers': even_numbers,
            'long_strings': long_strings,
            'pairs': [number_string_pair, boolean_float_pair]
        },
        'generic_classes': {
            'stack_operations': {
                'size': stack_size,
                'top': stack_top,
                'popped': popped_value
            },
            'pair_operations': {
                'original': (string_int_pair.first, string_int_pair.second),
                'swapped': (swapped_pair.first, swapped_pair.second)
            },
            'cache_operations': {
                'cached_value': cached_value,
                'contains_four': cache_contains,
                'cache_size': cache.size()
            }
        },
        'bounded_typevar': {
            'int_sum': int_sum,
            'float_sum': float_sum
        }
    }

# Protocols and structural typing
def protocols_demo():
    """Demonstrate Protocol usage for structural typing"""
    
    # Define protocols
    class Drawable(Protocol):
        def draw(self) -> str:
            ...
    
    class Sizeable(Protocol):
        def get_size(self) -> int:
            ...
    
    class Comparable(Protocol):
        def __lt__(self, other: 'Comparable') -> bool:
            ...
        
        def __eq__(self, other: object) -> bool:
            ...
    
    class Serializable(Protocol):
        def serialize(self) -> Dict[str, Any]:
            ...
        
        @classmethod
        def deserialize(cls, data: Dict[str, Any]) -> 'Serializable':
            ...
    
    # Classes that implement protocols (duck typing)
    class Circle:
        def __init__(self, radius: float) -> None:
            self.radius = radius
        
        def draw(self) -> str:
            return f"Drawing circle with radius {self.radius}"
        
        def get_size(self) -> int:
            return int(3.14 * self.radius ** 2)
        
        def __lt__(self, other: 'Circle') -> bool:
            return self.radius < other.radius
        
        def __eq__(self, other: object) -> bool:
            return isinstance(other, Circle) and self.radius == other.radius
    
    class Rectangle:
        def __init__(self, width: float, height: float) -> None:
            self.width = width
            self.height = height
        
        def draw(self) -> str:
            return f"Drawing rectangle {self.width}x{self.height}"
        
        def get_size(self) -> int:
            return int(self.width * self.height)
        
        def serialize(self) -> Dict[str, Any]:
            return {"width": self.width, "height": self.height}
        
        @classmethod
        def deserialize(cls, data: Dict[str, Any]) -> 'Rectangle':
            return cls(data["width"], data["height"])
    
    class TextFile:
        def __init__(self, content: str) -> None:
            self.content = content
        
        def get_size(self) -> int:
            return len(self.content)
        
        def serialize(self) -> Dict[str, Any]:
            return {"content": self.content}
        
        @classmethod
        def deserialize(cls, data: Dict[str, Any]) -> 'TextFile':
            return cls(data["content"])
    
    # Functions that work with protocols
    def draw_shape(shape: Drawable) -> str:
        return shape.draw()
    
    def get_total_size(items: List[Sizeable]) -> int:
        return sum(item.get_size() for item in items)
    
    def sort_items(items: List[Comparable]) -> List[Comparable]:
        return sorted(items)
    
    def save_objects(objects: List[Serializable]) -> List[Dict[str, Any]]:
        return [obj.serialize() for obj in objects]
    
    # Advanced protocol with generics
    T_co = TypeVar('T_co', covariant=True)
    
    class Container(Protocol[T_co]):
        def get_item(self) -> T_co:
            ...
    
    class Box(Generic[T]):
        def __init__(self, item: T) -> None:
            self._item = item
        
        def get_item(self) -> T:
            return self._item
    
    def extract_from_container(container: Container[T]) -> T:
        return container.get_item()
    
    # Runtime protocol checking
    def is_drawable(obj: object) -> bool:
        return (hasattr(obj, 'draw') and 
                callable(getattr(obj, 'draw')))
    
    def is_sizeable(obj: object) -> bool:
        return (hasattr(obj, 'get_size') and 
                callable(getattr(obj, 'get_size')))
    
    # Test protocols
    circle = Circle(5.0)
    rectangle = Rectangle(10.0, 20.0)
    text_file = TextFile("Hello, World!")
    
    # Test drawable protocol
    circle_drawing = draw_shape(circle)
    rectangle_drawing = draw_shape(rectangle)
    
    # Test sizeable protocol
    sizeable_items = [circle, rectangle, text_file]
    total_size = get_total_size(sizeable_items)
    
    # Test comparable protocol
    circles = [Circle(3), Circle(1), Circle(5), Circle(2)]
    sorted_circles = sort_items(circles)
    sorted_radii = [c.radius for c in sorted_circles]
    
    # Test serializable protocol
    serializable_items = [rectangle, text_file]
    serialized_data = save_objects(serializable_items)
    
    # Test generic protocol
    string_box: Box[str] = Box("hello")
    int_box: Box[int] = Box(42)
    
    extracted_string = extract_from_container(string_box)
    extracted_int = extract_from_container(int_box)
    
    # Test runtime checking
    runtime_checks = {
        'circle_drawable': is_drawable(circle),
        'rectangle_drawable': is_drawable(rectangle),
        'text_file_drawable': is_drawable(text_file),
        'circle_sizeable': is_sizeable(circle),
        'rectangle_sizeable': is_sizeable(rectangle),
        'text_file_sizeable': is_sizeable(text_file)
    }
    
    return {
        'protocol_usage': {
            'circle_drawing': circle_drawing,
            'rectangle_drawing': rectangle_drawing,
            'total_size': total_size,
            'sorted_radii': sorted_radii,
            'serialized_count': len(serialized_data)
        },
        'generic_protocols': {
            'extracted_string': extracted_string,
            'extracted_int': extracted_int
        },
        'runtime_checks': runtime_checks
    }

# TypedDict and structured data
def typed_dict_demo():
    """Demonstrate TypedDict for structured dictionaries"""
    
    # Basic TypedDict
    class Person(TypedDict):
        name: str
        age: int
        email: str
    
    class PersonOptional(TypedDict, total=False):
        name: str
        age: int
        email: str
        phone: Optional[str]
    
    # TypedDict with required and not required fields
    class Product(TypedDict):
        id: int
        name: str
        price: float
        description: NotRequired[str]
        tags: NotRequired[List[str]]
    
    class UserProfile(TypedDict):
        user_id: Required[int]
        username: Required[str]
        email: NotRequired[str]
        full_name: NotRequired[str]
        preferences: NotRequired[Dict[str, Any]]
    
    # Inheritance with TypedDict
    class BaseEntity(TypedDict):
        id: int
        created_at: str
    
    class User(BaseEntity):
        username: str
        email: str
    
    class Post(BaseEntity):
        title: str
        content: str
        author_id: int
    
    # Functions working with TypedDict
    def create_person(name: str, age: int, email: str) -> Person:
        return {"name": name, "age": age, "email": email}
    
    def update_person(person: Person, **updates: Any) -> Person:
        updated = person.copy()
        for key, value in updates.items():
            if key in updated:
                updated[key] = value  # type: ignore
        return updated
    
    def get_person_info(person: Person) -> str:
        return f"{person['name']} ({person['age']}) - {person['email']}"
    
    def create_product(id: int, name: str, price: float, **kwargs: Any) -> Product:
        product: Product = {"id": id, "name": name, "price": price}
        
        if "description" in kwargs:
            product["description"] = kwargs["description"]
        if "tags" in kwargs:
            product["tags"] = kwargs["tags"]
        
        return product
    
    def calculate_total_price(products: List[Product]) -> float:
        return sum(product["price"] for product in products)
    
    def filter_products_by_tag(products: List[Product], tag: str) -> List[Product]:
        return [
            product for product in products 
            if "tags" in product and tag in product["tags"]
        ]
    
    # JSON serialization with TypedDict
    def serialize_person(person: Person) -> str:
        return json.dumps(person)
    
    def deserialize_person(json_data: str) -> Person:
        data = json.loads(json_data)
        return {"name": data["name"], "age": data["age"], "email": data["email"]}
    
    # Validation functions
    def validate_person(data: Dict[str, Any]) -> bool:
        required_fields = {"name", "age", "email"}
        return (
            all(field in data for field in required_fields) and
            isinstance(data["name"], str) and
            isinstance(data["age"], int) and
            isinstance(data["email"], str)
        )
    
    def validate_product(data: Dict[str, Any]) -> bool:
        required_fields = {"id", "name", "price"}
        return (
            all(field in data for field in required_fields) and
            isinstance(data["id"], int) and
            isinstance(data["name"], str) and
            isinstance(data["price"], (int, float))
        )
    
    # Test TypedDict usage
    person1 = create_person("Alice", 30, "alice@example.com")
    person2 = update_person(person1, age=31, email="alice.new@example.com")
    
    person_info = get_person_info(person2)
    
    # Test optional fields
    person_optional: PersonOptional = {"name": "Bob", "age": 25}
    person_optional["phone"] = "+1234567890"
    
    # Test products
    products = [
        create_product(1, "Laptop", 999.99, description="Gaming laptop", tags=["electronics", "gaming"]),
        create_product(2, "Mouse", 29.99, tags=["electronics", "accessories"]),
        create_product(3, "Book", 19.99, description="Python programming guide")
    ]
    
    total_price = calculate_total_price(products)
    electronics = filter_products_by_tag(products, "electronics")
    
    # Test inheritance
    user: User = {
        "id": 1,
        "created_at": "2023-01-01T00:00:00Z",
        "username": "alice123",
        "email": "alice@example.com"
    }
    
    post: Post = {
        "id": 1,
        "created_at": "2023-01-02T10:00:00Z",
        "title": "Hello World",
        "content": "This is my first post!",
        "author_id": user["id"]
    }
    
    # Test serialization
    serialized = serialize_person(person1)
    deserialized = deserialize_person(serialized)
    
    # Test validation
    valid_person_data = {"name": "Charlie", "age": 35, "email": "charlie@example.com"}
    invalid_person_data = {"name": "Dave", "age": "thirty"}  # Invalid age type
    
    person_validation = {
        "valid_person": validate_person(valid_person_data),
        "invalid_person": validate_person(invalid_person_data)
    }
    
    return {
        'basic_typeddict': {
            'person1': person1,
            'person2': person2,
            'person_info': person_info,
            'person_optional': person_optional
        },
        'product_operations': {
            'product_count': len(products),
            'total_price': total_price,
            'electronics_count': len(electronics)
        },
        'inheritance': {
            'user': user,
            'post_title': post["title"],
            'post_author_matches': post["author_id"] == user["id"]
        },
        'serialization': {
            'serialized_length': len(serialized),
            'roundtrip_successful': person1 == deserialized
        },
        'validation': person_validation
    }

# Advanced typing features
def advanced_typing_demo():
    """Demonstrate advanced typing features"""
    
    # Literal types
    Mode = Literal["read", "write", "append"]
    Status = Literal["active", "inactive", "pending"]
    
    def open_file(filename: str, mode: Mode) -> str:
        return f"Opening {filename} in {mode} mode"
    
    def set_user_status(user_id: int, status: Status) -> str:
        return f"User {user_id} status set to {status}"
    
    # Final and ClassVar
    class Configuration:
        DEFAULT_TIMEOUT: Final[int] = 30
        instance_count: ClassVar[int] = 0
        
        def __init__(self, name: str) -> None:
            self.name: Final[str] = name
            Configuration.instance_count += 1
    
    # Type aliases
    UserId: TypeAlias = int
    UserName: TypeAlias = str
    UserData: TypeAlias = Dict[str, Union[str, int, bool]]
    
    def get_user_by_id(user_id: UserId) -> Optional[UserData]:
        users = {
            1: {"name": "Alice", "age": 30, "active": True},
            2: {"name": "Bob", "age": 25, "active": False}
        }
        return users.get(user_id)
    
    def create_username(first_name: str, last_name: str) -> UserName:
        return f"{first_name.lower()}.{last_name.lower()}"
    
    # Annotated types
    from typing import Annotated
    
    PositiveInt = Annotated[int, "Must be positive"]
    EmailString = Annotated[str, "Must be valid email format"]
    
    def create_user_account(user_id: PositiveInt, email: EmailString) -> Dict[str, Any]:
        return {"id": user_id, "email": email, "created": True}
    
    # Overloaded functions
    @overload
    def process_data(data: str) -> str:
        ...
    
    @overload
    def process_data(data: int) -> int:
        ...
    
    @overload
    def process_data(data: List[str]) -> List[str]:
        ...
    
    def process_data(data: Union[str, int, List[str]]) -> Union[str, int, List[str]]:
        if isinstance(data, str):
            return data.upper()
        elif isinstance(data, int):
            return data * 2
        else:
            return [item.upper() for item in data]
    
    # Type guards
    def is_string_list(obj: object) -> TypeGuard[List[str]]:
        return (isinstance(obj, list) and 
                all(isinstance(item, str) for item in obj))
    
    def is_int_dict(obj: object) -> TypeGuard[Dict[str, int]]:
        return (isinstance(obj, dict) and 
                all(isinstance(k, str) and isinstance(v, int) 
                    for k, v in obj.items()))
    
    def safe_process_list(data: object) -> Optional[List[str]]:
        if is_string_list(data):
            return [item.upper() for item in data]
        return None
    
    def safe_sum_values(data: object) -> Optional[int]:
        if is_int_dict(data):
            return sum(data.values())
        return None
    
    # NewType for strong typing
    from typing import NewType
    
    UserId = NewType('UserId', int)
    ProductId = NewType('ProductId', int)
    
    def get_user_orders(user_id: UserId) -> List[ProductId]:
        # Simulate getting orders for a user
        orders = {
            UserId(1): [ProductId(101), ProductId(102)],
            UserId(2): [ProductId(103)]
        }
        return orders.get(user_id, [])
    
    def get_product_name(product_id: ProductId) -> str:
        products = {
            ProductId(101): "Laptop",
            ProductId(102): "Mouse",
            ProductId(103): "Keyboard"
        }
        return products.get(product_id, "Unknown Product")
    
    # Callable with ParamSpec
    P = ParamSpec('P')
    R = TypeVar('R')
    
    def timing_decorator(func: Callable[P, R]) -> Callable[P, R]:
        import time
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} took {end - start:.4f} seconds")
            return result
        return wrapper
    
    @timing_decorator
    def expensive_function(n: int, multiplier: float = 1.0) -> float:
        import time
        time.sleep(0.01)
        return n * multiplier
    
    # Test advanced features
    file_operation = open_file("data.txt", "read")
    status_change = set_user_status(1, "active")
    
    config1 = Configuration("prod")
    config2 = Configuration("dev")
    instance_count = Configuration.instance_count
    
    user_data = get_user_by_id(UserId(1))
    username = create_username("John", "Doe")
    
    account = create_user_account(PositiveInt(123), EmailString("test@example.com"))
    
    # Test overloads
    processed_string = process_data("hello")
    processed_int = process_data(42)
    processed_list = process_data(["hello", "world"])
    
    # Test type guards
    test_data = ["hello", "world", "python"]
    mixed_data = ["hello", 123, "world"]
    int_dict_data = {"a": 1, "b": 2, "c": 3}
    
    safe_list_result = safe_process_list(test_data)
    safe_list_result_mixed = safe_process_list(mixed_data)
    safe_sum_result = safe_sum_values(int_dict_data)
    
    # Test NewType
    user_orders = get_user_orders(UserId(1))
    product_names = [get_product_name(pid) for pid in user_orders]
    
    # Test decorated function
    expensive_result = expensive_function(10, 2.5)
    
    return {
        'literal_types': {
            'file_operation': file_operation,
            'status_change': status_change
        },
        'final_classvar': {
            'instance_count': instance_count,
            'default_timeout': Configuration.DEFAULT_TIMEOUT
        },
        'type_aliases': {
            'user_data': user_data,
            'username': username,
            'account': account
        },
        'overloads': {
            'processed_string': processed_string,
            'processed_int': processed_int,
            'processed_list': processed_list
        },
        'type_guards': {
            'safe_list_success': safe_list_result,
            'safe_list_failure': safe_list_result_mixed,
            'safe_sum_result': safe_sum_result
        },
        'newtype': {
            'user_orders': user_orders,
            'product_names': product_names
        },
        'paramspec': {
            'expensive_result': expensive_result
        }
    }

# Type introspection and runtime checking
def type_introspection_demo():
    """Demonstrate type introspection and runtime type checking"""
    
    # Type introspection utilities
    def analyze_type_hints(func: Callable) -> Dict[str, Any]:
        hints = get_type_hints(func)
        signature = inspect.signature(func)
        
        return {
            'function_name': func.__name__,
            'type_hints': {name: str(hint) for name, hint in hints.items()},
            'parameters': {
                name: {
                    'annotation': str(param.annotation) if param.annotation != param.empty else None,
                    'default': param.default if param.default != param.empty else None
                }
                for name, param in signature.parameters.items()
            }
        }
    
    def get_generic_info(tp: Any) -> Dict[str, Any]:
        origin = get_origin(tp)
        args = get_args(tp)
        
        return {
            'original_type': str(tp),
            'origin': str(origin) if origin else None,
            'args': [str(arg) for arg in args] if args else []
        }
    
    # Runtime type validation
    def validate_type(value: Any, expected_type: Any) -> bool:
        """Simple runtime type validation"""
        try:
            if expected_type == Any:
                return True
            
            origin = get_origin(expected_type)
            args = get_args(expected_type)
            
            if origin is None:
                return isinstance(value, expected_type)
            
            if origin is Union:
                return any(validate_type(value, arg) for arg in args)
            
            if origin is list:
                if not isinstance(value, list):
                    return False
                if args:
                    return all(validate_type(item, args[0]) for item in value)
                return True
            
            if origin is dict:
                if not isinstance(value, dict):
                    return False
                if len(args) == 2:
                    key_type, value_type = args
                    return all(
                        validate_type(k, key_type) and validate_type(v, value_type)
                        for k, v in value.items()
                    )
                return True
            
            if origin is tuple:
                if not isinstance(value, tuple):
                    return False
                if args:
                    if len(args) == len(value):
                        return all(validate_type(v, t) for v, t in zip(value, args))
                return True
            
            return isinstance(value, origin)
        
        except Exception:
            return False
    
    def create_type_validator(func: Callable) -> Callable:
        """Create a decorator that validates function arguments at runtime"""
        hints = get_type_hints(func)
        signature = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            errors = []
            for param_name, value in bound_args.arguments.items():
                if param_name in hints:
                    expected_type = hints[param_name]
                    if not validate_type(value, expected_type):
                        errors.append(f"{param_name}: expected {expected_type}, got {type(value)}")
            
            if errors:
                raise TypeError(f"Type validation failed: {', '.join(errors)}")
            
            result = func(*args, **kwargs)
            
            # Validate return type
            if 'return' in hints:
                return_type = hints['return']
                if not validate_type(result, return_type):
                    raise TypeError(f"Return type validation failed: expected {return_type}, got {type(result)}")
            
            return result
        
        return wrapper
    
    # Test functions for analysis
    def sample_function(x: int, y: str, z: Optional[float] = None) -> Tuple[int, str]:
        return (x, y)
    
    def generic_function(items: List[Dict[str, int]]) -> Optional[Dict[str, Any]]:
        if items:
            return {"count": len(items), "first": items[0]}
        return None
    
    @create_type_validator
    def validated_function(numbers: List[int], factor: float) -> List[float]:
        return [n * factor for n in numbers]
    
    @create_type_validator
    def another_validated_function(data: Dict[str, Union[int, str]]) -> str:
        return json.dumps(data)
    
    # Type checking at runtime
    class TypeChecker:
        @staticmethod
        def check_list_of_ints(value: Any) -> bool:
            return validate_type(value, List[int])
        
        @staticmethod
        def check_dict_str_any(value: Any) -> bool:
            return validate_type(value, Dict[str, Any])
        
        @staticmethod
        def check_optional_str(value: Any) -> bool:
            return validate_type(value, Optional[str])
        
        @staticmethod
        def check_union_int_str(value: Any) -> bool:
            return validate_type(value, Union[int, str])
    
    # Test type introspection
    sample_analysis = analyze_type_hints(sample_function)
    generic_analysis = analyze_type_hints(generic_function)
    
    # Test generic type info
    list_int_info = get_generic_info(List[int])
    dict_str_any_info = get_generic_info(Dict[str, Any])
    optional_str_info = get_generic_info(Optional[str])
    union_info = get_generic_info(Union[int, str, float])
    
    # Test runtime validation
    validation_tests = {
        'valid_list_int': TypeChecker.check_list_of_ints([1, 2, 3]),
        'invalid_list_int': TypeChecker.check_list_of_ints([1, "2", 3]),
        'valid_dict': TypeChecker.check_dict_str_any({"a": 1, "b": "hello"}),
        'invalid_dict': TypeChecker.check_dict_str_any({1: "invalid_key"}),
        'valid_optional_str': TypeChecker.check_optional_str("hello"),
        'valid_optional_none': TypeChecker.check_optional_str(None),
        'invalid_optional': TypeChecker.check_optional_str(123),
        'valid_union_int': TypeChecker.check_union_int_str(42),
        'valid_union_str': TypeChecker.check_union_int_str("hello"),
        'invalid_union': TypeChecker.check_union_int_str([1, 2, 3])
    }
    
    # Test validated functions
    valid_result = validated_function([1, 2, 3], 2.5)
    
    try:
        validated_function(["1", "2", "3"], 2.5)  # Should fail
        validation_caught = False
    except TypeError:
        validation_caught = True
    
    valid_json_result = another_validated_function({"name": "Alice", "age": 30})
    
    return {
        'function_analysis': {
            'sample_function': sample_analysis,
            'generic_function': generic_analysis
        },
        'generic_type_info': {
            'list_int': list_int_info,
            'dict_str_any': dict_str_any_info,
            'optional_str': optional_str_info,
            'union': union_info
        },
        'validation_tests': validation_tests,
        'runtime_validation': {
            'valid_result': valid_result,
            'validation_caught_error': validation_caught,
            'valid_json_length': len(valid_json_result)
        }
    }

# MyPy integration and static analysis
def mypy_integration_demo():
    """Demonstrate MyPy integration concepts"""
    
    # MyPy configuration examples (would be in mypy.ini)
    mypy_config_example = """
    [mypy]
    python_version = 3.8
    warn_return_any = True
    warn_unused_configs = True
    disallow_untyped_defs = True
    disallow_incomplete_defs = True
    check_untyped_defs = True
    disallow_untyped_decorators = True
    no_implicit_optional = True
    warn_redundant_casts = True
    warn_unused_ignores = True
    warn_no_return = True
    warn_unreachable = True
    strict_equality = True
    """
    
    # Examples of MyPy directives
    def function_with_ignore() -> Any:
        x: int = "hello"  # type: ignore
        return x
    
    def function_with_type_comment():
        # type: () -> int
        return 42
    
    # Gradual typing examples
    def untyped_function(x, y):
        return x + y
    
    def partially_typed_function(x: int, y) -> int:
        return x + y  # type: ignore
    
    def fully_typed_function(x: int, y: int) -> int:
        return x + y
    
    # MyPy plugins and extensions (conceptual)
    class DataClassExample:
        """Example showing dataclass integration with MyPy"""
        
        @dataclass
        class Point:
            x: float
            y: float
            
            def distance_from_origin(self) -> float:
                return (self.x ** 2 + self.y ** 2) ** 0.5
        
        @dataclass
        class Person:
            name: str
            age: int
            email: Optional[str] = None
            
            def is_adult(self) -> bool:
                return self.age >= 18
    
    # Type narrowing examples
    def narrow_union_type(value: Union[str, int, None]) -> str:
        if value is None:
            return "none"
        elif isinstance(value, str):
            return value.upper()  # MyPy knows this is str
        else:
            return str(value)  # MyPy knows this is int
    
    def narrow_optional_type(value: Optional[List[int]]) -> int:
        if value is not None:
            return len(value)  # MyPy knows value is List[int]
        return 0
    
    # Reveal types (for MyPy debugging)
    def reveal_types_example():
        x = [1, 2, 3]
        # reveal_type(x)  # Would show List[int] in MyPy
        
        y = {"a": 1, "b": 2}
        # reveal_type(y)  # Would show Dict[str, int] in MyPy
        
        z = x[0] if x else None
        # reveal_type(z)  # Would show int in MyPy
        
        return x, y, z
    
    # Stub file examples (conceptual)
    stub_file_example = """
    # third_party_library.pyi
    from typing import Any, Optional, List
    
    class ExternalClass:
        def __init__(self, value: Any) -> None: ...
        def process(self, data: List[str]) -> Optional[str]: ...
        @property
        def result(self) -> Any: ...
    
    def external_function(x: int, y: str) -> bool: ...
    """
    
    # Common MyPy errors and solutions
    def demonstrate_common_issues():
        # Issue 1: Missing return type
        # def bad_function(x):  # error: Function is missing a type annotation
        #     return x * 2
        
        def good_function(x: int) -> int:
            return x * 2
        
        # Issue 2: Incompatible types
        # def bad_assignment() -> None:
        #     x: int = "hello"  # error: Incompatible types
        
        def good_assignment() -> None:
            x: int = 42
        
        # Issue 3: Optional handling
        def handle_optional_properly(value: Optional[str]) -> str:
            if value is not None:
                return value.upper()
            return ""
        
        # Issue 4: Union type handling
        def handle_union_properly(value: Union[int, str]) -> str:
            if isinstance(value, int):
                return str(value)
            return value
        
        return {
            'good_function_result': good_function(5),
            'optional_handling': handle_optional_properly("hello"),
            'union_handling_int': handle_union_properly(42),
            'union_handling_str': handle_union_properly("world")
        }
    
    # Testing with type annotations
    def test_function_types():
        """Examples of how type annotations help with testing"""
        
        def divide(a: float, b: float) -> float:
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b
        
        def find_max(numbers: List[int]) -> Optional[int]:
            return max(numbers) if numbers else None
        
        # Type-aware test cases
        test_results = {
            'divide_positive': divide(10.0, 2.0),
            'divide_negative': divide(-10.0, 2.0),
            'find_max_with_values': find_max([1, 5, 3, 9, 2]),
            'find_max_empty': find_max([])
        }
        
        return test_results
    
    # Run demonstrations
    narrowing_example1 = narrow_union_type("hello")
    narrowing_example2 = narrow_union_type(42)
    narrowing_example3 = narrow_union_type(None)
    
    optional_example1 = narrow_optional_type([1, 2, 3, 4])
    optional_example2 = narrow_optional_type(None)
    
    reveal_results = reveal_types_example()
    common_issues_results = demonstrate_common_issues()
    test_results = test_function_types()
    
    # Create sample dataclass instances
    point = DataClassExample.Point(3.0, 4.0)
    person = DataClassExample.Person("Alice", 25, "alice@example.com")
    
    return {
        'mypy_config': {
            'config_length': len(mypy_config_example.split('\n')),
            'strict_mode_enabled': 'disallow_untyped_defs = True' in mypy_config_example
        },
        'type_narrowing': {
            'string_narrowing': narrowing_example1,
            'int_narrowing': narrowing_example2,
            'none_narrowing': narrowing_example3,
            'optional_with_value': optional_example1,
            'optional_none': optional_example2
        },
        'dataclass_integration': {
            'point_distance': point.distance_from_origin(),
            'person_is_adult': person.is_adult(),
            'person_email_present': person.email is not None
        },
        'common_patterns': common_issues_results,
        'testing_integration': test_results,
        'stub_file_example_length': len(stub_file_example.split('\n'))
    }

# Comprehensive testing
def run_all_typing_demos():
    """Execute all type hints and MyPy demonstrations"""
    demo_functions = [
        ('basic_types', basic_types_demo),
        ('generics', generics_demo),
        ('protocols', protocols_demo),
        ('typed_dict', typed_dict_demo),
        ('advanced_typing', advanced_typing_demo),
        ('type_introspection', type_introspection_demo),
        ('mypy_integration', mypy_integration_demo)
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
    print("=== Python Type Hints and MyPy Demo ===")
    
    # Run all demonstrations
    all_results = run_all_typing_demos()
    
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
    
    print("\n=== TYPE HINT CONCEPTS ===")
    
    concepts = {
        "Type Hints": "Annotations that specify expected types for variables and functions",
        "Generic Types": "Types that work with multiple types using TypeVar",
        "Protocol": "Structural typing based on method signatures",
        "TypedDict": "Typed dictionaries with specific key-value types",
        "Union Types": "Allow values to be one of several types",
        "Optional": "Shorthand for Union[T, None]",
        "Literal": "Restrict values to specific literal values",
        "Type Aliases": "Create readable names for complex types",
        "NewType": "Create distinct types from existing types"
    }
    
    for concept, description in concepts.items():
        print(f"  {concept}: {description}")
    
    print("\n=== MYPY BENEFITS ===")
    
    benefits = [
        "Catch type errors before runtime",
        "Improve code documentation and readability",
        "Enable better IDE support and autocompletion",
        "Facilitate refactoring with confidence",
        "Enforce API contracts and interfaces",
        "Reduce debugging time",
        "Support gradual typing adoption",
        "Integrate with CI/CD pipelines"
    ]
    
    for benefit in benefits:
        print(f"  • {benefit}")
    
    print("\n=== TYPING BEST PRACTICES ===")
    
    best_practices = [
        "Start with function signatures, then add variable annotations",
        "Use Union sparingly, prefer Protocol for flexibility",
        "Leverage TypedDict for structured dictionary data",
        "Use Generic types for reusable data structures",
        "Apply Literal types for string/enum-like constants",
        "Use Optional explicitly instead of implicit None",
        "Create type aliases for complex type expressions",
        "Use Protocol for duck typing and interfaces",
        "Add return type annotations to all functions",
        "Use Final for constants that shouldn't change",
        "Leverage dataclasses with type hints",
        "Use mypy in CI/CD to enforce type checking",
        "Start with basic types and gradually add complexity",
        "Use type comments for Python < 3.6 compatibility"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== MYPY CONFIGURATION ===")
    
    config_options = {
        "strict": "Enable all strict checking options",
        "disallow_untyped_defs": "Require type annotations for functions",
        "warn_return_any": "Warn when returning Any type",
        "warn_unused_ignores": "Warn about unused type: ignore comments",
        "no_implicit_optional": "Don't treat arguments with None default as Optional",
        "warn_redundant_casts": "Warn about unnecessary type casts",
        "check_untyped_defs": "Type check unannotated functions",
        "show_error_codes": "Show error codes in output"
    }
    
    for option, description in config_options.items():
        print(f"  {option}: {description}")
    
    print("\n=== Type Hints and MyPy Complete! ===")
    print("  Advanced typing and static analysis mastered")
