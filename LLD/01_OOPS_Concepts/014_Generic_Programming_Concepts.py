"""
GENERIC PROGRAMMING CONCEPTS - Type Parameterization
====================================================

Problem Statement:
Demonstrate generic programming concepts in Python:
- Type variables and generic classes
- Generic functions and methods
- Constraints and bounds on type parameters
- Variance in generic types (covariance, contravariance)
- Generic protocols and interfaces

Learning Objectives:
- Understand generic programming principles
- Implement type-safe generic classes and functions
- Use type constraints effectively
- Handle variance in generic types
- Create reusable generic components
"""

from typing import (
    TypeVar, Generic, List, Dict, Optional, Union, Callable, 
    Protocol, runtime_checkable, Any, Tuple, Iterator
)
from abc import ABC, abstractmethod
from datetime import datetime
import json


# Type variables for generic programming
T = TypeVar('T')  # Generic type
K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type
N = TypeVar('N', int, float)  # Constrained to numeric types
Comparable = TypeVar('Comparable', bound='ComparableProtocol')  # Bounded type variable


# Protocol for comparable objects
@runtime_checkable
class ComparableProtocol(Protocol):
    """Protocol for objects that can be compared."""
    
    def __lt__(self, other: 'ComparableProtocol') -> bool:
        """Less than comparison."""
        ...
    
    def __le__(self, other: 'ComparableProtocol') -> bool:
        """Less than or equal comparison."""
        ...
    
    def __gt__(self, other: 'ComparableProtocol') -> bool:
        """Greater than comparison."""
        ...
    
    def __ge__(self, other: 'ComparableProtocol') -> bool:
        """Greater than or equal comparison."""
        ...


# Generic Stack implementation
class Stack(Generic[T]):
    """Generic stack data structure."""
    
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        """Push item onto stack."""
        self._items.append(item)
    
    def pop(self) -> Optional[T]:
        """Pop item from stack."""
        if self._items:
            return self._items.pop()
        return None
    
    def peek(self) -> Optional[T]:
        """Peek at top item without removing."""
        if self._items:
            return self._items[-1]
        return None
    
    def is_empty(self) -> bool:
        """Check if stack is empty."""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get stack size."""
        return len(self._items)
    
    def to_list(self) -> List[T]:
        """Convert stack to list."""
        return self._items.copy()
    
    def __str__(self) -> str:
        return f"Stack({self._items})"


# Generic Queue implementation
class Queue(Generic[T]):
    """Generic queue data structure."""
    
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def enqueue(self, item: T) -> None:
        """Add item to rear of queue."""
        self._items.append(item)
    
    def dequeue(self) -> Optional[T]:
        """Remove item from front of queue."""
        if self._items:
            return self._items.pop(0)
        return None
    
    def front(self) -> Optional[T]:
        """Get front item without removing."""
        if self._items:
            return self._items[0]
        return None
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._items) == 0
    
    def size(self) -> int:
        """Get queue size."""
        return len(self._items)
    
    def __str__(self) -> str:
        return f"Queue({self._items})"


# Generic Key-Value Store
class KeyValueStore(Generic[K, V]):
    """Generic key-value store."""
    
    def __init__(self) -> None:
        self._data: Dict[K, V] = {}
        self._access_count: Dict[K, int] = {}
    
    def put(self, key: K, value: V) -> None:
        """Store key-value pair."""
        self._data[key] = value
        self._access_count[key] = self._access_count.get(key, 0)
    
    def get(self, key: K) -> Optional[V]:
        """Get value by key."""
        if key in self._data:
            self._access_count[key] += 1
            return self._data[key]
        return None
    
    def remove(self, key: K) -> bool:
        """Remove key-value pair."""
        if key in self._data:
            del self._data[key]
            del self._access_count[key]
            return True
        return False
    
    def contains(self, key: K) -> bool:
        """Check if key exists."""
        return key in self._data
    
    def keys(self) -> List[K]:
        """Get all keys."""
        return list(self._data.keys())
    
    def values(self) -> List[V]:
        """Get all values."""
        return list(self._data.values())
    
    def items(self) -> List[Tuple[K, V]]:
        """Get all key-value pairs."""
        return list(self._data.items())
    
    def get_access_count(self, key: K) -> int:
        """Get access count for key."""
        return self._access_count.get(key, 0)
    
    def size(self) -> int:
        """Get store size."""
        return len(self._data)
    
    def __str__(self) -> str:
        return f"KeyValueStore({dict(list(self._data.items())[:3])}{'...' if len(self._data) > 3 else ''})"


# Generic Repository pattern
class Repository(Generic[T], ABC):
    """Generic repository interface."""
    
    @abstractmethod
    def save(self, entity: T) -> bool:
        """Save entity."""
        pass
    
    @abstractmethod
    def find_by_id(self, entity_id: str) -> Optional[T]:
        """Find entity by ID."""
        pass
    
    @abstractmethod
    def find_all(self) -> List[T]:
        """Find all entities."""
        pass
    
    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """Delete entity."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Count entities."""
        pass


# Entity base class
class Entity:
    """Base entity class."""
    
    def __init__(self, entity_id: str):
        self.id = entity_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()


# Concrete entity classes
class User(Entity):
    """User entity."""
    
    def __init__(self, user_id: str, name: str, email: str):
        super().__init__(user_id)
        self.name = name
        self.email = email
    
    def __str__(self) -> str:
        return f"User(id={self.id}, name={self.name}, email={self.email})"


class Product(Entity):
    """Product entity."""
    
    def __init__(self, product_id: str, name: str, price: float, category: str):
        super().__init__(product_id)
        self.name = name
        self.price = price
        self.category = category
    
    def __str__(self) -> str:
        return f"Product(id={self.id}, name={self.name}, price=${self.price:.2f})"


# Generic in-memory repository implementation
class InMemoryRepository(Repository[T]):
    """Generic in-memory repository implementation."""
    
    def __init__(self) -> None:
        self._entities: Dict[str, T] = {}
    
    def save(self, entity: T) -> bool:
        """Save entity."""
        if hasattr(entity, 'id'):
            if hasattr(entity, 'update_timestamp'):
                entity.update_timestamp()
            self._entities[entity.id] = entity
            return True
        return False
    
    def find_by_id(self, entity_id: str) -> Optional[T]:
        """Find entity by ID."""
        return self._entities.get(entity_id)
    
    def find_all(self) -> List[T]:
        """Find all entities."""
        return list(self._entities.values())
    
    def delete(self, entity_id: str) -> bool:
        """Delete entity."""
        if entity_id in self._entities:
            del self._entities[entity_id]
            return True
        return False
    
    def count(self) -> int:
        """Count entities."""
        return len(self._entities)
    
    def find_by_predicate(self, predicate: Callable[[T], bool]) -> List[T]:
        """Find entities matching predicate."""
        return [entity for entity in self._entities.values() if predicate(entity)]


# Generic functions
def find_max(items: List[Comparable]) -> Optional[Comparable]:
    """Find maximum item in list (bounded type variable)."""
    if not items:
        return None
    
    max_item = items[0]
    for item in items[1:]:
        if item > max_item:
            max_item = item
    return max_item


def find_min(items: List[Comparable]) -> Optional[Comparable]:
    """Find minimum item in list."""
    if not items:
        return None
    
    min_item = items[0]
    for item in items[1:]:
        if item < min_item:
            min_item = item
    return min_item


def sort_items(items: List[Comparable]) -> List[Comparable]:
    """Sort items in ascending order."""
    return sorted(items)


def map_function(items: List[T], func: Callable[[T], V]) -> List[V]:
    """Map function over list of items."""
    return [func(item) for item in items]


def filter_function(items: List[T], predicate: Callable[[T], bool]) -> List[T]:
    """Filter items based on predicate."""
    return [item for item in items if predicate(item)]


def reduce_function(items: List[T], func: Callable[[V, T], V], initial: V) -> V:
    """Reduce items using function."""
    result = initial
    for item in items:
        result = func(result, item)
    return result


# Numeric operations with constrained type variables
def add_numbers(a: N, b: N) -> N:
    """Add two numbers (constrained to int or float)."""
    return a + b


def multiply_numbers(a: N, b: N) -> N:
    """Multiply two numbers."""
    return a * b


def calculate_average(numbers: List[N]) -> Optional[float]:
    """Calculate average of numbers."""
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


# Generic cache implementation
class Cache(Generic[K, V]):
    """Generic LRU cache implementation."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[K, V] = {}
        self._access_order: List[K] = []
    
    def get(self, key: K) -> Optional[V]:
        """Get value from cache."""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key: K, value: V) -> None:
        """Put value in cache."""
        if key in self._cache:
            # Update existing
            self._cache[key] = value
            self._access_order.remove(key)
            self._access_order.append(key)
        else:
            # Add new
            if len(self._cache) >= self.max_size:
                # Remove least recently used
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
            
            self._cache[key] = value
            self._access_order.append(key)
    
    def remove(self, key: K) -> bool:
        """Remove key from cache."""
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)
            return True
        return False
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)
    
    def keys(self) -> List[K]:
        """Get cache keys in access order."""
        return self._access_order.copy()


# Generic event system
class Event(Generic[T]):
    """Generic event class."""
    
    def __init__(self, event_type: str, data: T, timestamp: Optional[datetime] = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.now()
        self.event_id = f"{event_type}_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
    
    def __str__(self) -> str:
        return f"Event({self.event_type}, {self.event_id})"


class EventHandler(Generic[T], ABC):
    """Generic event handler interface."""
    
    @abstractmethod
    def handle(self, event: Event[T]) -> bool:
        """Handle event."""
        pass
    
    @abstractmethod
    def can_handle(self, event_type: str) -> bool:
        """Check if handler can handle event type."""
        pass


class EventBus(Generic[T]):
    """Generic event bus."""
    
    def __init__(self):
        self._handlers: List[EventHandler[T]] = []
        self._event_history: List[Event[T]] = []
    
    def register_handler(self, handler: EventHandler[T]) -> None:
        """Register event handler."""
        self._handlers.append(handler)
    
    def unregister_handler(self, handler: EventHandler[T]) -> None:
        """Unregister event handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)
    
    def publish(self, event: Event[T]) -> int:
        """Publish event to all applicable handlers."""
        self._event_history.append(event)
        handled_count = 0
        
        for handler in self._handlers:
            if handler.can_handle(event.event_type):
                if handler.handle(event):
                    handled_count += 1
        
        return handled_count
    
    def get_event_history(self) -> List[Event[T]]:
        """Get event history."""
        return self._event_history.copy()


# Concrete event handler implementations
class UserEventHandler(EventHandler[User]):
    """Handler for user events."""
    
    def handle(self, event: Event[User]) -> bool:
        """Handle user event."""
        print(f"Handling user event: {event.event_type} for user {event.data.name}")
        return True
    
    def can_handle(self, event_type: str) -> bool:
        """Check if can handle event type."""
        return event_type.startswith("user_")


class ProductEventHandler(EventHandler[Product]):
    """Handler for product events."""
    
    def handle(self, event: Event[Product]) -> bool:
        """Handle product event."""
        print(f"Handling product event: {event.event_type} for product {event.data.name}")
        return True
    
    def can_handle(self, event_type: str) -> bool:
        """Check if can handle event type."""
        return event_type.startswith("product_")


def demonstrate_generic_programming():
    """
    Demonstrate generic programming concepts.
    """
    print("=== GENERIC PROGRAMMING CONCEPTS DEMONSTRATION ===\n")
    
    # 1. Generic Data Structures
    print("1. GENERIC DATA STRUCTURES:")
    
    # Generic Stack
    int_stack: Stack[int] = Stack()
    int_stack.push(1)
    int_stack.push(2)
    int_stack.push(3)
    print(f"Integer stack: {int_stack}")
    print(f"Popped: {int_stack.pop()}")
    print(f"Peek: {int_stack.peek()}")
    
    # String Stack
    str_stack: Stack[str] = Stack()
    str_stack.push("hello")
    str_stack.push("world")
    print(f"String stack: {str_stack}")
    
    # Generic Queue
    float_queue: Queue[float] = Queue()
    float_queue.enqueue(1.5)
    float_queue.enqueue(2.7)
    float_queue.enqueue(3.14)
    print(f"Float queue: {float_queue}")
    print(f"Dequeued: {float_queue.dequeue()}")
    print()
    
    # 2. Generic Key-Value Store
    print("2. GENERIC KEY-VALUE STORE:")
    
    # String to Integer store
    str_int_store: KeyValueStore[str, int] = KeyValueStore()
    str_int_store.put("apple", 5)
    str_int_store.put("banana", 3)
    str_int_store.put("orange", 8)
    
    print(f"Store: {str_int_store}")
    print(f"Apple count: {str_int_store.get('apple')}")
    print(f"Keys: {str_int_store.keys()}")
    print(f"Values: {str_int_store.values()}")
    
    # Integer to String store
    int_str_store: KeyValueStore[int, str] = KeyValueStore()
    int_str_store.put(1, "first")
    int_str_store.put(2, "second")
    print(f"Int-String store: {int_str_store}")
    print()
    
    # 3. Generic Repository Pattern
    print("3. GENERIC REPOSITORY PATTERN:")
    
    # User repository
    user_repo: Repository[User] = InMemoryRepository()
    
    user1 = User("U001", "Alice Johnson", "alice@example.com")
    user2 = User("U002", "Bob Smith", "bob@example.com")
    
    user_repo.save(user1)
    user_repo.save(user2)
    
    print(f"User repository count: {user_repo.count()}")
    retrieved_user = user_repo.find_by_id("U001")
    if retrieved_user:
        print(f"Retrieved user: {retrieved_user}")
    
    # Product repository
    product_repo: Repository[Product] = InMemoryRepository()
    
    product1 = Product("P001", "Laptop", 999.99, "Electronics")
    product2 = Product("P002", "Book", 29.99, "Education")
    
    product_repo.save(product1)
    product_repo.save(product2)
    
    print(f"Product repository count: {product_repo.count()}")
    all_products = product_repo.find_all()
    for product in all_products:
        print(f"  {product}")
    print()
    
    # 4. Generic Functions with Constraints
    print("4. GENERIC FUNCTIONS WITH CONSTRAINTS:")
    
    # Comparable items
    numbers = [3, 1, 4, 1, 5, 9, 2, 6]
    strings = ["apple", "banana", "cherry", "date"]
    
    print(f"Numbers: {numbers}")
    print(f"Max number: {find_max(numbers)}")
    print(f"Min number: {find_min(numbers)}")
    print(f"Sorted numbers: {sort_items(numbers)}")
    
    print(f"Strings: {strings}")
    print(f"Max string: {find_max(strings)}")
    print(f"Sorted strings: {sort_items(strings)}")
    
    # Numeric operations
    int_result = add_numbers(5, 3)
    float_result = add_numbers(2.5, 3.7)
    print(f"Add integers: {int_result}")
    print(f"Add floats: {float_result}")
    
    avg = calculate_average([1, 2, 3, 4, 5])
    print(f"Average: {avg}")
    print()
    
    # 5. Higher-order Generic Functions
    print("5. HIGHER-ORDER GENERIC FUNCTIONS:")
    
    # Map function
    squared = map_function(numbers, lambda x: x ** 2)
    print(f"Squared numbers: {squared}")
    
    upper_strings = map_function(strings, lambda s: s.upper())
    print(f"Upper strings: {upper_strings}")
    
    # Filter function
    even_numbers = filter_function(numbers, lambda x: x % 2 == 0)
    print(f"Even numbers: {even_numbers}")
    
    long_strings = filter_function(strings, lambda s: len(s) > 5)
    print(f"Long strings: {long_strings}")
    
    # Reduce function
    sum_result = reduce_function(numbers, lambda acc, x: acc + x, 0)
    print(f"Sum of numbers: {sum_result}")
    
    concat_result = reduce_function(strings, lambda acc, s: acc + s, "")
    print(f"Concatenated strings: {concat_result}")
    print()
    
    # 6. Generic Cache
    print("6. GENERIC CACHE:")
    
    # String cache
    str_cache: Cache[str, str] = Cache(max_size=3)
    str_cache.put("key1", "value1")
    str_cache.put("key2", "value2")
    str_cache.put("key3", "value3")
    
    print(f"Cache keys: {str_cache.keys()}")
    print(f"Get key1: {str_cache.get('key1')}")
    print(f"Cache keys after access: {str_cache.keys()}")
    
    # Add new item (should evict LRU)
    str_cache.put("key4", "value4")
    print(f"Cache keys after adding key4: {str_cache.keys()}")
    print()
    
    # 7. Generic Event System
    print("7. GENERIC EVENT SYSTEM:")
    
    # User event bus
    user_event_bus: EventBus[User] = EventBus()
    user_handler = UserEventHandler()
    user_event_bus.register_handler(user_handler)
    
    # Product event bus
    product_event_bus: EventBus[Product] = EventBus()
    product_handler = ProductEventHandler()
    product_event_bus.register_handler(product_handler)
    
    # Publish events
    user_event = Event("user_created", user1)
    user_event_bus.publish(user_event)
    
    product_event = Event("product_updated", product1)
    product_event_bus.publish(product_event)
    
    print(f"User events: {len(user_event_bus.get_event_history())}")
    print(f"Product events: {len(product_event_bus.get_event_history())}")
    print()
    
    # 8. Generic Programming Benefits
    print("8. GENERIC PROGRAMMING BENEFITS:")
    print("✓ Type safety with flexibility")
    print("✓ Code reusability across different types")
    print("✓ Better IDE support and autocompletion")
    print("✓ Compile-time type checking")
    print("✓ Self-documenting code with type hints")
    print("✓ Reduced code duplication")
    print("✓ Easier maintenance and refactoring")
    print()
    
    print("=== GENERIC PROGRAMMING CONCEPTS DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_generic_programming()
